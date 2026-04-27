"""
CausalAVTransformerBlock: Transformer block with causal attention for LTX-2.

This module implements a single transformer block with 6 attention types,
properly adapted for causal generation:

1. attn1 (video_self): Causal block mask - same block + previous blocks
2. attn2 (video_text): No causality needed - text is fixed
3. audio_attn1 (audio_self): Causal block mask
4. audio_attn2 (audio_text): No causality needed
5. audio_to_video_attn (A2V): Timestamp-based causal mask
6. video_to_audio_attn (V2A): Timestamp-based causal mask

Weight-compatible with original BasicAVTransformerBlock.
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from ltx_causal.attention.causal_attention import CausalLTXAttention
from ltx_causal.rope.causal_rope import CausalRopeType
from ltx_causal.transformer.compat import FeedForward

# Try to import BlockMask type
try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TransformerConfig:
    """Configuration for a transformer branch (video or audio)."""
    dim: int
    heads: int
    d_head: int
    context_dim: int  # Text context dimension


@dataclass
class CausalTransformerArgs:
    """
    Arguments for causal transformer forward pass (training only).
    """
    x: torch.Tensor                           # Hidden states [B, L, D]
    timesteps: torch.Tensor                   # Timestep embeddings for AdaLN
    positional_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # RoPE
    context: Optional[torch.Tensor] = None    # Text context
    context_mask: Optional[torch.Tensor] = None
    enabled: bool = True

    # Cross-attention RoPE (for A2V/V2A)
    cross_positional_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    cross_scale_shift_timestep: Optional[torch.Tensor] = None
    cross_gate_timestep: Optional[torch.Tensor] = None

    # Causal masks (training only)
    block_mask: Optional["BlockMask"] = None  # For self-attention
    cross_causal_mask: Optional[torch.Tensor] = None  # For cross-attention

    # Log-ratio scales for causal attention output (entropy-aligned rescaling)
    self_attn_log_scale: Optional[torch.Tensor] = None   # [1, L, 1]
    cross_attn_log_scale: Optional[torch.Tensor] = None  # [1, L, 1]

    # KV-cache fields for causal autoregressive inference / self-forcing
    # training. When provided, the block uses incremental KV-cache attention
    # instead of block_mask / cross_causal_mask. All fields are per-layer
    # (the caller passes per-layer dicts via the block-level forward kwargs).
    kv_cache_self: Optional[Dict[str, Any]] = None         # own-modality self-attn cache
    kv_cache_cross: Optional[Dict[str, Any]] = None        # cross-modal attn cache (this branch's Q -> other's K/V)
    crossattn_cache_text: Optional[Dict[str, Any]] = None  # text cross-attn cache (one-shot)
    current_token_start: int = 0                            # own-modality token offset (for cache writes & RoPE)
    cross_token_start: int = 0                              # other-modality token offset (for cross-modal cache writes & RoPE)


# FeedForward imported from compat.py (uses GELUApprox for weight-compatible state_dict keys)

# ============================================================================
# RMS Normalization
# ============================================================================

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization.

    Uses torch.nn.functional.rms_norm to match the original LTX-2 implementation
    (ltx_core.utils.rms_norm). The manual implementation (x * rsqrt(mean(x²) + eps))
    can produce different results under bf16 due to intermediate precision handling.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)


# ============================================================================
# Causal AV Transformer Block
# ============================================================================

class CausalAVTransformerBlock(nn.Module):
    """
    Causal Audio-Video Transformer Block.

    Contains 6 attention mechanisms with proper causality:
    - Video self-attention (causal)
    - Video-text cross-attention (non-causal, text is fixed)
    - Audio self-attention (causal)
    - Audio-text cross-attention (non-causal, text is fixed)
    - Audio-to-Video cross-attention (timestamp causal)
    - Video-to-Audio cross-attention (timestamp causal)

    Weight-compatible with original BasicAVTransformerBlock.
    """

    def __init__(
        self,
        idx: int,
        video: Optional[TransformerConfig] = None,
        audio: Optional[TransformerConfig] = None,
        rope_type: CausalRopeType = CausalRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        # Kept in signature for backward-compatible construction but unused
        local_attn_size: int = 16,
        sink_size: int = 1,
    ):
        """
        Initialize transformer block.

        Args:
            idx: Block index
            video: Video branch configuration
            audio: Audio branch configuration
            rope_type: Type of RoPE
            norm_eps: Epsilon for normalization
        """
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps

        # Diagnostic: when True, forward() collects gate/scale stats into _gate_stats
        self._store_gate_stats = True
        self._gate_stats = {}

        # Curriculum learning: skip A2V/V2A cross-modal attention
        self.skip_cross_modal_attention = False

        # === Video Branch ===
        if video is not None:
            # Video self-attention (CAUSAL) — with temperature cooling
            self.attn1 = CausalLTXAttention(
                query_dim=video.dim,
                context_dim=None,  # Self-attention
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Video-text cross-attention (NON-CAUSAL - text is fixed, no temperature)
            self.attn2 = CausalLTXAttention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Video feed-forward
            self.ff = FeedForward(video.dim, dim_out=video.dim)

            # AdaLN parameters (6 values: shift/scale/gate for MSA and MLP)
            self.scale_shift_table = nn.Parameter(torch.empty(6, video.dim))

        # === Audio Branch ===
        if audio is not None:
            # Audio self-attention (CAUSAL) — with temperature cooling
            self.audio_attn1 = CausalLTXAttention(
                query_dim=audio.dim,
                context_dim=None,  # Self-attention
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Audio-text cross-attention (NON-CAUSAL, no temperature)
            self.audio_attn2 = CausalLTXAttention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Audio feed-forward
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)

            # AdaLN parameters
            self.audio_scale_shift_table = nn.Parameter(torch.empty(6, audio.dim))

        # === Cross-Modal Attention (A2V and V2A) ===
        if audio is not None and video is not None:
            # Audio-to-Video: Q=Video, K/V=Audio (CAUSAL with timestamp mask + temperature)
            self.audio_to_video_attn = CausalLTXAttention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,  # Uses audio head config
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Video-to-Audio: Q=Audio, K/V=Video (CAUSAL with timestamp mask + temperature)
            self.video_to_audio_attn = CausalLTXAttention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,  # Uses audio head config
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # AdaLN for cross-attention (5 values: 4 scale/shift + 1 gate)
            self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, video.dim))

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get AdaLN values (shift, scale, gate) from scale_shift_table + timestep.
        """
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)

        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get AdaLN values for cross-attention.
        """
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
            slice(None, None),
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def _record_grad(self, name: str, grad: torch.Tensor):
        """Record gradient norm for diagnostics (called by tensor hooks)."""
        with torch.no_grad():
            self._gate_stats[f'grad_{name}_norm'] = grad.detach().float().norm().item()
            self._gate_stats[f'grad_{name}_absmax'] = grad.detach().float().abs().max().item()

    def forward(
        self,
        video: Optional[CausalTransformerArgs] = None,
        audio: Optional[CausalTransformerArgs] = None,
    ) -> Tuple[Optional[CausalTransformerArgs], Optional[CausalTransformerArgs]]:
        """
        Forward pass through the transformer block.

        Args:
            video: Video branch arguments
            audio: Audio branch arguments

        Returns:
            Updated (video, audio) arguments
        """
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        batch_size = (video or audio).x.shape[0]

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0) and not self.skip_cross_modal_attention
        run_v2a = run_ax and (video is not None and vx.numel() > 0) and not self.skip_cross_modal_attention

        # === Video Self-Attention (CAUSAL) ===
        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )

            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa

            # Causal self-attention with block_mask OR KV cache.
            vx_attn = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                block_mask=video.block_mask if video.kv_cache_self is None else None,
                logit_log_scale=video.self_attn_log_scale,
                kv_cache=video.kv_cache_self,
                kv_start=video.current_token_start,
            )

            # Collect gate stats for diagnostics (detach to avoid affecting
            # gradient checkpointing saved-tensor count)
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vgate_msa_mean'] = vgate_msa.detach().float().mean().item()
                    self._gate_stats['vgate_msa_std'] = vgate_msa.detach().float().std().item()
                    self._gate_stats['vscale_msa_mean'] = vscale_msa.detach().float().mean().item()
                    self._gate_stats['vscale_msa_std'] = vscale_msa.detach().float().std().item()
                    self._gate_stats['vshift_msa_mean'] = vshift_msa.detach().float().mean().item()
                    self._gate_stats['vx_attn_norm'] = vx_attn.detach().float().norm().item()
                    self._gate_stats['vx_self_attn_out_norm'] = vx_attn.detach().float().norm().item()
                    self._gate_stats['vx_self_attn_out_absmax'] = vx_attn.detach().float().abs().max().item()
                if vx_attn.requires_grad:
                    vx_attn.register_hook(lambda g, s=self: s._record_grad('vx_self_attn', g))

            vx = vx + vx_attn * vgate_msa

            # Video-text cross-attention (non-causal). Reuses crossattn cache
            # across denoising steps and rollout blocks when provided.
            vx_text_attn = self.attn2(
                rms_norm(vx, eps=self.norm_eps),
                context=video.context,
                mask=video.context_mask,
                crossattn_cache=video.crossattn_cache_text,
            )
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vx_text_attn_out_norm'] = vx_text_attn.detach().float().norm().item()
                    self._gate_stats['vx_text_attn_out_absmax'] = vx_text_attn.detach().float().abs().max().item()
                if vx_text_attn.requires_grad:
                    vx_text_attn.register_hook(lambda g, s=self: s._record_grad('vx_text_attn', g))
            vx = vx + vx_text_attn

            del vshift_msa, vscale_msa, vgate_msa

        # === Audio Self-Attention (CAUSAL) ===
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa

            # Causal self-attention with block_mask OR KV cache.
            ax_attn = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                block_mask=audio.block_mask if audio.kv_cache_self is None else None,
                logit_log_scale=audio.self_attn_log_scale,
                kv_cache=audio.kv_cache_self,
                kv_start=audio.current_token_start,
            )

            # Collect audio gate stats
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['agate_msa_mean'] = agate_msa.detach().float().mean().item()
                    self._gate_stats['agate_msa_std'] = agate_msa.detach().float().std().item()
                    self._gate_stats['ax_attn_norm'] = ax_attn.detach().float().norm().item()
                    self._gate_stats['ax_self_attn_out_norm'] = ax_attn.detach().float().norm().item()
                    self._gate_stats['ax_self_attn_out_absmax'] = ax_attn.detach().float().abs().max().item()
                if ax_attn.requires_grad:
                    ax_attn.register_hook(lambda g, s=self: s._record_grad('ax_self_attn', g))

            ax = ax + ax_attn * agate_msa

            # Audio-text cross-attention (non-causal). Reuses crossattn cache.
            ax_text_attn = self.audio_attn2(
                rms_norm(ax, eps=self.norm_eps),
                context=audio.context,
                mask=audio.context_mask,
                crossattn_cache=audio.crossattn_cache_text,
            )
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['ax_text_attn_out_norm'] = ax_text_attn.detach().float().norm().item()
                    self._gate_stats['ax_text_attn_out_absmax'] = ax_text_attn.detach().float().abs().max().item()
                if ax_text_attn.requires_grad:
                    ax_text_attn.register_hook(lambda g, s=self: s._record_grad('ax_text_attn', g))
            ax = ax + ax_text_attn

            del ashift_msa, ascale_msa, agate_msa

        # === Cross-Modal Attention (A2V and V2A - CAUSAL) ===
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            # Get AdaLN values for cross-attention
            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            # A2V: Video attends to Audio (with timestamp causal mask OR KV cache)
            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v

                a2v_out = self.audio_to_video_attn(
                    vx_scaled,
                    context=ax_scaled,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                    cross_causal_mask=video.cross_causal_mask if video.kv_cache_cross is None else None,
                    logit_log_scale=video.cross_attn_log_scale,
                    kv_cache=video.kv_cache_cross,
                    kv_start=video.cross_token_start,  # audio token offset
                )

                if self._store_gate_stats:
                    with torch.no_grad():
                        self._gate_stats['gate_a2v_mean'] = gate_out_a2v.detach().float().mean().item()
                        self._gate_stats['a2v_out_norm'] = a2v_out.detach().float().norm().item()
                        self._gate_stats['a2v_attn_out_norm'] = a2v_out.detach().float().norm().item()
                        self._gate_stats['a2v_attn_out_absmax'] = a2v_out.detach().float().abs().max().item()
                    if a2v_out.requires_grad:
                        a2v_out.register_hook(lambda g, s=self: s._record_grad('a2v_attn', g))

                vx = vx + a2v_out * gate_out_a2v

            # V2A: Audio attends to Video (with timestamp causal mask OR KV cache)
            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a

                v2a_out = self.video_to_audio_attn(
                    ax_scaled,
                    context=vx_scaled,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                    cross_causal_mask=audio.cross_causal_mask if audio.kv_cache_cross is None else None,
                    logit_log_scale=audio.cross_attn_log_scale,
                    kv_cache=audio.kv_cache_cross,
                    kv_start=audio.cross_token_start,  # video token offset
                )

                if self._store_gate_stats:
                    with torch.no_grad():
                        self._gate_stats['gate_v2a_mean'] = gate_out_v2a.detach().float().mean().item()
                        self._gate_stats['v2a_out_norm'] = v2a_out.detach().float().norm().item()
                        self._gate_stats['v2a_attn_out_norm'] = v2a_out.detach().float().norm().item()
                        self._gate_stats['v2a_attn_out_absmax'] = v2a_out.detach().float().abs().max().item()
                    if v2a_out.requires_grad:
                        v2a_out.register_hook(lambda g, s=self: s._record_grad('v2a_attn', g))

                ax = ax + v2a_out * gate_out_v2a

            del gate_out_a2v, gate_out_v2a

        # === Feed-Forward Networks ===
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vgate_mlp_mean'] = vgate_mlp.detach().float().mean().item()
                    self._gate_stats['vgate_mlp_std'] = vgate_mlp.detach().float().std().item()

            del vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['agate_mlp_mean'] = agate_mlp.detach().float().mean().item()
                    self._gate_stats['agate_mlp_std'] = agate_mlp.detach().float().std().item()

            del ashift_mlp, ascale_mlp, agate_mlp

        # Return updated arguments
        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )


# ============================================================================
# Factory Functions
# ============================================================================

def create_causal_av_block(
    idx: int,
    video_dim: int = 4096,
    audio_dim: int = 2048,
    video_heads: int = 32,
    audio_heads: int = 32,
    video_d_head: int = 128,
    audio_d_head: int = 64,
    context_dim: int = 4096,
    **kwargs,
) -> CausalAVTransformerBlock:
    """
    Create a causal AV transformer block with LTX-2 19B defaults.

    """
    video_config = TransformerConfig(
        dim=video_dim,
        heads=video_heads,
        d_head=video_d_head,
        context_dim=context_dim,
    )

    audio_config = TransformerConfig(
        dim=audio_dim,
        heads=audio_heads,
        d_head=audio_d_head,
        context_dim=context_dim,
    )

    return CausalAVTransformerBlock(
        idx=idx,
        video=video_config,
        audio=audio_config,
        **kwargs,
    )
