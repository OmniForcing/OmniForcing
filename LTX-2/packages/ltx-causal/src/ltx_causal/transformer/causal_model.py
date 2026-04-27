"""
CausalLTXModel: Full causal LTX-2 model for ODE causal training.

This module implements:
- Complete causal transformer with 48 layers
- Training mode: Flexattention with BlockMask for causal masking
- Weight loading from original LTX-2 checkpoints

Weight compatibility:
    All module names and state_dict keys match the original LTXModel exactly.
    The causal model adds block masks at runtime, which do not
    affect the checkpoint structure.

Based on LTX-2's LTXModel architecture (ltx_core/model/transformer/model.py).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn

from ltx_causal.config import (
    CausalMaskConfig,
    VIDEO_LATENT_FPS,
    AUDIO_LATENT_FPS,
)
from ltx_causal.attention.mask_builder import (
    AVCausalMaskBuilder,
    build_all_causal_masks,
    compute_aligned_audio_frames,
    compute_av_blocks,
    compute_causal_log_scales,
)
from ltx_causal.transformer.causal_block import (
    CausalAVTransformerBlock,
    CausalTransformerArgs,
    TransformerConfig,
    rms_norm,
)
from ltx_causal.transformer.compat import (
    AdaLayerNormSingle,
    PixArtAlphaTextProjection,
)
from ltx_causal.rope.causal_rope import (
    CausalRopeType,
    causal_precompute_freqs_cis,
)


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class CausalLTXModelConfig:
    """Configuration for CausalLTXModel."""

    # Model dimensions
    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_heads: int = 32
    audio_heads: int = 32
    video_d_head: int = 128
    audio_d_head: int = 64

    # Cross-attention context dimension
    cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048  # Also used as inner_dim for cross-modal RoPE

    # Patch embedding (LTX-2 uses patch_size=1 with nn.Linear)
    in_channels: int = 128
    out_channels: int = 128
    patch_size: Tuple[int, int, int] = (1, 1, 1)

    # Caption (text) projection
    caption_channels: int = 3840  # Gemma text encoder output dim

    # Position embedding
    pe_theta: float = 10000.0
    pe_max_pos: Tuple[int, int, int] = (20, 2048, 2048)
    audio_pe_max_pos: Tuple[int] = (20,)

    # Timestep embedding
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: int = 1

    # Normalization
    norm_eps: float = 1e-6

    # Causal generation: 4-3-3-3-... layout (Block 0 = 4 video frames + 26 audio,
    # subsequent blocks = 3 video + 25 audio).
    num_frame_per_block: int = 3
    num_frame_per_block_first: int = 4

    # RoPE
    rope_type: CausalRopeType = CausalRopeType.INTERLEAVED

    # Token sizes
    video_frame_seqlen: int = 384  # For 512x768: (512/32)*(768/32)
    audio_frame_seqlen: int = 1

    # Log-ratio entropy-aligned rescaling for causal attention outputs.
    # When True, each token's causal attention output is scaled by
    # log(1 + visible_tokens) / log(1 + total_tokens), compensating for
    # the information deficit caused by causal masking vs bidirectional.
    # No learnable parameters — purely structural rescaling.
    enable_causal_log_rescale: bool = False


# ============================================================================
# CausalLTXModel
# ============================================================================

class CausalLTXModel(nn.Module):
    """
    Causal LTX-2 Model for ODE causal training via masking.

    This model is weight-compatible with the original LTX-2 checkpoint.
    All module names match the original LTXModel to enable direct checkpoint loading.

    Module name mapping to original LTXModel:
        patchify_proj           → nn.Linear(128, 4096)
        audio_patchify_proj     → nn.Linear(128, 2048)
        adaln_single            → AdaLayerNormSingle(4096, coefficient=6)
        audio_adaln_single      → AdaLayerNormSingle(2048, coefficient=6)
        caption_projection      → PixArtAlphaTextProjection(3840, 4096)
        audio_caption_projection → PixArtAlphaTextProjection(3840, 2048)
        av_ca_video_scale_shift_adaln_single → AdaLayerNormSingle(4096, coefficient=4)
        av_ca_audio_scale_shift_adaln_single → AdaLayerNormSingle(2048, coefficient=4)
        av_ca_a2v_gate_adaln_single          → AdaLayerNormSingle(4096, coefficient=1)
        av_ca_v2a_gate_adaln_single          → AdaLayerNormSingle(2048, coefficient=1)
        transformer_blocks      → ModuleList of CausalAVTransformerBlock
        scale_shift_table       → Parameter([2, 4096])
        norm_out                → LayerNorm(4096, elementwise_affine=False)
        proj_out                → nn.Linear(4096, 128)
        audio_scale_shift_table → Parameter([2, 2048])
        audio_norm_out          → LayerNorm(2048, elementwise_affine=False)
        audio_proj_out          → nn.Linear(2048, 128)
    """

    def __init__(self, config: CausalLTXModelConfig):
        super().__init__()
        self.config = config

        # Store key parameters for easy access
        self.timestep_scale_multiplier = config.timestep_scale_multiplier
        self.av_ca_timestep_scale_multiplier = config.av_ca_timestep_scale_multiplier

        # === Patch Embedding (Linear, matching original) ===
        self.patchify_proj = nn.Linear(config.in_channels, config.video_dim, bias=True)
        self.audio_patchify_proj = nn.Linear(config.in_channels, config.audio_dim, bias=True)

        # === AdaLN Timestep Embedding ===
        # Main AdaLN: outputs [N, 6*D] for per-block shift/scale/gate + [N, D] embedded_timestep
        self.adaln_single = AdaLayerNormSingle(config.video_dim, embedding_coefficient=6)
        self.audio_adaln_single = AdaLayerNormSingle(config.audio_dim, embedding_coefficient=6)

        # === Caption (Text) Projection ===
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=config.caption_channels,
            hidden_size=config.video_dim,
        )
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=config.caption_channels,
            hidden_size=config.audio_dim,
        )

        # === Cross-Attention AdaLN (for A2V / V2A) ===
        # 4 additional AdaLN modules for cross-modal timestep conditioning
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            config.video_dim, embedding_coefficient=4,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            config.audio_dim, embedding_coefficient=4,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            config.video_dim, embedding_coefficient=1,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            config.audio_dim, embedding_coefficient=1,
        )

        # === Transformer Blocks ===
        video_config = TransformerConfig(
            dim=config.video_dim,
            heads=config.video_heads,
            d_head=config.video_d_head,
            context_dim=config.cross_attention_dim,
        )

        audio_config = TransformerConfig(
            dim=config.audio_dim,
            heads=config.audio_heads,
            d_head=config.audio_d_head,
            context_dim=config.audio_cross_attention_dim,
        )

        # Name must be `transformer_blocks` to match original state_dict
        self.transformer_blocks = nn.ModuleList([
            CausalAVTransformerBlock(
                idx=i,
                video=video_config,
                audio=audio_config,
                rope_type=config.rope_type,
                norm_eps=config.norm_eps,
            )
            for i in range(config.num_layers)
        ])

        # === Output Layers (matching original names exactly) ===
        # Video output
        self.scale_shift_table = nn.Parameter(torch.empty(2, config.video_dim))
        self.norm_out = nn.LayerNorm(
            config.video_dim, elementwise_affine=False, eps=config.norm_eps
        )
        self.proj_out = nn.Linear(config.video_dim, config.out_channels)

        # Audio output
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, config.audio_dim))
        self.audio_norm_out = nn.LayerNorm(
            config.audio_dim, elementwise_affine=False, eps=config.norm_eps
        )
        self.audio_proj_out = nn.Linear(config.audio_dim, config.out_channels)

        # === Mask Builder ===
        self.mask_builder = AVCausalMaskBuilder(
            video_frame_seqlen=config.video_frame_seqlen,
            audio_frame_seqlen=config.audio_frame_seqlen,
            num_frame_per_block=config.num_frame_per_block,
            num_frame_per_block_first=config.num_frame_per_block_first,
        )

        # Gradient checkpointing
        self.gradient_checkpointing = False

    # ================================================================
    # Timestep Processing
    # ================================================================

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings via AdaLayerNormSingle.

        Matches TransformerArgsPreprocessor._prepare_timestep exactly.

        Args:
            timestep: [B] or [B, F] raw sigma values
            adaln: AdaLayerNormSingle module
            batch_size: Batch size
            hidden_dtype: Target dtype

        Returns:
            (timestep_6d, embedded_timestep):
                timestep_6d: [B, L, 6*D] for block AdaLN
                embedded_timestep: [B, L, D] for output layer
        """
        timestep = timestep * self.timestep_scale_multiplier
        timestep_6d, embedded_timestep = adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        timestep_6d = timestep_6d.view(batch_size, -1, timestep_6d.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep_6d, embedded_timestep

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross-attention timestep embeddings.

        Matches MultiModalTransformerArgsPreprocessor._prepare_cross_attention_timestep.

        Args:
            timestep: [B] or [B, F] raw sigma values
            cross_scale_shift_adaln: AdaLN for scale/shift (coefficient=4)
            cross_gate_adaln: AdaLN for gate (coefficient=1)
            batch_size: Batch size
            hidden_dtype: Target dtype

        Returns:
            (scale_shift_timestep, gate_timestep):
                scale_shift_timestep: [B, L, 4*D]
                gate_timestep: [B, L, D]
        """
        timestep_scaled = timestep * self.timestep_scale_multiplier
        av_ca_factor = self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier

        scale_shift_timestep, _ = cross_scale_shift_adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(
            batch_size, -1, scale_shift_timestep.shape[-1]
        )

        gate_timestep, _ = cross_gate_adaln(
            timestep_scaled.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_timestep = gate_timestep.view(
            batch_size, -1, gate_timestep.shape[-1]
        )

        return scale_shift_timestep, gate_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        projection: PixArtAlphaTextProjection,
        target_dim: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Project and reshape text context.

        Args:
            context: [B, L_ctx, caption_channels] raw text embeddings
            projection: PixArtAlphaTextProjection module
            target_dim: Target hidden dimension
            batch_size: Batch size

        Returns:
            [B, L_ctx, target_dim] projected context
        """
        context = projection(context)
        context = context.view(batch_size, -1, target_dim)
        return context

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        x_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Prepare attention mask (convert bool/int to float mask)."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    # ================================================================
    # Output Processing
    # ================================================================

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: nn.LayerNorm,
        proj_out: nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output with timestep-conditioned modulation.

        Matches LTXModel._process_output exactly.

        Args:
            scale_shift_table: [2, D] learnable parameters
            norm_out: LayerNorm (elementwise_affine=False)
            proj_out: Linear projection to output channels
            x: [B, T, D] transformer output
            embedded_timestep: [B, L, D] from AdaLN

        Returns:
            [B, T, out_channels] projected output
        """
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        timesteps: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        video_context_mask: Optional[torch.Tensor] = None,
        audio_context_mask: Optional[torch.Tensor] = None,
        audio_timesteps: Optional[torch.Tensor] = None,
        masks: Optional[Dict[str, Any]] = None,
        # KV-cache path: when provided, dispatches to the incremental
        # forward_with_cache codepath. This MUST go through forward() so
        # that FSDP pre/post-forward hooks fire and unshard parameters.
        kv_caches: Optional[List[Dict[str, Any]]] = None,
        current_video_start_frame: int = 0,
        current_audio_start_frame: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        When ``kv_caches`` is provided, dispatches to the per-block KV-cache
        path (``forward_with_cache``). Otherwise runs the full-sequence path
        with optional block masks for causal training.

        Args:
            video_latent: [B, F_v, C, H, W] video latent
            audio_latent: [B, F_a, C] audio latent
            timesteps: [B] or [B, F_v] diffusion timesteps (sigma values)
            video_context: [B, L_ctx, caption_channels] video text context
            audio_context: [B, L_ctx, caption_channels] audio text context
            video_context_mask: Optional attention mask for video context
            audio_context_mask: Optional attention mask for audio context
            audio_timesteps: [B] or [B, F_a] audio timesteps (optional, defaults to video)
            masks: Pre-computed causal masks (from build_all_causal_masks)

        Returns:
            (video_velocity, audio_velocity): Velocity predictions
        """
        # === KV-cache dispatch (MUST go through forward() for FSDP) ===
        if kv_caches is not None:
            return self.forward_with_cache(
                video_latent=video_latent,
                audio_latent=audio_latent,
                timesteps=timesteps,
                video_context=video_context,
                audio_context=audio_context,
                kv_caches=kv_caches,
                current_video_start_frame=current_video_start_frame,
                current_audio_start_frame=current_audio_start_frame,
                video_context_mask=video_context_mask,
                audio_context_mask=audio_context_mask,
                audio_timesteps=audio_timesteps,
            )

        B = video_latent.shape[0]
        device = video_latent.device
        hidden_dtype = video_latent.dtype

        # === Patch Embedding ===
        # Video: [B, F, C, H, W] → patchify → [B, T, C] → project → [B, T, D]
        B_v, F_v, C_v, H_v, W_v = video_latent.shape
        video_flat = video_latent.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        video_flat = video_flat.reshape(B_v, C_v, -1).permute(0, 2, 1)  # [B, F*H*W, C]
        video_x = self.patchify_proj(video_flat)  # [B, T, D]
        video_grid_sizes = torch.tensor([F_v, H_v, W_v], device=device).unsqueeze(0)

        # Audio: [B, F_a, C] -> [B, F_a, audio_dim]
        audio_x = self.audio_patchify_proj(audio_latent)
        F_a_original = audio_x.shape[1]
        audio_grid_sizes = torch.tensor([F_a_original], device=device).unsqueeze(0)

        # === Context Projection ===
        video_ctx = self._prepare_context(
            video_context, self.caption_projection, self.config.video_dim, B
        )
        audio_ctx = self._prepare_context(
            audio_context, self.audio_caption_projection, self.config.audio_dim, B
        )

        # Prepare attention masks
        video_context_mask = self._prepare_attention_mask(video_context_mask, hidden_dtype)
        audio_context_mask = self._prepare_attention_mask(audio_context_mask, hidden_dtype)

        # === Timestep Embedding via AdaLayerNormSingle ===
        # Video timestep
        video_ts = timesteps  # [B] or [B, F_v]
        video_timestep_6d, video_embedded_ts = self._prepare_timestep(
            video_ts, self.adaln_single, B, hidden_dtype
        )

        # Audio timestep (defaults to video timestep if not provided)
        audio_ts = audio_timesteps if audio_timesteps is not None else timesteps

        audio_timestep_6d, audio_embedded_ts = self._prepare_timestep(
            audio_ts, self.audio_adaln_single, B, hidden_dtype
        )

        # === Cross-Attention Timesteps ===
        video_cross_ss, video_cross_gate = self._prepare_cross_attention_timestep(
            video_ts,
            self.av_ca_video_scale_shift_adaln_single,
            self.av_ca_a2v_gate_adaln_single,
            B, hidden_dtype,
        )
        audio_cross_ss, audio_cross_gate = self._prepare_cross_attention_timestep(
            audio_ts,
            self.av_ca_audio_scale_shift_adaln_single,
            self.av_ca_v2a_gate_adaln_single,
            B, hidden_dtype,
        )

        # === Expand per-frame timestep embeddings to per-token ===
        # When timesteps is [B, F_v] (per-frame), AdaLN output is [B, F_v, *]
        # but transformer blocks need [B, F_v*H*W, *] (per-token).
        # When timesteps is [B] (scalar), output is [B, 1, *] which broadcasts.
        # Audio has 1 token/frame so no expansion needed.
        frame_seqlen = H_v * W_v
        video_timestep_6d = self._expand_per_frame_to_per_token(video_timestep_6d, frame_seqlen)
        video_embedded_ts = self._expand_per_frame_to_per_token(video_embedded_ts, frame_seqlen)
        video_cross_ss = self._expand_per_frame_to_per_token(video_cross_ss, frame_seqlen)
        video_cross_gate = self._expand_per_frame_to_per_token(video_cross_gate, frame_seqlen)

        # Reuse wrapper-provided masks whenever available. The wrapper already
        # builds them with the correct num_frame_per_block_first.
        if masks is None:
            num_video_frames = video_grid_sizes[0, 0].item()
            num_audio_frames = audio_grid_sizes[0, 0].item()
            mask_config = CausalMaskConfig(
                video_frame_seqlen=self.config.video_frame_seqlen,
                num_frame_per_block=self.config.num_frame_per_block,
                num_frame_per_block_first=self.config.num_frame_per_block_first,
            )
            masks = build_all_causal_masks(
                num_video_frames, num_audio_frames,
                config=mask_config,
                device=device,
            )

        # === Compute log-ratio scales for causal attention rescaling ===
        log_scales = None
        if self.config.enable_causal_log_rescale:
            blocks = compute_av_blocks(
                F_v,
                num_frame_per_block=self.config.num_frame_per_block,
                num_frame_per_block_first=self.config.num_frame_per_block_first,
            )
            log_scales = compute_causal_log_scales(
                blocks,
                video_frame_seqlen=self.config.video_frame_seqlen,
                audio_frame_seqlen=self.config.audio_frame_seqlen,
                device=device,
            )

        # === Precompute RoPE ===
        video_pe = causal_precompute_freqs_cis(
            video_grid_sizes, self.config.video_d_head * self.config.video_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
        )

        audio_pe = causal_precompute_freqs_cis(
            audio_grid_sizes, self.config.audio_d_head * self.config.audio_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.audio_pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,
        )

        # === Cross-attention RoPE ===
        # Original uses 1D temporal-only positions at audio_cross_attention_dim (2048).
        # cross_pe_max_pos = max(pe_max_pos[0], audio_pe_max_pos[0]) = max(20, 20) = 20
        cross_pe_max_pos = max(
            self.config.pe_max_pos[0],
            self.config.audio_pe_max_pos[0],
        )
        # Video cross-PE: temporal positions from video grid (1D, video temporal)
        video_temporal_grid = torch.tensor(
            [[F_v]], device=device, dtype=torch.long
        )  # [1, 1]
        video_cross_pe = causal_precompute_freqs_cis(
            video_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=0,
            rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
            is_audio=False,  # Video temporal conversion
        )
        # Expand temporal PE to full video sequence: each frame's tokens share same temporal PE
        # video_cross_pe: [B, F_v, D] → need [B, F_v*H*W, D]
        video_cross_pe = (
            video_cross_pe[0].unsqueeze(2).expand(-1, -1, self.config.video_frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[0].shape[-1]),
            video_cross_pe[1].unsqueeze(2).expand(-1, -1, self.config.video_frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[1].shape[-1]),
        )

        # Audio cross-PE: temporal positions from audio grid (1D, audio temporal)
        # Use original audio frame count (without sinks) for cross-PE computation
        audio_temporal_grid = torch.tensor(
            [[F_a_original]], device=device, dtype=torch.long
        )  # [1, 1]
        audio_cross_pe = causal_precompute_freqs_cis(
            audio_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=0,
            rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,  # Audio temporal conversion
        )

        # === Prepare Transformer Args ===
        video_args = CausalTransformerArgs(
            x=video_x,
            timesteps=video_timestep_6d,
            positional_embeddings=video_pe,
            context=video_ctx,
            context_mask=video_context_mask,
            block_mask=masks.get('video_self'),
            cross_causal_mask=masks.get('a2v'),
            cross_positional_embeddings=video_cross_pe,
            cross_scale_shift_timestep=video_cross_ss,
            cross_gate_timestep=video_cross_gate,
            self_attn_log_scale=log_scales['video_self_scale'].to(hidden_dtype) if log_scales else None,
            cross_attn_log_scale=log_scales['a2v_scale'].to(hidden_dtype) if log_scales else None,
        )

        audio_args = CausalTransformerArgs(
            x=audio_x,
            timesteps=audio_timestep_6d,
            positional_embeddings=audio_pe,
            context=audio_ctx,
            context_mask=audio_context_mask,
            block_mask=masks.get('audio_self'),
            cross_causal_mask=masks.get('v2a'),
            cross_positional_embeddings=audio_cross_pe,
            cross_scale_shift_timestep=audio_cross_ss,
            cross_gate_timestep=audio_cross_gate,
            self_attn_log_scale=log_scales['audio_self_scale'].to(hidden_dtype) if log_scales else None,
            cross_attn_log_scale=log_scales['v2a_scale'].to(hidden_dtype) if log_scales else None,
        )

        # === Transformer Blocks ===
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                video_args, audio_args = torch.utils.checkpoint.checkpoint(
                    block, video_args, audio_args,
                    use_reentrant=False,
                )
            else:
                video_args, audio_args = block(video_args, audio_args)

        # === Output Layer (with timestep conditioning) ===
        video_out = self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out,
            video_args.x, video_embedded_ts,
        )
        audio_out = self._process_output(
            self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out,
            audio_args.x, audio_embedded_ts,
        )

        # === Unpatchify Video ===
        # [B, T, C] → [B, F, H, W, C] → [B, F, C, H, W]
        F, H, W = F_v, H_v, W_v
        C_out = self.config.out_channels
        video_out = video_out.reshape(B, F, H, W, C_out)
        video_out = video_out.permute(0, 1, 4, 2, 3)  # [B, F, C, H, W]

        return video_out, audio_out

    # ================================================================
    # Model Loading
    # ================================================================

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[CausalLTXModelConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "CausalLTXModel":
        """
        Load model from pretrained LTX-2 checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Model configuration (uses defaults if None)
            device: Target device
            dtype: Model dtype

        Returns:
            Initialized CausalLTXModel with loaded weights
        """
        if config is None:
            config = CausalLTXModelConfig()

        model = cls(config)
        model = model.to(device=device, dtype=dtype)

        # Load checkpoint
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        # Forward-compatible: strip legacy audio_sink_tokens entry if present.
        for legacy_key in list(state_dict.keys()):
            if 'audio_sink_tokens' in legacy_key:
                state_dict.pop(legacy_key)

        # Load with strict=False to ignore mask_builder buffers.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        expected_missing = ['mask_builder']
        real_missing = [
            key for key in missing
            if not any(pat in key for pat in expected_missing)
        ]
        if real_missing:
            raise RuntimeError(
                f"Missing keys in checkpoint: {real_missing[:10]}. "
                f"Total missing: {len(real_missing)}. "
                f"This likely means the checkpoint is incompatible with CausalLTXModel."
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in checkpoint: {unexpected[:10]}. "
                f"Total unexpected: {len(unexpected)}. "
                f"This likely means the checkpoint is incompatible with CausalLTXModel."
            )

        return model

    @staticmethod
    def _expand_per_frame_to_per_token(
        per_frame: torch.Tensor,
        frame_seqlen: int,
    ) -> torch.Tensor:
        """Expand per-frame tensor to per-token by repeating each frame's value.

        When timesteps is [B, F] (per-frame), AdaLN outputs [B, F, D].
        But transformer blocks expect [B, F*H*W, D] (per-token).
        Each frame's H*W tokens share the same timestep embedding.

        When timesteps is [B] (scalar), AdaLN outputs [B, 1, D] which
        broadcasts naturally, so no expansion is needed.

        Args:
            per_frame: [B, F, D] per-frame values
            frame_seqlen: Number of tokens per frame (H*W for video, 1 for audio)

        Returns:
            [B, F*frame_seqlen, D] per-token values
        """
        if per_frame.shape[1] <= 1 or frame_seqlen <= 1:
            return per_frame
        B, F, D = per_frame.shape
        return (
            per_frame.unsqueeze(2)
            .expand(-1, -1, frame_seqlen, -1)
            .reshape(B, F * frame_seqlen, D)
        )

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    # ================================================================
    # KV-cache Causal Inference / Self-Forcing Training
    # ================================================================

    def init_av_kv_caches(
        self,
        batch_size: int,
        max_video_frames: int,
        max_audio_frames: int,
        text_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Dict[str, Any]]:
        """Allocate per-layer KV caches for causal AV generation.

        Each layer's cache contains 6 buffers:
          - video_self: video self-attention K/V (V_cache tokens)
          - audio_self: audio self-attention K/V (A_cache tokens)
          - a2v: cross-modal A2V K/V (audio K/V projected by A2V's to_k/to_v)
          - v2a: cross-modal V2A K/V (video K/V projected by V2A's to_k/to_v)
          - video_text: video-text crossattn cache (one-shot init)
          - audio_text: audio-text crossattn cache (one-shot init)
        """
        v_cache_tokens = max_video_frames * self.config.video_frame_seqlen
        a_cache_tokens = max_audio_frames * self.config.audio_frame_seqlen

        v_h, v_d = self.config.video_heads, self.config.video_d_head
        a_h, a_d = self.config.audio_heads, self.config.audio_d_head

        def _alloc_self(B, T, H, D):
            return {
                "k": torch.zeros(B, T, H, D, device=device, dtype=dtype),
                "v": torch.zeros(B, T, H, D, device=device, dtype=dtype),
                "end": torch.tensor([0], dtype=torch.long, device=device),
            }

        def _alloc_text(B, T, H, D):
            return {
                "k": torch.zeros(B, T, H * D, device=device, dtype=dtype),
                "v": torch.zeros(B, T, H * D, device=device, dtype=dtype),
                "len": 0,
                "is_init": False,
            }

        caches: List[Dict[str, Any]] = []
        for _ in range(self.config.num_layers):
            caches.append({
                # video self-attention: video_dim split as v_h * v_d
                "video_self": _alloc_self(batch_size, v_cache_tokens, v_h, v_d),
                # audio self-attention: audio_dim split as a_h * a_d
                "audio_self": _alloc_self(batch_size, a_cache_tokens, a_h, a_d),
                # A2V: K/V come from the audio_to_video_attn's K/V projection
                # (output dim = a_h * a_d, since audio_to_video_attn uses audio's heads)
                "a2v": _alloc_self(batch_size, a_cache_tokens, a_h, a_d),
                # V2A: K/V come from the video_to_audio_attn's K/V projection
                # (also audio heads/d_head per causal_block.py construction)
                "v2a": _alloc_self(batch_size, v_cache_tokens, a_h, a_d),
                # Text cross-attn caches (full inner dim, one-shot init)
                "video_text": _alloc_text(batch_size, text_seq_len, v_h, v_d),
                "audio_text": _alloc_text(batch_size, text_seq_len, a_h, a_d),
            })
        return caches

    def forward_with_cache(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        timesteps: torch.Tensor,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        kv_caches: List[Dict[str, Any]],
        current_video_start_frame: int,
        current_audio_start_frame: int,
        video_context_mask: Optional[torch.Tensor] = None,
        audio_context_mask: Optional[torch.Tensor] = None,
        audio_timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal autoregressive forward for ONE block, with KV caches.

        Inputs cover the CURRENT block only (block-shaped tensors):
          - video_latent: [B, F_v, C, H, W] for current block (e.g. F_v=4 or 3)
          - audio_latent: [B, F_a, C]       for current block (e.g. F_a=26 or 25)
          - timesteps:    [B] or [B, F_v]   sigma values
          - audio_timesteps: [B] or [B, F_a] (optional, defaults to timesteps)
          - kv_caches: list[num_layers] of cache dicts (allocated by init_av_kv_caches)
          - current_video_start_frame: where this video block starts in the global sequence
          - current_audio_start_frame: where this audio block starts in the global sequence

        Causality is enforced by the cache mechanism: each call writes the
        block's K/V at offsets [current_*_start_frame * frame_seqlen,
        current_*_start_frame * frame_seqlen + L_block) and reads the full
        cache up to the new end. Future tokens are simply not yet written.
        """
        B = video_latent.shape[0]
        device = video_latent.device
        hidden_dtype = video_latent.dtype

        # === Patch embed (current block only) ===
        B_v, F_v, C_v, H_v, W_v = video_latent.shape
        video_flat = video_latent.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        video_flat = video_flat.reshape(B_v, C_v, -1).permute(0, 2, 1)
        video_x = self.patchify_proj(video_flat)
        video_grid_sizes = torch.tensor([F_v, H_v, W_v], device=device).unsqueeze(0)

        audio_x = self.audio_patchify_proj(audio_latent)
        F_a = audio_x.shape[1]
        audio_grid_sizes = torch.tensor([F_a], device=device).unsqueeze(0)

        # === Context projection ===
        video_ctx = self._prepare_context(
            video_context, self.caption_projection, self.config.video_dim, B
        )
        audio_ctx = self._prepare_context(
            audio_context, self.audio_caption_projection, self.config.audio_dim, B
        )
        video_context_mask = self._prepare_attention_mask(video_context_mask, hidden_dtype)
        audio_context_mask = self._prepare_attention_mask(audio_context_mask, hidden_dtype)

        # === Timestep embedding ===
        video_ts = timesteps
        video_timestep_6d, video_embedded_ts = self._prepare_timestep(
            video_ts, self.adaln_single, B, hidden_dtype
        )
        audio_ts = audio_timesteps if audio_timesteps is not None else timesteps
        audio_timestep_6d, audio_embedded_ts = self._prepare_timestep(
            audio_ts, self.audio_adaln_single, B, hidden_dtype
        )

        # === Cross-attention timesteps ===
        video_cross_ss, video_cross_gate = self._prepare_cross_attention_timestep(
            video_ts,
            self.av_ca_video_scale_shift_adaln_single,
            self.av_ca_a2v_gate_adaln_single,
            B, hidden_dtype,
        )
        audio_cross_ss, audio_cross_gate = self._prepare_cross_attention_timestep(
            audio_ts,
            self.av_ca_audio_scale_shift_adaln_single,
            self.av_ca_v2a_gate_adaln_single,
            B, hidden_dtype,
        )

        frame_seqlen = H_v * W_v
        video_timestep_6d = self._expand_per_frame_to_per_token(video_timestep_6d, frame_seqlen)
        video_embedded_ts = self._expand_per_frame_to_per_token(video_embedded_ts, frame_seqlen)
        video_cross_ss = self._expand_per_frame_to_per_token(video_cross_ss, frame_seqlen)
        video_cross_gate = self._expand_per_frame_to_per_token(video_cross_gate, frame_seqlen)

        # === RoPE for the CURRENT BLOCK at the given start_frame ===
        video_pe = causal_precompute_freqs_cis(
            video_grid_sizes, self.config.video_d_head * self.config.video_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.pe_max_pos),
            start_frame=current_video_start_frame, rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
        )
        audio_pe = causal_precompute_freqs_cis(
            audio_grid_sizes, self.config.audio_d_head * self.config.audio_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.audio_pe_max_pos),
            start_frame=current_audio_start_frame, rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,
        )

        # === Cross-PE for current block ===
        cross_pe_max_pos = max(
            self.config.pe_max_pos[0],
            self.config.audio_pe_max_pos[0],
        )
        # Video temporal cross-PE: [1, F_v, D_cross] for current block
        video_temporal_grid = torch.tensor([[F_v]], device=device, dtype=torch.long)
        video_cross_pe = causal_precompute_freqs_cis(
            video_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=current_video_start_frame,
            rope_type=self.config.rope_type,
            device=device, dtype=video_x.dtype,
            is_audio=False,
        )
        # Expand temporal -> per-spatial-token
        video_cross_pe = (
            video_cross_pe[0].unsqueeze(2).expand(-1, -1, self.config.video_frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[0].shape[-1]),
            video_cross_pe[1].unsqueeze(2).expand(-1, -1, self.config.video_frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[1].shape[-1]),
        )

        audio_temporal_grid = torch.tensor([[F_a]], device=device, dtype=torch.long)
        audio_cross_pe = causal_precompute_freqs_cis(
            audio_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=current_audio_start_frame,
            rope_type=self.config.rope_type,
            device=device, dtype=audio_x.dtype,
            is_audio=True,
        )

        # === Build per-layer args with caches ===
        v_token_start = current_video_start_frame * self.config.video_frame_seqlen
        a_token_start = current_audio_start_frame * self.config.audio_frame_seqlen

        # === Run transformer blocks with per-layer caches ===
        for layer_idx, block in enumerate(self.transformer_blocks):
            layer_caches = kv_caches[layer_idx]
            video_args = CausalTransformerArgs(
                x=video_x,
                timesteps=video_timestep_6d,
                positional_embeddings=video_pe,
                context=video_ctx,
                context_mask=video_context_mask,
                cross_positional_embeddings=video_cross_pe,
                cross_scale_shift_timestep=video_cross_ss,
                cross_gate_timestep=video_cross_gate,
                kv_cache_self=layer_caches["video_self"],
                kv_cache_cross=layer_caches["a2v"],          # video's cross-modal Q reads audio K/V via A2V cache
                crossattn_cache_text=layer_caches["video_text"],
                current_token_start=v_token_start,
                cross_token_start=a_token_start,
            )
            audio_args = CausalTransformerArgs(
                x=audio_x,
                timesteps=audio_timestep_6d,
                positional_embeddings=audio_pe,
                context=audio_ctx,
                context_mask=audio_context_mask,
                cross_positional_embeddings=audio_cross_pe,
                cross_scale_shift_timestep=audio_cross_ss,
                cross_gate_timestep=audio_cross_gate,
                kv_cache_self=layer_caches["audio_self"],
                kv_cache_cross=layer_caches["v2a"],          # audio's cross-modal Q reads video K/V via V2A cache
                crossattn_cache_text=layer_caches["audio_text"],
                current_token_start=a_token_start,
                cross_token_start=v_token_start,
            )

            # KV cache path: skip gradient checkpointing because in-place
            # cache writes are incompatible with recomputation.
            video_args, audio_args = block(video_args, audio_args)

            video_x = video_args.x
            audio_x = audio_args.x

        # === Output layer ===
        video_out = self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out,
            video_x, video_embedded_ts,
        )
        audio_out = self._process_output(
            self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out,
            audio_x, audio_embedded_ts,
        )

        # Unpatchify video
        video_out = video_out.reshape(B, F_v, H_v, W_v, self.config.out_channels)
        video_out = video_out.permute(0, 1, 4, 2, 3)
        return video_out, audio_out
