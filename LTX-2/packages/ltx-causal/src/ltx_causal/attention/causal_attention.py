"""
CausalLTXAttention: Causal attention module with Flexattention for training.

This module implements:
- Training mode: Flexattention with BlockMask for efficient block-wise causal attention
- Weight-compatible with original LTX-2 Attention module

Key Design Decisions:
1. Same projection layer structure as original Attention (to_q, to_k, to_v, to_out)
2. Same normalization (q_norm, k_norm with RMSNorm)
3. BlockMask for causal self-attention, dense mask for cross-attention
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from ltx_causal.attention.flex_attention_utils import (
    FLEX_ATTENTION_AVAILABLE,
    flex_attention_forward,
    standard_attention_forward,
)
from ltx_causal.rope.causal_rope import (
    CausalRopeType,
    apply_interleaved_rotary_emb,
)

# Import BlockMask type for annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


class CausalLTXAttention(nn.Module):
    """
    Causal attention module for LTX-2.

    This module is weight-compatible with the original LTX-2 Attention:
    - Same linear projections (to_q, to_k, to_v, to_out)
    - Same RMSNorm for Q/K normalization
    - Supports both self-attention and cross-attention

    Causal Features:
    - Uses Flexattention with BlockMask for efficient causal attention
    - Dense mask for cross-modal causal attention (A2V, V2A)

    Args:
        query_dim: Dimension of query input
        context_dim: Dimension of context input (None for self-attention)
        heads: Number of attention heads
        dim_head: Dimension per head
        norm_eps: Epsilon for RMSNorm
        rope_type: Type of RoPE (INTERLEAVED only)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: CausalRopeType = CausalRopeType.INTERLEAVED,
        # Kept in signature for backward-compatible construction but unused
        local_attn_size: int = -1,
        sink_size: int = 1,
    ):
        super().__init__()

        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.is_cross_attention = context_dim is not None
        context_dim = query_dim if context_dim is None else context_dim

        # === Projection Layers (Weight-Compatible with Original) ===
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=True)

        # Q/K Normalization
        self.q_norm = nn.RMSNorm(self.inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(self.inner_dim, eps=norm_eps)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=True),
            nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        k_pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # === Causal Training Parameters ===
        block_mask: Optional["BlockMask"] = None,
        cross_causal_mask: Optional[torch.Tensor] = None,
        logit_log_scale: Optional[torch.Tensor] = None,
        # === KV-cache Causal Inference / Self-Forcing Training Parameters ===
        kv_cache: Optional[Dict[str, Any]] = None,
        crossattn_cache: Optional[Dict[str, Any]] = None,
        kv_start: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for training with causal masks.

        Args:
            x: Query input [B, L, D]
            context: Context for cross-attention [B, L_ctx, D_ctx] (None for self-attn)
            mask: Optional attention mask (for non-causal attention, e.g. text)
            pe: RoPE frequencies for Q (cos, sin)
            k_pe: RoPE frequencies for K (if different from Q)
            block_mask: BlockMask for flexattention (causal self-attention)
            cross_causal_mask: Dense mask for cross-attention causality (A2V/V2A)
            logit_log_scale: Per-position log-ratio scale [1, L_q, 1] applied to Q
                before attention, making QK^T = (Q * scale) K^T. Acts as a
                position-dependent temperature: tokens seeing fewer KV tokens
                get scale < 1, softening their attention distribution.

        Returns:
            Attention output [B, L, D]
        """
        B, L, _ = x.shape
        context = x if context is None else context

        # ============================================================
        # Text cross-attention with one-shot crossattn_cache.
        # Text context is fixed across denoising steps and blocks; we
        # project K/V once, mark `is_init=True`, and reuse forever.
        # ============================================================
        if crossattn_cache is not None:
            q = self.q_norm(self.to_q(x))
            if not crossattn_cache.get("is_init", False):
                k_text = self.k_norm(self.to_k(context))
                v_text = self.to_v(context)
                ctx_len = k_text.shape[1]
                crossattn_cache["k"][:, :ctx_len] = k_text.detach()
                crossattn_cache["v"][:, :ctx_len] = v_text.detach()
                crossattn_cache["len"] = ctx_len
                crossattn_cache["is_init"] = True
            ctx_len = crossattn_cache["len"]
            # clone() to break view→cache link for autograd safety
            k_full = crossattn_cache["k"][:, :ctx_len].clone()
            v_full = crossattn_cache["v"][:, :ctx_len].clone()

            if logit_log_scale is not None:
                q = q * logit_log_scale

            q = q.view(B, -1, self.heads, self.dim_head)
            k_full = k_full.view(B, ctx_len, self.heads, self.dim_head)
            v_full = v_full.view(B, ctx_len, self.heads, self.dim_head)

            attn_mask = mask if mask is not None else None
            out = standard_attention_forward(q, k_full, v_full, attn_mask)
            out = out.reshape(B, -1, self.inner_dim)
            return self.to_out(out)

        # ============================================================
        # KV-cache causal path (self-attn or cross-modal attn).
        # On every call, we project the CURRENT block's Q/K/V, write
        # the new K/V into the ring at offset `kv_start`, and then
        # attend over `kv_cache[: end]` (no mask needed: causality is
        # enforced by the fact that future tokens are not yet written).
        # ============================================================
        if kv_cache is not None:
            q = self.q_norm(self.to_q(x))
            k = self.k_norm(self.to_k(context))
            v = self.to_v(context)

            if pe is not None:
                q = self._apply_rope(q, pe)
                k = self._apply_rope(k, pe if k_pe is None else k_pe)

            if logit_log_scale is not None:
                q = q * logit_log_scale

            L_kv = k.shape[1]
            ctx_start = kv_start
            ctx_end = ctx_start + L_kv

            q = q.view(B, -1, self.heads, self.dim_head)
            k = k.view(B, L_kv, self.heads, self.dim_head)
            v = v.view(B, L_kv, self.heads, self.dim_head)

            # Write new K/V into cache at [ctx_start, ctx_end). Use detached
            # copies for the in-place write so that subsequent overwrites (e.g.
            # context_noise refresh) don't break autograd's version tracking.
            kv_cache["k"][:, ctx_start:ctx_end] = k.detach()
            kv_cache["v"][:, ctx_start:ctx_end] = v.detach()
            cur_end = int(kv_cache["end"].item())
            new_end = max(cur_end, ctx_end)
            kv_cache["end"].fill_(new_end)

            # Read historical K/V (detached, from previous no_grad blocks)
            # and concatenate with current K/V (with grad for backward).
            # clone() breaks the view→cache link so in-place cache writes
            # in later layers / steps don't corrupt the backward graph
            # (Flash Attention does this implicitly via .contiguous()).
            k_hist = kv_cache["k"][:, :ctx_start].clone()
            v_hist = kv_cache["v"][:, :ctx_start].clone()
            k_full = torch.cat([k_hist, k], dim=1) if ctx_start > 0 else k
            v_full = torch.cat([v_hist, v], dim=1) if ctx_start > 0 else v

            out = standard_attention_forward(q, k_full, v_full)
            out = out.reshape(B, -1, self.inner_dim)
            return self.to_out(out)

        # ============================================================
        # Original full-sequence training paths (bidirectional, or
        # causal training with explicit block_mask / cross_causal_mask).
        # ============================================================
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Q/K Normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if provided
        if pe is not None:
            q = self._apply_rope(q, pe)
            k = self._apply_rope(k, pe if k_pe is None else k_pe)

        # Apply log-ratio scaling to Q (PaLM-style Log-N Scaling)
        # This scales QK^T by a position-dependent factor, acting as a
        # per-token temperature that softens attention for early causal blocks.
        # scale = log(1 + visible) / log(1 + total), applied BEFORE reshape
        # so it broadcasts across all heads: [1, L, 1] * [B, L, inner_dim]
        if logit_log_scale is not None:
            q = q * logit_log_scale

        # Reshape for attention: [B, L, H, D]
        q = q.view(B, -1, self.heads, self.dim_head)
        k = k.view(B, -1, self.heads, self.dim_head)
        v = v.view(B, -1, self.heads, self.dim_head)

        # Apply attention — all paths use SDPA (standard_attention_forward).
        # block_mask and cross_causal_mask are both dense bool tensors.
        attn_mask = block_mask if block_mask is not None else cross_causal_mask
        if attn_mask is None:
            attn_mask = mask
        out = standard_attention_forward(q, k, v, attn_mask)

        # Reshape and project output
        out = out.reshape(B, -1, self.inner_dim)
        return self.to_out(out)

    def _apply_rope(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply RoPE to input tensor. Only INTERLEAVED mode is supported."""
        if self.rope_type != CausalRopeType.INTERLEAVED:
            raise ValueError(
                f"Only CausalRopeType.INTERLEAVED is supported, got {self.rope_type}. "
                f"SPLIT mode is not implemented correctly for causal generation."
            )
        cos_freqs, sin_freqs = freqs_cis
        return apply_interleaved_rotary_emb(x, cos_freqs, sin_freqs)


# ============================================================================
# Factory Functions
# ============================================================================

def create_causal_attention(
    query_dim: int,
    context_dim: Optional[int] = None,
    heads: int = 32,
    dim_head: int = 128,
    **kwargs,
) -> CausalLTXAttention:
    """
    Factory function to create CausalLTXAttention with LTX-2 defaults.

    Args:
        query_dim: Query dimension
        context_dim: Context dimension (None for self-attention)
        heads: Number of attention heads
        dim_head: Dimension per head
    Returns:
        Configured CausalLTXAttention instance
    """
    return CausalLTXAttention(
        query_dim=query_dim,
        context_dim=context_dim,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_video_self_attention(
    dim: int = 4096,
    heads: int = 32,
    dim_head: int = 128,
    **kwargs,
) -> CausalLTXAttention:
    """Create video self-attention module with LTX-2 19B dimensions."""
    return create_causal_attention(
        query_dim=dim,
        context_dim=None,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_audio_self_attention(
    dim: int = 2048,
    heads: int = 32,
    dim_head: int = 64,
    **kwargs,
) -> CausalLTXAttention:
    """Create audio self-attention module with LTX-2 19B dimensions."""
    return create_causal_attention(
        query_dim=dim,
        context_dim=None,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_cross_attention(
    query_dim: int,
    context_dim: int,
    heads: int = 32,
    dim_head: int = 64,
    **kwargs,
) -> CausalLTXAttention:
    """Create cross-attention module (A2V or V2A)."""
    return create_causal_attention(
        query_dim=query_dim,
        context_dim=context_dim,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )
