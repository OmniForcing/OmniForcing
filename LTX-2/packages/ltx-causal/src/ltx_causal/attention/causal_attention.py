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

from typing import Optional, Tuple

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

        # Projections
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

        # Apply attention
        if block_mask is not None:
            if not FLEX_ATTENTION_AVAILABLE:
                raise RuntimeError(
                    "block_mask provided but flex_attention is not available. "
                    "PyTorch 2.2+ with CUDA is required for causal self-attention."
                )
            # === Flexattention Path (Self-Attention with BlockMask) ===
            out = flex_attention_forward(q, k, v, block_mask)

        elif cross_causal_mask is not None:
            # === Standard Attention with Dense Causal Mask (Cross-Attention) ===
            out = standard_attention_forward(q, k, v, cross_causal_mask)

        elif mask is not None:
            # === Standard Attention with Provided Mask (no temperature) ===
            out = standard_attention_forward(q, k, v, mask)

        else:
            # === Standard Attention (No Mask, no temperature) ===
            out = standard_attention_forward(q, k, v)

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
