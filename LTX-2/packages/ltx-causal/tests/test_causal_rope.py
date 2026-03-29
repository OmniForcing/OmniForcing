"""
Unit tests for causal RoPE (Rotary Position Embedding).

Tests the current implementation which matches ltx-core exactly,
with causal frame offset support for future real-time inference.
"""

import pytest
import torch

from ltx_causal.rope.causal_rope import (
    causal_precompute_freqs_cis,
    apply_interleaved_rotary_emb,
    CausalRopeType,
    generate_freq_grid,
)


class TestGenerateFreqGrid:
    """Tests for frequency grid generation."""

    def test_basic_grid(self):
        """Test basic frequency grid generation."""
        grid = generate_freq_grid(
            theta=10000.0,
            max_pos_count=3,
            inner_dim=128,
            device='cpu',
        )
        # inner_dim // (2 * max_pos_count) = 128 // 6 ≈ 21
        assert grid.ndim == 1
        assert grid.shape[0] == 128 // 6

    def test_1d_grid(self):
        """Test 1D frequency grid (for audio)."""
        grid = generate_freq_grid(
            theta=10000.0,
            max_pos_count=1,
            inner_dim=64,
            device='cpu',
        )
        # inner_dim // (2 * 1) = 32
        assert grid.shape[0] == 32


class TestCausalPrecomputeFreqsCis:
    """Tests for causal RoPE frequency precomputation."""

    def test_video_3d_output_shape(self):
        """Test output shape for video 3D positions."""
        grid_sizes = torch.tensor([[8, 16, 24]])  # F=8, H=16, W=24
        dim = 128 * 32  # d_head * n_heads = 4096

        cos_freq, sin_freq = causal_precompute_freqs_cis(
            grid_sizes=grid_sizes,
            dim=dim,
            device='cpu',
        )

        seq_len = 8 * 16 * 24  # 3072
        assert cos_freq.shape == (1, seq_len, dim)
        assert sin_freq.shape == (1, seq_len, dim)

    def test_audio_1d_output_shape(self):
        """Test output shape for audio 1D positions."""
        grid_sizes = torch.tensor([[125]])  # 125 audio frames
        dim = 64 * 32  # d_head * n_heads = 2048

        cos_freq, sin_freq = causal_precompute_freqs_cis(
            grid_sizes=grid_sizes,
            dim=dim,
            device='cpu',
            is_audio=True,
        )

        assert cos_freq.shape == (1, 125, dim)
        assert sin_freq.shape == (1, 125, dim)

    def test_start_frame_offset(self):
        """Test that start_frame offset shifts temporal positions."""
        grid_sizes = torch.tensor([[4, 4, 4]])
        dim = 128

        cos_0, sin_0 = causal_precompute_freqs_cis(
            grid_sizes=grid_sizes, dim=dim, start_frame=0, device='cpu',
        )

        cos_5, sin_5 = causal_precompute_freqs_cis(
            grid_sizes=grid_sizes, dim=dim, start_frame=5, device='cpu',
        )

        # Different start frames should produce different frequencies
        assert not torch.allclose(cos_0, cos_5)

    def test_split_mode_raises(self):
        """SPLIT mode should raise ValueError."""
        grid_sizes = torch.tensor([[4, 4, 4]])
        dim = 128

        with pytest.raises(ValueError, match="Only CausalRopeType.INTERLEAVED"):
            causal_precompute_freqs_cis(
                grid_sizes=grid_sizes, dim=dim,
                rope_type=CausalRopeType.SPLIT,
                device='cpu',
            )


class TestApplyInterleavedRotaryEmb:
    """Tests for interleaved rotary embedding application."""

    def test_shape_preservation(self):
        """Test that output shape matches input."""
        B, L, D = 2, 16, 128
        x = torch.randn(B, L, D)
        cos_freq = torch.ones(B, L, D)
        sin_freq = torch.zeros(B, L, D)

        out = apply_interleaved_rotary_emb(x, cos_freq, sin_freq)
        assert out.shape == x.shape

    def test_identity_with_zero_rotation(self):
        """cos=1, sin=0 should approximately preserve input."""
        B, L, D = 2, 16, 128
        x = torch.randn(B, L, D)
        cos_freq = torch.ones(B, L, D)
        sin_freq = torch.zeros(B, L, D)

        out = apply_interleaved_rotary_emb(x, cos_freq, sin_freq)
        assert torch.allclose(out, x, atol=1e-6)


class TestCausalRopeType:
    """Tests for RoPE type enum."""

    def test_interleaved_value(self):
        assert CausalRopeType.INTERLEAVED.value == "interleaved"

    def test_split_value(self):
        assert CausalRopeType.SPLIT.value == "split"
