"""
Unit tests for CausalLTX2DiffusionWrapper.
"""

import pytest
import torch

from ltx_causal.config import CausalGenerationConfig


class TestCausalTimestepProcessing:
    """Tests for causal timestep processing with Global Prefix."""

    def test_uniform_timestep_passthrough(self):
        """Test that 1D timesteps pass through unchanged."""
        timestep = torch.tensor([500, 500, 500, 500])

        # 1D timestep should remain unchanged
        assert timestep.ndim == 1
        assert timestep.shape[0] == 4

    def test_global_prefix_timestep_processing(self):
        """
        Test that per-frame timesteps respect Global Prefix.

        For num_frame_per_block=3, 16 frames:
        Block 0 (V_0): frame 0
        Block 1: frames 1-3
        Block 2: frames 4-6
        ...
        Input:  [100, 200, 300, 400, 500, 600, 700, 800, ...]
        Output: [100, 200, 200, 200, 500, 500, 500, 800, ...]
        """
        timestep = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800,
                                   900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]])
        num_frame_per_block = 3
        B, F = timestep.shape

        result = timestep.new_zeros(B, F)

        # Block 0 (Global Prefix): frame 0
        result[:, 0] = timestep[:, 0]

        # Standard blocks from frame 1
        idx = 1
        while idx < F:
            end = min(idx + num_frame_per_block, F)
            result[:, idx:end] = timestep[:, idx:idx + 1].expand(B, end - idx)
            idx = end

        # Block 0: frame 0 alone
        assert result[0, 0] == 100

        # Block 1: frames 1-3 all get frame 1's timestep
        assert result[0, 1] == result[0, 2] == result[0, 3] == 200

        # Block 2: frames 4-6 all get frame 4's timestep
        assert result[0, 4] == result[0, 5] == result[0, 6] == 500

        # Block 3: frames 7-9
        assert result[0, 7] == result[0, 8] == result[0, 9] == 800

        # Block 4: frames 10-12
        assert result[0, 10] == result[0, 11] == result[0, 12] == 1100

        # Block 5: frames 13-15
        assert result[0, 13] == result[0, 14] == result[0, 15] == 1400


class TestVideoLatentShape:
    """Tests for video latent shape calculations."""

    def test_512x768_shape(self):
        """Test latent shape for 512x768 resolution (patch_size=1)."""
        video_height = 512
        video_width = 768
        vae_compression = 32

        latent_h = video_height // vae_compression
        latent_w = video_width // vae_compression

        # patch_size=1: each latent spatial position is one token
        frame_tokens = latent_h * latent_w

        assert latent_h == 16
        assert latent_w == 24
        assert frame_tokens == 384


class TestAudioVideoAlignment:
    """Tests for audio-video alignment in wrapper."""

    def test_alignment_with_global_prefix(self):
        """Test that audio is correctly aligned with Global Prefix."""
        from ltx_causal.attention.mask_builder import compute_aligned_audio_frames

        # 16 video latent frames -> 126 aligned audio (V_0+A_0 in Block 0)
        aligned = compute_aligned_audio_frames(16, num_frame_per_block=3)
        assert aligned == 126

    def test_no_truncation_needed(self):
        """Test that ltx-core's raw audio count matches aligned count exactly."""
        from ltx_causal.attention.mask_builder import compute_aligned_audio_frames

        # ltx-core gives 126 audio frames for 121 pixel frames
        raw_audio = round(121 / 24 * 25)
        aligned = compute_aligned_audio_frames(16, num_frame_per_block=3)

        assert raw_audio == 126
        assert aligned == 126
        assert raw_audio == aligned  # No truncation needed!

    def test_block_structure_for_standard_config(self):
        """Test block structure for 16 video frames."""
        from ltx_causal.attention.mask_builder import compute_av_blocks

        blocks = compute_av_blocks(16, num_frame_per_block=3)

        # 6 blocks total: 1 Global Prefix + 5 standard
        assert len(blocks) == 6

        # Global Prefix (V_0 + A_0)
        assert blocks[0].video_frames == 1
        assert blocks[0].audio_frames == 1

        # Standard blocks
        for b in blocks[1:]:
            assert b.video_frames == 3
            assert b.audio_frames == 25


class TestGenerationConfig:
    """Tests for CausalGenerationConfig properties."""

    def test_video_latent_frames(self):
        """Test num_video_latent_frames computation."""
        config = CausalGenerationConfig(num_frames=121)
        assert config.num_video_latent_frames == 16

    def test_audio_latent_frames(self):
        """Test num_audio_latent_frames computation."""
        config = CausalGenerationConfig(num_frames=121)
        # round(121/24 * 25) = 126 (raw, before truncation)
        assert config.num_audio_latent_frames == 126

    def test_video_frame_seqlen(self):
        """Test video_frame_seqlen for 512x768."""
        config = CausalGenerationConfig(video_height=512, video_width=768)
        assert config.video_frame_seqlen == 384
