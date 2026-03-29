"""
Unit tests for ltx-causal configuration module.
"""

import pytest
import torch

from ltx_causal.config import (
    CausalMaskConfig,
    CausalGenerationConfig,
    VIDEO_LATENT_FPS,
    AUDIO_LATENT_FPS,
    compute_num_blocks,
)


class TestCausalMaskConfig:
    """Tests for CausalMaskConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CausalMaskConfig()

        assert config.video_latent_fps == VIDEO_LATENT_FPS
        assert config.audio_latent_fps == AUDIO_LATENT_FPS
        assert config.video_frame_seqlen == 384  # 512x768 with patch_size=1
        assert config.audio_frame_seqlen == 1
        assert config.num_frame_per_block == 3

    def test_audio_block_size(self):
        """Test audio block size is 25."""
        config = CausalMaskConfig()
        assert config.get_audio_block_size() == 25


class TestCausalGenerationConfig:
    """Tests for CausalGenerationConfig."""

    def test_default_values(self):
        """Test default generation config values."""
        config = CausalGenerationConfig()

        assert config.num_layers == 48
        assert config.video_num_heads == 32
        assert config.video_head_dim == 128
        assert config.audio_num_heads == 32
        assert config.audio_head_dim == 64
        assert config.num_frames == 121

    def test_video_latent_frames(self):
        """Test video latent frame count for 121 raw frames."""
        config = CausalGenerationConfig(num_frames=121)
        assert config.num_video_latent_frames == 16

    def test_audio_latent_frames(self):
        """Test raw audio latent count (before truncation)."""
        config = CausalGenerationConfig(num_frames=121)
        assert config.num_audio_latent_frames == 126

    def test_video_frame_seqlen_512x768(self):
        """Test frame seqlen for 512x768 (patch_size=1)."""
        config = CausalGenerationConfig(video_height=512, video_width=768)
        assert config.video_frame_seqlen == 384  # 16*24

    def test_video_frame_seqlen_720x1280(self):
        """Test frame seqlen for 720x1280 (patch_size=1)."""
        config = CausalGenerationConfig(video_height=704, video_width=1280)
        # 704/32 = 22, 1280/32 = 40, seqlen = 22*40 = 880
        assert config.video_frame_seqlen == 880


class TestComputeNumBlocks:
    """Tests for compute_num_blocks with Global Prefix."""

    def test_16_video_frames(self):
        """16 frames: 1 Global Prefix + 5 standard = 6 blocks."""
        assert compute_num_blocks(16, num_frame_per_block=3) == 6

    def test_1_video_frame(self):
        """1 frame: only Global Prefix."""
        assert compute_num_blocks(1, num_frame_per_block=3) == 1

    def test_4_video_frames(self):
        """4 frames: 1 Global Prefix + 1 standard = 2 blocks."""
        assert compute_num_blocks(4, num_frame_per_block=3) == 2

    def test_5_video_frames(self):
        """5 frames: 1 Global Prefix + 2 standard (1 partial) = 3 blocks."""
        assert compute_num_blocks(5, num_frame_per_block=3) == 3


class TestTimeAlignment:
    """Tests for audio-video time alignment calculations."""

    def test_alignment_ratio(self):
        """Test the audio-video alignment ratio."""
        # Video: 24fps -> 3fps latent (8x compression)
        # Audio: 16kHz -> 25fps latent (16000/160/4 = 25)
        video_fps = 3.0
        audio_fps = 25.0

        ratio = audio_fps / video_fps
        assert abs(ratio - 8.333) < 0.01

    def test_frame_count_alignment(self):
        """Test frame count alignment for 121 video frames."""
        num_frames = 121
        video_latent_frames = 1 + (num_frames - 1) // 8  # 16
        video_duration = num_frames / 24.0  # 5.04s

        audio_latent_frames = int(round(video_duration * AUDIO_LATENT_FPS))

        assert video_latent_frames == 16
        assert audio_latent_frames == 126

    def test_frame_count_alignment_241(self):
        """Test frame count alignment for 241 video frames."""
        num_frames = 241
        video_latent_frames = 1 + (num_frames - 1) // 8  # 31
        video_duration = num_frames / 24.0  # 10.04s

        audio_latent_frames = int(round(video_duration * AUDIO_LATENT_FPS))

        assert video_latent_frames == 31
        assert audio_latent_frames == 251
