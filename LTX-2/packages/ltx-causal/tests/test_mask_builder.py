"""
Unit tests for Global Prefix causal mask building.

Tests verify:
- Block structure with Global Prefix (V_0 + A_0 paired)
- Exact mask shapes
- Zero future information leakage
- V_0 <-> A_0 bidirectional in Block 0
- Audio alignment (no truncation needed)
"""

import pytest
import torch

from ltx_causal.attention.mask_builder import (
    AVBlock,
    AVCausalMaskBuilder,
    compute_av_blocks,
    compute_aligned_audio_frames,
    compute_total_audio_frames,
    build_all_causal_masks,
    verify_causal_masks,
    AUDIO_FRAMES_PER_BLOCK,
)
from ltx_causal.config import CausalMaskConfig


# ============================================================================
# Block Computation Tests
# ============================================================================

class TestComputeAVBlocks:
    """Tests for compute_av_blocks with Global Prefix."""

    def test_16_video_frames_standard(self):
        """Test 16 video latent frames (121 pixel frames)."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        # Should have 6 blocks: 1 Global Prefix + 5 standard
        assert len(blocks) == 6

        # Block 0: Global Prefix (V_0 + A_0)
        assert blocks[0].block_idx == 0
        assert blocks[0].video_start == 0
        assert blocks[0].video_end == 1
        assert blocks[0].audio_start == 0
        assert blocks[0].audio_end == 1
        assert blocks[0].is_global_prefix

        # Block 1: V_1-V_3, audio 1-26
        assert blocks[1].video_start == 1
        assert blocks[1].video_end == 4
        assert blocks[1].audio_start == 1
        assert blocks[1].audio_end == 26
        assert not blocks[1].is_global_prefix

        # Block 5 (last): V_13-V_15, audio 101-126
        assert blocks[5].video_start == 13
        assert blocks[5].video_end == 16
        assert blocks[5].audio_start == 101
        assert blocks[5].audio_end == 126

    def test_block_continuity(self):
        """Test that video and audio ranges are continuous with no gaps."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        # Video continuity
        for i in range(len(blocks) - 1):
            assert blocks[i].video_end == blocks[i + 1].video_start, (
                f"Video gap between block {i} and {i+1}"
            )

        # Audio continuity (skip Block 0 which has no audio)
        audio_blocks = [b for b in blocks if b.audio_end > b.audio_start]
        for i in range(len(audio_blocks) - 1):
            assert audio_blocks[i].audio_end == audio_blocks[i + 1].audio_start, (
                f"Audio gap between audio blocks {i} and {i+1}"
            )

    def test_total_coverage(self):
        """Test that blocks cover all video frames."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        assert blocks[0].video_start == 0
        assert blocks[-1].video_end == 16

    def test_global_prefix_has_one_audio(self):
        """Test that Block 0 (Global Prefix) has exactly 1 audio frame (A_0)."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        assert blocks[0].audio_frames == 1
        assert blocks[0].video_frames == 1

    def test_standard_blocks_have_25_audio(self):
        """Test that full standard blocks have exactly 25 audio frames."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        for block in blocks[1:]:
            assert block.audio_frames == 25
            assert block.video_frames == 3

    def test_single_frame(self):
        """Test with only 1 video frame (only Global Prefix)."""
        blocks = compute_av_blocks(1, num_frame_per_block=3)

        assert len(blocks) == 1
        assert blocks[0].is_global_prefix
        assert blocks[0].video_frames == 1
        assert blocks[0].audio_frames == 1

    def test_partial_last_block(self):
        """Test with partial last standard block."""
        # 5 video frames: Block 0 (1 frame) + Block 1 (3 frames) + Block 2 (1 frame)
        blocks = compute_av_blocks(5, num_frame_per_block=3)

        assert len(blocks) == 3
        assert blocks[0].is_global_prefix
        assert blocks[1].video_frames == 3
        assert blocks[1].audio_frames == 25
        assert blocks[2].video_frames == 1
        # Partial block: 25 * 1 / 3 = 8 audio frames
        assert blocks[2].audio_frames == 8

    def test_4_video_frames(self):
        """Test with 4 video frames: Global Prefix + 1 full standard block."""
        blocks = compute_av_blocks(4, num_frame_per_block=3)

        assert len(blocks) == 2
        assert blocks[0].is_global_prefix
        assert blocks[1].video_frames == 3
        assert blocks[1].audio_frames == 25


class TestComputeAlignedAudioFrames:
    """Tests for compute_aligned_audio_frames."""

    def test_16_video_frames(self):
        """16 video frames -> 126 aligned audio (matches ltx-core exactly)."""
        aligned = compute_aligned_audio_frames(16, num_frame_per_block=3)
        assert aligned == 126

    def test_1_video_frame(self):
        """1 video frame -> 1 audio (A_0 in Global Prefix)."""
        aligned = compute_aligned_audio_frames(1, num_frame_per_block=3)
        assert aligned == 1

    def test_4_video_frames(self):
        """4 video frames -> 26 audio (A_0 + 25 from 1 standard block)."""
        aligned = compute_aligned_audio_frames(4, num_frame_per_block=3)
        assert aligned == 26

    def test_matches_compute_total(self):
        """compute_total_audio_frames should match compute_aligned_audio_frames."""
        for vf in [1, 4, 7, 10, 13, 16]:
            assert compute_total_audio_frames(vf) == compute_aligned_audio_frames(vf)


# ============================================================================
# Mask Shape Tests (CPU-only, no FlexAttention needed)
# ============================================================================

class TestMaskShapesCPU:
    """Tests for mask shapes using small configs (no CUDA required for dense masks)."""

    def test_a2v_mask_shape(self):
        """Test A2V cross-attention mask shape."""
        video_frame_seqlen = 4  # Small for testing
        audio_frame_seqlen = 1
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        builder = AVCausalMaskBuilder.__new__(AVCausalMaskBuilder)
        builder.video_frame_seqlen = video_frame_seqlen
        builder.audio_frame_seqlen = audio_frame_seqlen
        builder.num_frame_per_block = 3

        mask = builder.build_a2v_causal_mask(blocks, device='cpu')

        # 16 video frames * 4 tokens = 64
        # 126 audio frames * 1 token = 126
        assert mask.shape == (64, 126)
        assert mask.dtype == torch.bool

    def test_v2a_mask_shape(self):
        """Test V2A cross-attention mask shape."""
        video_frame_seqlen = 4
        audio_frame_seqlen = 1
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        builder = AVCausalMaskBuilder.__new__(AVCausalMaskBuilder)
        builder.video_frame_seqlen = video_frame_seqlen
        builder.audio_frame_seqlen = audio_frame_seqlen
        builder.num_frame_per_block = 3

        mask = builder.build_v2a_causal_mask(blocks, device='cpu')

        # 126 audio tokens, 64 video tokens
        assert mask.shape == (126, 64)
        assert mask.dtype == torch.bool


# ============================================================================
# Causality Property Tests
# ============================================================================

class TestGlobalPrefixBidirectional:
    """Tests for Global Prefix (V_0 + A_0) bidirectional properties."""

    @pytest.fixture
    def blocks_and_builder(self):
        """Create blocks and builder with small dimensions."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)
        builder = AVCausalMaskBuilder.__new__(AVCausalMaskBuilder)
        builder.video_frame_seqlen = 4
        builder.audio_frame_seqlen = 1
        builder.num_frame_per_block = 3
        return blocks, builder

    def test_v0_sees_a0_in_a2v(self, blocks_and_builder):
        """Block 0 video (V_0) should attend to A_0 in A2V."""
        blocks, builder = blocks_and_builder
        mask = builder.build_a2v_causal_mask(blocks, device='cpu')

        # Block 0: video tokens 0..3 (4 tokens for V_0), audio token 0 (A_0)
        v0_a0 = mask[:4, :1]
        assert v0_a0.all(), "V_0 should attend to A_0 in A2V!"

    def test_v0_no_future_audio_a2v(self, blocks_and_builder):
        """Block 0 video (V_0) should NOT attend to audio beyond A_0."""
        blocks, builder = blocks_and_builder
        mask = builder.build_a2v_causal_mask(blocks, device='cpu')

        # V_0 should NOT see audio tokens 1+ (A_1, A_2, ...)
        v0_future = mask[:4, 1:]
        assert not v0_future.any(), "V_0 should not see audio beyond A_0!"

    def test_a0_sees_v0_in_v2a(self, blocks_and_builder):
        """Block 0 audio (A_0) should attend to V_0 in V2A."""
        blocks, builder = blocks_and_builder
        mask = builder.build_v2a_causal_mask(blocks, device='cpu')

        # A_0: audio token 0, V_0: video tokens 0..3
        a0_v0 = mask[:1, :4]
        assert a0_v0.all(), "A_0 should attend to V_0 in V2A!"

    def test_all_audio_sees_v0_v2a(self, blocks_and_builder):
        """All audio tokens should be able to attend to V_0 (V2A)."""
        blocks, builder = blocks_and_builder
        mask = builder.build_v2a_causal_mask(blocks, device='cpu')

        # V_0 video tokens: columns 0..3
        v0_columns = mask[:, :4]
        assert v0_columns.all(), "All audio should see V_0 in V2A!"

    def test_block1_video_sees_block0_and_block1_audio(self, blocks_and_builder):
        """Block 1 video should see Block 0 (A_0) and Block 1 audio in A2V."""
        blocks, builder = blocks_and_builder
        mask = builder.build_a2v_causal_mask(blocks, device='cpu')

        # Block 1: video tokens 4..15, should see audio tokens 0..25 (A_0 + A_1..A_25)
        block1_slice = mask[4:16, 0:26]
        assert block1_slice.all(), "Block 1 video should see Block 0 + Block 1 audio!"

    def test_block1_video_no_future_audio(self, blocks_and_builder):
        """Block 1 video should NOT see Block 2+ audio in A2V."""
        blocks, builder = blocks_and_builder
        mask = builder.build_a2v_causal_mask(blocks, device='cpu')

        # Block 1: video tokens 4..15, should NOT see audio tokens 26+
        future_slice = mask[4:16, 26:]
        assert not future_slice.any(), "Block 1 video should not see future audio!"


class TestFutureLeakage:
    """Tests to verify zero future information leakage."""

    @pytest.fixture
    def blocks_and_builder(self):
        blocks = compute_av_blocks(16, num_frame_per_block=3)
        builder = AVCausalMaskBuilder.__new__(AVCausalMaskBuilder)
        builder.video_frame_seqlen = 4
        builder.audio_frame_seqlen = 1
        builder.num_frame_per_block = 3
        return blocks, builder

    def test_no_a2v_future_leakage(self, blocks_and_builder):
        """No video token should see future audio tokens in A2V."""
        blocks, builder = blocks_and_builder
        mask = builder.build_a2v_causal_mask(blocks, device='cpu')

        for block in blocks:
            v_start = block.video_start * 4
            v_end = block.video_end * 4
            a_end = block.audio_end

            if a_end < 126:
                assert not mask[v_start:v_end, a_end:].any(), (
                    f"A2V future leakage in block {block.block_idx}!"
                )

    def test_no_v2a_future_leakage(self, blocks_and_builder):
        """No audio token should see future video tokens in V2A."""
        blocks, builder = blocks_and_builder
        mask = builder.build_v2a_causal_mask(blocks, device='cpu')

        for block in blocks:
            a_start = block.audio_start
            a_end = block.audio_end
            v_end = block.video_end * 4

            if v_end < 64:
                assert not mask[a_start:a_end, v_end:].any(), (
                    f"V2A future leakage in block {block.block_idx}!"
                )

    def test_causal_monotonicity_a2v(self, blocks_and_builder):
        """Later blocks should see strictly more audio than earlier blocks."""
        blocks, builder = blocks_and_builder
        mask = builder.build_a2v_causal_mask(blocks, device='cpu')

        # Blocks with audio
        audio_blocks = [b for b in blocks if b.audio_end > b.audio_start]
        for i in range(len(audio_blocks) - 1):
            curr = audio_blocks[i]
            next_b = audio_blocks[i + 1]

            # Count how many audio tokens each block's first video token can see
            curr_v_start = curr.video_start * 4
            next_v_start = next_b.video_start * 4

            curr_visible = mask[curr_v_start].sum().item()
            next_visible = mask[next_v_start].sum().item()

            assert next_visible > curr_visible, (
                f"Block {next_b.block_idx} should see more audio than "
                f"block {curr.block_idx}: {next_visible} <= {curr_visible}"
            )


class TestVerifyCausalMasks:
    """Tests for the verify_causal_masks function."""

    def test_valid_masks_pass_verification(self):
        """Correctly built masks should pass verification."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)
        builder = AVCausalMaskBuilder.__new__(AVCausalMaskBuilder)
        builder.video_frame_seqlen = 4
        builder.audio_frame_seqlen = 1
        builder.num_frame_per_block = 3

        masks = {
            'video_self': None,  # Would be BlockMask on CUDA
            'audio_self': None,
            'a2v': builder.build_a2v_causal_mask(blocks, device='cpu'),
            'v2a': builder.build_v2a_causal_mask(blocks, device='cpu'),
        }

        # Should not raise
        verify_causal_masks(masks, blocks, video_frame_seqlen=4, audio_frame_seqlen=1)

    def test_wrong_a2v_shape_fails(self):
        """Incorrect A2V mask shape should fail verification."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        masks = {
            'video_self': None,
            'audio_self': None,
            'a2v': torch.zeros(10, 10, dtype=torch.bool),  # Wrong shape
            'v2a': torch.zeros(126, 64, dtype=torch.bool),
        }

        with pytest.raises(AssertionError, match="A2V mask shape mismatch"):
            verify_causal_masks(masks, blocks, video_frame_seqlen=4, audio_frame_seqlen=1)

    def test_future_leakage_detected(self):
        """Mask with future leakage should fail verification."""
        blocks = compute_av_blocks(16, num_frame_per_block=3)

        # Build correct V2A mask
        builder = AVCausalMaskBuilder.__new__(AVCausalMaskBuilder)
        builder.video_frame_seqlen = 4
        builder.audio_frame_seqlen = 1
        builder.num_frame_per_block = 3

        # Correct masks
        a2v = builder.build_a2v_causal_mask(blocks, device='cpu')
        v2a = builder.build_v2a_causal_mask(blocks, device='cpu')

        # Introduce future leakage: Block 1 audio sees Block 5 video
        v2a_bad = v2a.clone()
        v2a_bad[1, 60:64] = True  # Audio token 1 (Block 1) sees last video block

        masks = {
            'video_self': None,
            'audio_self': None,
            'a2v': a2v,
            'v2a': v2a_bad,
        }

        with pytest.raises(AssertionError, match="V2A FUTURE LEAKAGE"):
            verify_causal_masks(masks, blocks, video_frame_seqlen=4, audio_frame_seqlen=1)


# ============================================================================
# Audio Alignment Tests
# ============================================================================

class TestAudioAlignment:
    """Tests for audio alignment with ltx-core."""

    def test_121_pixel_frames_alignment(self):
        """
        121 pixel frames -> 16 video latents, 126 raw audio latents.
        With V_0 + A_0 in Block 0: 126 aligned audio (NO truncation).
        """
        # ltx-core formula: F_latent = 1 + (121 - 1) // 8 = 16
        video_latent_frames = 1 + (121 - 1) // 8
        assert video_latent_frames == 16

        # ltx-core audio: round(121 / 24 * 25) = 126
        raw_audio_frames = round(121 / 24 * 25)
        assert raw_audio_frames == 126

        # Aligned audio matches raw count exactly (no truncation)
        aligned = compute_aligned_audio_frames(16)
        assert aligned == 126

        # No truncation needed!
        assert raw_audio_frames == aligned

    def test_vae_constraint_respected(self):
        """Verify that the standard video frame counts satisfy (n-1)%8==0."""
        # Common valid frame counts for ltx-core VAE
        for n_frames in [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]:
            assert (n_frames - 1) % 8 == 0, f"{n_frames} violates VAE constraint"
            video_latents = 1 + (n_frames - 1) // 8
            # Should produce valid blocks
            blocks = compute_av_blocks(video_latents)
            assert len(blocks) >= 1
            assert blocks[0].is_global_prefix

    def test_aligned_audio_for_various_lengths(self):
        """Test aligned audio frames for various video latent counts."""
        expected = {
            1: 1,     # Only Global Prefix (A_0)
            4: 26,    # A_0 + 1 standard block (25)
            7: 51,    # A_0 + 2 standard blocks (50)
            10: 76,   # A_0 + 3 standard blocks (75)
            13: 101,  # A_0 + 4 standard blocks (100)
            16: 126,  # A_0 + 5 standard blocks (125)
        }
        for vf, expected_af in expected.items():
            actual = compute_aligned_audio_frames(vf)
            assert actual == expected_af, (
                f"For {vf} video frames: expected {expected_af} audio, got {actual}"
            )
