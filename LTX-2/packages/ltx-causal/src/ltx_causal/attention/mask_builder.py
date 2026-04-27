"""
AVCausalMaskBuilder: Builds causal masks for audio-video joint attention.

Block layout (4-3-3-3-... uniform-after-first):
    Block 0:        4 video frames + 26 audio frames
    Block k >= 1:   3 video frames + 25 audio frames

For 16 video latent frames (121 pixel frames):
    16 = 4 + 4 * 3
    126 = 26 + 4 * 25

Attention rules:
    - Intra-block: bidirectional (within a block, video and audio attend freely)
    - Inter-block: causal (block k sees blocks 0..k)
    - No sink tokens, no Global Prefix special case
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        BlockMask,
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from torch.nn.attention.flex_attention import BlockMask

from ltx_causal.config import CausalMaskConfig


# Audio frame counts (LTX-2 VAE: 24fps video / 8 = 3 video latent fps;
# 16kHz audio / 160 hop / 4 = 25 audio latent fps -> ratio 25/3).
AUDIO_FRAMES_PER_BLOCK = 25         # standard block (3 video frames)
AUDIO_FRAMES_FIRST_BLOCK = 26       # Block 0 (4 video frames, absorbs causal-fix asymmetry)


# ============================================================================
# Block Computation
# ============================================================================

@dataclass
class AVBlock:
    """A synchronized audio-video generation block."""
    block_idx: int
    video_start: int
    video_end: int
    audio_start: int
    audio_end: int

    @property
    def video_frames(self) -> int:
        return self.video_end - self.video_start

    @property
    def audio_frames(self) -> int:
        return self.audio_end - self.audio_start

    def __repr__(self) -> str:
        return (
            f"AVBlock(idx={self.block_idx}, "
            f"video=[{self.video_start}:{self.video_end}], "
            f"audio=[{self.audio_start}:{self.audio_end}])"
        )


def compute_av_blocks(
    total_video_latent_frames: int,
    num_frame_per_block: int = 3,
    num_frame_per_block_first: int = 4,
) -> List[AVBlock]:
    """Compute audio-video block synchronization.

    Block 0:      num_frame_per_block_first video frames + AUDIO_FRAMES_FIRST_BLOCK audio.
    Block k>=1:   num_frame_per_block       video frames + AUDIO_FRAMES_PER_BLOCK   audio.

    Total video frames MUST equal num_frame_per_block_first + k * num_frame_per_block
    for some non-negative integer k. Misaligned inputs raise AssertionError.
    """
    assert total_video_latent_frames >= num_frame_per_block_first, (
        f"Need >= {num_frame_per_block_first} video frames, got {total_video_latent_frames}"
    )
    remaining = total_video_latent_frames - num_frame_per_block_first
    assert remaining % num_frame_per_block == 0, (
        f"Video frame count {total_video_latent_frames} not aligned to "
        f"{num_frame_per_block_first} + k * {num_frame_per_block}. Remainder={remaining % num_frame_per_block}."
    )

    blocks: List[AVBlock] = []

    # Block 0: 4 video + 26 audio
    blocks.append(AVBlock(
        block_idx=0,
        video_start=0,
        video_end=num_frame_per_block_first,
        audio_start=0,
        audio_end=AUDIO_FRAMES_FIRST_BLOCK,
    ))

    video_idx = num_frame_per_block_first
    audio_idx = AUDIO_FRAMES_FIRST_BLOCK
    block_idx = 1

    while video_idx < total_video_latent_frames:
        v_start = video_idx
        v_end = video_idx + num_frame_per_block
        a_start = audio_idx
        a_end = audio_idx + AUDIO_FRAMES_PER_BLOCK

        assert v_end <= total_video_latent_frames, (
            f"Misaligned tail: block {block_idx} would extend video past "
            f"{total_video_latent_frames} (v_end={v_end})."
        )
        blocks.append(AVBlock(
            block_idx=block_idx,
            video_start=v_start,
            video_end=v_end,
            audio_start=a_start,
            audio_end=a_end,
        ))

        video_idx = v_end
        audio_idx = a_end
        block_idx += 1

    assert video_idx == total_video_latent_frames, (
        f"Block layout did not consume all video frames "
        f"(consumed {video_idx}, expected {total_video_latent_frames})"
    )
    return blocks


def compute_aligned_audio_frames(
    total_video_latent_frames: int,
    num_frame_per_block: int = 3,
    num_frame_per_block_first: int = 4,
) -> int:
    """Return the audio frame count aligned to the block layout."""
    blocks = compute_av_blocks(
        total_video_latent_frames,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )
    return blocks[-1].audio_end if blocks else 0


def compute_total_audio_frames(
    total_video_latent_frames: int,
    num_frame_per_block: int = 3,
    num_frame_per_block_first: int = 4,
) -> int:
    """Alias for compute_aligned_audio_frames."""
    return compute_aligned_audio_frames(
        total_video_latent_frames,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )


# ============================================================================
# AVCausalMaskBuilder
# ============================================================================

class AVCausalMaskBuilder:
    """Builds causal attention masks for audio-video generation.

    Mask types:
        1. Video self-attention: block-wise causal
        2. Audio self-attention: block-wise causal
        3. A2V cross-attention (Q=video, K=audio): block-aligned causal
        4. V2A cross-attention (Q=audio, K=video): block-aligned causal

    No sink tokens. Block 0 follows the same rules as any other block:
    video tokens in block 0 attend to audio tokens in block 0 (intra-block
    bidirectional) and nothing else.
    """

    def __init__(
        self,
        video_frame_seqlen: int = 384,
        audio_frame_seqlen: int = 1,
        num_frame_per_block: int = 3,
        num_frame_per_block_first: int = 4,
    ):
        if not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError(
                "FlexAttention is REQUIRED for causal LTX-2 but is not available. "
                "Install PyTorch 2.2+ with CUDA support."
            )

        self.video_frame_seqlen = video_frame_seqlen
        self.audio_frame_seqlen = audio_frame_seqlen
        self.num_frame_per_block = num_frame_per_block
        self.num_frame_per_block_first = num_frame_per_block_first

    # ------------------------------------------------------------------
    # Video self-attention
    # ------------------------------------------------------------------

    def build_video_self_causal_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        """Returns a dense bool mask [total_video_tokens, total_video_tokens].

        Uses PyTorch SDPA instead of FlexAttention to avoid Triton backward
        shared-memory issues on certain GPU architectures.
        """
        total_video_frames = blocks[-1].video_end
        total_tokens = total_video_frames * self.video_frame_seqlen

        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
        for block in blocks:
            t_start = block.video_start * self.video_frame_seqlen
            t_end = block.video_end * self.video_frame_seqlen
            # Tokens in this block can see all tokens up to this block's end
            mask[t_start:t_end, :t_end] = True
        return mask

    # ------------------------------------------------------------------
    # Audio self-attention
    # ------------------------------------------------------------------

    def build_audio_self_causal_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> Optional[torch.Tensor]:
        """Returns a dense bool mask [total_audio_tokens, total_audio_tokens]."""
        total_audio_frames = blocks[-1].audio_end
        if total_audio_frames == 0:
            return None
        total_tokens = total_audio_frames * self.audio_frame_seqlen

        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
        for block in blocks:
            if block.audio_end <= block.audio_start:
                continue
            t_start = block.audio_start * self.audio_frame_seqlen
            t_end = block.audio_end * self.audio_frame_seqlen
            mask[t_start:t_end, :t_end] = True
        return mask

    # ------------------------------------------------------------------
    # A2V cross-attention (Q=video, K/V=audio)
    # ------------------------------------------------------------------

    def build_a2v_causal_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        total_video_frames = blocks[-1].video_end
        total_audio_frames = blocks[-1].audio_end
        num_video_tokens = total_video_frames * self.video_frame_seqlen
        num_audio_tokens = total_audio_frames * self.audio_frame_seqlen

        mask = torch.zeros(
            num_video_tokens, num_audio_tokens,
            dtype=torch.bool, device=device,
        )
        for block in blocks:
            v_token_start = block.video_start * self.video_frame_seqlen
            v_token_end = block.video_end * self.video_frame_seqlen
            a_causal_end = block.audio_end * self.audio_frame_seqlen
            mask[v_token_start:v_token_end, :a_causal_end] = True
        return mask

    # ------------------------------------------------------------------
    # V2A cross-attention (Q=audio, K/V=video)
    # ------------------------------------------------------------------

    def build_v2a_causal_mask(
        self,
        blocks: List[AVBlock],
        device: Union[torch.device, str] = "cuda",
    ) -> torch.Tensor:
        total_video_frames = blocks[-1].video_end
        total_audio_frames = blocks[-1].audio_end
        num_video_tokens = total_video_frames * self.video_frame_seqlen
        num_audio_tokens = total_audio_frames * self.audio_frame_seqlen

        mask = torch.zeros(
            num_audio_tokens, num_video_tokens,
            dtype=torch.bool, device=device,
        )
        for block in blocks:
            a_token_start = block.audio_start * self.audio_frame_seqlen
            a_token_end = block.audio_end * self.audio_frame_seqlen
            v_causal_end = block.video_end * self.video_frame_seqlen
            mask[a_token_start:a_token_end, :v_causal_end] = True
        return mask


# ============================================================================
# Mask Verification
# ============================================================================

def verify_causal_masks(
    masks: dict,
    blocks: List[AVBlock],
    video_frame_seqlen: int,
    audio_frame_seqlen: int,
) -> None:
    """Sanity-check causal mask correctness.

    Validates shape, intra-block bidirectional, and zero future leakage.
    """
    total_video_frames = blocks[-1].video_end
    total_audio_frames = blocks[-1].audio_end
    num_video_tokens = total_video_frames * video_frame_seqlen
    num_audio_tokens = total_audio_frames * audio_frame_seqlen

    for key in ('video_self', 'audio_self', 'a2v', 'v2a'):
        assert key in masks, f"Missing '{key}' mask"

    a2v_mask = masks['a2v']
    v2a_mask = masks['v2a']

    assert a2v_mask.shape == (num_video_tokens, num_audio_tokens), (
        f"A2V mask shape mismatch: {a2v_mask.shape} != "
        f"({num_video_tokens}, {num_audio_tokens})"
    )
    assert v2a_mask.shape == (num_audio_tokens, num_video_tokens), (
        f"V2A mask shape mismatch: {v2a_mask.shape} != "
        f"({num_audio_tokens}, {num_video_tokens})"
    )

    # Block 0 intra-block bidirectional (V0 <-> A0)
    blk0 = blocks[0]
    v0_end = blk0.video_end * video_frame_seqlen
    a0_end = blk0.audio_end * audio_frame_seqlen

    if a0_end > 0:
        assert a2v_mask[:v0_end, :a0_end].all(), (
            "Block 0 video should fully attend to Block 0 audio (A2V)."
        )
        assert v2a_mask[:a0_end, :v0_end].all(), (
            "Block 0 audio should fully attend to Block 0 video (V2A)."
        )

    # Block 0 video must NOT see audio beyond Block 0 in A2V
    if a0_end < num_audio_tokens:
        assert not a2v_mask[:v0_end, a0_end:].any(), (
            "Block 0 video attends to future audio in A2V (causality violation)."
        )

    # Future leakage checks
    for block in blocks:
        v_token_start = block.video_start * video_frame_seqlen
        v_token_end = block.video_end * video_frame_seqlen
        a_causal_end = block.audio_end * audio_frame_seqlen
        if a_causal_end < num_audio_tokens:
            assert not a2v_mask[v_token_start:v_token_end, a_causal_end:].any(), (
                f"A2V future leakage in block {block.block_idx}"
            )
        a_token_start = block.audio_start * audio_frame_seqlen
        a_token_end = block.audio_end * audio_frame_seqlen
        v_causal_end = block.video_end * video_frame_seqlen
        if v_causal_end < num_video_tokens:
            assert not v2a_mask[a_token_start:a_token_end, v_causal_end:].any(), (
                f"V2A future leakage in block {block.block_idx}"
            )

    # Intra-block bidirectional everywhere
    for block in blocks:
        v_token_start = block.video_start * video_frame_seqlen
        v_token_end = block.video_end * video_frame_seqlen
        a_token_start = block.audio_start * audio_frame_seqlen
        a_token_end = block.audio_end * audio_frame_seqlen
        assert a2v_mask[v_token_start:v_token_end, a_token_start:a_token_end].all(), (
            f"A2V intra-block visibility violated in block {block.block_idx}"
        )
        assert v2a_mask[a_token_start:a_token_end, v_token_start:v_token_end].all(), (
            f"V2A intra-block visibility violated in block {block.block_idx}"
        )


# ============================================================================
# Convenience functions
# ============================================================================

def compute_causal_log_scales(
    blocks: List[AVBlock],
    video_frame_seqlen: int = 384,
    audio_frame_seqlen: int = 1,
    device: Union[torch.device, str] = "cuda",
) -> dict:
    """Compute per-token log-ratio scale factors for causal attention outputs.

    Each token sees a strict subset of tokens vs. a bidirectional model.
    Scale = log(1 + visible) / log(1 + total) compensates softmax
    sharpening from the smaller K/V denominator.
    """
    total_video_frames = blocks[-1].video_end
    total_audio_frames = blocks[-1].audio_end
    total_video_tokens = total_video_frames * video_frame_seqlen
    total_audio_tokens = total_audio_frames * audio_frame_seqlen

    log_total_v = math.log(1 + total_video_tokens)
    log_total_a = math.log(1 + total_audio_tokens)

    video_self_scale = torch.ones(total_video_tokens, device=device)
    for block in blocks:
        t_start = block.video_start * video_frame_seqlen
        t_end = block.video_end * video_frame_seqlen
        visible = t_end
        video_self_scale[t_start:t_end] = math.log(1 + visible) / log_total_v

    audio_self_scale = torch.ones(total_audio_tokens, device=device)
    for block in blocks:
        if block.audio_end <= block.audio_start:
            continue
        t_start = block.audio_start * audio_frame_seqlen
        t_end = block.audio_end * audio_frame_seqlen
        visible = t_end
        audio_self_scale[t_start:t_end] = math.log(1 + visible) / log_total_a

    a2v_scale = torch.ones(total_video_tokens, device=device)
    for block in blocks:
        v_start = block.video_start * video_frame_seqlen
        v_end = block.video_end * video_frame_seqlen
        visible_audio = block.audio_end * audio_frame_seqlen
        a2v_scale[v_start:v_end] = math.log(1 + visible_audio) / log_total_a

    v2a_scale = torch.ones(total_audio_tokens, device=device)
    for block in blocks:
        if block.audio_end <= block.audio_start:
            continue
        a_start = block.audio_start * audio_frame_seqlen
        a_end = block.audio_end * audio_frame_seqlen
        visible_video = block.video_end * video_frame_seqlen
        v2a_scale[a_start:a_end] = math.log(1 + visible_video) / log_total_v

    return {
        'video_self_scale': video_self_scale.unsqueeze(0).unsqueeze(-1),
        'audio_self_scale': audio_self_scale.unsqueeze(0).unsqueeze(-1),
        'a2v_scale': a2v_scale.unsqueeze(0).unsqueeze(-1),
        'v2a_scale': v2a_scale.unsqueeze(0).unsqueeze(-1),
    }


def build_all_causal_masks(
    num_video_frames: int,
    num_audio_frames: int,
    config: CausalMaskConfig,
    device: Union[torch.device, str] = "cuda",
) -> dict:
    """Build all causal masks for LTX-2 audio-video generation."""
    num_first = getattr(config, 'num_frame_per_block_first', 4)
    blocks = compute_av_blocks(
        num_video_frames,
        num_frame_per_block=config.num_frame_per_block,
        num_frame_per_block_first=num_first,
    )
    expected_audio = blocks[-1].audio_end if blocks else 0
    assert num_audio_frames == expected_audio, (
        f"Audio frame count ({num_audio_frames}) does not match aligned count "
        f"({expected_audio}) for {num_video_frames} video frames with "
        f"first={num_first}, per_block={config.num_frame_per_block}."
    )

    builder = AVCausalMaskBuilder(
        video_frame_seqlen=config.video_frame_seqlen,
        audio_frame_seqlen=config.audio_frame_seqlen,
        num_frame_per_block=config.num_frame_per_block,
        num_frame_per_block_first=num_first,
    )

    masks = {
        'video_self': builder.build_video_self_causal_mask(blocks, device=device),
        'audio_self': builder.build_audio_self_causal_mask(blocks, device=device),
        'a2v': builder.build_a2v_causal_mask(blocks, device=device),
        'v2a': builder.build_v2a_causal_mask(blocks, device=device),
    }

    verify_causal_masks(
        masks, blocks,
        video_frame_seqlen=config.video_frame_seqlen,
        audio_frame_seqlen=config.audio_frame_seqlen,
    )
    return masks
