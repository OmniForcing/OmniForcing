"""
Causal RoPE: Rotary Position Embedding with temporal offset for causal training.

This module implements:
- causal_precompute_freqs_cis: Precompute RoPE frequencies with frame offset
- apply_interleaved_rotary_emb: Apply interleaved rotary embedding
- Supports both video (3D positions) and audio (1D positions)

Positions are in PHYSICAL coordinates (seconds / pixels), matching the
original LTX-2 pipeline.
"""

import math
from enum import Enum
from typing import Tuple, Optional, List, Union

import torch
from einops import rearrange


class CausalRopeType(Enum):
    """RoPE type compatible with LTX-2."""
    INTERLEAVED = "interleaved"
    SPLIT = "split"


# ============================================================================
# LTX-2 Physical Coordinate Constants
# ============================================================================
# Positions must be in PHYSICAL space (seconds / pixels), NOT latent indices.
# Using latent indices causes ~33x spatial error and ~3-5x temporal error,
# leading to gradient norms of 1e+15.

_VIDEO_TEMPORAL_SCALE = 8        # VAE temporal downsampling factor
_VIDEO_SPATIAL_SCALE = 32        # VAE spatial downsampling factor
_VIDEO_FPS = 24                  # Video frames per second
_AUDIO_DOWNSAMPLE_FACTOR = 4     # Audio latent temporal downsample factor
_AUDIO_HOP_LENGTH = 160          # Mel spectrogram hop length
_AUDIO_SAMPLE_RATE = 16000       # Audio mel processing sample rate


# ============================================================================
# Frequency Grid Generation
# ============================================================================

def generate_freq_grid(
    theta: float,
    max_pos_count: int,
    inner_dim: int,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Generate frequency grid for RoPE.

    Compatible with LTX-2's generate_freq_grid_pytorch.

    Args:
        theta: Base frequency (typically 10000)
        max_pos_count: Number of position dimensions
        inner_dim: Inner dimension of attention
        device: Target device

    Returns:
        Frequency indices tensor
    """
    start = 1
    end = theta
    n_elem = 2 * max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
            device=device,
        )
    )
    indices = indices * math.pi / 2

    return indices


# ============================================================================
# Standard RoPE Application
# ============================================================================

def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply interleaved rotary embedding (pairs adjacent dimensions)."""
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs
    return out


# ============================================================================
# Causal RoPE with Frame Offset
# ============================================================================

def causal_precompute_freqs_cis(
    grid_sizes: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
    max_pos: Optional[List[int]] = None,
    start_frame: int = 0,
    rope_type: CausalRopeType = CausalRopeType.INTERLEAVED,
    device: Union[torch.device, str] = "cuda",
    dtype: torch.dtype = torch.float32,
    is_audio: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE frequencies with causal frame offset.

    CRITICAL: Positions are in PHYSICAL coordinates (seconds / pixels), matching
    the original LTX-2 pipeline (ltx-core). Using bare latent indices causes
    gradient norm explosion (~1e15).

    For video 3D:
        - Temporal: pixel frames with causal fix → seconds (/ fps=24)
        - Spatial: pixel midpoints (latent * 32 + 16)
    For audio 1D (is_audio=True):
        - Temporal: mel frames with causal fix → seconds (* hop/sr)
    For video temporal 1D (is_audio=False):
        - Same as video temporal axis

    Args:
        grid_sizes: Position grid sizes [B, num_dims] or [B, num_dims, 2]
                   For video: [B, 3] with (F, H, W)
                   For audio: [B, 1] with (F,)
        dim: Inner dimension for RoPE
        theta: Base frequency
        max_pos: Maximum positions per dimension (default [20, 2048, 2048])
        start_frame: Starting frame offset for causal generation
        rope_type: INTERLEAVED or SPLIT
        device: Target device
        dtype: Output dtype
        is_audio: If True and num_pos_dims==1, use audio timing conversion.
                  If False and num_pos_dims==1, use video temporal conversion.

    Returns:
        (cos_freq, sin_freq): Tuple of frequency tensors
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    # Handle 2D grid_sizes (start/end format) by taking mean
    if grid_sizes.ndim == 3:
        grid_sizes = (grid_sizes[..., 0] + grid_sizes[..., 1]) / 2.0

    num_pos_dims = grid_sizes.shape[1]

    # Generate frequency indices
    indices = generate_freq_grid(theta, num_pos_dims, dim, device)

    # Build position indices with frame offset
    all_freqs = []

    for batch_idx in range(grid_sizes.shape[0]):
        sizes = grid_sizes[batch_idx].tolist()
        seq_len = int(math.prod(sizes))

        if num_pos_dims == 3:
            # Video: 3D positions (F, H, W) in PHYSICAL coordinates
            f, h, w = [int(s) for s in sizes]

            # Temporal: latent frame → pixel bounds → midpoint in seconds
            # Matches get_pixel_coords() + causal_fix + /fps in ltx-core
            latent_t = torch.arange(start_frame, start_frame + f, device=device, dtype=torch.float32)
            pixel_t_start = (latent_t * _VIDEO_TEMPORAL_SCALE + 1 - _VIDEO_TEMPORAL_SCALE).clamp(min=0)
            pixel_t_end = ((latent_t + 1) * _VIDEO_TEMPORAL_SCALE + 1 - _VIDEO_TEMPORAL_SCALE).clamp(min=0)
            t_seconds = (pixel_t_start + pixel_t_end) / 2.0 / _VIDEO_FPS

            # Spatial: latent coord → pixel bounds → midpoint
            # Matches get_pixel_coords() (no causal fix for spatial)
            h_pixels = torch.arange(h, device=device, dtype=torch.float32) * _VIDEO_SPATIAL_SCALE + _VIDEO_SPATIAL_SCALE / 2.0
            w_pixels = torch.arange(w, device=device, dtype=torch.float32) * _VIDEO_SPATIAL_SCALE + _VIDEO_SPATIAL_SCALE / 2.0

            # Normalize by max_pos (matching get_fractional_positions)
            t_frac = t_seconds / max_pos[0]
            h_frac = h_pixels / max_pos[1]
            w_frac = w_pixels / max_pos[2]

            # Build meshgrid: [F, H, W, 3]
            grid_t, grid_h, grid_w = torch.meshgrid(t_frac, h_frac, w_frac, indexing='ij')
            fractional_positions = torch.stack([grid_t, grid_h, grid_w], dim=-1)
            fractional_positions = fractional_positions.reshape(seq_len, num_pos_dims)

            # Compute frequencies matching original generate_freqs:
            #   freqs = (indices * (frac * 2 - 1)).transpose(-1, -2).flatten(2)
            # This interleaves: [freq0_t, freq0_h, freq0_w, freq1_t, freq1_h, freq1_w, ...]
            freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
            freqs = freqs.transpose(-1, -2).flatten(1)

        elif num_pos_dims == 1:
            # 1D temporal positions
            f = int(sizes[0])
            latent_t = torch.arange(start_frame, start_frame + f, device=device, dtype=torch.float32)

            if is_audio:
                # Audio: latent frame → mel frame with causal fix → seconds
                # Matches AudioPatchifier._get_audio_latent_time_in_sec()
                mel_start = (latent_t * _AUDIO_DOWNSAMPLE_FACTOR + 1 - _AUDIO_DOWNSAMPLE_FACTOR).clamp(min=0)
                mel_end = ((latent_t + 1) * _AUDIO_DOWNSAMPLE_FACTOR + 1 - _AUDIO_DOWNSAMPLE_FACTOR).clamp(min=0)
                t_seconds = (mel_start + mel_end) / 2.0 * _AUDIO_HOP_LENGTH / _AUDIO_SAMPLE_RATE
            else:
                # Video temporal: latent frame → pixel frame with causal fix → seconds
                # Same conversion as the temporal axis of the 3D video case
                pixel_t_start = (latent_t * _VIDEO_TEMPORAL_SCALE + 1 - _VIDEO_TEMPORAL_SCALE).clamp(min=0)
                pixel_t_end = ((latent_t + 1) * _VIDEO_TEMPORAL_SCALE + 1 - _VIDEO_TEMPORAL_SCALE).clamp(min=0)
                t_seconds = (pixel_t_start + pixel_t_end) / 2.0 / _VIDEO_FPS

            t_frac = t_seconds / max_pos[0]

            # For 1D: fractional_positions = [seq_len, 1]
            fractional_positions = t_frac.unsqueeze(-1)
            freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
            freqs = freqs.transpose(-1, -2).flatten(1)

        else:
            raise ValueError(f"Unsupported num_pos_dims: {num_pos_dims}")

        all_freqs.append(freqs)

    # Stack batches
    freqs = torch.stack(all_freqs, dim=0)  # [B, seq_len, D_freq * num_pos_dims]

    # Only INTERLEAVED mode is supported. SPLIT mode produces incorrect tensor
    # shapes (3D vs original's 4D after head reshape) and has never been tested.
    if rope_type != CausalRopeType.INTERLEAVED:
        raise ValueError(
            f"Only CausalRopeType.INTERLEAVED is supported, got {rope_type}. "
            f"SPLIT mode is not implemented correctly for causal generation."
        )

    # Compute cos/sin with padding matching original interleaved_freqs_cis
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)

    # Pad size matches original: dim % (2 * num_pos_dims)
    n_elem = 2 * num_pos_dims
    pad_size = dim % n_elem
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[..., :pad_size])
        sin_padding = torch.zeros_like(sin_freq[..., :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

    return cos_freq.to(dtype), sin_freq.to(dtype)


