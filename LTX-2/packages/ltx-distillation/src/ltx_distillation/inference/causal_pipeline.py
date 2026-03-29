"""
Causal benchmark inference pipeline for LTX-2 AV generation.

This pipeline keeps the CausVid-style autoregressive outer loop semantics:
each block is denoised from noise, conditioned on previously generated blocks.
Since the tracked causal wrapper still does not expose a stable KV-cache runtime
API, the implementation uses prefix-rerun instead of incremental cache updates.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ltx_causal.attention.mask_builder import (
    compute_aligned_audio_frames,
    compute_av_blocks,
)
class CausalAVInferencePipeline:
    """
    Prefix-rerun autoregressive pipeline for causal AV benchmark inference.

    `use_kv_cache` is kept for config compatibility, but the current causal
    wrapper does not expose a runnable KV-cache runtime API. We therefore keep
    the CausVid-style block-by-block outer loop but realize it through
    prefix-rerun instead of cache updates.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        num_frame_per_block: int = 3,
        use_kv_cache: bool = False,
        clear_cuda_cache_per_round: bool = True,
    ):
        if denoising_sigmas.ndim != 1 or denoising_sigmas.numel() < 2:
            raise ValueError(
                "denoising_sigmas must be a 1D tensor with at least 2 entries"
            )

        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.num_frame_per_block = max(1, int(num_frame_per_block))
        self.use_kv_cache_requested = bool(use_kv_cache)
        self.clear_cuda_cache_per_round = bool(clear_cuda_cache_per_round)

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
        param = next(module.parameters())
        return param.device, param.dtype

    @staticmethod
    def _zeros_sigma(
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, frames), device=device, dtype=dtype)

    @staticmethod
    def _full_sigma(
        sigma: torch.Tensor,
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sigma_value = sigma.to(device=device, dtype=dtype)
        return sigma_value.expand(batch_size, frames)

    def _renoise_block(self, clean_block: torch.Tensor, next_sigma: torch.Tensor) -> torch.Tensor:
        if clean_block is None:
            return None

        batch_size = clean_block.shape[0]
        num_frames = clean_block.shape[1]
        sigma = self._full_sigma(
            next_sigma,
            batch_size=batch_size,
            frames=num_frames,
            device=clean_block.device,
            dtype=clean_block.dtype,
        )
        return self.add_noise_fn(
            clean_block,
            torch.randn_like(clean_block),
            sigma,
        )

    @torch.no_grad()
    def generate(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Optional[Tuple[int, ...]],
        conditional_dict: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if len(video_shape) != 5:
            raise ValueError(f"Expected video_shape=[B,F,C,H,W], got {video_shape}")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        device, dtype = self._module_device_dtype(self.generator)

        batch_size = video_shape[0]
        total_video_frames = video_shape[1]
        blocks = compute_av_blocks(
            total_video_latent_frames=total_video_frames,
            num_frame_per_block=self.num_frame_per_block,
        )

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = None

        if audio_shape is not None:
            if len(audio_shape) != 3:
                raise ValueError(f"Expected audio_shape=[B,F,C], got {audio_shape}")
            expected_audio_frames = compute_aligned_audio_frames(
                total_video_latent_frames=total_video_frames,
                num_frame_per_block=self.num_frame_per_block,
            )
            if audio_shape[1] != expected_audio_frames:
                raise ValueError(
                    "audio_shape does not match causal block alignment: "
                    f"got F_a={audio_shape[1]}, expected {expected_audio_frames}"
                )
            audio = torch.zeros(audio_shape, device=device, dtype=dtype)

        for block in blocks:
            current_video = torch.randn(
                (batch_size, block.video_frames, *video_shape[2:]),
                device=device,
                dtype=dtype,
            )
            current_audio = None
            if audio is not None:
                current_audio = torch.randn(
                    (batch_size, block.audio_frames, audio_shape[2]),
                    device=device,
                    dtype=dtype,
                )

            prev_video = video[:, :block.video_start]
            prev_audio = audio[:, :block.audio_start] if audio is not None else None

            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                prefix_video = torch.cat([prev_video, current_video], dim=1)
                video_sigma = torch.cat(
                    [
                        self._zeros_sigma(batch_size, prev_video.shape[1], device, dtype),
                        self._full_sigma(sigma, batch_size, current_video.shape[1], device, dtype),
                    ],
                    dim=1,
                )

                prefix_audio = None
                audio_sigma = None
                if current_audio is not None:
                    prefix_audio = torch.cat([prev_audio, current_audio], dim=1)
                    audio_sigma = torch.cat(
                        [
                            self._zeros_sigma(batch_size, prev_audio.shape[1], device, dtype),
                            self._full_sigma(sigma, batch_size, current_audio.shape[1], device, dtype),
                        ],
                        dim=1,
                    )

                pred_video_prefix, pred_audio_prefix = self.generator(
                    noisy_image_or_video=prefix_video,
                    conditional_dict=conditional_dict,
                    timestep=video_sigma,
                    noisy_audio=prefix_audio,
                    audio_timestep=audio_sigma,
                    use_causal_timestep=False,
                    force_bidirectional=False,
                )

                current_video = pred_video_prefix[:, block.video_start:block.video_end]
                if current_audio is not None:
                    if pred_audio_prefix is None:
                        raise RuntimeError(
                            "Generator returned no audio prediction for audio benchmark inference"
                        )
                    current_audio = pred_audio_prefix[:, block.audio_start:block.audio_end]

                next_sigma = self.denoising_sigmas[sigma_idx + 1]
                if float(next_sigma.item()) > 0.0:
                    current_video = self._renoise_block(current_video, next_sigma)
                    if current_audio is not None:
                        current_audio = self._renoise_block(current_audio, next_sigma)

                if self.clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

            video[:, block.video_start:block.video_end] = current_video
            if audio is not None and current_audio is not None:
                audio[:, block.audio_start:block.audio_end] = current_audio

        return video, audio
