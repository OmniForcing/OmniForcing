"""
Causal autoregressive AV inference pipeline (KV-cache).

Mirrors Self-Forcing's `pipeline/causal_inference.py`:
  1. Allocate per-layer KV caches (video_self, audio_self, A2V, V2A,
     video_text, audio_text) once.
  2. For each block, run the denoising_step_list against pure noise.
  3. After each block, re-noise the denoised output with `context_noise`
     and run a no_grad refresh forward to overwrite cache entries with
     the representation the next block expects to read.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ltx_causal.attention.mask_builder import (
    compute_aligned_audio_frames,
    compute_av_blocks,
)


class CausalAVInferencePipeline:
    """KV-cache autoregressive inference for LTX-2 AV.

    The generator argument must be a `CausalLTX2DiffusionWrapper` (it exposes
    `init_av_kv_caches` and `forward_with_cache`).
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        num_frame_per_block: int = 3,
        num_frame_per_block_first: int = 4,
        context_noise: int = 0,
        num_train_timestep: int = 1000,
        clear_cuda_cache_per_round: bool = True,
        # Kept for backward-compatible config keys (no-op now).
        use_kv_cache: bool = True,
    ):
        if denoising_sigmas.ndim != 1 or denoising_sigmas.numel() < 2:
            raise ValueError(
                "denoising_sigmas must be a 1D tensor with at least 2 entries"
            )
        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.num_frame_per_block = max(1, int(num_frame_per_block))
        self.num_frame_per_block_first = max(1, int(num_frame_per_block_first))
        self.context_noise = int(context_noise)
        self.num_train_timestep = int(num_train_timestep)
        self.clear_cuda_cache_per_round = bool(clear_cuda_cache_per_round)
        del use_kv_cache  # legacy; pipeline always uses KV cache

    # ---------------- helpers ----------------

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
        param = next(module.parameters())
        return param.device, param.dtype

    def _full_sigma(
        self,
        sigma: torch.Tensor,
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return sigma.to(device=device, dtype=dtype).expand(batch_size, frames)

    def _renoise_block(self, clean_block: torch.Tensor, next_sigma: torch.Tensor) -> torch.Tensor:
        if clean_block is None:
            return None
        sigma = self._full_sigma(
            next_sigma,
            batch_size=clean_block.shape[0],
            frames=clean_block.shape[1],
            device=clean_block.device,
            dtype=clean_block.dtype,
        )
        return self.add_noise_fn(clean_block, torch.randn_like(clean_block), sigma)

    # ---------------- generate ----------------

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

        # Under FSDP, parameter tensors may be sharded/empty on non-rank-0.
        # Use cuda current device and a safe dtype probe.
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
        dtype = torch.bfloat16
        try:
            for p in self.generator.parameters():
                if p.numel() > 0:
                    dtype = p.dtype
                    break
        except Exception:
            pass

        batch_size = video_shape[0]
        total_video_frames = video_shape[1]
        blocks = compute_av_blocks(
            total_video_latent_frames=total_video_frames,
            num_frame_per_block=self.num_frame_per_block,
            num_frame_per_block_first=self.num_frame_per_block_first,
        )

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = None
        total_audio_frames = 0
        audio_channels = None
        if audio_shape is not None:
            if len(audio_shape) != 3:
                raise ValueError(f"Expected audio_shape=[B,F,C], got {audio_shape}")
            expected_audio_frames = compute_aligned_audio_frames(
                total_video_latent_frames=total_video_frames,
                num_frame_per_block=self.num_frame_per_block,
                num_frame_per_block_first=self.num_frame_per_block_first,
            )
            if audio_shape[1] != expected_audio_frames:
                raise ValueError(
                    "audio_shape does not match causal block alignment: "
                    f"got F_a={audio_shape[1]}, expected {expected_audio_frames}"
                )
            audio = torch.zeros(audio_shape, device=device, dtype=dtype)
            total_audio_frames = audio_shape[1]
            audio_channels = audio_shape[2]

        # Allocate KV caches once. Cache buffers are independent of FSDP-sharded
        # parameters, so we can access the unwrapped wrapper directly.
        gen = self.generator
        while hasattr(gen, "module"):
            gen = gen.module
        text_seq_len = conditional_dict["video_context"].shape[1]
        kv_caches = gen.init_av_kv_caches(
            batch_size=batch_size,
            max_video_frames=total_video_frames,
            max_audio_frames=total_audio_frames,
            text_seq_len=text_seq_len,
            device=device,
            dtype=dtype,
        )

        for block in blocks:
            v_len = block.video_end - block.video_start
            a_len = block.audio_end - block.audio_start

            current_video = torch.randn(
                (batch_size, v_len, *video_shape[2:]),
                device=device, dtype=dtype,
            )
            current_audio = None
            if audio is not None:
                current_audio = torch.randn(
                    (batch_size, a_len, audio_channels),
                    device=device, dtype=dtype,
                )

            # --- Block denoising loop ---
            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                v_sigma = self._full_sigma(sigma, batch_size, v_len, device, dtype)
                a_sigma = (
                    self._full_sigma(sigma, batch_size, a_len, device, dtype)
                    if current_audio is not None else None
                )
                pred_v, pred_a = self.generator(
                    noisy_image_or_video=current_video,
                    conditional_dict=conditional_dict,
                    timestep=v_sigma,
                    noisy_audio=current_audio,
                    audio_timestep=a_sigma,
                    kv_caches=kv_caches,
                    current_video_start_frame=block.video_start,
                    current_audio_start_frame=block.audio_start,
                )

                next_sigma = self.denoising_sigmas[sigma_idx + 1]
                if float(next_sigma.item()) > 0.0:
                    current_video = self._renoise_block(pred_v, next_sigma)
                    if pred_a is not None:
                        current_audio = self._renoise_block(pred_a, next_sigma)
                else:
                    current_video = pred_v
                    if pred_a is not None:
                        current_audio = pred_a

            # `pred_v`/`pred_a` from the last loop iteration are the denoised
            # outputs at the final non-zero sigma; treat them as the block's
            # denoised result.
            denoised_v = pred_v
            denoised_a = pred_a if current_audio is not None else None

            video[:, block.video_start:block.video_end] = denoised_v
            if audio is not None and denoised_a is not None:
                audio[:, block.audio_start:block.audio_end] = denoised_a

            # --- Context-noise cache refresh ---
            ctx_t = float(self.context_noise) / float(self.num_train_timestep)
            ctx_sigma_v = torch.full(
                (batch_size, v_len), ctx_t, device=device, dtype=dtype
            )
            if ctx_t > 0.0:
                noisy_ctx_v = self.add_noise_fn(
                    denoised_v, torch.randn_like(denoised_v), ctx_sigma_v,
                )
            else:
                noisy_ctx_v = denoised_v

            noisy_ctx_a = None
            ctx_sigma_a = None
            if denoised_a is not None:
                ctx_sigma_a = torch.full(
                    (batch_size, a_len), ctx_t, device=device, dtype=dtype
                )
                if ctx_t > 0.0:
                    noisy_ctx_a = self.add_noise_fn(
                        denoised_a, torch.randn_like(denoised_a), ctx_sigma_a,
                    )
                else:
                    noisy_ctx_a = denoised_a

            self.generator(
                noisy_image_or_video=noisy_ctx_v,
                conditional_dict=conditional_dict,
                timestep=ctx_sigma_v,
                noisy_audio=noisy_ctx_a,
                audio_timestep=ctx_sigma_a,
                kv_caches=kv_caches,
                current_video_start_frame=block.video_start,
                current_audio_start_frame=block.audio_start,
            )

            if self.clear_cuda_cache_per_round and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return video, audio
