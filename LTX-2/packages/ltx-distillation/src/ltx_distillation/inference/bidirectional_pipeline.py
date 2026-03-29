"""
Bidirectional Audio-Video Trajectory Pipeline for DMD backward simulation.

This pipeline generates denoising trajectories for backward simulation
in DMD training. It runs the generator through multiple denoising steps
and returns the intermediate states.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class BidirectionalAVTrajectoryPipeline:
    """
    Pipeline for generating audio-video denoising trajectories.

    Used in DMD training for backward simulation:
    1. Start from pure noise
    2. Denoise through multiple steps using the generator
    3. Return trajectory of intermediate states

    The trajectory can be used to sample training inputs at different noise levels.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
    ):
        """
        Args:
            generator: LTX2DiffusionWrapper instance
            add_noise_fn: Callable[[original, noise, sigma], noisy_sample]
                         Flow matching noise addition: (1-sigma)*x0 + sigma*eps
            denoising_sigmas: Tensor of sigma values for denoising steps
        """
        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas

    @torch.no_grad()
    def inference_with_trajectory(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        conditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate denoising trajectory from noise.

        This implements consistency backward simulation:
        At each step, predict x0 and then re-corrupt to the next noise level.

        Args:
            video_noise: Initial video noise [B, F_v, C, H, W]
            audio_noise: Initial audio noise [B, F_a, C]
            conditional_dict: Conditioning dictionary

        Returns:
            Tuple of:
                - video_trajectory: [B, T, F_v, C, H, W] where T is num steps
                - audio_trajectory: [B, T, F_a, C]
        """
        B = video_noise.shape[0]
        F_v = video_noise.shape[1]
        F_a = audio_noise.shape[1]
        device = video_noise.device

        video_trajectory = [video_noise]
        audio_trajectory = [audio_noise]

        noisy_video = video_noise
        noisy_audio = audio_noise

        # Iterate through denoising steps (except the last one which is t=0)
        for i, sigma in enumerate(self.denoising_sigmas[:-1]):
            # Prepare sigma tensors
            video_sigma = sigma * torch.ones([B, F_v], device=device)
            audio_sigma = sigma * torch.ones([B, F_a], device=device)

            # Predict x0
            pred_video, pred_audio = self.generator(
                noisy_image_or_video=noisy_video,
                conditional_dict=conditional_dict,
                timestep=video_sigma,
                noisy_audio=noisy_audio,
                audio_timestep=audio_sigma,
            )

            # Get next sigma
            next_sigma = self.denoising_sigmas[i + 1]

            if next_sigma > 0:
                # Re-corrupt with next sigma level
                # For flow matching: x_t = (1 - sigma) * x_0 + sigma * eps
                # We need to add noise at the next sigma level

                # Sample fresh noise
                fresh_noise_video = torch.randn_like(video_noise)
                fresh_noise_audio = torch.randn_like(audio_noise)

                next_video_sigma = next_sigma * torch.ones([B, F_v], device=device)
                next_audio_sigma = next_sigma * torch.ones([B, F_a], device=device)

                noisy_video = self.add_noise_fn(
                    pred_video.flatten(0, 1),
                    fresh_noise_video.flatten(0, 1),
                    next_video_sigma.flatten(0, 1),
                ).unflatten(0, (B, F_v))

                noisy_audio = self.add_noise_fn(
                    pred_audio, fresh_noise_audio, next_audio_sigma
                )
            else:
                # At t=0, just use the prediction
                noisy_video = pred_video
                noisy_audio = pred_audio

            video_trajectory.append(noisy_video)
            audio_trajectory.append(noisy_audio)

        # Stack trajectories: [B, T, F, C, H, W]
        video_trajectory = torch.stack(video_trajectory, dim=1)
        audio_trajectory = torch.stack(audio_trajectory, dim=1)

        return video_trajectory, audio_trajectory


class BidirectionalAVInferencePipeline:
    """
    Pipeline for few-step bidirectional inference.

    Used for validation after training to generate videos/audio
    using the distilled model.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
    ):
        """
        Args:
            generator: Distilled LTX2DiffusionWrapper
            add_noise_fn: Callable[[original, noise, sigma], noisy_sample]
            denoising_sigmas: Sigma values for few-step denoising
        """
        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas

    @torch.no_grad()
    def generate(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Tuple[int, ...],
        conditional_dict: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate video and audio using few-step denoising.

        Args:
            video_shape: (B, F_v, C, H, W) video latent shape
            audio_shape: (B, F_a, C) audio latent shape
            conditional_dict: Text conditioning
            seed: Random seed (optional)

        Returns:
            Tuple of (video_latent, audio_latent)
        """
        B = video_shape[0]
        F_v = video_shape[1]
        F_a = audio_shape[1]

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        device = next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype

        # Initialize with noise
        video = torch.randn(video_shape, device=device, dtype=dtype)
        audio = torch.randn(audio_shape, device=device, dtype=dtype)

        # Few-step denoising
        for i, sigma in enumerate(self.denoising_sigmas[:-1]):
            video_sigma = sigma * torch.ones([B, F_v], device=device)
            audio_sigma = sigma * torch.ones([B, F_a], device=device)

            # Predict x0
            pred_video, pred_audio = self.generator(
                noisy_image_or_video=video,
                conditional_dict=conditional_dict,
                timestep=video_sigma,
                noisy_audio=audio,
                audio_timestep=audio_sigma,
            )

            # Get next sigma
            next_sigma = self.denoising_sigmas[i + 1]

            if next_sigma > 0:
                # Euler step or re-corruption
                fresh_noise_video = torch.randn_like(video)
                fresh_noise_audio = torch.randn_like(audio)

                next_video_sigma = next_sigma * torch.ones([B, F_v], device=device)
                next_audio_sigma = next_sigma * torch.ones([B, F_a], device=device)

                video = self.add_noise_fn(
                    pred_video.flatten(0, 1),
                    fresh_noise_video.flatten(0, 1),
                    next_video_sigma.flatten(0, 1),
                ).unflatten(0, (B, F_v))

                audio = self.add_noise_fn(
                    pred_audio, fresh_noise_audio, next_audio_sigma
                )
            else:
                video = pred_video
                audio = pred_audio

        return video, audio
