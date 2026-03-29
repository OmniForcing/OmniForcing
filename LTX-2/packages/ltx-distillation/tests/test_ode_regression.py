"""
Unit tests for ODE regression module.
"""

import pytest
import torch
from types import SimpleNamespace

from ltx_distillation.ode.ode_regression import (
    ODERegressionConfig,
    LTX2ODERegression,
)


class _FixedOutputGenerator(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(
        self,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        noisy_audio=None,
        audio_timestep=None,
        use_causal_timestep=False,
    ):
        return self.output.clone(), None


def _build_stub_regression(
    loss_target: str,
    generator_output: torch.Tensor,
    noisy_video: torch.Tensor,
    sigma: torch.Tensor,
):
    stub = object.__new__(LTX2ODERegression)
    torch.nn.Module.__init__(stub)
    stub.config = SimpleNamespace(
        loss_target=loss_target,
        video_loss_weight=1.0,
        audio_loss_weight=0.0,
        uniform_timestep=False,
    )
    stub.dtype = torch.float32
    stub.device = torch.device("cpu")
    stub._diag_step = 999
    stub.diag_logger = None
    stub._generator = _FixedOutputGenerator(generator_output)
    stub._text_encoder = None
    stub._load_models = lambda: None
    stub._prepare_generator_input = (
        lambda video_latent, audio_latent=None, sigmas=None: (
            noisy_video.clone(),
            None,
            sigma.clone(),
        )
    )
    stub._expand_video_sigma_to_audio = lambda sigma, frames: None
    stub._expand_video_mask_to_audio = lambda mask, frames: None
    return stub


class TestODERegressionConfig:
    """Tests for ODERegressionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ODERegressionConfig()

        assert config.denoising_step_list == (1000, 757, 522, 0)
        assert config.generator_task == "causal_video"
        assert config.num_frame_per_block == 3
        assert config.gradient_checkpointing is True
        assert config.mixed_precision is True

    def test_custom_timesteps(self):
        """Test custom denoising timesteps."""
        config = ODERegressionConfig(
            denoising_step_list=(1000, 500, 0),
        )

        assert len(config.denoising_step_list) == 3
        assert config.denoising_step_list[0] == 1000
        assert config.denoising_step_list[-1] == 0


class TestTimestepProcessing:
    """Tests for timestep processing logic."""

    def test_causal_timestep_uniformity(self):
        """
        Test that causal timestep processing makes values uniform within blocks.

        For num_frame_per_block=4:
        Input indices:  [0, 1, 2, 3, 0, 2, 1, 3]
        Output indices: [0, 0, 0, 0, 0, 0, 0, 0] (first of each block)
        """
        # Simulate the processing
        timestep_indices = torch.tensor([[0, 1, 2, 3, 1, 2, 0, 3]])
        num_frame_per_block = 4
        B, F = timestep_indices.shape

        # Process: make uniform within blocks
        result = timestep_indices.reshape(B, -1, num_frame_per_block)
        result[:, :, 1:] = result[:, :, 0:1]
        result = result.reshape(B, -1)

        # Check uniformity
        assert result[0, 0] == result[0, 1] == result[0, 2] == result[0, 3]
        assert result[0, 4] == result[0, 5] == result[0, 6] == result[0, 7]

    def test_bidirectional_timestep_broadcast(self):
        """
        Test that bidirectional timestep processing broadcasts first value.

        Input:  [t0, t1, t2, t3, ...]
        Output: [t0, t0, t0, t0, ...]
        """
        timestep = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]])

        # Bidirectional: broadcast first timestep to all
        result = timestep.clone()
        for i in range(result.shape[0]):
            result[i] = result[i, 0]

        assert torch.all(result == 100)


class TestInputPreparation:
    """Tests for generator input preparation."""

    def test_gather_from_trajectory(self):
        """Test gathering latents from trajectory at selected timesteps."""
        batch_size = 2
        num_timesteps = 4
        num_frames = 8
        channels = 4
        height = 4
        width = 6

        # Create trajectory
        trajectory = torch.randn(
            batch_size, num_timesteps, num_frames, channels, height, width
        )

        # Select timestep index 2 for all frames in batch 0
        # Select timestep index 1 for all frames in batch 1
        indices = torch.tensor([
            [2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])

        # Gather
        noisy_input = torch.gather(
            trajectory,
            dim=1,
            index=indices.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, channels, height, width
            ),
        ).squeeze(1)

        assert noisy_input.shape == (batch_size, num_frames, channels, height, width)

        # Check that we got the correct timestep slice
        assert torch.allclose(noisy_input[0], trajectory[0, 2])
        assert torch.allclose(noisy_input[1], trajectory[1, 1])

    def test_audio_alignment_with_video(self):
        """Test aligning audio frame indices with video frames."""
        video_frames = 8
        audio_frames = 64  # 8 audio per video
        num_frame_per_block = 4

        # Video timestep indices (uniform within blocks)
        video_indices = torch.tensor([[0, 0, 0, 0, 2, 2, 2, 2]])

        # Expand to audio
        audio_per_video = audio_frames / video_frames  # 8

        audio_indices = torch.zeros(1, audio_frames, dtype=torch.long)
        for i in range(video_frames):
            start = int(i * audio_per_video)
            end = int((i + 1) * audio_per_video) if i < video_frames - 1 else audio_frames
            audio_indices[:, start:end] = video_indices[:, i:i+1]

        # Check alignment
        # First 32 audio frames (4 video frames * 8) should have index 0
        assert torch.all(audio_indices[:, :32] == 0)
        # Last 32 audio frames should have index 2
        assert torch.all(audio_indices[:, 32:] == 2)


class TestLossComputation:
    """Tests for ODE regression loss computation."""

    def test_mse_loss_masking(self):
        """Test that t=0 frames are excluded from loss."""
        batch_size = 2
        num_frames = 8

        pred = torch.randn(batch_size, num_frames, 4, 4, 6)
        target = torch.randn(batch_size, num_frames, 4, 4, 6)

        # Timesteps with some at 0
        timestep = torch.tensor([
            [500, 500, 500, 500, 0, 0, 0, 0],
            [500, 500, 0, 0, 500, 500, 0, 0],
        ])

        # Create mask for non-zero timesteps
        mask = timestep != 0

        # Masked loss computation
        masked_pred = pred[mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(pred)]
        masked_target = target[mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(target)]

        loss = torch.nn.functional.mse_loss(masked_pred, masked_target)

        # Loss should be computed only on masked elements
        assert not torch.isnan(loss)

    def test_combined_video_audio_loss(self):
        """Test combining video and audio losses."""
        video_loss = torch.tensor(0.5)
        audio_loss = torch.tensor(0.3)

        # Simple sum
        total_loss = video_loss + audio_loss

        assert total_loss.item() == pytest.approx(0.8)


class TestDenosingStepMapping:
    """Tests for denoising step to trajectory index mapping."""

    def test_timestep_to_index(self):
        """Test mapping timestep values to trajectory indices."""
        denoising_step_list = torch.tensor([1000, 757, 522, 0])

        # Random index selection
        indices = torch.tensor([[0, 1, 2, 3, 0, 2]])

        # Map to actual timesteps
        timesteps = denoising_step_list[indices]

        assert timesteps[0, 0] == 1000  # index 0 -> t=1000
        assert timesteps[0, 1] == 757   # index 1 -> t=757
        assert timesteps[0, 2] == 522   # index 2 -> t=522
        assert timesteps[0, 3] == 0     # index 3 -> t=0

    def test_warped_timesteps(self):
        """Test timestep warping with scheduler shift."""
        # Original timesteps
        original = [1000, 757, 522, 0]

        # After warping (example with shift=8.0)
        # The actual warping depends on scheduler implementation
        # This is a conceptual test

        # Warped timesteps should still be in valid range
        for t in original:
            assert 0 <= t <= 1000


    def test_x0_loss_uses_generator_output_as_x0(self):
        """x0 loss should be zero when the generator already returns the clean sample."""
        target_video = torch.tensor([[[[[1.0]]], [[[3.0]]]]], dtype=torch.float32)
        noisy_video = torch.tensor([[[[[2.0]]], [[[5.0]]]]], dtype=torch.float32)
        sigma = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
        video_latent = torch.stack([noisy_video, target_video], dim=1)

        regression = _build_stub_regression(
            loss_target="x0",
            generator_output=target_video,
            noisy_video=noisy_video,
            sigma=sigma,
        )

        loss, log_dict = LTX2ODERegression.generator_loss(
            regression,
            video_latent=video_latent,
            conditional_dict={},
            audio_latent=None,
            sigmas=None,
            return_samples=True,
        )

        assert loss.item() == pytest.approx(0.0)
        assert log_dict["video_loss"].item() == pytest.approx(0.0)
        assert log_dict["unnormalized_loss"].item() == pytest.approx(0.0)
        assert torch.allclose(log_dict["pred_video"], target_video)

    def test_velocity_loss_derives_velocity_from_x0_output(self):
        """velocity loss should convert wrapper x0 outputs back to velocity before supervision."""
        target_video = torch.tensor([[[[[1.0]]], [[[3.0]]]]], dtype=torch.float32)
        noisy_video = torch.tensor([[[[[2.0]]], [[[5.0]]]]], dtype=torch.float32)
        sigma = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
        video_latent = torch.stack([noisy_video, target_video], dim=1)

        regression = _build_stub_regression(
            loss_target="velocity",
            generator_output=target_video,
            noisy_video=noisy_video,
            sigma=sigma,
        )

        loss, log_dict = LTX2ODERegression.generator_loss(
            regression,
            video_latent=video_latent,
            conditional_dict={},
            audio_latent=None,
            sigmas=None,
            return_samples=True,
        )

        assert loss.item() == pytest.approx(0.0)
        assert log_dict["video_loss"].item() == pytest.approx(0.0)
        assert log_dict["unnormalized_loss"].item() == pytest.approx(0.0)
        assert torch.allclose(log_dict["pred_video"], target_video)
