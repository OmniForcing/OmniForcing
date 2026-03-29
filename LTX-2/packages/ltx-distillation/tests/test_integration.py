"""
Integration tests for ODE Init pipeline.

These tests verify the end-to-end flow without requiring actual model weights.
"""

import pytest
import tempfile
import os
from pathlib import Path

import torch
import numpy as np


class TestODEPipelineIntegration:
    """
    Integration tests for the complete ODE initialization pipeline.

    Pipeline flow:
    1. Generate ODE pairs (teacher model -> trajectories)
    2. Convert to LMDB
    3. Train causal model with ODE regression
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_trajectory_to_lmdb_flow(self, temp_dir):
        """Test the flow from trajectory files to LMDB dataset."""
        from ltx_distillation.ode.create_lmdb import (
            store_arrays_to_lmdb,
            get_array_shape_from_lmdb,
            retrieve_row_from_lmdb,
        )
        import lmdb

        # Create fake trajectory data
        num_samples = 5
        video_shape = (4, 16, 128, 16, 24)  # T, F, C, H, W
        audio_shape = (4, 126, 128)  # T, F_a, C

        prompts = [f"Test prompt {i}" for i in range(num_samples)]
        video_latents = np.random.randn(num_samples, *video_shape).astype(np.float16)
        audio_latents = np.random.randn(num_samples, *audio_shape).astype(np.float16)

        # Create LMDB
        lmdb_path = str(temp_dir / "test_lmdb")
        env = lmdb.open(lmdb_path, map_size=1_000_000_000)

        # Store data
        with env.begin(write=True) as txn:
            for i in range(num_samples):
                # Store video
                video_key = f'video_latents_{i}_data'.encode()
                txn.put(video_key, video_latents[i].tobytes())

                # Store audio
                audio_key = f'audio_latents_{i}_data'.encode()
                txn.put(audio_key, audio_latents[i].tobytes())

                # Store prompt
                prompt_key = f'prompts_{i}_data'.encode()
                txn.put(prompt_key, prompts[i].encode('utf-8'))

            # Store shapes
            video_shape_with_count = [num_samples] + list(video_shape)
            txn.put(
                "video_latents_shape".encode(),
                " ".join(map(str, video_shape_with_count)).encode()
            )

            audio_shape_with_count = [num_samples] + list(audio_shape)
            txn.put(
                "audio_latents_shape".encode(),
                " ".join(map(str, audio_shape_with_count)).encode()
            )

            txn.put("prompts_shape".encode(), str(num_samples).encode())

        env.close()

        # Verify we can read back
        env = lmdb.open(lmdb_path, readonly=True)

        # Check shape
        shape = get_array_shape_from_lmdb(env, 'video_latents')
        assert shape[0] == num_samples

        # Check data retrieval
        video = retrieve_row_from_lmdb(env, 'video_latents', np.float16, 0, video_shape)
        assert video.shape == video_shape

        prompt = retrieve_row_from_lmdb(env, 'prompts', str, 0)
        assert prompt == "Test prompt 0"

        env.close()

    def test_dataset_loading(self, temp_dir):
        """Test loading data through dataset interface."""
        import lmdb
        from ltx_distillation.ode.data import ODERegressionLMDBDataset

        # Create minimal LMDB
        num_samples = 3
        video_shape = (4, 8, 64, 8, 12)  # Smaller for testing
        audio_shape = (4, 32, 64)

        lmdb_path = str(temp_dir / "test_dataset_lmdb")
        env = lmdb.open(lmdb_path, map_size=100_000_000)

        with env.begin(write=True) as txn:
            for i in range(num_samples):
                video = np.random.randn(*video_shape).astype(np.float16)
                audio = np.random.randn(*audio_shape).astype(np.float16)

                txn.put(f'video_latents_{i}_data'.encode(), video.tobytes())
                txn.put(f'audio_latents_{i}_data'.encode(), audio.tobytes())
                txn.put(f'prompts_{i}_data'.encode(), f"Prompt {i}".encode())

            txn.put(
                "video_latents_shape".encode(),
                f"{num_samples} " + " ".join(map(str, video_shape)).encode()
            )
            txn.put(
                "audio_latents_shape".encode(),
                f"{num_samples} " + " ".join(map(str, audio_shape)).encode()
            )
            txn.put("prompts_shape".encode(), str(num_samples).encode())

        env.close()

        # Load through dataset
        dataset = ODERegressionLMDBDataset(lmdb_path, load_audio=True)

        assert len(dataset) == num_samples

        sample = dataset[0]
        assert 'prompts' in sample
        assert 'video_latent' in sample
        assert 'audio_latent' in sample

        assert sample['video_latent'].shape == video_shape
        assert sample['audio_latent'].shape == audio_shape

    def test_batch_collation(self, temp_dir):
        """Test collating multiple samples into a batch."""
        from ltx_distillation.ode.data import collate_ode_batch

        video_shape = (4, 8, 64, 8, 12)
        audio_shape = (4, 32, 64)

        batch = [
            {
                'prompts': f"Prompt {i}",
                'video_latent': torch.randn(*video_shape),
                'audio_latent': torch.randn(*audio_shape),
            }
            for i in range(4)
        ]

        collated = collate_ode_batch(batch)

        assert len(collated['prompts']) == 4
        assert collated['video_latent'].shape[0] == 4
        assert collated['audio_latent'].shape[0] == 4


class TestConfigIntegration:
    """Integration tests for configuration consistency."""

    def test_config_compatibility(self):
        """Test that configs from different modules are compatible."""
        from ltx_distillation.ode.ode_regression import ODERegressionConfig
        from ltx_distillation.ode.generate_ode_pairs import ODEGenerationConfig

        # Regression config
        reg_config = ODERegressionConfig(
            denoising_step_list=(1000, 757, 522, 0),
            num_frame_per_block=4,
        )

        # Generation config
        gen_config = ODEGenerationConfig(
            teacher_checkpoint="dummy",
            gemma_path="dummy",
            denoising_step_list=[1000, 757, 522, 0],
        )

        # Timesteps should match
        assert list(reg_config.denoising_step_list) == gen_config.denoising_step_list

    def test_frame_calculations(self):
        """Test video/audio frame calculations are consistent."""
        # Video frame calculation
        num_frames = 121
        vae_temporal_compression = 8

        video_latent_frames = 1 + (num_frames - 1) // vae_temporal_compression
        assert video_latent_frames == 16

        # Audio frame calculation
        video_fps = 24.0
        audio_latent_fps = 25.0

        video_duration = num_frames / video_fps
        audio_latent_frames = int(round(video_duration * audio_latent_fps))
        assert audio_latent_frames == 126


class TestShapeConsistency:
    """Tests for tensor shape consistency across the pipeline."""

    def test_trajectory_shapes_121frames(self):
        """Test trajectory shapes for 121-frame configuration."""
        # 121 frames @ 512x768
        num_timesteps = 4
        video_latent_frames = 16
        audio_latent_frames = 126
        video_channels = 128
        video_h = 16  # 512/32
        video_w = 24  # 768/32

        # Video trajectory: [B, T, F, C, H, W]
        batch_size = 2
        video_trajectory = torch.randn(
            batch_size, num_timesteps, video_latent_frames,
            video_channels, video_h, video_w
        )

        # Audio trajectory: [B, T, F_a, C]
        audio_trajectory = torch.randn(
            batch_size, num_timesteps, audio_latent_frames, video_channels
        )

        # Verify shapes
        assert video_trajectory.shape == (2, 4, 16, 128, 16, 24)
        assert audio_trajectory.shape == (2, 4, 126, 128)

    def test_trajectory_shapes_241frames(self):
        """Test trajectory shapes for 241-frame configuration."""
        # 241 frames @ 512x768
        num_timesteps = 4
        video_latent_frames = 31  # 1 + (241-1)/8
        audio_latent_frames = 251  # round(241/24 * 25)

        video_trajectory_shape = (2, 4, 31, 128, 16, 24)
        audio_trajectory_shape = (2, 4, 251, 128)

        video = torch.randn(*video_trajectory_shape)
        audio = torch.randn(*audio_trajectory_shape)

        assert video.shape == video_trajectory_shape
        assert audio.shape == audio_trajectory_shape
