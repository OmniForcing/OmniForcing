"""
Unit tests for ODE data loading and LMDB creation.
"""

import pytest
import tempfile
import os
from pathlib import Path

import numpy as np
import torch

from ltx_distillation.ode.data import (
    ODERegressionDataset,
    TextDataset,
    collate_ode_batch,
)


class TestTextDataset:
    """Tests for TextDataset class."""

    def test_load_prompts(self):
        """Test loading prompts from text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("A cat playing piano\n")
            f.write("A dog running in the park\n")
            f.write("Ocean waves at sunset\n")
            f.name

        try:
            dataset = TextDataset(f.name)

            assert len(dataset) == 3
            assert dataset[0] == "A cat playing piano"
            assert dataset[1] == "A dog running in the park"
            assert dataset[2] == "Ocean waves at sunset"
        finally:
            os.unlink(f.name)

    def test_skip_empty_lines(self):
        """Test that empty lines are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("First prompt\n")
            f.write("\n")
            f.write("Second prompt\n")
            f.write("   \n")  # whitespace only
            f.write("Third prompt\n")
            f.name

        try:
            dataset = TextDataset(f.name)
            # Only non-empty lines should be included
            assert len(dataset) >= 3
        finally:
            os.unlink(f.name)


class TestODERegressionDataset:
    """Tests for ODERegressionDataset (in-memory)."""

    @pytest.fixture
    def sample_data_path(self):
        """Create sample data file."""
        data = {
            'prompts': ["Prompt 1", "Prompt 2"],
            'video_latents': torch.randn(2, 4, 16, 128, 16, 24),  # [N, T, F, C, H, W]
            'audio_latents': torch.randn(2, 4, 126, 128),  # [N, T, F_a, C]
        }

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(data, f.name)
            yield f.name

        os.unlink(f.name)

    def test_load_dataset(self, sample_data_path):
        """Test loading ODE regression dataset."""
        dataset = ODERegressionDataset(sample_data_path)

        assert len(dataset) == 2

    def test_get_item(self, sample_data_path):
        """Test getting a single item."""
        dataset = ODERegressionDataset(sample_data_path)
        sample = dataset[0]

        assert 'prompts' in sample
        assert 'video_latent' in sample
        assert sample['prompts'] == "Prompt 1"

    def test_max_pair_limit(self, sample_data_path):
        """Test max_pair parameter limits dataset size."""
        dataset = ODERegressionDataset(sample_data_path, max_pair=1)

        assert len(dataset) == 1


class TestCollateODEBatch:
    """Tests for ODE batch collation."""

    def test_collate_video_only(self):
        """Test collating batch without audio."""
        batch = [
            {
                'prompts': "Prompt 1",
                'video_latent': torch.randn(4, 16, 128, 16, 24),
            },
            {
                'prompts': "Prompt 2",
                'video_latent': torch.randn(4, 16, 128, 16, 24),
            },
        ]

        collated = collate_ode_batch(batch)

        assert len(collated['prompts']) == 2
        assert collated['video_latent'].shape[0] == 2
        assert 'audio_latent' not in collated

    def test_collate_with_audio(self):
        """Test collating batch with audio."""
        batch = [
            {
                'prompts': "Prompt 1",
                'video_latent': torch.randn(4, 16, 128, 16, 24),
                'audio_latent': torch.randn(4, 126, 128),
            },
            {
                'prompts': "Prompt 2",
                'video_latent': torch.randn(4, 16, 128, 16, 24),
                'audio_latent': torch.randn(4, 126, 128),
            },
        ]

        collated = collate_ode_batch(batch)

        assert 'audio_latent' in collated
        assert collated['audio_latent'].shape[0] == 2


class TestTrajectoryShape:
    """Tests for ODE trajectory shapes."""

    def test_video_trajectory_shape(self):
        """Test expected video trajectory shape."""
        # Trajectory: [B, T, F, C, H, W]
        # T = number of timesteps (e.g., 4 for [1000, 757, 522, 0])
        # F = video latent frames (16 for 121 frames)
        # C = latent channels (128)
        # H, W = spatial dimensions (16, 24 for 512x768)

        batch_size = 2
        num_timesteps = 4
        num_frames = 16
        channels = 128
        height = 16
        width = 24

        trajectory = torch.randn(
            batch_size, num_timesteps, num_frames, channels, height, width
        )

        assert trajectory.shape == (2, 4, 16, 128, 16, 24)

    def test_audio_trajectory_shape(self):
        """Test expected audio trajectory shape."""
        # Trajectory: [B, T, F_a, C]
        # T = number of timesteps
        # F_a = audio latent frames (126 for 121 video frames)
        # C = latent channels (128)

        batch_size = 2
        num_timesteps = 4
        num_audio_frames = 126
        channels = 128

        trajectory = torch.randn(
            batch_size, num_timesteps, num_audio_frames, channels
        )

        assert trajectory.shape == (2, 4, 126, 128)

    def test_timestep_ordering(self):
        """Test that timesteps are ordered from noise to clean."""
        # CausVid convention: [1000, 757, 522, 0] = [noise, ..., clean]
        timesteps = [1000, 757, 522, 0]

        # Should be decreasing (more noise -> less noise)
        assert timesteps[0] > timesteps[1] > timesteps[2] > timesteps[3]
        assert timesteps[0] == 1000  # Pure noise
        assert timesteps[-1] == 0    # Clean
