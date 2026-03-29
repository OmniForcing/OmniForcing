"""
ODE Regression Datasets for LTX-2 causal model training.

This module provides dataset classes for loading ODE trajectory pairs
used in the ODE initialization stage of causal model training.

Key Classes:
    - ODERegressionLMDBDataset: Efficient LMDB-based dataset for large-scale training
    - ODERegressionDataset: Simple in-memory dataset for small-scale experiments

Data Format:
    Each sample contains:
        - prompts: Text prompt string
        - video_latent: [T, F, C, H, W] video latents at T timesteps
        - audio_latent: [T, F_a, C] audio latents at T timesteps (optional)
        - sigmas: [T] actual sigma values per trajectory entry (from LTX2Scheduler)
"""

from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import lmdb

from ltx_distillation.ode.create_lmdb import (
    get_array_shape_from_lmdb,
    retrieve_row_from_lmdb,
)


class ODERegressionLMDBDataset(Dataset):
    """
    LMDB-based dataset for ODE regression training.

    Efficiently loads video and audio ODE trajectories from LMDB database.
    Supports both video-only and audio-video joint training.

    Args:
        data_path: Path to LMDB database
        max_pair: Maximum number of samples to use
        load_audio: Whether to load audio trajectories

    Example:
        dataset = ODERegressionLMDBDataset("./ode_lmdb")
        sample = dataset[0]
        # sample = {
        #     "prompts": "A cat playing piano",
        #     "video_latent": tensor([T, F, C, H, W]),
        #     "audio_latent": tensor([T, F_a, C]),  # if available
        # }
    """

    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
        load_audio: bool = True,
    ):
        self.data_path = data_path
        self.max_pair = max_pair
        self.load_audio = load_audio

        # Open LMDB in read-only mode
        self.env = lmdb.open(
            data_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Get shapes
        self.video_shape = get_array_shape_from_lmdb(self.env, 'video_latents')

        # Check for sigmas
        self.has_sigmas = False
        try:
            self.sigmas_shape = get_array_shape_from_lmdb(self.env, 'sigmas')
            self.has_sigmas = True
        except KeyError:
            self.sigmas_shape = None

        # Check for audio
        self.has_audio = False
        if load_audio:
            try:
                self.audio_shape = get_array_shape_from_lmdb(self.env, 'audio_latents')
                self.has_audio = True
            except KeyError:
                self.audio_shape = None

    def __len__(self) -> int:
        return min(self.video_shape[0], self.max_pair)

    def get_prompt(self, idx: int) -> str:
        """Read only the prompt string for a dataset item."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Prompt index out of range: {idx}")

        return retrieve_row_from_lmdb(
            self.env,
            "prompts",
            str,
            idx,
        )

    def get_prompts(self, num_prompts: int) -> list[str]:
        """Read the first ``num_prompts`` prompt strings without loading latents."""
        return [self.get_prompt(i) for i in range(min(len(self), max(0, int(num_prompts))))]

    def get_sigmas(self, idx: int = 0) -> torch.Tensor:
        """Read one sigma trajectory row without loading the large latent tensors."""
        if not self.has_sigmas or self.sigmas_shape is None:
            raise KeyError("LMDB does not contain stored sigma trajectories")
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Sigma index out of range: {idx}")

        sigmas = retrieve_row_from_lmdb(
            self.env,
            "sigmas",
            np.float32,
            idx,
            shape=self.sigmas_shape[1:],
        )
        return torch.tensor(sigmas, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single ODE trajectory pair.

        Returns:
            Dictionary with:
                - prompts: Text prompt string
                - video_latent: [T, F, C, H, W] video latents ordered noise->clean
                - audio_latent: [T, F_a, C] audio latents (if available)
                - sigmas: [T] actual sigma values per trajectory entry (if available)
        """
        # Load video latents
        video_latents = retrieve_row_from_lmdb(
            self.env,
            "video_latents",
            np.float16,
            idx,
            shape=self.video_shape[1:],  # [T, F, C, H, W]
        )

        # Load prompt
        prompt = retrieve_row_from_lmdb(
            self.env,
            "prompts",
            str,
            idx,
        )

        result = {
            "prompts": prompt,
            "video_latent": torch.tensor(video_latents, dtype=torch.float32),
        }

        # Load sigma values if available
        if self.has_sigmas:
            sigmas = retrieve_row_from_lmdb(
                self.env,
                "sigmas",
                np.float32,
                idx,
                shape=self.sigmas_shape[1:],  # [T]
            )
            result["sigmas"] = torch.tensor(sigmas, dtype=torch.float32)

        # Load audio latents if available
        if self.has_audio and self.load_audio:
            audio_latents = retrieve_row_from_lmdb(
                self.env,
                "audio_latents",
                np.float16,
                idx,
                shape=self.audio_shape[1:],  # [T, F_a, C]
            )
            result["audio_latent"] = torch.tensor(audio_latents, dtype=torch.float32)

        return result

    def close(self):
        """Close the LMDB environment."""
        self.env.close()


class ODERegressionDataset(Dataset):
    """
    Simple in-memory dataset for ODE regression training.

    Loads all data into memory. Suitable for small-scale experiments
    or debugging.

    Args:
        data_path: Path to .pt file containing trajectory data
        max_pair: Maximum number of samples to use

    Data format in .pt file:
        {
            "prompts": List[str],
            "video_latents": Tensor[N, T, F, C, H, W],
            "audio_latents": Tensor[N, T, F_a, C],  # optional
        }
    """

    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
    ):
        self.data_dict = torch.load(data_path, weights_only=False)
        self.max_pair = max_pair

        # Check for audio
        self.has_audio = 'audio_latents' in self.data_dict

    def __len__(self) -> int:
        return min(len(self.data_dict['prompts']), self.max_pair)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single ODE trajectory pair.

        Returns:
            Dictionary with:
                - prompts: Text prompt string
                - video_latent: [T, F, C, H, W] video latents
                - audio_latent: [T, F_a, C] audio latents (if available)
        """
        result = {
            "prompts": self.data_dict['prompts'][idx],
            "video_latent": self.data_dict['video_latents'][idx],
        }

        if self.has_audio:
            result["audio_latent"] = self.data_dict['audio_latents'][idx]

        return result


class TextDataset(Dataset):
    """
    Simple text prompt dataset for ODE pair generation.

    Args:
        data_path: Path to text file with one prompt per line
    """

    def __init__(self, data_path: str):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                text = line.strip()
                if text:
                    self.texts.append(text)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def collate_ode_batch(batch: list) -> Dict[str, Any]:
    """
    Custom collate function for ODE regression batches.

    Handles variable-length prompts and stacks tensors.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with:
            - prompts: List[str]
            - video_latent: Tensor[B, T, F, C, H, W]
            - audio_latent: Tensor[B, T, F_a, C] (if present)
            - sigmas: Tensor[B, T] actual sigma values (if present)
    """
    prompts = [sample['prompts'] for sample in batch]
    video_latents = torch.stack([sample['video_latent'] for sample in batch], dim=0)

    result = {
        'prompts': prompts,
        'video_latent': video_latents,
    }

    # Handle sigmas if present
    if 'sigmas' in batch[0]:
        sigmas = torch.stack([sample['sigmas'] for sample in batch], dim=0)
        result['sigmas'] = sigmas

    # Handle audio if present
    if 'audio_latent' in batch[0]:
        audio_latents = torch.stack([sample['audio_latent'] for sample in batch], dim=0)
        result['audio_latent'] = audio_latents

    return result
