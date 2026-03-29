"""
Dataset classes for DMD distillation.
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Simple text prompt dataset.

    Reads prompts from a text file where each line is one prompt.
    The prompts are assumed to be already processed (no enhancement needed).
    """

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to text file with one prompt per line
            max_samples: Maximum number of samples to load (None for all)
        """
        self.data_path = data_path
        self.prompts = self._load_prompts(data_path, max_samples)

    def _load_prompts(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """Load prompts from file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        if max_samples is not None:
            prompts = prompts[:max_samples]

        print(f"Loaded {len(prompts)} prompts from {data_path}")
        return prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


class ODERegressionLMDBDataset(Dataset):
    """
    LMDB dataset for ODE regression training.

    Stores pre-computed ODE trajectories for faster training.
    This is used when backward_simulation=False.

    LMDB stores:
        - video_latents_{idx}_data: video trajectory bytes [T, F, C, H, W]
        - audio_latents_{idx}_data: audio trajectory bytes [T, F_a, C]
        - prompts_{idx}_data: prompt string bytes
        - video_latents_shape: "[total, T, F, C, H, W]"
        - audio_latents_shape: "[total, T, F_a, C]"
    """

    def __init__(
        self,
        lmdb_path: str,
        max_pair: int = int(1e8),
    ):
        """
        Args:
            lmdb_path: Path to LMDB database
            max_pair: Maximum number of pairs to load
        """
        self.lmdb_path = lmdb_path
        self.max_pair = max_pair

        try:
            import lmdb
            self.env = lmdb.open(
                lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

            # Get shape metadata
            with self.env.begin(write=False) as txn:
                # Parse video shape: "[total, T, F, C, H, W]"
                video_shape_bytes = txn.get("video_latents_shape".encode())
                if video_shape_bytes is None:
                    raise ValueError("Missing video_latents_shape in LMDB")
                video_shape_str = video_shape_bytes.decode()
                video_shape = list(map(int, video_shape_str.split()))
                self.length = min(video_shape[0], max_pair)
                self.video_entry_shape = video_shape[1:]  # [T, F, C, H, W]

                # Parse audio shape: "[total, T, F_a, C]" (may not exist)
                audio_shape_bytes = txn.get("audio_latents_shape".encode())
                if audio_shape_bytes is not None:
                    audio_shape_str = audio_shape_bytes.decode()
                    audio_shape = list(map(int, audio_shape_str.split()))
                    self.audio_entry_shape = audio_shape[1:]  # [T, F_a, C]
                    self.has_audio = True
                else:
                    self.audio_entry_shape = None
                    self.has_audio = False

        except ImportError:
            raise ImportError("lmdb package required for ODERegressionLMDBDataset")
        except Exception as e:
            raise RuntimeError(f"Failed to open LMDB at {lmdb_path}: {e}")

        print(f"Loaded LMDB dataset with {self.length} samples from {lmdb_path}")
        print(f"  Video shape per entry: {self.video_entry_shape}")
        if self.has_audio:
            print(f"  Audio shape per entry: {self.audio_entry_shape}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from LMDB.

        Returns:
            Dictionary containing:
                - prompts: Text prompt
                - ode_latent: ODE video trajectory [T, F, C, H, W]
                - ode_audio_latent: ODE audio trajectory [T, F_a, C] (if available)
        """
        with self.env.begin(write=False) as txn:
            # Load prompt
            prompt_key = f"prompts_{idx}_data".encode()
            prompt_bytes = txn.get(prompt_key)
            if prompt_bytes is None:
                raise KeyError(f"Prompt key {idx} not found in LMDB")
            prompt = prompt_bytes.decode('utf-8')

            # Load video latents
            video_key = f"video_latents_{idx}_data".encode()
            video_bytes = txn.get(video_key)
            if video_bytes is None:
                raise KeyError(f"Video latents key {idx} not found in LMDB")
            video_array = np.frombuffer(video_bytes, dtype=np.float16)
            video_array = video_array.reshape(self.video_entry_shape)
            video_tensor = torch.from_numpy(video_array.copy()).float()

            # Load audio latents (if available)
            audio_tensor = None
            if self.has_audio:
                audio_key = f"audio_latents_{idx}_data".encode()
                audio_bytes = txn.get(audio_key)
                if audio_bytes is not None:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float16)
                    audio_array = audio_array.reshape(self.audio_entry_shape)
                    audio_tensor = torch.from_numpy(audio_array.copy()).float()

        result = {
            "prompts": prompt,
            "ode_latent": video_tensor,  # [T, F, C, H, W]
        }

        if audio_tensor is not None:
            result["ode_audio_latent"] = audio_tensor  # [T, F_a, C]

        return result


def collate_text_prompts(batch: List[str]) -> List[str]:
    """Simple collate function for text prompts."""
    return batch


def collate_ode_data(batch: List[dict]) -> dict:
    """
    Collate function for ODE regression data.

    Handles both video and audio latents.
    Audio may be None if not available in the dataset.
    """
    prompts = [item["prompts"] for item in batch]
    ode_latents = torch.stack([item["ode_latent"] for item in batch])

    result = {
        "prompts": prompts,
        "ode_latent": ode_latents,  # [B, T, F, C, H, W]
    }

    # Check if audio is available (first item determines availability)
    if "ode_audio_latent" in batch[0] and batch[0]["ode_audio_latent"] is not None:
        ode_audio_latents = torch.stack([item["ode_audio_latent"] for item in batch])
        result["ode_audio_latent"] = ode_audio_latents  # [B, T, F_a, C]

    return result
