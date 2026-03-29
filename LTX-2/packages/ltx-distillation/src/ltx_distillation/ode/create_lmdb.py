"""
LMDB Creation Script for ODE Trajectory Pairs.

This script converts individual .pt trajectory files into an LMDB database
for efficient training data loading.

Each .pt file contains:
    - "prompt": str
    - "video_trajectory": [1, T, F, C, H, W] video latents at T timesteps
    - "audio_trajectory": [1, T, F_a, C] audio latents at T timesteps

The LMDB stores:
    - video_latents_{idx}_data: video trajectory bytes
    - audio_latents_{idx}_data: audio trajectory bytes
    - sigmas_{idx}_data: sigma values (float32) per trajectory entry
    - prompts_{idx}_data: prompt string bytes
    - *_shape: array shapes for reconstruction
"""

import os
import glob
import argparse
from typing import Dict, Any, Set
from tqdm import tqdm

import numpy as np
import torch
import lmdb


def store_arrays_to_lmdb(
    env: lmdb.Environment,
    arrays_dict: Dict[str, Any],
    start_index: int = 0,
) -> int:
    """
    Store arrays in LMDB database.

    Args:
        env: LMDB environment
        arrays_dict: Dictionary with array names and values
        start_index: Starting index for keys

    Returns:
        Number of entries stored
    """
    count = 0
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            if isinstance(array, (list, np.ndarray)):
                for i, row in enumerate(array):
                    if isinstance(row, str):
                        row_bytes = row.encode('utf-8')
                    else:
                        row_bytes = row.tobytes()

                    data_key = f'{array_name}_{start_index + i}_data'.encode()
                    txn.put(data_key, row_bytes)
                count = max(count, len(array))
            else:
                # Single value
                if isinstance(array, str):
                    row_bytes = array.encode('utf-8')
                else:
                    row_bytes = array.tobytes()
                data_key = f'{array_name}_{start_index}_data'.encode()
                txn.put(data_key, row_bytes)
                count = 1

    return count


def get_array_shape_from_lmdb(
    env: lmdb.Environment,
    array_name: str,
) -> tuple:
    """Get array shape from LMDB metadata."""
    with env.begin() as txn:
        shape_bytes = txn.get(f"{array_name}_shape".encode())
        if shape_bytes is None:
            raise KeyError(f"Shape not found for {array_name}")
        shape_str = shape_bytes.decode()
        shape = tuple(map(int, shape_str.split()))
    return shape


def retrieve_row_from_lmdb(
    env: lmdb.Environment,
    array_name: str,
    dtype,
    row_index: int,
    shape: tuple = None,
):
    """Retrieve a specific row from LMDB."""
    data_key = f'{array_name}_{row_index}_data'.encode()

    with env.begin() as txn:
        row_bytes = txn.get(data_key)

    if row_bytes is None:
        raise KeyError(f"Key not found: {array_name}_{row_index}")

    if dtype == str:
        return row_bytes.decode('utf-8')
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)
        if shape is not None and len(shape) > 0:
            array = array.reshape(shape)
        return array


def process_trajectory_file(
    file_path: str,
    seen_prompts: Set[str],
) -> Dict[str, Any]:
    """
    Process a single trajectory .pt file.

    Args:
        file_path: Path to .pt file
        seen_prompts: Set of already processed prompts (for deduplication)

    Returns:
        Dictionary with processed arrays, or None if duplicate
    """
    data = torch.load(file_path, map_location='cpu', weights_only=False)

    prompt = data.get('prompt', '')

    # Deduplicate by prompt
    if prompt in seen_prompts:
        return None

    seen_prompts.add(prompt)

    # Extract trajectories
    video_trajectory = data.get('video_trajectory')
    audio_trajectory = data.get('audio_trajectory')
    sigmas = data.get('sigmas')  # [T] actual sigma values per trajectory entry

    if video_trajectory is None:
        raise ValueError(f"Missing video_trajectory in {file_path}")

    # Convert to numpy float16 for storage efficiency
    video_latents = video_trajectory.half().numpy()
    audio_latents = audio_trajectory.half().numpy() if audio_trajectory is not None else None
    sigmas_np = sigmas.float().numpy() if sigmas is not None else None

    return {
        'video_latents': video_latents,
        'audio_latents': audio_latents,
        'prompts': [prompt],
        'sigmas': sigmas_np,
    }


def create_lmdb_from_trajectories(
    data_path: str,
    lmdb_path: str,
    map_size: int = 5_000_000_000_000,  # 5TB default
) -> None:
    """
    Create LMDB database from trajectory .pt files.

    Args:
        data_path: Directory containing .pt trajectory files
        lmdb_path: Output LMDB path
        map_size: Maximum database size in bytes
    """
    # Find all .pt files
    all_files = sorted(glob.glob(os.path.join(data_path, "*.pt")))
    print(f"Found {len(all_files)} trajectory files")

    if len(all_files) == 0:
        raise ValueError(f"No .pt files found in {data_path}")

    # Create LMDB
    os.makedirs(os.path.dirname(lmdb_path) if os.path.dirname(lmdb_path) else '.', exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)

    counter = 0
    seen_prompts: Set[str] = set()
    last_video_shape = None
    last_audio_shape = None

    for file_path in tqdm(all_files, desc="Processing trajectory files"):
        try:
            data_dict = process_trajectory_file(file_path, seen_prompts)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        if data_dict is None:
            # Duplicate prompt, skip
            continue

        # Store arrays
        video_latents = data_dict['video_latents']
        audio_latents = data_dict['audio_latents']
        prompts = data_dict['prompts']
        sigmas = data_dict['sigmas']  # [T] float32 actual sigma values

        with env.begin(write=True) as txn:
            for i, prompt in enumerate(prompts):
                # Store video latents
                video_key = f'video_latents_{counter}_data'.encode()
                txn.put(video_key, video_latents[i].tobytes())

                # Store audio latents (if available)
                if audio_latents is not None:
                    audio_key = f'audio_latents_{counter}_data'.encode()
                    txn.put(audio_key, audio_latents[i].tobytes())

                # Store sigma values (float32, same for all prompts in this file)
                if sigmas is not None:
                    sigma_key = f'sigmas_{counter}_data'.encode()
                    txn.put(sigma_key, sigmas.tobytes())

                # Store prompt
                prompt_key = f'prompts_{counter}_data'.encode()
                txn.put(prompt_key, prompt.encode('utf-8'))

                counter += 1

        last_video_shape = video_latents.shape
        last_audio_shape = audio_latents.shape if audio_latents is not None else None
        last_sigmas_shape = sigmas.shape if sigmas is not None else None

    # Save shapes to LMDB
    if last_video_shape is not None:
        with env.begin(write=True) as txn:
            # Video shape: [B, T, F, C, H, W] -> store [T, F, C, H, W] per entry
            video_entry_shape = list(last_video_shape[1:])  # Remove batch dim
            video_entry_shape = [counter] + video_entry_shape  # Add total count
            shape_str = " ".join(map(str, video_entry_shape))
            txn.put("video_latents_shape".encode(), shape_str.encode())

            # Audio shape: [B, T, F_a, C] -> store [T, F_a, C] per entry
            if last_audio_shape is not None:
                audio_entry_shape = list(last_audio_shape[1:])
                audio_entry_shape = [counter] + audio_entry_shape
                shape_str = " ".join(map(str, audio_entry_shape))
                txn.put("audio_latents_shape".encode(), shape_str.encode())

            # Sigmas shape: [T] per entry (same T as video trajectory dim)
            if last_sigmas_shape is not None:
                sigmas_entry_shape = [counter] + list(last_sigmas_shape)
                shape_str = " ".join(map(str, sigmas_entry_shape))
                txn.put("sigmas_shape".encode(), shape_str.encode())

            # Prompts shape
            txn.put("prompts_shape".encode(), f"{counter}".encode())

    env.close()
    print(f"Created LMDB at {lmdb_path} with {counter} entries")


def main():
    """Command line interface for LMDB creation."""
    parser = argparse.ArgumentParser(
        description="Convert ODE trajectory .pt files to LMDB format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory containing .pt trajectory files"
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        required=True,
        help="Output LMDB database path"
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=5_000_000_000_000,
        help="Maximum LMDB size in bytes (default: 5TB)"
    )

    args = parser.parse_args()

    create_lmdb_from_trajectories(
        data_path=args.data_path,
        lmdb_path=args.lmdb_path,
        map_size=args.map_size,
    )


if __name__ == "__main__":
    main()
