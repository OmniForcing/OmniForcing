"""Decode a single ODE LMDB sample's clean endpoint to MP4 for inspection.

Usage:
    python visualize_lmdb_sample.py --lmdb ode_lmdb_4step_512 --index 0 --output sample0.mp4
"""

import argparse
import os
import sys

import lmdb
import numpy as np
import torch
import torchvision
from pathlib import Path


def decode_sample(lmdb_path: str, index: int, output_path: str, checkpoint_path: str):
    """Decode one LMDB sample's clean endpoint to MP4."""

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        # Shapes
        v_shape_str = txn.get(b"video_latents_shape").decode()
        a_shape_str = txn.get(b"audio_latents_shape").decode()
        v_shape = [int(x) for x in v_shape_str.split()]   # [N, T, F, C, H, W]
        a_shape = [int(x) for x in a_shape_str.split()]   # [N, T, F_a, C_a]

        N, T, F_v, C_v, H, W = v_shape
        _, _, F_a, C_a = a_shape
        print(f"LMDB contains {N} samples, trajectory T={T}")
        print(f"video shape per sample: [T={T}, F={F_v}, C={C_v}, H={H}, W={W}]")
        print(f"audio shape per sample: [T={T}, F_a={F_a}, C_a={C_a}]")

        # Prompt
        prompt = txn.get(f"prompts_{index}_data".encode()).decode()
        print(f"\nSample {index} prompt:\n  {prompt[:300]}...\n")

        # Sigmas
        sigmas = np.frombuffer(
            txn.get(f"sigmas_{index}_data".encode()), dtype=np.float32
        )
        print(f"sigmas: {sigmas.tolist()}")

        # Clean endpoint = trajectory step -1
        video_raw = np.frombuffer(
            txn.get(f"video_latents_{index}_data".encode()),
            dtype=np.float16,
        ).reshape(T, F_v, C_v, H, W)
        video_clean = torch.from_numpy(video_raw[-1].copy()).unsqueeze(0)  # [1, F, C, H, W]
        print(f"clean video latent: shape={tuple(video_clean.shape)} dtype={video_clean.dtype}")

        audio_raw = np.frombuffer(
            txn.get(f"audio_latents_{index}_data".encode()),
            dtype=np.float16,
        ).reshape(T, F_a, C_a)
        audio_clean = torch.from_numpy(audio_raw[-1].copy()).unsqueeze(0)  # [1, F_a, C_a]
        print(f"clean audio latent: shape={tuple(audio_clean.shape)}")

    # Load VAEs
    print("\nLoading VAEs (video + audio + vocoder)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from ltx_distillation.models.vae_wrapper import create_vae_wrappers
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )

    # Decode video: latent [1, F, C, H, W] -> pixel [1, 3, F_out, H_out, W_out] in [-1, 1]
    print("Decoding video latent to pixels...")
    with torch.no_grad():
        video_clean = video_clean.to(device=device, dtype=dtype)
        pixel_video = video_vae.decode(video_clean)
        pixel_video = (pixel_video + 1) / 2
        pixel_video = pixel_video.clamp(0, 1)
        print(f"  pixel video: shape={tuple(pixel_video.shape)} dtype={pixel_video.dtype}")

    # Decode audio
    print("Decoding audio latent to waveform...")
    with torch.no_grad():
        audio_clean = audio_clean.to(device=device, dtype=dtype)
        waveform = audio_vae.decode_to_waveform(audio_clean)
        print(f"  waveform: shape={tuple(waveform.shape)}")

    # Save mp4 via torchvision or ffmpeg
    # pixel_video: [1, 3, F, H, W] → [F, H, W, 3] uint8
    video_np = pixel_video.squeeze(0).permute(1, 2, 3, 0).float().cpu().numpy()  # [F, H, W, 3]
    video_uint8 = (video_np * 255).clip(0, 255).astype(np.uint8)
    print(f"\nVideo tensor for encoding: {video_uint8.shape} {video_uint8.dtype}")

    # Waveform: [1, C, samples] → numpy float32, transpose to (samples, C) for scipy
    waveform_np = waveform.squeeze(0).float().cpu().numpy()  # [C, samples]
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.T  # [samples, C]
    print(f"Audio waveform: {waveform_np.shape}")

    # Save via torchvision.io.write_video (silent) then mux audio via ffmpeg
    video_only_path = output_path.replace(".mp4", "_video_only.mp4")
    audio_wav_path = output_path.replace(".mp4", ".wav")

    torchvision.io.write_video(
        video_only_path,
        torch.from_numpy(video_uint8),
        fps=24,
        video_codec="h264",
    )
    print(f"Video-only saved to: {video_only_path}")

    # Save audio WAV
    import scipy.io.wavfile
    sample_rate = 24000
    scipy.io.wavfile.write(audio_wav_path, sample_rate, waveform_np.astype(np.float32))
    print(f"Audio WAV saved to: {audio_wav_path}")

    # Mux with ffmpeg
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_only_path,
        "-i", audio_wav_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"\nFinal MP4 saved to: {output_path}")
        # Clean up intermediates
        os.remove(video_only_path)
        os.remove(audio_wav_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg mux failed: {e}")
        print(f"Intermediate files kept: {video_only_path}, {audio_wav_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="sample_viz.mp4")
    parser.add_argument(
        "--checkpoint",
        default="/pfs/suyaofeng/workspace/omni-forcing/checkpoints/LTX-2/ltx-2-19b-dev.safetensors",
    )
    args = parser.parse_args()
    decode_sample(args.lmdb, args.index, args.output, args.checkpoint)
