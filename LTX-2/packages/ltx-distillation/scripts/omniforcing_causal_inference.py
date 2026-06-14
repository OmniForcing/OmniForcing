#!/usr/bin/env python3
"""Standalone OmniForcing causal AV inference.

This is the GitHub/Hugging Face release copy of the single-file inference
entry point. The checkpoint path below is intentionally a placeholder: upload
the OmniForcing checkpoint to Hugging Face, download it locally, then pass that
local file path to ``--generator-ckpt``.

Recommended checkpoint
----------------------
For the current 5-second model, use the OmniForcing causal checkpoint:

    <HF_ORG>/<HF_REPO>/omniforcing_ltx2_5s_causal.safetensors.index.json

When running this script, ``--generator-ckpt`` should point to the downloaded
checkpoint file, for example:

    /path/to/downloaded/omniforcing_ltx2_5s_causal.safetensors.index.json

The script runs the causal KV-cache generator with the released 5-second
settings:

    - 5-second output: 121 frames at 24 FPS
    - resolution: 512 x 768
    - causal blocks: 4, 3, 3, 3, 3 latent video frames
    - denoising timesteps: 1000, 909, 725, 421, 0
    - prompt seeds: base seed + prompt index

Example
-------
Run one prompt:

    python scripts/omniforcing_causal_inference.py \
        --base-checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --vae-checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --gemma-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
        --generator-ckpt /path/to/downloaded/omniforcing_ltx2_5s_causal.safetensors.index.json \
        --prompt "Realistic. Rain falls on a quiet street at night." \
        --output-dir outputs/demo

Run a prompt file, one prompt per line:

    python scripts/omniforcing_causal_inference.py \
        --base-checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --vae-checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --gemma-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
        --generator-ckpt /path/to/downloaded/omniforcing_ltx2_5s_causal.safetensors.index.json \
        --prompt-file prompts/demo.txt \
        --output-dir outputs/demo

Outputs
-------
Each sample is saved as:

    sample_000.mp4
    sample_000.txt

The mp4 writer tries to mux generated audio into the video. If audio muxing is
not available in the local torchvision/ffmpeg build, a silent mp4 and a sidecar
sample_000.wav are written instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _add_repo_paths() -> None:
    """Make the script runnable from a source checkout without pip install -e."""
    script_path = Path(__file__).resolve()
    # LTX-2/packages/ltx-distillation/scripts/this_file.py
    packages_dir = script_path.parents[2]
    for package in ("ltx-distillation", "ltx-causal", "ltx-core", "ltx-pipelines"):
        src = packages_dir / package / "src"
        if src.exists():
            sys.path.insert(0, str(src))


_add_repo_paths()

import torch

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader.registry import StateDictRegistry
from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
from ltx_causal.wrapper import CausalLTX2DiffusionWrapper
from ltx_distillation.inference.causal_pipeline import CausalAVInferencePipeline
from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import create_vae_wrappers


def load_checkpoint_state_dict(path: str, prefer_ema: bool = False) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return the generator state dict when present."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        index_files = sorted(checkpoint_path.glob("*.safetensors.index.json"))
        if index_files:
            return load_checkpoint_state_dict(str(index_files[0]), prefer_ema=prefer_ema)
        shard_files = sorted(checkpoint_path.glob("*.safetensors"))
        if shard_files:
            return load_safetensors_shards(shard_files)
        raise FileNotFoundError(f"No safetensors checkpoint shards found in {checkpoint_path}")

    if path.endswith(".safetensors.index.json"):
        with open(path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shard_files = [checkpoint_path.parent / name for name in shard_names]
        return load_safetensors_shards(shard_files)

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if prefer_ema and isinstance(checkpoint, dict) and "generator_ema" in checkpoint:
        return checkpoint["generator_ema"]
    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        return checkpoint["generator"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_safetensors_shards(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Load and merge a sharded safetensors checkpoint."""
    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}
    for path in paths:
        state_dict.update(load_file(str(path)))
    return state_dict


def remap_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map common LTX checkpoint key layouts to CausalLTX2DiffusionWrapper keys."""
    if not state_dict:
        return state_dict

    non_transformer_prefixes = (
        "vae.",
        "audio_vae.",
        "vocoder.",
        "model.vae.",
        "model.audio_vae.",
        "model.vocoder.",
    )
    remapped_non_transformer_prefixes = (
        "model.audio_embeddings_connector.",
        "model.video_embeddings_connector.",
    )

    if any(key.startswith("model.diffusion_model.") for key in state_dict):
        remapped = {}
        for key, value in state_dict.items():
            if not key.startswith("model.diffusion_model."):
                continue
            new_key = "model." + key[len("model.diffusion_model.") :]
            if any(new_key.startswith(prefix) for prefix in remapped_non_transformer_prefixes):
                continue
            remapped[new_key] = value
        return remapped

    first_key = next(iter(state_dict))
    if first_key.startswith("model.velocity_model."):
        return {
            "model." + key[len("model.velocity_model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.velocity_model.")
        }
    if first_key.startswith("model."):
        return {
            key: value
            for key, value in state_dict.items()
            if not any(key.startswith(prefix) for prefix in non_transformer_prefixes)
        }
    return {
        "model." + key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in non_transformer_prefixes)
    }


def add_noise(original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Flow-matching interpolation: x_t = (1 - sigma) * x_0 + sigma * noise."""
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    sigma = sigma.to(dtype=original.dtype)
    return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def denoising_sigmas(
    denoising_step_list: list[int],
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    full_sigmas = LTX2Scheduler().execute(steps=num_inference_steps)
    selected = []
    for timestep in denoising_step_list:
        target_sigma = float(timestep) / 1000.0
        idx = (full_sigmas - target_sigma).abs().argmin().item()
        selected.append(full_sigmas[idx])
    return torch.stack(selected).to(device)


def compute_latent_shapes(
    num_frames: int,
    height: int,
    width: int,
    fps: float = 24.0,
    batch_size: int = 1,
) -> tuple[list[int], list[int]]:
    if (num_frames - 1) % 8 != 0:
        raise ValueError(f"num_frames must be 1 + 8*k, got {num_frames}")
    latent_frames = 1 + (num_frames - 1) // 8
    latent_h = height // 32
    latent_w = width // 32

    # Matches the training benchmark shape calculation. LTX-2 audio latents are
    # aligned to 16 kHz / 160 hop / 4 downsample = 25 latent frames per second.
    video_duration = float(num_frames) / float(fps)
    audio_frames = round(video_duration * 25.0)
    return (
        [batch_size, latent_frames, 128, latent_h, latent_w],
        [batch_size, audio_frames, 128],
    )


def read_prompts(prompt: list[str] | None, prompt_file: str | None, num_prompts: int | None) -> list[str]:
    prompts: list[str] = []
    if prompt:
        prompts.extend(prompt)
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    if not prompts:
        raise ValueError("No prompts provided. Use --prompt or --prompt-file.")
    return prompts


def build_generator(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    causal_config = CausalLTXModelConfig(
        num_frame_per_block=args.num_frame_per_block,
        num_frame_per_block_first=args.num_frame_per_block_first,
        enable_causal_log_rescale=args.enable_causal_log_rescale,
    )
    model = CausalLTXModel(causal_config).to(device=device, dtype=dtype)
    generator = CausalLTX2DiffusionWrapper(
        model=model,
        video_height=args.height,
        video_width=args.width,
        num_frame_per_block=args.num_frame_per_block,
        num_frame_per_block_first=args.num_frame_per_block_first,
        disable_causal_mask=args.disable_causal_mask,
    )

    print(f"[init] Loading base checkpoint: {args.base_checkpoint}", flush=True)
    base_sd = remap_state_dict_keys(load_checkpoint_state_dict(args.base_checkpoint))
    for key in [key for key in list(base_sd.keys()) if "audio_sink_tokens" in key]:
        base_sd.pop(key)
    missing, unexpected = generator.load_state_dict(base_sd, strict=False)
    real_missing = [key for key in missing if "mask_builder" not in key and "causal_gate" not in key]
    if real_missing or unexpected:
        print(f"[init] base load missing={len(real_missing)} unexpected={len(unexpected)}", flush=True)

    print(f"[init] Loading distilled generator: {args.generator_ckpt}", flush=True)
    gen_sd = remap_state_dict_keys(
        load_checkpoint_state_dict(args.generator_ckpt, prefer_ema=args.use_ema)
    )
    missing, unexpected = generator.load_state_dict(gen_sd, strict=False)
    real_missing = [key for key in missing if "mask_builder" not in key]
    if real_missing or unexpected:
        print(f"[init] generator load missing={len(real_missing)} unexpected={len(unexpected)}", flush=True)

    generator.requires_grad_(False)
    return generator.eval()


def save_sample(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor | None,
    video_vae,
    audio_vae,
    output_path: Path,
    fps: int,
    audio_sample_rate: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_pixel = video_vae.decode_to_pixel(video_latent)
    audio_waveform = None
    if audio_latent is not None:
        try:
            audio_waveform = audio_vae.decode_to_waveform(audio_latent)
        except Exception as exc:
            print(f"[warn] audio decode failed for {output_path.name}: {exc}", flush=True)

    video = video_pixel[0]
    if video.shape[0] == 3:
        video = video.permute(1, 0, 2, 3)
    video = video.permute(0, 2, 3, 1)
    video = (video.clamp(0, 1) * 255).cpu().to(torch.uint8)

    written_with_audio = False
    if audio_waveform is not None:
        try:
            from torchvision.io import write_video

            write_video(
                str(output_path),
                video,
                fps=fps,
                audio_array=audio_waveform[0].cpu().float(),
                audio_fps=audio_sample_rate,
                audio_codec="aac",
            )
            written_with_audio = True
        except Exception as exc:
            print(f"[warn] write_video with audio failed for {output_path.name}: {exc}", flush=True)

    if not written_with_audio:
        from torchvision.io import write_video

        write_video(str(output_path), video, fps=fps)
        if audio_waveform is not None:
            try:
                import torchaudio

                torchaudio.save(
                    str(output_path.with_suffix(".wav")),
                    audio_waveform[0].cpu().float(),
                    audio_sample_rate,
                )
            except Exception as exc:
                print(f"[warn] torchaudio.save failed for {output_path.name}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniForcing causal AV inference")
    parser.add_argument("--base-checkpoint", required=True, help="LTX-2 base .safetensors checkpoint.")
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Optional original LTX-2 checkpoint used to load video/audio VAEs. Defaults to --base-checkpoint.",
    )
    parser.add_argument("--gemma-path", required=True, help="Gemma text encoder directory.")
    parser.add_argument(
        "--generator-ckpt",
        required=True,
        help=(
            "Downloaded OmniForcing causal checkpoint. "
            "Recommended release checkpoint placeholder: "
            "<HF_ORG>/<HF_REPO>/omniforcing_ltx2_5s_causal.safetensors.index.json"
        ),
    )
    parser.add_argument("--prompt", action="append", help="Prompt text. Can be passed multiple times.")
    parser.add_argument("--prompt-file", default=None, help="Text file with one prompt per line.")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/omniforcing_causal")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--start-index", type=int, default=0)

    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio-sample-rate", type=int, default=24000)

    parser.add_argument("--denoising-step-list", default="1000,909,725,421,0")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--num-frame-per-block", type=int, default=3)
    parser.add_argument("--num-frame-per-block-first", type=int, default=4)
    parser.add_argument("--context-noise", type=int, default=0)
    parser.add_argument("--num-train-timestep", type=int, default=1000)
    parser.add_argument("--disable-causal-mask", action="store_true")
    parser.add_argument("--enable-causal-log-rescale", action="store_true")

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()

    prompts = read_prompts(args.prompt, args.prompt_file, args.num_prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[init] device={device} dtype={dtype}", flush=True)
    generator = build_generator(args, device=device, dtype=dtype)

    registry = StateDictRegistry()
    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=args.base_checkpoint,
        gemma_path=args.gemma_path,
        device=device,
        dtype=dtype,
        registry=registry,
    ).eval()
    vae_checkpoint = args.vae_checkpoint or args.base_checkpoint
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=vae_checkpoint,
        device=device,
        dtype=dtype,
        registry=registry,
    )

    sigmas = denoising_sigmas(
        denoising_step_list=parse_int_list(args.denoising_step_list),
        num_inference_steps=args.num_inference_steps,
        device=device,
    )
    print(f"[init] denoising_sigmas={sigmas.detach().cpu().tolist()}", flush=True)

    pipeline = CausalAVInferencePipeline(
        generator=generator,
        add_noise_fn=add_noise,
        denoising_sigmas=sigmas,
        num_frame_per_block=args.num_frame_per_block,
        num_frame_per_block_first=args.num_frame_per_block_first,
        context_noise=args.context_noise,
        num_train_timestep=args.num_train_timestep,
        clear_cuda_cache_per_round=True,
    )

    video_shape, audio_shape = compute_latent_shapes(
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        batch_size=1,
    )
    print(f"[init] video_shape={video_shape} audio_shape={audio_shape}", flush=True)

    metadata = {
        "base_checkpoint": args.base_checkpoint,
        "vae_checkpoint": vae_checkpoint,
        "gemma_path": args.gemma_path,
        "generator_ckpt": args.generator_ckpt,
        "seed": args.seed,
        "prompts": prompts,
        "video_shape": video_shape,
        "audio_shape": audio_shape,
        "denoising_sigmas": sigmas.detach().cpu().tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for local_idx, prompt in enumerate(prompts):
        prompt_idx = args.start_index + local_idx
        prompt_seed = args.seed + prompt_idx
        print(f"[infer] sample_{prompt_idx:03d} seed={prompt_seed}: {prompt}", flush=True)

        with torch.no_grad():
            conditional_dict = text_encoder(text_prompts=[prompt])
            fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(prompt_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(prompt_seed)
                video_latent, audio_latent = pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                )

        sample_path = output_dir / f"sample_{prompt_idx:03d}.mp4"
        save_sample(
            video_latent=video_latent,
            audio_latent=audio_latent,
            video_vae=video_vae,
            audio_vae=audio_vae,
            output_path=sample_path,
            fps=args.fps,
            audio_sample_rate=args.audio_sample_rate,
        )
        (output_dir / f"sample_{prompt_idx:03d}.txt").write_text(prompt + "\n", encoding="utf-8")

        if args.save_latents:
            torch.save(
                {
                    "prompt": prompt,
                    "seed": prompt_seed,
                    "video_latent": video_latent.detach().cpu(),
                    "audio_latent": audio_latent.detach().cpu() if audio_latent is not None else None,
                },
                output_dir / f"sample_{prompt_idx:03d}_latents.pt",
            )

        del video_latent, audio_latent, conditional_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[done] saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
