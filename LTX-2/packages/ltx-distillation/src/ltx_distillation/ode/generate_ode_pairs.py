"""
ODE Pair Generation for LTX-2 causal model initialization.

Generates ODE trajectory pairs by running the bidirectional teacher model
and collecting intermediate denoising states.

Supports single-GPU, single-node multi-GPU, and multi-node multi-GPU via torchrun:

    # Single GPU
    python -m ltx_distillation.ode.generate_ode_pairs \
        --teacher_checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --gemma_path /path/to/gemma \
        --prompts_file prompts.txt --output_dir ./ode_pairs

    # Single node, 8 GPUs
    torchrun --nproc_per_node=8 -m ltx_distillation.ode.generate_ode_pairs \
        --teacher_checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --gemma_path /path/to/gemma \
        --prompts_file prompts.txt --output_dir ./ode_pairs

    # Multi-node (e.g. 2 nodes x 8 GPUs)
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m ltx_distillation.ode.generate_ode_pairs \
        --teacher_checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --gemma_path /path/to/gemma \
        --prompts_file prompts.txt --output_dir ./ode_pairs

Based on CausVid's generate_ode_pairs.py, adapted for LTX-2 audio-video.
"""

import os
import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _is_torchrun() -> bool:
    """Check whether the process was launched via torchrun / torch.distributed.launch."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _init_distributed():
    """Initialise the distributed process group (nccl) and set the local device."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    init_method = f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)


def _get_rank_info() -> Tuple[int, int]:
    """Return (global_rank, world_size). Works with or without torchrun."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ODEGenerationConfig:
    """Configuration for ODE trajectory generation."""

    # Model
    teacher_checkpoint: str = ""
    gemma_path: str = ""

    # Resolution
    video_height: int = 512
    video_width: int = 768
    num_frames: int = 121

    # Denoising
    num_inference_steps: int = 40
    video_guidance_scale: float = 3.0   # LTX-2 native default for video
    audio_guidance_scale: float = 7.0   # LTX-2 native default for audio
    denoising_step_list: List[int] = field(default_factory=lambda: [1000, 757, 522, 0])

    # Negative prompt for CFG (matches LTX-2 official DEFAULT_NEGATIVE_PROMPT)
    negative_prompt: str = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )

    # Output
    output_dir: str = "./ode_pairs"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LTX2ODEPairGenerator(nn.Module):
    """
    Generates ODE trajectory pairs for causal model initialization.

    Uses bidirectional teacher model to denoise from pure noise to clean,
    collecting intermediate states at specified timesteps.
    """

    # Time alignment constants
    VIDEO_LATENT_FPS = 3.0
    AUDIO_LATENT_FPS = 25.0
    VAE_TEMPORAL_COMPRESSION = 8

    def __init__(self, config: ODEGenerationConfig):
        super().__init__()
        self.config = config

        # Lazy loading – models loaded on first use
        self._teacher = None
        self._text_encoder = None
        self._scheduler = None

    def _load_models(self, device: torch.device):
        """Load teacher model and text encoder."""
        if self._teacher is not None:
            return

        rank, _ = _get_rank_info()
        if rank == 0:
            print("Loading teacher model...")

        from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
        from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
        from ltx_core.components.schedulers import LTX2Scheduler

        self._text_encoder = create_text_encoder_wrapper(
            checkpoint_path=self.config.teacher_checkpoint,
            gemma_path=self.config.gemma_path,
            device=device,
        )
        self._text_encoder.eval()

        self._teacher = create_ltx2_wrapper(
            checkpoint_path=self.config.teacher_checkpoint,
            gemma_path=self.config.gemma_path,
            device=device,
            video_height=self.config.video_height,
            video_width=self.config.video_width,
        )
        self._teacher.eval()

        # Use LTX-2's native scheduler (sigmoid shift + stretch) for generating
        # the denoising sigma schedule. This matches the original inference pipeline.
        self._scheduler = LTX2Scheduler()

        if rank == 0:
            print("Models loaded.")

    @property
    def num_video_latent_frames(self) -> int:
        return 1 + (self.config.num_frames - 1) // self.VAE_TEMPORAL_COMPRESSION

    @property
    def num_audio_latent_frames(self) -> int:
        video_duration = self.config.num_frames / 24.0
        return int(round(video_duration * self.AUDIO_LATENT_FPS))

    def _compute_video_shape(self, batch_size: int) -> Tuple[int, ...]:
        F = self.num_video_latent_frames
        H = self.config.video_height // 32
        W = self.config.video_width // 32
        return (batch_size, F, 128, H, W)

    def _compute_audio_shape(self, batch_size: int) -> Tuple[int, ...]:
        F = self.num_audio_latent_frames
        return (batch_size, F, 128)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_trajectory(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        """
        Generate a single ODE trajectory.

        Returns dict with keys: prompt, video_trajectory, audio_trajectory.
        """
        device = torch.device(device)
        self._load_models(device)

        if seed is not None:
            torch.manual_seed(seed)

        # Encode text
        if negative_prompt is None:
            negative_prompt = self.config.negative_prompt
        conditional_dict = self._text_encoder(text_prompts=[prompt])
        unconditional_dict = self._text_encoder(
            text_prompts=[negative_prompt],
        )

        # Initialise noise
        video = torch.randn(self._compute_video_shape(1), device=device, dtype=dtype)
        audio = torch.randn(self._compute_audio_shape(1), device=device, dtype=dtype)

        video_trajectory = [video.cpu()]
        audio_trajectory = [audio.cpu()]

        # Use LTX-2's native scheduler to get the sigma schedule.
        # This produces a shifted + stretched schedule matching the real inference.
        # sigmas: [steps+1] values from ~1.0 → 0.0 (shifted by sigmoid + stretched)
        num_steps = self.config.num_inference_steps
        sigmas = self._scheduler.execute(steps=num_steps).to(device)

        rank, _ = _get_rank_info()
        if rank == 0 and not hasattr(self, '_sigmas_logged'):
            print(f"Sigma schedule ({num_steps} steps):")
            print(f"  first 5: {sigmas[:5].tolist()}")
            print(f"  last 5:  {sigmas[-5:].tolist()}")
            self._sigmas_logged = True

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_tensor = sigma * torch.ones([1], device=device, dtype=dtype)

            video_x0_cond, audio_x0_cond = self._teacher(
                noisy_image_or_video=video,
                conditional_dict=conditional_dict,
                timestep=sigma_tensor,
                noisy_audio=audio,
            )
            video_x0_uncond, audio_x0_uncond = self._teacher(
                noisy_image_or_video=video,
                conditional_dict=unconditional_dict,
                timestep=sigma_tensor,
                noisy_audio=audio,
            )

            # CFG (LTX-2 uses different scales for video and audio)
            video_x0 = video_x0_uncond + self.config.video_guidance_scale * (video_x0_cond - video_x0_uncond)
            audio_x0 = audio_x0_uncond + self.config.audio_guidance_scale * (audio_x0_cond - audio_x0_uncond)

            # Euler step: x_{t+1} = x_t + velocity * dt
            # velocity = (x_t - x_0) / sigma
            sigma_next = sigmas[i + 1]
            if sigma_next > 0 and sigma > 0:
                video_velocity = (video.float() - video_x0.float()) / sigma.float()
                audio_velocity = (audio.float() - audio_x0.float()) / sigma.float()
                dt = (sigma_next - sigma).float()
                video = (video.float() + video_velocity * dt).to(dtype)
                audio = (audio.float() + audio_velocity * dt).to(dtype)
            else:
                video = video_x0
                audio = audio_x0

            video_trajectory.append(video.cpu())
            audio_trajectory.append(audio.cpu())

        # Stack trajectory: [1, num_steps+1, ...]
        # Index 0 = pure noise (sigma≈1), index -1 = clean (sigma=0)
        video_trajectory = torch.stack(video_trajectory, dim=1)  # [1, T+1, F, C, H, W]
        audio_trajectory = torch.stack(audio_trajectory, dim=1)  # [1, T+1, F_a, C]

        # Store the sigma values alongside trajectory for exact reconstruction
        # sigmas_trajectory[i] is the sigma of trajectory[:, i]
        # sigmas_trajectory[0] = sigmas[0] ≈ 1.0 (noise), sigmas_trajectory[-1] = 0.0 (clean)
        sigmas_trajectory = torch.cat([sigmas[:1], sigmas[1:]])  # same as sigmas, just explicit

        # Subsample to key timesteps
        if self.config.denoising_step_list:
            indices = self._compute_subsample_indices(sigmas, self.config.denoising_step_list)
            video_trajectory = video_trajectory[:, indices]
            audio_trajectory = audio_trajectory[:, indices]
            sigmas_trajectory = sigmas_trajectory[indices]

        return {
            "prompt": prompt,
            "video_trajectory": video_trajectory,
            "audio_trajectory": audio_trajectory,
            "sigmas": sigmas_trajectory.cpu(),  # [T] actual sigma values for each trajectory entry
        }

    def _compute_subsample_indices(
        self,
        sigmas: torch.Tensor,
        target_timesteps: List[int],
    ) -> List[int]:
        """Find trajectory indices closest to the requested sigma values.

        Args:
            sigmas: Full sigma schedule [steps+1], e.g. from LTX2Scheduler
            target_timesteps: Desired timesteps in [0, 1000] format.
                E.g. [1000, 757, 522, 0] → sigmas [1.0, 0.757, 0.522, 0.0]

        Returns:
            List of indices into the trajectory (len = len(target_timesteps))
        """
        indices = []
        for t in target_timesteps:
            target_sigma = t / 1000.0
            # Find the closest sigma in the schedule
            diffs = (sigmas - target_sigma).abs()
            idx = diffs.argmin().item()
            indices.append(idx)
        return indices

    # ------------------------------------------------------------------
    # Distributed batch generation
    # ------------------------------------------------------------------

    def generate_distributed(
        self,
        prompts: List[str],
        output_dir: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        skip_existing: bool = True,
    ) -> None:
        """
        Generate ODE trajectories with automatic distributed sharding.

        Each rank processes a disjoint subset of prompts (interleaved) and
        saves results to the shared *output_dir*.  Already-generated files
        are skipped when *skip_existing* is True, making the job resumable.
        """
        output_dir = output_dir or self.config.output_dir
        rank, world_size = _get_rank_info()
        device = torch.cuda.current_device()

        # Rank 0 creates output dir before anyone writes
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        total_prompts = len(prompts)
        # Number of iterations each rank must run (may include padding)
        iters_per_rank = int(math.ceil(total_prompts / world_size))

        generated = 0
        skipped = 0

        pbar = tqdm(
            range(iters_per_rank),
            desc=f"[rank {rank}] Generating ODE pairs",
            disable=(rank != 0),
        )

        for local_idx in pbar:
            # Interleaved assignment: rank 0 gets 0, W, 2W, …; rank 1 gets 1, W+1, …
            prompt_idx = local_idx * world_size + rank
            if prompt_idx >= total_prompts:
                continue

            output_path = os.path.join(output_dir, f"{prompt_idx:06d}.pt")

            # Resume: skip already-generated files
            if skip_existing and os.path.exists(output_path):
                skipped += 1
                continue

            prompt = prompts[prompt_idx]

            try:
                trajectory = self.generate_trajectory(
                    prompt=prompt,
                    seed=prompt_idx,
                    device=device,
                    dtype=dtype,
                )
                torch.save(trajectory, output_path)
                generated += 1

            except Exception as e:
                print(f"[rank {rank}] Error generating prompt {prompt_idx}: {e}")
                continue

        # Wait for all ranks before printing summary
        if dist.is_initialized():
            dist.barrier()

        print(f"[rank {rank}] Done. generated={generated}, skipped={skipped}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ODE trajectory pairs (supports torchrun for multi-GPU)",
    )
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--gemma_path", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ode_pairs")
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--video_height", type=int, default=512)
    parser.add_argument("--video_width", type=int, default=768)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--video_guidance_scale", type=float, default=3.0,
                        help="CFG scale for video (LTX-2 native default: 3.0)")
    parser.add_argument("--audio_guidance_scale", type=float, default=7.0,
                        help="CFG scale for audio (LTX-2 native default: 7.0)")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt for CFG (default: LTX-2 official)")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="Disable skipping already-generated files")

    args = parser.parse_args()

    # ---------- distributed init ----------
    if _is_torchrun():
        _init_distributed()
    rank, world_size = _get_rank_info()

    if rank == 0:
        print(f"World size: {world_size}")

    # Deterministic settings
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------- load prompts ----------
    with open(args.prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if rank == 0:
        print(f"Total prompts: {len(prompts)}, ~{math.ceil(len(prompts) / world_size)} per rank")

    # ---------- generate ----------
    config_kwargs = dict(
        teacher_checkpoint=args.teacher_checkpoint,
        gemma_path=args.gemma_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        video_height=args.video_height,
        video_width=args.video_width,
        num_inference_steps=args.num_inference_steps,
        video_guidance_scale=args.video_guidance_scale,
        audio_guidance_scale=args.audio_guidance_scale,
    )
    if args.negative_prompt is not None:
        config_kwargs["negative_prompt"] = args.negative_prompt
    config = ODEGenerationConfig(**config_kwargs)

    generator = LTX2ODEPairGenerator(config)
    generator.generate_distributed(
        prompts=prompts,
        skip_existing=not args.no_skip_existing,
    )

    # ---------- cleanup ----------
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
