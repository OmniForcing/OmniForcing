"""
LTX-2 DMD (Distribution Matching Distillation) Module.

This module implements DMD for LTX-2 audio-video joint generation,
adapted from CausVid's DMD implementation.

Key differences from CausVid:
- Handles both video and audio modalities jointly
- Uses LTX-2's sigma-based timestep format
- Supports audio-video time alignment
"""

from contextlib import nullcontext
from typing import Tuple, Dict, Any, Optional, List
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ltx_core.loader.registry import StateDictRegistry

from ltx_core.components.schedulers import LTX2Scheduler

from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper, create_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper, create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import VideoVAEWrapper, AudioVAEWrapper, create_vae_wrappers
from ltx_distillation.loss import get_denoising_loss
try:
    from ltx_causal.wrapper import CausalLTX2DiffusionWrapper
    from ltx_causal.attention.mask_builder import compute_av_blocks
    from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
except ImportError:
    CausalLTX2DiffusionWrapper = None
    compute_av_blocks = None
    CausalLTXModel = None
    CausalLTXModelConfig = None


class LTX2DMD(nn.Module):
    """
    DMD (Distribution Matching Distillation) module for LTX-2.

    Implements the DMD algorithm for distilling a multi-step diffusion model
    to a few-step model, supporting audio-video joint generation.

    The module contains three diffusion models:
    - generator: Student model being trained
    - real_score: Teacher model (frozen)
    - fake_score: Critic model for discriminating real vs fake

    Training alternates between:
    1. Generator training: minimize DMD loss (KL divergence from teacher)
    2. Critic training: learn to distinguish generator outputs from teacher
    """

    # Audio-video time alignment constants
    VIDEO_LATENT_FPS = 3.0  # 24fps / 8
    AUDIO_LATENT_FPS = 25.0  # 16kHz / 160 / 4

    def __init__(self, args, device: torch.device):
        """
        Initialize the DMD module.

        Args:
            args: Configuration object with:
                - checkpoint_path: Path to LTX-2 checkpoint
                - gemma_path: Path to Gemma text encoder
                - denoising_step_list: List of denoising timesteps
                - num_train_timestep: Total training timesteps
                - real_video_guidance_scale: CFG scale for teacher (video)
                - real_audio_guidance_scale: CFG scale for teacher (audio)
                - gradient_checkpointing: Enable gradient checkpointing
                - mixed_precision: Use bfloat16
                - denoising_loss_type: Type of denoising loss
                - video_shape: [B, F, C, H, W]
                - audio_shape: [B, F_a, C]
            device: Target device
        """
        super().__init__()

        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        # Task types
        self.generator_task_type = getattr(args, "generator_task_type", args.generator_task)
        self.real_task_type = getattr(args, "real_task_type", args.generator_task)
        self.fake_task_type = getattr(args, "fake_task_type", args.generator_task)
        self.training_mode = getattr(args, "training_mode", "bidirectional")
        self.enable_self_forcing = "self_forcing" in str(self.training_mode).lower()
        inferred_causal = (
            "causal" in str(self.training_mode).lower()
            or "causal" in str(self.generator_task_type).lower()
            or "causal" in str(self.real_task_type).lower()
            or "causal" in str(self.fake_task_type).lower()
        )
        self.use_causal_wrapper = bool(getattr(args, "use_causal_wrapper", inferred_causal))
        # Per-model causal wrapper switches (CausVid-style hybrid default).
        # By default:
        # - generator follows global use_causal_wrapper
        # - real/fake follow their task types, enabling bidirectional teacher/critic
        self.generator_use_causal_wrapper = bool(
            getattr(args, "generator_use_causal_wrapper", self.use_causal_wrapper)
        )
        self.real_score_use_causal_wrapper = bool(
            getattr(args, "real_score_use_causal_wrapper", "causal" in str(self.real_task_type).lower())
        )
        self.fake_score_use_causal_wrapper = bool(
            getattr(args, "fake_score_use_causal_wrapper", "causal" in str(self.fake_task_type).lower())
        )
        self.alignment_rounding = str(getattr(args, "alignment_rounding", "round")).lower()
        if self.alignment_rounding not in {"round", "floor", "ceil"}:
            raise ValueError(
                f"Invalid alignment_rounding={self.alignment_rounding}, expected round|floor|ceil"
            )
        if (
            self.generator_use_causal_wrapper
            or self.real_score_use_causal_wrapper
            or self.fake_score_use_causal_wrapper
        ) and CausalLTX2DiffusionWrapper is None:
            raise ImportError(
                "Causal wrapper requires ltx-causal package. "
                "Install with: pip install -e packages/ltx-causal"
            )
        if self.enable_self_forcing and not self.generator_use_causal_wrapper:
            raise ValueError("Stage3 Self-Forcing requires generator_use_causal_wrapper=true")
        self.self_forcing_runtime = str(
            getattr(args, "self_forcing_runtime", "prefix_rerun")
        ).lower()
        if self.self_forcing_runtime not in {"prefix_rerun", "kv_cache"}:
            raise ValueError(
                f"Invalid self_forcing_runtime={self.self_forcing_runtime}, "
                "expected prefix_rerun|kv_cache"
            )
        self.self_forcing_min_generated_blocks = getattr(
            args, "self_forcing_min_generated_blocks", None
        )
        self.self_forcing_max_generated_blocks = getattr(
            args, "self_forcing_max_generated_blocks", None
        )
        self.self_forcing_loss_scope = str(
            getattr(args, "self_forcing_loss_scope", "last_block")
        ).lower()
        if self.self_forcing_loss_scope != "last_block":
            raise ValueError(
                f"Invalid self_forcing_loss_scope={self.self_forcing_loss_scope}, "
                "only last_block is currently supported"
            )

        # Initialize models (will be populated by _init_models or external loading)
        self.generator: LTX2DiffusionWrapper = None
        self.real_score: LTX2DiffusionWrapper = None
        self.fake_score: LTX2DiffusionWrapper = None
        self.text_encoder: GemmaTextEncoderWrapper = None
        self.video_vae: VideoVAEWrapper = None
        self.audio_vae: AudioVAEWrapper = None

        # DMD hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_video_guidance_scale = getattr(args, "real_video_guidance_scale", 3.0)
        self.real_audio_guidance_scale = getattr(args, "real_audio_guidance_scale", 7.0)

        # DMD latent noise mode for KL gradient computation.
        # "direct_noise": add Gaussian noise at target sigma (standard DMD)
        # "teacher_denoise": teacher denoises from high noise to target sigma
        self.dmd_latent_mode = getattr(args, "dmd_latent_mode", "direct_noise")

        # Video/Audio loss weighting for ablation experiments.
        # video_loss_weight + audio_loss_weight need not sum to 1.
        # Supports two-phase training: video-only phase then joint phase.
        self.video_loss_weight = getattr(args, "video_loss_weight", 1.0)
        self.audio_loss_weight = getattr(args, "audio_loss_weight", 1.0)
        # Two-phase: if audio_start_step > 0, audio_loss_weight=0 until that step
        self.audio_start_step = getattr(args, "audio_start_step", 0)

        # Denoising sigmas from LTX-2's native scheduler (shifted + stretched).
        # Number of denoising steps = len(denoising_step_list) - 1
        # (the list includes both start=1.0 and end=0.0, so 4 entries = 3 steps)
        # This matches the sigma schedule used in ODE pair generation and inference.
        num_denoising_steps = len(args.denoising_step_list) - 1
        self.denoising_sigmas = LTX2Scheduler().execute(
            steps=num_denoising_steps
        ).to(device)

        # Pre-compute sigma lookup table for random timestep → sigma conversion.
        # This matches CausVid's approach where scheduler.add_noise() internally
        # does argmin lookup against the scheduler's sigma schedule.
        # We compute a 1001-entry table (timestep 0..1000) using the native
        # LTX2Scheduler's shifted+stretched sigmoid formula.
        # sigma_lookup[t] gives the actual sigma for integer timestep t.
        scheduler = LTX2Scheduler()
        full_sigmas = scheduler.execute(steps=self.num_train_timestep).to(device)  # [1001] values
        # full_sigmas goes from ~1.0 (noise) to 0.0 (clean), same order as timesteps 1000→0
        # We need sigma_lookup[t] where t=0 → sigma=0 (clean) and t=1000 → sigma≈1 (noise)
        # full_sigmas is ordered: sigma[0]=high (noise), sigma[-1]=0 (clean)
        # So sigma_lookup[t] = full_sigmas[1000 - t] maps t=1000→full_sigmas[0], t=0→full_sigmas[1000]
        self.register_buffer(
            'sigma_lookup',
            full_sigmas.flip(0),  # Reverse so index 0=clean(σ≈0), index 1000=noise(σ≈1)
        )

        # Teacher denoise config (only used when dmd_latent_mode == "teacher_denoise")
        if self.dmd_latent_mode == "teacher_denoise":
            self.teacher_num_steps = getattr(args, "teacher_num_steps", 40)
            # How many teacher schedule steps above target_sigma to start from.
            # E.g. offset=5 means: start 5 steps before target in the sigma schedule,
            # so the teacher only runs ~5 Euler steps regardless of target sigma.
            # Smaller = faster + more student structure preserved.
            # Larger = more "on teacher trajectory" but slower.
            self.teacher_start_offset = getattr(args, "teacher_start_offset", 5)
            # Pre-compute fine-grained teacher sigma schedule.
            # teacher_sigmas[0] ≈ 1.0 (noise), teacher_sigmas[-1] = 0.0 (clean)
            teacher_sigmas = LTX2Scheduler().execute(steps=self.teacher_num_steps).to(device)
            self.register_buffer('teacher_sigmas', teacher_sigmas)

        # Loss function
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

        # Block-aware loss weighting for over-exposure suppression.
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.block_weight_mode = getattr(args, "block_weight_mode", "uniform")
        self.block_weight_min = getattr(args, "block_weight_min", 0.5)

        # Inference pipeline (lazy init)
        self.inference_pipeline = None

        # Current training step (updated by trainer)
        self.current_step = 0

    def get_loss_weights(self) -> Tuple[float, float]:
        """Get current video/audio loss weights based on training step."""
        video_w = self.video_loss_weight
        audio_w = self.audio_loss_weight
        if self.audio_start_step > 0 and self.current_step < self.audio_start_step:
            audio_w = 0.0
        return video_w, audio_w

    def timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert integer timestep (0-1000) to sigma using LTX2Scheduler's lookup table.

        Uses a pre-computed lookup table from the native LTX2Scheduler (shifted+stretched
        sigmoid schedule) instead of a linear t/1000 mapping. This matches CausVid's
        approach where scheduler.add_noise() does internal argmin lookup.

        Args:
            timestep: Integer timestep tensor [B, F] in range [0, num_train_timestep]

        Returns:
            Sigma tensor with same shape, values in [0, 1]
        """
        # Clamp to valid range and index into pre-computed lookup table
        t_clamped = timestep.long().clamp(0, self.num_train_timestep)
        return self.sigma_lookup[t_clamped]

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples using flow matching interpolation.

        Flow matching formula: x_t = (1 - sigma) * x_0 + sigma * epsilon

        Args:
            original: Clean samples x_0, shape [B, ...]
            noise: Gaussian noise epsilon, shape [B, ...]
            sigma: Noise level, shape [B] or [B, T] or scalar

        Returns:
            Noisy samples x_t
        """
        # Reshape sigma for broadcasting
        if sigma.dim() == 1:
            # [B] -> [B, 1, 1, 1, ...] for proper broadcasting
            sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
        elif sigma.dim() == 2:
            # [B, T] -> [B, T, 1, 1, ...] for video/audio
            sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
        sigma = sigma.to(dtype=original.dtype)
        return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)

    def init_models(self):
        """
        Initialize all models from checkpoints.

        This method should be called BEFORE FSDP wrapping in distributed training.
        Models must exist before they can be wrapped with FSDP.
        """
        args = self.args

        def _init_log(message: str) -> None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[DMDInit] {message}", flush=True)

        # Get video dimensions from config
        video_height = getattr(args, "video_height", 512)
        video_width = getattr(args, "video_width", 768)

        # Create diffusion wrappers per model (CausVid-style hybrid setup):
        # generator can be causal while real/fake remain bidirectional.
        if isinstance(self.device, int):
            target_device = f"cuda:{self.device}"
        else:
            target_device = str(self.device)

        def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
            if checkpoint_path in checkpoint_state_cache:
                return checkpoint_state_cache[checkpoint_path]
            if checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                loaded = load_file(checkpoint_path)
                checkpoint_state_cache[checkpoint_path] = loaded
                return loaded

            loaded = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(loaded, dict) and "generator" in loaded:
                loaded = loaded["generator"]
            elif isinstance(loaded, dict) and "model" in loaded:
                loaded = loaded["model"]
            elif isinstance(loaded, dict) and "state_dict" in loaded:
                loaded = loaded["state_dict"]
            checkpoint_state_cache[checkpoint_path] = loaded
            return loaded

        def _remap_state_dict_keys(state_dict: dict) -> dict:
            if not state_dict:
                return state_dict

            non_transformer_prefixes = (
                "vae.", "audio_vae.", "vocoder.",
                "model.vae.", "model.audio_vae.", "model.vocoder.",
            )
            remapped_non_transformer_prefixes = (
                "model.audio_embeddings_connector.",
                "model.video_embeddings_connector.",
            )

            sample_keys = list(state_dict.keys())[:20]
            has_diffusion_model = any(k.startswith("model.diffusion_model.") for k in sample_keys)
            if not has_diffusion_model:
                has_diffusion_model = any(k.startswith("model.diffusion_model.") for k in state_dict)

            if has_diffusion_model:
                remapped = {}
                for k, v in state_dict.items():
                    if not k.startswith("model.diffusion_model."):
                        continue
                    new_key = "model." + k[len("model.diffusion_model."):]
                    if any(new_key.startswith(p) for p in remapped_non_transformer_prefixes):
                        continue
                    remapped[new_key] = v
                return remapped

            first_key = next(iter(state_dict))
            if first_key.startswith("model.velocity_model."):
                return {
                    "model." + k[len("model.velocity_model."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("model.velocity_model.")
                }
            if first_key.startswith("model."):
                return {
                    k: v for k, v in state_dict.items()
                    if not any(k.startswith(p) for p in non_transformer_prefixes)
                }
            return {
                "model." + k: v
                for k, v in state_dict.items()
                if not any(k.startswith(p) for p in non_transformer_prefixes)
            }

        def _is_bidirectional_wrapper_state_dict(state_dict: dict) -> bool:
            if not state_dict:
                return False
            sample_keys = list(state_dict.keys())[:20]
            return any(k.startswith("model.velocity_model.") for k in sample_keys) or any(
                k.startswith("model.velocity_model.") for k in state_dict
            )

        def _build_bidirectional_delegate(delegate_checkpoint_path: Optional[str] = None):
            _init_log("build bidirectional delegate wrapper start")
            delegate = create_ltx2_wrapper(
                checkpoint_path=args.checkpoint_path,
                gemma_path=args.gemma_path,
                device=torch.device("cpu"),
                dtype=self.dtype,
                video_height=video_height,
                video_width=video_width,
                registry=shared_registry,
            )
            if delegate_checkpoint_path:
                _init_log(f"load bidirectional delegate state start path={delegate_checkpoint_path}")
                delegate_state_dict = _load_checkpoint_state_dict(delegate_checkpoint_path)
                load_result = delegate.load_state_dict(delegate_state_dict, strict=False)
                if load_result is None:
                    missing, unexpected = [], []
                else:
                    missing, unexpected = load_result
                real_missing = [k for k in missing if "model.velocity_model" in k]
                if real_missing or unexpected:
                    print(
                        f"[Stage3] Bidirectional delegate load from {delegate_checkpoint_path}: "
                        f"missing={len(real_missing)} unexpected={len(unexpected)}"
                    )
            _init_log("build bidirectional delegate wrapper done")
            delegate.eval()
            return delegate

        def _resolve_bidirectional_delegate_checkpoint() -> Optional[str]:
            explicit_delegate_ckpt = getattr(args, "bootstrap_bidirectional_ckpt_path", None)
            if explicit_delegate_ckpt:
                return explicit_delegate_ckpt

            generator_ckpt = getattr(args, "generator_ckpt", None)
            if generator_ckpt:
                return generator_ckpt

            stage1_ckpt = getattr(args, "stage1_ckpt_path", None)
            if stage1_ckpt:
                stage1_state_dict = _load_checkpoint_state_dict(stage1_ckpt)
                if _is_bidirectional_wrapper_state_dict(stage1_state_dict):
                    return stage1_ckpt

            return None

        def _build_wrapper(use_causal: bool):
            if use_causal:
                _init_log("build causal wrapper start")
                causal_config = CausalLTXModelConfig(
                    num_frame_per_block=self.num_frame_per_block,
                    enable_causal_log_rescale=getattr(args, "enable_causal_log_rescale", False),
                    num_audio_sink_tokens=getattr(args, "num_audio_sink_tokens", 0),
                )
                model = CausalLTXModel(causal_config).to(device=target_device, dtype=self.dtype)
                wrapper = CausalLTX2DiffusionWrapper(
                    model=model,
                    video_height=video_height,
                    video_width=video_width,
                    num_frame_per_block=self.num_frame_per_block,
                    disable_causal_mask=getattr(args, "disable_causal_mask", False),
                    num_audio_sink_tokens=getattr(args, "num_audio_sink_tokens", 0),
                )
                state_dict = _remap_state_dict_keys(
                    _load_checkpoint_state_dict(args.checkpoint_path)
                )
                _init_log("load causal wrapper base state done")
                missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
                real_missing = [
                    k for k in missing
                    if "mask_builder" not in k and "audio_sink_tokens" not in k and "causal_gate" not in k
                ]
                if real_missing:
                    print(
                        f"[Stage3] Causal init from {args.checkpoint_path}: "
                        f"missing={len(real_missing)} unexpected={len(unexpected)}"
                    )
                delegate = _build_bidirectional_delegate(_resolve_bidirectional_delegate_checkpoint())
                wrapper.set_bidirectional_delegate(delegate)
                _init_log("build causal wrapper done")
                return wrapper
            _init_log("build bidirectional wrapper start")
            return create_ltx2_wrapper(
                checkpoint_path=args.checkpoint_path,
                gemma_path=args.gemma_path,
                device=self.device,
                dtype=self.dtype,
                video_height=video_height,
                video_width=video_width,
                registry=shared_registry,
            )

        checkpoint_state_cache: Dict[str, dict] = {}
        shared_registry = StateDictRegistry()
        _init_log("generator wrapper init start")
        self.generator = _build_wrapper(self.generator_use_causal_wrapper)
        _init_log("generator wrapper init done")
        _init_log("real_score wrapper init start")
        self.real_score = _build_wrapper(self.real_score_use_causal_wrapper)
        _init_log("real_score wrapper init done")
        _init_log("fake_score wrapper init start")
        self.fake_score = _build_wrapper(self.fake_score_use_causal_wrapper)
        _init_log("fake_score wrapper init done")

        _init_log("text encoder init start")
        self.text_encoder = create_text_encoder_wrapper(
            checkpoint_path=args.checkpoint_path,
            gemma_path=args.gemma_path,
            device=self.device,
            dtype=self.dtype,
            registry=shared_registry,
        )
        _init_log("text encoder init done")

        _init_log("vae init start")
        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=args.checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            registry=shared_registry,
        )
        _init_log("vae init done")

        # Set gradients
        self.generator.set_module_grad(args.generator_grad)
        self.real_score.set_module_grad(args.real_score_grad)
        self.fake_score.set_module_grad(args.fake_score_grad)
        self.text_encoder.requires_grad_(False)
        self.video_vae.requires_grad_(False)
        self.audio_vae.requires_grad_(False)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # Checkpoint loading with priority:
        #   resume_checkpoint > generator_ckpt > stage1_ckpt_path
        stage1_ckpt = getattr(args, "stage1_ckpt_path", None)
        stage1_strict = getattr(args, "stage1_ckpt_strict", False)
        generator_ckpt = getattr(args, "generator_ckpt", None)
        generator_ckpt_strict = getattr(args, "generator_ckpt_strict", False)

        if generator_ckpt:
            print(f"Loading pretrained generator from {generator_ckpt}")
            ckpt = torch.load(generator_ckpt, map_location="cpu")
            gen_sd = ckpt.get("generator", ckpt)
            if self.generator_use_causal_wrapper:
                gen_sd = _remap_state_dict_keys(gen_sd)
            missing_g, unexpected_g = self.generator.load_state_dict(gen_sd, strict=generator_ckpt_strict)
            real_missing_g = [k for k in missing_g if "mask_builder" not in k]
            if real_missing_g:
                print(f"  [generator] missing keys ({len(real_missing_g)}): {real_missing_g[:10]}...")
            if unexpected_g:
                print(f"  [generator] unexpected keys ({len(unexpected_g)}): {unexpected_g[:10]}...")

            sink_key = None
            for k in gen_sd:
                if "audio_sink_tokens" in k:
                    sink_key = k
                    break
            if sink_key is not None:
                for pname, param in self.generator.named_parameters():
                    if "audio_sink_tokens" in pname:
                        assert param.shape == gen_sd[sink_key].shape, (
                            f"[Stage3] Sink token shape mismatch in generator: "
                            f"model={param.shape} vs ckpt={gen_sd[sink_key].shape}"
                        )
                        break
            print("[Stage3] Generator checkpoint load complete")

        elif stage1_ckpt:
            print(f"[Stage2] Loading Stage 1 checkpoint from {stage1_ckpt}")
            ckpt = torch.load(stage1_ckpt, map_location="cpu")

            gen_sd = ckpt.get("generator", ckpt)
            # Stage 3 configs may point stage1_ckpt_path at either:
            # 1. a causal/ODE checkpoint already keyed as model.*
            # 2. a bidirectional DMD checkpoint keyed as model.velocity_model.*
            # The causal generator expects model.* keys, so remap before load.
            if self.generator_use_causal_wrapper:
                gen_sd = _remap_state_dict_keys(gen_sd)
            missing_g, unexpected_g = self.generator.load_state_dict(gen_sd, strict=stage1_strict)
            real_missing_g = [k for k in missing_g if "mask_builder" not in k]
            if real_missing_g:
                print(f"  [generator] missing keys ({len(real_missing_g)}): {real_missing_g[:10]}...")
            if unexpected_g:
                print(f"  [generator] unexpected keys ({len(unexpected_g)}): {unexpected_g[:10]}...")

            # CausVid-style hybrid setup: only load Stage1 ckpt into fake_score
            # when fake_score itself is causal.
            if self.fake_score_use_causal_wrapper:
                missing_f, unexpected_f = self.fake_score.load_state_dict(gen_sd, strict=stage1_strict)
                real_missing_f = [k for k in missing_f if "mask_builder" not in k]
                if real_missing_f:
                    print(f"  [fake_score] missing keys ({len(real_missing_f)}): {real_missing_f[:10]}...")
                if unexpected_f:
                    print(f"  [fake_score] unexpected keys ({len(unexpected_f)}): {unexpected_f[:10]}...")
            else:
                print("[Stage2] fake_score is bidirectional, skip Stage1 causal ckpt load for fake_score")

            # Validate sink token shape consistency
            sink_key = None
            for k in gen_sd:
                if "audio_sink_tokens" in k:
                    sink_key = k
                    break
            if sink_key is not None:
                models_to_check = [("generator", self.generator)]
                if self.fake_score_use_causal_wrapper:
                    models_to_check.append(("fake_score", self.fake_score))
                for name, model in models_to_check:
                    for pname, param in model.named_parameters():
                        if "audio_sink_tokens" in pname:
                            assert param.shape == gen_sd[sink_key].shape, (
                                f"[Stage2] Sink token shape mismatch in {name}: "
                                f"model={param.shape} vs ckpt={gen_sd[sink_key].shape}"
                            )
                            break
            print("[Stage2] Stage1 checkpoint load complete")

    def _round_align(self, value: float) -> int:
        if self.alignment_rounding == "floor":
            return int(torch.floor(torch.tensor(value)).item())
        if self.alignment_rounding == "ceil":
            return int(torch.ceil(torch.tensor(value)).item())
        return int(round(value))

    @staticmethod
    def _is_bidirectional_task(task_type: Optional[str]) -> bool:
        return "bidirectional" in str(task_type).lower()

    @staticmethod
    def _is_causal_task(task_type: Optional[str]) -> bool:
        return "causal" in str(task_type).lower()

    def _get_causal_blocks(self, num_video_frames: int):
        if compute_av_blocks is None:
            raise ImportError("Causal block utilities require the ltx-causal package")
        return compute_av_blocks(
            total_video_latent_frames=num_video_frames,
            num_frame_per_block=self.num_frame_per_block,
        )

    def _build_current_block_masks(
        self,
        num_video_frames: int,
        num_audio_frames: int,
        block_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        blocks = self._get_causal_blocks(num_video_frames)
        batch_size = block_indices.shape[0]

        video_mask = torch.zeros(
            batch_size, num_video_frames, device=block_indices.device, dtype=torch.bool
        )
        audio_mask = torch.zeros(
            batch_size, num_audio_frames, device=block_indices.device, dtype=torch.bool
        )

        for batch_idx, block_idx in enumerate(block_indices.tolist()):
            block = blocks[block_idx]
            video_mask[batch_idx, block.video_start:block.video_end] = True
            audio_end = min(block.audio_end, num_audio_frames)
            if audio_end > block.audio_start:
                audio_mask[batch_idx, block.audio_start:audio_end] = True

        return video_mask, audio_mask

    def _sample_causal_training_blocks(
        self,
        batch_size: int,
        num_video_frames: int,
    ) -> torch.Tensor:
        blocks = self._get_causal_blocks(num_video_frames)
        if len(blocks) <= 1:
            raise ValueError(
                f"Causal training requires at least one standard block, got {num_video_frames} video frames"
            )
        return torch.randint(
            1,
            len(blocks),
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def _sample_causal_supervision_timesteps(
        self,
        batch_size: int,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_timestep = torch.zeros(
            video_mask.shape,
            device=self.device,
            dtype=torch.long,
        )
        audio_timestep = torch.zeros(
            audio_mask.shape,
            device=self.device,
            dtype=torch.long,
        )

        sampled = torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        for batch_idx in range(batch_size):
            video_timestep[batch_idx, video_mask[batch_idx]] = sampled[batch_idx]
            audio_timestep[batch_idx, audio_mask[batch_idx]] = sampled[batch_idx]

        return video_timestep, audio_timestep

    def _prepare_causal_generator_inputs(
        self,
        clean_video: torch.Tensor,
        clean_audio: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_video_frames = clean_video.shape[:2]
        num_audio_frames = clean_audio.shape[1] if clean_audio is not None else 0

        block_indices = self._sample_causal_training_blocks(batch_size, num_video_frames)
        video_mask, audio_mask = self._build_current_block_masks(
            num_video_frames=num_video_frames,
            num_audio_frames=num_audio_frames,
            block_indices=block_indices,
        )

        num_steps = len(self.denoising_sigmas)
        if num_steps > 1:
            sampled_indices = torch.randint(
                0,
                num_steps - 1,
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            sampled_indices = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        sampled_sigmas = self.denoising_sigmas[sampled_indices]

        video_sigma = torch.zeros(
            batch_size, num_video_frames, device=self.device, dtype=self.denoising_sigmas.dtype
        )
        for batch_idx in range(batch_size):
            video_sigma[batch_idx, video_mask[batch_idx]] = sampled_sigmas[batch_idx]

        noise_video = torch.randn_like(clean_video)
        noisy_video = self.add_noise(
            clean_video.flatten(0, 1),
            noise_video.flatten(0, 1),
            video_sigma.flatten(0, 1),
        ).unflatten(0, (batch_size, num_video_frames))

        if clean_audio is None:
            return noisy_video, None, video_sigma, None, video_mask, None

        audio_sigma = torch.zeros(
            batch_size, num_audio_frames, device=self.device, dtype=self.denoising_sigmas.dtype
        )
        for batch_idx in range(batch_size):
            audio_sigma[batch_idx, audio_mask[batch_idx]] = sampled_sigmas[batch_idx]

        noise_audio = torch.randn_like(clean_audio)
        noisy_audio = self.add_noise(clean_audio, noise_audio, audio_sigma)

        return noisy_video, noisy_audio, video_sigma, audio_sigma, video_mask, audio_mask

    def _sample_synced_int(self, min_value: int, max_value: int) -> int:
        if min_value > max_value:
            raise ValueError(f"Invalid synced sampling range [{min_value}, {max_value}]")
        if dist.is_initialized():
            if dist.get_rank() == 0:
                sampled = torch.randint(
                    min_value,
                    max_value + 1,
                    (1,),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                sampled = torch.empty((1,), device=self.device, dtype=torch.long)
            dist.broadcast(sampled, src=0)
            return int(sampled.item())
        return int(
            torch.randint(
                min_value,
                max_value + 1,
                (1,),
                device=self.device,
                dtype=torch.long,
            ).item()
        )

    def _get_self_forcing_rollout_blocks(self, num_video_frames: int):
        blocks = self._get_causal_blocks(num_video_frames)
        standard_blocks = blocks[1:]
        if not standard_blocks:
            raise ValueError(
                f"Self-forcing requires at least one standard causal block, got {num_video_frames} video frames"
            )

        total_blocks = len(standard_blocks)
        max_cfg = self.self_forcing_max_generated_blocks
        if max_cfg is None:
            max_blocks = total_blocks
        else:
            max_blocks = min(total_blocks, max(1, int(max_cfg)))

        min_cfg = self.self_forcing_min_generated_blocks
        if min_cfg is None:
            min_blocks = max_blocks
        else:
            min_blocks = min(max_blocks, max(1, int(min_cfg)))

        rollout_blocks = self._sample_synced_int(min_blocks, max_blocks)
        return blocks[0], standard_blocks[:rollout_blocks]

    def _build_masks_for_blocks(
        self,
        batch_size: int,
        num_video_frames: int,
        num_audio_frames: int,
        blocks,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        video_mask = torch.zeros(
            batch_size, num_video_frames, device=self.device, dtype=torch.bool
        )
        audio_mask = None
        if num_audio_frames > 0:
            audio_mask = torch.zeros(
                batch_size, num_audio_frames, device=self.device, dtype=torch.bool
            )

        for block in blocks:
            video_mask[:, block.video_start:block.video_end] = True
            if audio_mask is not None:
                audio_end = min(block.audio_end, num_audio_frames)
                if audio_end > block.audio_start:
                    audio_mask[:, block.audio_start:audio_end] = True

        return video_mask, audio_mask

    @staticmethod
    def _unwrap_module(module: nn.Module) -> nn.Module:
        current = module
        while hasattr(current, "module"):
            current = current.module
        return current

    def _summon_generator_full_params(self):
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError:
            return nullcontext()

        if isinstance(self.generator, FSDP):
            return FSDP.summon_full_params(self.generator, recurse=True, writeback=False)
        return nullcontext()

    @staticmethod
    def _reshape_sigma_for_block(
        sigma: torch.Tensor,
        target: torch.Tensor,
        wrapper: nn.Module,
    ) -> torch.Tensor:
        reshape_sigma = getattr(wrapper, "_reshape_sigma_for_broadcast", None)
        if callable(reshape_sigma):
            return reshape_sigma(sigma, target)
        if sigma.dim() == 1:
            return sigma.reshape(-1, *[1] * (target.dim() - 1))
        if sigma.dim() == 2:
            return sigma.reshape(*sigma.shape, *[1] * (target.dim() - 2))
        return sigma

    def _renoise_block(self, clean_block: Optional[torch.Tensor], next_sigma: torch.Tensor) -> Optional[torch.Tensor]:
        if clean_block is None:
            return None
        sigma = next_sigma.to(device=clean_block.device, dtype=clean_block.dtype).expand(
            clean_block.shape[0], clean_block.shape[1]
        )
        return self.add_noise(clean_block, torch.randn_like(clean_block), sigma)

    def _run_prefix_rerun_block(
        self,
        *,
        prev_video: torch.Tensor,
        prev_audio: Optional[torch.Tensor],
        block,
        conditional_dict: Dict[str, Any],
        requires_grad: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = prev_video.shape[0]
        video_tail_shape = prev_video.shape[2:]
        current_video = torch.randn(
            (batch_size, block.video_frames, *video_tail_shape),
            device=self.device,
            dtype=self.dtype,
        )
        current_audio = None
        if prev_audio is not None:
            current_audio = torch.randn(
                (batch_size, block.audio_frames, prev_audio.shape[2]),
                device=self.device,
                dtype=self.dtype,
            )

        was_training = self.generator.training
        if not requires_grad:
            self.generator.eval()

        try:
            grad_context = nullcontext() if requires_grad else torch.no_grad()
            with grad_context:
                for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                    prefix_video = torch.cat([prev_video, current_video], dim=1)
                    video_sigma = torch.cat(
                        [
                            torch.zeros(
                                (batch_size, prev_video.shape[1]),
                                device=self.device,
                                dtype=self.denoising_sigmas.dtype,
                            ),
                            sigma.to(device=self.device, dtype=self.denoising_sigmas.dtype).expand(
                                batch_size, current_video.shape[1]
                            ),
                        ],
                        dim=1,
                    )

                    prefix_audio = None
                    audio_sigma = None
                    if current_audio is not None and prev_audio is not None:
                        prefix_audio = torch.cat([prev_audio, current_audio], dim=1)
                        audio_sigma = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prev_audio.shape[1]),
                                    device=self.device,
                                    dtype=self.denoising_sigmas.dtype,
                                ),
                                sigma.to(device=self.device, dtype=self.denoising_sigmas.dtype).expand(
                                    batch_size, current_audio.shape[1]
                                ),
                            ],
                            dim=1,
                        )

                    pred_video_prefix, pred_audio_prefix = self.generator(
                        noisy_image_or_video=prefix_video,
                        conditional_dict=conditional_dict,
                        timestep=video_sigma,
                        noisy_audio=prefix_audio,
                        audio_timestep=audio_sigma,
                        use_causal_timestep=False,
                        force_bidirectional=False,
                    )

                    current_video = pred_video_prefix[:, block.video_start:block.video_end]
                    if current_audio is not None:
                        if pred_audio_prefix is None:
                            raise RuntimeError(
                                "Generator returned no audio prediction during self-forcing rollout"
                            )
                        current_audio = pred_audio_prefix[:, block.audio_start:block.audio_end]

                    next_sigma = self.denoising_sigmas[sigma_idx + 1]
                    if float(next_sigma.item()) > 0.0:
                        current_video = self._renoise_block(current_video, next_sigma)
                        current_audio = self._renoise_block(current_audio, next_sigma)
        finally:
            if not requires_grad and was_training:
                self.generator.train()

        return current_video, current_audio

    def _build_self_forcing_prefix_cache(
        self,
        prefix_video: torch.Tensor,
        prefix_audio: Optional[torch.Tensor],
        conditional_dict: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if prefix_video.shape[1] == 0:
            return None

        with self._summon_generator_full_params():
            generator_module = self._unwrap_module(self.generator)
            kv_cache = None
            prefix_blocks = self._get_causal_blocks(prefix_video.shape[1])

            for block in prefix_blocks:
                video_block = prefix_video[:, block.video_start:block.video_end]
                audio_block = None
                audio_sigma = None
                if prefix_audio is not None:
                    audio_end = min(block.audio_end, prefix_audio.shape[1])
                    if audio_end > block.audio_start:
                        audio_block = prefix_audio[:, block.audio_start:audio_end]
                        audio_sigma = torch.zeros(
                            (video_block.shape[0], audio_block.shape[1]),
                            device=audio_block.device,
                            dtype=self.denoising_sigmas.dtype,
                        )

                video_sigma = torch.zeros(
                    (video_block.shape[0], video_block.shape[1]),
                    device=video_block.device,
                    dtype=self.denoising_sigmas.dtype,
                )
                _, _, kv_cache = generator_module.model.forward_inference(
                    video_latent=video_block,
                    audio_latent=audio_block,
                    timesteps=video_sigma,
                    audio_timesteps=audio_sigma,
                    video_context=conditional_dict["video_context"],
                    audio_context=conditional_dict["audio_context"],
                    video_context_mask=conditional_dict.get("video_context_mask"),
                    audio_context_mask=conditional_dict.get("audio_context_mask"),
                    kv_cache=kv_cache,
                    video_start_frame=block.video_start,
                    audio_start_frame=block.audio_start,
                    include_audio_sinks=(block.block_idx == 0),
                )

        return kv_cache

    def _run_kv_rollout_block(
        self,
        *,
        prev_video: torch.Tensor,
        prev_audio: Optional[torch.Tensor],
        block,
        conditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kv_cache = self._build_self_forcing_prefix_cache(prev_video, prev_audio, conditional_dict)

        batch_size = prev_video.shape[0]
        video_tail_shape = prev_video.shape[2:]
        current_video = torch.randn(
            (batch_size, block.video_frames, *video_tail_shape),
            device=self.device,
            dtype=self.dtype,
        )
        current_audio = None
        if prev_audio is not None:
            current_audio = torch.randn(
                (batch_size, block.audio_frames, prev_audio.shape[2]),
                device=self.device,
                dtype=self.dtype,
            )

        with self._summon_generator_full_params():
            generator_module = self._unwrap_module(self.generator)
            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                video_sigma = sigma.to(device=self.device, dtype=self.denoising_sigmas.dtype).expand(
                    batch_size, current_video.shape[1]
                )
                audio_sigma = None
                if current_audio is not None:
                    audio_sigma = sigma.to(device=self.device, dtype=self.denoising_sigmas.dtype).expand(
                        batch_size, current_audio.shape[1]
                    )

                pred_video, pred_audio, _ = generator_module.model.forward_inference(
                    video_latent=current_video,
                    audio_latent=current_audio,
                    timesteps=video_sigma,
                    audio_timesteps=audio_sigma,
                    video_context=conditional_dict["video_context"],
                    audio_context=conditional_dict["audio_context"],
                    video_context_mask=conditional_dict.get("video_context_mask"),
                    audio_context_mask=conditional_dict.get("audio_context_mask"),
                    kv_cache=kv_cache,
                    video_start_frame=block.video_start,
                    audio_start_frame=block.audio_start,
                    include_audio_sinks=False,
                )

                sigma_video_broadcast = self._reshape_sigma_for_block(video_sigma, current_video, generator_module)
                current_video = (
                    current_video.to(torch.float32)
                    - pred_video.to(torch.float32) * sigma_video_broadcast.to(torch.float32)
                ).to(self.dtype)

                if current_audio is not None:
                    sigma_audio_broadcast = self._reshape_sigma_for_block(audio_sigma, current_audio, generator_module)
                    current_audio = (
                        current_audio.to(torch.float32)
                        - pred_audio.to(torch.float32) * sigma_audio_broadcast.to(torch.float32)
                    ).to(self.dtype)

                next_sigma = self.denoising_sigmas[sigma_idx + 1]
                if float(next_sigma.item()) > 0.0:
                    current_video = self._renoise_block(current_video, next_sigma)
                    current_audio = self._renoise_block(current_audio, next_sigma)

        return current_video, current_audio

    def _run_self_forcing_rollout(
        self,
        *,
        clean_video: torch.Tensor,
        clean_audio: Optional[torch.Tensor],
        conditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if clean_video is None:
            raise ValueError("Self-forcing rollout requires clean_video from ODE data")
        if getattr(self.args, "backward_simulation", True):
            raise ValueError("Self-forcing rollout requires backward_simulation=false")

        prefix_block, rollout_blocks = self._get_self_forcing_rollout_blocks(clean_video.shape[1])
        rollout_video = clean_video[:, prefix_block.video_start:prefix_block.video_end].clone()
        rollout_audio = None
        if clean_audio is not None:
            audio_end = min(prefix_block.audio_end, clean_audio.shape[1])
            rollout_audio = clean_audio[:, prefix_block.audio_start:audio_end].clone()

        for idx, block in enumerate(rollout_blocks):
            is_last = idx == len(rollout_blocks) - 1
            if self.self_forcing_runtime == "kv_cache" and not is_last:
                current_video, current_audio = self._run_kv_rollout_block(
                    prev_video=rollout_video,
                    prev_audio=rollout_audio,
                    block=block,
                    conditional_dict=conditional_dict,
                )
            else:
                current_video, current_audio = self._run_prefix_rerun_block(
                    prev_video=rollout_video,
                    prev_audio=rollout_audio,
                    block=block,
                    conditional_dict=conditional_dict,
                    requires_grad=is_last,
                )

            if not is_last:
                current_video = current_video.detach()
                if current_audio is not None:
                    current_audio = current_audio.detach()

            rollout_video = torch.cat([rollout_video, current_video], dim=1)
            if current_audio is not None:
                if rollout_audio is None:
                    rollout_audio = current_audio
                else:
                    rollout_audio = torch.cat([rollout_audio, current_audio], dim=1)

        num_audio_frames = rollout_audio.shape[1] if rollout_audio is not None else 0
        video_loss_mask, audio_loss_mask = self._build_masks_for_blocks(
            batch_size=rollout_video.shape[0],
            num_video_frames=rollout_video.shape[1],
            num_audio_frames=num_audio_frames,
            blocks=[rollout_blocks[-1]],
        )
        rollout_log = {
            "self_forcing_rollout_blocks": len(rollout_blocks),
            "self_forcing_rollout_video_frames": rollout_video.shape[1],
            "self_forcing_rollout_audio_frames": num_audio_frames,
            "self_forcing_runtime": 0 if self.self_forcing_runtime == "prefix_rerun" else 1,
        }
        return rollout_video, rollout_audio, video_loss_mask, audio_loss_mask, rollout_log

    def _process_timestep(self, timestep: torch.Tensor, task_type: str) -> torch.Tensor:
        """
        Process timestep based on task type.

        For causal tasks, each block of num_frame_per_block frames shares the
        same timestep (noise level), matching CausVid semantics.

        Args:
            timestep: [B, F] tensor of timesteps
            task_type: "bidirectional_av", "bidirectional_video", "causal_av", etc.

        Returns:
            Processed timestep tensor
        """
        if self._is_bidirectional_task(task_type):
            for i in range(timestep.shape[0]):
                timestep[i] = timestep[i, 0]
            return timestep
        elif "causal" in task_type:
            result = timestep.clone()
            if result.shape[1] <= 1:
                return result
            idx = 1
            while idx < result.shape[1]:
                end = min(idx + self.num_frame_per_block, result.shape[1])
                result[:, idx:end] = result[:, idx:idx + 1].expand(-1, end - idx)
                idx = end
            return result
        else:
            return timestep

    def _compute_audio_timestep(
        self,
        video_timestep: torch.Tensor,
        num_audio_frames: int,
        task_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute audio timestep from video timestep.

        In bidirectional mode, all frames use the same timestep.
        In causal mode, audio frames inherit the timestep from their
        corresponding video block via the AV alignment ratio.
        """
        B = video_timestep.shape[0]
        num_video_frames = video_timestep.shape[1]
        mode = task_type or self.real_task_type

        if self._is_bidirectional_task(mode):
            return video_timestep[:, 0:1].expand(B, num_audio_frames)

        # Causal/non-bidirectional: map audio blocks to the video block sigma
        # defined by the causal wrapper's Global Prefix schedule.
        audio_timestep = torch.zeros(
            B, num_audio_frames, device=video_timestep.device, dtype=video_timestep.dtype
        )
        for block in self._get_causal_blocks(num_video_frames):
            if block.audio_start >= num_audio_frames:
                break
            audio_end = min(block.audio_end, num_audio_frames)
            if audio_end <= block.audio_start:
                continue
            audio_timestep[:, block.audio_start:audio_end] = video_timestep[
                :, block.video_start:block.video_start + 1
            ].expand(B, audio_end - block.audio_start)
        return audio_timestep

    @torch.no_grad()
    def _teacher_denoise_cfg_step(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One teacher denoising step with classifier-free guidance.

        Calls real_score twice (conditional + unconditional) and applies
        LTX-2's CFG formula with separate video/audio guidance scales.

        Returns:
            Tuple of (pred_video_x0, pred_audio_x0)
        """
        pred_cond_video, pred_cond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        pred_uncond_video, pred_uncond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=unconditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        # CFG: output = cond + (scale - 1) * (cond - uncond)
        pred_video_x0 = pred_cond_video + (self.real_video_guidance_scale - 1) * (
            pred_cond_video - pred_uncond_video
        )
        pred_audio_x0 = pred_cond_audio + (self.real_audio_guidance_scale - 1) * (
            pred_cond_audio - pred_uncond_audio
        )

        return pred_video_x0, pred_audio_x0

    @torch.no_grad()
    def _get_noisy_latent_via_teacher_denoise(
        self,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        target_video_sigma: torch.Tensor,
        target_audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher denoises from high noise to target sigma, producing latents
        on the teacher's ODE trajectory instead of the Gaussian interpolation line.

        The number of Euler steps is determined solely by where target_sigma
        falls in the pre-computed teacher_sigmas schedule — no randomization.

        Args:
            clean_video: Generator-predicted clean video [B, F_v, C, H, W]
            clean_audio: Generator-predicted clean audio [B, F_a, C]
            target_video_sigma: Target sigma [B, F_v]
            target_audio_sigma: Target sigma [B, F_a]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings

        Returns:
            (noisy_video, noisy_audio, actual_video_sigma, actual_audio_sigma)
            where actual sigmas are the exact teacher schedule values at the
            stop point, guaranteed to match the returned latents' noise level.
        """
        B = clean_video.shape[0]
        F_v = clean_video.shape[1]
        F_a = clean_audio.shape[1]
        device = clean_video.device

        teacher_sigmas = self.teacher_sigmas  # [N+1], descending: [0]≈1.0, [-1]=0.0
        offset = self.teacher_start_offset

        # In bidirectional mode all frames share the same sigma
        target_scalar = target_video_sigma[:, 0]  # [B]

        # Find the closest index in teacher_sigmas for each batch element
        target_idx = torch.argmin(
            (teacher_sigmas.unsqueeze(0) - target_scalar.unsqueeze(1)).abs(),
            dim=1,
        )  # [B]
        target_idx = target_idx.clamp(min=1)  # at least 1 denoising step

        # Start index: `offset` steps before target in the schedule.
        # start_idx < target_idx, so teacher_sigmas[start_idx] > teacher_sigmas[target_idx].
        # Clamped to 0 so we never go before the schedule start.
        start_idx = (target_idx - offset).clamp(min=0)  # [B]
        num_steps = target_idx - start_idx  # [B], exactly how many Euler steps each element needs

        # NCCL safety: all ranks must call real_score() the same number of times.
        max_steps = num_steps.max().item()
        if dist.is_initialized():
            max_steps_tensor = torch.tensor(max_steps, device=device, dtype=torch.long)
            dist.all_reduce(max_steps_tensor, op=dist.ReduceOp.MAX)
            max_steps = max_steps_tensor.item()

        # Add noise at each element's start sigma (not necessarily pure noise)
        noise_video = torch.randn_like(clean_video)
        noise_audio = torch.randn_like(clean_audio)

        start_sigma_per_elem = teacher_sigmas[start_idx]  # [B]
        s_v = start_sigma_per_elem.unsqueeze(1).expand(B, F_v)
        s_a = start_sigma_per_elem.unsqueeze(1).expand(B, F_a)

        current_video = self.add_noise(
            clean_video.flatten(0, 1),
            noise_video.flatten(0, 1),
            s_v.flatten(0, 1),
        ).unflatten(0, (B, F_v))
        current_audio = self.add_noise(clean_audio, noise_audio, s_a)

        # Teacher Euler denoising: each element runs from its own start_idx to target_idx.
        # We iterate max_steps times; each element's absolute schedule index is start_idx + step_i.
        for step_i in range(max_steps):
            active = (step_i < num_steps)  # [B]

            # Each element may be at a different position in the schedule
            abs_idx = start_idx + step_i  # [B]
            cur_sigma = teacher_sigmas[abs_idx]    # [B]
            nxt_sigma = teacher_sigmas[(abs_idx + 1).clamp(max=len(teacher_sigmas) - 1)]  # [B]

            v_sigma = cur_sigma.unsqueeze(1).expand(B, F_v)
            a_sigma = cur_sigma.unsqueeze(1).expand(B, F_a)

            pred_v_x0, pred_a_x0 = self._teacher_denoise_cfg_step(
                noisy_video=current_video,
                noisy_audio=current_audio,
                video_sigma=v_sigma,
                audio_sigma=a_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

            # Euler step: v = (x_t - x_0) / sigma, x_{t'} = x_t + v * (sigma' - sigma)
            cur_sigma_bcast = cur_sigma.view(B, 1, 1, 1, 1)
            dsigma = (nxt_sigma - cur_sigma).view(B, 1, 1, 1, 1)
            vel_v = (current_video - pred_v_x0) / cur_sigma_bcast
            nxt_v = current_video + vel_v * dsigma

            cur_sigma_audio = cur_sigma.view(B, 1, 1)
            dsigma_audio = (nxt_sigma - cur_sigma).view(B, 1, 1)
            vel_a = (current_audio - pred_a_x0) / cur_sigma_audio
            nxt_a = current_audio + vel_a * dsigma_audio

            m_v = active.view(B, 1, 1, 1, 1).expand_as(current_video)
            m_a = active.view(B, 1, 1).expand_as(current_audio)
            current_video = torch.where(m_v, nxt_v, current_video)
            current_audio = torch.where(m_a, nxt_a, current_audio)

        # Actual sigma at the exact stopping point (from the teacher schedule)
        actual_sigma = teacher_sigmas[target_idx]  # [B]
        actual_video_sigma = actual_sigma.unsqueeze(1).expand(B, F_v)
        actual_audio_sigma = actual_sigma.unsqueeze(1).expand(B, F_a)

        return current_video, current_audio, actual_video_sigma, actual_audio_sigma

    def _compute_kl_grad(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compute KL gradient for both video and audio.

        This implements Equation 7 from the DMD paper.

        Args:
            video_sigma: Noise level sigma [B, F_v], passed directly to score networks.
            audio_sigma: Noise level sigma [B, F_a], passed directly to score networks.
        """
        # Step 1: Fake score prediction
        pred_fake_video, pred_fake_audio = self.fake_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        # Step 2: Real score prediction with CFG
        pred_real_cond_video, pred_real_cond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        pred_real_uncond_video, pred_real_uncond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=unconditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        # Apply CFG: output = cond + (scale - 1) * (cond - uncond)
        # This matches LTX-2's native CFGGuider.delta = (scale - 1) * (cond - uncond)
        # With video_scale=3.0: effective = 3.0*cond - 2.0*uncond
        # With audio_scale=7.0: effective = 7.0*cond - 6.0*uncond
        pred_real_video = pred_real_cond_video + (self.real_video_guidance_scale - 1) * (
            pred_real_cond_video - pred_real_uncond_video
        )
        pred_real_audio = pred_real_cond_audio + (self.real_audio_guidance_scale - 1) * (
            pred_real_cond_audio - pred_real_uncond_audio
        )

        # Step 3: Compute DMD gradient
        grad_video = pred_fake_video - pred_real_video
        grad_audio = pred_fake_audio - pred_real_audio

        # Step 4: Gradient normalization (Eq. 8)
        if normalization:
            # Video normalization
            p_real_video = clean_video - pred_real_video
            if video_loss_mask is not None:
                video_mask = video_loss_mask.to(p_real_video.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # Count all active latent elements, not just active frames.
                video_active = video_mask.expand_as(p_real_video).sum(dim=[1, 2, 3, 4], keepdim=True).clamp_min(1.0)
                normalizer_video = (torch.abs(p_real_video) * video_mask).sum(dim=[1, 2, 3, 4], keepdim=True)
                normalizer_video = normalizer_video / video_active
            else:
                normalizer_video = torch.abs(p_real_video).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad_video = grad_video / (normalizer_video + 1e-8)

            # Audio normalization
            p_real_audio = clean_audio - pred_real_audio
            if audio_loss_mask is not None:
                audio_mask = audio_loss_mask.to(p_real_audio.dtype).unsqueeze(-1)
                audio_active = audio_mask.expand_as(p_real_audio).sum(dim=[1, 2], keepdim=True).clamp_min(1.0)
                normalizer_audio = (torch.abs(p_real_audio) * audio_mask).sum(dim=[1, 2], keepdim=True)
                normalizer_audio = normalizer_audio / audio_active
            else:
                normalizer_audio = torch.abs(p_real_audio).mean(dim=[1, 2], keepdim=True)
            grad_audio = grad_audio / (normalizer_audio + 1e-8)

        grad_video = torch.nan_to_num(grad_video)
        grad_audio = torch.nan_to_num(grad_audio)

        log_dict = {
            "dmdtrain_gradient_norm_video": torch.mean(torch.abs(grad_video)).item(),
            "dmdtrain_gradient_norm_audio": torch.mean(torch.abs(grad_audio)).item(),
            "real_score_video": torch.mean(torch.abs(pred_real_video)).item(),
            "real_score_audio": torch.mean(torch.abs(pred_real_audio)).item(),
            "fake_score_video": torch.mean(torch.abs(pred_fake_video)).item(),
            "fake_score_audio": torch.mean(torch.abs(pred_fake_audio)).item(),
        }

        return grad_video, grad_audio, log_dict

    def _compute_block_weights(self, num_frames: int, *, is_audio: bool = False) -> torch.Tensor:
        """
        Compute per-frame loss weights based on block position.

        For "linear_ramp", early blocks get lower weight (block_weight_min)
        ramping linearly to 1.0 at the last block.
        For "uniform" or "none", returns all-ones.

        Returns:
            Tensor [num_frames] of per-frame weights on self.device.
        """
        if self.block_weight_mode == "uniform" or self.block_weight_mode == "none":
            return torch.ones(num_frames, device=self.device, dtype=torch.float64)

        if self._is_causal_task(self.generator_task_type):
            blocks = self._get_causal_blocks(
                math.ceil((num_frames - 1) / 25) * self.num_frame_per_block + 1
            ) if is_audio else self._get_causal_blocks(num_frames)
            if is_audio:
                blocks = [block for block in blocks if block.audio_start < num_frames]
            else:
                blocks = [block for block in blocks if block.video_start < num_frames]
            n_blocks = len(blocks)
        else:
            nfpb = self.num_frame_per_block
            n_blocks = math.ceil(num_frames / nfpb)
            blocks = None

        if n_blocks <= 1:
            return torch.ones(num_frames, device=self.device, dtype=torch.float64)

        weights = torch.ones(num_frames, device=self.device, dtype=torch.float64)
        if blocks is not None:
            for blk_idx, block in enumerate(blocks):
                w = self.block_weight_min + (1.0 - self.block_weight_min) * blk_idx / (n_blocks - 1)
                start = block.audio_start if is_audio else block.video_start
                end = min(block.audio_end if is_audio else block.video_end, num_frames)
                weights[start:end] = w
        else:
            for blk in range(n_blocks):
                start = blk * nfpb
                end = min(start + nfpb, num_frames)
                w = self.block_weight_min + (1.0 - self.block_weight_min) * blk / (n_blocks - 1)
                weights[start:end] = w

        return weights

    @staticmethod
    def _masked_weighted_mean(
        values: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return (values * weights).mean()
        mask_f = mask.to(values.dtype)
        weighted = values * weights * mask_f
        denom = (weights * mask_f).sum().clamp_min(1.0)
        return weighted.sum() / denom

    def _compute_masked_denoising_loss(
        self,
        *,
        target: torch.Tensor,
        prediction: torch.Tensor,
        noise: torch.Tensor,
        flow_pred: Optional[torch.Tensor],
        timestep: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return self.denoising_loss_func(
                x=target,
                x_pred=prediction,
                noise=noise,
                noise_pred=None,
                alphas_cumprod=None,
                timestep=timestep,
                flow_pred=flow_pred,
            )

        loss_type = str(getattr(self.args, "denoising_loss_type", "velocity")).lower()
        if loss_type == "x0":
            diff = (target.double() - prediction.double()) ** 2
        elif loss_type in {"velocity", "flow"}:
            pred = flow_pred.double() if flow_pred is not None else (noise.double() - prediction.double())
            diff = (pred - (noise.double() - target.double())) ** 2
        else:
            raise NotImplementedError(
                f"Masked causal critic loss does not support denoising_loss_type={loss_type}"
            )

        reduce_dims = tuple(range(2, diff.dim()))
        per_frame = diff.mean(dim=reduce_dims)
        return self._masked_weighted_mean(
            per_frame,
            torch.ones_like(per_frame, dtype=per_frame.dtype),
            mask,
        ).to(target.dtype)

    def compute_distribution_matching_loss(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the DMD loss for video and audio jointly.

        Supports block-aware per-frame weighting for causal over-exposure suppression
        and causal block-wise timestep unification.

        Args:
            video_latent: Clean video latent [B, F, C, H, W]
            audio_latent: Clean audio latent [B, F_a, C]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings

        Returns:
            Tuple of (total_loss, log_dict)
        """
        B, F_v = video_latent.shape[:2]
        F_a = audio_latent.shape[1]

        with torch.no_grad():
            if video_loss_mask is not None and audio_loss_mask is not None:
                video_timestep, audio_timestep = self._sample_causal_supervision_timesteps(
                    B,
                    video_loss_mask,
                    audio_loss_mask,
                )
            else:
                video_timestep = torch.randint(
                    0, self.num_train_timestep,
                    [B, F_v],
                    device=self.device,
                    dtype=torch.long,
                )
                video_timestep = self._process_timestep(video_timestep, self.real_task_type)
                video_timestep = video_timestep.clamp(self.min_step, self.max_step)
                audio_timestep = self._compute_audio_timestep(
                    video_timestep, F_a, task_type=self.real_task_type
                )

            video_sigma = self.timestep_to_sigma(video_timestep)
            audio_sigma = self.timestep_to_sigma(audio_timestep)

            if self.dmd_latent_mode == "teacher_denoise":
                noisy_video, noisy_audio, video_sigma, audio_sigma = \
                    self._get_noisy_latent_via_teacher_denoise(
                        clean_video=video_latent,
                        clean_audio=audio_latent,
                        target_video_sigma=video_sigma,
                        target_audio_sigma=audio_sigma,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                    )
            else:
                noise_video = torch.randn_like(video_latent)
                noise_audio = torch.randn_like(audio_latent)

                noisy_video = self.add_noise(
                    video_latent.flatten(0, 1),
                    noise_video.flatten(0, 1),
                    video_sigma.flatten(0, 1),
                ).unflatten(0, (B, F_v))

                noisy_audio = self.add_noise(
                    audio_latent,
                    noise_audio,
                    audio_sigma,
                )

            grad_video, grad_audio, log_dict = self._compute_kl_grad(
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                clean_video=video_latent,
                clean_audio=audio_latent,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_loss_mask=video_loss_mask,
                audio_loss_mask=audio_loss_mask,
            )

        # Block-aware per-frame loss weighting (over-exposure suppression)
        video_block_w = self._compute_block_weights(F_v)  # [F_v]
        audio_block_w = self._compute_block_weights(F_a, is_audio=True)  # [F_a]

        # Per-frame MSE then weight
        video_diff = video_latent.double() - (video_latent.double() - grad_video.double()).detach()
        video_per_frame = (video_diff ** 2).mean(dim=[2, 3, 4])  # [B, F_v]
        video_loss = 0.5 * self._masked_weighted_mean(
            video_per_frame,
            video_block_w.unsqueeze(0),
            video_loss_mask,
        )

        audio_diff = audio_latent.double() - (audio_latent.double() - grad_audio.double()).detach()
        audio_per_frame = (audio_diff ** 2).mean(dim=2)  # [B, F_a]
        audio_loss = 0.5 * self._masked_weighted_mean(
            audio_per_frame,
            audio_block_w.unsqueeze(0),
            audio_loss_mask,
        )

        video_w, audio_w = self.get_loss_weights()
        total_loss = video_w * video_loss + audio_w * audio_loss

        log_dict["video_dmd_loss"] = video_loss.detach()
        log_dict["audio_dmd_loss"] = audio_loss.detach()
        log_dict["video_loss_weight"] = video_w
        log_dict["audio_loss_weight"] = audio_w
        log_dict["alignment/video_sigma_mean"] = video_sigma.float().mean().item()
        log_dict["alignment/audio_sigma_mean"] = audio_sigma.float().mean().item()

        return total_loss, log_dict

    def _initialize_inference_pipeline(self):
        """Initialize the inference pipeline for backward simulation."""
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVTrajectoryPipeline

        self.inference_pipeline = BidirectionalAVTrajectoryPipeline(
            generator=self.generator,
            add_noise_fn=self.add_noise,
            denoising_sigmas=self.denoising_sigmas,
        )

    @torch.no_grad()
    def _consistency_backward_simulation(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        conditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Simulate generator input using backward simulation.

        Returns trajectory of noisy inputs at each denoising step.

        Note: The generator is temporarily switched to eval() mode during
        backward simulation. This disables gradient checkpointing, which
        would otherwise conflict with FSDP under torch.no_grad() (checkpoint
        requires grad-enabled tensors). After simulation, the generator is
        restored to train() mode so that gradient checkpointing remains
        active for the subsequent gradient-enabled forward pass — essential
        for the 19B model's memory footprint.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        # Temporarily disable gradient checkpointing by switching to eval().
        # Under @torch.no_grad(), FSDP + gradient checkpointing conflicts
        # because checkpoint requires grad-enabled tensors.
        self.generator.eval()
        try:
            result = self.inference_pipeline.inference_with_trajectory(
                video_noise=video_noise,
                audio_noise=audio_noise,
                conditional_dict=conditional_dict,
            )
        finally:
            # Restore train() so gradient checkpointing is active for
            # the gradient-enabled generator forward pass that follows.
            self.generator.train()

        return result

    def _run_generator(
        self,
        video_shape: List[int],
        audio_shape: List[int],
        conditional_dict: Dict[str, Any],
        clean_video: Optional[torch.Tensor] = None,
        clean_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """
        Run generator with backward simulation.

        Returns predicted clean video and audio.
        """
        B = video_shape[0]
        F_v = video_shape[1]
        F_a = audio_shape[1]

        video_loss_mask = None
        audio_loss_mask = None
        rollout_log: Dict[str, Any] = {}

        if self.enable_self_forcing:
            pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log = (
                self._run_self_forcing_rollout(
                    clean_video=clean_video,
                    clean_audio=clean_audio,
                    conditional_dict=conditional_dict,
                )
            )
            return pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log

        # Step 1: Backward simulation or ODE data
        if getattr(self.args, "backward_simulation", True):
            if self._is_causal_task(self.generator_task_type):
                raise NotImplementedError(
                    "Causal Stage-3 DMD currently requires backward_simulation=false so training "
                    "matches the clean-prefix/current-block-noisy inference distribution."
                )
            video_noise = torch.randn(video_shape, device=self.device, dtype=self.dtype)
            audio_noise = torch.randn(audio_shape, device=self.device, dtype=self.dtype)

            simulated_video, simulated_audio = self._consistency_backward_simulation(
                video_noise=video_noise,
                audio_noise=audio_noise,
                conditional_dict=conditional_dict,
            )
        else:
            if self._is_causal_task(self.generator_task_type):
                noisy_video, noisy_audio, video_sigma, audio_sigma, video_loss_mask, audio_loss_mask = (
                    self._prepare_causal_generator_inputs(
                        clean_video=clean_video,
                        clean_audio=clean_audio,
                    )
                )
                pred_video, pred_audio = self.generator(
                    noisy_image_or_video=noisy_video,
                    conditional_dict=conditional_dict,
                    timestep=video_sigma,
                    noisy_audio=noisy_audio,
                    audio_timestep=audio_sigma,
                    use_causal_timestep=False,
                )
                return pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log

            # Use provided clean latents
            simulated_video = []
            simulated_audio = []

            for sigma in self.denoising_sigmas:
                noise_v = torch.randn(video_shape, device=self.device, dtype=self.dtype)
                noise_a = torch.randn(audio_shape, device=self.device, dtype=self.dtype)

                sigma_tensor = sigma * torch.ones([B, F_v], device=self.device)
                sigma_tensor_a = sigma * torch.ones([B, F_a], device=self.device)

                if sigma > 0:
                    noisy_video = self.add_noise(
                        clean_video.flatten(0, 1),
                        noise_v.flatten(0, 1),
                        sigma_tensor.flatten(0, 1),
                    ).unflatten(0, (B, F_v))
                    noisy_audio = self.add_noise(clean_audio, noise_a, sigma_tensor_a)
                else:
                    noisy_video = clean_video
                    noisy_audio = clean_audio

                simulated_video.append(noisy_video)
                simulated_audio.append(noisy_audio)

            simulated_video = torch.stack(simulated_video, dim=1)
            simulated_audio = torch.stack(simulated_audio, dim=1)

        # Step 2: Random timestep selection
        num_steps = len(self.denoising_sigmas)
        index = torch.randint(0, num_steps, [B, F_v], device=self.device, dtype=torch.long)
        index = self._process_timestep(index, self.generator_task_type)
        if self._is_bidirectional_task(self.generator_task_type):
            # Keep the Stage-1 bidirectional path byte-for-byte aligned with
            # the 88fb145 DMD fix semantics: one shared step per sample across
            # all video and audio frames.
            noisy_video = torch.gather(
                simulated_video,
                dim=1,
                index=index[:, :1, None, None, None, None].expand(-1, -1, F_v, *video_shape[2:]),
            ).squeeze(1)
            noisy_audio = torch.gather(
                simulated_audio,
                dim=1,
                index=index[:, :1, None, None].expand(-1, -1, F_a, audio_shape[2]),
            ).squeeze(1)

            sigma = self.denoising_sigmas[index[:, 0]]
            video_sigma = sigma.unsqueeze(1).expand(B, F_v)
            audio_sigma = sigma.unsqueeze(1).expand(B, F_a)
        else:
            audio_index = self._compute_audio_timestep(
                index, F_a, task_type=self.generator_task_type
            ).clamp(0, num_steps - 1)

            noisy_video = torch.gather(
                simulated_video, dim=1,
                index=index.reshape(B, 1, F_v, 1, 1, 1).expand(-1, -1, -1, *video_shape[2:])
            ).squeeze(1)

            noisy_audio = torch.gather(
                simulated_audio, dim=1,
                index=audio_index.reshape(B, 1, F_a, 1).expand(-1, -1, -1, audio_shape[2])
            ).squeeze(1)

            # Step 3: Generator prediction (per-frame/per-audio-frame sigma)
            video_sigma = self.denoising_sigmas[index]
            audio_sigma = self.denoising_sigmas[audio_index]

        pred_video, pred_audio = self.generator(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        return pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log

    def generator_loss(
        self,
        video_shape: List[int],
        audio_shape: List[int],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_video: Optional[torch.Tensor] = None,
        clean_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute generator loss using DMD.

        Args:
            video_shape: [B, F, C, H, W]
            audio_shape: [B, F_a, C]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings
            clean_video: Clean video latent (optional, for non-backward-simulation)
            clean_audio: Clean audio latent (optional)

        Returns:
            Tuple of (loss, log_dict)
        """
        # Run generator
        pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log = self._run_generator(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            clean_video=clean_video,
            clean_audio=clean_audio,
        )

        # Compute DMD loss
        dmd_loss, log_dict = self.compute_distribution_matching_loss(
            video_latent=pred_video,
            audio_latent=pred_audio,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )
        log_dict.update(rollout_log)

        return dmd_loss, log_dict

    def critic_loss(
        self,
        video_shape: List[int],
        audio_shape: List[int],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_video: Optional[torch.Tensor] = None,
        clean_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute critic (fake_score) loss.

        The critic learns to denoise generated samples.
        """
        # Step 1: Generate samples (no gradient)
        with torch.no_grad():
            generated_video, generated_audio, video_loss_mask, audio_loss_mask, _ = self._run_generator(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                clean_video=clean_video,
                clean_audio=clean_audio,
            )

        B = generated_video.shape[0]
        F_v = generated_video.shape[1]
        F_a = generated_audio.shape[1]

        # Step 2: Sample critic timestep
        if video_loss_mask is not None and audio_loss_mask is not None:
            critic_timestep, audio_critic_timestep = self._sample_causal_supervision_timesteps(
                B,
                video_loss_mask,
                audio_loss_mask,
            )
            critic_sigma = self.timestep_to_sigma(critic_timestep)
            audio_critic_sigma = self.timestep_to_sigma(audio_critic_timestep)
        else:
            critic_timestep = torch.randint(
                0, self.num_train_timestep,
                [B, F_v],
                device=self.device,
                dtype=torch.long,
            )
            critic_timestep = self._process_timestep(critic_timestep, self.fake_task_type)
            critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

            critic_sigma = self.timestep_to_sigma(critic_timestep)
            if self._is_bidirectional_task(self.fake_task_type):
                audio_critic_timestep = critic_timestep[:, 0]
                audio_critic_sigma = critic_sigma[:, 0:1].expand(B, F_a)
            else:
                audio_critic_timestep = self._compute_audio_timestep(
                    critic_timestep, F_a, task_type=self.fake_task_type
                ).clamp(self.min_step, self.max_step)
                audio_critic_sigma = self.timestep_to_sigma(audio_critic_timestep)

        # Step 3: Add noise to generated samples
        noise_video = torch.randn_like(generated_video)
        noise_audio = torch.randn_like(generated_audio)

        noisy_generated_video = self.add_noise(
            generated_video.flatten(0, 1),
            noise_video.flatten(0, 1),
            critic_sigma.flatten(0, 1),
        ).unflatten(0, (B, F_v))

        noisy_generated_audio = self.add_noise(
            generated_audio, noise_audio, audio_critic_sigma
        )

        # Step 4: Critic prediction
        pred_video, pred_audio = self.fake_score(
            noisy_image_or_video=noisy_generated_video,
            conditional_dict=conditional_dict,
            timestep=critic_sigma,
            noisy_audio=noisy_generated_audio,
            audio_timestep=audio_critic_sigma,
        )

        # Step 5: Compute flow matching loss for critic
        # CausVid uses flow_pred = (xt - x0_pred) / sigma, NOT simple x0 MSE.
        # The 1/sigma factor gives implicit 1/sigma^2 gradient weighting,
        # making the critic accurate at low-noise timesteps (critical for DMD).
        # Float64 for numerical stability, then cast back (matches CausVid).
        video_sigma_4d = critic_sigma.flatten(0, 1).double().reshape(-1, 1, 1, 1).clamp_min(1e-8)
        flow_pred_video = (
            (noisy_generated_video.flatten(0, 1).double() - pred_video.flatten(0, 1).double())
            / video_sigma_4d
        ).to(self.dtype)

        audio_sigma_2d = audio_critic_sigma.double().unsqueeze(-1).clamp_min(1e-8)
        flow_pred_audio = (
            (noisy_generated_audio.double() - pred_audio.double())
            / audio_sigma_2d
        ).to(self.dtype)

        # flow_true = noise - x0 (target flow)
        video_loss = self._compute_masked_denoising_loss(
            target=generated_video,
            prediction=pred_video,
            noise=noise_video,
            flow_pred=flow_pred_video.unflatten(0, (B, F_v)),
            timestep=critic_timestep,
            mask=video_loss_mask,
        )

        audio_loss = self._compute_masked_denoising_loss(
            target=generated_audio,
            prediction=pred_audio,
            noise=noise_audio,
            flow_pred=flow_pred_audio,
            timestep=audio_critic_timestep,
            mask=audio_loss_mask,
        )

        video_w, audio_w = self.get_loss_weights()
        total_loss = video_w * video_loss + audio_w * audio_loss

        log_dict = {
            "critic_video_loss": video_loss.item(),
            "critic_audio_loss": audio_loss.item(),
        }

        return total_loss, log_dict
