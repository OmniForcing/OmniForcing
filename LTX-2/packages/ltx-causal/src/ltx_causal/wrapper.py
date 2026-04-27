"""
CausalLTX2DiffusionWrapper: Diffusion wrapper for causal LTX-2.

This module provides:
- Interface compatible with ltx-distillation's LTX2DiffusionWrapper
- Support for ODE causal training via masking
- Returns x0 predictions (matching original X0Model behavior)

The wrapper bridges CausalLTXModel with the DMD training framework.
"""

from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from ltx_causal.config import (
    VIDEO_LATENT_FPS,
    AUDIO_LATENT_FPS,
    CausalMaskConfig,
)
from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
from ltx_causal.attention.mask_builder import (
    build_all_causal_masks,
    compute_aligned_audio_frames,
)
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model


class CausalLTX2DiffusionWrapper(nn.Module):
    """
    Wrapper for CausalLTXModel providing DMD-compatible interface.

    Returns x0 predictions (denoised latents), matching the behavior of
    the original LTX2DiffusionWrapper which uses X0Model internally.

    The conversion is: x0 = sample - velocity * sigma

    Example:
        wrapper = CausalLTX2DiffusionWrapper(model)
        video_x0, audio_x0 = wrapper(
            video_latent, audio_latent, timestep, conditional_dict
        )
    """

    # Time alignment constants (same as ltx-distillation)
    VIDEO_LATENT_FPS = VIDEO_LATENT_FPS
    AUDIO_LATENT_FPS = AUDIO_LATENT_FPS
    ALIGNMENT_RATIO = AUDIO_LATENT_FPS / VIDEO_LATENT_FPS

    # Latent dimensions
    VIDEO_FRAME_SEQLEN = 384  # For 512x768 with patch_size=1: (512/32)*(768/32)
    AUDIO_FRAME_SEQLEN = 1

    def __init__(
        self,
        model: CausalLTXModel,
        video_height: int = 512,
        video_width: int = 768,
        vae_spatial_compression: int = 32,
        num_frame_per_block: int = 3,
        num_frame_per_block_first: int = 4,
        disable_causal_mask: bool = False,
    ):
        """
        Args:
            model: CausalLTXModel instance
            video_height: Video height in pixels
            video_width: Video width in pixels
            vae_spatial_compression: VAE spatial compression factor
            num_frame_per_block: Frames per causal generation block (>=1)
            num_frame_per_block_first: Frames in Block 0 (4-3-3-3-... layout)
            disable_causal_mask: If True, disable all causal masks (fully bidirectional,
                equivalent to original LTX-2). Used for ablation studies.
        """
        super().__init__()
        self.model = model
        self.video_height = video_height
        self.video_width = video_width
        self.vae_spatial_compression = vae_spatial_compression
        self.num_frame_per_block = num_frame_per_block
        self.num_frame_per_block_first = num_frame_per_block_first
        self.disable_causal_mask = disable_causal_mask

        # Validate patch_size compatibility
        if hasattr(model, 'config') and hasattr(model.config, 'patch_size'):
            ps = model.config.patch_size
            if ps != (1, 1, 1):
                raise ValueError(
                    f"CausalLTX2DiffusionWrapper requires patch_size=(1,1,1), "
                    f"but model has patch_size={ps}. The causal masks assume "
                    f"video_frame_seqlen = (H/32)*(W/32) tokens per frame."
                )

        # Compute latent dimensions
        self.latent_height = video_height // vae_spatial_compression
        self.latent_width = video_width // vae_spatial_compression

        # Compute frame sequence length
        # LTX-2 uses patch_size=1 (no spatial grouping), so each spatial
        # position in the latent grid is one token.
        # For 512x768: (512/32) * (768/32) = 16 * 24 = 384
        self.video_frame_seqlen = self.latent_height * self.latent_width

        # Mask config with actual resolution parameters (not defaults)
        self.mask_config = CausalMaskConfig(
            video_frame_seqlen=self.video_frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            num_frame_per_block_first=num_frame_per_block_first,
        )

        # Cache masks per (shape, device, sink-token config) so training and
        # benchmark block shapes can reuse prebuilt masks instead of rebuilding.
        self._mask_cache: Dict[Tuple, Dict[str, Optional[torch.Tensor]]] = {}
        # Keep the eval-only delegate out of nn.Module registration so FSDP/state_dict
        # do not treat it as part of the trainable causal model.
        self.__dict__["_bidirectional_delegate"] = None

    def set_bidirectional_delegate(self, delegate: nn.Module) -> None:
        """Install an explicit bidirectional delegate for eval-time fallback."""
        self.__dict__["_bidirectional_delegate"] = delegate

    def has_bidirectional_delegate(self) -> bool:
        """Return whether a delegate was explicitly installed."""
        return self._bidirectional_delegate is not None

    def set_module_grad(self, module_grad: Dict[str, bool]) -> None:
        """Set gradient requirements for model components."""
        if module_grad.get("model", True):
            self.model.requires_grad_(True)
        else:
            self.model.requires_grad_(False)
            self.model.eval()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.model.enable_gradient_checkpointing()

    def _build_bidirectional_delegate(self) -> nn.Module:
        """Build an eval-only wrapper that uses the original bidirectional path."""
        from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper

        model_config = self.model.config
        velocity_model = LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=model_config.video_heads,
            attention_head_dim=model_config.video_d_head,
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            num_layers=model_config.num_layers,
            cross_attention_dim=model_config.cross_attention_dim,
            norm_eps=model_config.norm_eps,
            caption_channels=model_config.caption_channels,
            positional_embedding_theta=model_config.pe_theta,
            positional_embedding_max_pos=list(model_config.pe_max_pos),
            timestep_scale_multiplier=model_config.timestep_scale_multiplier,
            audio_num_attention_heads=model_config.audio_heads,
            audio_attention_head_dim=model_config.audio_d_head,
            audio_in_channels=model_config.in_channels,
            audio_out_channels=model_config.out_channels,
            audio_cross_attention_dim=model_config.audio_cross_attention_dim,
            audio_positional_embedding_max_pos=list(model_config.audio_pe_max_pos),
            av_ca_timestep_scale_multiplier=model_config.av_ca_timestep_scale_multiplier,
        ).to(device=next(self.model.parameters()).device, dtype=next(self.model.parameters()).dtype)

        state_dict = {
            key: value
            for key, value in self.model.state_dict().items()
            if "mask_builder" not in key and "causal_gate" not in key and "audio_sink_tokens" not in key
        }
        missing, unexpected = velocity_model.load_state_dict(state_dict, strict=False)
        real_missing = [key for key in missing if "mask_builder" not in key]
        if real_missing or unexpected:
            raise RuntimeError(
                "Failed to build bidirectional delegate from causal model: "
                f"missing={real_missing[:10]}, unexpected={unexpected[:10]}"
            )

        delegate = LTX2DiffusionWrapper(
            model=X0Model(velocity_model),
            video_height=self.video_height,
            video_width=self.video_width,
            vae_spatial_compression=self.vae_spatial_compression,
        )
        delegate.eval()
        return delegate

    def _get_bidirectional_delegate(self) -> nn.Module:
        if self._bidirectional_delegate is None:
            if dist.is_initialized():
                raise RuntimeError(
                    "Bidirectional delegate was not pre-installed before distributed/FSDP wrapping. "
                    "Lazy delegate construction calls self.model.state_dict(), which is unsafe under "
                    "FSDP and can leave unmanaged all-gathered parameter storage alive. Install the "
                    "delegate via set_bidirectional_delegate() before wrapping the causal generator."
                )
            self._bidirectional_delegate = self._build_bidirectional_delegate()
        return self._bidirectional_delegate

    def get_scheduler(self):
        """Get the scheduler for this model."""
        try:
            from ltx_distillation.schedulers import LTX2FlowScheduler
            return LTX2FlowScheduler()
        except ImportError:
            raise ImportError(
                "LTX2FlowScheduler not found. "
                "Please install ltx-distillation package."
            )

    def _process_timestep_causal(
        self,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process timestep for causal generation (4-3-3-3-... layout).

        Makes timesteps uniform within each generation block:
            Block 0: frames [0, num_frame_per_block_first)
            Block k>=1: frames [num_frame_per_block_first + (k-1)*num_frame_per_block,
                                num_frame_per_block_first + k*num_frame_per_block)
        """
        if timestep.ndim == 1:
            return timestep

        B, F = timestep.shape
        if F <= 1:
            return timestep

        result = timestep.new_zeros(B, F)

        first = self.num_frame_per_block_first
        end0 = min(first, F)
        result[:, :end0] = timestep[:, :1].expand(B, end0)

        block_size = self.num_frame_per_block
        idx = end0
        while idx < F:
            end = min(idx + block_size, F)
            result[:, idx:end] = timestep[:, idx:idx + 1].expand(B, end - idx)
            idx = end

        return result

    def _get_or_build_masks(
        self,
        num_video_frames: int,
        num_audio_frames: int,
        device: torch.device,
    ) -> Dict:
        """Get cached masks or build new ones.

        When disable_causal_mask=True, returns all-None masks so attention
        falls through to standard bidirectional path (no mask), matching
        original LTX-2 behavior exactly.
        """
        if self.disable_causal_mask:
            return {'video_self': None, 'audio_self': None, 'a2v': None, 'v2a': None}

        cache_key = (
            num_video_frames,
            num_audio_frames,
            str(device),
            self.mask_config.num_frame_per_block,
            self.mask_config.num_frame_per_block_first,
            self.mask_config.video_frame_seqlen,
        )

        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = build_all_causal_masks(
                num_video_frames,
                num_audio_frames,
                config=self.mask_config,
                device=device,
            )
        return self._mask_cache[cache_key]

    @staticmethod
    def _reshape_sigma_for_broadcast(
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Reshape sigma tensor for broadcasting with target.

        Args:
            sigma: [B] or [B, F] sigma values
            target: Target tensor to broadcast against
                    Video: [B, F, C, H, W] or Audio: [B, F, C]

        Returns:
            Sigma reshaped for broadcasting
        """
        if sigma.ndim == 1:
            # [B] → add trailing dims to match target
            return sigma.view(-1, *([1] * (target.ndim - 1)))
        elif sigma.ndim == 2:
            # [B, F] → add trailing dims
            return sigma.view(sigma.shape[0], sigma.shape[1], *([1] * (target.ndim - 2)))
        else:
            return sigma

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        use_causal_timestep: bool = True,
        force_bidirectional: bool = False,
        # KV-cache autoregressive path (Self-Forcing). When kv_caches is
        # provided the wrapper dispatches to the per-block KV-cache forward
        # so that FSDP parameter gather works correctly via __call__.
        kv_caches=None,
        current_video_start_frame: int = 0,
        current_audio_start_frame: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Returns x0 predictions (denoised latents) matching the original
        X0Model wrapper behavior: x0 = sample - velocity * sigma.

        Args:
            noisy_image_or_video: Noisy video latent [B, F, C, H, W]
            conditional_dict: Text conditioning dictionary with:
                - video_context: [B, L, D] video text embeddings
                - audio_context: [B, L, D] audio text embeddings
                - video_context_mask: Optional attention mask
                - audio_context_mask: Optional attention mask
            timestep: Diffusion timestep (sigma) [B] or [B, F] (video timestep)
            noisy_audio: Noisy audio latent [B, F_a, C] (optional)
            audio_timestep: Audio diffusion timestep (sigma) [B] or [B, F_a] (optional)
                If not provided, uses video timestep expanded to audio frames
            use_causal_timestep: Whether to apply causal timestep processing

        Returns:
            (video_x0, audio_x0): Denoised x0 predictions.
        """
        # === KV-cache autoregressive path (Self-Forcing) ===
        if kv_caches is not None:
            return self._forward_with_cache_impl(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                noisy_audio=noisy_audio,
                audio_timestep=audio_timestep,
                kv_caches=kv_caches,
                current_video_start_frame=current_video_start_frame,
                current_audio_start_frame=current_audio_start_frame,
            )

        B, F_v, C, H, W = noisy_image_or_video.shape
        device = noisy_image_or_video.device

        if (
            not self.training
            and (self.disable_causal_mask or force_bidirectional)
            and not dist.is_initialized()  # unsafe under FSDP: see _get_bootstrap_generator note
        ):
            delegate = self._get_bidirectional_delegate()
            delegate = delegate.to(device=device, dtype=noisy_image_or_video.dtype)
            return delegate(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                noisy_audio=noisy_audio,
                audio_timestep=audio_timestep,
            )

        # Compute aligned audio frame count for the 4-3-3-3-... block layout.
        aligned_audio = compute_aligned_audio_frames(
            F_v,
            num_frame_per_block=self.num_frame_per_block,
            num_frame_per_block_first=self.num_frame_per_block_first,
        )

        # Compute audio frames if not provided
        if noisy_audio is None:
            # Create audio with aligned frame count
            F_a = aligned_audio
            noisy_audio = torch.randn(B, F_a, C, device=device, dtype=noisy_image_or_video.dtype)
        else:
            F_a = noisy_audio.shape[1]

        # Validate audio frame count matches aligned count
        if F_a != aligned_audio:
            raise ValueError(
                f"Audio frames ({F_a}) does not match aligned count ({aligned_audio}) "
                f"for {F_v} video frames. With V_0 + A_0 in Block 0, the aligned "
                f"count should match ltx-core's raw audio count exactly."
            )

        # Process timestep
        if use_causal_timestep and timestep.ndim > 1:
            timestep = self._process_timestep_causal(timestep)

        # Extract context from conditional_dict
        video_context = conditional_dict.get('video_context')
        audio_context = conditional_dict.get('audio_context')
        video_context_mask = conditional_dict.get('video_context_mask')
        audio_context_mask = conditional_dict.get('audio_context_mask')

        # Handle shared context
        if video_context is None and 'context' in conditional_dict:
            video_context = conditional_dict['context']
        if audio_context is None:
            audio_context = video_context

        # Handle shared attention mask fallback
        shared_mask = conditional_dict.get('attention_mask')
        if video_context_mask is None and shared_mask is not None:
            video_context_mask = shared_mask
        if audio_context_mask is None and shared_mask is not None:
            audio_context_mask = shared_mask

        # Build or get cached masks
        masks = self._get_or_build_masks(F_v, F_a, device)

        # Call model (returns velocity predictions)
        video_velocity, audio_velocity = self.model(
            video_latent=noisy_image_or_video,
            audio_latent=noisy_audio,
            timesteps=timestep,
            video_context=video_context,
            audio_context=audio_context,
            video_context_mask=video_context_mask,
            audio_context_mask=audio_context_mask,
            audio_timesteps=audio_timestep,
            masks=masks,
        )

        # === Convert velocity to x0 predictions ===
        # Matches X0Model / to_denoised: x0 = sample - velocity * sigma
        # Use float32 for numerical stability (matching to_denoised in ltx-core)
        compute_dtype = torch.float32

        # Video: x0 = noisy_video - velocity * sigma
        video_sigma = self._reshape_sigma_for_broadcast(timestep, noisy_image_or_video)
        video_x0 = (
            noisy_image_or_video.to(compute_dtype)
            - video_velocity.to(compute_dtype) * video_sigma.to(compute_dtype)
        ).to(noisy_image_or_video.dtype)

        # Audio: x0 = noisy_audio - velocity * sigma
        audio_sigma_raw = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = self._reshape_sigma_for_broadcast(audio_sigma_raw, noisy_audio)
        audio_x0 = (
            noisy_audio.to(compute_dtype)
            - audio_velocity.to(compute_dtype) * audio_sigma.to(compute_dtype)
        ).to(noisy_audio.dtype)

        return video_x0, audio_x0

    # ------------------------------------------------------------------
    # KV-cache causal forward (Self-Forcing autoregressive path)
    # ------------------------------------------------------------------

    def init_av_kv_caches(
        self,
        batch_size: int,
        max_video_frames: int,
        max_audio_frames: int,
        text_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Allocate per-layer AV KV caches."""
        return self.model.init_av_kv_caches(
            batch_size=batch_size,
            max_video_frames=max_video_frames,
            max_audio_frames=max_audio_frames,
            text_seq_len=text_seq_len,
            device=device,
            dtype=dtype,
        )

    def _forward_with_cache_impl(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        kv_caches,
        current_video_start_frame: int,
        current_audio_start_frame: int,
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-block KV-cache forward; returns x0 predictions.

        timestep is expected to be block-uniform (same sigma across all
        frames of the block). Velocity-to-x0 conversion mirrors `forward`.
        Callers under FSDP MUST go through `__call__` (i.e. dispatch via
        `forward(... kv_caches=...)`) so that pre/post-forward hooks run.
        """
        video_context = conditional_dict.get('video_context')
        audio_context = conditional_dict.get('audio_context')
        video_context_mask = conditional_dict.get('video_context_mask')
        audio_context_mask = conditional_dict.get('audio_context_mask')
        if video_context is None and 'context' in conditional_dict:
            video_context = conditional_dict['context']
        if audio_context is None:
            audio_context = video_context
        shared_mask = conditional_dict.get('attention_mask')
        if video_context_mask is None and shared_mask is not None:
            video_context_mask = shared_mask
        if audio_context_mask is None and shared_mask is not None:
            audio_context_mask = shared_mask

        # Call through self.model(...) (not .forward_with_cache) so that if
        # self.model is FSDP-wrapped, the FSDP pre/post-forward hooks fire
        # and unshard parameters. CausalLTXModel.forward() dispatches to
        # forward_with_cache internally when kv_caches is not None.
        video_velocity, audio_velocity = self.model(
            video_latent=noisy_image_or_video,
            audio_latent=noisy_audio,
            timesteps=timestep,
            audio_timesteps=audio_timestep,
            video_context=video_context,
            audio_context=audio_context,
            video_context_mask=video_context_mask,
            audio_context_mask=audio_context_mask,
            kv_caches=kv_caches,
            current_video_start_frame=current_video_start_frame,
            current_audio_start_frame=current_audio_start_frame,
        )

        compute_dtype = torch.float32
        video_sigma = self._reshape_sigma_for_broadcast(timestep, noisy_image_or_video)
        video_x0 = (
            noisy_image_or_video.to(compute_dtype)
            - video_velocity.to(compute_dtype) * video_sigma.to(compute_dtype)
        ).to(noisy_image_or_video.dtype)

        if noisy_audio is None:
            return video_x0, None

        audio_sigma_raw = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = self._reshape_sigma_for_broadcast(audio_sigma_raw, noisy_audio)
        audio_x0 = (
            noisy_audio.to(compute_dtype)
            - audio_velocity.to(compute_dtype) * audio_sigma.to(compute_dtype)
        ).to(noisy_audio.dtype)
        return video_x0, audio_x0

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        video_height: int = 512,
        video_width: int = 768,
        num_frame_per_block: int = 3,
        num_frame_per_block_first: int = 4,
        disable_causal_mask: bool = False,
        enable_causal_log_rescale: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "CausalLTX2DiffusionWrapper":
        """Load wrapper from pretrained checkpoint."""
        config = CausalLTXModelConfig(
            num_frame_per_block=num_frame_per_block,
            num_frame_per_block_first=num_frame_per_block_first,
            enable_causal_log_rescale=enable_causal_log_rescale,
        )
        model = CausalLTXModel.from_pretrained(
            checkpoint_path, config, device, dtype
        )
        return cls(
            model=model,
            video_height=video_height,
            video_width=video_width,
            num_frame_per_block=num_frame_per_block,
            num_frame_per_block_first=num_frame_per_block_first,
            disable_causal_mask=disable_causal_mask,
        )


# ============================================================================
# Factory Functions
# ============================================================================

def get_causal_diffusion_wrapper(
    checkpoint_path: Optional[str] = None,
    model: Optional[CausalLTXModel] = None,
    **kwargs,
) -> CausalLTX2DiffusionWrapper:
    """
    Factory function to create CausalLTX2DiffusionWrapper.

    Args:
        checkpoint_path: Path to checkpoint (if model not provided)
        model: Pre-loaded CausalLTXModel (if checkpoint not provided)
        **kwargs: Additional arguments for wrapper

    Returns:
        Configured wrapper instance
    """
    if model is not None:
        return CausalLTX2DiffusionWrapper(model=model, **kwargs)
    elif checkpoint_path is not None:
        return CausalLTX2DiffusionWrapper.from_pretrained(checkpoint_path, **kwargs)
    else:
        raise ValueError("Either checkpoint_path or model must be provided")
