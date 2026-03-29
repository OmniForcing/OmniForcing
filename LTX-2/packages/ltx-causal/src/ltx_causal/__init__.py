"""
LTX-Causal: Causal LTX-2 model for ODE causal training via masking.

This package implements causal attention mechanisms for
converting the bidirectional LTX-2 model into a causal training model.

Key Components:
- CausalLTXModel: Full causal transformer with 48 layers
- CausalLTX2DiffusionWrapper: DMD-compatible diffusion wrapper
- AVCausalMaskBuilder: Causal mask construction for audio-video
"""

from ltx_causal.config import (
    CausalGenerationConfig,
    CausalMaskConfig,
    VIDEO_LATENT_FPS,
    AUDIO_LATENT_FPS,
)
from ltx_causal.attention.mask_builder import (
    AVCausalMaskBuilder,
    compute_av_blocks,
    build_all_causal_masks,
)
from ltx_causal.transformer.causal_model import (
    CausalLTXModel,
    CausalLTXModelConfig,
)
from ltx_causal.wrapper import (
    CausalLTX2DiffusionWrapper,
    get_causal_diffusion_wrapper,
)

__version__ = "0.2.0"

__all__ = [
    # Config
    "CausalGenerationConfig",
    "CausalMaskConfig",
    "CausalLTXModelConfig",
    # Constants
    "VIDEO_LATENT_FPS",
    "AUDIO_LATENT_FPS",
    # Mask building
    "AVCausalMaskBuilder",
    "compute_av_blocks",
    "build_all_causal_masks",
    # Model
    "CausalLTXModel",
    # Wrapper
    "CausalLTX2DiffusionWrapper",
    "get_causal_diffusion_wrapper",
]
