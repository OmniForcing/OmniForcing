"""
LTX-2 DMD Distillation Package

This package implements Distribution Matching Distillation (DMD) for
LTX-2 audio-video generation, based on the algorithm from the CausVid paper
(https://arxiv.org/abs/2412.07772) with LTX-2-specific implementation.

Key Modules:
- dmd: DMD distillation module for bidirectional models
- ode: ODE initialization module for causal models
- models: Model wrappers for LTX-2
- inference: Inference pipelines

Training Pipeline:
1. Stage 0: Bidirectional DMD (dmd.py, train_distillation.py)
2. Stage 1: ODE Init (ode/train_ode.py)
3. Stage 2: Self-Forcing DMD (to be implemented in ltx-self-forcing)
"""

from ltx_distillation.dmd import LTX2DMD
from ltx_distillation.loss import get_denoising_loss
from ltx_distillation.ode import (
    LTX2ODEPairGenerator,
    ODEGenerationConfig,
    LTX2ODERegression,
    ODERegressionConfig,
    ODERegressionLMDBDataset,
    ODERegressionDataset,
)

__version__ = "0.1.0"

__all__ = [
    # DMD
    "LTX2DMD",
    "get_denoising_loss",
    # ODE Init
    "LTX2ODEPairGenerator",
    "ODEGenerationConfig",
    "LTX2ODERegression",
    "ODERegressionConfig",
    "ODERegressionLMDBDataset",
    "ODERegressionDataset",
]
