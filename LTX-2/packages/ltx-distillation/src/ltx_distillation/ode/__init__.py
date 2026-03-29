"""
ODE Init module for causal LTX-2 training.

This module implements the ODE initialization stage (Stage 1) from CausVid:
1. Generate ODE trajectories using bidirectional teacher model
2. Convert trajectories to LMDB format
3. Train causal model with ODE regression

Key Components:
- LTX2ODEPairGenerator: Generates ODE trajectories from teacher
- ODERegressionLMDBDataset: LMDB dataset for ODE pairs
- ODERegressionDataset: In-memory dataset for small experiments
- LTX2ODERegression: Training module for ODE regression
- ODERegressionConfig: Configuration for ODE regression

Scripts:
- generate_ode_pairs.py: Generate ODE trajectories from teacher
- create_lmdb.py: Convert .pt files to LMDB format
- train_ode.py: Train causal model with ODE regression
"""

from ltx_distillation.ode.generate_ode_pairs import (
    LTX2ODEPairGenerator,
    ODEGenerationConfig,
)
from ltx_distillation.ode.data import (
    ODERegressionLMDBDataset,
    ODERegressionDataset,
    TextDataset,
    collate_ode_batch,
)
from ltx_distillation.ode.ode_regression import (
    LTX2ODERegression,
    ODERegressionConfig,
)
from ltx_distillation.ode.create_lmdb import (
    create_lmdb_from_trajectories,
)

__all__ = [
    # ODE Generation
    "LTX2ODEPairGenerator",
    "ODEGenerationConfig",
    # Datasets
    "ODERegressionLMDBDataset",
    "ODERegressionDataset",
    "TextDataset",
    "collate_ode_batch",
    # Training
    "LTX2ODERegression",
    "ODERegressionConfig",
    # LMDB
    "create_lmdb_from_trajectories",
]
