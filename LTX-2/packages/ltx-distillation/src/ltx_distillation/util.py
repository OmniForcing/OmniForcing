"""
Utility functions for DMD distillation training.

Includes:
- FSDP wrapping and state dict handling
- Distributed training utilities
- Logging and checkpointing helpers
"""

import os
import random
from typing import Optional, Tuple, Any
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def launch_distributed_job() -> None:
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    return rank, world_size, local_rank


def barrier() -> None:
    """Synchronization barrier for distributed training."""
    if dist.is_initialized():
        dist.barrier()


def get_sharding_strategy(strategy_name: str) -> ShardingStrategy:
    """Get FSDP sharding strategy by name."""
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_grad_op": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    return strategy_map.get(strategy_name, ShardingStrategy.HYBRID_SHARD)


def fsdp_wrap(
    module: nn.Module,
    sharding_strategy: str = "hybrid_full",
    mixed_precision: bool = True,
    wrap_strategy: str = "size",
    transformer_module: Optional[Tuple[type, ...]] = None,
    min_num_params: int = 1e8,
    cpu_offload: bool = False,
) -> FSDP:
    """
    Wrap module with FSDP for distributed training.

    Args:
        module: Module to wrap
        sharding_strategy: Sharding strategy name
        mixed_precision: Use bfloat16 mixed precision
        wrap_strategy: "size" or "transformer"
        transformer_module: Transformer block classes for transformer wrapping
        min_num_params: Minimum parameters for size-based wrapping
        cpu_offload: Enable CPU offload

    Returns:
        FSDP-wrapped module
    """
    # Mixed precision policy
    # Match CausVid: param in bfloat16, but reduce/buffer in float32
    # for gradient all-reduce precision and buffer accuracy.
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mp_policy = None

    # Wrap policy
    if wrap_strategy == "transformer" and transformer_module is not None:
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module,
        )
    else:
        wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(min_num_params),
        )

    # CPU offload
    offload_policy = CPUOffload(offload_params=True) if cpu_offload else None

    # Wrap with FSDP
    wrapped = FSDP(
        module,
        sharding_strategy=get_sharding_strategy(sharding_strategy),
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        cpu_offload=offload_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    return wrapped


def fsdp_state_dict(module: FSDP) -> dict:
    """
    Get full state dict from FSDP module.

    Gathers sharded parameters to rank 0.
    """
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = module.state_dict()

    return state_dict


def init_logging_folder(config) -> Tuple[str, str]:
    """
    Initialize output and wandb folders.

    The run directory name follows the pattern: ``{MMDD}_{HHMMSS}_{wandb_name}``
    and the WandB run name is set to the same directory name so they stay in sync.

    A copy of the full config is saved as ``config.yaml`` inside the run directory.

    Args:
        config: Configuration object with output_path and wandb settings

    Returns:
        Tuple of (output_path, wandb_folder)
    """
    import wandb
    from omegaconf import OmegaConf

    # Create output directory – naming: {MMDD}_{HHMMSS}_{wandb_name}
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{config.wandb_name}"
    output_path = os.path.join(config.output_path, run_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # Save a copy of the config for reproducibility
    OmegaConf.save(config, os.path.join(output_path, "config.yaml"))

    # Initialize wandb
    wandb_folder = os.path.join(output_path, "wandb")
    os.makedirs(wandb_folder, exist_ok=True)

    # Set wandb API key from config (required for multi-node without shared ~/.netrc)
    wandb_api_key = getattr(config, "wandb_api_key", "")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # Get wandb entity (None means use default logged-in account)
    wandb_entity = getattr(config, "wandb_entity", None)
    if wandb_entity == "null" or wandb_entity == "":
        wandb_entity = None

    wandb_kwargs = dict(
        project=config.wandb_project,
        entity=wandb_entity,
        name=run_dir_name,
        config=dict(config),
        dir=wandb_folder,
    )
    try:
        wandb.init(**wandb_kwargs)
    except Exception as exc:
        # If rank 0 dies here, other ranks only report a later NCCL/TCPStore
        # failure during output_path broadcast. Fall back to disabled WandB so
        # training can proceed and the root cause remains visible on rank 0.
        print(
            f"[WandB] init failed ({type(exc).__name__}: {exc}). "
            "Falling back to disabled WandB mode."
        )
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled", **wandb_kwargs)

    return output_path, wandb_folder


def prepare_for_saving(tensor: torch.Tensor, max_frames: int = 16) -> Any:
    """
    Prepare tensor for wandb logging/saving.

    Args:
        tensor: Video tensor [B, C, F, H, W] or [B, F, C, H, W]
        max_frames: Maximum frames to save

    Returns:
        Wandb-compatible video object
    """
    import wandb

    # Ensure correct format [B, F, C, H, W]
    if tensor.dim() == 5:
        if tensor.shape[1] == 3:
            # [B, C, F, H, W] -> [B, F, C, H, W]
            tensor = tensor.permute(0, 2, 1, 3, 4)

    # Take first sample and limit frames
    video = tensor[0, :max_frames].cpu()

    # Normalize to [0, 255] uint8
    video = (video.clamp(0, 1) * 255).to(torch.uint8)

    # Convert to [F, H, W, C] for wandb
    video = video.permute(0, 2, 3, 1).numpy()

    return wandb.Video(video, fps=8, format="mp4")


def cycle(dataloader):
    """Infinite dataloader iterator."""
    while True:
        for batch in dataloader:
            yield batch


class AverageMeter:
    """Compute and store running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
