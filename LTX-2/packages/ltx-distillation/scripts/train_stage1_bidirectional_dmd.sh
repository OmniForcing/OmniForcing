#!/bin/bash
# =============================================================================
# Stage 1: Bidirectional DMD Distillation
# =============================================================================
# Distills the LTX-2 bidirectional teacher model from 1000-step to 4-step
# fast inference. This produces the bidirectional DMD model used as teacher
# and critic in later stages.
#
# Usage:
#   ./scripts/train_stage1_bidirectional_dmd.sh [config_path] [extra args...]
#
# Environment variables:
#   NUM_GPUS          - GPUs per node (default: 8)
#   NNODES            - Number of nodes (default: 1, auto-detected from SLURM)
#   NODE_RANK         - Current node rank (auto-detected from SLURM/env)
#   MASTER_ADDR       - Master node address (default: localhost)
#   MASTER_PORT       - Master node port (default: 29500)
#   VENV_PATH         - Python venv path (optional)
#   NUM_CPUS          - CPUs per node for OMP threading (default: 128)
#
# Recommended: 32 GPUs (4 nodes x 8 GPUs), or 8 GPUs with gradient_accumulation_steps=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LTX2_ROOT="$(cd "${DISTILLATION_ROOT}/../.." && pwd)"

cd "${DISTILLATION_ROOT}"
echo "Working dir: $(pwd)"

# Virtual Environment (optional)
if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated venv: ${VENV_PATH}"
fi

# Config
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    CONFIG_PATH="$1"
    shift
else
    CONFIG_PATH="configs/stage1_bidirectional_dmd.yaml"
fi

export PYTHONPATH="${DISTILLATION_ROOT}/src:${LTX2_ROOT}/packages/ltx-causal/src:${LTX2_ROOT}/packages/ltx-core/src:${LTX2_ROOT}/packages/ltx-pipelines/src${PYTHONPATH:+:${PYTHONPATH}}"

# Distributed settings (auto-detect SLURM/scheduler env)
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS:-${LOCAL_WORLD_SIZE:-8}}}"
if [ -z "${NNODES:-}" ]; then
    if [ -n "${SLURM_NNODES:-}" ]; then
        NNODES="${SLURM_NNODES}"
    elif [ -n "${GROUP_WORLD_SIZE:-}" ]; then
        NNODES="${GROUP_WORLD_SIZE}"
    elif [ -n "${WORLD_SIZE:-}" ] && [ -n "${LOCAL_WORLD_SIZE:-}" ] && [ "${LOCAL_WORLD_SIZE}" -gt 0 ] && [ $((WORLD_SIZE % LOCAL_WORLD_SIZE)) -eq 0 ]; then
        NNODES="$((WORLD_SIZE / LOCAL_WORLD_SIZE))"
    else
        NNODES=1
    fi
fi

if [ -z "${NODE_RANK:-}" ]; then
    if [ -n "${SLURM_NODEID:-}" ]; then
        NODE_RANK="${SLURM_NODEID}"
    elif [ -n "${GROUP_RANK:-}" ]; then
        NODE_RANK="${GROUP_RANK}"
    elif [ -n "${RANK:-}" ] && [ -n "${LOCAL_WORLD_SIZE:-}" ] && [ "${LOCAL_WORLD_SIZE}" -gt 0 ]; then
        NODE_RANK="$((RANK / LOCAL_WORLD_SIZE))"
    else
        NODE_RANK=0
    fi
fi

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

NUM_CPUS="${NUM_CPUS:-128}"
export OMP_NUM_THREADS=$((NUM_CPUS / NPROC_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOTAL_GPUS=$((NPROC_PER_NODE * NNODES))

echo "========================================================"
echo "Stage 1: Bidirectional DMD Distillation"
echo "========================================================"
echo "Config:        $CONFIG_PATH"
echo "Nodes:         $NNODES  |  GPUs/Node: $NPROC_PER_NODE  |  Total: $TOTAL_GPUS"
echo "Master:        $MASTER_ADDR:$MASTER_PORT"
echo "========================================================"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m ltx_distillation.train_distillation \
    --config_path "$CONFIG_PATH" \
    "$@"
