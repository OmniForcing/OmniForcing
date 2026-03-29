#!/bin/bash
# =============================================================================
# Stage 3: Causal DMD Training
# =============================================================================
# Trains the causal autoregressive model with DMD loss using the ODE-initialized
# generator and bidirectional teacher/critic.
#
# Two variants available:
#   - Causal DMD (this script, main branch): block-wise DMD training
#   - Self-Forcing DMD (self-forcing branch): autoregressive self-forcing rollout
#
# Prerequisites:
#   - Stage 1 bidirectional DMD checkpoint (teacher/critic)
#   - Stage 2 ODE-initialized causal checkpoint (generator)
#   - ODE LMDB dataset
#
# Usage:
#   ./scripts/train_stage3_causal_dmd.sh [config_path] [extra args...]
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
    CONFIG_PATH="configs/stage3_causal_dmd.yaml"
fi

export PYTHONPATH="${DISTILLATION_ROOT}/src:${LTX2_ROOT}/packages/ltx-causal/src:${LTX2_ROOT}/packages/ltx-core/src:${LTX2_ROOT}/packages/ltx-pipelines/src${PYTHONPATH:+:${PYTHONPATH}}"

# Distributed settings
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
echo "Stage 3: Causal DMD Training"
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
