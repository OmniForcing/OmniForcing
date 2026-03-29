#!/bin/bash
# =============================================================================
# ODE Pair Generation - Multi-Node Multi-GPU
# =============================================================================
# Generates ODE trajectory pairs using the bidirectional teacher model.
# Each GPU across all nodes processes a disjoint shard of prompts,
# saving to the same shared output directory.
#
# Usage (2 nodes, 8 GPUs each):
#
#   Node 0 (Master, IP: 10.0.0.1):
#       NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#           ./scripts/generate_ode_pairs_multi_node.sh
#
#   Node 1 (Worker, IP: 10.0.0.2):
#       NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
#           ./scripts/generate_ode_pairs_multi_node.sh
#
# Environment Variables:
#   NNODES       - Total number of nodes (required)
#   NODE_RANK    - Current node rank, master=0 (required)
#   MASTER_ADDR  - Master node IP (required)
#   MASTER_PORT  - Communication port (default: 29501)
#   NUM_GPUS     - GPUs per node (default: 8)
#
# =============================================================================

set -e

# =============================================================================
# Virtual Environment
# =============================================================================
if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated venv: ${VENV_PATH}"
fi

# =============================================================================
# Check required environment variables
# =============================================================================
if [ -z "$NNODES" ]; then
    echo "Error: NNODES not set."
    echo "Usage: NNODES=2 NODE_RANK=0 MASTER_ADDR=x.x.x.x ./scripts/generate_ode_pairs_multi_node.sh"
    exit 1
fi

if [ -z "$NODE_RANK" ]; then
    echo "Error: NODE_RANK not set."
    echo "Usage: NNODES=2 NODE_RANK=0 MASTER_ADDR=x.x.x.x ./scripts/generate_ode_pairs_multi_node.sh"
    exit 1
fi

if [ -z "$MASTER_ADDR" ]; then
    echo "Error: MASTER_ADDR not set."
    echo "Usage: NNODES=2 NODE_RANK=0 MASTER_ADDR=x.x.x.x ./scripts/generate_ode_pairs_multi_node.sh"
    exit 1
fi

# =============================================================================
# Configuration
# =============================================================================

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

# CPU threading (128 CPUs per node)
NUM_CPUS="${NUM_CPUS:-128}"
export OMP_NUM_THREADS=$((NUM_CPUS / NUM_GPUS))

TOTAL_GPUS=$((NUM_GPUS * NNODES))

# Teacher model checkpoint
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/path/to/checkpoints/ltx-2-19b-dev.safetensors}"

# Gemma text encoder
GEMMA_PATH="${GEMMA_PATH:-/path/to/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized}"

# Prompts file (one prompt per line, generate with pe/batch_enhance.py)
PROMPTS_FILE="${PROMPTS_FILE:-/path/to/data/prompts.txt}"

# Output directory (shared across all nodes, must be on shared filesystem)
OUTPUT_DIR="${OUTPUT_DIR:-./ode_pairs}"

# Video configuration
NUM_FRAMES="${NUM_FRAMES:-121}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-512}"
VIDEO_WIDTH="${VIDEO_WIDTH:-768}"

# Generation parameters
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"

# =============================================================================
# Run Generation
# =============================================================================

echo "=============================================="
echo "ODE Pair Generation (Multi-Node)"
echo "=============================================="
echo "Teacher:       ${TEACHER_CHECKPOINT}"
echo "Gemma:         ${GEMMA_PATH}"
echo "Prompts:       ${PROMPTS_FILE}"
echo "Output:        ${OUTPUT_DIR}"
echo "Nodes:         ${NNODES}"
echo "Node Rank:     ${NODE_RANK}"
echo "GPUs/Node:     ${NUM_GPUS}"
echo "Total GPUs:    ${TOTAL_GPUS}"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Frames:        ${NUM_FRAMES}"
echo "Resolution:    ${VIDEO_HEIGHT}x${VIDEO_WIDTH}"
echo "Steps:         ${NUM_INFERENCE_STEPS}"
echo "Guidance:      ${GUIDANCE_SCALE}"
echo "=============================================="

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NUM_GPUS} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m ltx_distillation.ode.generate_ode_pairs \
    --teacher_checkpoint ${TEACHER_CHECKPOINT} \
    --gemma_path ${GEMMA_PATH} \
    --prompts_file ${PROMPTS_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --num_frames ${NUM_FRAMES} \
    --video_height ${VIDEO_HEIGHT} \
    --video_width ${VIDEO_WIDTH} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    "$@"

echo "=============================================="
echo "ODE pair generation complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=============================================="
