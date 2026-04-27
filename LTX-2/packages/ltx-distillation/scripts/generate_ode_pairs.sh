#!/bin/bash
# =============================================================================
# ODE Pair Generation - Single GPU
# =============================================================================
# Generates ODE trajectory pairs using the bidirectional teacher model.
# These pairs are used for causal model initialization (Stage 1).
#
# Usage:
#   ./scripts/generate_ode_pairs.sh
#
# Prerequisites:
#   - Bidirectional teacher model (either original or DMD-distilled)
#   - Gemma text encoder
#   - Prompts file (one prompt per line)

set -e

# =============================================================================
# Virtual Environment
# =============================================================================
if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated venv: ${VENV_PATH}"
fi

# =============================================================================
# Configuration
# =============================================================================

# Teacher model checkpoint
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/path/to/checkpoints/ltx-2-19b-dev.safetensors}"

# Gemma text encoder
GEMMA_PATH="${GEMMA_PATH:-/path/to/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized}"

# Prompts file (one prompt per line, generate with pe/batch_enhance.py)
PROMPTS_FILE="${PROMPTS_FILE:-/path/to/data/prompts.txt}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./ode_pairs}"

# Video configuration
NUM_FRAMES="${NUM_FRAMES:-121}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-512}"
VIDEO_WIDTH="${VIDEO_WIDTH:-768}"

# Generation parameters
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
VIDEO_GUIDANCE_SCALE="${VIDEO_GUIDANCE_SCALE:-3.0}"
AUDIO_GUIDANCE_SCALE="${AUDIO_GUIDANCE_SCALE:-7.0}"
RESCALE_SCALE="${RESCALE_SCALE:-0.7}"

# CPU threading (128 CPUs per node, single GPU)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-128}"

# =============================================================================
# Run Generation
# =============================================================================

echo "=============================================="
echo "ODE Pair Generation (Single GPU)"
echo "=============================================="
echo "Teacher:       ${TEACHER_CHECKPOINT}"
echo "Gemma:         ${GEMMA_PATH}"
echo "Prompts:       ${PROMPTS_FILE}"
echo "Output:        ${OUTPUT_DIR}"
echo "Frames:        ${NUM_FRAMES}"
echo "Resolution:    ${VIDEO_HEIGHT}x${VIDEO_WIDTH}"
echo "Steps:         ${NUM_INFERENCE_STEPS}"
echo "Video CFG:     ${VIDEO_GUIDANCE_SCALE}"
echo "Audio CFG:     ${AUDIO_GUIDANCE_SCALE}"
echo "Rescale:       ${RESCALE_SCALE}"
echo "=============================================="

python -m ltx_distillation.ode.generate_ode_pairs \
    --teacher_checkpoint ${TEACHER_CHECKPOINT} \
    --gemma_path ${GEMMA_PATH} \
    --prompts_file ${PROMPTS_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --num_frames ${NUM_FRAMES} \
    --video_height ${VIDEO_HEIGHT} \
    --video_width ${VIDEO_WIDTH} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --video_guidance_scale ${VIDEO_GUIDANCE_SCALE} \
    --audio_guidance_scale ${AUDIO_GUIDANCE_SCALE} \
    --rescale_scale ${RESCALE_SCALE} \
    "$@"

echo "=============================================="
echo "ODE pair generation complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=============================================="
