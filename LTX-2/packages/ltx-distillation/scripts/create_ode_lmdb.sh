#!/bin/bash
# =============================================================================
# LMDB Creation Script for ODE Pairs
# =============================================================================
# Converts individual .pt trajectory files to LMDB format for efficient loading.
#
# Usage:
#   ./scripts/create_ode_lmdb.sh
#
# Prerequisites:
#   - Generated ODE pairs in .pt format (from generate_ode_pairs.sh)

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

# Input directory containing .pt files
DATA_PATH="${DATA_PATH:-./ode_pairs}"

# Output LMDB path
LMDB_PATH="${LMDB_PATH:-./ode_lmdb}"

# Maximum LMDB size (default: 5TB)
MAP_SIZE="${MAP_SIZE:-5000000000000}"

# CPU threading (128 CPUs per node, single process)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-128}"

# =============================================================================
# Run Conversion
# =============================================================================

echo "=============================================="
echo "LMDB Creation for ODE Pairs"
echo "=============================================="
echo "Input:         ${DATA_PATH}"
echo "Output:        ${LMDB_PATH}"
echo "Max Size:      ${MAP_SIZE} bytes"
echo "=============================================="

python -m ltx_distillation.ode.create_lmdb \
    --data_path ${DATA_PATH} \
    --lmdb_path ${LMDB_PATH} \
    --map_size ${MAP_SIZE}

echo "=============================================="
echo "LMDB creation complete!"
echo "Output saved to: ${LMDB_PATH}"
echo "=============================================="
