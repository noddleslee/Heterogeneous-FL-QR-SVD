#!/bin/bash
# smart_submit.sh <submit_script.sh> [extra sbatch args...]
#
# Detects which GPU type (A100 or A40) has more free GPUs on alvis,
# then submits the given SLURM script with --gpus-per-node overridden.
# Command-line --gpus-per-node takes precedence over the #SBATCH directive
# inside the script, so the script itself never needs editing.
#
# Examples:
#   bash smart_submit.sh scripts/submit_fedavg.sh
#   bash smart_submit.sh scripts/submit_flex_64_32_16_8_0.8B.sh --time=12:00:00
#   bash smart_submit.sh scripts/submit_het_64_32_16_8_g099.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HFEDLORA_PYTHON:-python}"
# Fall back to any python3 in PATH if the conda env path is missing
command -v "$PYTHON" &>/dev/null || PYTHON="python3"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <submit_script.sh> [extra sbatch args...]" >&2
    exit 1
fi

SCRIPT="$1"; shift
EXTRA_ARGS=("$@")

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: script not found: $SCRIPT" >&2
    exit 1
fi

# ── detect how many GPUs the script requests ──────────────────────────────────
# Read #SBATCH --gpus-per-node=TYPE:N and extract N (default 2)
N_GPUS=$(grep -oE '\-\-gpus-per-node=[A-Za-z0-9]+:([0-9]+)' "$SCRIPT" 2>/dev/null \
         | grep -oE '[0-9]+$' | head -1)
N_GPUS="${N_GPUS:-2}"

# ── pick the GPU type with the most free GPUs ─────────────────────────────────
# pick_gpu.py prints the winner to stdout and free-count info to stderr
GPU_TYPE=$("$PYTHON" "$SCRIPT_DIR/../core/pick_gpu.py")
if [[ -z "$GPU_TYPE" ]]; then
    echo "[smart_submit] GPU detection failed, defaulting to A40" >&2
    GPU_TYPE="A40"
fi

echo "[smart_submit] Submitting $SCRIPT with --gpus-per-node=${GPU_TYPE}:${N_GPUS}"

sbatch --gpus-per-node="${GPU_TYPE}:${N_GPUS}" "${EXTRA_ARGS[@]}" "$SCRIPT"
