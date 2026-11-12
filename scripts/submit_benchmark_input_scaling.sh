#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --job-name=bench_input_scaling
#SBATCH --tasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1

PYTHON="${HFEDLORA_PYTHON:-python}"
SIM_DIR="${HFEDLORA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

for i in $(seq 1 4); do
    echo "=== Run $i / 4 (append run2-5) ==="
    "$PYTHON" -u "$SIM_DIR/core/benchmark_input_scaling.py"
done

echo "Done: input-size scaling benchmark (append run2-5)"
