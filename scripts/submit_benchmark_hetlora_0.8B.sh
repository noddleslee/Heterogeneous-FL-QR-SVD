#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --job-name=bench_hetlora
#SBATCH --tasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1

PYTHON="${HFEDLORA_PYTHON:-python}"
SIM_DIR="${HFEDLORA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

for i in $(seq 1 5); do
    echo "=== Run $i / 5 ==="
    "$PYTHON" -u "$SIM_DIR/core/benchmark_hetlora.py"
done

echo "Done: HETLoRA benchmark, ranks [8,16,32,64] (5 runs)"
