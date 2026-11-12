#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH --job-name=qrsvd_c8_0.8B
#SBATCH --tasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON="${HFEDLORA_PYTHON:-python}"
SIM_DIR="${HFEDLORA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RESULTS_DIR="${HFEDLORA_RESULTS_DIR:-$SIM_DIR/results/runs}"

"$PYTHON" -u "$SIM_DIR/core/run_simulation.py" \
    --qrsvd \
    --ranks 8 16 32 64 \
    --num_rounds 10 --num_clients 8 --num_gpus 4 \
    --output_dir $RESULTS_DIR/fed_sim_qrsvd_c8_0.8B \
    --base_model Qwen/Qwen3.5-0.8B --epochs 1 --max_steps 100

echo "Done: QR-SVD c=8 [8,16,32,64]×2 0.8B"
