#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH --job-name=het_64_32_16_8_g099
#SBATCH --tasks-per-node=1
#SBATCH --time=05:00:00
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
    --hetlora --ranks 64 32 16 8 \
    --het_lambda 0.01 --het_gamma 0.99 \
    --num_rounds 10 --num_clients 4 --num_gpus 4 \
    --output_dir $RESULTS_DIR/fed_sim_hetlora_64_32_16_8_g099 \
    --base_model Qwen/Qwen3.5-0.8B --epochs 1 --max_steps 100

echo "Done: HetLoRA [64,32,16,8] gamma=0.99"
