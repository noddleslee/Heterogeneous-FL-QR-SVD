#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH --job-name=flex_32_16_8_4_fast_verify
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

# Verification run: same FlexLoRA [32,16,8,4] config as baseline, but with
# --fast_agg (aggregate_flexlora_fast.py). Output to a SEPARATE dir so the
# original baseline (fed_sim_flexlora_32_16_8_4_0.8B) is never overwritten.
"$PYTHON" -u "$SIM_DIR/core/run_simulation.py" \
    --flexlora \
    --fast_agg \
    --ranks 32 16 8 4 \
    --num_rounds 10 --num_clients 4 --num_gpus 4 \
    --output_dir $RESULTS_DIR/fed_sim_flexlora_32_16_8_4_0.8B_fast_verify \
    --base_model Qwen/Qwen3.5-0.8B --epochs 1 --max_steps 100

echo "Done: FlexLoRA [32,16,8,4] 0.8B  (fast_agg verification)"
