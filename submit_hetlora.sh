#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:2
#SBATCH --job-name=fed_hetlora
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/fed_hetlora_%j.out

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON="/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python -u"
SIM_DIR=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora

# HetLoRA (EMNLP 2024) with heterogeneous ranks [4, 16, 32, 64]
# lambda=0.01, gamma=0.5 (paper defaults)
$PYTHON $SIM_DIR/run_simulation.py \
    --hetlora \
    --ranks 4 16 32 64 \
    --het_lambda 0.01 \
    --het_gamma 0.5 \
    --num_rounds 10 \
    --num_clients 4 \
    --num_gpus 2 \
    --output_dir results/fed_sim_hetlora \
    --base_model Qwen/Qwen3.5-0.8B \
    --epochs 1 \
    --max_steps 100

echo "HetLoRA Simulation Done."
