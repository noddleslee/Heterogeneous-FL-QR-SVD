#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:2
#SBATCH --job-name=fed_rank64
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/fed_rank64_%j.out

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON="/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python -u"
SIM_DIR=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora

# Fixed Rank 64
$PYTHON $SIM_DIR/run_simulation.py \
    --num_rounds 10 \
    --num_clients 4 \
    --num_gpus 2 \
    --output_dir results/fed_sim_rank64 \
    --rank 64 \
    --base_model Qwen/Qwen3.5-0.8B \
    --epochs 1 \
    --max_steps 100

echo "Rank 64 Simulation Done."
