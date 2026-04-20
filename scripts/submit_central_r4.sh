#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --job-name=central_r4
#SBATCH --tasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/cephyr/users/huanli/Alvis/hfedlora/logs/central_r4_%j.out

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON="/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python -u"
SIM_DIR=/cephyr/users/huanli/Alvis/hfedlora

$PYTHON $SIM_DIR/core/train_centralized.py \
    --rank 4 \
    --output_dir $SIM_DIR/results/centralized_r4 \
    --base_model Qwen/Qwen3.5-0.8B \
    --max_steps 1000

echo "Training done. Running evaluation on validation split ..."

$PYTHON $SIM_DIR/eval/eval_pathvqa.py \
    --base_model Qwen/Qwen3.5-0.8B \
    --peft_path $SIM_DIR/results/centralized_r4/adapter \
    --split validation \
    --n_samples none \
    --output_json $SIM_DIR/results/centralized_r4/eval_results.json

echo "Centralized rank 4 done."
