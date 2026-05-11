#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:2
#SBATCH --job-name=2ep_r4
#SBATCH --tasks-per-node=1
#SBATCH --exclude=alvis3-08
#SBATCH --time=2:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/slurm_2epoch_rank4_%j.out

ml load CUDA/12.6.0
ml swap GCCcore/13.3.0 GCCcore/12.3.0 || ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON_BIN=/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin
TRAIN=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/train_1epoch.py
OUTPUT=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/2epoch_rank4

$PYTHON_BIN/torchrun --nproc_per_node=2 $TRAIN \
    --rank 4 \
    --num_epochs 2 \
    --batch_size 64 \
    --output_dir $OUTPUT

echo "Done."
