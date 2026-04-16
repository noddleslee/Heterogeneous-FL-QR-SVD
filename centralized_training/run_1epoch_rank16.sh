#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --job-name=1ep_r16
#SBATCH --tasks-per-node=1
#SBATCH --exclude=alvis3-08
#SBATCH --time=6:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/slurm_1epoch_rank16_%j.out

ml load CUDA/12.6.0
ml swap GCCcore/13.3.0 GCCcore/12.3.0 || ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON=/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python

$PYTHON /mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/train_1epoch.py \
    --rank 16 \
    --output_dir /mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/1epoch_rank16

echo "Done."
