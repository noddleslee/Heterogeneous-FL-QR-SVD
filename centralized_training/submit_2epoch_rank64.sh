#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:2
#SBATCH --job-name=2ep_r64
#SBATCH --tasks-per-node=1
#SBATCH --exclude=alvis3-08
#SBATCH --time=2:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/slurm_2epoch_rank64_%j.out

ml load CUDA/12.6.0
ml swap GCCcore/13.3.0 GCCcore/12.3.0 || ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON_BIN=/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin
TRAIN=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/train_1epoch.py
OUTPUT=/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/centralized_training/2epoch_rank64

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
echo "Using master port: $MASTER_PORT"

$PYTHON_BIN/torchrun --nproc_per_node=2 --master_port=$MASTER_PORT $TRAIN \
    --rank 64 \
    --num_epochs 2 \
    --batch_size 64 \
    --output_dir $OUTPUT

echo "Done."
