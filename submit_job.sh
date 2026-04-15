#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:2
#SBATCH --job-name=fed_sim
#SBATCH --tasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/mimer/NOBACKUP/groups/bloom/shenghui/hfl/federated_pathvqa/fed_sim_%j.out

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON="/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python -u"
SIM_DIR=/mimer/NOBACKUP/groups/bloom/shenghui/hfl/federated_pathvqa

$PYTHON $SIM_DIR/run_simulation.py \
    --num_rounds 3 \
    --num_clients 4 \
    --output_dir /mimer/NOBACKUP/groups/bloom/shenghui/hfl/fed_sim_output \
    --rank 4 \
    --epochs 1 \
    --max_steps 100

echo "Federated Simulation Done."
