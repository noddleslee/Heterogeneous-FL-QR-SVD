#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH --job-name=fed_qrsvd
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/cephyr/users/huanli/Alvis/hfedlora/logs/fed_qrsvd_%j.out

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON="/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python -u"
SIM_DIR=/cephyr/users/huanli/Alvis/hfedlora

# QR-SVD with heterogeneous ranks: 4, 16, 32, 64
# Client training identical to FlexLoRA; aggregation uses QR + mini-SVD
# instead of full-matrix SVD (avoids forming the [out x in] matrix).
$PYTHON $SIM_DIR/core/run_simulation.py \
    --qrsvd \
    --ranks 4 16 32 64 \
    --num_rounds 10 \
    --num_clients 4 \
    --num_gpus 4 \
    --output_dir $SIM_DIR/results/fed_sim_qrsvd \
    --base_model Qwen/Qwen3.5-0.8B \
    --epochs 1 \
    --max_steps 100

echo "QR-SVD Simulation Done."
