#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH --job-name=fedavg
#SBATCH --tasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ���� configurable ����������������������������������������������������������������������������������������������������������������������������
RANK=${RANK:-64}
ROUNDS=${ROUNDS:-10}
CLIENTS=${CLIENTS:-4}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-100}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen3.5-0.8B"}
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

set -e

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON="${HFEDLORA_PYTHON:-python}"
SIM_DIR="${HFEDLORA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RESULTS_DIR="${HFEDLORA_RESULTS_DIR:-$SIM_DIR/results/runs}"

OUTPUT_DIR=$RESULTS_DIR/fed_sim_fedavg_r${RANK}

echo "FedAvg: rank=${RANK}, rounds=${ROUNDS}, clients=${CLIENTS}, output=${OUTPUT_DIR}"

"$PYTHON" -u "$SIM_DIR/core/run_simulation.py" \
    --rank        $RANK \
    --num_rounds  $ROUNDS \
    --num_clients $CLIENTS \
    --num_gpus    4 \
    --epochs      $EPOCHS \
    --max_steps   $MAX_STEPS \
    --base_model  $BASE_MODEL \
    --output_dir  $OUTPUT_DIR

echo "Done: FedAvg rank=${RANK}"
