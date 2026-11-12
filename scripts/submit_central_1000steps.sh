#!/bin/bash
# Submit centralized training jobs: 2 batch sizes × 5 ranks = 10 jobs
# Batch 64  → ~3.25 epochs (matches FL optimizer-step count)
# Batch 256 → ~13.02 epochs (matches FL total sample views)

set -e

ACCOUNT="NAISS2025-22-1535"
PYTHON="${HFEDLORA_PYTHON:-python}"
SIM_DIR="${HFEDLORA_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RESULTS_DIR="${HFEDLORA_RESULTS_DIR:-$SIM_DIR/results/runs}"
LOG_DIR="$SIM_DIR/logs"
BASE_MODEL="Qwen/Qwen3.5-0.8B"

mkdir -p "$LOG_DIR"

for BATCH in 64 256; do
  for RANK in 4 8 16 32 64; do
    OUTPUT_DIR="$RESULTS_DIR/centralized_r${RANK}_b${BATCH}_1000s"
    JOB_NAME="cen_r${RANK}_b${BATCH}"

    sbatch <<EOF
#!/bin/bash
#SBATCH -A $ACCOUNT -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --job-name=$JOB_NAME
#SBATCH --tasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.err

set -e
ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

"$PYTHON" -u "$SIM_DIR/core/train_centralized.py" \\
    --rank       $RANK \\
    --batch_size $BATCH \\
    --base_model $BASE_MODEL \\
    --output_dir $OUTPUT_DIR

echo "Done: rank=$RANK batch=$BATCH"
EOF

    echo "Submitted: rank=$RANK batch=$BATCH → $OUTPUT_DIR"
  done
done

echo ""
echo "All 10 jobs submitted."
