#!/bin/bash
# Small scale test for Federated Simulation

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1

PYTHON=/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python
SIM_DIR=/mimer/NOBACKUP/groups/bloom/shenghui/hfl/federated_pathvqa

$PYTHON -u $SIM_DIR/run_simulation.py \
    --num_rounds 1 \
    --num_clients 4 \
    --output_dir /mimer/NOBACKUP/groups/bloom/shenghui/hfl/fed_sim_test \
    --rank 4 \
    --epochs 1 \
    --max_steps 5

echo "Test Done."
