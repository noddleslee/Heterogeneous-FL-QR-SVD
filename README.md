# HFedLoRA — Heterogeneous Federated LoRA on PathVQA

Federated learning simulation for vision-language models using LoRA adapters.
Supports three FL modes (FedAvg, FlexLoRA, HetLoRA) with automatic GPU batching
and persistent worker processes that eliminate repeated model-loading overhead.

---

## Project structure

```
hfedlora/
├── run_simulation.py      # FL orchestrator — launches workers, aggregates, submits eval
├── train_client.py        # Client training script (single-round or persistent worker)
├── aggregate.py           # FedAvg / FlexLoRA / HetLoRA aggregation
├── eval_pathvqa.py        # Evaluation on PathVQA test set
│
├── submit_rank4.sh        # FedAvg rank=4,  4 GPUs
├── submit_rank16.sh       # FedAvg rank=16, 2 GPUs
├── submit_rank64.sh       # FedAvg rank=64, 4 GPUs
├── submit_flexlora.sh     # FlexLoRA ranks=[4,16,32,64], 4 GPUs
└── submit_hetlora.sh      # HetLoRA  ranks=[4,16,32,64], 4 GPUs
```

---

## FL modes

| Mode | Flag | Client ranks | Server rank | Notes |
|------|------|-------------|-------------|-------|
| **FedAvg** | *(default)* | uniform `--rank` | = `--rank` | Homogeneous LoRA |
| **FlexLoRA** | `--flexlora` | heterogeneous `--ranks` | max of ranks | SVD projection at distribution |
| **HetLoRA** | `--hetlora` | heterogeneous `--ranks` | max of ranks | Adaptive rank pruning (EMNLP 2024) |

---

## GPU scheduling

The orchestrator auto-detects available GPUs and tiles clients across them in batches:

```
num_clients=4, num_gpus=4  →  1 batch  of 4  (all parallel)
num_clients=4, num_gpus=2  →  2 batches of 2 (serial batches, parallel within)
num_clients=6, num_gpus=4  →  batch[4] then batch[2]
```

Aggregation always sees all client adapters at once — batching is transparent to the FL algorithm.

---

## Persistent worker optimization

### Problem

In the naive implementation every FL round relaunches each client as a fresh subprocess, paying the full model-loading cost on every round:

- Python / unsloth library import: ~30–60 s
- `FastVisionModel.from_pretrained`: ~1–3 min
- Total idle time per round (4 clients, 2 GPUs): ~3–4 min of GPU sitting unused

### Solution

`--persistent` (default: **on**) keeps each client process alive for the entire simulation.
The model is loaded **once** at startup; between rounds the orchestrator sends file-based
signals to tell workers when to start/stop each round.

Signal protocol (`{output_dir}/_worker_signals/`):

| File | Written by | Meaning |
|------|-----------|---------|
| `ready_c{C}.json` | worker C | model loaded, ready for round 0 |
| `start_r{R}_c{C}.json` | orchestrator | start round R |
| `done_r{R}_c{C}.json` | worker C | round R complete |

GPU batching is preserved: the orchestrator only sends start signals to the current batch,
so clients sharing a GPU never train simultaneously.

### Measured results (rank=16, 4 clients, 2 GPUs, 10 rounds, Qwen/Qwen3.5-0.8B on A40)

| Round | OLD per-round (s) | NEW per-round (s) | Speedup |
|------:|------------------:|------------------:|--------:|
| 0 | 439 | 240 | 1.83× |
| 1 | 439 | 191 | 2.30× |
| 2 | 445 | 187 | 2.38× |
| 3 | 447 | 196 | 2.28× |
| 4 | 444 | 188 | 2.36× |
| 5 | 448 | 197 | 2.27× |
| 6 | 450 | 199 | 2.26× |
| 7 | 448 | 189 | 2.37× |
| 8 | 439 | 190 | 2.31× |
| 9 | 442 | 191 | 2.31× |
| **Total** | **74 min** | **35 min** | **2.1×** |

Round 0 is slower to start (one-time model loading for all 4 workers: ~132 s).
Rounds 1–9 save ~250 s each — the model-loading overhead that was previously paid every round.

### Effect on accuracy

The persistent mode produces near-identical results to per-round subprocesses.
Differences arise from `dataloader_prefetch_factor=4` (changed from default 2),
which introduces minor non-determinism in data loading order.

| Round | OLD overall | NEW overall | Δ |
|------:|------------:|------------:|--:|
| 0 | 0.4652 | 0.4632 | −0.0020 |
| 1 | 0.5190 | 0.5236 | +0.0046 |
| 2 | 0.5492 | 0.5514 | +0.0022 |
| 3 | 0.5526 | 0.5541 | +0.0015 |
| 4 | 0.5557 | 0.5587 | +0.0030 |
| 5 | 0.5688 | 0.5706 | +0.0018 |
| 6 | 0.5718 | 0.5781 | +0.0063 |
| 7 | 0.5721 | 0.5718 | −0.0003 |
| 8 | 0.5730 | 0.5767 | +0.0037 |
| 9 | 0.5679 | 0.5739 | +0.0060 |

Per-sample agreement: ~84–87% of test samples receive identical predictions.
The remaining ~15% differ due to non-determinism in data loading order;
there is no systematic accuracy regression.

To disable persistent mode (e.g. for exact reproducibility):

```bash
python run_simulation.py ... --no-persistent
```

### HetLoRA

HetLoRA requires dynamic rank updates between rounds (the LoRA module is rebuilt
with a smaller rank after pruning), which is incompatible with a persistent process.
`--hetlora` **automatically disables** persistent mode — no flag needed.

---

## Usage

### Quick start

```bash
# FedAvg, rank=16, 4 clients, 2 GPUs, 10 rounds
python run_simulation.py \
    --num_rounds 10 --num_clients 4 --num_gpus 2 \
    --rank 16 --max_steps 100 \
    --output_dir results/my_run

# FlexLoRA, heterogeneous ranks, 4 GPUs
python run_simulation.py \
    --flexlora --ranks 4 16 32 64 \
    --num_rounds 10 --num_clients 4 --num_gpus 4 \
    --max_steps 100 --output_dir results/my_flexlora

# HetLoRA (persistent auto-disabled)
python run_simulation.py \
    --hetlora --ranks 4 16 32 64 --het_lambda 0.01 --het_gamma 0.5 \
    --num_rounds 10 --num_clients 4 --num_gpus 4 \
    --max_steps 100 --output_dir results/my_hetlora
```

### SLURM submission

```bash
sbatch submit_rank16.sh     # FedAvg rank=16,  2×A40
sbatch submit_rank4.sh      # FedAvg rank=4,   4×A40
sbatch submit_rank64.sh     # FedAvg rank=64,  4×A40
sbatch submit_flexlora.sh   # FlexLoRA,        4×A40
sbatch submit_hetlora.sh    # HetLoRA,         4×A40
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_rounds` | 3 | Number of FL rounds |
| `--num_clients` | 4 | Number of federated clients |
| `--num_gpus` | auto | GPUs to use (auto-detects all visible) |
| `--gpu_ids` | — | Specific GPU IDs, e.g. `--gpu_ids 0 1` |
| `--rank` | 16 | Uniform LoRA rank (FedAvg) |
| `--ranks` | — | Per-client ranks (FlexLoRA / HetLoRA) |
| `--max_steps` | −1 | Steps per round; −1 = full epoch |
| `--epochs` | 1 | Epochs per round (used when `--max_steps -1`) |
| `--persistent` | **True** | Persistent worker processes |
| `--no-persistent` | — | Disable persistent mode |
| `--base_model` | `Qwen/Qwen3.5-0.8B` | HuggingFace model ID |

### Monitoring a running job

```bash
# Main orchestrator log
tail -f fed_rank16_<jobid>.out

# Per-worker logs (model loading, per-round training progress)
tail -f results/<run>/worker_client_{0,1,2,3}.log

# Check eval results as they come in
for r in $(seq 0 9); do
  echo -n "Round $r: "
  python -c "
import json; s = json.load(open('results/<run>/round_$r/eval_results.json'))['summary']
print(f\"overall={s['overall_accuracy']:.4f}  y/n={s['yn_accuracy']:.4f}\")
" 2>/dev/null || echo "(pending)"
done
```

---

## Results summary (PathVQA, Qwen/Qwen3.5-0.8B, 10 rounds, max_steps=100)

| Method | Rank | Final overall acc | Final y/n acc | Final open exact |
|--------|------|------------------:|-------------:|----------------:|
| FedAvg | 16 | 0.574 | 0.857 | 0.290 |

*(FlexLoRA and HetLoRA results to be added)*
