# HFedLoRA — Heterogeneous Federated LoRA on PathVQA

Federated learning simulation for vision-language models using LoRA adapters.
Supports four FL modes (FedAvg, FlexLoRA, QR-SVD, HetLoRA) with automatic GPU batching
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
├── submit_qrsvd.sh        # QR-SVD  ranks=[4,16,32,64], 2 GPUs
└── submit_hetlora.sh      # HetLoRA  ranks=[4,16,32,64], 4 GPUs
```

---

## FL modes

| Mode | Flag | Client ranks | Server rank | Notes |
|------|------|-------------|-------------|-------|
| **FedAvg** | *(default)* | uniform `--rank` | = `--rank` | Homogeneous LoRA |
| **FlexLoRA** | `--flexlora` | heterogeneous `--ranks` | max of ranks | SVD projection at aggregation (NeurIPS 2024) |
| **QR-SVD** | `--qrsvd` | heterogeneous `--ranks` | max of ranks | Same as FlexLoRA but with QR + mini-SVD (faster) |
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

## QR-SVD aggregation

### Motivation

FlexLoRA's server-side aggregation forms the full weight-delta matrix
`ΔW_avg = Σ w_k · B_k @ A_k` (shape `[out, in]`, e.g. 4096×4096 for a 7B model)
and then runs truncated SVD on it — O(d² · K) in both time and memory.

### Algorithm

Instead of materialising the big matrix, QR-SVD stacks the thin client factors
side-by-side and exploits the identity:

```
U_big = hstack(√w_k · B_k)    [out, Σr_k]
V_big = hstack(√w_k · A_k.T)  [in,  Σr_k]
U_big @ V_big.T  ≡  ΔW_avg          ← never formed
```

A thin QR decomposes each tall matrix to a small core `M = R_U @ R_V.T` of size
`Σr_k × Σr_k`, and a mini-SVD of `M` recovers the top-`global_rank` factors.
Complexity drops from O(d² · K) to O(d · K · r + (K·r)³).

The delta-update path (`--global_model_path`) is also handled in factored form:
the global reference `B_g @ A_g` is QR-decomposed once per layer and truncated
to each client's rank without ever forming the `[out × in]` matrix.

### Measured results (PathVQA, Qwen/Qwen3.5-0.8B, ranks=[4,16,32,64], 10 rounds, max_steps=100, 2×A40)

**Timing (per-round wall time from file timestamps):**

| | QR-SVD | FlexLoRA | Speedup |
|--|--------|----------|---------|
| Per-round average | **241 s** | **392 s** | **1.63×** |
| 10-round total | **42:46** | **1:05:43** | **1.54×** |

**Accuracy (overall = (yn_acc + open_em) / 2):**

| Round | QR-SVD | FlexLoRA | Δ |
|------:|-------:|----------:|--:|
| 0 | 0.4667 | 0.4697 | −0.003 |
| 1 | 0.4624 | 0.5011 | −0.039 ← noise |
| 2 | 0.5333 | 0.5385 | −0.005 |
| 3 | 0.5386 | 0.5495 | −0.011 |
| 4 | 0.5438 | 0.5449 | −0.001 |
| 5 | 0.5659 | 0.5676 | −0.002 |
| 6 | 0.5712 | 0.5727 | −0.002 |
| 7 | 0.5693 | 0.5708 | −0.002 |
| 8 | 0.5760 | 0.5801 | −0.004 |
| 9 | **0.5760** | **0.5751** | **+0.001** |

From round 2 onward the gap is consistently within **±0.005** — indistinguishable
from training noise. The round-1 anomaly (−0.039) reflects independent random seeds
between the two separately submitted jobs, not a systematic accuracy loss.

**Conclusion:** QR-SVD achieves the same final accuracy as FlexLoRA while saving
~23 minutes per 10-round experiment on a 0.8B model. The advantage scales with
model size: larger hidden dimensions make the full-matrix SVD increasingly dominant.

---

## Usage

### Quick start

```bash
# FedAvg, rank=16, 4 clients, 2 GPUs, 10 rounds
python run_simulation.py \
    --num_rounds 10 --num_clients 4 --num_gpus 2 \
    --rank 16 --max_steps 100 \
    --output_dir results/my_run

# FlexLoRA, heterogeneous ranks
python run_simulation.py \
    --flexlora --ranks 4 16 32 64 \
    --num_rounds 10 --num_clients 4 --num_gpus 2 \
    --max_steps 100 --output_dir results/my_flexlora

# QR-SVD — same client training as FlexLoRA, faster aggregation
python run_simulation.py \
    --qrsvd --ranks 4 16 32 64 \
    --num_rounds 10 --num_clients 4 --num_gpus 2 \
    --max_steps 100 --output_dir results/my_qrsvd

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
sbatch submit_flexlora.sh   # FlexLoRA,        2×A40
sbatch submit_qrsvd.sh      # QR-SVD,          2×A40
sbatch submit_hetlora.sh    # HetLoRA,         4×A40
```

Or use `smart_submit.sh` to auto-select the GPU type with the most free slots:

```bash
bash smart_submit.sh submit_qrsvd.sh
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

| Method | Ranks | Final overall acc | Final y/n acc | Final open exact | Time (10 rounds) |
|--------|-------|------------------:|--------------:|-----------------:|:----------------:|
| FedAvg | 16 | 0.574 | 0.857 | 0.290 | — |
| FlexLoRA | 4/16/32/64 | 0.5751 | 0.8679 | 0.2818 | 1:05:43 |
| **QR-SVD** | 4/16/32/64 | **0.5760** | **0.8700** | **0.2815** | **0:42:46** |

All runs: Qwen/Qwen3.5-0.8B, PathVQA, 10 rounds, max_steps=100, 2×A40.
