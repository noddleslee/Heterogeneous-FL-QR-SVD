# Running the thesis code

Run commands from the repository root in a Linux environment with CUDA and SLURM. The submission scripts target NAISS Alvis; change the account, partition and module commands for another cluster. Activate the ML environment before submitting. The Python entry points use the active interpreter and the evaluator included in `eval/`.

## Federated experiment

```bash
python core/run_simulation.py \
  --qrsvd --ranks 64 32 16 8 \
  --num_rounds 10 --num_clients 4 --num_gpus 4 \
  --batch_size 64 --max_steps 100 \
  --output_dir results/runs/qrsvd_64_32_16_8
```

Use `--flexlora` instead of `--qrsvd` for the direct comparison. For HETLoRA, use `--hetlora --het_gamma 0.99 --het_lambda 0.01`.

```bash
sbatch scripts/submit_qrsvd_64_32_16_8_0.8B.sh
sbatch scripts/submit_flex_64_32_16_8_0.8B.sh
sbatch scripts/submit_het_64_32_16_8_g099.sh
```

Each round starts fresh client processes and retains training state through checkpoints. Evaluation is submitted as a separate validation job. The current entry point does not implement persistent worker processes or `--no-persistent`.

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HFEDLORA_PYTHON` | Active Python interpreter / `python` in submission scripts | Interpreter for workers and jobs |
| `HFEDLORA_EVAL_SCRIPT` | `eval/eval_pathvqa.py` in this checkout | Explicit evaluator override |
| `HFEDLORA_CACHE_DIR` | `.dataset_cache/` in this checkout | Dataset cache |
| `HFEDLORA_PROJECT_DIR` | SLURM submission directory | Checkout root for batch scripts |
| `HFEDLORA_RESULTS_DIR` | `results/runs/` in this checkout | Training outputs |

## Benchmarks and supplied results

The eight benchmark families have five original run JSON files each in [`results/benchmarks`](../results/benchmarks). The square-matrix workloads use dimensions and matrix counts as model proxies. They do not load the complete 9B or 27B models.

Recompute the summary without a GPU or ML dependencies:

```bash
python eval/summarize_benchmarks.py
```

This reads runs 1–5, checks their configurations, and writes `summary_all.csv`. Means, sample standard deviations and 95% Student t intervals use unrounded JSON values (`t=2.776445`, four degrees of freedom). Small differences from printed confidence-interval endpoints can result from rounding in the thesis. The CSV distinguishes the ratio of mean times from the mean of the five run-level time ratios; the client-count speedup figure uses the latter.

To run a new five-run benchmark on Alvis:

```bash
sbatch scripts/submit_benchmark_client_scaling_0.8B.sh
sbatch scripts/submit_benchmark_client_scaling_9B.sh
sbatch scripts/submit_benchmark_client_scaling_27B.sh
```

Benchmark scripts choose new run numbers, so the supplied runs are preserved. The summary command deliberately selects the supplied runs 1–5.

Accuracy values in the README are reported thesis results. The per-sample validation prediction JSON files are not included in this snapshot.

## Checks

```bash
# Launch configuration checks; Python standard library only
python -m unittest discover -s tests -p test_launch_config.py -v

# QR-SVD numerical checks; require PyTorch, safetensors and pytest
pytest tests/test_qrsvd.py -v
```

All tests locate `core/` and `eval/` through `tests/conftest.py`. A full training run additionally needs the Unsloth / Transformers / TRL / PEFT stack, datasets, Pillow and bitsandbytes.

## Recorded thesis environment

These versions are recorded in the final thesis, Chapter 4; they are an environment record rather than a newly validated installation recipe.

| Component | Recorded version |
|---|---|
| Python | 3.10.8 |
| PyTorch / CUDA | 2.7.1 / 12.6 |
| Unsloth | 2026.6.8 |
| Transformers | 5.15.1 |
| PEFT | 0.20.0 |
| TRL | 1.10.0 |
| bitsandbytes | 0.50.1 |
| Federated hardware | Four A100 80 GB GPUs |
