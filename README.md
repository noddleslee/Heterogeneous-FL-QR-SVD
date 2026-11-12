# QR-SVD for Heterogeneous Federated LoRA

Efficient server aggregation for federated fine-tuning when clients use different LoRA ranks. QR-SVD reformulates FlexLoRA to avoid reconstructing and decomposing full weight-update matrices.

**Master's thesis:** [Heterogeneous Federated Fine-Tuning of Foundation Models (PDF)](https://uu.diva-portal.org/smash/get/diva2:2108786/FULLTEXT01.pdf)

## What is QR-SVD?

QR-SVD is a server-side aggregation method proposed in this thesis for federated fine-tuning with heterogeneous LoRA ranks. FlexLoRA reconstructs and averages the client weight updates into a full $d_{\mathrm{out}}\times d_{\mathrm{in}}$ matrix, then performs SVD on this full matrix and truncates the result to the target rank. Here, $d_{\mathrm{out}}$ and $d_{\mathrm{in}}$ are the output and input dimensions of the adapted weight matrix.

QR-SVD instead stacks the weighted low-rank factors, performs two thin QR decompositions, and applies SVD to a $\rho\times\rho$ core matrix. Mapping the singular vectors back to the original dimensions produces a numerically equivalent rank-constrained update without explicitly forming the full weight-update matrix. Local training follows FlexLoRA.

**The key efficiency condition:** in the intended low-rank setting, the total client rank is much smaller than the full matrix dimensions:

$$
\rho=\sum_{k=1}^{K}r_k \ll \min(d_{\mathrm{out}},d_{\mathrm{in}}).
$$

For example, with $\rho=120$ and $d_{\mathrm{out}}=d_{\mathrm{in}}=1024$, the core SVD operates on a **120 × 120** matrix instead of a **1024 × 1024** matrix.

![Original FlexLoRA workflow with a comparison of the server computation in QR-SVD](assets/01_flexlora_vs_qrsvd.png)

Method: thesis §3.3–3.3.1. Left workflow: Figure 2 of [Bai et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1a134b50202088aa8c595cc99b310e5a-Abstract-Conference.html); the right panel highlights the QR-SVD reformulation.

## Does QR-SVD preserve model accuracy?

**The evaluated runs achieve similar accuracy:** QR-SVD reaches **59.00%**, compared with FlexLoRA's **59.10%**, a difference of **0.10 percentage points**.

| Method | LoRA rank configuration | Final validation accuracy |
|---|---|---:|
| Centralized | r = 32 | 60.23% |
| FedAvg | r = 64 | 61.89% |
| FedAvg | r = 32 | 59.34% |
| HETLoRA | [64, 32, 16, 8], γ = 0.99 | 59.27% |
| FlexLoRA | [64, 32, 16, 8] | 59.10% |
| **QR-SVD** | **[64, 32, 16, 8]** | **59.00%** |

Source: thesis Table 5.5. Qwen3.5-0.8B on PathVQA validation (6,259 samples); federated scores are at round 9. Centralized r = 32 uses 1,000 steps with batch 256. The three heterogeneous methods differ by at most 0.27 percentage points in these runs.

## How much faster is QR-SVD aggregation?

The figure combines the three model-proxy curves from thesis Figure 5.9 / Table 5.4. Each curve uses five runs on an A100; increasing the client count also increases the total rank.

$$
\mathrm{Speedup}_i=\frac{t_{\mathrm{FlexLoRA},i}}{t_{\mathrm{QR\text{-}SVD},i}},\qquad
\mathrm{plotted\ speedup}=\frac{1}{5}\sum_{i=1}^{5}\mathrm{Speedup}_i.
$$

Here, $t$ denotes aggregation time; larger values mean greater acceleration.

![QR-SVD aggregation speedup across client counts for three model proxy workloads](assets/03_when_qrsvd_helps.png)

[Original five-run benchmark JSON](results/benchmarks) · [Recompute the summary](eval/summarize_benchmarks.py)

## What scenarios is QR-SVD best suited for?

**Cross-silo federated learning** involves a small, stable set of organizations training a shared model while keeping their data local. As the figure above shows, QR-SVD is particularly attractive for large weight matrices when the total participating LoRA rank $\rho$ is small relative to the matrix dimensions.

For setup and commands, see the [running guide](docs/running.md).

Accuracy values are reported thesis results; per-sample validation prediction files are not included in this snapshot.
