"""
HETLoRA aggregation-time benchmark (single rank configuration).

Uses the SAME harness as benchmark_rank_scaling.py (identical tensor
allocation, 186-layer loop, cuda.synchronize timing, d=1024, all on GPU),
so the HETLoRA number is directly comparable to the FlexLoRA / QR-SVD
results at ranks [8,16,32,64] (rho=120, 186 layers, A100 GPU).

HETLoRA aggregation (EMNLP 2024): Frobenius-norm-weighted average of
zero-padded LoRA factors. No SVD, but forms the d×d matrix dW = B@A per
client to compute the Frobenius norm.
"""

import sys, os, time, json, math, re, glob
sys.path.insert(0, os.path.dirname(__file__))

import torch

NUM_CLIENTS = 4
D           = 1024    # hidden dim (Qwen3-0.8B)
NUM_LAYERS  = 186     # 24 LLM layers: 18 DeltaNet×8 + 6 Attention×7

RANK_CONFIGS = [
    [8, 16, 32, 64],   # single point: global_rank=64, rho=120
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def hetlora_agg_layer(B_list, A_list, global_rank):
    """HETLoRA: Frobenius-norm-weighted average of zero-padded factors.
    Forms d×d dW = B@A per client to compute the norm (no SVD).
    B_list : list of [d, r_k] float32 GPU tensors
    A_list : list of [r_k, d] float32 GPU tensors
    """
    d_out = B_list[0].shape[0]
    d_in  = A_list[0].shape[1]
    r_max = global_rank

    # step 1: per-client Frobenius norm of dW = B@A (forms d×d, no SVD)
    norms = []
    for B_k, A_k in zip(B_list, A_list):
        dW = B_k @ A_k
        norms.append(torch.linalg.norm(dW))
    total = sum(norms)
    p = [n / total for n in norms]

    # step 2: zero-pad each factor to r_max, then sparsity-weighted average
    B_g = torch.zeros(d_out, r_max, dtype=torch.float32, device=DEVICE)
    A_g = torch.zeros(r_max, d_in, dtype=torch.float32, device=DEVICE)
    for pk, B_k, A_k in zip(p, B_list, A_list):
        r_k = B_k.shape[1]
        B_g[:, :r_k] += pk * B_k
        A_g[:r_k, :] += pk * A_k
    _ = (B_g, A_g)


# ── benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark():
    results = {}
    base = os.path.join(os.path.dirname(__file__), "..", "results", "benchmarks", "benchmark_hetlora.json")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    existing = [int(re.search(r"_run(\d+)\.json$", f).group(1))
                for f in glob.glob(base.replace(".json", "_run*.json"))
                if re.search(r"_run(\d+)\.json$", f)]
    run_n = max(existing, default=0) + 1
    out_path = base.replace(".json", f"_run{run_n}.json")
    print(f"Run {run_n} → {out_path}")

    sync = lambda: torch.cuda.synchronize() if DEVICE == "cuda" else None

    for ranks in RANK_CONFIGS:
        global_rank = max(ranks)
        rho         = sum(ranks)
        key         = str(global_rank)
        w           = 1.0 / NUM_CLIENTS
        sqrt_w      = math.sqrt(w)
        print(f"\nranks = {ranks}  global_rank = {global_rank}  ρ = {rho}")

        # Pre-build GPU tensors once; reuse across all layers (same as
        # benchmark_rank_scaling.py, so timing is directly comparable)
        B_list = [torch.randn(D, r, dtype=torch.float32, device=DEVICE) * sqrt_w
                  for r in ranks]
        A_list = [torch.randn(r, D, dtype=torch.float32, device=DEVICE) * sqrt_w
                  for r in ranks]
        sync()

        # ── HETLoRA ───────────────────────────────────────────────────────────
        sync()
        t0 = time.time()
        for _ in range(NUM_LAYERS):
            hetlora_agg_layer(B_list, A_list, global_rank)
        sync()
        t_het     = round(time.time() - t0, 3)
        t_het_per = round(t_het / NUM_LAYERS, 4)

        print(f"  HETLoRA: {t_het}s ({t_het_per}s/layer)")

        results[key] = {
            "config": {
                "ranks":       ranks,
                "global_rank": global_rank,
                "rho":         rho,
                "num_clients": NUM_CLIENTS,
                "d":           D,
                "num_layers":  NUM_LAYERS,
            },
            "hetlora_total_s":     t_het,
            "hetlora_per_layer_s": t_het_per,
        }

        with open(out_path, "w") as f:
            json.dump({"num_clients": NUM_CLIENTS, "d": D,
                       "num_layers": NUM_LAYERS, "results": results},
                      f, indent=2)
        print(f"  (saved → {out_path})")

    print(f"\nAll done. Results in {out_path}")


if __name__ == "__main__":
    run_benchmark()
