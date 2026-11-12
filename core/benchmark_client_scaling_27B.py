"""
Client scaling benchmark: FlexLoRA vs QR-SVD aggregation time.

Fixes rank pattern [8,16,32,64] (cycled across clients) and hidden dim d=5120.
Varies num_clients to measure how aggregation time scales with client count.

Both algorithms use standard formulations, all operations on GPU:
  FlexLoRA : average client deltas �� single economy SVD  O(d_min^2 * d_max)
  QR-SVD   : stack factors �� two thin QR �� small core SVD  O((d_out+d_in)*rho^2)

Model: Qwen3.5-27B  (d=5120, 64 LLM layers: 48 DeltaNet��8 + 16 Attention��7 = 496)
"""

import sys, os, time, json, math, re, glob
sys.path.insert(0, os.path.dirname(__file__))

import torch

BASE_RANKS    = [8, 16, 32, 64]
GLOBAL_RANK   = max(BASE_RANKS)   # 64
D             = 5120               # hidden dim (Qwen3.5-27B)
NUM_LAYERS    = 496                # 48��8 + 16��7
CLIENT_COUNTS = [4, 8, 12, 16, 20]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def flexlora_agg_layer(B_list, A_list, global_rank):
    """Standard FlexLoRA: weighted average of client deltas, then one economy SVD.
    B_list : list of [d, r_k] float32 GPU tensors (pre-scaled by sqrt(w_k))
    A_list : list of [r_k, d] float32 GPU tensors (pre-scaled by sqrt(w_k))
    """
    d_out = B_list[0].shape[0]
    d_in  = A_list[0].shape[1]
    W_avg = torch.zeros(d_out, d_in, dtype=torch.float32, device=DEVICE)
    for B_k, A_k in zip(B_list, A_list):
        W_avg.addmm_(B_k, A_k)                              # in-place accumulate
    U, S, Vh = torch.linalg.svd(W_avg, full_matrices=False) # single economy SVD
    r = min(global_rank, S.shape[0])
    sqrt_S = S[:r].clamp(min=0).sqrt()
    _ = (U[:, :r] * sqrt_S, Vh[:r] * sqrt_S.unsqueeze(1))  # truncated factors


def qrsvd_agg_layer(B_list, A_list, global_rank):
    """QR-SVD: stack pre-scaled factors, two thin QRs, small core SVD.
    B_list : list of [d, r_k] float32 GPU tensors (pre-scaled by sqrt(w_k))
    A_list : list of [r_k, d] float32 GPU tensors (pre-scaled by sqrt(w_k))
    """
    B_stack = torch.cat(B_list, dim=1)    # [d_out, rho]
    A_stack = torch.cat(A_list, dim=0)    # [rho,   d_in]

    Q_B, R_B = torch.linalg.qr(B_stack,   mode="reduced")  # [d_out, rho], [rho, rho]
    Q_A, R_A = torch.linalg.qr(A_stack.T, mode="reduced")  # [d_in,  rho], [rho, rho]

    M = R_B @ R_A.T                                          # [rho, rho]
    U_m, S_m, Vh_m = torch.linalg.svd(M, full_matrices=False)

    r = min(global_rank, S_m.shape[0])
    sqrt_S = S_m[:r].clamp(min=0).sqrt()
    _ = (Q_B @ (U_m[:, :r] * sqrt_S),
         (sqrt_S.unsqueeze(1) * Vh_m[:r]) @ Q_A.T)


def run_benchmark():
    results = {}
    base = os.path.join(os.path.dirname(__file__), "..", "results", "benchmarks",
                        "benchmark_client_scaling_27B.json")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    existing = [int(re.search(r"_run(\d+)\.json$", f).group(1))
                for f in glob.glob(base.replace(".json", "_run*.json"))
                if re.search(r"_run(\d+)\.json$", f)]
    run_n    = max(existing, default=0) + 1
    out_path = base.replace(".json", f"_run{run_n}.json")
    print(f"Run {run_n} �� {out_path}")

    sync = lambda: torch.cuda.synchronize() if DEVICE == "cuda" else None

    for n in CLIENT_COUNTS:
        ranks = [BASE_RANKS[i % len(BASE_RANKS)] for i in range(n)]
        rho   = sum(ranks)
        w     = 1.0 / n
        sqrt_w = math.sqrt(w)
        print(f"\nnum_clients = {n}  ranks = {ranks}  �� = {rho}")

        # Pre-build scaled GPU tensors once; reuse across all layers
        B_list = [torch.randn(D, r, dtype=torch.float32, device=DEVICE) * sqrt_w
                  for r in ranks]
        A_list = [torch.randn(r, D, dtype=torch.float32, device=DEVICE) * sqrt_w
                  for r in ranks]
        sync()  # wait for allocations

        # ���� FlexLoRA ��������������������������������������������������������������������������������������������������������������������
        sync()
        t0 = time.time()
        for _ in range(NUM_LAYERS):
            flexlora_agg_layer(B_list, A_list, GLOBAL_RANK)
        sync()
        t_flex = round(time.time() - t0, 3)

        # ���� QR-SVD ������������������������������������������������������������������������������������������������������������������������
        sync()
        t0 = time.time()
        for _ in range(NUM_LAYERS):
            qrsvd_agg_layer(B_list, A_list, GLOBAL_RANK)
        sync()
        t_qrsvd = round(time.time() - t0, 3)

        t_flex_per  = round(t_flex  / NUM_LAYERS, 4)
        t_qrsvd_per = round(t_qrsvd / NUM_LAYERS, 4)
        speedup     = round(t_flex / t_qrsvd, 2) if t_qrsvd > 0 else None

        print(f"  FlexLoRA: {t_flex}s ({t_flex_per}s/layer)"
              f"  |  QR-SVD: {t_qrsvd}s ({t_qrsvd_per}s/layer)"
              f"  |  speedup: {speedup}x")

        results[n] = {
            "config": {
                "num_clients": n,
                "ranks":       ranks,
                "global_rank": GLOBAL_RANK,
                "d":           D,
                "num_layers":  NUM_LAYERS,
                "rho":         rho,
            },
            "flexlora_total_s":     t_flex,
            "flexlora_per_layer_s": t_flex_per,
            "qrsvd_total_s":        t_qrsvd,
            "qrsvd_per_layer_s":    t_qrsvd_per,
            "speedup":              speedup,
        }

        with open(out_path, "w") as f:
            json.dump({"base_ranks": BASE_RANKS, "global_rank": GLOBAL_RANK,
                       "d": D, "num_layers": NUM_LAYERS, "results": results},
                      f, indent=2)
        print(f"  (checkpoint saved �� {out_path})")

    print(f"\nAll done. Results in {out_path}")


if __name__ == "__main__":
    run_benchmark()
