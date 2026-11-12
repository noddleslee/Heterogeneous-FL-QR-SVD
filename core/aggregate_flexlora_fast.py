"""
aggregate_fast.py — optimised FlexLoRA aggregation (drop-in replacement).

Two fixes over aggregate.py::aggregate_flexlora:
  1. _load_adapter(global_model_path) moved before the 196-layer loop
     (original calls it once per layer = 196 disk reads per round).
  2. W_g = B_g_ref @ A_g_ref and its full SVD are computed once per layer;
     each client's projection is obtained by truncating the shared decomposition
     (original recomputes the same full SVD once per client per layer = 4×).

Everything else — delta-update logic, save format, CLI — is identical to
aggregate.py so run_simulation.py can be pointed at this file with no other
changes.
"""

import argparse
import json
import math
import os
import shutil
import time

import torch
from safetensors.torch import save_file

# reuse helpers from aggregate.py (no duplication)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from aggregate import (
    _load_adapter,
    _infer_rank,
    _lora_base_keys,
    _svd_to_rank,
    _qr_svd_rank,
    aggregate_lora,
    aggregate_qrsvd,
    aggregate_hetlora,
    project_adapter_to_rank,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def aggregate_flexlora_fast(
    client_dirs,
    output_dir,
    global_rank,
    client_weights=None,
    global_model_path=None,
):
    """
    FlexLoRA aggregation with two performance fixes:
      - global reference adapter loaded once (not once per layer)
      - W_g SVD computed once per layer (not once per client per layer)
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dicts  = [_load_adapter(d) for d in client_dirs]
    num_clients  = len(state_dicts)
    client_ranks = [_infer_rank(sd) for sd in state_dicts]
    print(f"FlexLoRA-fast: {num_clients} clients, ranks={client_ranks}, global_rank={global_rank}")

    _agg_start = time.time()

    if client_weights is None:
        client_weights = [1.0] * num_clients
    total_w = sum(client_weights)

    # ── FIX 1: load global reference once, before the layer loop ──
    global_ref = None
    if global_model_path and os.path.exists(global_model_path):
        global_ref = _load_adapter(global_model_path)

    pairs = {}
    for sd in state_dicts:
        pairs.update(_lora_base_keys(sd))
    global_sd = {}

    for bk, keys in sorted(pairs.items()):
        if "A" not in keys or "B" not in keys:
            continue
        a_key, b_key = keys["A"], keys["B"]

        # ── FIX 2: compute W_g and its full SVD once per layer ──
        W_g      = None
        U_g      = None
        S_g      = None
        Vh_g     = None
        has_ref  = (
            global_ref is not None
            and a_key in global_ref
            and b_key in global_ref
        )
        if has_ref:
            A_g_ref = global_ref[a_key].float().to(DEVICE)
            B_g_ref = global_ref[b_key].float().to(DEVICE)
            W_g = B_g_ref @ A_g_ref                              # [out, in] — once
            U_g, S_g, Vh_g = torch.linalg.svd(W_g, full_matrices=False)  # once

        delta_W_sum = None
        effective_w = 0.0
        orig_dtype  = next(sd[a_key].dtype for sd in state_dicts if a_key in sd)

        for w, sd, r_k in zip(client_weights, state_dicts, client_ranks):
            if a_key not in sd or b_key not in sd:
                continue
            A = sd[a_key].float()
            B = sd[b_key].float()
            dW_client = B @ A

            if has_ref:
                # truncate shared SVD to this client's r_k (no extra full SVD)
                r_act   = min(r_k, S_g.shape[0])
                sqrt_S  = S_g[:r_act].clamp(min=0.0).sqrt()
                B_proj  = (U_g[:, :r_act] * sqrt_S).cpu()       # [out, r_act]
                A_proj  = (Vh_g[:r_act] * sqrt_S.unsqueeze(1)).cpu()  # [r_act, in]
                if r_act < r_k:
                    pad_A = torch.zeros(r_k, A_proj.shape[1])
                    pad_B = torch.zeros(B_proj.shape[0], r_k)
                    pad_A[:r_act] = A_proj
                    pad_B[:, :r_act] = B_proj
                    A_proj, B_proj = pad_A, pad_B
                dW_start = B_proj.float() @ A_proj.float()
                delta_k  = dW_client - dW_start
            else:
                delta_k = dW_client

            delta_W_sum = (w * delta_k) if delta_W_sum is None else delta_W_sum + w * delta_k
            effective_w += w

        if delta_W_sum is None:
            continue

        avg_delta = delta_W_sum / effective_w

        if has_ref:
            W_new = W_g.cpu() + avg_delta
        else:
            W_new = avg_delta

        A_g, B_g = _svd_to_rank(W_new, global_rank, orig_dtype)
        global_sd[a_key] = A_g
        global_sd[b_key] = B_g

    # ── average non-lora keys normally ──
    lora_keys = {k for p in pairs.values() for k in p.values()}
    all_other_keys: set = set()
    for sd in state_dicts:
        all_other_keys.update(k for k in sd if k not in lora_keys)
    for key in sorted(all_other_keys):
        tensors = [sd[key] for sd in state_dicts if key in sd]
        if len(tensors) == num_clients:
            try:
                global_sd[key] = torch.stack(tensors).mean(dim=0)
            except Exception:
                global_sd[key] = tensors[0]

    save_file(global_sd, os.path.join(output_dir, "adapter_model.safetensors"))

    cfg_src = os.path.join(client_dirs[0], "adapter_config.json")
    if os.path.exists(cfg_src):
        with open(cfg_src) as f:
            cfg = json.load(f)
        cfg["r"]          = global_rank
        cfg["lora_alpha"] = global_rank * 2
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    for fn in ["tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]:
        src = os.path.join(client_dirs[0], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fn))

    agg_elapsed = time.time() - _agg_start
    print(f"FlexLoRA-fast aggregation done in {agg_elapsed:.2f}s. "
          f"Global adapter (rank={global_rank}) saved to {output_dir}")

    with open(os.path.join(output_dir, "agg_timing.json"), "w") as _f:
        json.dump({"mode": "flexlora_fast", "aggregation_time_s": round(agg_elapsed, 4)}, _f)

    return agg_elapsed


# ── CLI (same interface as aggregate.py) ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LoRA adapters — optimised FlexLoRA + standard modes"
    )
    parser.add_argument("--client_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir",  type=str,   required=True)
    parser.add_argument("--mode",        type=str,   default="flexlora",
                        choices=["fedavg", "flexlora", "qrsvd", "hetlora"])
    parser.add_argument("--global_rank", type=int,   default=64)
    parser.add_argument("--client_weights", type=float, nargs="*", default=None)
    parser.add_argument("--global_model_path", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "flexlora":
        aggregate_flexlora_fast(
            args.client_dirs, args.output_dir, args.global_rank,
            client_weights=args.client_weights,
            global_model_path=args.global_model_path,
        )
    elif args.mode == "qrsvd":
        aggregate_qrsvd(
            args.client_dirs, args.output_dir, args.global_rank,
            client_weights=args.client_weights,
            global_model_path=args.global_model_path,
        )
    elif args.mode == "hetlora":
        aggregate_hetlora(args.client_dirs, args.output_dir, args.global_rank)
    else:
        aggregate_lora(args.client_dirs, args.output_dir)


if __name__ == "__main__":
    main()
