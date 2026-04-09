"""
Aggregation scripts for Federated Learning on PathVQA.

Supports two modes:
  - fedavg   : simple FedAvg (all clients must have the same LoRA rank)
  - flexlora : FlexLoRA (NeurIPS 2024) — handles heterogeneous client ranks
               via full-matrix averaging + SVD re-decomposition
"""

import argparse
import json
import os
import shutil

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_adapter(adapter_dir: str) -> dict:
    st = os.path.join(adapter_dir, "adapter_model.safetensors")
    bn = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(st):
        return load_file(st)
    if os.path.exists(bn):
        return torch.load(bn, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def _infer_rank(state_dict: dict) -> int:
    for key, val in state_dict.items():
        if ".lora_A.weight" in key:
            return val.shape[0]   # lora_A: [r, in_features]
    raise ValueError("Cannot infer rank: no lora_A.weight key found")


def _lora_base_keys(state_dict: dict) -> dict:
    """Return {base_key: {'A': a_key, 'B': b_key}} for every lora pair."""
    pairs = {}
    for key in state_dict:
        if ".lora_A.weight" in key:
            bk = key.replace(".lora_A.weight", "")
            pairs.setdefault(bk, {})["A"] = key
        elif ".lora_B.weight" in key:
            bk = key.replace(".lora_B.weight", "")
            pairs.setdefault(bk, {})["B"] = key
    return pairs


def _svd_to_rank(delta_W: torch.Tensor, rank: int, orig_dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose delta_W (shape [out, in]) via truncated SVD at the given rank.

    Returns (A, B) where:
      A : [rank, in]   (lora_A.weight)
      B : [out, rank]  (lora_B.weight)
    such that B @ A ≈ delta_W (best rank-r approximation).

    The split is symmetric: B = U * sqrt(S), A = sqrt(S) * Vh,
    so B @ A = U @ diag(S) @ Vh (the standard SVD approximation).

    If min(out, in) < rank (layer too small for full rank), the output is
    zero-padded to always return tensors of exactly [rank, in] and [out, rank].
    This ensures shape consistency with adapter_config.json (r = global_rank).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = delta_W.float().to(device)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)   # economy SVD
    r_actual = min(rank, S.shape[0])   # may be < rank for small layers

    sqrt_S = S[:r_actual].clamp(min=0.0).sqrt()
    B_r = (U[:, :r_actual] * sqrt_S.unsqueeze(0)).contiguous().cpu()   # [out, r_actual]
    A_r = (Vh[:r_actual, :] * sqrt_S.unsqueeze(1)).contiguous().cpu()  # [r_actual, in]

    if r_actual < rank:
        # Zero-pad so shapes always match adapter_config.json (r = rank)
        A_full = torch.zeros(rank, W.shape[1], dtype=A_r.dtype)
        B_full = torch.zeros(W.shape[0], rank, dtype=B_r.dtype)
        A_full[:r_actual, :] = A_r
        B_full[:, :r_actual] = B_r
        return A_full.to(orig_dtype), B_full.to(orig_dtype)

    return A_r.to(orig_dtype), B_r.to(orig_dtype)


# ── FedAvg ────────────────────────────────────────────────────────────────────

def aggregate_lora(client_dirs, output_dir):
    """
    Standard FedAvg: element-wise average of all adapter tensors.
    All clients must share the same LoRA rank and target modules.
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dicts = [_load_adapter(d) for d in client_dirs]
    num_clients = len(state_dicts)
    print(f"FedAvg: averaging {num_clients} clients …")

    global_sd = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        global_sd[key] = torch.stack(tensors).mean(dim=0)

    save_file(global_sd, os.path.join(output_dir, "adapter_model.safetensors"))

    for fn in ["adapter_config.json", "tokenizer_config.json",
               "special_tokens_map.json", "tokenizer.json"]:
        src = os.path.join(client_dirs[0], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fn))

    print(f"FedAvg done. Global model saved to {output_dir}")


# ── FlexLoRA ──────────────────────────────────────────────────────────────────

def aggregate_flexlora(client_dirs, output_dir, global_rank, client_weights=None):
    """
    FlexLoRA aggregation (NeurIPS 2024):

    For each LoRA layer:
      1. Compute the effective weight delta for each client:
             ΔW_k = lora_B_k @ lora_A_k   (raw, before lora_alpha scaling)
      2. Weighted-average across clients:
             ΔW_avg = Σ w_k * ΔW_k / Σ w_k
      3. SVD-decompose ΔW_avg back to global_rank:
             (A_global, B_global) ← SVD_r(ΔW_avg)
         so that  B_global @ A_global ≈ ΔW_avg

    The resulting global adapter has rank = global_rank and
    lora_alpha = global_rank * 2  (scaling factor = 2, same as clients).

    Non-lora_A/lora_B keys (e.g. biases, embeddings) are averaged normally.

    Args:
        client_dirs:    list of client adapter directories
        output_dir:     where to write the aggregated global adapter
        global_rank:    target rank of the global adapter (≥ max client rank)
        client_weights: optional list of per-client weights (e.g. dataset sizes);
                        defaults to uniform weights
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dicts = [_load_adapter(d) for d in client_dirs]
    num_clients = len(state_dicts)

    client_ranks = [_infer_rank(sd) for sd in state_dicts]
    print(f"FlexLoRA: {num_clients} clients, ranks={client_ranks}, global_rank={global_rank}")

    if client_weights is None:
        client_weights = [1.0] * num_clients
    total_w = sum(client_weights)

    # ── aggregate lora_A / lora_B pairs via SVD ──
    pairs = _lora_base_keys(state_dicts[0])
    global_sd = {}

    for bk, keys in sorted(pairs.items()):
        if "A" not in keys or "B" not in keys:
            continue
        a_key, b_key = keys["A"], keys["B"]

        # Compute weighted average of ΔW = B @ A across clients
        delta_W_sum = None
        for w, sd in zip(client_weights, state_dicts):
            if a_key not in sd or b_key not in sd:
                continue
            A = sd[a_key].float()   # [r_k, in]
            B = sd[b_key].float()   # [out, r_k]
            dW = B @ A              # [out, in]
            delta_W_sum = (w * dW) if delta_W_sum is None else delta_W_sum + w * dW

        if delta_W_sum is None:
            continue

        avg_dW = delta_W_sum / total_w

        orig_dtype = state_dicts[0][a_key].dtype
        A_g, B_g = _svd_to_rank(avg_dW, global_rank, orig_dtype)
        global_sd[a_key] = A_g
        global_sd[b_key] = B_g

    # ── average non-lora_A/lora_B keys normally ──
    lora_keys = {k for p in pairs.values() for k in p.values()}
    other_keys = [k for k in state_dicts[0] if k not in lora_keys]
    # Only average keys that exist and have matching shapes in all clients
    for key in other_keys:
        tensors = [sd[key] for sd in state_dicts if key in sd]
        if len(tensors) == num_clients:
            try:
                global_sd[key] = torch.stack(tensors).mean(dim=0)
            except Exception:
                global_sd[key] = tensors[0]   # fallback: use first client

    save_file(global_sd, os.path.join(output_dir, "adapter_model.safetensors"))

    # ── write adapter_config.json with updated global_rank ──
    cfg_src = os.path.join(client_dirs[0], "adapter_config.json")
    if os.path.exists(cfg_src):
        with open(cfg_src) as f:
            cfg = json.load(f)
        cfg["r"] = global_rank
        cfg["lora_alpha"] = global_rank * 2
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    for fn in ["tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]:
        src = os.path.join(client_dirs[0], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fn))

    print(f"FlexLoRA aggregation done. Global adapter (rank={global_rank}) saved to {output_dir}")


# ── project global → local rank (for client initialisation) ──────────────────

def project_adapter_to_rank(
    global_adapter_dir: str,
    local_rank: int,
    output_dir: str,
) -> None:
    """
    Project a global adapter (rank=global_rank) down to local_rank via SVD
    and save to output_dir.  Used by clients to initialise their local adapter
    from the global model.

    If local_rank >= global_rank the adapter is copied unchanged.
    """
    global_sd = _load_adapter(global_adapter_dir)
    global_rank = _infer_rank(global_sd)

    os.makedirs(output_dir, exist_ok=True)

    if local_rank >= global_rank:
        # No projection needed: copy everything as-is
        save_file(global_sd, os.path.join(output_dir, "adapter_model.safetensors"))
        cfg_src = os.path.join(global_adapter_dir, "adapter_config.json")
        if os.path.exists(cfg_src):
            shutil.copy(cfg_src, os.path.join(output_dir, "adapter_config.json"))
        print(f"  local_rank ({local_rank}) >= global_rank ({global_rank}): copied adapter directly")
        return

    pairs = _lora_base_keys(global_sd)
    projected_sd = {}

    for bk, keys in sorted(pairs.items()):
        if "A" not in keys or "B" not in keys:
            continue
        a_key, b_key = keys["A"], keys["B"]

        A_g = global_sd[a_key].float()   # [global_rank, in]
        B_g = global_sd[b_key].float()   # [out, global_rank]
        delta_W = B_g @ A_g              # [out, in]

        orig_dtype = global_sd[a_key].dtype
        A_l, B_l = _svd_to_rank(delta_W, local_rank, orig_dtype)
        projected_sd[a_key] = A_l.contiguous()
        projected_sd[b_key] = B_l.contiguous()

    # keep non-lora_A/lora_B keys unchanged
    lora_keys = {k for p in pairs.values() for k in p.values()}
    for k, v in global_sd.items():
        if k not in lora_keys:
            projected_sd[k] = v

    save_file(projected_sd, os.path.join(output_dir, "adapter_model.safetensors"))

    # write adapter_config.json with local_rank
    cfg_src = os.path.join(global_adapter_dir, "adapter_config.json")
    if os.path.exists(cfg_src):
        with open(cfg_src) as f:
            cfg = json.load(f)
        cfg["r"] = local_rank
        cfg["lora_alpha"] = local_rank * 2
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"  Projected global adapter (rank={global_rank}) → local adapter (rank={local_rank})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate LoRA adapters (FedAvg or FlexLoRA)")
    parser.add_argument("--client_dirs", nargs="+", required=True,
                        help="Directories containing per-client adapter weights")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the aggregated global adapter")
    parser.add_argument("--mode", type=str, default="fedavg",
                        choices=["fedavg", "flexlora"],
                        help="Aggregation mode (default: fedavg)")
    parser.add_argument("--global_rank", type=int, default=64,
                        help="Global LoRA rank for FlexLoRA (default: 64)")
    parser.add_argument("--client_weights", type=float, nargs="*", default=None,
                        help="Optional per-client weights for FlexLoRA (e.g. dataset sizes). "
                             "Defaults to uniform weighting.")
    args = parser.parse_args()

    if args.mode == "flexlora":
        aggregate_flexlora(
            args.client_dirs,
            args.output_dir,
            args.global_rank,
            client_weights=args.client_weights,
        )
    else:
        aggregate_lora(args.client_dirs, args.output_dir)


if __name__ == "__main__":
    main()
