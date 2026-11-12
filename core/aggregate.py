"""
Aggregation scripts for Federated Learning on PathVQA.

Supports four modes:
  - fedavg   : simple FedAvg (all clients must have the same LoRA rank)
  - flexlora : FlexLoRA (NeurIPS 2024) �� full-matrix averaging + SVD
  - qrsvd    : QR-SVD �� same math as FlexLoRA but avoids forming the full
               [out��in] matrix; faster for large models
  - hetlora  : HetLoRA (EMNLP 2024) �� zero-padding + sparsity-weighted avg
"""

import argparse
import json
import math
import os
import shutil
import time

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file


# ���� helpers ��������������������������������������������������������������������������������������������������������������������������������������

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
    such that B @ A �� delta_W (best rank-r approximation).

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


# ���� QR + mini-SVD helper ������������������������������������������������������������������������������������������������������������

def _qr_svd_rank(B_list, A_list, global_rank, orig_dtype):
    """
    Rank-global_rank approximation of ��_k B_k @ A_k without forming [out��in].

    Algorithm:
      1. U_big = cat(B_k, dim=1)  [out, ��r_k]
         V_big = cat(A_k.T, dim=1) [in, ��r_k]
      2. Thin QR: U_big = Q_U R_U,  V_big = Q_V R_V
      3. Core:    M = R_U @ R_V.T   [��r_k �� ��r_k]
      4. Mini-SVD on M, back-project top-r components
    For total rank rho much smaller than the matrix dimensions, the cost is
    O((out + in) * rho**2 + rho**3), versus a full-matrix SVD.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B_dev = [b.float().to(device) for b in B_list]
    A_dev = [a.float().to(device) for a in A_list]

    out_dim = B_dev[0].shape[0]
    in_dim  = A_dev[0].shape[1]

    U_big = torch.cat(B_dev, dim=1)                        # [out, ��r_k]
    V_big = torch.cat([a.T for a in A_dev], dim=1)         # [in,  ��r_k]

    Q_U, R_U = torch.linalg.qr(U_big, mode="reduced")     # [out, p], [p, ��r_k]
    Q_V, R_V = torch.linalg.qr(V_big, mode="reduced")     # [in,  p], [p, ��r_k]

    M = R_U @ R_V.T                                         # [p, p]
    U_t, S, Vt_t = torch.linalg.svd(M, full_matrices=False)

    r_actual = min(global_rank, S.shape[0])
    sqrt_S   = S[:r_actual].clamp(min=0.0).sqrt()

    B_g = (Q_U @ U_t[:, :r_actual] * sqrt_S).cpu()                 # [out, r_actual]
    A_g = (sqrt_S.unsqueeze(1) * (Vt_t[:r_actual] @ Q_V.T)).cpu()  # [r_actual, in]

    if r_actual < global_rank:
        A_full = torch.zeros(global_rank, in_dim)
        B_full = torch.zeros(out_dim, global_rank)
        A_full[:r_actual] = A_g
        B_full[:, :r_actual] = B_g
        return A_full.to(orig_dtype), B_full.to(orig_dtype)

    return A_g.to(orig_dtype), B_g.to(orig_dtype)


# ���� FedAvg ����������������������������������������������������������������������������������������������������������������������������������������

def aggregate_lora(client_dirs, output_dir):
    """
    Standard FedAvg: element-wise average of all adapter tensors.
    All clients must share the same LoRA rank and target modules.
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dicts = [_load_adapter(d) for d in client_dirs]
    num_clients = len(state_dicts)
    print(f"FedAvg: averaging {num_clients} clients ��")

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


# ���� FlexLoRA ������������������������������������������������������������������������������������������������������������������������������������

def aggregate_flexlora(client_dirs, output_dir, global_rank, client_weights=None, global_model_path=None):
    """
    FlexLoRA aggregation (NeurIPS 2024):

    For each LoRA layer:
      1. Compute the effective weight delta for each client:
             ��W_k = lora_B_k @ lora_A_k   (raw, before lora_alpha scaling)
      2. Weighted-average across clients:
             ��W_avg = �� w_k * ��W_k / �� w_k
      3. SVD-decompose ��W_avg back to global_rank:
             (A_global, B_global) �� SVD_r(��W_avg)
         so that  B_global @ A_global �� ��W_avg

    The resulting global adapter has rank = global_rank and
    lora_alpha = global_rank * 2  (scaling factor = 2, same as clients).

    Non-lora_A/lora_B keys (e.g. biases, embeddings) are averaged normally.

    Args:
        client_dirs:    list of client adapter directories
        output_dir:     where to write the aggregated global adapter
        global_rank:    target rank of the global adapter (�� max client rank)
        client_weights: optional list of per-client weights (e.g. dataset sizes);
                        defaults to uniform weights
        global_model_path: path to the previous global model for delta-updates
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dicts = [_load_adapter(d) for d in client_dirs]
    num_clients = len(state_dicts)

    client_ranks = [_infer_rank(sd) for sd in state_dicts]
    print(f"FlexLoRA: {num_clients} clients, ranks={client_ranks}, global_rank={global_rank}")

    _agg_start = time.time()

    if client_weights is None:
        client_weights = [1.0] * num_clients
    total_w = sum(client_weights)

    # ���� aggregate lora_A / lora_B pairs via SVD ����
    # Build union of all client LoRA keys (robust to heterogeneous target modules)
    pairs = {}
    for sd in state_dicts:
        pairs.update(_lora_base_keys(sd))
    global_sd = {}

    for bk, keys in sorted(pairs.items()):
        if "A" not in keys or "B" not in keys:
            continue
        a_key, b_key = keys["A"], keys["B"]

        # Compute weighted average of ��W = B @ A across clients.
        delta_W_sum = None
        effective_w = 0.0
        
        global_ref = None
        if global_model_path and os.path.exists(global_model_path):
            global_ref = _load_adapter(global_model_path)

        for w, sd, r_k in zip(client_weights, state_dicts, client_ranks):
            if a_key not in sd or b_key not in sd:
                continue
            A = sd[a_key].float()   # [r_k, in]
            B = sd[b_key].float()   # [out, r_k]
            dW_client = B @ A       # [out, in]
            
            if global_ref is not None and a_key in global_ref and b_key in global_ref:
                A_g_ref = global_ref[a_key].float()
                B_g_ref = global_ref[b_key].float()
                W_g = B_g_ref @ A_g_ref
                orig_dtype = A.dtype
                A_proj, B_proj = _svd_to_rank(W_g, r_k, orig_dtype)
                dW_start = (B_proj.float() @ A_proj.float())
                delta_k = dW_client - dW_start
            else:
                delta_k = dW_client
                
            delta_W_sum = (w * delta_k) if delta_W_sum is None else delta_W_sum + w * delta_k
            effective_w += w

        if delta_W_sum is None:
            continue

        avg_delta = delta_W_sum / effective_w
        
        if global_ref is not None and a_key in global_ref and b_key in global_ref:
            A_g_ref = global_ref[a_key].float()
            B_g_ref = global_ref[b_key].float()
            W_g = B_g_ref @ A_g_ref
            W_new = W_g + avg_delta
        else:
            W_new = avg_delta

        orig_dtype = next(sd[a_key].dtype for sd in state_dicts if a_key in sd)
        A_g, B_g = _svd_to_rank(W_new, global_rank, orig_dtype)
        global_sd[a_key] = A_g
        global_sd[b_key] = B_g

    # ���� average non-lora_A/lora_B keys normally ����
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
                global_sd[key] = tensors[0]   # fallback: use first client

    save_file(global_sd, os.path.join(output_dir, "adapter_model.safetensors"))

    # ���� write adapter_config.json with updated global_rank ����
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

    agg_elapsed = time.time() - _agg_start
    print(f"FlexLoRA aggregation done in {agg_elapsed:.2f}s. Global adapter (rank={global_rank}) saved to {output_dir}")

    # Write timing to output_dir for run_simulation to pick up
    with open(os.path.join(output_dir, "agg_timing.json"), "w") as _f:
        json.dump({"mode": "flexlora", "aggregation_time_s": round(agg_elapsed, 4)}, _f)

    return agg_elapsed


# ���� QR-SVD ����������������������������������������������������������������������������������������������������������������������������������������

def aggregate_qrsvd(client_dirs, output_dir, global_rank, client_weights=None, global_model_path=None):
    """
    QR-SVD aggregation: same math as FlexLoRA but replaces the expensive
    full-matrix SVD with QR + mini-SVD.  Never materialises the [out��in] ��W.

    Delta-update mode (global_model_path provided):
      ��W_new = W_g + ��_k w_k (B_k@A_k ? W_g��r_k)
    The global reference factors are appended to U_big/V_big as extra columns.
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dicts  = [_load_adapter(d) for d in client_dirs]
    num_clients  = len(state_dicts)
    client_ranks = [_infer_rank(sd) for sd in state_dicts]
    print(f"QR-SVD: {num_clients} clients, ranks={client_ranks}, global_rank={global_rank}")

    _agg_start = time.time()

    if client_weights is None:
        client_weights = [1.0] * num_clients
    total_w = sum(client_weights)
    norm_w  = [w / total_w for w in client_weights]

    pairs = {}
    for sd in state_dicts:
        pairs.update(_lora_base_keys(sd))
    global_sd = {}

    global_ref = None
    if global_model_path and os.path.exists(global_model_path):
        global_ref = _load_adapter(global_model_path)

    for bk, keys in sorted(pairs.items()):
        if "A" not in keys or "B" not in keys:
            continue
        a_key, b_key = keys["A"], keys["B"]

        # Pre-compute QR-SVD of global reference (once per layer, avoids [out��in])
        B_g_svd = A_g_svd = None
        if global_ref is not None and a_key in global_ref and b_key in global_ref:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            B_gref = global_ref[b_key].float().to(device)
            A_gref = global_ref[a_key].float().to(device)
            Q_Ug, R_Ug = torch.linalg.qr(B_gref,   mode="reduced")
            Q_Vg, R_Vg = torch.linalg.qr(A_gref.T, mode="reduced")
            Mg = R_Ug @ R_Vg.T
            Utg, Sg, Vttg = torch.linalg.svd(Mg, full_matrices=False)
            sqrt_Sg = Sg.clamp(min=0.0).sqrt()
            B_g_svd = ((Q_Ug @ Utg) * sqrt_Sg).cpu()
            A_g_svd = (sqrt_Sg.unsqueeze(1) * (Vttg @ Q_Vg.T)).cpu()

        B_factors = []
        A_factors = []

        for w, sd, r_k in zip(norm_w, state_dicts, client_ranks):
            if a_key not in sd or b_key not in sd:
                continue
            sw = math.sqrt(w)
            B_factors.append(sd[b_key].float() * sw)
            A_factors.append(sd[a_key].float() * sw)
            if B_g_svd is not None:
                B_factors.append(-B_g_svd[:, :r_k] * sw)
                A_factors.append( A_g_svd[:r_k, :] * sw)

        if not B_factors:
            continue

        if B_g_svd is not None:
            B_factors.append(B_g_svd)
            A_factors.append(A_g_svd)

        orig_dtype = next(sd[a_key].dtype for sd in state_dicts if a_key in sd)
        A_g, B_g = _qr_svd_rank(B_factors, A_factors, global_rank, orig_dtype)
        global_sd[a_key] = A_g
        global_sd[b_key] = B_g

    # Average non-LoRA keys normally
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
    print(f"QR-SVD aggregation done in {agg_elapsed:.2f}s. Global adapter (rank={global_rank}) saved to {output_dir}")

    with open(os.path.join(output_dir, "agg_timing.json"), "w") as _f:
        json.dump({"mode": "qrsvd", "aggregation_time_s": round(agg_elapsed, 4)}, _f)

    return agg_elapsed


# ���� HetLoRA ��������������������������������������������������������������������������������������������������������������������������������������

def aggregate_hetlora(client_dirs, output_dir, global_rank):
    """
    HetLoRA aggregation (EMNLP 2024):
    Zero-padding + sparsity-weighted direct LoRA module averaging.

    Key difference from FlexLoRA: we aggregate LoRA modules (B, A) directly
    with zero-padding, NOT by reconstructing ��W = B@A. This preserves
    cross-client information (see paper Section 3.4, Table 3).

    Algorithm:
      1. Compute per-client sparsity weight:
         p_k = ||��W_k||_F / ��||��W_j||_F   where ��W_k = B_k @ A_k
         (global weight, averaged across all LoRA layers)
      2. Zero-pad each client's LoRA modules to r_max:
         B_k �� ?^{d��r_k} �� ?^{d��r_max}  (right-pad with zeros)
         A_k �� ?^{r_k��l} �� ?^{r_max��l}  (bottom-pad with zeros)
      3. Weighted average of padded modules:
         B_global = �� p_k * B_k_padded
         A_global = �� p_k * A_k_padded
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dicts = [_load_adapter(d) for d in client_dirs]
    num_clients = len(state_dicts)

    client_ranks = [_infer_rank(sd) for sd in state_dicts]
    print(f"HetLoRA: aggregating {num_clients} clients, ranks={client_ranks}, "
          f"global_rank={global_rank} ��")

    # ���� Step 1: compute global sparsity weight per client ����
    # Sum ||��W_k||_F across all LoRA layers for each client
    client_norm_sums = [0.0] * num_clients

    # Collect all LoRA pairs (use union of all client keys for robustness)
    all_pairs = {}
    for sd in state_dicts:
        all_pairs.update(_lora_base_keys(sd))

    for bk, keys in sorted(all_pairs.items()):
        if "A" not in keys or "B" not in keys:
            continue
        a_key, b_key = keys["A"], keys["B"]

        for c_idx, sd in enumerate(state_dicts):
            if a_key not in sd or b_key not in sd:
                continue
            A = sd[a_key].float()   # [r_k, in]
            B = sd[b_key].float()   # [out, r_k]
            dW = B @ A              # [out, in]
            client_norm_sums[c_idx] += torch.norm(dW, p='fro').item()

    total_norm = sum(client_norm_sums)
    if total_norm < 1e-9:
        sparsity_weights = [1.0 / num_clients] * num_clients
    else:
        sparsity_weights = [n / total_norm for n in client_norm_sums]

    print(f"  Sparsity weights: {[f'{w:.4f}' for w in sparsity_weights]}")

    # ���� Step 2 & 3: zero-pad and weighted-average LoRA modules ����
    global_sd = {}
    r_max = global_rank  # should be max(client_ranks) or the specified global_rank

    for bk, keys in sorted(all_pairs.items()):
        if "A" not in keys or "B" not in keys:
            continue
        a_key, b_key = keys["A"], keys["B"]

        # Determine layer dimensions from first available client
        ref_sd = None
        for sd in state_dicts:
            if a_key in sd and b_key in sd:
                ref_sd = sd
                break
        if ref_sd is None:
            continue

        in_dim = ref_sd[a_key].shape[1]    # l
        out_dim = ref_sd[b_key].shape[0]   # d
        orig_dtype = ref_sd[a_key].dtype

        A_avg = torch.zeros(r_max, in_dim, dtype=torch.float32)
        B_avg = torch.zeros(out_dim, r_max, dtype=torch.float32)
        weight_sum = torch.zeros(r_max, dtype=torch.float32)

        for c_idx, (sd, w) in enumerate(zip(state_dicts, sparsity_weights)):
            if a_key not in sd or b_key not in sd:
                continue

            A_k = sd[a_key].float()   # [r_k, in]
            B_k = sd[b_key].float()   # [out, r_k]
            r_k = A_k.shape[0]

            if r_k > r_max:
                raise ValueError(
                    f"Client {c_idx} rank {r_k} > global_rank {r_max} for layer {bk}"
                )

            A_avg[:r_k, :] += w * A_k
            B_avg[:, :r_k] += w * B_k
            weight_sum[:r_k] += w

        weight_sum = weight_sum.clamp(min=1e-9)
        A_avg = A_avg / weight_sum.unsqueeze(1)
        B_avg = B_avg / weight_sum.unsqueeze(0)

        global_sd[a_key] = A_avg.to(orig_dtype)
        global_sd[b_key] = B_avg.to(orig_dtype)

    # ���� Average non-LoRA keys normally (uniform average) ����
    lora_keys = {k for p in all_pairs.values() for k in p.values()}
    all_other_keys = set()
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

    # ���� write adapter_config.json with global_rank ����
    # Find a client config to use as base (prefer the highest-rank client)
    cfg_src = None
    for d in client_dirs:
        cfg_candidate = os.path.join(d, "adapter_config.json")
        if os.path.exists(cfg_candidate):
            cfg_src = cfg_candidate
            break

    if cfg_src:
        with open(cfg_src) as f:
            cfg = json.load(f)
        cfg["r"] = global_rank
        cfg["lora_alpha"] = global_rank * 2
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    for fn in ["tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]:
        for d in client_dirs:
            src = os.path.join(d, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(output_dir, fn))
                break

    print(f"HetLoRA aggregation done. Global adapter (rank={global_rank}) saved to {output_dir}")


# ���� project global �� local rank (for client initialisation) ������������������������������������

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

    print(f"  Projected global adapter (rank={global_rank}) �� local adapter (rank={local_rank})")


# ���� CLI ����������������������������������������������������������������������������������������������������������������������������������������������

def main():
    parser = argparse.ArgumentParser(description="Aggregate LoRA adapters (FedAvg or FlexLoRA)")
    parser.add_argument("--client_dirs", nargs="+", required=True,
                        help="Directories containing per-client adapter weights")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the aggregated global adapter")
    parser.add_argument("--mode", type=str, default="fedavg",
                        choices=["fedavg", "flexlora", "qrsvd", "hetlora"],
                        help="Aggregation mode (default: fedavg)")
    parser.add_argument("--global_rank", type=int, default=64,
                        help="Global LoRA rank for FlexLoRA (default: 64)")
    parser.add_argument("--client_weights", type=float, nargs="*", default=None,
                        help="Optional per-client weights for FlexLoRA (e.g. dataset sizes). "
                             "Defaults to uniform weighting.")
    parser.add_argument("--global_model_path", type=str, default=None,
                        help="Optional path to previous global model for FlexLoRA Delta update.")
    args = parser.parse_args()

    if args.mode == "hetlora":
        aggregate_hetlora(
            args.client_dirs,
            args.output_dir,
            args.global_rank,
        )
    elif args.mode == "flexlora":
        aggregate_flexlora(
            args.client_dirs,
            args.output_dir,
            args.global_rank,
            client_weights=args.client_weights,
            global_model_path=args.global_model_path,
        )
    elif args.mode == "qrsvd":
        aggregate_qrsvd(
            args.client_dirs,
            args.output_dir,
            args.global_rank,
            client_weights=args.client_weights,
            global_model_path=args.global_model_path,
        )
    else:
        aggregate_lora(args.client_dirs, args.output_dir)


if __name__ == "__main__":
    main()
