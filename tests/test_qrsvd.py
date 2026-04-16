"""
Tests: QR-SVD aggregation produces the same rank-r approximation as pure SVD.

Both _svd_to_rank and _qr_svd_rank compute the best rank-r approximation of the
same matrix (by Eckart-Young, this is unique), so their reconstructions B @ A
must agree up to floating-point precision, and their singular values must match.

Run with:
    pytest tests/test_qrsvd.py -v
"""

import json
import math
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# adjust import path so tests can be run from repo root or tests/ dir
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aggregate import (
    _svd_to_rank,
    _qr_svd_rank,
    aggregate_flexlora,
    aggregate_qrsvd,
)

# ---------------------------------------------------------------------------
# helpers

TOL_REL  = 1e-3   # relative Frobenius-norm tolerance for reconstructed matrix
TOL_SV   = 1e-3   # relative tolerance for leading singular values


def _make_clients(out, in_, ranks, seed=42):
    """Return lists of (B_k, A_k) tensors with given heterogeneous ranks."""
    torch.manual_seed(seed)
    Bs = [torch.randn(out, r) for r in ranks]
    As = [torch.randn(r, in_) for r in ranks]
    return Bs, As


def _weighted_sum(Bs, As, weights):
    """Explicitly form Σ w_k · B_k @ A_k (used as ground truth)."""
    total = sum(weights)
    W = sum(w / total * (B @ A) for w, B, A in zip(weights, Bs, As))
    return W


def _svd_result(W, rank):
    """Reference: full-matrix truncated SVD."""
    return _svd_to_rank(W, rank, torch.float32)


def _qrsvd_result(Bs, As, weights, rank):
    """QR-SVD on weight-scaled factor lists."""
    total = sum(weights)
    sw = [math.sqrt(w / total) for w in weights]
    B_scaled = [B * s for B, s in zip(Bs, sw)]
    A_scaled = [A * s for A, s in zip(As, sw)]
    return _qr_svd_rank(B_scaled, A_scaled, rank, torch.float32)


def _reconstruction_rel_err(B1, A1, B2, A2, W_ref):
    W1 = B1.float() @ A1.float()
    W2 = B2.float() @ A2.float()
    return ((W1 - W2).norm() / W_ref.norm()).item()


def _sv_rel_err(B1, A1, B2, A2):
    _, S1, _ = torch.linalg.svd(B1.float() @ A1.float(), full_matrices=False)
    _, S2, _ = torch.linalg.svd(B2.float() @ A2.float(), full_matrices=False)
    n = min(S1.shape[0], S2.shape[0])
    return ((S1[:n] - S2[:n]).abs() / S1[0].clamp(min=1e-9)).max().item()


# ---------------------------------------------------------------------------
# 1. Basic equivalence: heterogeneous ranks, uniform weights

@pytest.mark.parametrize("out,in_,ranks,global_rank", [
    (128, 128, [4, 16, 32, 64], 64),
    (256, 256, [4, 16, 32, 64], 64),
    (512, 512, [4, 16, 32, 64], 64),
    (256, 512, [4, 16, 32, 64], 64),   # non-square
    (512, 256, [4, 16, 32, 64], 64),   # non-square, transposed
])
def test_basic_equivalence(out, in_, ranks, global_rank):
    Bs, As = _make_clients(out, in_, ranks)
    weights = [1.0] * len(ranks)
    W = _weighted_sum(Bs, As, weights)

    A_svd, B_svd = _svd_result(W, global_rank)
    A_qr,  B_qr  = _qrsvd_result(Bs, As, weights, global_rank)

    rel_err = _reconstruction_rel_err(B_svd, A_svd, B_qr, A_qr, W)
    sv_err  = _sv_rel_err(B_svd, A_svd, B_qr, A_qr)

    assert rel_err < TOL_REL, f"Reconstruction error {rel_err:.2e} ≥ {TOL_REL}"
    assert sv_err  < TOL_SV,  f"Singular value error {sv_err:.2e} ≥ {TOL_SV}"


# ---------------------------------------------------------------------------
# 2. Non-uniform weights

@pytest.mark.parametrize("weights", [
    [1.0, 2.0, 3.0, 4.0],
    [0.1, 0.9, 0.5, 0.5],
    [10.0, 1.0, 1.0, 1.0],
])
def test_nonuniform_weights(weights):
    out, in_, ranks, global_rank = 256, 256, [4, 16, 32, 64], 64
    Bs, As = _make_clients(out, in_, ranks)
    W = _weighted_sum(Bs, As, weights)

    A_svd, B_svd = _svd_result(W, global_rank)
    A_qr,  B_qr  = _qrsvd_result(Bs, As, weights, global_rank)

    rel_err = _reconstruction_rel_err(B_svd, A_svd, B_qr, A_qr, W)
    assert rel_err < TOL_REL, f"Weighted error {rel_err:.2e} ≥ {TOL_REL}"


# ---------------------------------------------------------------------------
# 3. Homogeneous ranks (all clients same rank)

def test_homogeneous_ranks():
    out, in_, ranks, global_rank = 256, 256, [16, 16, 16, 16], 16
    Bs, As = _make_clients(out, in_, ranks)
    weights = [1.0] * len(ranks)
    W = _weighted_sum(Bs, As, weights)

    A_svd, B_svd = _svd_result(W, global_rank)
    A_qr,  B_qr  = _qrsvd_result(Bs, As, weights, global_rank)

    rel_err = _reconstruction_rel_err(B_svd, A_svd, B_qr, A_qr, W)
    assert rel_err < TOL_REL, f"Homogeneous error {rel_err:.2e} ≥ {TOL_REL}"


# ---------------------------------------------------------------------------
# 4. Single client

def test_single_client():
    out, in_, rank, global_rank = 256, 256, 32, 64
    torch.manual_seed(0)
    B = torch.randn(out, rank)
    A = torch.randn(rank, in_)
    W = B @ A

    A_svd, B_svd = _svd_result(W, global_rank)
    A_qr,  B_qr  = _qr_svd_rank([B], [A], global_rank, torch.float32)

    rel_err = _reconstruction_rel_err(B_svd, A_svd, B_qr, A_qr, W)
    assert rel_err < TOL_REL, f"Single-client error {rel_err:.2e} ≥ {TOL_REL}"


# ---------------------------------------------------------------------------
# 5. Zero-padding: Σr_k < global_rank
#    When there are fewer combined rank slots than global_rank, both methods
#    must zero-pad and the non-zero portion must match.

def test_zero_padding():
    out, in_, ranks, global_rank = 256, 256, [4, 8], 64  # Σr_k=12 << 64
    Bs, As = _make_clients(out, in_, ranks)
    weights = [1.0] * len(ranks)
    W = _weighted_sum(Bs, As, weights)

    A_svd, B_svd = _svd_result(W, global_rank)
    A_qr,  B_qr  = _qrsvd_result(Bs, As, weights, global_rank)

    # Shapes must be exactly [global_rank, in] and [out, global_rank]
    assert A_svd.shape == (global_rank, in_)
    assert B_svd.shape == (out, global_rank)
    assert A_qr.shape  == (global_rank, in_)
    assert B_qr.shape  == (out, global_rank)

    rel_err = _reconstruction_rel_err(B_svd, A_svd, B_qr, A_qr, W)
    assert rel_err < TOL_REL, f"Zero-pad error {rel_err:.2e} ≥ {TOL_REL}"


# ---------------------------------------------------------------------------
# 6. Output shapes are always correct

@pytest.mark.parametrize("out,in_,ranks,global_rank", [
    (64,  128, [4, 8],        16),
    (128, 64,  [4, 16, 32],   32),
    (256, 256, [4, 16, 32, 64], 64),
])
def test_output_shapes(out, in_, ranks, global_rank):
    Bs, As = _make_clients(out, in_, ranks)
    sw = math.sqrt(1.0 / len(ranks))
    A_g, B_g = _qr_svd_rank([B * sw for B in Bs], [A * sw for A in As],
                             global_rank, torch.float32)
    assert A_g.shape == (global_rank, in_),  f"A shape {A_g.shape}"
    assert B_g.shape == (out, global_rank),  f"B shape {B_g.shape}"


# ---------------------------------------------------------------------------
# 7. End-to-end: aggregate_qrsvd vs aggregate_flexlora (no delta mode)

def _write_adapter(directory, B_dict, A_dict, rank, out_dim, in_dim):
    """Write a minimal adapter directory for testing."""
    os.makedirs(directory, exist_ok=True)
    sd = {}
    for key in B_dict:
        sd[f"{key}.lora_B.weight"] = B_dict[key]
        sd[f"{key}.lora_A.weight"] = A_dict[key]
    save_file(sd, os.path.join(directory, "adapter_model.safetensors"))
    cfg = {"r": rank, "lora_alpha": rank * 2, "target_modules": list(B_dict.keys())}
    with open(os.path.join(directory, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)


def test_end_to_end_no_delta():
    torch.manual_seed(7)
    out, in_, ranks, global_rank = 128, 128, [4, 16, 32, 64], 64
    keys = ["layer0", "layer1"]

    with tempfile.TemporaryDirectory() as tmp:
        client_dirs = []
        for k, r in enumerate(ranks):
            d = os.path.join(tmp, f"client_{k}")
            B_dict = {key: torch.randn(out, r) for key in keys}
            A_dict = {key: torch.randn(r, in_) for key in keys}
            _write_adapter(d, B_dict, A_dict, r, out, in_)
            client_dirs.append(d)

        out_flex = os.path.join(tmp, "global_flex")
        out_qr   = os.path.join(tmp, "global_qr")

        aggregate_flexlora(client_dirs, out_flex, global_rank)
        aggregate_qrsvd(client_dirs, out_qr,   global_rank)

        from safetensors.torch import load_file
        sd_flex = load_file(os.path.join(out_flex, "adapter_model.safetensors"))
        sd_qr   = load_file(os.path.join(out_qr,   "adapter_model.safetensors"))

        for key in keys:
            B_flex = sd_flex[f"{key}.lora_B.weight"].float()
            A_flex = sd_flex[f"{key}.lora_A.weight"].float()
            B_qr   = sd_qr  [f"{key}.lora_B.weight"].float()
            A_qr   = sd_qr  [f"{key}.lora_A.weight"].float()

            W_ref = B_flex @ A_flex
            rel_err = ((B_flex @ A_flex) - (B_qr @ A_qr)).norm() / W_ref.norm()
            assert rel_err < TOL_REL, \
                f"End-to-end [{key}] error {rel_err:.2e} ≥ {TOL_REL}"


# ---------------------------------------------------------------------------
# 8. End-to-end: aggregate_qrsvd vs aggregate_flexlora WITH delta mode

def test_end_to_end_with_delta():
    torch.manual_seed(13)
    out, in_, ranks, global_rank = 128, 128, [4, 16, 32, 64], 64
    keys = ["layer0", "layer1"]

    with tempfile.TemporaryDirectory() as tmp:
        # Write a fake "previous global model" at global_rank
        global_prev = os.path.join(tmp, "global_prev")
        B_g = {key: torch.randn(out, global_rank) for key in keys}
        A_g = {key: torch.randn(global_rank, in_) for key in keys}
        _write_adapter(global_prev, B_g, A_g, global_rank, out, in_)

        # Write client adapters
        client_dirs = []
        for k, r in enumerate(ranks):
            d = os.path.join(tmp, f"client_{k}")
            B_dict = {key: torch.randn(out, r) for key in keys}
            A_dict = {key: torch.randn(r, in_) for key in keys}
            _write_adapter(d, B_dict, A_dict, r, out, in_)
            client_dirs.append(d)

        out_flex = os.path.join(tmp, "global_flex")
        out_qr   = os.path.join(tmp, "global_qr")

        aggregate_flexlora(client_dirs, out_flex, global_rank,
                           global_model_path=global_prev)
        aggregate_qrsvd(client_dirs, out_qr, global_rank,
                        global_model_path=global_prev)

        from safetensors.torch import load_file
        sd_flex = load_file(os.path.join(out_flex, "adapter_model.safetensors"))
        sd_qr   = load_file(os.path.join(out_qr,   "adapter_model.safetensors"))

        for key in keys:
            B_flex = sd_flex[f"{key}.lora_B.weight"].float()
            A_flex = sd_flex[f"{key}.lora_A.weight"].float()
            B_qr   = sd_qr  [f"{key}.lora_B.weight"].float()
            A_qr   = sd_qr  [f"{key}.lora_A.weight"].float()

            W_ref = B_flex @ A_flex
            rel_err = ((B_flex @ A_flex) - (B_qr @ A_qr)).norm() / W_ref.norm()
            assert rel_err < TOL_REL, \
                f"Delta end-to-end [{key}] error {rel_err:.2e} ≥ {TOL_REL}"


# ---------------------------------------------------------------------------
# 9. dtype preservation

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_preserved(dtype):
    out, in_, ranks, global_rank = 128, 128, [4, 16], 16
    Bs, As = _make_clients(out, in_, ranks)
    sw = math.sqrt(0.5)
    A_g, B_g = _qr_svd_rank(
        [B.to(dtype) * sw for B in Bs],
        [A.to(dtype) * sw for A in As],
        global_rank, dtype,
    )
    assert A_g.dtype == dtype, f"Expected {dtype}, got {A_g.dtype}"
    assert B_g.dtype == dtype, f"Expected {dtype}, got {B_g.dtype}"
