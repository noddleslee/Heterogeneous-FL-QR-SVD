"""
Unit tests for aggregate.py — SVD math, FedAvg, and FlexLoRA aggregation.
No model loading required; pure PyTorch + safetensors.
"""
import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from aggregate import (
    _infer_rank,
    _lora_base_keys,
    _svd_to_rank,
    aggregate_flexlora,
    aggregate_lora,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_adapter_sd(rank, in_dim=32, out_dim=64, n_layers=2):
    """Return a fake adapter state dict with n_layers LoRA pairs."""
    torch.manual_seed(rank)
    sd = {}
    for i in range(n_layers):
        prefix = f"base_model.model.layer{i}.proj"
        sd[f"{prefix}.lora_A.weight"] = torch.randn(rank, in_dim)
        sd[f"{prefix}.lora_B.weight"] = torch.randn(out_dim, rank)
    return sd


def _save_adapter(sd, directory, rank):
    os.makedirs(directory, exist_ok=True)
    save_file(sd, os.path.join(directory, "adapter_model.safetensors"))
    with open(os.path.join(directory, "adapter_config.json"), "w") as f:
        json.dump({"r": rank, "lora_alpha": rank * 2, "target_modules": ["proj"]}, f)


# ── _svd_to_rank ───────────────────────────────────────────────────────────────

class TestSvdToRank:
    def test_output_shapes(self):
        """A: [rank, in], B: [out, rank]."""
        delta_W = torch.randn(64, 32)
        A, B = _svd_to_rank(delta_W, rank=8, orig_dtype=torch.float32)
        assert A.shape == (8, 32)
        assert B.shape == (64, 8)

    def test_reconstruction_quality(self):
        """B @ A must be a good low-rank approximation of delta_W."""
        torch.manual_seed(0)
        delta_W = torch.randn(64, 32)
        A, B = _svd_to_rank(delta_W, rank=16, orig_dtype=torch.float32)
        recon = B.float() @ A.float()
        # Relative Frobenius error — rank-16 approx of a full-rank 64×32 matrix
        rel_err = (recon - delta_W).norm() / delta_W.norm()
        assert rel_err < 0.9, f"Reconstruction too poor: rel_err={rel_err:.3f}"

    def test_exact_for_low_rank_input(self):
        """If delta_W already has rank r, SVD at that r must reconstruct exactly."""
        torch.manual_seed(1)
        # True rank-4 matrix
        U = torch.randn(64, 4)
        V = torch.randn(4, 32)
        delta_W = U @ V
        A, B = _svd_to_rank(delta_W, rank=8, orig_dtype=torch.float32)
        recon = B.float() @ A.float()
        assert torch.allclose(recon, delta_W, atol=1e-4), \
            f"Low-rank input not reconstructed exactly; max_err={((recon-delta_W).abs().max()):.2e}"

    def test_zero_padding_when_matrix_smaller_than_rank(self):
        """
        When min(out, in) < rank, output must be zero-padded to [rank, in] / [out, rank].
        A 3×3 matrix has at most 3 singular values, but we request rank 8.
        """
        delta_W = torch.randn(3, 3)
        A, B = _svd_to_rank(delta_W, rank=8, orig_dtype=torch.float32)
        assert A.shape == (8, 3), f"A shape={A.shape}"
        assert B.shape == (3, 8), f"B shape={B.shape}"
        # Padded rows/columns should be zero
        assert torch.all(A[3:] == 0), "Padded rows of A are not zero"
        assert torch.all(B[:, 3:] == 0), "Padded cols of B are not zero"

    def test_dtype_preserved(self):
        """Output dtype must match orig_dtype, not the float32 computation dtype."""
        delta_W = torch.randn(64, 32)
        for dtype in (torch.float16, torch.bfloat16):
            A, B = _svd_to_rank(delta_W, rank=8, orig_dtype=dtype)
            assert A.dtype == dtype, f"A dtype={A.dtype}, expected {dtype}"
            assert B.dtype == dtype, f"B dtype={B.dtype}, expected {dtype}"

    def test_contiguous_output(self):
        """Outputs must be contiguous (required by safetensors.save_file)."""
        delta_W = torch.randn(64, 32)
        A, B = _svd_to_rank(delta_W, rank=8, orig_dtype=torch.float32)
        assert A.is_contiguous(), "A is not contiguous"
        assert B.is_contiguous(), "B is not contiguous"

    def test_non_negative_singular_values(self):
        """sqrt of singular values must be real (no NaN from negative values)."""
        delta_W = torch.randn(64, 32)
        A, B = _svd_to_rank(delta_W, rank=8, orig_dtype=torch.float32)
        assert not torch.isnan(A).any(), "A contains NaN"
        assert not torch.isnan(B).any(), "B contains NaN"


# ── _infer_rank ────────────────────────────────────────────────────────────────

class TestInferRank:
    def test_infer_from_state_dict(self):
        sd = _make_adapter_sd(rank=16)
        assert _infer_rank(sd) == 16

    def test_infer_rank_4(self):
        sd = _make_adapter_sd(rank=4)
        assert _infer_rank(sd) == 4

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            _infer_rank({})


# ── FedAvg ─────────────────────────────────────────────────────────────────────

class TestFedAvg:
    def test_two_clients_exact_mean(self, tmp_path):
        """FedAvg of two adapters must equal the element-wise mean."""
        torch.manual_seed(42)
        rank = 4
        sd0 = _make_adapter_sd(rank)
        sd1 = _make_adapter_sd(rank + 1)  # different seed, same rank via override
        sd1 = {k: torch.randn_like(v) for k, v in sd0.items()}

        expected = {k: (sd0[k] + sd1[k]) / 2 for k in sd0}

        dirs = [str(tmp_path / f"c{i}") for i in range(2)]
        for d, sd in zip(dirs, [sd0, sd1]):
            _save_adapter(sd, d, rank)

        out = str(tmp_path / "global")
        aggregate_lora(dirs, out)

        result = load_file(os.path.join(out, "adapter_model.safetensors"))
        for k in expected:
            assert torch.allclose(result[k], expected[k], atol=1e-6), \
                f"FedAvg mismatch on {k}"

    def test_single_client_passthrough(self, tmp_path):
        """FedAvg of one client should return its weights unchanged."""
        sd = _make_adapter_sd(rank=8)
        dirs = [str(tmp_path / "c0")]
        _save_adapter(sd, dirs[0], 8)

        aggregate_lora(dirs, str(tmp_path / "global"))
        result = load_file(str(tmp_path / "global" / "adapter_model.safetensors"))

        for k in sd:
            assert torch.allclose(result[k], sd[k], atol=1e-6)

    def test_adapter_config_copied(self, tmp_path):
        """adapter_config.json should be copied to the output directory."""
        sd = _make_adapter_sd(rank=4)
        dirs = [str(tmp_path / "c0")]
        _save_adapter(sd, dirs[0], 4)

        out = str(tmp_path / "global")
        aggregate_lora(dirs, out)
        assert os.path.exists(os.path.join(out, "adapter_config.json"))


# ── FlexLoRA ───────────────────────────────────────────────────────────────────

class TestFlexLoRA:
    def test_output_shapes_match_global_rank(self, tmp_path):
        """All LoRA tensors in the global adapter must have dim == global_rank."""
        global_rank = 16
        for i, r in enumerate([4, 8, 16]):
            _save_adapter(_make_adapter_sd(r), str(tmp_path / f"c{i}"), r)

        dirs = [str(tmp_path / f"c{i}") for i in range(3)]
        out = str(tmp_path / "global")
        aggregate_flexlora(dirs, out, global_rank=global_rank)

        result = load_file(os.path.join(out, "adapter_model.safetensors"))
        for k, v in result.items():
            if ".lora_A.weight" in k:
                assert v.shape[0] == global_rank, f"{k}: A rank={v.shape[0]}"
            elif ".lora_B.weight" in k:
                assert v.shape[1] == global_rank, f"{k}: B rank={v.shape[1]}"

    def test_adapter_config_updated_to_global_rank(self, tmp_path):
        """adapter_config.json r must be global_rank, lora_alpha = 2 × global_rank."""
        for i, r in enumerate([4, 16]):
            _save_adapter(_make_adapter_sd(r), str(tmp_path / f"c{i}"), r)

        dirs = [str(tmp_path / f"c{i}") for i in range(2)]
        out = str(tmp_path / "global")
        aggregate_flexlora(dirs, out, global_rank=32)

        with open(os.path.join(out, "adapter_config.json")) as f:
            cfg = json.load(f)
        assert cfg["r"] == 32
        assert cfg["lora_alpha"] == 64

    def test_heterogeneous_ranks_no_crash(self, tmp_path):
        """FlexLoRA with ranks [4, 16, 32, 64] must aggregate without error."""
        for i, r in enumerate([4, 16, 32, 64]):
            _save_adapter(_make_adapter_sd(r), str(tmp_path / f"c{i}"), r)

        dirs = [str(tmp_path / f"c{i}") for i in range(4)]
        out = str(tmp_path / "global")
        aggregate_flexlora(dirs, out, global_rank=64)  # global = max client rank

        # Sanity: output file exists and has expected keys
        result = load_file(os.path.join(out, "adapter_model.safetensors"))
        assert len(result) > 0

    def test_single_client_weight_zero_excluded(self, tmp_path):
        """
        With client_weights=[0, 1], the zero-weight client must not affect
        the aggregated ΔW.  The global adapter must reconstruct client 1's
        weight delta with near-zero error.
        """
        torch.manual_seed(7)
        in_dim, out_dim, rank = 16, 32, 4

        # client 0: all zeros (should be excluded by weight=0)
        sd0 = {
            "layer.lora_A.weight": torch.zeros(rank, in_dim),
            "layer.lora_B.weight": torch.zeros(out_dim, rank),
        }
        # client 1: random weights
        sd1 = {
            "layer.lora_A.weight": torch.randn(rank, in_dim),
            "layer.lora_B.weight": torch.randn(out_dim, rank),
        }
        expected_dW = sd1["layer.lora_B.weight"].float() @ sd1["layer.lora_A.weight"].float()

        for i, (sd, r) in enumerate([(sd0, rank), (sd1, rank)]):
            _save_adapter(sd, str(tmp_path / f"c{i}"), r)

        out = str(tmp_path / "global")
        aggregate_flexlora(
            [str(tmp_path / "c0"), str(tmp_path / "c1")],
            out, global_rank=rank, client_weights=[0.0, 1.0],
        )

        result = load_file(os.path.join(out, "adapter_model.safetensors"))
        A = result["layer.lora_A.weight"].float()
        B = result["layer.lora_B.weight"].float()
        recon_dW = B @ A

        rel_err = (recon_dW - expected_dW).norm() / (expected_dW.norm() + 1e-8)
        assert rel_err < 1e-3, \
            f"Zero-weight client contaminated result: rel_err={rel_err:.4f}"

    def test_output_contiguous(self, tmp_path):
        """All output tensors must be contiguous (safetensors requirement)."""
        for i, r in enumerate([4, 16]):
            _save_adapter(_make_adapter_sd(r), str(tmp_path / f"c{i}"), r)

        out = str(tmp_path / "global")
        aggregate_flexlora(
            [str(tmp_path / "c0"), str(tmp_path / "c1")],
            out, global_rank=16,
        )
        result = load_file(os.path.join(out, "adapter_model.safetensors"))
        for k, v in result.items():
            assert v.is_contiguous(), f"{k} is not contiguous"
