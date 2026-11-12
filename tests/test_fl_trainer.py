"""
Tests for FLSFTTrainer and the stateful FL training loop.

Covers:
  - _find_latest_checkpoint path logic
  - FLSFTTrainer._load_from_checkpoint is a no-op (injected adapter preserved)
  - FLSFTTrainer.create_scheduler uses total_rounds * steps_per_round
  - [integration] global_step is NOT reset between FL rounds
  - [integration] LR scheduler is continuous across round boundary
  - [integration] injected adapter weights survive checkpoint resume

Integration tests use a tiny GPT-2 (n_embd=32) + LoRA on CPU.
They are marked with @pytest.mark.integration and can be skipped
in fast CI runs via: pytest -m "not integration"
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# conftest.py stubs unsloth before any import; handle standalone runs too
import importlib
for _mod in ["unsloth", "unsloth.trainer"]:
    if _mod not in sys.modules:
        _m = MagicMock()
        _m.__spec__ = importlib.machinery.ModuleSpec(_mod, None)
        sys.modules[_mod] = _m
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_client import FLSFTTrainer, _find_latest_checkpoint
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


# ── tiny model & dataset helpers ───────────────────────────────────────────────

def make_tiny_peft_model(seed: int = 0):
    """
    A tiny GPT-2 (32-dim, 2 layers) wrapped with rank-4 LoRA on c_attn.
    Runs fast on CPU; vocab_size=50257 matches the GPT-2 tokenizer.
    """
    torch.manual_seed(seed)
    config = GPT2Config(
        n_embd=32, n_head=2, n_layer=2, n_positions=32, vocab_size=50257,
    )
    base = GPT2LMHeadModel(config)
    lora_cfg = LoraConfig(
        r=4, lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base, lora_cfg)


_TOKENIZER = None  # lazily loaded once

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    return _TOKENIZER


def make_tiny_dataset(n: int = 64, seq_len: int = 16):
    """
    Pre-tokenized dataset — skip_prepare_dataset=True bypasses SFTTrainer's
    tokenisation so we never need to call the tokenizer on data.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, 200, (n, seq_len)).tolist()
    return Dataset.from_dict({
        "input_ids":      ids,
        "attention_mask": [[1] * seq_len] * n,
        "labels":         ids,
    })


def make_fl_trainer(
    model,
    dataset,
    ckpt_dir: str,
    round_idx: int,
    steps_per_round: int,
    total_rounds: int,
    **extra_sft_kwargs,
):
    """Build a FLSFTTrainer for a single FL round (CPU, adamw_torch)."""
    abs_max_steps = (round_idx + 1) * steps_per_round
    sft_args = SFTConfig(
        output_dir=ckpt_dir,
        max_steps=abs_max_steps,
        save_strategy="steps",
        save_steps=steps_per_round,
        save_total_limit=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        logging_steps=1,
        report_to="none",
        eval_strategy="no",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        use_cpu=True,
        **extra_sft_kwargs,
    )
    return FLSFTTrainer(
        total_rounds=total_rounds,
        steps_per_round=steps_per_round,
        model=model,
        tokenizer=get_tokenizer(),
        train_dataset=dataset,
        args=sft_args,
    )


def lora_param_snapshot(model):
    """Return a dict of cloned LoRA parameter tensors."""
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if "lora_A" in n or "lora_B" in n
    }


# ── _find_latest_checkpoint ────────────────────────────────────────────────────

class TestFindLatestCheckpoint:
    def test_empty_dir_returns_none(self, tmp_path):
        assert _find_latest_checkpoint(str(tmp_path)) is None

    def test_single_checkpoint(self, tmp_path):
        (tmp_path / "checkpoint-10").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert result.endswith("checkpoint-10")

    def test_returns_highest_number(self, tmp_path):
        for n in [10, 50, 30]:
            (tmp_path / f"checkpoint-{n}").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert result.endswith("checkpoint-50"), f"Got: {result}"

    def test_ignores_non_checkpoint_dirs(self, tmp_path):
        (tmp_path / "logs").mkdir()
        (tmp_path / "checkpoint-5").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert result.endswith("checkpoint-5")

    def test_ignores_checkpoint_files_not_dirs(self, tmp_path):
        (tmp_path / "checkpoint-100").write_text("not a dir")
        result = _find_latest_checkpoint(str(tmp_path))
        assert result is None


# ── FLSFTTrainer unit tests ────────────────────────────────────────────────────

class TestFLSFTTrainerUnit:
    def test_load_from_checkpoint_is_noop(self, tmp_path):
        """
        _load_from_checkpoint must not modify any model parameter.
        We call it with a non-existent path; our override does nothing.
        """
        model = make_tiny_peft_model()
        dataset = make_tiny_dataset()
        trainer = make_fl_trainer(
            model, dataset, str(tmp_path), round_idx=0, steps_per_round=2, total_rounds=2
        )

        before = lora_param_snapshot(model)

        # Non-existent path — a real implementation would crash; ours must not
        trainer._load_from_checkpoint("/nonexistent/fake/path")

        after = lora_param_snapshot(model)
        for name in before:
            assert torch.equal(before[name], after[name]), \
                f"_load_from_checkpoint modified {name}"

    def test_create_scheduler_uses_total_steps(self, tmp_path):
        """
        create_scheduler must call super().create_scheduler with
        num_training_steps = total_rounds * steps_per_round.
        """
        total_rounds, steps_per_round = 5, 10
        model = make_tiny_peft_model()
        dataset = make_tiny_dataset()

        trainer = make_fl_trainer(
            model, dataset, str(tmp_path),
            round_idx=0, steps_per_round=steps_per_round, total_rounds=total_rounds,
        )

        captured = {}

        original = SFTTrainer.create_scheduler

        def spy_create_scheduler(self, num_training_steps, optimizer=None):
            captured["num_training_steps"] = num_training_steps
            return original(self, num_training_steps, optimizer=optimizer)

        with patch.object(SFTTrainer, "create_scheduler", spy_create_scheduler):
            trainer.create_scheduler(num_training_steps=steps_per_round)

        expected = total_rounds * steps_per_round
        assert captured.get("num_training_steps") == expected, (
            f"Expected {expected}, got {captured.get('num_training_steps')}"
        )


# ── integration tests (tiny GPT-2, CPU training) ──────────────────────────────

@pytest.mark.integration
class TestStatefulTraining:
    """
    End-to-end FL round simulation.

    Each test trains a tiny GPT-2 + LoRA model for a few steps,
    simulates FL aggregation (injects known adapter weights), then resumes
    and verifies the trainer state carries over correctly.

    These run on CPU and take ~10–30 s total.
    """

    def test_global_step_continues_across_rounds(self, tmp_path):
        """
        global_step must NOT reset between FL rounds.

        Round 0: trains 0→N steps,   saves checkpoint-N.
        Round 1: resumes from ckpt,  trains N→2N steps.
        Final state.global_step == 2N.
        """
        model = make_tiny_peft_model()
        dataset = make_tiny_dataset()
        ckpt_dir = str(tmp_path / "ckpt")
        os.makedirs(ckpt_dir)

        N = 3  # steps per round

        # ── Round 0 ──────────────────────────────────────────────────────────
        t0 = make_fl_trainer(model, dataset, ckpt_dir, round_idx=0, steps_per_round=N, total_rounds=2)
        t0.train()

        assert t0.state.global_step == N, \
            f"Round 0: expected {N} steps, got {t0.state.global_step}"

        ckpt = _find_latest_checkpoint(ckpt_dir)
        assert ckpt is not None, "No checkpoint saved after round 0"
        assert Path(ckpt).name == f"checkpoint-{N}", \
            f"Expected checkpoint-{N}, found {Path(ckpt).name}"

        # ── Round 1 ──────────────────────────────────────────────────────────
        t1 = make_fl_trainer(model, dataset, ckpt_dir, round_idx=1, steps_per_round=N, total_rounds=2)
        t1.train(resume_from_checkpoint=ckpt)

        assert t1.state.global_step == 2 * N, \
            f"Round 1: expected {2*N} steps, got {t1.state.global_step}"

    def test_injected_adapter_not_overwritten_on_resume(self, tmp_path):
        """
        After FL aggregation injects a sentinel adapter, resuming from a
        checkpoint must NOT restore the stale checkpoint weights.

        If _load_from_checkpoint were NOT a no-op, the adapter would revert
        to the round-0 trained values, and the sentinel would be lost.
        """
        model = make_tiny_peft_model()
        dataset = make_tiny_dataset()
        ckpt_dir = str(tmp_path / "ckpt")
        os.makedirs(ckpt_dir)

        N = 2

        # Round 0: train and checkpoint
        t0 = make_fl_trainer(model, dataset, ckpt_dir, 0, N, total_rounds=2)
        t0.train()
        ckpt = _find_latest_checkpoint(ckpt_dir)

        # ── Simulate FL aggregation: fill all LoRA weights with sentinel ──
        SENTINEL = 42.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.fill_(SENTINEL)

        # Capture the first lora_A weight at the very start of the first step
        # in round 1 (before any gradient update, after _load_from_checkpoint).
        captured = {}

        class CaptureWeightsAtFirstStep(TrainerCallback):
            def on_step_begin(self, args, state, control, model=None, **kwargs):
                if not captured:  # only capture once
                    for name, param in model.named_parameters():
                        if "lora_A" in name:
                            captured["name"] = name
                            captured["weight"] = param.detach().clone()
                            break

        t1 = make_fl_trainer(model, dataset, ckpt_dir, 1, N, total_rounds=2)
        t1.add_callback(CaptureWeightsAtFirstStep())
        t1.train(resume_from_checkpoint=ckpt)

        assert captured, "CaptureWeightsAtFirstStep never fired"

        mean_val = captured["weight"].float().mean().item()
        assert abs(mean_val - SENTINEL) < 0.5, (
            f"Adapter weight at start of round 1 is {mean_val:.2f}; "
            f"expected ~{SENTINEL} (injected sentinel). "
            f"_load_from_checkpoint may have overwritten the injected weights.\n"
            f"  Parameter: {captured['name']}"
        )

    def test_lr_schedule_continuous_across_rounds(self, tmp_path):
        """
        The LR at the start of round 1 must equal the LR at the end of round 0.
        If the scheduler restarted each round, round 1 would begin at the
        initial (higher) LR — which would be a discontinuity.
        """
        model = make_tiny_peft_model()
        dataset = make_tiny_dataset()
        ckpt_dir = str(tmp_path / "ckpt")
        os.makedirs(ckpt_dir)

        N = 4  # enough steps for the LR to decay measurably

        # Round 0
        t0 = make_fl_trainer(model, dataset, ckpt_dir, 0, N, total_rounds=2)
        t0.train()
        lr_end_round0 = t0.lr_scheduler.get_last_lr()[0]
        ckpt = _find_latest_checkpoint(ckpt_dir)

        # Capture LR at the very first optimizer step of round 1
        lr_start_round1 = {}
        t1 = make_fl_trainer(model, dataset, ckpt_dir, 1, N, total_rounds=2)

        original_step = t1.optimizer.step.__class__  # for type check

        class LRCapture(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if not lr_start_round1:
                    lr_start_round1["lr"] = t1.lr_scheduler.get_last_lr()[0]

        t1.add_callback(LRCapture())
        t1.train(resume_from_checkpoint=ckpt)

        assert "lr" in lr_start_round1, "LR callback never fired"

        lr0 = lr_end_round0
        lr1 = lr_start_round1["lr"]
        assert abs(lr1 - lr0) < 1e-6, (
            f"LR discontinuity: end of round 0 = {lr0:.8f}, "
            f"start of round 1 = {lr1:.8f}.\n"
            f"The LR scheduler was not resumed from checkpoint correctly."
        )

    def test_checkpoint_deleted_by_save_total_limit(self, tmp_path):
        """
        With save_total_limit=1, only the latest checkpoint is kept.
        After round 1, checkpoint-N must be deleted and checkpoint-2N kept.
        """
        model = make_tiny_peft_model()
        dataset = make_tiny_dataset()
        ckpt_dir = str(tmp_path / "ckpt")
        os.makedirs(ckpt_dir)

        N = 2

        t0 = make_fl_trainer(model, dataset, ckpt_dir, 0, N, total_rounds=2)
        t0.train()
        assert (Path(ckpt_dir) / f"checkpoint-{N}").is_dir(), \
            f"checkpoint-{N} not saved after round 0"

        ckpt = _find_latest_checkpoint(ckpt_dir)
        t1 = make_fl_trainer(model, dataset, ckpt_dir, 1, N, total_rounds=2)
        t1.train(resume_from_checkpoint=ckpt)

        # Old checkpoint deleted
        assert not (Path(ckpt_dir) / f"checkpoint-{N}").exists(), \
            f"checkpoint-{N} should have been deleted (save_total_limit=1)"
        # New checkpoint present
        assert (Path(ckpt_dir) / f"checkpoint-{2*N}").is_dir(), \
            f"checkpoint-{2*N} not present after round 1"
