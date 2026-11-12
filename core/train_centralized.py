"""
Centralized training script for PathVQA �� matched to FL-B for fair comparison.

Config alignment with FL-B:
  1. Model        : Qwen/Qwen3.5-0.8B           (same as FL-B)
  2. Eval split   : official PathVQA validation  (not random 10% of train)
  3. Yes/no prompt: adds "? Please answer yes or no."  (matches FL-B)
  4. Batch size   : configurable via --batch_size (64 or 256)
  5. Total steps  : 1,000                        (matches FL-B: 10 rounds �� 100 steps)
  6. warmup_ratio : 0.03
  7. weight_decay : 0.001                        (matches FL-B)
  8. max_length   : 512                          (matches FL-B)

Two configurations:
  --batch_size 64  �� 1,000 steps �� 64  / 19,654 samples ��  3.25 epochs (matches FL optimizer-step count)
  --batch_size 256 �� 1,000 steps �� 256 / 19,654 samples �� 13.02 epochs (matches FL total sample views)

Usage:
  python train_centralized.py --rank 4  --batch_size 64  --output_dir results/centralized_r4_b64
  python train_centralized.py --rank 64 --batch_size 256 --output_dir results/centralized_r64_b256
"""

import argparse
import json
import os
import re
import subprocess
import warnings
from pathlib import Path
from runtime_config import dataset_cache_dir, python_executable, evaluation_script

import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", message="Truncated File Read")

_CACHE_DIR = dataset_cache_dir()
SEED = 3407

_PYTHON     = python_executable()
_EVAL_SCRIPT = evaluation_script()


# ���� dataset helpers �� identical to train_client.py ��������������������������������������������������������

def _is_valid_image(sample):
    try:
        sample["image"].convert("RGB").load()
        return True
    except Exception:
        return False


def _is_yes_no(answer: str) -> bool:
    return answer.lower().strip() in ("yes", "no")


class LazyPathVQADataset(torch.utils.data.Dataset):
    """
    Change 3: adds "? Please answer yes or no." to yes/no questions.
    Identical to train_client.py �� ensures the same prompt format as FL-B.
    """
    def __init__(self, hf_ds):
        self.hf_ds = hf_ds

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        try:
            img = sample["image"].convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        question = sample["question"]
        if _is_yes_no(sample["answer"]):
            question = question.rstrip("?") + "? Please answer yes or no."
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text",  "text": question},
                    {"type": "image", "image": img},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": sample["answer"]},
                ]},
            ]
        }


def load_split(split: str) -> LazyPathVQADataset:
    """
    Change 2: loads the official train / validation split.
    Uses arrow cache to avoid repeated filtering (same cache dir as FL-B).
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    filter_cache = os.path.join(_CACHE_DIR, f"pathvqa_{split}_f_central.arrow")

    print(f"Loading PathVQA {split} split ...")
    ds = load_dataset("flaviagiammarino/path-vqa", split=split)
    print(f"  Raw size: {len(ds)}")

    ds = ds.filter(
        _is_valid_image,
        num_proc=4,
        cache_file_name=filter_cache,
        desc=f"Filtering {split}",
    )
    print(f"  After filter: {len(ds)}")
    return LazyPathVQADataset(ds)


# ���� results helpers ����������������������������������������������������������������������������������������������������������������������

def _submit_eval_job(base_model: str, checkpoint_dir: str, output_dir: str) -> str | None:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    ckpt_name = Path(checkpoint_dir).name

    rank_match = re.search(r"rank(\d+)", output_dir)
    rank_suffix = rank_match.group(1) if rank_match else "ckpt"

    script_path  = os.path.join(output_dir, f"submit_eval_{ckpt_name}.sh")
    log_path     = os.path.join(output_dir, f"eval_{ckpt_name}_%j.out")
    output_json  = os.path.join(output_dir, f"eval_results_{ckpt_name}.json")

    tmp_dir      = os.path.join(output_dir, ".eval_tmp")
    inductor_dir = os.path.join(output_dir, ".torchinductor")
    triton_dir   = os.path.join(output_dir, ".triton")
    for d in (tmp_dir, inductor_dir, triton_dir):
        os.makedirs(d, exist_ok=True)

    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --job-name=eval_r{rank_suffix}
#SBATCH --tasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output={log_path}

ml load CUDA/12.6.0
ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1
export TMPDIR={tmp_dir}
export TORCHINDUCTOR_CACHE_DIR={inductor_dir}
export TRITON_CACHE_DIR={triton_dir}

{_PYTHON} {_EVAL_SCRIPT} \\
    --base_model {base_model} \\
    --peft_path {checkpoint_dir} \\
    --split validation \\
    --n_samples none \\
    --output_json {output_json}
""")

    os.chmod(script_path, 0o755)
    try:
        submission = subprocess.check_output(["sbatch", script_path], text=True).strip()
        print(f"  Submitted eval job: {submission}")
        parts = submission.split()
        return parts[-1] if parts and parts[-1].isdigit() else None
    except subprocess.CalledProcessError as exc:
        print(f"  WARNING: failed to submit eval job: {exc}")
        return None


# ���� main ��������������������������������������������������������������������������������������������������������������������������������������������

def main():
    parser = argparse.ArgumentParser(
        description="Centralized PathVQA fine-tuning matched to FL-B"
    )
    parser.add_argument("--rank",       type=int, default=4,
                        help="LoRA rank: 4 / 8 / 16 / 32 / 64")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Per-device training batch size: 64 or 256")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save adapter, logs, and checkpoints")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-0.8B")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":8192:8"
    os.environ["UNSLOTH_RETURN_LOGITS"]   = "1"

    # ���� Change 1: Qwen/Qwen3.5-0.8B ����
    print(f"\nLoading {args.base_model}  rank={args.rank} ...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0,
        bias="none",
        random_state=SEED,
    )
    model.print_trainable_parameters()

    # ���� Changes 2 + 3: official validation split + yes/no prompt ����
    train_ds = load_split("train")
    val_ds   = load_split("validation")
    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps = 1000
    print(f"\nTrain size : {len(train_ds)}")
    print(f"Val size   : {len(val_ds)}")
    print(f"Batch size : {args.batch_size}  steps/epoch: {steps_per_epoch}  total steps: {total_steps}  (~{total_steps / steps_per_epoch:.2f} epochs)")

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=8,
        max_steps=1000,
        eval_strategy="no",
        save_strategy="epoch",                # save one checkpoint at end of epoch
        save_total_limit=1,
        logging_steps=10,
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        seed=SEED,
        report_to="none",
        max_length=512,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_args,
    )

    print(f"\nStarting centralized training  rank={args.rank}  batch={args.batch_size}  1,000 steps (~{total_steps / steps_per_epoch:.2f} epochs) ...")
    train_result = trainer.train()

    # extract loss curve from trainer state
    train_log = [
        {"step": e["step"], "loss": e["loss"]}
        for e in trainer.state.log_history
        if "loss" in e and "eval_loss" not in e
    ]

    # submit async eval job for each saved checkpoint
    all_ckpts = sorted(
        Path(args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    eval_job_ids = {}
    for ckpt in all_ckpts:
        job_id = _submit_eval_job(args.base_model, str(ckpt), args.output_dir)
        eval_job_ids[ckpt.name] = job_id

    latest_ckpt = str(all_ckpts[-1]) if all_ckpts else None

    # write results.json
    results = {
        "rank":             args.rank,
        "total_steps":      1000,
        "base_model":       args.base_model,
        "train_samples":    len(train_ds),
        "batch_size":       args.batch_size,
        "train_metrics":    train_result.metrics,
        "train_loss_curve": train_log,
        "checkpoints":      [str(c) for c in all_ckpts],
        "final_checkpoint": latest_ckpt,
        "eval_job_ids":     eval_job_ids,
    }

    out_path = Path(args.output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"  rank={args.rank}  1,000-step results")
    print("="*60)
    print(f"  Train loss (final): {train_log[-1]['loss'] if train_log else 'N/A'}")
    print(f"  Saved checkpoints:  {[c.name for c in all_ckpts]}")
    for ckpt_name, job_id in eval_job_ids.items():
        print(f"  Eval job {ckpt_name}: {job_id or 'submission failed'}")
    print(f"  Results saved -> {out_path}")
    print("="*60)

    # ���� save final adapter ����
    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nAdapter saved to {adapter_dir}")
    print("Centralized training complete.")


if __name__ == "__main__":
    main()
