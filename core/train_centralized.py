"""
Centralized training script for PathVQA — matched to FL-B for fair comparison.

All 8 changes vs existing centralized scripts:
  1. Model        : Qwen/Qwen3.5-0.8B           (same as FL-B)
  2. Eval split   : official PathVQA validation  (not random 10% of train)
  3. Yes/no prompt: adds "? Please answer yes or no."  (matches FL-B)
  4. Batch size   : 16                           (matches FL-B)
  5. max_steps    : 1000                         (matches FL-B total steps)
  6. warmup_ratio : 0.03                         (~30 warmup steps, matches FL-B)
  7. weight_decay : 0.001                        (matches FL-B)
  8. max_length   : 512                          (matches FL-B)

LR scheduler is already "stateful" in centralized — one continuous linear decay
over 1000 steps, no per-round restarts.

Usage:
  python train_centralized.py --rank 4  --output_dir results/centralized_r4
  python train_centralized.py --rank 16 --output_dir results/centralized_r16
  python train_centralized.py --rank 64 --output_dir results/centralized_r64
"""

import argparse
import os
import warnings

import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", message="Truncated File Read")

_CACHE_DIR = "/mimer/NOBACKUP/groups/bloom/shenghui/hfl/.dataset_cache"
SEED = 3407


# ── dataset helpers — identical to train_client.py ────────────────────────────

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
    Identical to train_client.py — ensures the same prompt format as FL-B.
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Centralized PathVQA fine-tuning matched to FL-B"
    )
    parser.add_argument("--rank",       type=int,  default=4,
                        help="LoRA rank: 4 / 16 / 64")
    parser.add_argument("--output_dir", type=str,  required=True,
                        help="Directory to save adapter, logs, and checkpoints")
    parser.add_argument("--base_model", type=str,  default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max_steps",  type=int,  default=1000,
                        help="Total training steps (default=1000, matches FL-B)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":8192:8"
    os.environ["UNSLOTH_RETURN_LOGITS"]   = "1"

    # ── Change 1: Qwen/Qwen3.5-0.8B ──
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

    # ── Changes 2 + 3: official validation split + yes/no prompt ──
    train_ds = load_split("train")
    val_ds   = load_split("validation")
    print(f"\nTrain size : {len(train_ds)}")
    print(f"Val size   : {len(val_ds)}")
    print(f"Steps at batch=16: {len(train_ds) // 16} per epoch  "
          f"(training stops at max_steps={args.max_steps})")

    # ── Changes 4–8: batch=16, max_steps=1000, warmup_ratio=0.03,
    #                weight_decay=0.001, max_length=512 ──
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=16,       # Change 4
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=8,
        max_steps=args.max_steps,             # Change 5
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.max_steps,            # one checkpoint at the very end
        save_total_limit=1,
        logging_steps=10,
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.001,                   # Change 7
        lr_scheduler_type="linear",
        warmup_ratio=0.03,                    # Change 6
        seed=SEED,
        report_to="none",
        max_length=512,                       # Change 8
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

    print(f"\nStarting centralized training  "
          f"rank={args.rank}  max_steps={args.max_steps} ...")
    trainer.train()

    # ── save final adapter ──
    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nAdapter saved to {adapter_dir}")
    print("Centralized training complete.")


if __name__ == "__main__":
    main()
