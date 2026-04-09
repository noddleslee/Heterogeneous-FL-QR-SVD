"""
Client training script for Federated Learning on PathVQA.

Stateful training across FL rounds
------------------------------------
When --max_steps > 0 (steps per round), the trainer state persists across
rounds so the dataset cursor, optimizer, and LR scheduler are NOT reset at
each round boundary.

  - Persistent checkpoint dir : <output_dir>/client_<id>_ckpt/
      stores optimizer.pt / scheduler.pt / trainer_state.json / rng_state*.pth
  - Per-round adapter output   : <output_dir>/round_<R>/client_<id>/
      contains only adapter_model.safetensors + adapter_config.json (for aggregation)

  Round 0 : trains steps 1..steps_per_round,   saves checkpoint-{steps_per_round}
  Round 1 : resumes checkpoint, trains steps steps_per_round+1..2*steps_per_round
  …

  FLSFTTrainer overrides two things:
    create_scheduler  → spans total_rounds * steps_per_round so LR doesn't restart
    _load_from_checkpoint → no-op for model weights; the caller already loaded the
                            aggregated (+ projected) adapter into the model, so we
                            must not let HuggingFace overwrite it with stale weights.
                            Optimizer / scheduler / dataloader state still resume.

FlexLoRA support
-----------------
When --flexlora is passed, the global adapter (at --global_rank) is projected
down to the client's --rank via truncated SVD before local training begins.
"""

import argparse, json, os, shutil, warnings
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", message="Truncated File Read")

_CACHE_DIR = "/mimer/NOBACKUP/groups/bloom/shenghui/hfl/.dataset_cache"


# ── dataset helpers ────────────────────────────────────────────────────────────

def _is_valid_image(sample):
    try:
        sample["image"].convert("RGB").load()
        return True
    except Exception:
        return False


def _is_yes_no(answer: str) -> bool:
    return answer.lower().strip() in ("yes", "no")


class LazyPathVQADataset(torch.utils.data.Dataset):
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


def load_split(split: str, client_id: int, num_clients: int):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    filter_cache = os.path.join(_CACHE_DIR, f"pathvqa_{split}_f_c{client_id}.arrow")

    print(f"Loading raw PathVQA {split} split …")
    ds = load_dataset("flaviagiammarino/path-vqa", split=split)

    ds = ds.shuffle(seed=42).shard(num_shards=num_clients, index=client_id)
    print(f"  Shard size for client {client_id}: {len(ds)}")

    ds = ds.filter(_is_valid_image, num_proc=4,
                   cache_file_name=filter_cache, desc=f"Filtering {split} shard")

    return LazyPathVQADataset(ds)


# ── stateful FL trainer ────────────────────────────────────────────────────────

def _find_latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return path to the highest-numbered checkpoint-N dir, or None."""
    ckpts = sorted(
        [p for p in Path(ckpt_dir).glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else None


class FLSFTTrainer(SFTTrainer):
    """
    SFTTrainer extended for stateful federated learning.

    Two key overrides
    -----------------
    create_scheduler
        Creates the LR schedule for the FULL training horizon
        (total_rounds × steps_per_round) so the LR curve is continuous
        across all FL rounds rather than restarting each round.

    _load_from_checkpoint
        Skips model-weight loading when resuming from a checkpoint.
        The correct (aggregated + projected) adapter has already been loaded
        into the model before train() is called; we must not overwrite it
        with the stale weights stored in the checkpoint.
        Optimizer state, scheduler state, and dataloader position (via
        trainer_state.json + rng_state*.pth) still resume normally because
        those are loaded by separate HuggingFace Trainer methods.
    """

    def __init__(self, total_rounds: int, steps_per_round: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_rounds = total_rounds
        self.steps_per_round = steps_per_round

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        total = self.total_rounds * self.steps_per_round
        print(f"  [FLSFTTrainer] scheduler total_steps={total} "
              f"({self.total_rounds} rounds × {self.steps_per_round} steps/round)")
        super().create_scheduler(num_training_steps=total, optimizer=optimizer)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # Intentionally skip model-weight loading.
        # The caller loaded the correct aggregated adapter before train().
        print(f"  [FLSFTTrainer] Skipping model-weight load from checkpoint "
              f"(adapter already loaded). Optimizer/scheduler will still resume.")


# ── FlexLoRA helpers ───────────────────────────────────────────────────────────

def _load_safetensors(path: str) -> dict:
    from safetensors.torch import load_file
    st = os.path.join(path, "adapter_model.safetensors")
    bn = os.path.join(path, "adapter_model.bin")
    if os.path.exists(st):
        return load_file(st)
    if os.path.exists(bn):
        return torch.load(bn, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights in {path}")


def _infer_adapter_rank(adapter_dir: str) -> int:
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return json.load(f)["r"]
    sd = _load_safetensors(adapter_dir)
    for k, v in sd.items():
        if ".lora_A.weight" in k:
            return v.shape[0]
    raise ValueError(f"Cannot infer rank from {adapter_dir}")


def project_and_load_adapter(global_adapter_dir: str, model, local_rank: int):
    """
    Load the global adapter (possibly at global_rank > local_rank) and
    project it down to local_rank via truncated SVD, then load into model.

    ΔW_global = B_global @ A_global
    U, S, Vh  = SVD(ΔW_global)
    B_local   = U[:, :r] * sqrt(S[:r])   [out, r]
    A_local   = Vh[:r, :] * sqrt(S[:r])  [r, in]
    → B_local @ A_local ≈ ΔW_global  (best rank-r approximation)
    """
    from peft import set_peft_model_state_dict

    global_rank = _infer_adapter_rank(global_adapter_dir)
    global_sd = _load_safetensors(global_adapter_dir)

    if local_rank >= global_rank:
        incompatible = set_peft_model_state_dict(model, global_sd)
        _warn_incompatible(incompatible)
        print(f"  Loaded global adapter (rank={global_rank}) without projection")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    projected = {}
    processed = set()

    for key in global_sd:
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key:
            projected[key] = global_sd[key]
            continue

        bk = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        if bk in processed:
            continue
        processed.add(bk)

        a_key = bk + ".lora_A.weight"
        b_key = bk + ".lora_B.weight"
        if a_key not in global_sd or b_key not in global_sd:
            projected[key] = global_sd[key]
            continue

        A_g = global_sd[a_key].float().to(device)   # [global_rank, in]
        B_g = global_sd[b_key].float().to(device)   # [out, global_rank]
        delta_W = B_g @ A_g                          # [out, in]

        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        r_actual = min(local_rank, S.shape[0])
        sqrt_S = S[:r_actual].clamp(min=0.0).sqrt()

        orig_dtype = global_sd[a_key].dtype
        A_r = (Vh[:r_actual, :] * sqrt_S.unsqueeze(1)).contiguous().cpu()
        B_r = (U[:, :r_actual]  * sqrt_S.unsqueeze(0)).contiguous().cpu()

        if r_actual < local_rank:
            A_full = torch.zeros(local_rank, delta_W.shape[1], dtype=A_r.dtype)
            B_full = torch.zeros(delta_W.shape[0], local_rank, dtype=B_r.dtype)
            A_full[:r_actual, :] = A_r
            B_full[:, :r_actual] = B_r
            projected[a_key] = A_full.to(orig_dtype)
            projected[b_key] = B_full.to(orig_dtype)
        else:
            projected[a_key] = A_r.to(orig_dtype)
            projected[b_key] = B_r.to(orig_dtype)

    incompatible = set_peft_model_state_dict(model, projected)
    _warn_incompatible(incompatible)
    print(f"  Projected global adapter rank {global_rank} → local rank {local_rank}")


def _warn_incompatible(result):
    if result is None:
        return
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        print(f"  Warning: unexpected adapter keys: {unexpected[:3]}{'…' if len(unexpected) > 3 else ''}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id",    type=int, required=True)
    parser.add_argument("--num_clients",  type=int, default=4)
    parser.add_argument("--round",        type=int, default=0)
    parser.add_argument("--total_rounds", type=int, default=10,
                        help="Total number of FL rounds (used to span the LR scheduler)")
    parser.add_argument("--rank",         type=int, default=16,
                        help="Local LoRA rank for this client")
    parser.add_argument("--global_rank",  type=int, default=None,
                        help="Global LoRA rank on the server (FlexLoRA only). "
                             "Defaults to --rank (no projection).")
    parser.add_argument("--flexlora",     action="store_true",
                        help="Enable FlexLoRA: project global adapter to local rank "
                             "before training")
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--base_model",   type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--global_model_path", type=str, default=None,
                        help="Path to global adapter from previous round")
    parser.add_argument("--epochs",       type=int, default=1,
                        help="Fallback epoch count when --max_steps is -1")
    parser.add_argument("--max_steps",    type=int, default=-1,
                        help="Steps per FL round (>0 enables stateful training). "
                             "-1 = full-epoch mode (legacy, no state persistence).")
    args = parser.parse_args()

    if args.global_rank is None:
        args.global_rank = args.rank

    stateful = args.max_steps > 0

    # ── directories ──
    # Persistent checkpoint dir — optimizer/scheduler/dataloader state across rounds
    client_ckpt_dir = os.path.join(args.output_dir, f"client_{args.client_id}_ckpt")
    os.makedirs(client_ckpt_dir, exist_ok=True)

    # Per-round adapter output — contains only the adapter for aggregation
    client_output_dir = os.path.join(
        args.output_dir, f"round_{args.round}", f"client_{args.client_id}"
    )
    os.makedirs(client_output_dir, exist_ok=True)

    # ── load data ──
    train_ds = load_split("train",      args.client_id, args.num_clients)
    val_ds   = load_split("validation", args.client_id, args.num_clients)

    # ── load base model ──
    print(f"\n[Client {args.client_id}] Loading model: {args.base_model}  "
          f"(local_rank={args.rank})")
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
        random_state=3407,
    )

    # ── load global adapter (aggregated from previous round) ──
    if args.round > 0:
        adapter_path = (
            args.global_model_path
            or os.path.join(args.output_dir, f"round_{args.round - 1}", "global_model")
        )
        print(f"[Client {args.client_id}] Loading global adapter from {adapter_path}")

        if args.flexlora and args.rank < args.global_rank:
            project_and_load_adapter(adapter_path, model, args.rank)
        else:
            from peft import load_peft_weights, set_peft_model_state_dict
            weights = load_peft_weights(adapter_path)
            set_peft_model_state_dict(model, weights)
            print(f"  Loaded global adapter directly (rank={args.rank})")

    model.print_trainable_parameters()

    # ── configure trainer ──
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":8192:8"
    os.environ["UNSLOTH_RETURN_LOGITS"]   = "1"

    if stateful:
        # Absolute max_steps grows each round so the trainer continues where
        # it left off without resetting the global step counter.
        abs_max_steps = (args.round + 1) * args.max_steps
        sft_args = SFTConfig(
            output_dir=client_ckpt_dir,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=8,
            eval_strategy="no",
            # Save a full checkpoint (optimizer+scheduler+dataloader) at the
            # end of each round so the next round can resume from it.
            save_strategy="steps",
            save_steps=args.max_steps,
            save_total_limit=1,      # keep only the latest checkpoint
            logging_steps=10,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            seed=3407 + args.client_id,
            report_to="none",
            max_length=512,
            max_steps=abs_max_steps,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )
        trainer = FLSFTTrainer(
            total_rounds=args.total_rounds,
            steps_per_round=args.max_steps,
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=sft_args,
        )
        # Locate the checkpoint from the previous round (if any)
        resume_ckpt = _find_latest_checkpoint(client_ckpt_dir) if args.round > 0 else None
        if resume_ckpt:
            print(f"[Client {args.client_id}] Resuming trainer state from {resume_ckpt}")
        elif args.round > 0:
            print(f"[Client {args.client_id}] Warning: round>0 but no checkpoint found — "
                  f"starting fresh (optimizer/scheduler will reset)")
    else:
        # Legacy: full-epoch mode, no state persistence
        sft_args = SFTConfig(
            output_dir=client_output_dir,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=8,
            num_train_epochs=args.epochs,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=10,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            seed=3407 + args.client_id,
            report_to="none",
            max_length=512,
            max_steps=-1,
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
        resume_ckpt = None

    # ── train ──
    print(f"\n[Client {args.client_id}] Starting round {args.round} training …")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── save adapter to per-round dir for aggregation ──
    model.save_pretrained(client_output_dir)
    print(f"[Client {args.client_id}] Training done. "
          f"Adapter (rank={args.rank}) saved to {client_output_dir}")


if __name__ == "__main__":
    main()
