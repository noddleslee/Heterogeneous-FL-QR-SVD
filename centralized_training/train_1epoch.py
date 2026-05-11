"""
Standalone centralized fine-tuning on full PathVQA for rank comparison.

Usage:
    python train_1epoch.py --rank 16 --num_epochs 2 --output_dir /path/to/rank16
    python train_1epoch.py --rank 64 --num_epochs 2 --output_dir /path/to/rank64

Produces:
  - Trained LoRA adapter checkpoints
  - centralized_training/results/<run_name>/*: train/eval summaries
  - submit_eval.sh: evaluation job script for eval_pathvqa.py
"""

import argparse, json, os, re, subprocess, warnings
from collections import Counter
from pathlib import Path

import torch.distributed as _dist


def _is_main_process() -> bool:
    return not _dist.is_initialized() or _dist.get_rank() == 0

import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from tqdm import tqdm
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Truncated File Read")

_CACHE_DIR = "/mimer/NOBACKUP/groups/bloom/shenghui/hfl/.dataset_cache"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── dataset helpers ────────────────────────────────────────────────────────────

def _is_valid_image(sample):
    try:
        sample["image"].convert("RGB").load()
        return True
    except Exception:
        return False


def _to_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "text",  "text": sample["question"]},
                {"type": "image", "image": sample["image"].convert("RGB")},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["answer"]},
            ]},
        ]
    }


class LazyPathVQADataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.hf_ds = hf_ds

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        sample = self.hf_ds[idx]
        return _to_conversation(sample)


def load_split(split: str):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    filter_cache = os.path.join(_CACHE_DIR, f"pathvqa_{split}_filtered.arrow")

    print(f"Loading PathVQA {split} split …")
    ds = load_dataset("flaviagiammarino/path-vqa", split=split)
    print(f"  Raw size: {len(ds)}")

    ds = ds.filter(_is_valid_image, num_proc=8,
                   cache_file_name=filter_cache, desc=f"Filtering {split}")
    print(f"  After filter: {len(ds)}")
    return LazyPathVQADataset(ds)


def _latest_checkpoint(output_dir: str) -> str | None:
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def _results_dir() -> Path:
    _RESULTS_DIR.mkdir(exist_ok=True)
    return _RESULTS_DIR


def _result_run_dir(output_dir: str, num_epochs: int, rank: int) -> Path:
    output_name = Path(output_dir).name
    run_name = output_name if output_name else f"{num_epochs}epoch_rank{rank}"
    run_dir = _results_dir() / run_name
    run_dir.mkdir(exist_ok=True)
    return run_dir


def _submit_eval_job(base_model: str, checkpoint_dir: str) -> str | None:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    output_dir = os.path.dirname(checkpoint_dir)
    run_dir = _result_run_dir(output_dir, 0, 0)
    rank_match = re.search(r"rank(\d+)", output_dir)
    rank_suffix = rank_match.group(1) if rank_match else "ckpt"
    ckpt_name = Path(checkpoint_dir).name
    script_path = os.path.join(output_dir, f"submit_eval_{ckpt_name}.sh")
    log_path = run_dir / f"eval_{ckpt_name}_%j.out"
    output_json = run_dir / f"pathvqa_eval_{ckpt_name}.json"
    tmp_dir = os.path.join(output_dir, ".eval_tmp")
    inductor_dir = os.path.join(output_dir, ".torchinductor")
    triton_dir = os.path.join(output_dir, ".triton")

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(inductor_dir, exist_ok=True)
    os.makedirs(triton_dir, exist_ok=True)

    with open(script_path, "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH -A NAISS2025-22-1535 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:1
#SBATCH --job-name=eval_r{rank_suffix}
#SBATCH --tasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output={str(log_path)}

ml load CUDA/12.6.0
ml is-loaded GCCcore/13.3.0 && ml swap GCCcore/13.3.0 GCCcore/12.3.0 || ml load GCCcore/12.3.0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_RETURN_LOGITS=1
export TMPDIR={tmp_dir}
export TORCHINDUCTOR_CACHE_DIR={inductor_dir}
export TRITON_CACHE_DIR={triton_dir}

/mimer/NOBACKUP/groups/bloom/shenghui/miniconda3/envs/visper/bin/python /mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/evaluation/eval_pathvqa.py \\
    --base_model {base_model} \\
    --peft_path {checkpoint_dir} \\
    --output_json {str(output_json)}
"""
        )

    os.chmod(script_path, 0o755)
    try:
        submission = subprocess.check_output(["sbatch", script_path], text=True).strip()
        print(f"Submitted async evaluation job: {submission}")
        parts = submission.split()
        return parts[-1] if parts and parts[-1].isdigit() else None
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: failed to submit eval job: {exc}")
        return None


# ── VQA eval helpers ───────────────────────────────────────────────────────────

def _normalise(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip()).strip()

def _is_yn(answer):
    return _normalise(answer) in ("yes", "no")

def _bleu1(pred, ref):
    pt = _normalise(pred).split(); rt = _normalise(ref).split()
    if not pt: return 0.0
    rc = Counter(rt)
    return sum(min(c, rc[t]) for t, c in Counter(pt).items()) / len(pt)

def _extract_answer(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    for line in raw.strip().splitlines():
        if line.strip(): return line.strip()
    return raw.strip()

def _yn_logit_score(model, tokenizer, inputs):
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    yes_ids = _tok.encode("yes", add_special_tokens=False)
    no_ids  = _tok.encode("no",  add_special_tokens=False)
    if not yes_ids or not no_ids: return ""
    gen_out = model.generate(
        **inputs, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_scores=True, use_cache=True,
    )
    logits = gen_out.scores[0][0]
    return "yes" if logits[yes_ids[0]] >= logits[no_ids[0]] else "no"


@torch.inference_mode()
def run_vqa_eval(model, tokenizer, eval_ds):
    """Run VQA generation eval on the full evaluation split."""
    FastVisionModel.for_inference(model)

    yn_c, yn_t = 0, 0
    op_em, op_bl, op_t = 0, 0.0, 0
    per_sample = []

    _n_errors = 0
    for sample in tqdm(eval_ds, desc="VQA eval"):
        ref = sample["messages"][1]["content"][0]["text"]
        try:
            raw_img = sample["messages"][0]["content"][1]["image"]
            # Arrow cache may deserialize PIL Image as bytes or dict — normalise back
            if isinstance(raw_img, dict) and "bytes" in raw_img:
                from io import BytesIO
                img = Image.open(BytesIO(raw_img["bytes"])).convert("RGB")
            elif isinstance(raw_img, bytes):
                from io import BytesIO
                img = Image.open(BytesIO(raw_img)).convert("RGB")
            else:
                img = raw_img.convert("RGB")
            q   = sample["messages"][0]["content"][0]["text"]
            is_yn = _is_yn(ref)
            if is_yn:
                q = q.rstrip("?. ") + "? Please answer yes or no."
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": q}]}]
            input_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = tokenizer(img, input_text, add_special_tokens=False,
                               return_tensors="pt").to(model.device)
            if is_yn:
                pred = _yn_logit_score(model, tokenizer, inputs)
            else:
                ids = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
                pred = _extract_answer(tokenizer.decode(ids[0][inputs["input_ids"].shape[-1]:],
                                                        skip_special_tokens=True))
        except Exception as e:
            if _n_errors < 3:
                print(f"\n[eval error] {type(e).__name__}: {e}")
            _n_errors += 1
            pred = ""

        # log first 5 predictions for debugging
        if len(per_sample) < 5:
            print(f"\n[eval sample] type={'yn' if _is_yn(ref) else 'open'} ref={repr(ref)} pred={repr(pred)}")

        if _is_yn(ref):
            yn_t += 1
            correct = (_normalise(pred) == _normalise(ref))
            yn_c += correct
            per_sample.append({"type":"yn","ref":ref,"pred":pred,"correct":correct})
        else:
            op_t += 1
            em   = int(_normalise(pred) == _normalise(ref))
            bl   = _bleu1(pred, ref)
            op_em += em; op_bl += bl
            per_sample.append({"type":"open","ref":ref,"pred":pred,"em":em,"bleu1":round(bl,4)})

    summary = {
        "yn_accuracy":      round(yn_c / yn_t if yn_t else 0, 4),
        "yn_total":         yn_t,
        "open_exact_match": round(op_em / op_t if op_t else 0, 4),
        "open_bleu1":       round(op_bl / op_t if op_t else 0, 4),
        "open_total":       op_t,
        "overall_accuracy": round((yn_c + op_em) / (yn_t + op_t) if (yn_t+op_t) else 0, 4),
        "total_samples":    yn_t + op_t,
    }
    return summary, per_sample


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",       type=int, default=64)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_seq_len",type=int, default=512)
    parser.add_argument("--max_steps",  type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            f"/mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/"
            f"centralized_training/{args.num_epochs}epoch_rank{args.rank}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load data ──
    train_ds = load_split("train")
    print(f"\nTrain: {len(train_ds)}")

    # ── load model ──
    print(f"\nLoading model: {args.base_model}  rank={args.rank}")
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
    model.print_trainable_parameters()

    # ── train ──
    eff_batch = args.batch_size
    total_steps = len(train_ds) // eff_batch
    print(
        f"\n{args.num_epochs}-epoch training: "
        f"{len(train_ds)} samples / batch {eff_batch} = {total_steps} steps/epoch"
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":8192:8"
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=8,
            num_train_epochs=args.num_epochs,
            eval_strategy="no",
            save_strategy="epoch",
            logging_steps=1,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            seed=3407,
            report_to="none",
            save_total_limit=2,
            disable_tqdm=False,
            remove_unused_columns=True,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            max_length=args.max_seq_len,
            max_steps=args.max_steps,
            label_names=["labels"],
        ),
    )

    print("\nStarting training …")
    train_result = trainer.train()
    print(f"Training done. Runtime: {train_result.metrics['train_runtime']:.1f}s")

    if not _is_main_process():
        return

    # ── collect train loss curve from trainer state ──
    train_log = [
        {"step": e["step"], "loss": e["loss"]}
        for e in trainer.state.log_history
        if "loss" in e and "eval_loss" not in e
    ]

    # ── collect all saved epoch checkpoints and submit eval for each ──
    all_ckpts = sorted(
        Path(args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    eval_job_ids = {}
    for ckpt in all_ckpts:
        job_id = _submit_eval_job(args.base_model, str(ckpt))
        eval_job_ids[ckpt.name] = job_id

    latest_ckpt = str(all_ckpts[-1]) if all_ckpts else None

    # ── save results ──
    run_dir = _result_run_dir(args.output_dir, args.num_epochs, args.rank)
    results = {
        "rank":             args.rank,
        "num_epochs":       args.num_epochs,
        "base_model":       args.base_model,
        "train_samples":    len(train_ds),
        "batch_size":       args.batch_size,
        "train_metrics":    train_result.metrics,
        "train_loss_curve": train_log,
        "checkpoints":      [str(c) for c in all_ckpts],
        "final_checkpoint": latest_ckpt,
        "eval_job_ids":     eval_job_ids,
    }
    out_path = run_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"  rank={args.rank}  {args.num_epochs}-epoch results")
    print("="*60)
    print(f"  Train loss (final): {train_log[-1]['loss'] if train_log else 'N/A'}")
    print(f"  Saved checkpoints:  {[c.name for c in all_ckpts]}")
    for ckpt_name, job_id in eval_job_ids.items():
        print(f"  Eval job {ckpt_name}: {job_id or 'submission failed'}")
    print(f"  Results saved → {str(out_path)}")
    print("="*60)


if __name__ == "__main__":
    main()
