"""
PathVQA evaluation for Qwen3.5-0.8B (base or fine-tuned via LoRA).

Metrics
-------
- yes/no accuracy   : binary questions (answer in {yes, no})
- open exact match  : open-ended questions, case-insensitive full-string match
- open BLEU-1       : unigram overlap (standard VQA open-ended proxy)
- overall accuracy  : yes/no acc + open exact match (averaged over all samples)

Usage
-----
# Baseline (no fine-tuning):
python eval_pathvqa.py --base_model Qwen/Qwen3.5-0.8B

# Fine-tuned checkpoint:
python eval_pathvqa.py --base_model Qwen/Qwen3.5-0.8B \
    --peft_path /path/to/checkpoint/peft_model \
    --output_json /path/to/results.json
"""

import argparse
import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from tqdm import tqdm
from unsloth import FastVisionModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Truncated File Read")


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation/whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _is_yes_no(answer: str) -> bool:
    return _normalise(answer) in ("yes", "no")


def _bleu1(pred: str, ref: str) -> float:
    """Unigram precision (BLEU-1)."""
    pred_tokens = _normalise(pred).split()
    ref_tokens  = _normalise(ref).split()
    if not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    matches = sum(min(cnt, ref_counts[tok]) for tok, cnt in Counter(pred_tokens).items())
    return matches / len(pred_tokens)


def _extract_answer(raw: str) -> str:
    """
    Strip thinking tags and extract the actual answer from model output.
    Qwen3 sometimes wraps reasoning in <think>…</think>.
    """
    # Remove thinking block
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Return first non-empty line
    for line in raw.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return raw.strip()


def _batched(items: list, batch_size: int) -> Iterable[list]:
    """Yield fixed-size chunks from a list."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


# ── dataset ───────────────────────────────────────────────────────────────────

def load_eval_samples(n_samples: int = None, seed: int = 42):
    """
    Load n_samples from the PathVQA test split, balanced between
    yes/no and open-ended questions. If n_samples is None, use all samples.
    """
    print("Loading PathVQA test split …")
    ds = load_dataset("flaviagiammarino/path-vqa", split="test")

    if n_samples is None:
        print(f"  Using all {len(ds)} samples.")
        return ds

    yn_indices   = [i for i in range(len(ds)) if _is_yes_no(ds[i]["answer"])]
    open_indices = [i for i in range(len(ds)) if not _is_yes_no(ds[i]["answer"])]

    import random
    rng = random.Random(seed)
    half = n_samples // 2
    sampled_yn   = rng.sample(yn_indices,   min(half, len(yn_indices)))
    sampled_open = rng.sample(open_indices, min(n_samples - half, len(open_indices)))

    indices = sorted(sampled_yn + sampled_open)
    subset  = ds.select(indices)

    print(f"  Sampled {len(sampled_yn)} yes/no  +  {len(sampled_open)} open-ended"
          f"  =  {len(subset)} total")
    return subset


# ── inference ─────────────────────────────────────────────────────────────────

def build_prompt(sample, yn_question: bool = False):
    """
    Return (image, question_text).

    For yes/no questions we append a format instruction so the model is
    guided to answer concisely — matching training data format and aligning
    with common practice in PathVQA / MedVQA literature (e.g. LLaVA-Med,
    Med-Flamingo).  Open-ended questions are left unchanged.
    """
    try:
        img = sample["image"].convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224))

    question = sample["question"]
    if yn_question:
        question = question.rstrip("?. ") + "? Please answer yes or no."
    return img, question


def _yn_logit_score(model, tokenizer, inputs) -> str:
    """
    Logit-based yes/no decision: compare P('yes') vs P('no') at the first
    generated token.  Uses generate(max_new_tokens=1, output_scores=True) so
    dtype handling is identical to normal generation — avoids the bfloat16/
    float32 mismatch that occurs with a raw model(**inputs) forward pass.
    """
    # Qwen3VLProcessor wraps a tokenizer; fall back to .tokenizer if needed
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    yes_ids = _tok.encode("yes", add_special_tokens=False)
    no_ids  = _tok.encode("no",  add_special_tokens=False)
    if not yes_ids or not no_ids:
        return ""
    yes_id, no_id = yes_ids[0], no_ids[0]

    gen_out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )
    # scores[0]: logits over vocab for the first generated token, shape [1, vocab]
    logits = gen_out.scores[0][0]
    return "yes" if logits[yes_id].item() >= logits[no_id].item() else "no"


def _prepare_batch(tokenizer, batch_samples):
    images, texts = [], []
    for sample in batch_samples:
        is_yn = _is_yes_no(sample["answer"])
        img, question = build_prompt(sample, yn_question=is_yn)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
        )
        images.append(img)

    return tokenizer(
        images,
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )


def _run_yn_batch(model, tokenizer, batch_samples):
    inputs = _prepare_batch(tokenizer, batch_samples).to(model.device)
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    yes_ids = _tok.encode("yes", add_special_tokens=False)
    no_ids = _tok.encode("no", add_special_tokens=False)
    if not yes_ids or not no_ids:
        return [""] * len(batch_samples)

    gen_out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )
    logits = gen_out.scores[0]
    yes_id, no_id = yes_ids[0], no_ids[0]
    return [
        "yes" if row[yes_id].item() >= row[no_id].item() else "no"
        for row in logits
    ]


def _run_open_batch(model, tokenizer, batch_samples, max_new_tokens: int):
    inputs = _prepare_batch(tokenizer, batch_samples).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    new_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    raw_outputs = _tok.batch_decode(new_ids, skip_special_tokens=True)
    return [_extract_answer(raw) for raw in raw_outputs]


def _run_buckets(model, tokenizer, bucket, batch_size: int, desc: str, batch_fn):
    if not bucket:
        return {}

    results = {}
    current_bs = min(batch_size, len(bucket))
    cursor = 0

    with tqdm(total=len(bucket), desc=desc) as pbar:
        while cursor < len(bucket):
            batch = bucket[cursor:cursor + current_bs]
            try:
                preds = batch_fn([sample for _, sample in batch])
                for (sample_idx, _), pred in zip(batch, preds):
                    results[sample_idx] = pred
                cursor += len(batch)
                pbar.update(len(batch))
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or current_bs == 1:
                    if current_bs == 1:
                        sample_idx, _ = batch[0]
                        print(f"  Warning: inference failed for sample {sample_idx} — {e}")
                        results[sample_idx] = ""
                        cursor += 1
                        pbar.update(1)
                        continue
                    current_bs = 1
                    print(f"  Batch failed in {desc}; falling back to per-sample mode — {e}")
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"  OOM in {desc}; retrying with batch_size={current_bs}")
            except Exception as e:
                if current_bs == 1:
                    sample_idx, _ = batch[0]
                    print(f"  Warning: inference failed for sample {sample_idx} — {e}")
                    results[sample_idx] = ""
                    cursor += 1
                    pbar.update(1)
                    continue
                current_bs = 1
                print(f"  Batch failed in {desc}; falling back to per-sample mode — {e}")

    return results


@torch.inference_mode()
def run_inference(
    model,
    tokenizer,
    samples,
    max_new_tokens: int = 32,
    batch_size: int = 8,
    yn_batch_size: int | None = None,
) -> list[str]:
    """
    Two-track inference:
    - Yes/No questions : logit-based (P(yes) vs P(no) at first token)
    - Open-ended       : free generation with format-guiding prompt
    """
    FastVisionModel.for_inference(model)
    yn_batch_size = yn_batch_size or max(batch_size, batch_size * 2)
    indexed_samples = list(enumerate(samples))
    yn_bucket = [(idx, sample) for idx, sample in indexed_samples if _is_yes_no(sample["answer"])]
    open_bucket = [(idx, sample) for idx, sample in indexed_samples if not _is_yes_no(sample["answer"])]

    predictions = [""] * len(samples)

    yn_results = _run_buckets(
        model,
        tokenizer,
        yn_bucket,
        batch_size=yn_batch_size,
        desc="Inference (yes/no)",
        batch_fn=lambda batch: _run_yn_batch(model, tokenizer, batch),
    )
    open_results = _run_buckets(
        model,
        tokenizer,
        open_bucket,
        batch_size=batch_size,
        desc="Inference (open)",
        batch_fn=lambda batch: _run_open_batch(model, tokenizer, batch, max_new_tokens),
    )

    for idx, pred in {**yn_results, **open_results}.items():
        predictions[idx] = pred
    return predictions


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(samples, predictions) -> dict:
    yn_correct, yn_total = 0, 0
    open_em, open_bleu, open_total = 0, 0.0, 0

    per_sample = []
    for sample, pred in zip(samples, predictions):
        ref  = sample["answer"]
        norm_pred = _normalise(pred)
        norm_ref  = _normalise(ref)

        if _is_yes_no(ref):
            yn_total += 1
            # Logit-based pred is already "yes"/"no"; exact match
            correct = (norm_pred == norm_ref)
            yn_correct += int(correct)
            per_sample.append({"type": "yn",   "ref": ref, "pred": pred, "correct": correct})
        else:
            open_total += 1
            em    = int(norm_pred == norm_ref)
            bleu  = _bleu1(pred, ref)
            open_em   += em
            open_bleu += bleu
            per_sample.append({"type": "open", "ref": ref, "pred": pred,
                                "exact_match": em, "bleu1": round(bleu, 4)})

    yn_acc    = yn_correct   / yn_total    if yn_total    else 0.0
    open_em_r = open_em      / open_total  if open_total  else 0.0
    open_bleu_r = open_bleu  / open_total  if open_total  else 0.0
    overall   = (yn_correct + open_em) / (yn_total + open_total)

    summary = {
        "yn_accuracy":        round(yn_acc,      4),
        "yn_total":           yn_total,
        "open_exact_match":   round(open_em_r,   4),
        "open_bleu1":         round(open_bleu_r, 4),
        "open_total":         open_total,
        "overall_accuracy":   round(overall,     4),
        "total_samples":      yn_total + open_total,
    }
    return summary, per_sample


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",    type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--peft_path",     type=str, default=None,
                        help="Path to LoRA adapter directory (None = base model only)")
    parser.add_argument("--n_samples",     type=str, default=None,
                        help="Number of samples (int) or 'none' to use all")
    parser.add_argument("--max_new_tokens",type=int, default=32)
    parser.add_argument("--batch_size",     type=int, default=32,
                        help="Batch size for open-ended generation")
    parser.add_argument("--yn_batch_size",  type=int, default=None,
                        help="Batch size for yes/no scoring (default: max(batch_size, 2*batch_size))")
    parser.add_argument("--output_json",   type=str, default=None,
                        help="Where to save JSON results (default: auto-named next to peft_path)")
    parser.add_argument("--load_in_4bit",  action="store_true")
    args = parser.parse_args()

    # ── output path ──
    if args.output_json is None:
        if args.peft_path:
            args.output_json = os.path.join(
                os.path.dirname(args.peft_path), "pathvqa_eval.json"
            )
        else:
            args.output_json = "pathvqa_eval_base.json"

    # ── load model ──
    print(f"\nLoading base model: {args.base_model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=False,  # inference only
    )

    if args.peft_path:
        import json
        import safetensors.torch
        from peft import get_peft_model, LoraConfig, set_peft_model_state_dict

        peft_path = os.path.abspath(args.peft_path)
        print(f"Loading LoRA adapter from: {peft_path}")

        with open(os.path.join(peft_path, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)

        lora_config = LoraConfig(
            r=adapter_cfg["r"],
            lora_alpha=adapter_cfg["lora_alpha"],
            target_modules=adapter_cfg.get("target_modules"),
            lora_dropout=adapter_cfg.get("lora_dropout", 0),
            bias=adapter_cfg.get("bias", "none"),
            task_type=adapter_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_config)

        weights = safetensors.torch.load_file(
            os.path.join(peft_path, "adapter_model.safetensors"),
            device=str(next(model.parameters()).device),
        )
        incompatible = set_peft_model_state_dict(model, weights)
        if incompatible.unexpected_keys:
            print(f"  Warning: unexpected keys: {incompatible.unexpected_keys[:5]}")

        model = model.merge_and_unload()
        print(f"  Adapter (r={adapter_cfg['r']}) merged.")

    model.eval()

    # ── load data ──
    if args.n_samples is None or args.n_samples.lower() == "none":
        n_samples = None
    else:
        n_samples = int(args.n_samples)
    samples = load_eval_samples(n_samples)

    # ── inference ──
    predictions = run_inference(
        model,
        tokenizer,
        samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        yn_batch_size=args.yn_batch_size,
    )

    # ── metrics ──
    summary, per_sample = compute_metrics(samples, predictions)

    print("\n" + "=" * 60)
    print(f"  PathVQA Evaluation Results")
    print(f"  Base model : {args.base_model}")
    print(f"  LoRA path  : {args.peft_path or '(none — base model)'}")
    print("=" * 60)
    print(f"  Yes/No accuracy   : {summary['yn_accuracy']:.4f}  ({summary['yn_total']} samples)")
    print(f"  Open exact match  : {summary['open_exact_match']:.4f}  ({summary['open_total']} samples)")
    print(f"  Open BLEU-1       : {summary['open_bleu1']:.4f}")
    print(f"  Overall accuracy  : {summary['overall_accuracy']:.4f}  ({summary['total_samples']} samples)")
    print("=" * 60)

    out = {
        "base_model":  args.base_model,
        "peft_path":   args.peft_path,
        "summary":     summary,
        "per_sample":  per_sample,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
