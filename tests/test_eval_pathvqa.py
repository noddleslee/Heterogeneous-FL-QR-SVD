from unittest.mock import MagicMock

import eval_pathvqa


def test_batched_chunks():
    chunks = list(eval_pathvqa._batched(list(range(7)), 3))
    assert chunks == [[0, 1, 2], [3, 4, 5], [6]]


def test_run_inference_restores_original_order(monkeypatch):
    monkeypatch.setattr(eval_pathvqa.FastVisionModel, "for_inference", lambda model: None)

    def fake_run_buckets(model, tokenizer, bucket, batch_size, desc, batch_fn):
        if "yes/no" in desc:
            return {idx: f"yn-{sample['answer']}" for idx, sample in bucket}
        return {idx: f"open-{sample['answer']}" for idx, sample in bucket}

    monkeypatch.setattr(eval_pathvqa, "_run_buckets", fake_run_buckets)

    samples = [
        {"answer": "yes", "question": "Q1"},
        {"answer": "cells", "question": "Q2"},
        {"answer": "no", "question": "Q3"},
        {"answer": "tumor", "question": "Q4"},
    ]

    preds = eval_pathvqa.run_inference(
        model=MagicMock(),
        tokenizer=MagicMock(),
        samples=samples,
        max_new_tokens=16,
        batch_size=4,
        yn_batch_size=8,
    )

    assert preds == ["yn-yes", "open-cells", "yn-no", "open-tumor"]
