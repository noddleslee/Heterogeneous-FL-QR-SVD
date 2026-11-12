"""
Unit tests for prompt formatting in train_client.py.

Verifies that yes/no questions get "? Please answer yes or no." appended
(matching eval_pathvqa.py's build_prompt logic) and that open-ended
questions are left unchanged.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# conftest.py stubs unsloth, but if run standalone we need it here too
import importlib
for _mod in ["unsloth", "unsloth.trainer"]:
    if _mod not in sys.modules:
        _m = MagicMock()
        _m.__spec__ = importlib.machinery.ModuleSpec(_mod, None)
        sys.modules[_mod] = _m
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_client import _is_yes_no, LazyPathVQADataset


# ���� _is_yes_no ����������������������������������������������������������������������������������������������������������������������������������

class TestIsYesNo:
    def test_yes_lowercase(self):
        assert _is_yes_no("yes")

    def test_yes_uppercase(self):
        assert _is_yes_no("YES")

    def test_yes_mixed(self):
        assert _is_yes_no("Yes")

    def test_yes_with_whitespace(self):
        assert _is_yes_no("  yes  ")

    def test_no_lowercase(self):
        assert _is_yes_no("no")

    def test_no_uppercase(self):
        assert _is_yes_no("NO")

    def test_no_mixed(self):
        assert _is_yes_no("No")

    def test_non_binary_maybe(self):
        assert not _is_yes_no("maybe")

    def test_non_binary_sentence(self):
        assert not _is_yes_no("The answer is yes.")

    def test_non_binary_empty(self):
        assert not _is_yes_no("")

    def test_non_binary_number(self):
        assert not _is_yes_no("1")

    def test_non_binary_partial(self):
        assert not _is_yes_no("yes please")


# ���� LazyPathVQADataset prompt formatting ����������������������������������������������������������������������������

class _FakeHFDataset:
    """Minimal stand-in for a HuggingFace Dataset with image support."""

    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        from PIL import Image
        sample = dict(self._samples[idx])
        sample["image"] = Image.new("RGB", (64, 64), color=(128, 128, 128))
        return sample


def _get_user_text(dataset, idx):
    """Extract the text part of the user turn from a LazyPathVQADataset sample."""
    item = dataset[idx]
    user_content = item["messages"][0]["content"]
    for part in user_content:
        if part["type"] == "text":
            return part["text"]
    raise AssertionError("No text found in user content")


def _get_assistant_text(dataset, idx):
    item = dataset[idx]
    return item["messages"][1]["content"][0]["text"]


class TestLazyDatasetPrompt:
    def _make_dataset(self, samples):
        return LazyPathVQADataset(_FakeHFDataset(samples))

    def test_yn_question_gets_suffix(self):
        ds = self._make_dataset([{"question": "Is there a tumor", "answer": "yes"}])
        text = _get_user_text(ds, 0)
        assert text.endswith("? Please answer yes or no."), \
            f"Expected yn suffix, got: {text!r}"

    def test_yn_no_double_question_mark(self):
        """Question already ending in '?' must not become '??'."""
        ds = self._make_dataset([{"question": "Is there a tumor?", "answer": "yes"}])
        text = _get_user_text(ds, 0)
        assert "??" not in text, f"Double question mark in: {text!r}"
        assert text.endswith("? Please answer yes or no."), f"Got: {text!r}"

    def test_yn_no_answer(self):
        ds = self._make_dataset([{"question": "Is this benign", "answer": "no"}])
        text = _get_user_text(ds, 0)
        assert text.endswith("? Please answer yes or no."), f"Got: {text!r}"

    def test_open_question_unchanged(self):
        question = "What type of cells are present?"
        ds = self._make_dataset([{"question": question, "answer": "lymphocytes"}])
        text = _get_user_text(ds, 0)
        assert text == question, \
            f"Open-ended question was modified: {text!r}"

    def test_answer_preserved_in_assistant_turn(self):
        ds = self._make_dataset([{"question": "Is this cancer?", "answer": "yes"}])
        ans = _get_assistant_text(ds, 0)
        assert ans == "yes"

    def test_messages_structure(self):
        """Output must have the expected two-turn messages structure."""
        ds = self._make_dataset([{"question": "What is shown?", "answer": "cells"}])
        item = ds[0]
        assert "messages" in item
        msgs = item["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_image_in_user_turn(self):
        """User turn must contain an image content part."""
        ds = self._make_dataset([{"question": "Describe the image.", "answer": "tissue"}])
        user_content = ds[0]["messages"][0]["content"]
        types = [p["type"] for p in user_content]
        assert "image" in types, f"No image in user content: {types}"

    def test_broken_image_fallback(self):
        """A sample with an unreadable image must not raise; fallback RGB is used."""
        from unittest.mock import patch, MagicMock
        from PIL import Image

        bad_image = MagicMock()
        bad_image.convert.side_effect = OSError("corrupt image")

        class BrokenHFDataset:
            def __len__(self): return 1
            def __getitem__(self, idx):
                return {"question": "Q", "answer": "no", "image": bad_image}

        ds = LazyPathVQADataset(BrokenHFDataset())
        item = ds[0]  # must not raise
        image_part = next(p for p in item["messages"][0]["content"] if p["type"] == "image")
        assert isinstance(image_part["image"], Image.Image)
        assert image_part["image"].mode == "RGB"
