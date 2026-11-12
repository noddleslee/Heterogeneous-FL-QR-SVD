"""Resolve local paths without depending on the original Alvis checkout."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def python_executable():
    return os.environ.get("HFEDLORA_PYTHON") or sys.executable


def evaluation_script():
    return os.environ.get("HFEDLORA_EVAL_SCRIPT") or str(PROJECT_ROOT / "eval" / "eval_pathvqa.py")


def dataset_cache_dir():
    return os.environ.get("HFEDLORA_CACHE_DIR") or str(PROJECT_ROOT / ".dataset_cache")
