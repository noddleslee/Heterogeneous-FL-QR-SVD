"""
pytest configuration: stub heavy optional imports and set up sys.path.

This runs before any test file is collected, so the stubs are in place
when test modules do their top-level imports.
"""
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ── stub unsloth (requires GPU / custom CUDA kernels, not available in tests) ──
# We must set __spec__ on the mock module so that importlib.util.find_spec()
# (called by trl's import machinery) does not crash with
# "ValueError: unsloth.__spec__ is not set".
_UNSLOTH_MODULES = ["unsloth", "unsloth.trainer"]
for _mod_name in _UNSLOTH_MODULES:
    if _mod_name not in sys.modules:
        _mock = MagicMock()
        _mock.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None)
        sys.modules[_mod_name] = _mock

# ── make project root importable as a package ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
