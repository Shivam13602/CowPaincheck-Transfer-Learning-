"""
The evaluator file is named `evaluate_test_set_v2.9_cli.py` (not a valid Python module name for
`import ...` because of the dot). Load it via importlib from sibling path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_evaluate_test_cli_module() -> ModuleType:
    path = Path(__file__).resolve().parent / "evaluate_test_set_v2.9_cli.py"
    name = "ucaps_evaluate_test_cli"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
