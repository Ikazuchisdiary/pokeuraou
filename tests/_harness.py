"""Loads a ``tools/`` script as a module.

The differential harnesses live under ``tools/`` because they are also standalone
commands. Importing them here keeps the tests from duplicating several hundred lines of
battle driving, and keeps the command and the test measuring the same thing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"


def load_tool(name: str) -> ModuleType:
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    module_name = f"pokeuraou_tool_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _TOOLS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
