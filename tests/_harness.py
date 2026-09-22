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
    # In `sys.modules` while it runs, as the import system does: a dataclass defined under
    # `from __future__ import annotations` looks its module up there, and several tools
    # have them. Out again if running it raises anything -- SystemExit included, which is
    # what a tool that parses its arguments at import raises under pytest -- as the import
    # system also does. Left there, it is what the check above hands the next caller: a
    # half-run module, so the error that stopped the file reads as a missing attribute
    # (IKA-40: `book_stem` "missing" below a failed `import torch`).
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
