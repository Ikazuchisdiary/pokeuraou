"""IKA-360: the entry points a person starts put torch's OpenMP threads to sleep between
regions (``KMP_BLOCKTIME=0``, `pokeuraou.openmp`) before anything loads torch.

What these hold:

- **before torch, by the source**: each entry point's module-level code calls
  `openmp.quiet_wait()` before any import but the standard library's and the call's own
  (a call after another import would hold only while that import stays torch-free);
- **before torch, when run**: importing each entry point in a fresh process leaves
  ``KMP_BLOCKTIME`` at 0 with torch not yet loaded; a value already set is kept;
- the source check fails on the shape it names (the call moved below the imports, or
  missing), so it is not vacuous.

The measurement (the same games, 4.3 times less CPU) is in records/IKA-360.md.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ("tools/play_human.py", "tools/analyze.py", "tools/play.py")


def _is_call(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "openmp.quiet_wait")


def _harmless(node: ast.stmt) -> bool:
    """An import that cannot load torch: the standard library, ``__future__``, and the
    call's own ``from pokeuraou import openmp``."""
    if isinstance(node, ast.ImportFrom):
        if node.module == "pokeuraou" and [a.name for a in node.names] == ["openmp"]:
            return True
        return (node.module or "").split(".")[0] in sys.stdlib_module_names | {"__future__"}
    assert isinstance(node, ast.Import)
    return all(a.name.split(".")[0] in sys.stdlib_module_names for a in node.names)


def quiet_first(source: str) -> list[str]:
    """What is wrong with this entry point's order: an import that could load torch before
    `openmp.quiet_wait()`, or no call at all."""
    problems = []
    for node in ast.parse(source).body:
        if _is_call(node):
            return problems
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not _harmless(node):
            problems.append(f"line {node.lineno} imports {ast.unparse(node)!r} first")
    return [*problems, "never calls openmp.quiet_wait()"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_entry_quiets_openmp_before_any_import_that_could_load_torch(entry: str) -> None:
    assert quiet_first((ROOT / entry).read_text(encoding="utf-8")) == []


def test_the_order_check_fails_on_the_shape_it_names() -> None:
    source = (ROOT / "tools/play_human.py").read_text(encoding="utf-8")
    call = "openmp.quiet_wait()  # before anything loads torch (IKA-360)\n"
    assert source.count(call) == 1
    late = source.replace(call, "").replace(
        "from pokeuraou.teams import load_roster  # noqa: E402\n",
        "from pokeuraou.teams import load_roster  # noqa: E402\n" + call,
    )
    assert any("from pokeuraou import analysis" in p for p in quiet_first(late))
    assert quiet_first(source.replace(call, "")) [-1] == "never calls openmp.quiet_wait()"


_PROBE = """
import importlib.util, json, os, sys
# As when run as a script: its own directory on the path (analyze imports play_human).
sys.path.insert(0, os.path.dirname(sys.argv[1]))
spec = importlib.util.spec_from_file_location("entry_under_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules["entry_under_test"] = module
spec.loader.exec_module(module)
from pokeuraou import openmp
print(json.dumps({"blocktime": os.environ.get("KMP_BLOCKTIME"),
                  "beforeTorch": openmp.before_torch, "torch": "torch" in sys.modules}))
"""


def _probe(entry: str, env_value: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "KMP_BLOCKTIME"}
    if env_value is not None:
        env["KMP_BLOCKTIME"] = env_value
    env["PYTHONPATH"] = str(ROOT / "src")
    got = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE, str(ROOT / entry)], env=env, cwd=str(ROOT),
        capture_output=True, text=True, check=True, timeout=120,
    )
    return json.loads(got.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("entry", ENTRIES)
def test_importing_the_entry_sets_the_wait_before_torch_loads(entry: str) -> None:
    got = _probe(entry, None)
    assert got == {"blocktime": "0", "beforeTorch": True, "torch": False}


def test_a_wait_already_set_is_kept() -> None:
    assert _probe("tools/play_human.py", "50")["blocktime"] == "50"
