"""A tool that fails to load has to fail the same way the next time it is asked for.

`load_tool` puts the module in `sys.modules` before running it, as the import system does.
Until IKA-40 it did not take it out again when running it raised -- which the import system
also does -- so the next call found the half-run module under its name and returned it, and
the caller read a missing attribute instead of the error that had stopped the file. On 9/19,
with torch absent, `tests/test_selection_book.py` reported two failures for one cause:

    1st  ModuleNotFoundError: No module named 'torch'
    2nd  AttributeError: module 'pokeuraou_tool_solve_selection_book' has no attribute 'book_stem'

`book_stem` exists. It is defined below the `import torch` that stopped the first run, so
the message named a real function as missing, and the issue records two misdiagnoses of
that one run made the same day.

The tools here are written to `tmp_path`, so these tests do not depend on which optional
groups are installed. Each fails at module scope above a definition, which is the shape of
`tools/solve_selection_book.py`: `import torch` on line 36, `book_stem` on line 74.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from . import _harness

#: A module no environment has, standing in for torch.
ABSENT = "ika40_absent_dependency"
BELOW = "\n\ndef defined_below() -> int:\n    return 1\n"

#: How each tool stops, what that raises, and the message that names the real cause.
FAILURES = {
    "import": (f"import {ABSENT}\n", ModuleNotFoundError, f"No module named '{ABSENT}'"),
    # Not an Exception, so a fix that caught Exception would leave this one behind. The tree
    # had the real thing: `tools/cells_needed.py` parsed its arguments at module scope, and
    # under pytest's argv that is SystemExit(2) (the tool went in IKA-212).
    "exit": ("raise SystemExit('ika40: stopped at module scope')\n", SystemExit, "ika40: stopped"),
}


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A `tools/` directory of our own.

    `load_tool` puts its directory on `sys.path`, so it gets a copy of the list. What these
    tests load is taken out of `sys.modules` afterwards, so that a harness which leaves a
    half-run module behind fails here and not in whichever test happens to run next.
    """
    monkeypatch.setattr(_harness, "_TOOLS", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield tmp_path
    for name in [n for n in sys.modules if n == ABSENT or n.startswith("pokeuraou_tool_ika40_")]:
        del sys.modules[name]


@pytest.mark.parametrize("how", sorted(FAILURES))
def test_a_tool_that_fails_to_load_fails_the_same_way_twice(tools: Path, how: str) -> None:
    """Both calls raise what stopped the file, and nothing is left under its name.

    Before IKA-40 the second call raised nothing: it returned the half-run module.
    """
    source, error, message = FAILURES[how]
    name = f"ika40_fails_by_{how}"
    (tools / f"{name}.py").write_text(source + BELOW, encoding="utf-8")

    with pytest.raises(error, match=message):
        _harness.load_tool(name)
    with pytest.raises(error, match=message):
        _harness.load_tool(name)
    assert f"pokeuraou_tool_{name}" not in sys.modules


def test_once_the_cause_is_gone_the_next_call_runs_the_whole_file(tools: Path) -> None:
    """Nothing of the failed run is kept, so the call after the cause is fixed loads it all.

    The half-run module held everything above the failing import and nothing below it,
    which is how a function that exists came to be reported missing.
    """
    name = "ika40_fails_until_installed"
    (tools / f"{name}.py").write_text(f"import {ABSENT}\n" + BELOW, encoding="utf-8")
    with pytest.raises(ModuleNotFoundError, match=ABSENT):
        _harness.load_tool(name)

    # `load_tool` put the tools directory on sys.path, so this makes the import succeed.
    (tools / f"{ABSENT}.py").write_text("", encoding="utf-8")
    importlib.invalidate_caches()
    module = _harness.load_tool(name)
    assert module.defined_below() == 1
    # And a load that succeeded is still kept: the next call is the same module.
    assert _harness.load_tool(name) is module
