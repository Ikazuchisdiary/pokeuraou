"""Which test files the `learn` job runs, and whether they hold every torch skip of `suite`.

The `learn` job exists so that the tests which skip for want of torch run somewhere (IKA-51).
Until IKA-465 it ran the whole suite a second time with torch installed: 3,048 tests, 782 s
of pytest, of which 2,837 tests had already passed in `suite`. It now runs only the test
files this lists: every `tests/test_*.py` whose text names torch at all. That is wider than
the files that skip for it, on purpose -- a file that says "no torch" in its docstring, or
that checks torch is not loaded, is a file whose answer may move when torch is installed.

A list read from the source can miss a skip that comes from somewhere else (a helper module
that calls `importorskip`, say). So `suite` checks the list against what it actually
skipped: ``--check`` reads its junit report and fails if any skip the audit classes as torch
(`tools/ci_skip_audit.py`) lies in a file this would not hand to `learn`.

    uv run python tools/ci_torch_tests.py                      # the files, one per line
    uv run python tools/ci_torch_tests.py --check reports/pytest.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ci_skip_audit  # noqa: E402


def torch_files(root: Path = ROOT) -> list[str]:
    """The test files, as paths relative to the repository, whose text names torch."""
    return sorted(
        p.relative_to(root).as_posix()
        for p in (root / "tests").glob("test_*.py")
        if "torch" in p.read_text(encoding="utf-8")
    )


def module_file(where: str) -> str:
    """``tests.test_foo::test_bar`` or ``tests.test_foo.TestCls::test_bar`` or a module's
    own collection skip ``tests.test_foo`` -> ``tests/test_foo.py``."""
    dotted = where.split("::", 1)[0]
    parts = dotted.split(".")
    return f"{parts[0]}/{parts[1]}.py" if len(parts) >= 2 else dotted


def check(junit: Path, root: Path = ROOT) -> int:
    skips, _, _ = ci_skip_audit.parse(junit)
    listed = set(torch_files(root))
    torch = [
        (where, reason)
        for where, reason in skips
        if (cls := ci_skip_audit.classify(reason)) is not None and cls.name == "torch"
    ]
    missed = [(where, reason) for where, reason in torch if module_file(where) not in listed]
    print(f"  {len(torch)} torch skip(s) in {junit}; {len(listed)} file(s) go to `learn`")
    if missed:
        print(f"  {len(missed)} torch skip(s) in a file `learn` would not run:")
        for where, reason in missed:
            print(f"    {module_file(where)}: {where}: {reason}")
        print("  The test skips for torch without naming it in its own file; name it there.")
        return 1
    if not torch:
        print("  no torch skips at all: a suite without torch that skips none is not this job")
        return 1
    print("  every torch skip is in a file `learn` runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", type=Path, metavar="JUNIT", help="suite's pytest --junitxml")
    args = ap.parse_args(argv)
    if args.check:
        return check(args.check)
    print("\n".join(torch_files()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
