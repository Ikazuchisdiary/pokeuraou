"""A tracked file may not name one machine's home directory.

The obvious cost is portability, and it is the smaller one. The larger one is that a
hardcoded path does not fail -- it succeeds against the wrong tree. `tools/` scripts run
from a worktree while their `sys.path` points at the main checkout give a run that is half
one tree and half the other, and nothing in the output says so; the same trap the editable
install sets for a test suite that forgets `PYTHONPATH`.

Ten files carried one on 2026-09-23 (IKA-76), six of them written the day before. Whoever
writes the eleventh will not be doing anything different from what those six did, so the
check is here rather than in a habit.

What to write instead, both already the house pattern:

    ROOT = Path(__file__).resolve().parents[1]      # tools/*.py
    cd "$(dirname "$0")/.."                         # tools/*.sh

The needles are assembled rather than spelled, so this file is subject to its own rule and
needs no exception for itself.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Built from pieces, and the comments do not spell them either. The first draft assembled
# the needles and then wrote each one out in the comment beside it, which this test caught
# the moment it was tracked -- it had passed until then because `git ls-files` does not
# list a file nobody has added yet, so the run that "passed" had never read it.
_DRIVE = "C:"
_USERS = "Users"
_HOME = "home"
NEEDLES = (
    f"{_DRIVE}\\{_USERS}",  # the Windows spelling, as Python and PowerShell write it
    f"{_DRIVE}/{_USERS}",  # the same with forward slashes
    f"/c/{_USERS.lower()}",  # the MSYS spelling, as bash writes it
    f"/{_HOME}/",  # a Linux home
)

# A macOS home. The bare prefix is also the tail of the Windows one, so it is matched with
# a name after it rather than by itself.
MAC_HOME = re.compile(rf"(?<![A-Za-z:])/{_USERS}/[A-Za-z0-9._-]+/")

EXEMPT = (
    # The record. TODO.md and GENERATIONS.md quote commands as they were actually run, and
    # a quotation that edits the path is a quotation that no longer matches its run.
    "TODO.md",
    "GENERATIONS.md",
    "README.md",
    "docs/",
    "rust/README.md",
    # The same argument, one step further: scratchpad/ IS the record of what was typed on
    # a particular day. Nothing there is called by anything, and rewriting it would make
    # the files disagree with the runs they document.
    "scratchpad/",
    # Generated, and not ours to edit.
    "uv.lock",
    "package-lock.json",
)

SUFFIXES = {".py", ".sh", ".toml", ".yml", ".yaml", ".json", ".cfg", ".rs", ".ts", ".js"}


def tracked_files() -> list[Path]:
    if shutil.which("git") is None:  # pragma: no cover - depends on the machine
        pytest.skip("no git on PATH, so there is no list of tracked files to check")
    out = subprocess.run(  # noqa: S603
        ["git", "ls-files"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    return [Path(line) for line in out.splitlines() if line]


def offences(paths: list[Path]) -> list[str]:
    found = []
    for rel in paths:
        if any(str(rel).replace("\\", "/").startswith(skip) or str(rel) == skip for skip in EXEMPT):
            continue
        if rel.suffix not in SUFFIXES:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            hit = next((n for n in NEEDLES if n in line), None) or (
                MAC_HOME.search(line) and "a home directory"
            )
            if hit:
                found.append(f"{rel}:{number}: {line.strip()[:100]}")
    return found


def test_the_tree_is_not_pinned_to_one_machine() -> None:
    """No tracked source file names a home directory."""
    found = offences(tracked_files())
    assert not found, (
        "these files name one machine's home directory, so they are not portable and, "
        "worse, resolve to the wrong tree from a worktree without failing:\n  "
        + "\n  ".join(found)
        + "\nUse `Path(__file__).resolve().parents[1]` in Python or "
        '`cd "$(dirname "$0")/.."` in a shell script.'
    )


def test_the_check_would_notice_one() -> None:
    """The control: a clean tree and a check that inspects nothing look identical.

    So the detector is run against a file that does carry the thing. Without this, the
    test above passes just as well with an empty needle list, an EXEMPT entry that
    swallowed `tools/`, or a suffix set that matched nothing.
    """
    planted = ROOT / "tools" / "_ika76_planted_probe.py"
    planted.write_text(
        "\n".join(("home = " + repr(f"{_DRIVE}/{_USERS}/somebody/repos/pokeuraou/src"), "")),
        encoding="utf-8",
    )
    try:
        found = offences([Path("tools/_ika76_planted_probe.py")])
    finally:
        planted.unlink()
    assert len(found) == 1, f"the detector missed a planted absolute path: {found}"
    assert "_ika76_planted_probe.py:1" in found[0]


def test_the_exemptions_are_for_files_that_exist() -> None:
    """An exemption whose path has moved is a hole nobody can see."""
    missing = [
        skip
        for skip in EXEMPT
        if not (ROOT / skip).exists() and not any((ROOT / skip.rstrip("/")).glob("*"))
    ]
    assert not missing, f"EXEMPT names paths that are not in the tree any more: {missing}"


if __name__ == "__main__":  # pragma: no cover - a hand run, for the message
    bad = offences(tracked_files())
    print("\n".join(bad) if bad else "clean")
    sys.exit(1 if bad else 0)
