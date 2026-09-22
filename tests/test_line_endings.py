"""Every tracked text file is LF, in the index and on disk.

`.gitattributes` does most of this by itself. With `* text=auto eol=lf`, `git add` stores
LF whatever the file on disk holds, and `git diff` compares after that conversion, so a
writer that emits CRLF no longer shows up as a whole-file change. What this checks is the
two things the attributes do not do.

- They do not rewrite the working tree. A file written CRLF over an LF one -- Python's
  text mode does that on Windows -- shows in `git status` as modified while `git diff`
  shows nothing, and once it has been added git calls the tree clean, while its bytes still
  differ from a checkout of the same commit. `engine_fingerprint` hashes those bytes:
  measured on a scratch copy, one commit with `dirty` false named two different `sources`.
- They do not convert a blob that is already CRLF. Under `text=auto`, git leaves a file
  alone when the copy in the index has a CR in it, so a CRLF blob that arrives by a merge
  from a branch cut before the normalisation stays CRLF through every later edit.

Until 2026-09-23 nothing held the endings at all: 87 files were CRLF, one was both, and 51
had changed endings at least once after they were created, TODO.md seven times. Each
change was a whole-file diff that buried the edit made with it.

To repair what this names, keeping any edit in the files:

    uv run python tests/test_line_endings.py --fix     # LF on disk, as bytes
    git add <the files it named "in the index">        # and LF in the index
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: What `git ls-files --eol` calls a file that is not purely LF.
NOT_LF = {"crlf", "mixed"}


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    if shutil.which("git") is None:  # pragma: no cover - depends on the machine
        pytest.skip("no git on PATH, so there is no list of tracked files to check")
    # Without the GIT_* variables a hook or a `rebase --exec` may have set, so that the
    # scratch repository below is the one written to and not the one the run came from.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", *args], cwd=root, env=env, capture_output=True, check=check, timeout=60
    )


def entries(root: Path) -> list[tuple[str, str, str, str]]:
    """(index, disk, attributes, path) for every tracked file, as `git ls-files --eol` has them."""
    out = git(root, "ls-files", "--eol", "-z").stdout.decode("utf-8")
    rows = []
    for entry in filter(None, out.split("\0")):
        info, _, rel = entry.partition("\t")
        index, disk, *attr = info.split()
        rows.append(
            (
                index.removeprefix("i/"),
                disk.removeprefix("w/"),
                " ".join(attr).removeprefix("attr/"),
                rel,
            )
        )
    return rows


def is_text(index: str) -> bool:
    # Empty for a submodule, `-text` for a file git has decided is binary.
    return index not in ("", "-text")


def offences(root: Path) -> list[str]:
    """Tracked text files that are not LF somewhere, or that nothing holds to LF."""
    found = []
    for index, disk, attr, rel in entries(root):
        if not is_text(index):
            continue
        wrong = []
        if index in NOT_LF:
            wrong.append(f"{index} in the index")
        if disk in NOT_LF:
            wrong.append(f"{disk} on disk")
        if "eol=lf" not in attr.split():
            wrong.append(f"attributes {attr!r}, which do not say eol=lf")
        if wrong:
            found.append(f"{rel}: " + ", ".join(wrong))
    return found


def not_lf_on_disk(root: Path) -> list[str]:
    return [rel for index, disk, _, rel in entries(root) if is_text(index) and disk in NOT_LF]


def to_lf(root: Path, rels: list[str]) -> None:
    """CRLF -> LF on disk, read and written as bytes so nothing else about the file moves."""
    for rel in rels:
        path = root / rel
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def test_every_tracked_text_file_is_lf() -> None:
    """In the index, on disk, and held there by the attributes."""
    rows = entries(ROOT)
    # This file is in its own scope. The guard IKA-76 added passed in its worktree because
    # `git ls-files` does not list a file nobody has added, so that run never read it.
    assert "tests/test_line_endings.py" in {rel for *_, rel in rows}, (
        "this file is not tracked yet, so the run has not checked it; `git add` it first"
    )
    found = offences(ROOT)
    shown = found[:30] + ([f"... and {len(found) - 30} more"] if len(found) > 30 else [])
    assert not found, (
        "these tracked files are not LF:\n  "
        + "\n  ".join(shown)
        + "\nOn disk: `uv run python tests/test_line_endings.py --fix` rewrites them as bytes, "
        "keeping any edit. In the index: `git add` them after that. Whatever wrote them wants "
        "`newline=\"\\n\"` -- text mode on Windows writes CRLF."
    )


def test_the_attributes_and_the_check_do_what_this_file_says(tmp_path: Path) -> None:
    """The control, on a scratch repository that carries this tree's `.gitattributes`.

    A clean tree and a check that reads nothing look the same, and so do attributes that
    normalise and attributes nobody consulted. So each claim in the module docstring is made
    to happen here, and then the repair it recommends is run.
    """

    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        # autocrlf and safecrlf pinned off, so whatever normalises here is the attributes,
        # and a machine whose config refuses a lossy conversion does not fail this for a
        # reason that has nothing to do with the tree.
        return git(tmp_path, "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", *args, check=check)

    run("init", "-q")
    # Added before the normalisation, the way the 87 were: CRLF in the index.
    (tmp_path / "old.py").write_bytes(b"a = 1\r\n")
    run("add", "old.py")
    shutil.copyfile(ROOT / ".gitattributes", tmp_path / ".gitattributes")
    (tmp_path / "old.py").write_bytes(b"a = 1\r\nb = 2\r\n")  # edited since, still CRLF
    (tmp_path / "new.py").write_bytes(b"a = 1\r\n")  # what text mode writes on Windows
    (tmp_path / "fine.py").write_bytes(b"a = 1\n")
    run("add", ".")

    # What the attributes do: a CRLF file with no CRLF history is stored LF...
    assert run("show", ":new.py").stdout == b"a = 1\n"
    # ...and git sees no difference between that and the CRLF still on disk: no diff, and
    # a blank second column in the status, which is the working tree against the index.
    assert run("diff", "--quiet", "--", "new.py", check=False).returncode == 0
    assert run("status", "--porcelain", "--", "new.py").stdout == b"A  new.py\n"
    # What they do not: a file whose index copy already had a CR is left as it was.
    assert run("show", ":old.py").stdout == b"a = 1\r\nb = 2\r\n"

    assert sorted(offences(tmp_path)) == [
        "new.py: crlf on disk",
        "old.py: crlf in the index, crlf on disk",
    ]

    # The repair the message recommends: bytes on disk, then an add.
    to_lf(tmp_path, not_lf_on_disk(tmp_path))
    run("add", ".")
    assert offences(tmp_path) == []
    assert run("show", ":old.py").stdout == b"a = 1\nb = 2\n"


if __name__ == "__main__":  # pragma: no cover - a hand run, for the message and the repair
    if "--fix" in sys.argv[1:]:
        fixed = not_lf_on_disk(ROOT)
        to_lf(ROOT, fixed)
        print("\n".join(f"now LF on disk: {rel}" for rel in fixed) or "nothing on disk to fix")
    bad = offences(ROOT)
    print("\n".join(bad) if bad else "clean")
    sys.exit(1 if bad else 0)
