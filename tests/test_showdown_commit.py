"""The Showdown commit a dump records is the vendor submodule's gitlink (IKA-152).

`meta.showdownCommit` is how a reader tells which dex produced a dump. The old builders ran
`git -C vendor/pokemon-showdown rev-parse HEAD`; in a worktree whose submodule was never
initialised that directory is not a repository, git walks up, and the *parent's* HEAD was
written as if it were Showdown's. So:

- every committed dump's commit is checked against the gitlink the parent records;
- the builders' resolver (packages/sim-bridge/src/showdown-commit.ts) is run, when it has
  been built, on the two shapes that must stop: a directory that git answers for with the
  parent (the uninitialised submodule), and a checkout of its own at another commit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = "vendor/pokemon-showdown"
RESOLVER = ROOT / "packages" / "sim-bridge" / "dist" / "showdown-commit.js"


def git(cwd: Path, *args: str) -> str:
    if shutil.which("git") is None:  # pragma: no cover - depends on the machine
        pytest.skip("no git on PATH")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    out = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, check=True, timeout=60)
    return out.stdout.decode("utf-8").strip()


def gitlink() -> str:
    line = git(ROOT, "ls-files", "--stage", "--", SUBMODULE)
    mode, sha, _stage_path = line.split(" ", 2)
    assert mode == "160000", line
    return sha


def dumps() -> list[Path]:
    """Tracked JSON under configs/ that records a Showdown commit (hand-written files do not)."""
    tracked = git(ROOT, "ls-files", "--", "configs").splitlines()
    out = []
    for p in tracked:
        if not p.endswith(".json"):
            continue
        meta = json.loads((ROOT / p).read_text(encoding="utf-8")).get("meta")
        if isinstance(meta, dict) and "showdownCommit" in meta:
            out.append(ROOT / p)
    return out


def test_there_are_dumps_to_check() -> None:
    names = {p.name for p in dumps()}
    assert {"gen9championsvgc2026regmc.json", "gen9championsvgc2026regmb.json", "ja.json"} <= names


@pytest.mark.parametrize("path", dumps(), ids=lambda p: p.name)
def test_committed_dump_names_the_vendor_gitlink(path: Path) -> None:
    meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
    assert meta["showdownCommit"] == gitlink(), path


def resolve(showdown_dir: Path) -> subprocess.CompletedProcess[str]:
    if shutil.which("node") is None:
        pytest.skip("no node on PATH")
    if not RESOLVER.exists():
        pytest.skip("packages/sim-bridge is not built (npm run build)")
    script = (
        "const s = require(process.argv[1]);"
        "try { console.log(s.showdownCommit(process.argv[2], process.argv[3])); }"
        "catch (e) { console.error(e.message); process.exit(3); }"
    )
    return subprocess.run(
        ["node", "-e", script, str(RESOLVER), str(ROOT), str(showdown_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_directory_the_parent_answers_for_stops() -> None:
    """The uninitialised-submodule shape: git would answer with the parent's HEAD."""
    out = resolve(ROOT / "packages")
    assert out.returncode == 3, out
    assert "not a git checkout of its own" in out.stderr
    assert git(ROOT, "rev-parse", "HEAD") not in out.stdout


def test_a_checkout_at_another_commit_stops(tmp_path: Path) -> None:
    other = tmp_path / "showdown"
    other.mkdir()
    git(other, "init", "-q")
    git(other, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    out = resolve(other)
    assert out.returncode == 3, out
    assert gitlink() in out.stderr


def test_an_initialised_submodule_gives_the_gitlink() -> None:
    vendor = ROOT / SUBMODULE
    if not (vendor / ".git").exists():
        pytest.skip("vendor submodule not initialised in this checkout")
    out = resolve(vendor)
    assert out.returncode == 0, out
    assert out.stdout.strip() == gitlink()
