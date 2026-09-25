"""tools/queue_restart.py: where a stopped run resumes, and folding the resume back (IKA-77)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("queue_restart", ROOT / "tools" / "queue_restart.py")
queue_restart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue_restart)


def game(index: int, tag: str = "a") -> bytes:
    return json.dumps({"gameIndex": index, "run": tag}).encode() + b"\n"


def write(path: Path, indices: list[int], tag: str = "a", torn: bytes = b"") -> None:
    path.write_bytes(b"".join(game(i, tag) for i in indices) + torn)


def test_where_finds_the_first_gap_and_repairs_a_torn_line(tmp_path: Path) -> None:
    write(tmp_path / "games-worker0.jsonl", [0, 2, 4, 7])
    write(tmp_path / "games-worker1.jsonl", [1, 3, 6], torn=b'{"gameIndex": 5, "deci')
    assert queue_restart.where(tmp_path, 10) == 5
    # the torn bytes moved aside, where no *.jsonl reader sees them
    assert (tmp_path / "games-worker1.jsonl").read_bytes().endswith(game(6))
    assert (tmp_path / "games-worker1.jsonl.torn").read_bytes() == b'{"gameIndex": 5, "deci'
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == [
        "games-worker0.jsonl", "games-worker1.jsonl"]
    # a second look finds the same gap and nothing more to repair
    assert queue_restart.where(tmp_path, 10) == 5


def test_where_on_a_finished_run(tmp_path: Path) -> None:
    write(tmp_path / "games-worker0.jsonl", [0, 1, 2])
    assert queue_restart.where(tmp_path, 3) == 3


def test_merge_drops_what_the_first_run_wrote(tmp_path: Path) -> None:
    out, resume = tmp_path / "out", tmp_path / "r1"
    out.mkdir()
    resume.mkdir()
    write(out / "games-worker0.jsonl", [0, 1, 2, 4], "a")
    # the resume started at K=3 and replayed 4, which the first run had
    write(resume / "games-worker0.jsonl", [3, 4], "b")
    write(resume / "games-worker1.jsonl", [5, 6], "b")
    assert queue_restart.merge(out, resume, "r1") == 3
    indices = [
        json.loads(raw)["gameIndex"]
        for p in out.glob("*.jsonl")
        for raw in p.read_bytes().splitlines()
    ]
    assert sorted(indices) == list(range(7))
    assert json.loads((resume / "dropped-duplicates.jsonl").read_bytes())["gameIndex"] == 4
    assert not list(resume.glob("games-worker*.jsonl"))
    assert (out / "games-r1-worker1.jsonl").exists()
    assert queue_restart.where(out, 7) == 7


def test_merge_refuses_a_used_tag(tmp_path: Path) -> None:
    out, resume = tmp_path / "out", tmp_path / "r1"
    out.mkdir()
    resume.mkdir()
    write(out / "games-r1-worker0.jsonl", [0])
    write(resume / "games-worker0.jsonl", [1])
    try:
        queue_restart.merge(out, resume, "r1")
    except SystemExit as stop:
        assert "exists" in str(stop)
    else:
        raise AssertionError("merge overwrote a file")
    assert (resume / "games-worker0.jsonl").exists()


def _rank_member(index: int) -> bytes:
    import gzip

    return gzip.compress(json.dumps({"gameIndex": index}).encode() + b"\n", mtime=0)


def test_the_rank_files_are_repaired_and_moved_with_the_games(tmp_path: Path) -> None:
    """IKA-278: rank-*.jsonl.gz beside the games get the same care, and no *.jsonl reader
    sees them."""
    from pokeuraou.rank_scores import iter_records

    out, resume = tmp_path / "out", tmp_path / "r1"
    out.mkdir()
    resume.mkdir()
    write(out / "games-worker0.jsonl", [0, 1])
    torn = _rank_member(2)[:7]
    (out / "rank-worker0.jsonl.gz").write_bytes(_rank_member(0) + _rank_member(1) + torn)
    assert queue_restart.where(out, 4) == 2
    assert (out / "rank-worker0.jsonl.gz.torn").read_bytes() == torn
    write(resume / "games-worker0.jsonl", [2, 3])
    (resume / "rank-worker0.jsonl.gz").write_bytes(_rank_member(2) + _rank_member(3))
    queue_restart.merge(out, resume, "r1")
    assert not list(resume.glob("rank-*"))
    assert sorted(line["gameIndex"] for line in iter_records(out)) == [0, 1, 2, 3]
    assert sorted(p.name for p in out.glob("*.jsonl")) == [
        "games-r1-worker0.jsonl", "games-worker0.jsonl"]
