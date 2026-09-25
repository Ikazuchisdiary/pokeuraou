"""Picking a stopped `generate_queue.py` run back up (IKA-77).

A game is fixed by (run seed, gameIndex), so a game played twice is the same game twice --
checked on the M-C pool: 600 games replayed from the shared selection store came out
600/600 identical. That makes the restart simple: start again from the first missing
index into a NEW directory, then fold that directory back in, dropping the indices the
first run had already written. The queue hands indices out in order, so what is written
past the first gap is at most about one game per worker.

    uv run python tools/queue_restart.py where data/selfplay-mc0 40000
    # -> first missing K = 31234 ...  resume with: --first-game 31234 --games 8766
    uv run python tools/queue_restart.py merge data/selfplay-mc0 data/selfplay-mc0-r1 r1

`where` also drops a torn last line (a worker killed mid-write) and keeps the torn bytes
beside the file as `<name>.jsonl.torn`, which no reader globs. `merge` moves the resume
run's files in as `games-<tag>-workerN.jsonl` and keeps the dropped duplicates in
`<resume dir>/dropped-duplicates.jsonl`. Nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _index(raw: bytes, path: Path) -> int:
    try:
        return int(json.loads(raw)["gameIndex"])
    except (ValueError, KeyError) as error:
        raise SystemExit(f"{path}: a bad line that is not the last one; look by hand") from error


def where(out: Path, total: int) -> int:
    """Prints the first missing index and returns it, after repairing torn last lines."""
    seen: Counter = Counter()
    for path in sorted(out.glob("*.jsonl")):
        data = path.read_bytes()
        keep = data if data.endswith(b"\n") or not data else data[: data.rfind(b"\n") + 1]
        seen.update(_index(raw, path) for raw in keep.splitlines() if raw.strip())
        if keep != data:
            torn = path.with_name(path.name + ".torn")
            torn.write_bytes(data[len(keep):])
            path.write_bytes(keep)
            print(f"  {path.name}: dropped a torn last line ({len(data) - len(keep)} bytes, "
                  f"kept in {torn.name})")
    _repair_rank_files(out)
    first = next((i for i in range(total) if i not in seen), total)
    ahead = sum(1 for i in seen if i >= first)
    duplicates = sum(n - 1 for n in seen.values() if n > 1)
    print(f"written {len(seen)} distinct of {total} (duplicate lines {duplicates})")
    print(f"first missing K = {first}; already written at or past K: {ahead}")
    if first < total:
        print(f"resume with: --first-game {first} --games {total - first}")
    return first


def _repair_rank_files(out: Path) -> None:
    """IKA-278: a torn last gzip member of a rank file, kept beside as ``.torn``."""
    from pokeuraou.rank_scores import repair

    for path in sorted(out.glob("rank-*.jsonl.gz")):
        cut = repair(path)
        if cut:
            print(f"  {path.name}: dropped a torn last member ({cut} bytes, "
                  f"kept in {path.name}.torn)")


def _move_rank_files(out: Path, resume: Path, tag: str) -> None:
    """IKA-278: the resume run's rank files, moved in whole under the tag.

    Not filtered: a replayed index is the same game, and `rank_scores.iter_records` keeps
    the first line of an index.
    """
    sources = sorted(resume.glob("rank-worker*.jsonl.gz"))
    targets = [out / path.name.replace("rank-", f"rank-{tag}-", 1) for path in sources]
    clash = [t for t in targets if t.exists()]
    if clash:
        raise SystemExit(f"{clash[0]} exists; pick another tag")
    for path, target in zip(sources, targets, strict=True):
        path.replace(target)
    if sources:
        print(f"moved {len(sources)} rank files into {out}")


def merge(out: Path, resume: Path, tag: str) -> int:
    """Moves `resume`'s games into `out`, dropping indices `out` already has."""
    have = {
        _index(raw, path)
        for path in out.glob("*.jsonl")
        for raw in path.read_bytes().splitlines()
        if raw.strip()
    }
    sources = sorted(resume.glob("games-worker*.jsonl"))
    targets = [out / path.name.replace("games-", f"games-{tag}-", 1) for path in sources]
    clash = [t for t in targets if t.exists()]
    if clash:
        raise SystemExit(f"{clash[0]} exists; pick another tag")
    dropped = bytearray()
    kept = 0
    for path, target in zip(sources, targets, strict=True):
        keep = bytearray()
        for raw in path.read_bytes().splitlines():
            if not raw.strip():
                continue
            index = _index(raw, path)
            if index in have:
                dropped += raw + b"\n"
                continue
            have.add(index)
            keep += raw + b"\n"
            kept += 1
        target.write_bytes(bytes(keep))
        path.unlink()
    (resume / "dropped-duplicates.jsonl").write_bytes(bytes(dropped))
    _move_rank_files(out, resume, tag)
    print(f"moved {len(sources)} files ({kept} games) into {out}; dropped "
          f"{dropped.count(b'\n')} duplicates; {len(have)} distinct games now")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    w = sub.add_parser("where", help="first missing game index, torn lines repaired")
    w.add_argument("out", type=Path)
    w.add_argument("games", type=int, help="games in the whole run")
    m = sub.add_parser("merge", help="fold a resume directory into the run")
    m.add_argument("out", type=Path)
    m.add_argument("resume", type=Path)
    m.add_argument("tag", help="file prefix for the moved files, e.g. r1")
    args = ap.parse_args()
    if args.command == "where":
        where(args.out, args.games)
    else:
        merge(args.out, args.resume, args.tag)


if __name__ == "__main__":
    main()
