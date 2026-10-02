"""Positions where two conditions play different moves, to begin a match from (IKA-423).

A match of ordinary games shows a change in how a ladder reads a deep cell only where it
changes a move, which is about 1.5% of the moves (IKA-422: 200 games, not one move apart).
This tool finds the positions where two conditions' answers differ, and `tools/time_match.py
--starts` begins its pairs there.

    uv run python tools/targeted_positions.py candidates --set C:/tmp/ika421/set \\
        --out C:/tmp/ika423/cand.jsonl
    uv run python tools/time_match.py --starts C:/tmp/ika423/cand.jsonl --probe --pairs N \\
        --arm A:seconds=4,... --arm B:seconds=4,... --out C:/tmp/ika423/probe-AB ...
    uv run python tools/targeted_positions.py select --starts C:/tmp/ika423/cand.jsonl \\
        --probe C:/tmp/ika423/probe-AB --out C:/tmp/ika423/sel-AB.jsonl
    uv run python tools/time_match.py --starts C:/tmp/ika423/sel-AB.jsonl --pairs M \\
        --arm A:... --arm B:... --out C:/tmp/ika423/match-AB ...

* **candidates**: the positions of a position set (`tools/depth_outcome.py build`: one
  move decision of a recorded game, with the identities each side had shown) where both sides
  have shown all four of their Pokemon. A bench nobody has seen is the one thing a start
  cannot restore (the recorded game's belief is not in the position), so such a position is
  left out: the game from it is the open game for both seats, and no belief is drawn. Each
  line is a `humanplay.GameStart` (`id`, `teams`, `picks`, `position`, `seenIds`) plus where
  it came from. The set's order is kept.
* **select**: from a probe run (`time_match --probe`: each candidate's first move read by both
  seats under both conditions: the two games of a pair are the four reads), the candidates
  where the two conditions' most played moves differ. **Both seats are read under both
  conditions, and a candidate is kept when they differ on either seat**, so the rule asks
  nothing of which condition is which: swapping the two names keeps the same positions (the
  test). A difference that is a tie between equal moves (the other condition's answer plays
  the move as much as its own top, to 1e-6) is not a difference. The kept lines carry both
  conditions' reads (``probe``) for a reader.
* **summary**: how many candidates differ, on which seat, and how far apart the mixtures are
  (total variation).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Two moves whose probabilities differ by less than this are a tie, not a difference.
TIE = 1e-6


def _write_lines(path: Path, lines: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    path.write_bytes(body.encode("utf-8"))  # LF on every platform (tests/test_line_endings.py)
    return body.count("\n")


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(raw) for raw in path.read_bytes().splitlines() if raw.strip()]


# ------------------------------------------------------------------------------ candidates


def start_line(entry: dict[str, Any], teams: Sequence[Any], reg: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """The start of one position-set entry, or None when a side has not shown all four."""
    from pokeuraou.hidden import identity
    from pokeuraou.position import Position
    from pokeuraou.regulation import to_id

    pos = Position.from_json(entry["position"])
    seen_ids: list[list[str]] = []
    picks: list[list[int]] = []
    for side in (0, 1):
        mons = pos.sides[side].pokemon
        if len(entry["seen"][side]) != len(mons):
            return None  # a bench nobody has seen
        sheet = [to_id(s.species) for s in teams[side].sets]
        slots = []
        for mon in sorted(mons, key=lambda m: m.slot):
            ident = identity(mon)
            if ident not in sheet:
                return None  # the six does not hold it (a form the sheet spells otherwise)
            slots.append(sheet.index(ident))
        if len(set(slots)) != len(slots):
            return None
        seen_ids.append(sorted(identity(m) for m in mons))
        picks.append(slots)
    return {
        "id": f"{entry['game']}-t{entry['turn']}", "teams": list(entry["teams"]),
        "picks": picks, "seenIds": seen_ids, "turn": entry["turn"], "game": entry["game"],
        "outcome": entry.get("outcome"), "position": entry["position"],
    }


def candidates(args: argparse.Namespace) -> int:
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.pool import load_pool

    pool = load_pool(args.pool)
    register_mega_stones(pool.reg)
    roster = {t.id: t for t in pool.teams}
    entries = read_json_lines(Path(args.set) / "positions.jsonl")
    out, dropped = [], {"hidden bench": 0, "turn": 0}
    for entry in entries:
        if entry["turn"] < args.min_turn:
            dropped["turn"] += 1
            continue
        line = start_line(entry, [roster[i] for i in entry["teams"]], pool.reg)
        if line is None:
            dropped["hidden bench"] += 1
            continue
        out.append(line)
        if args.limit and len(out) >= args.limit:
            break
    n = _write_lines(Path(args.out), out)
    print(f"{len(entries)} positions read, {n} candidates (both sides have shown all four), "
          f"left out {dropped} -> {args.out}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------------------ select


def _mixture(read: dict[str, Any]) -> dict[str, float]:
    return dict(zip(read["menu"], read["p"], strict=True))


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Two reads of one seat at one position, one under each condition: do their most played
    moves differ, and how far apart are the mixtures. Symmetric in ``a`` and ``b``."""
    ma, mb = _mixture(a), _mixture(b)
    tv = 0.5 * sum(abs(ma.get(k, 0.0) - mb.get(k, 0.0)) for k in set(ma) | set(mb))
    differ = a["top"] != b["top"] and (
        mb.get(a["top"], 0.0) < b["topP"] - TIE or ma.get(b["top"], 0.0) < a["topP"] - TIE
    )
    return {"differ": bool(differ), "tv": round(tv, 6), "tops": [a["top"], b["top"]],
            "tie": a["top"] != b["top"] and not differ}


def reads_of(lines: Sequence[dict[str, Any]], tested: str, other: str) -> dict[int, dict[str, Any]]:
    """By pair: ``{(condition, seat): read}`` from a probe run's game lines."""
    out: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    for line in lines:
        if "probe" not in line:
            raise SystemExit("a line without 'probe': this is not a --probe run")
        for seat_text, read in line["probe"].items():
            seat = int(seat_text)
            out.setdefault(int(line["pair"]), {})[line["conditions"][seat], seat] = read
    names = {tested, other}
    for pair, got in out.items():
        if {name for name, _seat in got} != names or len(got) != 4:
            raise SystemExit(f"pair {pair}: reads of {sorted(got)} are not each condition on each seat")
    return out


def select_positions(
    starts: Sequence[dict[str, Any]], reads: dict[int, dict[tuple[str, int], dict[str, Any]]],
    names: tuple[str, str], min_tv: float = 0.0, band: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The starts kept (those where either seat's most played move differs between the two
    conditions, the mixtures at least ``min_tv`` apart, and no read's value (side 0's units) is
    within ``band`` of a win or a loss) and the counts. The rule is symmetric in the two names
    (`compare`; the band asks it of all four reads).

    The band: a decided position has every move at the same value, so its most played move is
    whichever the solver reached first, and two conditions "differ" there for nothing; a game
    from it is adjudicated at once. Left in, they would dilute the pairs that matter."""
    first, second = names
    kept: list[dict[str, Any]] = []
    stats = {"probed": 0, "differ": 0, "seat0": 0, "seat1": 0, "both": 0, "ties": 0,
             "tvMean": 0.0, "decided": 0}
    tvs = []
    for pair in sorted(reads):
        got = reads[pair]
        stats["probed"] += 1
        seats = [compare(got[first, s], got[second, s]) for s in (0, 1)]
        stats["ties"] += sum(c["tie"] for c in seats)
        values = [got[name, seat]["value0"] for name in names for seat in (0, 1)]
        if band > 0 and any(v is None or v < band or v > 1.0 - band for v in values):
            stats["decided"] += any(c["differ"] for c in seats)
            continue
        wins = [c["differ"] and c["tv"] >= min_tv for c in seats]
        if not any(wins):
            continue
        stats["differ"] += 1
        stats["both"] += all(wins)
        stats["seat0"] += wins[0] and not wins[1]
        stats["seat1"] += wins[1] and not wins[0]
        tvs.extend(c["tv"] for c, w in zip(seats, wins, strict=True) if w)
        kept.append({**starts[pair], "source": pair, "probe": {
            "seats": seats,
            "reads": {f"{name}@{seat}": {k: v for k, v in got[name, seat].items()
                                         if k in ("top", "topP", "value0", "stage")}
                      for name in names for seat in (0, 1)}}})
    stats["tvMean"] = round(sum(tvs) / len(tvs), 4) if tvs else 0.0
    return kept, stats


def select(args: argparse.Namespace) -> int:
    starts = read_json_lines(Path(args.starts))
    lines = [line for p in sorted(Path(args.probe).glob("games-*worker*.jsonl"))
             for line in read_json_lines(p)]
    settings = json.loads((Path(args.probe) / "settings.json").read_bytes())
    names = (settings["tested"]["name"], settings["other"]["name"])
    reads = reads_of(lines, *names)
    kept, stats = select_positions(starts, reads, names, args.min_tv, args.band)
    n = _write_lines(Path(args.out), kept)
    rate = stats["differ"] / max(stats["probed"], 1)
    print(f"conditions {names[0]} / {names[1]}: {stats['probed']} candidates probed, "
          f"{stats['differ']} differ ({rate:.1%}; seat 0 only {stats['seat0']}, seat 1 only "
          f"{stats['seat1']}, both {stats['both']}; ties not counted {stats['ties']}), mean "
          f"distance of the mixtures {stats['tvMean']}; left out as decided (value within "
          f"{args.band} of a win or a loss) while differing: {stats['decided']} -> {args.out} "
          f"({n} lines)", file=sys.stderr)
    (Path(args.out).with_suffix(".summary.json")).write_bytes(
        (json.dumps({"conditions": names, **stats}, indent=1) + "\n").encode("utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--set", required=True, help="a depth_outcome position set (positions.jsonl)")
    c.add_argument("--pool", default="regmc-matchupweb")
    c.add_argument("--out", required=True)
    c.add_argument("--min-turn", type=int, default=1)
    c.add_argument("--limit", type=int, default=0, help="at most this many (0: all)")
    c.set_defaults(fn=candidates)
    s = sub.add_parser("select")
    s.add_argument("--starts", required=True, help="the candidates the probe read")
    s.add_argument("--probe", required=True, help="the time_match --probe run's --out")
    s.add_argument("--out", required=True)
    s.add_argument("--min-tv", type=float, default=0.0,
                   help="also ask the mixtures to be this far apart (total variation)")
    s.add_argument("--band", type=float, default=0.1,
                   help="leave out a position where any of the four reads' value is within this "
                   "of a win or a loss (decided: its moves are all equal). 0: keep them")
    s.set_defaults(fn=select)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
