"""`tools/refusal_replay.py` draws its sample streaming, and draws the sample it always drew.

IKA-154: `sample` held every candidate decision and shuffled the list, so a whole-pool run
(`--decisions 1000000` over data/ika73/w12) reached 13 GB. It now shuffles indices and
reads the games twice, one at a time. These tests pin both halves: the drawn set equals the
old implementation's for the same seed, and a consumer that drops each decision never has
more than about one game's text alive.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import tracemalloc
from pathlib import Path

import pytest

from tests._harness import load_tool

replay = load_tool("refusal_replay")

PAYLOAD = 20_000  # characters of filler in each decision's position


def old_sample(args: argparse.Namespace) -> list[dict]:
    """The sampler as it stood before IKA-154, verbatim but for the name."""
    files = [p for d in args.games_dir for p in sorted(Path(d).glob("*.jsonl"))]
    per_file = -(-args.decisions // len(files))
    taken: list[dict] = []
    for path in files:
        here = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if here >= per_file:
                    break
                if args.holding and not any(item in line for item in args.holding):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                from_game = 0
                for decision in record.get("decisions", ()):
                    if decision.get("kind") != "move":
                        continue
                    if int(decision.get("turn", 0)) < args.min_turn:
                        continue
                    taken.append(decision)
                    here += 1
                    from_game += 1
                    if from_game >= args.per_game or here >= per_file:
                        break
    random.Random(args.seed).shuffle(taken)
    return taken[: args.decisions]


def games_dir(root: Path, files: int = 3, games: int = 12, payload: int = PAYLOAD) -> Path:
    rng = random.Random(1)
    out = root / "games"
    out.mkdir()
    for f in range(files):
        lines = []
        for g in range(games):
            decisions = []
            for t in range(rng.randint(2, 9)):
                decisions.append(
                    {
                        "kind": rng.choice(["move", "move", "move", "switch"]),
                        "turn": t,
                        "id": f"{f}-{g}-{t}",
                        "position": {"filler": "x" * payload},
                    }
                )
            item = "choicescarf" if g % 3 == 0 else "leftovers"
            lines.append(json.dumps({"item": item, "decisions": decisions}))
            if g == 4:
                lines.append("{not json")
        (out / f"games-worker{f}.jsonl").write_bytes(("\n".join(lines) + "\n").encode())
    return out


def args_for(directory: Path, **over) -> argparse.Namespace:
    base = {
        "games_dir": [directory],
        "decisions": 20,
        "per_game": 3,
        "min_turn": 1,
        "seed": 7,
        "holding": frozenset(),
    }
    base.update(over)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"seed": 11},
        {"decisions": 5},
        {"decisions": 60, "per_game": 2},
        {"decisions": 10_000},  # more than there are: every candidate
        {"min_turn": 3, "decisions": 17},
        {"holding": frozenset({"choicescarf"}), "decisions": 9},
    ],
)
def test_draws_the_set_the_old_sampler_drew(tmp_path: Path, over: dict) -> None:
    directory = games_dir(tmp_path)
    args = args_for(directory, **over)
    old = [d["id"] for d in old_sample(args)]
    new = [d["id"] for d in replay.sample(args)]
    assert len(new) == len(set(new))
    assert sorted(new) == sorted(old)


def test_sample_is_a_generator(tmp_path: Path) -> None:
    assert inspect.isgenerator(replay.sample(args_for(games_dir(tmp_path))))


def test_memory_is_bounded_by_one_game(tmp_path: Path) -> None:
    """A whole-pool draw with each decision dropped holds about one game, not the pool."""
    directory = games_dir(tmp_path, files=2, games=30)
    args = args_for(directory, decisions=10_000, per_game=100, min_turn=0)
    one_game = 9 * PAYLOAD  # the largest a synthetic game gets
    pool = sum(p.stat().st_size for p in directory.glob("*.jsonl"))
    assert pool > 30 * one_game
    tracemalloc.start()
    try:
        seen = 0
        for decision in replay.sample(args):
            seen += 1
            del decision
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert seen > 100
    assert peak < 4 * one_game, f"peak {peak:,} bytes over a {pool:,}-byte pool"
