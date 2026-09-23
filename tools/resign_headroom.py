"""How much of generation's search is spent after the game is decided -- IKA-88.

Generation is bound by wall clock, and a doubling of games is worth +2.50 points (IKA-66).
A game whose result is settled but which goes on being searched to the last turn is
spending time that could have been the start of another game. AlphaGo Zero resigned when
the value crossed a threshold, and kept 10% of games playing on regardless so that it
could measure how often a resignation would have thrown a won game away (Silver et al.
2017). Every game in a recorded pool played on, so the whole pool is that 10%: the rule
can be replayed over it without playing a move.

    uv run python tools/resign_headroom.py data/ika73/armD
    uv run python tools/resign_headroom.py data/ika73/w12 --worker-logs data/ika73/w12/logs
    uv run python tools/resign_headroom.py data/ika73/armD --games 300        # a trial
    uv run python tools/resign_headroom.py data/ika73/w12 --save w12.npz
    uv run python tools/resign_headroom.py --load w12.npz --min-index 37402   # re-read nothing

Answer that day (2026-09-23, `data/ika73/armD`, 37,401 games, the shipping generation):
no (v, k) saves 15% of `searchSeconds` with at most 2% of resignations false. At v 0.97,
k 2 a resignation saves 6.4% on the workload split -- the decision count says 15.3% --
with 1.15% [1.03, 1.29] false, and removes 62% of the decisions at turn 15 and later.
The largest saving under 2% false is 8.1% (v 0.95, k 2). The 6,598 w12 games armD does
not hold, and IKA-66's width-24 pool, give the same table to within half a point.

## The rule

At a move decision, side 1 resigns once side 0's value has been at least `v` at `k`
consecutive move decisions, and side 0 once it has been at most `1 - v`. Replacement and
self-switch decisions neither count towards a streak nor break one: they carry no
recorded seconds and are solved differently. The decision that completes the streak is
still searched -- its value is what triggers -- so what a resignation saves is every
move decision after it, and what it loses is every decision of any kind after it.

A false resignation is one where the side that would have resigned went on to win.

## What the record holds, and what this has to fit

- `searchValue` is **side 0's** number. Under a hidden bench it is side 0's
  `belief_solve` value -- what side 0 guarantees against an opponent who knows their own
  four -- in side 0's win-probability units. Side 1's own value, from its own information
  state, was computed in `play_game` and not recorded (`answers[1].value`), so a rule
  that asks each side about its own position cannot be replayed. Both resignations here
  are read off side 0's number, and the two directions are reported separately because
  they are not the same rule: side 0's value is conservative for side 0.
- `searchSeconds` is **per game**, one entry per side, move nodes only. For one agent
  playing itself the two entries are halves of the same work (`play_game`, `one_agent`),
  so a game's search cost is their sum. **No decision carries its own seconds.**

So the seconds after a resignation are not in the record and have to be split out of the
game's total. Each game's own total is divided among its move decisions in proportion to
a fitted cost -- a non-negative least-squares fit, across games, of the game's total on
what its decisions had to do (IKA-97's method for an untimed stage: regress the total on
the workload count). The total across games is exact; only the split inside a game is
modelled. Three splits, declared before the pool was read:

- `uniform` -- every move decision of a game costs the same: the decision-count ratio,
  weighted by what each game cost. If late decisions are cheaper, this is an upper bound.
- `by-turn` -- one cost per turn (15 and later pooled), identified only by how long the
  games ran. Knows nothing about the node, so it checks the next one's shape.
- `workload` -- what the node does, from the recorded position and menus, replayed the
  way `play_game` builds it. **The bar is judged on this one.**

      decision  1      what every node pays whatever its size
      legal     L0+L1  the legal pools the leaf ranking resolves against two replies each
                       (`_menus` -> `narrow(rank=leaf_ranking)`), recomputed from the
                       position as `narrow` builds them
      cells     r x c  the matrix, resolved and encoded once
      scored    cells x completions scored, both sides: 1 when a side hides nothing,
                3 or 6 when it hides one or two slots (`hidden.completions`), with the
                hidden slots replayed the way `play_game` carries `shown`
      twice     cells when neither side hides anything -- `_per_completion` solves that
                node once a side, the same Position both times (IKA-97, IKA-104)

- `workload+turns` -- the workload with a free cost per turn on top. Added after the
  first full read (armD, 9/23), where `by-turn` and `workload` disagreed about the late
  turns, to ask whether the workload misses something late. It fits best and moves the
  saving *down*: the free turn costs go to turn 1, not to the end of the game. `by-turn`
  is biased the other way, because it only sees how long a game ran and long games are
  dearer from the start (their benches stay hidden longer).

A resignation also saves what is not search -- resolving the turns that are no longer
played, the replacement nodes -- which `searchSeconds` does not include. `--worker-logs`
reads the workers' own `N games in Xs` lines, so the search share of the wall clock can
be printed beside the table when the pool read is the whole run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: The grid IKA-88 fixed before measuring.
THRESHOLDS = (0.95, 0.97, 0.99)
STREAKS = (1, 2, 3)

#: What IKA-88 would ship on: at least this share of the search saved, with at most this
#: share of resignations false. Written in the issue before anything was measured.
SAVING_BAR = 0.15
FALSE_BAR = 0.02

KINDS = {"move": 0, "replacement": 1, "selfswitch": 2}
KIND_OTHER = 3

#: The turn IKA-88 calls the endgame: 5.4% of games reached it in the pool that number
#: came from, and resignation removes positions from exactly that end.
ENDGAME_TURN = 15
TURN_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-9", 7, 9),
    ("10-12", 10, 12),
    ("13-14", 13, 14),
    ("15+", 15, 10**6),
)
#: `by-turn` pools every turn from this one on into one coefficient.
LAST_TURN_BIN = 15

_INDEX = re.compile(rb'"gameIndex":\s*(\d+)\s*\}\s*$')
_WORKER_LINE = re.compile(r"(\d+) games in ([\d.]+)s")

#: (position JSON) -> each side's legal pool size, as `narrow` builds the pool.
LegalCounter = Callable[[dict], tuple[int, int]]


# --------------------------------------------------------------------------- reading


def pool_files(paths: Sequence[Path]) -> list[Path]:
    """Every `.jsonl` a path names: the file itself, or a directory's files in order."""
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.glob("*.jsonl")))
        else:
            out.append(path)
    return out


def _in_range(index: int | None, low: int | None, high: int | None) -> bool:
    if index is None:
        return low is None and high is None
    return (low is None or index >= low) and (high is None or index < high)


def iter_games(
    files: Iterable[Path],
    *,
    min_index: int | None = None,
    max_index: int | None = None,
    limit: int | None = None,
) -> Iterator[dict]:
    """Games one line at a time. The index filter reads `gameIndex` off the line's tail
    before parsing it, so a range that skips most of a pool does not pay for the skip."""
    taken = 0
    ranged = min_index is not None or max_index is not None
    for path in files:
        with path.open("rb") as handle:
            for raw in handle:
                if limit is not None and taken >= limit:
                    return
                if not raw.strip():
                    continue
                if ranged:
                    found = _INDEX.search(raw[-64:])
                    if found is not None and not _in_range(int(found.group(1)), min_index, max_index):
                        continue
                game = json.loads(raw)
                if ranged and not _in_range(game.get("gameIndex"), min_index, max_index):
                    continue
                taken += 1
                yield game


def seen_in(side: dict) -> set[int]:
    """`hidden.seen_slots` for one side, read off the position's JSON.

    The same seven tests, so that the replay does not build a `Position` for every
    decision just to ask them. `tests/test_resign_headroom.py` compares the two.
    """
    out: set[int] = set()
    for mon in side["pokemon"]:
        if (
            mon.get("activeIndex") is not None
            or mon.get("fainted")
            or mon["hp"] != mon["maxhp"]
            or mon.get("status") is not None
            or any(mon.get("boosts", {}).values())
            or mon.get("volatiles")
            or mon.get("isMega")
        ):
            out.add(mon["slot"])
    return out


def completions_for(hidden: int) -> int:
    """How many benches `hidden.completions` fills `hidden` unseen slots with: the four
    brought are six minus the ones on show, so `hidden` slots come from `2 + hidden`
    candidates. One when nothing is hidden -- the position itself, `exact`."""
    return comb(2 + hidden, hidden) if hidden > 0 else 1


def engine_legal_counter() -> LegalCounter:
    """Each side's pool as `narrow` builds it: `side_actions`, then `drop_dead_actions`."""
    from pokeuraou.actions import side_actions
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.narrow import drop_dead_actions
    from pokeuraou.position import Position
    from pokeuraou.regulation import load_regulation

    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)

    def count(position: dict) -> tuple[int, int]:
        pos = Position.from_json(position)
        sizes = [len(drop_dead_actions(reg, pos, s, side_actions(reg, pos, s))) for s in (0, 1)]
        return sizes[0], sizes[1]

    return count


@dataclass
class Pool:
    """A pool as arrays: per game, per move decision, and per decision of any kind."""

    seconds: np.ndarray  # (G,) the game's searchSeconds, both sides summed
    outcome: np.ndarray  # (G,) 1 when side 0 won
    index: np.ndarray  # (G,) gameIndex, -1 when absent
    move_start: np.ndarray  # (G+1,) offsets into the move arrays
    all_start: np.ndarray  # (G+1,) offsets into the all-decision arrays
    value: np.ndarray  # (M,) searchValue
    own: np.ndarray  # (M,) len(ownActions)
    foe: np.ndarray  # (M,) len(foeActions)
    legal0: np.ndarray  # (M,) side 0's legal pool
    legal1: np.ndarray  # (M,) side 1's legal pool
    hidden0: np.ndarray  # (M,) side 0's slots still unseen
    hidden1: np.ndarray  # (M,) side 1's slots still unseen
    move_turn: np.ndarray  # (M,) the move decision's turn
    move_at: np.ndarray  # (M,) this move decision's place in its game's decision list
    turn: np.ndarray  # (D,) every decision's turn
    kind: np.ndarray  # (D,) KINDS code

    @property
    def games(self) -> int:
        return len(self.seconds)

    def game_of_move(self) -> np.ndarray:
        return np.repeat(np.arange(self.games), np.diff(self.move_start))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> Pool:
        with np.load(path) as data:
            return cls(**{k: data[k] for k in cls.__dataclass_fields__})

    def subset(self, keep: np.ndarray) -> Pool:
        """The games `keep` marks, with their decisions."""
        chosen = np.flatnonzero(keep)

        def spans(start: np.ndarray) -> np.ndarray:
            if not len(chosen):
                return np.zeros(0, dtype=np.int64)
            return np.concatenate([np.arange(start[g], start[g + 1]) for g in chosen])

        moves, rows = spans(self.move_start), spans(self.all_start)
        per_move = ("value", "own", "foe", "legal0", "legal1", "hidden0", "hidden1",
                    "move_turn", "move_at")
        return Pool(
            seconds=self.seconds[chosen],
            outcome=self.outcome[chosen],
            index=self.index[chosen],
            move_start=np.concatenate([[0], np.cumsum(np.diff(self.move_start)[chosen])]).astype(np.int64),
            all_start=np.concatenate([[0], np.cumsum(np.diff(self.all_start)[chosen])]).astype(np.int64),
            turn=self.turn[rows],
            kind=self.kind[rows],
            **{name: getattr(self, name)[moves] for name in per_move},
        )


def build_pool(
    games: Iterable[dict], legal: LegalCounter | None = None
) -> tuple[Pool, dict[str, int]]:
    """Reads what the replay needs out of each game and drops the rest as it goes."""
    counter = legal if legal is not None else engine_legal_counter()
    per_game: dict[str, list] = {k: [] for k in ("seconds", "outcome", "index")}
    per_move: dict[str, list] = {
        k: [] for k in ("value", "own", "foe", "legal0", "legal1", "hidden0", "hidden1",
                        "move_turn", "move_at")
    }
    turn: list[int] = []
    kind: list[int] = []
    move_start = [0]
    all_start = [0]
    flagged = {"unlabelled": 0, "no move decision": 0, "sides differ in seconds": 0}

    for game in games:
        result = game.get("outcome")
        if result not in (0.0, 1.0):
            flagged["unlabelled"] += 1
            continue
        decisions = game.get("decisions") or []
        if not any(d.get("kind") == "move" for d in decisions):
            flagged["no move decision"] += 1
            continue
        halves = game.get("searchSeconds") or [0.0, 0.0]
        if abs(float(halves[0]) - float(halves[1])) > 1e-9:
            # Not an error -- a match charges each agent its own solve -- but then the
            # two entries are not halves of one agent's work, and the sum is not what
            # one agent's generation paid. Counted so a pool of that kind says so.
            flagged["sides differ in seconds"] += 1
        shown: list[set[int]] = [set(), set()]
        for place, decision in enumerate(decisions):
            code = KINDS.get(decision.get("kind"), KIND_OTHER)
            turn.append(int(decision.get("turn", 0)))
            kind.append(code)
            position = decision["position"]
            if code in (0, 1):
                # `play_game` carries `shown` through move and replacement nodes, from
                # the position each node was solved at; a self-switch does not update it.
                for side in (0, 1):
                    shown[side] |= seen_in(position["sides"][side])
            if code != 0:
                continue
            hidden = [
                sum(1 for mon in position["sides"][side]["pokemon"] if mon["slot"] not in shown[side])
                for side in (0, 1)
            ]
            pools = counter(position)
            per_move["value"].append(float(decision["searchValue"]))
            per_move["own"].append(len(decision["ownActions"]))
            per_move["foe"].append(len(decision["foeActions"]))
            per_move["legal0"].append(pools[0])
            per_move["legal1"].append(pools[1])
            per_move["hidden0"].append(hidden[0])
            per_move["hidden1"].append(hidden[1])
            per_move["move_turn"].append(int(decision.get("turn", 0)))
            per_move["move_at"].append(place)
        per_game["seconds"].append(float(halves[0]) + float(halves[1]))
        per_game["outcome"].append(1 if result == 1.0 else 0)
        per_game["index"].append(int(game.get("gameIndex", -1)))
        move_start.append(len(per_move["value"]))
        all_start.append(len(turn))

    types = {
        "seconds": np.float64, "outcome": np.int8, "index": np.int64, "value": np.float64,
        "own": np.int16, "foe": np.int16, "legal0": np.int16, "legal1": np.int16,
        "hidden0": np.int8, "hidden1": np.int8, "move_turn": np.int16, "move_at": np.int32,
    }
    pool = Pool(
        move_start=np.asarray(move_start, dtype=np.int64),
        all_start=np.asarray(all_start, dtype=np.int64),
        turn=np.asarray(turn, dtype=np.int16),
        kind=np.asarray(kind, dtype=np.int8),
        **{k: np.asarray(v, dtype=types[k]) for k, v in {**per_game, **per_move}.items()},
    )
    return pool, flagged


# ------------------------------------------------------------------------ cost model

WORKLOAD = ("decision", "legal", "cells", "scored", "twice")
BY_TURN = tuple(f"turn {t}" for t in range(1, LAST_TURN_BIN)) + (f"turn {LAST_TURN_BIN}+",)
#: The first three were declared before the pool was read; `workload+turns` was added
#: after the first full read (see the module docstring). The bar is judged on `HEADLINE`.
MODELS: dict[str, tuple[str, ...]] = {
    "uniform": ("decision",),
    "by-turn": BY_TURN,
    "workload": WORKLOAD,
    "workload+turns": WORKLOAD + BY_TURN,
}
HEADLINE = "workload"


def features(pool: Pool, terms: Sequence[str]) -> np.ndarray:
    """(M, len(terms)): the workload of each move decision, one column per term."""
    if not len(pool.value):
        return np.zeros((0, len(terms)))
    cells = pool.own.astype(np.float64) * pool.foe.astype(np.float64)
    table = np.asarray([completions_for(h) for h in range(7)], dtype=np.float64)
    completions = table[pool.hidden0.astype(np.int64)] + table[pool.hidden1.astype(np.int64)]
    columns = {
        "decision": np.ones_like(cells),
        "legal": pool.legal0.astype(np.float64) + pool.legal1.astype(np.float64),
        "cells": cells,
        "scored": cells * completions,
        "twice": cells * ((pool.hidden0 == 0) & (pool.hidden1 == 0)),
    }
    turns = np.minimum(pool.move_turn.astype(np.int64), LAST_TURN_BIN)
    for t in range(1, LAST_TURN_BIN + 1):
        name = f"turn {t}" if t < LAST_TURN_BIN else f"turn {LAST_TURN_BIN}+"
        columns[name] = (turns == t).astype(np.float64)
    return np.column_stack([columns[t] for t in terms])


def per_game(pool: Pool, rows: np.ndarray) -> np.ndarray:
    """Sums per-move-decision rows into per-game rows."""
    out = np.zeros((pool.games, rows.shape[1]))
    np.add.at(out, pool.game_of_move(), rows)
    return out


@dataclass
class Fit:
    name: str
    terms: tuple[str, ...]
    coef: np.ndarray
    r2: float
    cv_r2: float


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    total = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - fitted) ** 2).sum()) / total if total > 0 else float("nan")


def fit_cost(pool: Pool, name: str, *, folds: int = 5, seed: int = 88) -> Fit:
    """Non-negative least squares of each game's seconds on its decisions' workload,
    with a held-out R^2 from `folds` splits by game."""
    from scipy.optimize import nnls

    terms = MODELS[name]
    design = per_game(pool, features(pool, terms))
    y = pool.seconds
    coef, _residual = nnls(design, y)
    order = np.random.default_rng(seed).permutation(pool.games)
    held = np.zeros_like(y)
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test)
        part, _ = nnls(design[train], y[train])
        held[test] = design[test] @ part
    return Fit(name, terms, coef, _r2(y, design @ coef), _r2(y, held))


def split_seconds(pool: Pool, fit: Fit) -> np.ndarray:
    """(M,) each move decision's share of its own game's seconds.

    The fitted cost only apportions: a game's decisions add up to exactly what the game
    recorded, so the across-game total never depends on the fit.
    """
    cost = features(pool, fit.terms) @ fit.coef
    game_of = pool.game_of_move()
    totals = np.zeros(pool.games)
    np.add.at(totals, game_of, cost)
    counts = np.diff(pool.move_start).astype(np.float64)
    # A game whose fitted cost is zero everywhere falls back to the uniform split.
    safe = totals > 0
    share = np.where(
        safe[game_of],
        cost / np.where(safe, totals, 1.0)[game_of],
        1.0 / counts[game_of],
    )
    return share * pool.seconds[game_of]


# ---------------------------------------------------------------------------- replay


@dataclass
class Trigger:
    """Where the rule fires in each game: the move ordinal, or -1, and who resigns."""

    at: np.ndarray  # (G,) 0-based move ordinal of the deciding decision, -1 when none
    resigner: np.ndarray  # (G,) 0 or 1, -1 when none


def find_triggers(pool: Pool, threshold: float, streak: int) -> Trigger:
    at = np.full(pool.games, -1, dtype=np.int64)
    resigner = np.full(pool.games, -1, dtype=np.int8)
    high = (pool.value >= threshold).tolist()
    low = (pool.value <= 1.0 - threshold).tolist()
    starts = pool.move_start.tolist()
    for g in range(pool.games):
        start, end = starts[g], starts[g + 1]
        up = down = 0
        for i in range(start, end):
            up = up + 1 if high[i] else 0
            down = down + 1 if low[i] else 0
            if up >= streak:
                at[g], resigner[g] = i - start, 1
                break
            if down >= streak:
                at[g], resigner[g] = i - start, 0
                break
    return Trigger(at, resigner)


def clopper_pearson(hits: int, n: int, level: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - level
    lo = 0.0 if hits == 0 else float(beta.ppf(alpha / 2, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(beta.ppf(1 - alpha / 2, hits + 1, n - hits))
    return lo, hi


def turn_shares(turns: np.ndarray) -> list[float]:
    total = max(len(turns), 1)
    return [float(((turns >= lo) & (turns <= hi)).sum()) / total for _n, lo, hi in TURN_BUCKETS]


def replay(pool: Pool, threshold: float, streak: int, split: dict[str, np.ndarray]) -> dict:
    """One row of the table: what the rule saves, what it loses, and how often it lies."""
    trig = find_triggers(pool, threshold, streak)
    fired = trig.at >= 0
    moves = np.diff(pool.move_start)

    saved = dict.fromkeys(split, 0.0)
    moves_cut = 0
    kept_row = np.ones(len(pool.turn), dtype=bool)
    endgame = pool.turn >= ENDGAME_TURN
    for g in np.flatnonzero(fired):
        a0, a1 = pool.all_start[g], pool.all_start[g + 1]
        m0 = pool.move_start[g]
        first_cut = m0 + trig.at[g] + 1
        for name, seconds in split.items():
            saved[name] += float(seconds[first_cut : pool.move_start[g + 1]].sum())
        moves_cut += int(moves[g] - trig.at[g] - 1)
        place = int(pool.move_at[m0 + trig.at[g]])
        kept_row[a0 + place + 1 : a1] = False

    reached = np.zeros(pool.games, dtype=bool)
    reached_kept = np.zeros(pool.games, dtype=bool)
    game_of_row = np.repeat(np.arange(pool.games), np.diff(pool.all_start))
    reached[game_of_row[endgame]] = True
    reached_kept[game_of_row[endgame & kept_row]] = True

    won_anyway = ((trig.resigner == 0) & (pool.outcome == 1)) | (
        (trig.resigner == 1) & (pool.outcome == 0)
    )
    by_side = {
        side: (int(won_anyway[trig.resigner == side].sum()), int((trig.resigner == side).sum()))
        for side in (0, 1)
    }
    lost = ~kept_row
    total_seconds = float(pool.seconds.sum())
    n_fired = int(fired.sum())
    false_total = int(won_anyway[fired].sum())
    return {
        "threshold": threshold,
        "streak": streak,
        "games": pool.games,
        "fired": n_fired,
        "saved": {name: s / total_seconds for name, s in saved.items()},
        "moves_cut": moves_cut / max(int(moves.sum()), 1),
        "rows_cut": int(lost.sum()),
        "rows_cut_share": float(lost.sum()) / max(len(pool.turn), 1),
        "false": false_total,
        "false_rate": false_total / max(n_fired, 1),
        "false_ci": clopper_pearson(false_total, n_fired),
        "false_by_side": by_side,
        "endgame_before": float(endgame.mean()) if len(pool.turn) else 0.0,
        "endgame_after": float(endgame[kept_row].mean()) if kept_row.any() else 0.0,
        "endgame_rows_kept": int((endgame & kept_row).sum()),
        "endgame_rows": int(endgame.sum()),
        "reached_before": float(reached.mean()) if pool.games else 0.0,
        "reached_after": float(reached_kept.mean()) if pool.games else 0.0,
        "buckets_before": turn_shares(pool.turn),
        "buckets_after": turn_shares(pool.turn[kept_row]),
        "buckets_lost": turn_shares(pool.turn[lost]),
    }


def calibration(pool: Pool) -> list[tuple[str, int, float]]:
    """Positive control for the direction: side 0's win rate by side 0's value.

    If `searchValue` were side 1's number, or 1 - p, the top row would read near zero and
    every false-resignation rate below would be read upside down.
    """
    edges = (0.0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 1.0 + 1e-12)
    won = pool.outcome[pool.game_of_move()] == 1
    out = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        mask = (pool.value >= lo) & (pool.value < hi)
        n = int(mask.sum())
        out.append((f"[{lo:.2f}, {min(hi, 1.0):.2f})", n, float(won[mask].mean()) if n else float("nan")))
    return out


def cost_by_turn(pool: Pool, split: dict[str, np.ndarray]) -> list[tuple[str, int, dict[str, float]]]:
    """Mean seconds a move decision is charged, by turn, under each split -- the shape of
    what each model believes, so a split that makes late decisions cheap says how cheap."""
    turns = np.minimum(pool.move_turn.astype(np.int64), LAST_TURN_BIN)
    out = []
    for t in range(1, LAST_TURN_BIN + 1):
        mask = turns == t
        label = f"{t}" if t < LAST_TURN_BIN else f"{LAST_TURN_BIN}+"
        out.append((label, int(mask.sum()), {n: float(s[mask].mean()) if mask.any() else 0.0
                                              for n, s in split.items()}))
    return out


def worker_clock(log_dir: Path) -> tuple[int, float] | None:
    """Games and seconds the workers themselves reported (`N games in Xs`), summed."""
    games = 0
    seconds = 0.0
    found = False
    for path in sorted(log_dir.glob("worker*.log")):
        text = path.read_bytes().decode("utf-8", errors="replace")
        for match in _WORKER_LINE.finditer(text):
            games += int(match.group(1))
            seconds += float(match.group(2))
            found = True
    return (games, seconds) if found else None


def meets_bar(row: dict, split: str = HEADLINE) -> bool:
    return row["saved"][split] >= SAVING_BAR and row["false_rate"] <= FALSE_BAR


# ---------------------------------------------------------------------------- output


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def render(
    pool: Pool,
    fits: list[Fit],
    split: dict[str, np.ndarray],
    rows: list[dict],
    clock: tuple[int, float] | None,
) -> str:
    lines: list[str] = []
    moves = int(pool.move_start[-1])
    lines.append(
        f"{pool.games} games, {moves} move decisions, {int(pool.all_start[-1])} decisions "
        f"of any kind, search {pool.seconds.sum():.0f} s ({pool.seconds.mean():.3f} s a game, "
        f"{1000 * pool.seconds.sum() / max(moves, 1):.1f} ms a move decision)"
    )
    if clock is not None:
        games, seconds = clock
        share = pool.seconds.sum() / seconds if seconds > 0 else float("nan")
        note = "" if games == pool.games else f"  ! the logs cover {games} games, this read {pool.games}"
        lines.append(
            f"worker logs: {games} games in {seconds:.0f} worker-seconds; searchSeconds is "
            f"{_pct(share)} of that{note}"
        )

    lines.append("")
    lines.append("direction (positive control): side 0's win rate by side 0's searchValue")
    for label, n, rate in calibration(pool):
        lines.append(f"  {label:<14} {n:>8}  {_pct(rate, 2) if n else '-':>8}")

    lines.append("")
    lines.append("cost split: each game's seconds on its move decisions' workload (NNLS, across games)")
    for fit in fits:
        coef = "  ".join(
            f"{t} {c * 1000:.3g}" for t, c in zip(fit.terms, fit.coef, strict=True)
        )
        lines.append(f"  {fit.name:<9} R2 {fit.r2:.3f}  5-fold R2 {fit.cv_r2:.3f}   ms: {coef}")
    lines.append("  mean ms charged to a move decision, by turn")
    names = [f.name for f in fits]
    lines.append(f"    {'turn':>5} {'n':>8} " + " ".join(f"{n:>9}" for n in names))
    for label, n, means in cost_by_turn(pool, split):
        lines.append(f"    {label:>5} {n:>8} " + " ".join(f"{1000 * means[m]:>9.1f}" for m in names))

    lines.append("")
    lines.append("replay: saved = searchSeconds after the deciding decision / all searchSeconds")
    width = max(len("saved " + n) for n in names)
    lines.append(
        f"  {'v':>4} {'k':>2} {'resigned':>8} "
        + " ".join(f"{'saved ' + n:>{width}}" for n in names)
        + f" {'moves cut':>9} {'rows cut':>8}   {'false':<14} {'95% CI':<17}"
        + f" {'side 0 resigns':<18} {'side 1 resigns':<18}"
    )
    for row in rows:
        f0, n0 = row["false_by_side"][0]
        f1, n1 = row["false_by_side"][1]
        lo, hi = row["false_ci"]
        lines.append(
            f"  {row['threshold']:>4.2f} {row['streak']:>2d} {_pct(row['fired'] / max(row['games'], 1)):>8} "
            + " ".join(f"{_pct(row['saved'][n]):>{width}}" for n in names)
            + f" {_pct(row['moves_cut']):>9} {_pct(row['rows_cut_share']):>8}"
            f"   {_pct(row['false_rate'], 2):>6} {row['false']:>3}/{row['fired']:<5}"
            f" [{_pct(lo, 2)}, {_pct(hi, 2)}]"
            f"  {_pct(f0 / n0, 2) if n0 else '-':>6} of {n0:<7}"
            f"  {_pct(f1 / n1, 2) if n1 else '-':>6} of {n1:<7}"
        )

    lines.append("")
    lines.append(
        f"teacher positions lost, and the endgame (decisions at turn >= {ENDGAME_TURN}; games with one)"
    )
    for row in rows:
        lines.append(
            f"  {row['threshold']:>4.2f} {row['streak']:>2d}  lost {row['rows_cut']:>7} "
            f"({_pct(row['rows_cut_share'])})   turn>={ENDGAME_TURN} decisions "
            f"{row['endgame_rows']} -> {row['endgame_rows_kept']} "
            f"({_pct(row['endgame_before'], 2)} -> {_pct(row['endgame_after'], 2)} of the pool)"
            f"   games {_pct(row['reached_before'], 2)} -> {_pct(row['reached_after'], 2)}"
        )
    lines.append("")
    labels = " ".join(f"{name:>7}" for name, _lo, _hi in TURN_BUCKETS)
    lines.append(f"  turn distribution of the decisions   {labels}")
    lines.append("  as generated              " + "".join(f"{_pct(x):>8}" for x in rows[0]["buckets_before"]))
    for row in rows:
        lines.append(
            f"  {row['threshold']:>4.2f} {row['streak']:>2d}  kept           "
            + "".join(f"{_pct(x):>8}" for x in row["buckets_after"])
        )
        lines.append("           lost           " + "".join(f"{_pct(x):>8}" for x in row["buckets_lost"]))

    lines.append("")
    lines.append(
        f"bar (IKA-88, fixed before measuring): saved >= {_pct(SAVING_BAR, 0)} on the "
        f"{HEADLINE} split, and false <= {_pct(FALSE_BAR, 0)}"
    )
    passing = [r for r in rows if meets_bar(r)]
    for r in passing:
        lines.append(f"  meets it: v {r['threshold']:.2f}, k {r['streak']}")
    if not passing:
        lines.append("  no (v, k) meets it")
    upper = [r for r in rows if meets_bar(r, "uniform")]
    if upper:
        lines.append(
            "  on the uniform split (every decision of a game costing the same) it would be met by "
            + ", ".join(f"v {r['threshold']:.2f} k {r['streak']}" for r in upper)
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="*", type=Path, help="games .jsonl files, or directories of them")
    ap.add_argument("--games", type=int, default=None, help="read at most this many games (a trial)")
    ap.add_argument("--min-index", type=int, default=None, help="keep gameIndex >= this")
    ap.add_argument("--max-index", type=int, default=None, help="keep gameIndex < this")
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    ap.add_argument("--streaks", type=int, nargs="+", default=list(STREAKS))
    ap.add_argument("--worker-logs", type=Path, default=None, help="the run's logs/ directory")
    ap.add_argument("--save", type=Path, default=None, help="write the extracted arrays (.npz)")
    ap.add_argument("--load", type=Path, default=None, help="read arrays --save wrote, instead of a pool")
    args = ap.parse_args(argv)

    if args.load is not None:
        pool = Pool.load(args.load)
        keep = np.ones(pool.games, dtype=bool)
        if args.min_index is not None:
            keep &= pool.index >= args.min_index
        if args.max_index is not None:
            keep &= pool.index < args.max_index
        if args.games is not None:
            keep &= np.cumsum(keep) <= args.games
        if not keep.all():
            pool = pool.subset(keep)
        flagged: dict[str, int] = {}
        print(f"loaded {args.load}")
    else:
        if not args.pool:
            ap.error("name a pool, or --load a saved one")
        files = pool_files(args.pool)
        pool, flagged = build_pool(
            iter_games(files, min_index=args.min_index, max_index=args.max_index, limit=args.games)
        )
        shown = ", ".join(str(f) for f in files[:3]) + (" ..." if len(files) > 3 else "")
        print(f"read {len(files)} file(s): {shown}")
    for reason, count in flagged.items():
        if count:
            print(f"  flagged, {reason}: {count}")
    if args.save is not None:
        pool.save(args.save)
        print(f"  arrays -> {args.save}")
    if not pool.games:
        print("no games")
        return

    fits = [fit_cost(pool, name) for name in MODELS]
    split = {fit.name: split_seconds(pool, fit) for fit in fits}
    rows = [
        replay(pool, threshold, streak, split)
        for threshold in args.thresholds
        for streak in args.streaks
    ]
    clock = worker_clock(args.worker_logs) if args.worker_logs is not None else None
    print(render(pool, fits, split, rows, clock))


if __name__ == "__main__":
    main()
