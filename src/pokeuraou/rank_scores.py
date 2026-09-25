"""The leaf ranking's own numbers, written beside a generation run (IKA-278).

`search.leaf_ranking` resolves every candidate of a side against its first damage replies,
scores each resulting position with the leaf, and hands `narrow` the mean -- and the matrix
it averaged is thrown away. IKA-274 wants a head that gives those scores from the position
directly, and the cheapest teacher data for it is the numbers generation already computes.
So a generation worker can write them down, off the games file:

    games-worker3.jsonl        the games, byte for byte what they were
    rank-worker3.jsonl.gz      one JSON line per game, one gzip member per line

A line::

    {"gameIndex": 18, "rankScores": 1, "orphans": 0, "rankings": [
      {"decision": 0, "turn": 1, "agent": 0, "side": 0,
       "completion": {"index": 2, "species": ["glimmora", "hatterene"], "slots": [2, 3],
                      "weight": 0.61, "of": 6},
       "candidates": ["move 1 1, move 2 2", ...],
       "score": [0.4127, ...],
       "fills": [{"weight": 1.0, "replies": ["move 1 1, move 1 2", ...],
                  "values": [[0.40, 0.42], ...]}]},
      ...]}

* ``decision`` indexes the game's ``decisions`` (the same game's line in games-*.jsonl,
  matched on ``gameIndex``); ``turn`` repeats its turn as a check. Ranking runs at move
  decisions only -- a replacement or a self-switch node narrows nothing (`_menus` is
  called from the move node alone) -- so every row is a ``"move"`` decision.
* ``agent`` is whose menu construction it was: 0 for side 0's, which serves both sides
  when the two agents build the same menu (all of generation), 1 for side 1's own when
  they do not. ``side`` is the side whose candidates were ranked. The menu a side played
  is its own agent's construction when there is one, otherwise agent 0's (`played`).
* ``completion`` is the world of the opponent's unseen slots the ranking read
  (`_menus.views`: the heaviest), with its belief weight and how many there were; None in
  the open game. The completion's position is not written: it is the decision's
  ``position`` with ``slots`` of side ``1 - side`` rebuilt from the sheet (the pool team
  the game names), and `completion_position` rebuilds it.
* ``candidates`` is the pool `narrow` scored, in its order (the legal actions left after
  `drop_dead_actions`), and ``score`` what `narrow` ordered it by -- `believed_ranking`'s
  fold of the fills, higher is better for ``side``.
* ``fills`` are the leaf ranking's matrices, one per completion folded (one today): the
  ``replies`` of the other side and ``values[i][j]``, candidate i against reply j, in side
  0's win probability (the leaf's own units, not ``side``'s). ``score`` is the weighted mean
  over the fills of each row's mean, negated for side 1.
* ``orphans`` counts rankings dropped because their decision was never written (the game
  stopped between the menus and the record).

The line is written before the game's own, so a worker killed between the two leaves a
ranking whose game is replayed on the restart -- the same game, since a game is fixed by
its index -- rather than a game with no ranking. Readers keep the first line of an index.
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np

#: The line's format, so a reader can tell it from a later one.
FORMAT = 1


class _Collector:
    """One game's rankings, gathered while `play_game` runs."""

    __slots__ = ("node", "pending", "rows")

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.node: tuple[int, int, int] | None = None


_SINK: ContextVar[_Collector | None] = ContextVar("pokeuraou_rank_scores", default=None)


@contextmanager
def collecting() -> Iterator[_Collector]:
    """Gathers every ranking made inside the block; nothing is gathered outside one."""
    sink = _Collector()
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)


def at_node(decision: int, turn: int, agent: int) -> None:
    """The next rankings build agent `agent`'s menus for decision `decision`."""
    sink = _SINK.get()
    if sink is not None:
        sink.node = (decision, turn, agent)


def saw_fill(side: int, pool: Sequence[Any], replies: Sequence[Any], payoff: np.ndarray) -> None:
    """`leaf_ranking`'s matrix: `payoff` is side 0's win probability, ours x theirs."""
    sink = _SINK.get()
    if sink is None:
        return
    values = payoff if side == 0 else payoff.T
    sink.pending.append(
        {
            "replies": [a.to_choice() for a in replies],
            "values": np.asarray(values, dtype=np.float64).tolist(),
        }
    )


def watch(
    rank: Callable[..., np.ndarray],
    side: int,
    used: Mapping[int, tuple[int, tuple[str, ...]]] | None,
    spreads: Mapping[int, Sequence[Any]] | None,
    weights: Sequence[float] = (1.0,),
) -> Callable[..., np.ndarray]:
    """`rank` itself when nothing is gathering; otherwise `rank` that also writes a row.

    `used` and `spreads` are `_menus`' own: which completion side `side` ranked from, and
    the completions it chose among. `weights` are the parts' weights as `believed_ranking`
    folds them, in the order the fills arrive.
    """
    if _SINK.get() is None:
        return rank

    def watched(pool: list[Any], scored: object = None) -> np.ndarray:
        sink = _SINK.get()
        start = len(sink.pending) if sink is not None else 0
        values = rank(pool, scored)
        if sink is None or sink.node is None:
            return values
        fills = sink.pending[start:]
        del sink.pending[start:]
        decision, turn, agent = sink.node
        completion = None
        if spreads is not None and used is not None and side in used:
            index, species = used[side]
            items = spreads[1 - side]
            completion = {
                "index": int(index),
                "species": list(species),
                "slots": [int(s) for s in items[index].slots],
                "weight": float(items[index].weight),
                "of": len(items),
            }
        sink.rows.append(
            {
                "decision": decision,
                "turn": turn,
                "agent": agent,
                "side": side,
                "completion": completion,
                "candidates": [a.to_choice() for a in pool],
                "score": np.asarray(values, dtype=np.float64).tolist(),
                "fills": [
                    {"weight": float(w), "replies": f["replies"], "values": f["values"]}
                    for f, w in zip(fills, _padded(weights, len(fills)), strict=True)
                ],
            }
        )
        return values

    return watched


def _padded(weights: Sequence[float], n: int) -> list[float]:
    got = list(weights)[:n]
    return got + [1.0] * (n - len(got))


def game_line(game_index: int | None, sink: _Collector, decisions: int) -> bytes:
    """The game's line: its rankings whose decision was written, as UTF-8 with a newline."""
    kept = [row for row in sink.rows if row["decision"] < decisions]
    payload = {
        "gameIndex": game_index,
        "rankScores": FORMAT,
        "orphans": len(sink.rows) - len(kept),
        "rankings": kept,
    }
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def path_for(games: Path) -> Path:
    """``games-worker3.jsonl`` -> ``rank-worker3.jsonl.gz``, in the same directory."""
    stem = games.name.removesuffix(".jsonl").removeprefix("games-")
    return games.with_name(f"rank-{stem}.jsonl.gz")


def _members(data: bytes) -> Iterator[tuple[int, bytes]]:
    """(end offset, content) of each complete gzip member; stops at a torn one."""
    offset = 0
    while offset < len(data):
        inflate = zlib.decompressobj(wbits=31)
        try:
            content = inflate.decompress(data[offset:])
        except zlib.error:
            return
        if not inflate.eof:
            return
        offset = len(data) - len(inflate.unused_data)
        yield offset, content


def repair(path: Path) -> int:
    """Cuts a torn last member off `path`, keeping it beside as ``<name>.torn``.

    Returns the bytes cut. A worker killed mid-write leaves one, and anything appended
    after it would be unreachable by a reader that stops there.
    """
    if not path.exists():
        return 0
    data = path.read_bytes()
    end = max((stop for stop, _content in _members(data)), default=0)
    if end == len(data):
        return 0
    path.with_name(path.name + ".torn").write_bytes(data[end:])
    path.write_bytes(data[:end])
    return len(data) - end


#: gzip level. 1, not the usual 6: on 300 recorded M-C games a line took 1.5 ms to write at
#: 1 and 2.9 ms at 6, for 28.5 KB against 24.5 KB a game -- the floats, which are most of a
#: line, hardly compress at any level (IKA-278).
LEVEL = 1


class Writer:
    """Appends one gzip member per game to a rank file, repairing a torn tail first."""

    def __init__(self, path: Path, level: int = LEVEL) -> None:
        self.path = path
        self.level = level
        path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired = repair(path)
        self._handle = path.open("ab")

    def write(self, line: bytes) -> int:
        # mtime 0: the same games make the same bytes.
        member = gzip.compress(line, compresslevel=self.level, mtime=0)
        self._handle.write(member)
        self._handle.flush()
        return len(member)

    def close(self) -> None:
        self._handle.close()


def iter_file(path: Path) -> Iterator[dict[str, Any]]:
    """Every complete line of one rank file, a torn tail ignored."""
    for _end, content in _members(path.read_bytes()):
        for raw in content.splitlines():
            if raw.strip():
                yield json.loads(raw)


def iter_records(where: Path) -> Iterator[dict[str, Any]]:
    """Every game's rankings under `where` (a rank file or a directory of them).

    A directory is read as ``rank-*.jsonl.gz`` sorted by name, and an index seen twice --
    a game replayed after a restart -- is kept once, the first time.
    """
    paths = [where] if where.is_file() else sorted(where.glob("rank-*.jsonl.gz"))
    seen: set[Any] = set()
    for path in paths:
        for line in iter_file(path):
            key = line.get("gameIndex")
            if key is not None and key in seen:
                continue
            seen.add(key)
            yield line


def folded(row: Mapping[str, Any]) -> np.ndarray:
    """``score`` recomputed from ``fills``, as `leaf_ranking` and `believed_ranking` do."""
    fills = row["fills"]
    if not fills:
        return np.zeros(len(row["candidates"]))
    total = sum(f["weight"] for f in fills) or 1.0
    out = None
    for fill in fills:
        values = np.asarray(fill["values"], dtype=np.float64)
        mean = values.mean(axis=1) if row["side"] == 0 else -values.mean(axis=1)
        got = mean if len(fills) == 1 else mean * (fill["weight"] / total)
        out = got if out is None else out + got
    assert out is not None
    return out


def played(rankings: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
    """(decision, side) -> the row whose menu that side played (its own agent's, else 0's)."""
    out: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rankings:
        key = (row["decision"], row["side"])
        if key not in out or row["agent"] == row["side"]:
            out[key] = row
    return out


def check_game(game: Mapping[str, Any], line: Mapping[str, Any]) -> list[str]:
    """What does not match between a game's record and its rankings; empty when all does.

    For each ranking whose menu was played: the decision is a move at the same turn; the
    score is the fold of the fills to the bit; and the recorded menu is a subset of the
    candidates, in the order `narrow` keeps it -- descending score, ties by choice.
    """
    problems: list[str] = []
    if game.get("gameIndex") != line.get("gameIndex"):
        problems.append(f"game {game.get('gameIndex')} against line {line.get('gameIndex')}")
    decisions = game["decisions"]
    for (index, side), row in sorted(played(line["rankings"]).items()):
        where = f"game {line.get('gameIndex')} decision {index} side {side}"
        if index >= len(decisions):
            problems.append(f"{where}: no such decision")
            continue
        decision = decisions[index]
        if decision["kind"] != "move" or decision["turn"] != row["turn"]:
            problems.append(f"{where}: {decision['kind']} at turn {decision['turn']}, "
                            f"ranked at turn {row['turn']}")
        score = np.asarray(row["score"], dtype=np.float64)
        if len(score) != len(row["candidates"]) or any(
            len(f["values"]) != len(row["candidates"]) for f in row["fills"]
        ):
            problems.append(f"{where}: lengths differ")
            continue
        if not np.array_equal(folded(row), score):
            problems.append(f"{where}: score is not the fold of the fills")
        menu = decision["ownActions"] if side == 0 else decision["foeActions"]
        at = {choice: i for i, choice in enumerate(row["candidates"])}
        if any(choice not in at for choice in menu):
            problems.append(f"{where}: a menu action is not a candidate")
            continue
        ordered = sorted(menu, key=lambda c: (-score[at[c]], c))
        if ordered != list(menu):
            problems.append(f"{where}: the menu is not in the score's order")
        views = decision.get("rankViews")
        completion = row["completion"]
        if (
            completion is not None
            and views is not None
            and views[side] is not None
            and [completion["index"], completion["species"]] != views[side]
        ):
            problems.append(f"{where}: completion {completion['index']} but rankViews "
                            f"says {views[side]}")
    return problems


def completion_position(reg: Any, pool: Any, game: Mapping[str, Any], row: Mapping[str, Any]) -> Any:  # noqa: ANN401
    """The position the ranking read: the decision's, with the guessed slots rebuilt.

    `pool` is the `pool.Pool` the game names (its ``pool.sha256`` is checked); the sheet of
    side ``1 - side`` is that seat's team.
    """
    from .hidden import substitute
    from .position import Position
    from .selfplay import _make_pokemon

    decision = game["decisions"][row["decision"]]
    pos = Position.from_json(decision["position"])
    completion = row["completion"]
    if completion is None or not completion["slots"]:
        return pos
    named = game["pool"]
    if named["sha256"] != pool.sha256:
        raise ValueError(f"the game was played from pool {named['sha256']}, not {pool.sha256}")
    other = 1 - row["side"]
    team = next(t for t in pool.teams if t.id == named["teams"][other])
    by_species = {entry.species: entry for entry in team.sets}
    sets = [by_species[name] for name in completion["species"]]
    return substitute(reg, pos, other, completion["slots"], sets, _make_pokemon)


__all__ = [
    "FORMAT",
    "Writer",
    "at_node",
    "check_game",
    "collecting",
    "completion_position",
    "folded",
    "game_line",
    "iter_file",
    "iter_records",
    "path_for",
    "played",
    "repair",
    "saw_fill",
    "watch",
]
