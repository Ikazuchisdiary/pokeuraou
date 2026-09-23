"""Self-play with both seats drawn from one pool, and the selection solved on the spot (IKA-81).

`pokeuraou.selfplay.generate` plays a fixed roster at seat 0 against a field. This plays
two teams of one `pool.Pool` against each other, neither of them "ours", and it is what M-C
generation runs (IKA-77).

**Selection, solved where the game starts** (the user's decision of 9/24 on IKA-77: 選出は
その場で考える). Each game solves the 90 x 90 selection game of its two sheets with the
leaf it plays with (`selection.solve_selection`), and both seats draw their ordered four
from that one solve -- seat 0 from the row strategy, seat 1 from the column strategy,
softened by the same epsilon and temperature a selection book uses. No book is built in
advance (IKA-84 is not needed); the same pair under the same leaf is the same answer, so a
worker solves a pair once and keeps it (`SolvedSelections`). The record says
``selectionSource: "solved"``.

What is solved, exactly:

- **one spread class.** The pool carries every team's SP and the user decided on 9/24 not
  to hide spreads in M-C yet (only the bench), so the column player has no private type:
  the Bayesian game is a plain matrix game;
- **the pair in index order.** The lower pool index is always the row player, whichever
  seat it sits in, and a game with the seats the other way reads the same solve transposed
  (the row and column strategies swap, the value becomes one minus it). Exact for a matrix
  game, and it is what makes the memo invisible: without it the two seatings would be two
  LPs and could land on different vertices of a degenerate optimum, so a memo keyed on the
  unordered pair would change games;
- **a mirror uses the row strategy for both seats.** Same six, same spreads: the matrix is
  antisymmetric, so the row strategy is also optimal for the column player, and giving both
  seats the same one makes the two seats exchangeable -- the true mirror is then exactly
  50% by symmetry, not only in value. The LP's own column strategy can be a different
  vertex and would break that.

Both seats' bench beliefs come from the same solve (IKA-128: `bench_prior` for both sides
from the pair-keyed selection): seat ``s``'s belief about its opponent is the opponent's own
drawn mixture, which is what `BenchPrior.of` returns for the other index.

**Randomness.** Per game, from ``[seed, index]`` when a queue deals indices and from one
stream otherwise, as in `generate`. The pair, then the seats, then the selection draw (ours
first, `BookEntry.draw`), then the game. The solve draws nothing, so solving or reading it
from the memo leaves the stream where it was.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .payoff import HP_SHARE, Objective
from .pool import Pool, draw_pair
from .regulation import Regulation
from .selection_book import (
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    BenchPrior,
    BookEntry,
)
from .selfplay import MAX_TURNS, SEARCH_LIMIT, LeafEvaluator, play_game
from .teams import Roster, pick_four_indices

#: `selectionSource` of a game whose selection was solved at the start of the game.
SOLVED = "solved"

SELECTIONS = ("solved", "uniform")


@dataclass
class SolvedSelections:
    """The selection game of each pair, solved on first sight and kept (one worker, one leaf).

    ``memo=False`` solves every time, which is the control that the memo changes no game.

    ``store`` shares the solves between workers: a directory where each pair's solve is
    written once (`<lo>-<hi>.json`, atomically) and read by every worker that meets the pair
    after that. Measured on the M-C pool a solve is about 2.9 s of worker CPU, 2.1 s of it
    building the 8,100 positions, and a 24-worker run of 24,000 games meets about 800
    distinct pairs per worker, so a memo per worker alone would solve ~19,000 times
    against 2,145 shared (TODO.md, IKA-81). ``tag`` names what the answers depend on --
    the pool's sha256 and the leaf -- and a stored file with another tag stops the run
    rather than being re-solved or trusted.
    """

    reg: Regulation
    teams: Sequence[Roster]
    evaluate: Callable[[list[Any]], np.ndarray]
    memo: bool = True
    model: str = ""
    solves: int = 0
    reused: int = 0
    seconds: float = 0.0
    positions: int = 0
    #: Seconds of each solve, in the order they happened.
    per_solve: list[float] = field(default_factory=list)
    worst_antisymmetry: float = 0.0
    store: Path | None = None
    tag: str = ""
    #: Pairs read from `store` that another worker (or an earlier run) had solved.
    loaded: int = 0
    _kept: dict[tuple[int, int], BookEntry] = field(default_factory=dict)

    def _solve(self, lo: int, hi: int) -> BookEntry:
        # Imported here: `selection` imports `selfplay`, which this module also imports,
        # and nothing that only draws uniformly should need the LP.
        from .selection import SpreadClass, book_entry, solve_selection

        row, col = self.teams[lo], self.teams[hi]
        started = time.perf_counter()
        analysis = solve_selection(
            self.reg,
            row.sets,
            [SpreadClass(weight=1.0, sets=tuple(col.sets), label="sheet")],
            self.evaluate,
        )
        entry = book_entry(
            analysis, key=f"{row.id}|{col.id}", player=col.name, model=self.model
        )
        if lo == hi:
            entry.their_strategies = (np.array(entry.our_strategy, dtype=np.float64),)
            entry.their_ev_loss = (np.array(entry.our_ev_loss, dtype=np.float64),)
        took = time.perf_counter() - started
        self.solves += 1
        self.seconds += took
        self.per_solve.append(took)
        self.positions += analysis.positions_evaluated
        self.worst_antisymmetry = max(self.worst_antisymmetry, analysis.antisymmetry_error)
        return entry

    def canonical(self, a: int, b: int) -> BookEntry:
        """The solve of the pair with the lower index as the row player."""
        lo, hi = min(a, b), max(a, b)
        if self.memo and (lo, hi) in self._kept:
            self.reused += 1
            return self._kept[(lo, hi)]
        entry = self._read(lo, hi) if self.memo else None
        if entry is None:
            entry = self._solve(lo, hi)
            if self.memo:
                self._write(lo, hi, entry)
        if self.memo:
            self._kept[(lo, hi)] = entry
        return entry

    def _path(self, lo: int, hi: int) -> Path | None:
        return None if self.store is None else self.store / f"{lo:03d}-{hi:03d}.json"

    def _read(self, lo: int, hi: int) -> BookEntry | None:
        path = self._path(lo, hi)
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_bytes().decode("utf-8"))
        teams = [self.teams[lo].id, self.teams[hi].id]
        if data.get("tag") != self.tag or data.get("teams") != teams:
            raise ValueError(
                f"{path} was solved for tag {data.get('tag')!r} and teams "
                f"{data.get('teams')}, not {self.tag!r} and {teams}: another pool or "
                "another leaf. Give this run its own store."
            )
        self.loaded += 1
        return BookEntry.from_json(data["entry"])

    def _write(self, lo: int, hi: int, entry: BookEntry) -> None:
        path = self._path(lo, hi)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tag": self.tag,
            "pair": [lo, hi],
            "teams": [self.teams[lo].id, self.teams[hi].id],
            "seconds": self.per_solve[-1],
            "entry": entry.to_json(),
        }
        # Another worker may be writing the same pair: both write the same answer, and the
        # rename makes whichever lands last whole rather than interleaved.
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(json.dumps(payload).encode("utf-8"))
        os.replace(temporary, path)

    def entry(self, seat0: int, seat1: int) -> BookEntry:
        """The pair's selection game with ``seat0``'s team as the row player."""
        entry = self.canonical(seat0, seat1)
        if seat0 <= seat1:
            return entry
        return transposed(entry, self.teams[seat1].sets, key=f"{self.teams[seat0].id}|"
                          f"{self.teams[seat1].id}", player=self.teams[seat1].name)


def transposed(
    entry: BookEntry, row_sets: Sequence[Any], *, key: str, player: str
) -> BookEntry:
    """The same one-class matrix game read from the column player's chair.

    ``row_sets`` are the old row player's six, now the column (the "class") side.
    """
    if len(entry.their_strategies) != 1:
        raise ValueError("only a one-class selection game can be read transposed")
    return BookEntry(
        key=key,
        player=player,
        place=0,
        selections=entry.selections,
        our_strategy=np.array(entry.their_strategies[0], dtype=np.float64),
        our_ev_loss=np.array(entry.their_ev_loss[0], dtype=np.float64),
        class_weights=np.array([1.0]),
        class_sets=(tuple(row_sets),),
        their_strategies=(np.array(entry.our_strategy, dtype=np.float64),),
        their_ev_loss=(np.array(entry.our_ev_loss, dtype=np.float64),),
        value=1.0 - float(entry.value),
        duality_gap=entry.duality_gap,
        antisymmetry_error=entry.antisymmetry_error,
        seconds=entry.seconds,
        model=entry.model,
        notes=entry.notes,
    )


def generate_pool(
    reg: Regulation,
    pool: Pool,
    *,
    games: int,
    hide_bench: bool,
    seed: int = 0,
    out: Path,
    selection: str = SOLVED,
    evaluate: LeafEvaluator | None = None,
    solver: SolvedSelections | None = None,
    memo: bool = True,
    store: Path | None = None,
    objective: Objective = HP_SHARE,
    search_limit: int | tuple[int, int] = SEARCH_LIMIT,
    max_turns: int = MAX_TURNS,
    leaf: str | None = None,
    explore_epsilon: float = DEFAULT_EPSILON,
    explore_temperature: float = DEFAULT_TEMPERATURE,
    rank_by_leaf: bool = False,
    policy: Any = None,
    indices: Iterable[int] | None = None,
    on_finish: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Plays pool-against-pool games and appends one JSON line per finished game.

    ``selection`` is "solved" (what M-C ships: the selection game solved with the leaf at
    the start of each game, both seats drawn from it, and both seats' bench beliefs taken
    from it) or "uniform" (a reference: four of six uniformly on both sides and uniform bench
    beliefs). "solved" needs a leaf that answers a turn-1 position -- ``evaluate`` or
    ``solver`` -- because the hp-share proxy scores every full-HP position alike and the LP
    would return an arbitrary vertex.

    ``hide_bench`` is keyword-only and has no default here either (IKA-123); the CLI gives
    the pool path the hidden bench when neither flag is passed (IKA-128).

    ``store`` is the directory the solves are shared through (`SolvedSelections`); None
    keeps them in this process only.

    Each record carries ``pool``: the pool id and sha256, the two teams by id and name in
    seat order, the pair index and whether it is a mirror. ``selectionSource`` is "solved"
    or "uniform".
    """
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")
    if selection == SOLVED and solver is None:
        if evaluate is None:
            raise ValueError(
                "a solved selection needs a leaf: the hp-share proxy scores every turn-1 "
                "position at 0.5 and the LP would return an arbitrary vertex. Pass a value "
                "function, or selection='uniform' to mean the uniform reference."
            )
        solver = SolvedSelections(
            reg, pool.teams, evaluate, memo=memo, model=leaf or "",
            store=store, tag=f"{pool.sha256}|{leaf or ''}",
        )
    if pool.reg.meta.format_id != reg.meta.format_id:
        raise ValueError(
            f"the pool is {pool.reg.meta.format_id}, the regulation {reg.meta.format_id}"
        )
    pairs = pool.pairs
    size = reg.meta.picked_team_size

    def scheduled() -> Iterator[tuple[int | None, np.random.Generator]]:
        if indices is None:
            rng = np.random.default_rng(seed)
            for _ in range(games):
                yield None, rng
        else:
            for index in indices:
                yield index, np.random.default_rng([seed, index])

    out.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "games": 0,
        "finished": 0,
        "discarded_unfinished": 0,
        "wins": 0,
        "decisions": 0,
        "turns": 0,
        "mirror_games": 0,
        "mirror_wins": 0,
        "mirror_draws": 0,
        "selection_seconds": 0.0,
    }
    with out.open("a", encoding="utf-8") as handle:
        for index, rng in scheduled():
            k, a, b = draw_pair(rng, pairs)
            team0, team1 = pool.teams[a], pool.teams[b]
            six0, six1 = list(team0.sets), list(team1.sets)
            mirror = a == b
            drawn = None
            priors: tuple[BenchPrior | None, BenchPrior | None] | None = None
            started = time.perf_counter()
            if selection == SOLVED:
                assert solver is not None
                entry = solver.entry(a, b)
                drawn = entry.draw(
                    rng, epsilon=explore_epsilon, temperature=explore_temperature
                )
                pick0, pick1 = drawn.our_pick, drawn.foe_pick
                if hide_bench:
                    priors = (
                        BenchPrior.of(
                            entry, 0, [s.species for s in six0],
                            epsilon=explore_epsilon, temperature=explore_temperature,
                        ),
                        BenchPrior.of(
                            entry, 1, [s.species for s in six1],
                            epsilon=explore_epsilon, temperature=explore_temperature,
                        ),
                    )
            else:
                pick0 = pick_four_indices(rng, len(six0), size=size)
                pick1 = pick_four_indices(rng, len(six1), size=size)
            stats["selection_seconds"] += time.perf_counter() - started

            record = play_game(
                reg, rng, [six0[i] for i in pick0], [six1[i] for i in pick1],
                "mirror" if mirror else "pool",
                objective=objective, search_limit=search_limit, max_turns=max_turns,
                evaluate=evaluate,
                rank_by_leaf=rank_by_leaf,
                policy=policy,
                sheets=(six0, six1) if hide_bench else None,
                open_information=not hide_bench,
                bench_prior=priors,
                selection=(
                    [s.species for s in six0],
                    [s.species for s in six1],
                    tuple(pick0),
                    tuple(pick1),
                ),
            )
            record.selection_source = selection
            if drawn is not None:
                record.own_selection_policy = [float(x) for x in drawn.our_equilibrium]
                record.foe_selection_policy = [float(x) for x in drawn.foe_equilibrium]
                record.own_selection_mixture = [float(x) for x in drawn.our_mixture]
                record.foe_selection_mixture = [float(x) for x in drawn.foe_mixture]
                record.selection_value = drawn.value

            stats["games"] += 1
            if mirror:
                stats["mirror_games"] += 1
            if record.outcome is None:
                stats["discarded_unfinished"] += 1
            else:
                stats["finished"] += 1
                stats["wins"] += int(record.outcome > 0.5)
                stats["decisions"] += len(record.decisions)
                stats["turns"] += record.turns
                if mirror:
                    stats["mirror_wins"] += int(record.outcome > 0.5)
                    stats["mirror_draws"] += int(record.outcome == 0.5)
                payload = record.to_json(
                    objective=leaf or objective.name, search_limit=search_limit
                )
                payload["pool"] = {
                    "id": pool.id,
                    "sha256": pool.sha256,
                    "teams": [team0.id, team1.id],
                    "names": [team0.name, team1.name],
                    "pair": k,
                    "mirror": mirror,
                    "benchPrior": (
                        None if not hide_bench
                        else ["solved", "solved"] if priors is not None
                        else ["uniform", "uniform"]
                    ),
                }
                if index is not None:
                    payload["gameIndex"] = index
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
            if index is not None and on_finish is not None:
                on_finish(index)
    stats["path"] = str(out)
    if solver is not None:
        stats["solves"] = solver.solves
        stats["solves_reused"] = solver.reused
        stats["solves_loaded"] = solver.loaded
        stats["solve_seconds"] = solver.seconds
        stats["solve_positions"] = solver.positions
        stats["solve_worst_antisymmetry"] = solver.worst_antisymmetry
    return stats


__all__ = [
    "SELECTIONS",
    "SOLVED",
    "SolvedSelections",
    "generate_pool",
    "transposed",
]
