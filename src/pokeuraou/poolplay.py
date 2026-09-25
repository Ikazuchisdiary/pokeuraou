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
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import rank_scores, timing
from .deepen import DEFAULT_DEEPEN
from .hidden import DEFAULT_BENCH_DROP
from .payoff import HP_SHARE, Objective
from .pool import Pool, draw_pair
from .regulation import Regulation
from .search import DEFAULT_RANK_FILL
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


def _read_shared(path: Path, *, attempts: int = 50, pause: float = 0.02) -> bytes:
    """A store file's bytes, waiting out another worker's rename onto it.

    Windows refuses to open a file while another process is replacing it, the reading half
    of the race `SolvedSelections._write` settles for writing: in the first M-C board
    match (IKA-82, warm2x2 vs scratchx2) one worker of 24 died on it after 341 s. The
    rename is atomic, so a later attempt reads the whole file, old or new -- the same
    answer either way.
    """
    for _ in range(attempts - 1):
        try:
            return path.read_bytes()
        except PermissionError:
            time.sleep(pause)
    return path.read_bytes()


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

    @timing.timed("selection.solve")
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
        data = json.loads(_read_shared(path).decode("utf-8"))
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
        try:
            os.replace(temporary, path)
        except PermissionError:
            # Windows refuses to replace a file another process has open -- the other
            # worker that solved the same pair, reading or replacing it. A match deals
            # both seats of a game at once, so two workers meet the same new pair in the
            # same second every game (IKA-259). Its file is the same answer (same pool,
            # same leaf), so ours is dropped; anything else is still an error.
            if not path.exists():
                raise
            temporary.unlink(missing_ok=True)

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
    rank_fill: str = DEFAULT_RANK_FILL,
    bench_drop: str = DEFAULT_BENCH_DROP,
    deepen: str = DEFAULT_DEEPEN,
    indices: Iterable[int] | None = None,
    on_finish: Callable[[int], None] | None = None,
    rank_scores_out: Path | None = None,
) -> dict[str, Any]:
    """Plays pool-against-pool games and appends one JSON line per finished game.

    ``rank_scores_out`` (IKA-278), when given, is a second file that gets each written
    game's leaf rankings (`rank_scores`), written just before the game's own line. It
    changes no game and no byte of ``out``.

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
    writer = rank_scores.Writer(rank_scores_out) if rank_scores_out is not None else None
    if writer is not None:
        stats["rank_scores_path"] = str(rank_scores_out)
        stats["rank_scores_bytes"] = 0
        stats["rank_scores_repaired"] = writer.repaired
    with out.open("a", encoding="utf-8") as handle:
        for index, rng in scheduled():
            k, a, b = draw_pair(rng, pairs)
            team0, team1 = pool.teams[a], pool.teams[b]
            six0, six1 = list(team0.sets), list(team1.sets)
            mirror = a == b
            drawn = None
            priors: tuple[BenchPrior | None, BenchPrior | None] | None = None
            started = time.perf_counter()
            # IKA-258: the stretch before a game's first decision had no row of its own.
            with timing.stage("selection"):
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

            with rank_scores.collecting() if writer is not None else nullcontext() as sink:
                record = play_game(
                    reg, rng, [six0[i] for i in pick0], [six1[i] for i in pick1],
                    "mirror" if mirror else "pool",
                    objective=objective, search_limit=search_limit, max_turns=max_turns,
                    evaluate=evaluate,
                    rank_by_leaf=rank_by_leaf,
                    policy=policy,
                    rank_fill=rank_fill,
                    bench_drop=bench_drop,
                    deepen=deepen,
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
                with timing.stage("record"):  # IKA-258
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
                    if writer is not None:
                        # Before the game's own line: a kill between the two leaves a
                        # ranking whose game the restart replays, not a game without one.
                        stats["rank_scores_bytes"] += writer.write(
                            rank_scores.game_line(index, sink, len(record.decisions))
                        )
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
            if index is not None and on_finish is not None:
                on_finish(index)
    stats["path"] = str(out)
    if writer is not None:
        writer.close()
    if solver is not None:
        stats["solves"] = solver.solves
        stats["solves_reused"] = solver.reused
        stats["solves_loaded"] = solver.loaded
        stats["solve_seconds"] = solver.seconds
        stats["solve_positions"] = solver.positions
        stats["solve_worst_antisymmetry"] = solver.worst_antisymmetry
    return stats


@dataclass
class PoolArm:
    """One agent of a pool-against-pool match (IKA-259).

    An agent here is a leaf, a width, a narrowing order and its own selection: ``solver``
    is the arm's `SolvedSelections` over ITS leaf, so each arm draws its four and believes
    the opponent's bench from the selection game as its own value function defines it --
    generation's "solved on the spot" (IKA-81), per arm. ``solver=None`` is an arm with no
    selection model (hp-share, the M-C origin): four of six uniformly, uniform bench belief.

    ``name`` is what the records call the leaf (a model stem such as ``value-gen11L``, or
    ``hp-share``); a rating is keyed on it.
    """

    name: str
    evaluate: LeafEvaluator | None
    solver: SolvedSelections | None
    limit: int
    rank_by_leaf: bool
    #: How its leaf ranking fills its cells (`search.parse_rank_fill`, IKA-268).
    rank_fill: str = DEFAULT_RANK_FILL
    #: Which completions its belief drops (`hidden.parse_bench_drop`, IKA-283).
    bench_drop: str = DEFAULT_BENCH_DROP
    #: How it deepens its move decisions (`deepen.parse_deepen`, IKA-33); none is off.
    deepen: str = DEFAULT_DEEPEN

    @property
    def selection(self) -> str:
        return SOLVED if self.solver is not None else "uniform"


def selection_rng(seed: int, game_index: int, side: int) -> np.random.Generator:
    """The stream side ``side`` draws its four from, in game ``game_index``.

    Its own stream per SIDE, not the game's: an arm that draws from a solve takes one
    uniform and an arm that draws uniformly takes several, so drawn from the game's stream
    the two would leave it in different places and the pair's two seats would play
    different games from the same seed for no reason but the draw's arithmetic.
    """
    return np.random.default_rng([seed, game_index, 1, side])


def draw_side(
    arm: PoolArm,
    entry: BookEntry | None,
    side: int,
    six: Sequence[Any],
    rng: np.random.Generator,
    *,
    size: int,
    epsilon: float,
    temperature: float,
) -> tuple[int, ...]:
    """The ordered four ``arm`` brings when it sits at ``side``.

    ``entry`` is the arm's own solve read with the seat-0 team as the row player, so side
    0 draws from the row strategy and side 1 from the column strategy.
    """
    if entry is None:
        return tuple(pick_four_indices(rng, len(six), size=size))
    mixture = (
        entry.our_mixture(epsilon=epsilon, temperature=temperature)
        if side == 0
        else entry.their_mixture(0, epsilon=epsilon, temperature=temperature)
    )
    cumulative = np.cumsum(np.asarray(mixture, dtype=np.float64))
    cumulative /= cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, rng.random(), side="right")), len(cumulative) - 1
    )
    return tuple(entry.selections[index])


def pool_match_game(
    reg: Regulation,
    pool: Pool,
    arms: tuple[PoolArm, PoolArm],
    *,
    seed: int,
    game_index: int,
    which: int,
    hide_bench: bool,
    objective: Objective = HP_SHARE,
    max_turns: int = MAX_TURNS,
    epsilon: float = 0.0,
    temperature: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Plays game ``game_index`` of a pool match with the tested arm (``arms[0]``) at
    side ``which`` (IKA-259).

    The pair and its seats come from ``[seed, game_index]`` alone, so both seats of a game
    are the same two teams in the same seats with the ARMS swapped -- what makes the two a
    pair. Each side's four comes from that side's own stream (`selection_rng`) and from
    the solve of the arm sitting there; each side's bench belief is the one its own arm
    holds (never the opponent's actual mixture, which would leak its private strategy).

    ``epsilon``/``temperature`` default to 0/1, the pure equilibrium: a rating asks what
    the strategy is worth, and exploration belongs to generation (`generation_match` does
    the same). The belief is built with the same pair, as `BenchPrior.of` requires.

    Returns the record and what each side was: leaf name, selection, belief, pick.
    """
    if pool.reg.meta.format_id != reg.meta.format_id:
        raise ValueError(
            f"the pool is {pool.reg.meta.format_id}, the regulation {reg.meta.format_id}"
        )
    if which not in (0, 1):
        raise ValueError(f"which must be 0 or 1, got {which}")
    rng = np.random.default_rng([seed, game_index])
    k, a, b = draw_pair(rng, pool.pairs)
    team0, team1 = pool.teams[a], pool.teams[b]
    six = (list(team0.sets), list(team1.sets))
    species = ([s.species for s in six[0]], [s.species for s in six[1]])
    mirror = a == b
    side_arms = (arms[0], arms[1]) if which == 0 else (arms[1], arms[0])
    size = reg.meta.picked_team_size

    entries = tuple(
        arm.solver.entry(a, b) if arm.solver is not None else None for arm in side_arms
    )
    picks = tuple(
        draw_side(
            side_arms[side], entries[side], side, six[side],
            selection_rng(seed, game_index, side),
            size=size, epsilon=epsilon, temperature=temperature,
        )
        for side in (0, 1)
    )
    # bench_prior[s] prices side s's bench and is read by side 1 - s: so it is built from
    # the entry of the arm sitting at 1 - s, about side s.
    priors: tuple[BenchPrior | None, BenchPrior | None] | None = None
    if hide_bench:
        about0 = (
            BenchPrior.of(entries[1], 0, species[0], epsilon=epsilon, temperature=temperature)
            if entries[1] is not None
            else None
        )
        about1 = (
            BenchPrior.of(entries[0], 1, species[1], epsilon=epsilon, temperature=temperature)
            if entries[0] is not None
            else None
        )
        if about0 is not None or about1 is not None:
            priors = (about0, about1)

    record = play_game(
        reg, rng, [six[0][i] for i in picks[0]], [six[1][i] for i in picks[1]],
        "mirror" if mirror else "pool",
        objective=objective,
        search_limit=(side_arms[0].limit, side_arms[1].limit),
        max_turns=max_turns,
        evaluate=(side_arms[0].evaluate, side_arms[1].evaluate),
        rank_by_leaf=(side_arms[0].rank_by_leaf, side_arms[1].rank_by_leaf),
        # Two agents even when they hold one leaf: each is charged its whole search.
        one_agent=False,
        sheets=six if hide_bench else None,
        open_information=not hide_bench,
        bench_prior=priors,
        rank_view="heaviest",
        rank_fill=(side_arms[0].rank_fill, side_arms[1].rank_fill),
        bench_drop=(side_arms[0].bench_drop, side_arms[1].bench_drop),
        deepen=(side_arms[0].deepen, side_arms[1].deepen),
        selection=(species[0], species[1], picks[0], picks[1]),
    )
    sources = tuple(arm.selection for arm in side_arms)
    record.selection_source = sources[0] if sources[0] == sources[1] else "mixed"
    beliefs = tuple(
        "uniform" if not hide_bench or entries[s] is None else SOLVED for s in (0, 1)
    )
    sides = {
        "pair": k,
        "teams": (a, b),
        "mirror": mirror,
        "leaves": tuple(arm.name for arm in side_arms),
        "limits": tuple(arm.limit for arm in side_arms),
        "rankings": tuple("leaf" if arm.rank_by_leaf else "damage" for arm in side_arms),
        "rank_fills": tuple(arm.rank_fill for arm in side_arms),
        "bench_drops": tuple(arm.bench_drop for arm in side_arms),
        "deepens": tuple(arm.deepen for arm in side_arms),
        "selections": sources,
        "beliefs": beliefs,
        "picks": picks,
    }
    return record, sides


__all__ = [
    "SELECTIONS",
    "SOLVED",
    "PoolArm",
    "SolvedSelections",
    "draw_side",
    "generate_pool",
    "pool_match_game",
    "selection_rng",
    "transposed",
]
