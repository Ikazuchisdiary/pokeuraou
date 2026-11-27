"""Both seats drawn from one pool, and the selection solved on the spot (IKA-81).

M-C generation and boards play two teams of one `pool.Pool` against each other, neither of
them "ours" (IKA-77). The games are played by the Rust workers (`rust/src/worker.rs`,
`rust/src/boardworker.rs`, IKA-458 stage 3.7), which solve and store the selections the same
way (`rust/src/selection.rs`); the Python game loops (`generate_pool`, `pool_match_game`) were
removed then. What stays here is the selection solve and its shared store, which the human
play, the analysis tools and the stores' readers use, and a board arm's settings (`PoolArm`).

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
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import timing
from .regulation import Regulation
from .search import resolve_rank_fill
from .selection_book import BookEntry
from .selfplay import LeafEvaluator
from .teams import Roster

#: `selectionSource` of a game whose selection was solved at the start of the game.
SOLVED = "solved"


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
    #: The pairs this process solved itself (IKA-458 stage 3.5: the port reads the files, and
    #: counts a pair read from one of them as loaded only if another process wrote it).
    solved_pairs: set[tuple[int, int]] = field(default_factory=set)
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
            self.solved_pairs.add((lo, hi))
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


@dataclass
class PoolArm:
    """One agent of a pool-against-pool match (IKA-259), as `tools/pool_match.py` hands it to
    `rustworker.board_config` (the board's games are the Rust worker's since IKA-458 stage 3.7).

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
    #: How its leaf ranking fills its cells (`search.parse_rank_fill`, IKA-268). Unnamed
    #: (None), resolved at construction as generation's tools resolve it
    #: (`search.resolve_rank_fill`, IKA-341): q-nocover for a leaf-ranked arm.
    rank_fill: str | None = None
    #: The port executable this arm's move-node search is read through, or None for the
    #: process's own (IKA-413). Only for measuring a port change against the port before it;
    #: the game itself is resolved by the worker's own executable.
    rust_binary: Path | None = None

    def __post_init__(self) -> None:
        self.rank_fill = resolve_rank_fill(self.rank_fill, self.rank_by_leaf)

    @property
    def selection(self) -> str:
        return SOLVED if self.solver is not None else "uniform"
