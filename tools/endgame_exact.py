"""How far the search's small endgame answers are from the game solved to the end (IKA-254).

IKA-254 asks whether a position whose matrix is small should be read to the end instead of
to a fixed depth. IKA-279 counted how often that comes up (a menu product of 16 or less
is 3.4% of M-C move decisions, 17.4% once the benches are empty) and what one more ply
would cost there (almost nothing). This measures the other half: how wrong depth 1 and
depth 2 are on those positions, against the answer with no leaf at all.

  scan      Stream the first games of every worker file of a recorded pool and list the
            move decisions whose recorded menu product is at most `--max-product`.
  collect   Per listed position (in parallel, `--jobs`): the shipped search at depth 1,
            depth 2 (the mixed refine) and depth 2 restricted, over the shipped menu
            (`selfplay._menus`, leaf ranking, open information), under the shipped leaf
            and under the same leaf with every ended position scored 1/0 (the IKA-253
            fix); the same game read full-width to 2 and 3 turns (`Tree` with a horizon);
            and the game solved to the end (`Tree` without one): every legal action on
            both sides at every node, every chance branch the matrix budget gives, the
            replacement phase and mid-turn replacements folded as the search folds them,
            1/0 at the end. A root whose tree outgrows `--max-cells` or `--max-turns` is
            recorded as not reached and reported apart.
  analyse   From the stored numbers: per bucket of menu product, each reading's value
            error, strategy distance (total variation) and what its strategy pair gives
            up in the solved game (NashConv against the solved root matrix), and what the
            solve cost.

Chance is the matrix budget's (`Budget.matrix()`: the median damage roll, crits
collapsed), the model every cell of the search is filled with, so the "exact" answer is
exact for the game the search believes it is in. Positions are solved open, as IKA-68
and IKA-281 did: the recorded position is the true one.

    uv run --group learn python tools/endgame_exact.py scan --games-dir data/selfplay-mc0 \\
        --games-per-file 60 --out C:/tmp/ika254/refs.jsonl
    uv run --group learn python tools/endgame_exact.py collect --refs C:/tmp/ika254/refs.jsonl \\
        --games-dir data/selfplay-mc0 --per-bucket 20 --jobs 8 --eps 0.01 --max-cells 30000 \\
        --out C:/tmp/ika254/runs.jsonl \\
        --value data/models/value-mc0.pt data/models/value-mc0-s1.pt
    uv run python tools/endgame_exact.py analyse C:/tmp/ika254/runs.jsonl

Each worker appends one line per position to `<out>.part<N>` as it finishes it, so a run
that is stopped keeps what it did (`cat <out>.part*`); the parts are merged in order at the
end. A position costs about 1.4 ms of wall clock per resolved cell: size `--max-cells` and
the sample so that one run stays within ten to fifteen minutes (IKA-254 §6).

Controls (tests/test_endgame_exact.py): the solver gives 1/9 on the Garchomp-Kingambit
end of IKA-253; `Tree` with a horizon of 1 over the menu equals `search(depth=1)`; and
`--jobs 1` and `--jobs N` write the same records apart from the timings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Menu-product buckets the analysis reports (upper bounds, inclusive).
BUCKETS = (16, 64)

#: The readings compared, in the order they are reported.
READINGS = ("d1", "d2m", "d2r", "w2", "w3")

LeafEvaluator = Callable[[list[Any]], np.ndarray]


# --------------------------------------------------------------------------------------
# the tree


class Capped(RuntimeError):
    """The tree outgrew its caps; the root is recorded as not reached."""


def legal(reg: Any, pos: Any, side: int) -> list[Any]:
    """Every action worth a row: the legal set less the provably dead (`narrow`'s pool)."""
    from pokeuraou.actions import side_actions
    from pokeuraou.narrow import drop_dead_actions

    return drop_dead_actions(reg, pos, side, side_actions(reg, pos, side))


def decided(pos: Any) -> float:
    from pokeuraou.payoff import _decided

    value = _decided(pos)
    assert value is not None
    return value


def with_ends_decided(leaf: LeafEvaluator) -> LeafEvaluator:
    """The leaf with every ended position scored 1/0 (0.5 for a draw), as IKA-253 fixes it.

    Since IKA-253 landed, `BatchedValue` and `RemoteValue` do this themselves
    (`encode.settle`), so over a learned leaf this is a no-op and `asis` equals `fixed`;
    it stays for a leaf that is neither.
    """

    def evaluate(positions: list[Any]) -> np.ndarray:
        out = np.asarray(leaf(positions), dtype=np.float64).copy()
        for k, pos in enumerate(positions):
            if pos.ended:
                out[k] = decided(pos)
        return out

    evaluate.evaluated = 0  # type: ignore[attr-defined]
    return evaluate


def turns(reg: Any, pos: Any, pairs: Sequence[tuple[Any, Any]], budget: Any) -> list[Any]:
    """`port.turn(full=True)` for every pair, pipelined down one pipe (`_exchange_many`).

    The same requests `RustNode.turn` sends, answered in order; the round trip per cell
    was most of a node's cost. tests/test_endgame_exact.py holds it equal to `port.turn`.
    """
    from pokeuraou import port
    from pokeuraou.rustnode import PortTurn, dump_action, dump_budget

    def call(node: Any) -> list[Any]:
        where = pos.to_json()
        spent = dump_budget(budget)
        answers = node._exchange_many([
            {
                "kind": "turn",
                "position": where,
                "actions": [[dump_action(a) for a in side.slots] for side in pair],
                "budget": spent,
                "full": True,
                "select": None,
                "events": False,
            }
            for pair in pairs
        ])
        out = []
        for answer in answers:
            if answer.get("refused"):
                node.refusal = str(answer["refused"])
                raise port.PortRefused(f"the port refused a turn: {node.refusal}")
            out.append(PortTurn.read(answer))
        return out

    return port.ask(reg, call)


def game_value(payoff: np.ndarray) -> float:
    """A zero-sum matrix game's value: a pure saddle read off directly, else the LP."""
    from pokeuraou.equilibrium import solve

    lower = float(payoff.min(axis=1).max())
    upper = float(payoff.max(axis=0).min())
    if upper - lower <= 1e-12:
        return lower
    return float(solve(payoff).value)


@dataclass
class Root:
    """A solved root: the equilibrium of the midpoint matrix, and the bounds around it."""

    equilibrium: Any
    lo: float
    hi: float
    payoff_lo: np.ndarray
    payoff_hi: np.ndarray
    rows: list[Any]
    cols: list[Any]

    @property
    def payoff(self) -> np.ndarray:
        return (self.payoff_lo + self.payoff_hi) / 2


class Tree:
    """A simultaneous-move game tree, solved by a matrix game at every node.

    With `horizon=k` the positions `k` turns below the root are scored by `leaf`, and so is
    every ended position above them (`search._subgame_value` scores an ended position with
    the leaf too; `with_ends_decided` is the IKA-253 fix). Without a horizon there is no
    leaf: an ended position is 1/0, and a position `cap` turns below the root is bounded by
    [0, 1] -- every node carries a lower and an upper value, from the same matrix with the
    bounds put in, and the game's value lies between them because a matrix game's value is
    monotone in its entries. `solve_to_end` deepens `cap` until the root's bounds meet.

    `root_actions` overrides the root's rows and columns (the menu); every other node uses
    `legal`. A move node resolves every cell with `port.turn(full=True)` and folds it with
    `port.turn_leaves` -- chance branches averaged, a mid-turn replacement taken as the
    chooser's best, as the search folds it. A position that owes a replacement is a node of
    its own, the replacement phase as a matrix game (`selfplay._do_replacement_node`'s
    options), and does not use up a turn.
    """

    def __init__(
        self,
        reg: Any,
        budget: Any,
        *,
        horizon: int | None = None,
        leaf: LeafEvaluator | None = None,
        max_nodes: int = 20_000,
        max_cells: int = 10**9,
        cap: int = 64,
        solved: dict[str, float] | None = None,
        delta: float = 0.0,
    ) -> None:
        if (horizon is None) != (leaf is None):
            raise ValueError("a horizon and a leaf go together")
        self.reg = reg
        self.budget = budget
        self.horizon = horizon
        self.leaf = leaf
        self.max_nodes = max_nodes
        self.max_cells = max_cells
        self.cap = cap
        #: A move node stops solving cells once its pure bounds are this close; it then
        #: carries both (they still bracket its value), so the root's bounds stay honest.
        self.delta = delta
        self.memo: dict[str, tuple[float, float]] = {}
        #: Positions whose bounds met, by key: their value holds under any cut, so a deeper
        #: pass of `solve_to_end` reads them instead of solving them again.
        self.solved: dict[str, float] = {} if solved is None else solved
        self.nodes = 0
        self.replacement_nodes = 0
        self.cells = 0
        self.positions = 0
        self.leaves = 0
        self.cut = 0
        self.deepest = 0

    # -- public

    def solve_root(self, pos: Any, root_actions: tuple[list[Any], list[Any]] | None = None) -> Root:
        from pokeuraou.equilibrium import solve

        rows, cols = root_actions or (legal(self.reg, pos, 0), legal(self.reg, pos, 1))
        lo, hi = self._move_matrix(pos, 0, rows, cols)
        low = float(solve(lo).value)
        high = low if np.array_equal(lo, hi) else float(solve(hi).value)
        return Root(solve((lo + hi) / 2), low, high, lo, hi, rows, cols)

    # -- nodes

    def value(self, pos: Any, depth: int) -> tuple[float, float]:
        if pos.ended:
            if self.leaf is None:
                d = decided(pos)
                return d, d
            self.leaves += 1
            v = float(self.leaf([pos])[0])
            return v, v
        # The JSON carries the turn, so a key is also a depth.
        key = json.dumps(pos.to_json(), sort_keys=True)
        known = self.solved.get(key)
        if known is not None:
            return known, known
        if self.horizon is None and depth >= self.cap:
            self.cut += 1
            return 0.0, 1.0
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        from pokeuraou import port

        if self.nodes + self.replacement_nodes >= self.max_nodes:
            raise Capped(f"more than {self.max_nodes} nodes")
        if self.cells >= self.max_cells:
            raise Capped(f"more than {self.max_cells} cells")
        self.deepest = max(self.deepest, depth)
        owed = port.replacements_needed(self.reg, pos)
        if any(owed[0]) or any(owed[1]):
            lo, hi = self._replacement_matrix(pos, depth, owed)
            low, high = game_value(lo), game_value(hi)
        elif self.horizon is None:
            low, high = self._pruned_move(
                pos, depth, legal(self.reg, pos, 0), legal(self.reg, pos, 1)
            )
        else:
            lo, hi = self._move_matrix(
                pos, depth, legal(self.reg, pos, 0), legal(self.reg, pos, 1)
            )
            low, high = game_value(lo), game_value(hi)
        self.memo[key] = (low, high)
        if self.horizon is None and high - low <= 0:
            self.solved[key] = low
        return low, high

    def _known(self, pos: Any) -> tuple[float, float]:
        """What is already known of a position without solving it: [0, 1] if nothing."""
        if pos.ended:
            d = decided(pos)
            return d, d
        key = json.dumps(pos.to_json(), sort_keys=True)
        known = self.solved.get(key)
        if known is not None:
            return known, known
        return self.memo.get(key, (0.0, 1.0))

    def _pruned_move(
        self, pos: Any, depth: int, rows: list[Any], cols: list[Any]
    ) -> tuple[float, float]:
        """A move node's bounds, solving its cells only until a pure saddle closes it.

        Every cell is resolved (one turn each), then priced by what is already known of its
        children. Rows are then solved best-guarantee first and columns best-cap first, in
        turn, and the node stops as soon as the best pure guarantee of the row player (from
        the lower matrix) reaches the best pure cap of the column player (from the upper
        one): the value is then pinned between them. A position one side wins whatever
        happens stops after that side's winning action -- most of the small endgames.
        Otherwise every cell is solved and the matrices are solved as games.
        """
        from pokeuraou import port
        from pokeuraou.fold import fold_value

        self.nodes += 1
        if not rows or not cols:
            raise Capped("a side with no legal action")
        m, n = len(rows), len(cols)
        plans = [
            port.turn_leaves(self.reg, result)
            for result in turns(self.reg, pos, [(r, c) for r in rows for c in cols], self.budget)
        ]
        self.cells += len(plans)
        self.positions += sum(len(p.positions) for p in plans)
        lo = np.empty((m, n), dtype=np.float64)
        hi = np.empty((m, n), dtype=np.float64)
        done = np.zeros((m, n), dtype=bool)

        def price(i: int, j: int, deep: bool) -> None:
            plan = plans[i * n + j]
            got = [self.value(q, depth + 1) if deep else self._known(q) for q in plan.positions]
            lo[i, j] = fold_value(plan.root, [b[0] for b in got])
            hi[i, j] = fold_value(plan.root, [b[1] for b in got])
            if deep or lo[i, j] == hi[i, j]:
                done[i, j] = True

        def closed() -> tuple[float, float] | None:
            lower = float(lo.min(axis=1).max())
            upper = float(hi.max(axis=0).min())
            if upper - lower <= 1e-12:
                return lower, lower
            return (lower, upper) if upper - lower <= self.delta else None

        for i in range(m):
            for j in range(n):
                price(i, j, False)
        pinned = closed()
        if pinned is not None:
            return pinned
        row_order = list(np.argsort(-lo.min(axis=1), kind="stable"))
        col_order = list(np.argsort(hi.max(axis=0), kind="stable"))
        for k in range(max(m, n)):
            for line in ((row_order[k], None) if k < m else None,
                         (None, col_order[k]) if k < n else None):
                if line is None:
                    continue
                i0, j0 = line
                cells = [(i0, j) for j in range(n)] if i0 is not None else [(i, j0) for i in range(m)]
                for i, j in cells:
                    if not done[i, j]:
                        price(i, j, True)
                pinned = closed()
                if pinned is not None:
                    return pinned
        return game_value(lo), game_value(hi)

    def _children(self, positions: list[Any], depth: int) -> list[tuple[float, float]]:
        """Bounds of positions `depth` turns below the root."""
        if self.horizon is not None and depth >= self.horizon:
            assert self.leaf is not None
            self.leaves += len(positions)
            values = [float(v) for v in self.leaf(positions)] if positions else []
            return [(v, v) for v in values]
        return [self.value(p, depth) for p in positions]

    def _move_matrix(
        self, pos: Any, depth: int, rows: list[Any], cols: list[Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        from pokeuraou import port
        from pokeuraou.fold import fold_value

        self.nodes += 1
        if not rows or not cols:
            raise Capped("a side with no legal action")
        plans = [
            port.turn_leaves(self.reg, result)
            for result in turns(self.reg, pos, [(r, c) for r in rows for c in cols], self.budget)
        ]
        self.cells += len(plans)
        self.positions += sum(len(p.positions) for p in plans)
        if self.horizon is not None and depth + 1 >= self.horizon:
            # One leaf call for the whole node, as the search makes it.
            flat = [q for p in plans for q in p.positions]
            bounds = self._children(flat, depth + 1)
            per, at = [], 0
            for p in plans:
                per.append(bounds[at : at + len(p.positions)])
                at += len(p.positions)
        else:
            per = [self._children(p.positions, depth + 1) for p in plans]
        lo = [fold_value(p.root, [b[0] for b in got]) for p, got in zip(plans, per, strict=True)]
        hi = [fold_value(p.root, [b[1] for b in got]) for p, got in zip(plans, per, strict=True)]
        shape = (len(rows), len(cols))
        return (np.array(lo, dtype=np.float64).reshape(shape),
                np.array(hi, dtype=np.float64).reshape(shape))

    def _replacement_matrix(
        self, pos: Any, depth: int, owed: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        from pokeuraou import port
        from pokeuraou.actions import PassAction, SideAction, switch_actions_after_faint

        self.replacement_nodes += 1
        options: list[list[Any]] = []
        for side in range(2):
            must = list(owed[side])
            found = switch_actions_after_faint(self.reg, pos, side, must) if any(must) else []
            if not found:
                found = [
                    SideAction(
                        slots=tuple(
                            PassAction(slot=s) for s in range(len(pos.sides[side].active))
                        )
                    )
                ]
            options.append(found)
        after = [
            port.resolve_replacements(self.reg, pos, [a, b]).position
            for a in options[0]
            for b in options[1]
        ]
        # The phase happens within the turn: the positions it leads to are at this depth.
        bounds = [self.value(p, depth) for p in after]
        shape = (len(options[0]), len(options[1]))
        return (np.array([b[0] for b in bounds], dtype=np.float64).reshape(shape),
                np.array([b[1] for b in bounds], dtype=np.float64).reshape(shape))


def solve_to_end(
    reg: Any,
    pos: Any,
    budget: Any,
    *,
    eps: float,
    max_turns: int,
    max_nodes: int,
    max_cells: int = 10**9,
    needed: tuple[set[str], set[str]] | None = None,
) -> tuple[Root, dict[str, Any]]:
    """The game from `pos` solved to within `eps`, deepening the [0, 1] cut a turn at a time.

    `needed` names root actions (as choices) some strategy to be judged plays. What that
    strategy gives up is read from its rows against every column and every row against its
    columns, so once the value has met the cut keeps deepening until those cells are within
    `eps` as well -- or until the caps, and then the root of the last pass is returned with
    `open` saying how far they still were. Raises `Capped` when the value itself never met.
    """
    spent: dict[str, Any] = {"nodes": 0, "replacement_nodes": 0, "cells": 0, "positions": 0}
    last: tuple[float, float] | None = None
    best: Root | None = None
    previous: Root | None = None
    solved: dict[str, float] = {}

    def open_cells(root: Root) -> float:
        gap = root.payoff_hi - root.payoff_lo
        if needed is None:
            return 0.0
        rows = [i for i, a in enumerate(root.rows) if a.to_choice() in needed[0]]
        cols = [j for j, a in enumerate(root.cols) if a.to_choice() in needed[1]]
        parts = [gap[rows, :].ravel(), gap[:, cols].ravel(), np.zeros(1)]
        return float(np.concatenate(parts).max())

    for cap in range(1, max_turns + 1):
        tree = Tree(reg, budget, max_nodes=max_nodes - spent["nodes"] - spent["replacement_nodes"],
                    max_cells=max_cells - spent["cells"], cap=cap, solved=solved, delta=eps / 4)
        try:
            root = tree.solve_root(pos)
        except Capped as why:
            for k in ("nodes", "replacement_nodes", "cells", "positions"):
                spent[k] += getattr(tree, k)
            if best is not None:
                spent["open"] = open_cells(best)
                spent["stopped"] = str(why)
                return best, spent
            raise Capped(
                f"{why} (turn cap {cap}, last bounds {last})", spent, last, previous
            ) from None
        for k in ("nodes", "replacement_nodes", "cells", "positions"):
            spent[k] += getattr(tree, k)
        last = (root.lo, root.hi)
        previous = root
        if root.hi - root.lo <= eps:
            if best is None:
                # What the value alone cost, before any deepening for the judged cells.
                spent["turns"] = cap
                spent["deepest"] = tree.deepest
                spent["value_nodes"] = spent["nodes"] + spent["replacement_nodes"]
                spent["value_cells"] = spent["cells"]
            best = root
            if open_cells(root) <= eps:
                spent["open"] = open_cells(root)
                spent["cell_turns"] = cap
                return root, spent
    if best is not None:
        spent["open"] = open_cells(best)
        spent["stopped"] = f"turn cap {max_turns}"
        return best, spent
    raise Capped(f"bounds {last} after {max_turns} turns", spent, last, previous)


# --------------------------------------------------------------------------------------
# scan


def scan(games_dir: Path, games_per_file: int, max_product: int) -> list[dict[str, Any]]:
    """Every move decision with a recorded menu product at most `max_product`."""
    found: list[dict[str, Any]] = []
    for path in sorted(games_dir.glob("games-worker*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if line_no >= games_per_file:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k, d in enumerate(record.get("decisions", ())):
                    if d.get("kind") != "move":
                        continue
                    product = len(d.get("ownActions") or ()) * len(d.get("foeActions") or ())
                    if 0 < product <= max_product:
                        found.append({
                            "ref": [path.name, line_no, k],
                            "turn": int(d.get("turn", 0)),
                            "product": product,
                            "bench": bench_counts(d["position"]),
                        })
    return found


def bench_counts(position: dict[str, Any]) -> list[int]:
    """Standing Pokemon off the field, per side (from the position JSON)."""
    out = []
    for side in position["sides"]:
        n = 0
        for mon in side.get("pokemon") or side.get("team") or ():
            if mon.get("fainted") or (mon.get("hp", 1) <= 0):
                continue
            if mon.get("activeIndex") is None and mon.get("active_index") is None:
                n += 1
        out.append(n)
    return out


def read_decision(games_dir: Path, ref: Sequence[Any]) -> dict[str, Any]:
    name, line_no, k = ref
    with (games_dir / name).open(encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            if n == line_no:
                return json.loads(line)["decisions"][k]
    raise KeyError(ref)


def choose(refs: list[dict[str, Any]], per_bucket: int, seed: int) -> list[dict[str, Any]]:
    """`per_bucket` positions from each product bucket, shuffled with `seed`."""
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    low = 0
    for high in BUCKETS:
        members = [r for r in refs if low < r["product"] <= high]
        rng.shuffle(members)
        chosen.extend(members[:per_bucket])
        low = high
    return chosen


# --------------------------------------------------------------------------------------
# collect


def strategy(actions: Sequence[Any], probs: Sequence[float]) -> dict[str, float]:
    return {a.to_choice(): float(p) for a, p in zip(actions, probs, strict=True) if p > 1e-9}


def collect_one(
    reg: Any,
    pos: Any,
    leaf: Any,
    *,
    limit: int,
    max_nodes: int,
    max_cells: int,
    max_turns: int,
    wide: int,
    eps: float,
    exact_budget: Any = None,
) -> dict[str, Any]:
    from pokeuraou.budget import Budget
    from pokeuraou.port import PortRefused
    from pokeuraou.search import search
    from pokeuraou.selfplay import _menus

    budget = Budget.matrix()
    ours, theirs = _menus(reg, pos, (limit, limit), leaf, budget, True)
    pools = (legal(reg, pos, 0), legal(reg, pos, 1))
    out: dict[str, Any] = {
        "menu": [[a.to_choice() for a in ours], [a.to_choice() for a in theirs]],
        "pool": [[a.to_choice() for a in pools[0]], [a.to_choice() for a in pools[1]]],
    }
    fixed = with_ends_decided(leaf)
    readings: dict[str, dict[str, Any]] = {}
    for name, evaluate in (("asis", leaf), ("fixed", fixed)):
        got: dict[str, Any] = {}
        for label, kwargs in (
            ("d1", {"depth": 1}),
            ("d2m", {"depth": 2}),
            ("d2r", {"depth": 2, "solve_restricted": True}),
        ):
            before = leaf.evaluated
            started = time.perf_counter()
            result = search(reg, pos, ours, theirs, evaluate, budget=budget, **kwargs)
            eq = result.equilibrium
            got[label] = {
                "value": float(eq.value),
                "row": strategy(ours, eq.row_strategy),
                "col": strategy(theirs, eq.col_strategy),
                "seconds": round(time.perf_counter() - started, 4),
                "leaves": int(leaf.evaluated - before),
                "refined": int(result.refined),
                "subgames": int(result.subgames),
            }
        # Full width to 2 and 3 turns, when the pools are small enough to afford it.
        for label, horizon in (("w2", 2), ("w3", 3)):
            if len(pools[0]) * len(pools[1]) > (wide if horizon == 2 else wide // 4):
                continue
            before = leaf.evaluated
            started = time.perf_counter()
            tree = Tree(reg, budget, horizon=horizon, leaf=evaluate, max_nodes=max_nodes,
                        max_cells=max_cells)
            try:
                root = tree.solve_root(pos)
            except (Capped, PortRefused) as why:
                got[label] = {"status": f"capped: {why}"}
                continue
            eq = root.equilibrium
            got[label] = {
                "value": float(eq.value),
                "row": strategy(root.rows, eq.row_strategy),
                "col": strategy(root.cols, eq.col_strategy),
                "seconds": round(time.perf_counter() - started, 4),
                "leaves": int(leaf.evaluated - before),
                "nodes": tree.nodes + tree.replacement_nodes,
            }
        readings[name] = got
    out["readings"] = readings

    started = time.perf_counter()
    cpu = time.process_time()
    spent: dict[str, Any] = {}
    last = None
    try:
        needed: tuple[set[str], set[str]] = (set(), set())
        for got in readings.values():
            for reading in got.values():
                needed[0].update(reading.get("row", {}))
                needed[1].update(reading.get("col", {}))
        root, spent = solve_to_end(
            reg, pos, exact_budget or budget, eps=eps, max_turns=max_turns, max_nodes=max_nodes,
            max_cells=max_cells,
            needed=needed,
        )
        status = "ok"
    except Capped as why:
        status = f"capped: {why.args[0]}"
        if len(why.args) > 1:
            spent, last = why.args[1], why.args[2]
            root = why.args[3]
    except PortRefused as why:
        status = f"refused: {why}"
    exact: dict[str, Any] = {
        "status": status,
        "seconds": round(time.perf_counter() - started, 4),
        "cpu": round(time.process_time() - cpu, 4),
        **spent,
    }
    if status == "ok":
        eq = root.equilibrium
        exact.update({
            "value": float(eq.value),
            "lo": root.lo,
            "hi": root.hi,
            "row": strategy(root.rows, eq.row_strategy),
            "col": strategy(root.cols, eq.col_strategy),
            "payoff": np.round(root.payoff, 12).tolist(),
            "payoff_gap": np.round(root.payoff_hi - root.payoff_lo, 12).tolist(),
        })
    elif last is not None:
        exact["bounds"] = list(last)
        # The last pass that finished: its bounds on every root cell, so a strategy's
        # NashConv can still be bracketed where the value was not pinned.
        eq = root.equilibrium
        exact["partial"] = {
            "value": float(eq.value),
            "lo": root.lo,
            "hi": root.hi,
            "row": strategy(root.rows, eq.row_strategy),
            "col": strategy(root.cols, eq.col_strategy),
            "payoff": np.round(root.payoff, 12).tolist(),
            "payoff_gap": np.round(root.payoff_hi - root.payoff_lo, 12).tolist(),
        }
    out["exact"] = exact
    return out


def run_collect(args: argparse.Namespace) -> None:
    if args.jobs > 1 and args.worker is None:
        spawn(args)
        return
    import torch

    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.position import Position
    from pokeuraou.regulation import load_regulation
    from pokeuraou.value import BatchedValue, load_ensemble

    torch.set_num_threads(1)
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    refs = [json.loads(line) for line in Path(args.refs).read_bytes().splitlines() if line]
    chosen = choose(refs, args.per_bucket, args.seed)
    worker = args.worker or 0
    mine = [(k, r) for k, r in enumerate(chosen) if k % args.jobs == worker]
    out = Path(args.out) if args.worker is None else Path(f"{args.out}.part{worker}")
    reg = None
    leaf: Any = None
    with out.open("wb") as handle:
        for k, ref in mine:
            decision = read_decision(args.games_dir, ref["ref"])
            pos = Position.from_json(decision["position"])
            if reg is None:
                reg = load_regulation(pos.format)
                register_mega_stones(reg)
                encoder = Encoder(reg)
                nets, _ = load_ensemble(list(args.value), encoder)
                device = torch.device(args.device)
                leaf = BatchedValue([n.to(device) for n in nets], encoder, device=device)
            started = time.perf_counter()
            row: dict[str, Any] = {"index": k, **ref}
            row.update(collect_one(
                reg, pos, leaf, limit=args.limit, max_nodes=args.max_nodes,
                max_cells=args.max_cells,
                max_turns=args.max_turns, wide=args.wide, eps=args.eps,
            ))
            row["seconds"] = round(time.perf_counter() - started, 3)
            handle.write((json.dumps(row) + "\n").encode("utf-8"))
            handle.flush()
            print(f"  [{worker}] #{k} p{ref['product']} {row['exact']['status'][:6]} "
                  f"{row['seconds']:.1f}s", flush=True)


def spawn(args: argparse.Namespace) -> None:
    """One process per job, each taking every `jobs`-th position; then one file."""
    base = [sys.executable, str(Path(__file__).resolve()), "collect"]
    passed = [
        "--refs", str(args.refs), "--games-dir", str(args.games_dir),
        "--per-bucket", str(args.per_bucket), "--seed", str(args.seed),
        "--limit", str(args.limit), "--max-nodes", str(args.max_nodes),
        "--max-cells", str(args.max_cells),
        "--max-turns", str(args.max_turns), "--wide", str(args.wide),
        "--eps", str(args.eps),
        "--device", args.device, "--jobs", str(args.jobs), "--out", str(args.out),
        "--value", *args.value,
    ]
    procs = [subprocess.Popen([*base, *passed, "--worker", str(w)]) for w in range(args.jobs)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a worker failed: {codes}")
    rows = []
    for w in range(args.jobs):
        part = Path(f"{args.out}.part{w}")
        rows.extend(part.read_bytes().splitlines())
        part.unlink()
    rows.sort(key=lambda line: json.loads(line)["index"])
    Path(args.out).write_bytes(b"".join(line + b"\n" for line in rows))


# --------------------------------------------------------------------------------------
# analyse


def as_vector(strat: dict[str, float], actions: Sequence[str]) -> np.ndarray:
    """A stored strategy over `actions` (probability outside them is dropped and counted)."""
    index = {a: i for i, a in enumerate(actions)}
    out = np.zeros(len(actions), dtype=np.float64)
    for a, p in strat.items():
        if a in index:
            out[index[a]] += p
    return out


def compare(reading: dict[str, Any], exact: dict[str, Any], pool: list[list[str]]) -> dict[str, float]:
    """Value error, strategy distance and NashConv of one reading in the solved game."""
    a = np.asarray(exact["payoff"], dtype=np.float64)
    x = as_vector(reading["row"], pool[0])
    y = as_vector(reading["col"], pool[1])
    xs = as_vector(exact["row"], pool[0])
    ys = as_vector(exact["col"], pool[1])
    v = float(exact["value"])
    row_loss = v - float((x @ a).min())
    col_loss = float((a @ y).max()) - v
    # NashConv is max_i (A y)_i - min_j (x A)_j, so with every cell in [lo, hi] it lies
    # between the same sum at the two corners of the box the [0, 1] cut leaves.
    gap = np.asarray(exact.get("payoff_gap", np.zeros_like(a)), dtype=np.float64)
    lo, hi = a - gap / 2, a + gap / 2
    least = max(0.0, float((lo @ y).max()) - float((x @ hi).min()))
    most = float((hi @ y).max()) - float((x @ lo).min())
    return {
        "err": float(reading["value"]) - v,
        "tv_row": 0.5 * float(np.abs(x - xs).sum()),
        "tv_col": 0.5 * float(np.abs(y - ys).sum()),
        "row_loss": row_loss,
        "col_loss": col_loss,
        "exploit": row_loss + col_loss,
        "exploit_open": most - least,
        "exploit_min": least,
        "exploit_max": most,
        "outside": float(2 - x.sum() - y.sum()),
    }


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.5, 0.9, 0.99)) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "-"
    return "/".join(f"{np.quantile(arr, q):.3g}" for q in qs) + f" ({arr.mean():.3g})"


def bucket_of(product: int) -> str:
    low = 0
    for high in BUCKETS:
        if low < product <= high:
            return f"<={high}" if low == 0 else f"{low + 1}-{high}"
        low = high
    return f">{BUCKETS[-1]}"


def interval(values: Sequence[float], rng: np.random.Generator, draws: int = 2000) -> str:
    """Mean and a bootstrap 95% interval."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "-"
    means = arr[rng.integers(0, arr.size, size=(draws, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return f"{arr.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"


#: A solved value this close to 0 or 1 is a decided position: every strategy wins (loses).
DECIDED = 0.01

#: One decision's budget in generation, in CPU seconds on one logical core: the median
#: 1,670 leaves at 26,000 leaves a second per logical core (IKA-279).
DECISION_SECONDS = 1670 / 26000


def _status(row: dict[str, Any]) -> str:
    s = row["exact"]["status"]
    if s == "ok":
        return "ok"
    if "nodes" in s:
        return "nodes"
    if " cells" in s:
        return "cells"
    if "bounds" in s:
        return "turns"
    return s.split(":")[0]


def analyse(rows: list[dict[str, Any]]) -> str:
    rng = np.random.default_rng(254)
    out: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(bucket_of(r["product"]), []).append(r)
        groups.setdefault("<=64", []).append(r)
    order = [bucket_of(b) for b in BUCKETS] + ["<=64"]

    out.append("## reach (solved: root bounds within eps; not reached: the node or turn cap)")
    for g in order:
        members = groups.get(g, [])
        by: dict[str, int] = {}
        for r in members:
            by[_status(r)] = by.get(_status(r), 0) + 1
        benched = [r for r in members if sum(r["bench"]) > 0]
        same = sum(
            1 for r in members
            if sorted(r["menu"][0]) == sorted(r["pool"][0])
            and sorted(r["menu"][1]) == sorted(r["pool"][1])
        )
        ok = [r for r in members if _status(r) == "ok"]
        decided_n = sum(1 for r in ok if min(r["exact"]["value"], 1 - r["exact"]["value"]) <= DECIDED)
        out.append(
            f"  {g:6s} positions {len(members):4d}  {by}  bench {len(benched)} "
            f"(solved {sum(1 for r in benched if _status(r) == 'ok')})  menu==pool {same}  "
            f"solved and decided (<{DECIDED} from 0/1) {decided_n}"
        )
        widths = [
            r["exact"]["bounds"][1] - r["exact"]["bounds"][0]
            for r in members if r["exact"].get("bounds")
        ]
        if widths:
            out.append(f"         not reached: root bounds width p50/p90/p99 (mean) {quantiles(widths)}")

    for part, keep in (
        ("all solved", lambda r: True),
        ("open (solved value in (0.01, 0.99))",
         lambda r: min(r["exact"]["value"], 1 - r["exact"]["value"]) > DECIDED),
    ):
        for leafname in ("asis", "fixed"):
            out.append(f"\n## {part}, leaf {leafname}")
            out.append(
                "  reading   n    V-V* mean   |V-V*| p50/p90/p99 (mean)   TV p50/p90 (mean)   "
                "exploit mean [95%]   p50/p90/p99   >0.01  >0.05   open"
            )
            for g in order:
                ok = [r for r in groups.get(g, []) if _status(r) == "ok" and keep(r)]
                if not ok:
                    continue
                out.append(f"  -- {g} ({len(ok)} positions)")
                for reading in READINGS:
                    got = [
                        compare(r["readings"][leafname][reading], r["exact"], r["pool"])
                        for r in ok
                        if "value" in r["readings"][leafname].get(reading, {})
                    ]
                    if not got:
                        continue
                    err = np.array([c["err"] for c in got])
                    tv = [(c["tv_row"] + c["tv_col"]) / 2 for c in got]
                    ex = np.array([c["exploit"] for c in got])
                    op = [c["exploit_max"] - c["exploit_min"] for c in got]
                    out.append(
                        f"  {reading:4s} {len(got):4d}  {err.mean():+.4f}   {quantiles(np.abs(err))}   "
                        f"{quantiles(tv, (0.5, 0.9))}   {interval(ex, rng)}   {quantiles(ex)}   "
                        f"{np.mean(ex > 0.01):5.1%}  {np.mean(ex > 0.05):5.1%}   {np.max(op):.1e}"
                    )

    out.append(
        "\n## not reached: NashConv bracketed by the last finished pass "
        "(every root cell within its [lo, hi])"
    )
    out.append(
        "  reading   n   width of the root value (mean)   NashConv at least (mean)   at most (mean)   "
        "at least > 0.01   at most <= 0.01"
    )
    for leafname in ("asis", "fixed"):
        for g in order:
            part = [r for r in groups.get(g, []) if r["exact"].get("partial")]
            if not part:
                continue
            out.append(f"  -- leaf {leafname}, {g} ({len(part)} positions)")
            width = np.mean([r["exact"]["partial"]["hi"] - r["exact"]["partial"]["lo"] for r in part])
            for reading in READINGS:
                got = [
                    compare(r["readings"][leafname][reading], r["exact"]["partial"], r["pool"])
                    for r in part
                    if "value" in r["readings"][leafname].get(reading, {})
                ]
                if not got:
                    continue
                least = np.array([c["exploit_min"] for c in got])
                most = np.array([c["exploit_max"] for c in got])
                out.append(
                    f"  {reading:4s} {len(got):4d}   {width:.3f}   {least.mean():.4f}   {most.mean():.4f}   "
                    f"{np.mean(least > 0.01):5.1%}   {np.mean(most <= 0.01):5.1%}"
                )

    out.append("\n## paired differences in exploit (same positions), all solved")
    pairs = (
        ("asis d2r - d1", ("asis", "d2r"), ("asis", "d1")),
        ("asis d2m - d1", ("asis", "d2m"), ("asis", "d1")),
        ("fixed d1 - asis d1", ("fixed", "d1"), ("asis", "d1")),
        ("fixed d2r - asis d2r", ("fixed", "d2r"), ("asis", "d2r")),
        ("fixed d2r - fixed d1", ("fixed", "d2r"), ("fixed", "d1")),
    )
    for g in order:
        ok = [r for r in groups.get(g, []) if _status(r) == "ok"]
        for label, (la, ra), (lb, rb) in pairs:
            diffs = [
                compare(r["readings"][la][ra], r["exact"], r["pool"])["exploit"]
                - compare(r["readings"][lb][rb], r["exact"], r["pool"])["exploit"]
                for r in ok
            ]
            if diffs:
                out.append(f"  {g:6s} {label:22s} {interval(diffs, rng)}")

    out.append(
        f"\n## cost of the solve (one logical core; a decision's budget = {DECISION_SECONDS:.3f} s)"
    )
    for g in order:
        members = groups.get(g, [])
        for label, keep in (("solved", lambda r: _status(r) == "ok"),
                            ("all", lambda r: True)):
            e = [r["exact"] for r in members if keep(r)]
            if not e:
                continue
            nodes = [x.get("nodes", 0) + x.get("replacement_nodes", 0) for x in e]
            cells = [x.get("cells", 0) for x in e]
            depth = [x.get("turns", 0) for x in e]
            wall = [x["seconds"] for x in e]
            out.append(
                f"  {g:6s} {label:6s} n {len(e):3d}  nodes {quantiles(nodes)}  "
                f"cells {quantiles(cells)}  turns {quantiles(depth)}  wall s {quantiles(wall)}  "
                f"x budget {quantiles([w / DECISION_SECONDS for w in wall])}"
            )
        ok = [r for r in members if _status(r) == "ok"]
        for reading in READINGS:
            got = [r["readings"]["asis"].get(reading, {}) for r in ok]
            got = [x for x in got if "seconds" in x]
            if got:
                out.append(
                    f"         {reading:4s} wall s {quantiles([x['seconds'] for x in got])}  "
                    f"leaves {quantiles([x['leaves'] for x in got])}"
                )
    return "\n".join(out)


def run_analyse(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.runs).read_bytes().splitlines() if line]
    print(analyse(rows))


# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--games-dir", type=Path, default=ROOT / "data" / "selfplay-mc0")
    s.add_argument("--games-per-file", type=int, default=60)
    s.add_argument("--max-product", type=int, default=BUCKETS[-1])
    s.add_argument("--out", type=Path, required=True)

    c = sub.add_parser("collect")
    c.add_argument("--refs", type=Path, required=True)
    c.add_argument("--games-dir", type=Path, default=ROOT / "data" / "selfplay-mc0")
    c.add_argument("--per-bucket", type=int, default=150)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--limit", type=int, default=12, help="menu width (generation's 12)")
    c.add_argument("--max-nodes", type=int, default=20_000)
    c.add_argument("--max-cells", type=int, default=30_000,
                   help="resolved cells per position (all passes); about 1.4 ms each, so the "
                   "default caps a position near 40 s")
    c.add_argument("--max-turns", type=int, default=30)
    c.add_argument("--eps", type=float, default=1e-3,
                   help="the solved root's bounds must meet to within this")
    c.add_argument("--wide", type=int, default=64,
                   help="full-width 2-turn reading up to this pool product, 3-turn up to a quarter")
    c.add_argument("--device", default="cuda")
    c.add_argument("--jobs", type=int, default=1)
    c.add_argument("--worker", type=int, default=None)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--value", nargs="+", required=True)

    a = sub.add_parser("analyse")
    a.add_argument("runs", type=Path)

    args = parser.parse_args(argv)
    if args.command == "scan":
        found = scan(args.games_dir, args.games_per_file, args.max_product)
        args.out.write_bytes(b"".join((json.dumps(r) + "\n").encode("utf-8") for r in found))
        by = {bucket_of(r["product"]): 0 for r in found}
        for r in found:
            by[bucket_of(r["product"])] += 1
        print(f"{len(found)} decisions: {by}")
    elif args.command == "collect":
        run_collect(args)
    else:
        run_analyse(args)


if __name__ == "__main__":
    main()
