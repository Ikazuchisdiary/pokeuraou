"""IKA-381: the ladder's LPs and sub-game folds solved in the port (HiGHS in Rust), off by default.

`equilibrium.solve` spends more of an LP in Python than in HiGHS: building the model, handing
it over and reading the answer (0.31 ms an LP against 0.22 ms of HiGHS, IKA-379), and a
ladder worker solves about 110 of the Q's games and 70 sub-games a node second. The port
solves them with the same HiGHS (1.12.0, scipy's commit), the same model and the same options
(`rust/src/lp.rs`), so an answer is `solve`'s to the bit wherever HiGHS takes the same pivots
-- all 59,050 LPs captured from open and hidden reads did (records/IKA-381.md).

Two roads, both one crossing for many games:

* `solve_many`: the Q's games of one `deepen._q_menus` call. The port answers both LPs' values
  and the cleaned strategies; the rest of `Equilibrium` is built here as `solve` builds it
  (`equilibrium.assemble`), so the menus are the same menus.
* `solve_subs`: the sub-games of one `search._refine_cells` call once their leaves are scored.
  A filled node is folded in the port from the span block its `fills` answer carried, its
  replacements' fold trees and the leaf values -- `port._folded`'s arithmetic: numpy's dot in
  the order this machine's OpenBLAS takes it (the same bits here, the last place elsewhere),
  `fold.fold_value`'s Python sums -- and solved; a matrix already whole (the hp-share road) is
  solved as it is.

IKA-387 adds the rest of the ladder's LPs, on the same `lp` command:

* `solve_many` again for a deep cell's children's own matrices (`ladder._kids_fill`), all of
  one call in one crossing;
* `solve_bayesian_many` for their pass rectangles (`ladder._kids_solve`: a pass's children
  in one crossing; the reader's `deep_passes` gathers the children whose pass came back in
  one round of answers), the answer built by `equilibrium.assemble_bayesian` as
  `solve_bayesian` builds it;
* `solve_bayesian_one` for a stage's rectangle (`ladder.read`) and the answer a stage's
  tail reads ahead from (`ladder._Ahead._answer`).

    POKEURAOU_LADDER_PORT_LP=1     # on (default off; data generation never sets it)

Off, nothing here runs and every answer is Python's. A game the port cannot solve is
`solve`'s error (`EquilibriumError`, or `ValueError` for a non-finite matrix), where `solve`
would have raised it.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import rustnode
from .equilibrium import (
    BayesianEquilibrium,
    Equilibrium,
    EquilibriumError,
    assemble,
    assemble_bayesian,
    bayesian_inputs,
    solve,
    solve_bayesian,
)

ENV = "POKEURAOU_LADDER_PORT_LP"
#: Whether the ladder's LPs go to the port. A list so a test can turn it on in one process.
ON = [os.environ.get(ENV, "0").strip() == "1"]
#: What went: crossings, games (`solve_many`), Bayesian games (`solve_bayesian_many`),
#: sub-games folded in the port and given whole, and the LPs the port reports it solved (the
#: positive control).
COUNTS = {"crossings": 0, "games": 0, "bayes": 0, "folded": 0, "whole": 0, "lps": 0}


def set_on(on: bool) -> None:
    """Turn the port's LPs on or off in this process (and the span blocks they need)."""
    ON[0] = bool(on)
    rustnode._SPAN_BLOCKS[0] = bool(on)


set_on(ON[0])


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f8").tobytes()).decode("ascii")


def _unb64(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype="<f8").astype(np.float64)


def _game(a: np.ndarray) -> dict[str, Any]:
    return {"rows": int(a.shape[0]), "cols": int(a.shape[1]), "data": _b64(a)}


def _ask(reg: Any, request: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401 - a Regulation
    from . import port

    answer = port.ask(reg, lambda node: node.lp(request))
    COUNTS["crossings"] += 1
    COUNTS["lps"] += int(answer.get("lps", 0))
    return answer


def _error(answer: dict[str, Any]) -> Exception | None:
    if "failed" in answer:
        return EquilibriumError(str(answer["failed"]))
    if "invalid" in answer:
        return ValueError(str(answer["invalid"]))
    return None


def solve_many(
    reg: Any, payoffs: Sequence[np.ndarray]  # noqa: ANN401 - a Regulation
) -> list[Equilibrium | Exception]:
    """`equilibrium.solve` of each matrix in one crossing: its `Equilibrium`, or the error
    `solve` would have raised (returned in its place, not raised)."""
    if not payoffs:
        return []
    mats = [np.asarray(p, dtype=np.float64) for p in payoffs]
    asked = [a for a in mats if a.ndim == 2 and a.size]
    answers = (_ask(reg, {"kind": "lp", "games": [_game(a) for a in asked]})["answers"]
               if asked else [])
    if len(answers) != len(asked):
        raise RuntimeError(f"`lp` answered {len(answers)} of {len(asked)} games")
    got = iter(answers)
    COUNTS["games"] += len(asked)
    out: list[Equilibrium | Exception] = []
    for a in mats:
        if a.ndim != 2 or a.size == 0:
            out.append(ValueError(f"payoff must be a non-empty 2-D array, got shape {a.shape}"))
            continue
        one = next(got)
        error = _error(one)
        if error is not None:
            out.append(error)
            continue
        out.append(assemble(a, float(one["valueRow"]), float(one["valueCol"]),
                            _unb64(one["x"]), _unb64(one["ys"][0])))
    return out


def solve_bayesian_many(
    reg: Any,  # noqa: ANN401 - a Regulation
    games: Sequence[tuple[Sequence[np.ndarray], np.ndarray]],
) -> list[BayesianEquilibrium | Exception]:
    """`equilibrium.solve_bayesian` of each ``(matrices, weights)`` in one crossing (IKA-387):
    its `BayesianEquilibrium`, or the error `solve_bayesian` would have raised (returned in
    its place). The port is sent the weights as given and normalises them as Python does."""
    if not games:
        return []
    checked: list[tuple[list[np.ndarray], np.ndarray] | Exception] = []
    asked: list[dict[str, Any]] = []
    for mats, weights in games:
        try:
            got = bayesian_inputs(list(mats), weights)
        except ValueError as exc:
            checked.append(exc)
            continue
        checked.append(got)
        asked.append({"bayes": [_game(a) for a in got[0]],
                      "weights": _b64(np.asarray(weights, dtype=np.float64).reshape(-1))})
    answers = _ask(reg, {"kind": "lp", "games": asked})["answers"] if asked else []
    if len(answers) != len(asked):
        raise RuntimeError(f"`lp` answered {len(answers)} of {len(asked)} games")
    got_answers = iter(answers)
    COUNTS["bayes"] += len(asked)
    out: list[BayesianEquilibrium | Exception] = []
    for one_game in checked:
        if isinstance(one_game, Exception):
            out.append(one_game)
            continue
        mats, w = one_game
        one = next(got_answers)
        error = _error(one)
        if error is not None:
            out.append(error)
            continue
        out.append(assemble_bayesian(mats, w, float(one["valueRow"]), float(one["valueCol"]),
                                     _unb64(one["x"]), tuple(_unb64(y) for y in one["ys"])))
    return out


def solve_one(reg: Any, payoff: np.ndarray) -> Equilibrium:  # noqa: ANN401 - a Regulation
    """`equilibrium.solve` of one matrix -- in the port when `ON`, else in Python -- raising
    as `solve` raises (IKA-387)."""
    if not ON[0]:
        return solve(payoff)
    got = solve_many(reg, [payoff])[0]
    if isinstance(got, Exception):
        raise got
    return got


def solve_bayesian_one(
    reg: Any, matrices: Sequence[np.ndarray], weights: np.ndarray  # noqa: ANN401 - a Regulation
) -> BayesianEquilibrium:
    """`equilibrium.solve_bayesian` of one game -- in the port when `ON`, else in Python --
    raising as it raises (IKA-387)."""
    if not ON[0]:
        return solve_bayesian(list(matrices), weights)
    got = solve_bayesian_many(reg, [(matrices, weights)])[0]
    if isinstance(got, Exception):
        raise got
    return got


def solve_subs(reg: Any, subs: Sequence[Any]) -> None:  # noqa: ANN401 - a Regulation, `search._Sub`s
    """Solve, in one crossing, every sub-game of `subs` that `search._sub_value` would solve:
    each filled one not solved yet (an alias's first in its place). Sets its ``solved`` and
    ``done`` as `_sub_value` does; a non-finite matrix raises `solve`'s ValueError."""
    todo: list[Any] = []
    seen: set[int] = set()
    for sub in subs:
        first = sub.alias if sub.alias is not None else sub
        if (id(first) in seen or first.done or first.error is not None or first.shared
                or first.ended or first.empty):
            continue
        pending = first.pending
        if pending is None and first.payoff is None:
            continue
        if pending is not None and pending.values is None:
            continue
        seen.add(id(first))
        todo.append(first)
    if not todo:
        return
    nodes: list[dict[str, Any]] = []
    for sub in todo:
        pending = sub.pending
        block = None if pending is None else pending.filled.span_block
        if pending is not None and block is not None:
            raw, count, total = block
            m, n = pending.shape
            node = {
                "rows": m, "cols": n, "spanCount": count, "spanLeaves": total,
                "spans": base64.b64encode(raw).decode("ascii"),
                "values": _b64(pending.values),
            }
            if pending.filled.folded:
                # The replacements' trees as JSON text, each weight its shortest round-trip
                # text: the port reads it back correctly rounded, the double it first wrote.
                node["folded"] = json.dumps(pending.filled.folded)
            nodes.append(node)
            COUNTS["folded"] += 1
        else:
            payoff = sub.payoff if pending is None else pending.finish()
            nodes.append(_game(np.asarray(payoff, dtype=np.float64)))
            COUNTS["whole"] += 1
    answer = _ask(reg, {"kind": "folds", "nodes": nodes})
    got = answer["answers"]
    if len(got) != len(todo):
        raise RuntimeError(f"`folds` answered {len(got)} of {len(todo)} sub-games")
    for sub, one in zip(todo, got, strict=True):
        if "invalid" in one:
            raise ValueError(str(one["invalid"]))
        sub.solved = None if "failed" in one else float(one["value"])
        sub.done = True


def counts() -> dict[str, int]:
    return dict(COUNTS)


__all__ = ["COUNTS", "ENV", "ON", "counts", "set_on", "solve_bayesian_many",
           "solve_bayesian_one", "solve_many", "solve_one", "solve_subs"]
