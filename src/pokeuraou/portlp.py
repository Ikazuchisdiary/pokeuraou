"""The LPs the port solves (HiGHS in Rust), and the one road a game's nodes take.

`equilibrium.solve` spends more of an LP in Python than in HiGHS: building the model, handing
it over and reading the answer (0.31 ms an LP against 0.22 ms of HiGHS, IKA-379). The port
solves the games with the same HiGHS (1.12.0, scipy's commit) and the same model, with the
options of IKA-470 (`rust/src/lp.rs`: no presolve, no scaling, tolerance 1e-8; scipy's
options only when those fail). Until IKA-470 the options were scipy's and an answer was
`solve`'s to the bit (all 59,050 LPs captured from open and hidden reads, records/IKA-381.md).
Now the game value is `solve`'s to 1e-7 and the strategies are an equilibrium of the game, but
on a degenerate game they may be another optimal vertex than `solve`'s
(records/IKA-470.md). Tests hold the port to the value and to "is an equilibrium"
(`tests/_equilibrium.py`), not to the bits.

Two requests, both one crossing for many games, on the `lp` command:

* `solve_many`: the Q's games of a menu's Q ranking (`qrank`). The port answers both LPs'
  values and the cleaned strategies; the rest of `Equilibrium` is built here as `solve` builds
  it (`equilibrium.assemble`), so the menus are the same menus.
* `solve_bayesian_many` (IKA-387): a side's Bayesian game (`search.belief_solve`) and the
  selection's deep reading (`selection_deep`), the answer built by
  `equilibrium.assemble_bayesian` as `solve_bayesian` builds it.

A game the port cannot solve is `solve`'s error (`EquilibriumError`, or `ValueError` for a
non-finite matrix), returned in its place. IKA-381 to IKA-389 also folded and solved a
depth-2 read's sub-games here (`solve_subs`) behind a switch (``POKEURAOU_LADDER_PORT_LP``);
IKA-464 removed them with the depth-2 read.

IKA-460 stage 4: the roads of a game's nodes are not switches any more. The hidden bench's node
(`beliefnode.belief_payoffs`), the menus' Q games (`qrank.prefetch`), a Bayesian game
(`search.belief_solve`), the replacement node, the menus (`rootmenus`), the move node over them
(`moveNode`), the turn's advance, the self-switch of a game the port holds, the game's
preparation and the game's steps are the port's, on the inference server's leaf. A request the
port answers with a ``fault`` is a mistake that stops the game with a sentence saying what
happened (`PortFault`); it is not solved again in Python.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import numpy as np

from .equilibrium import (
    BayesianEquilibrium,
    Equilibrium,
    EquilibriumError,
    assemble,
    assemble_bayesian,
    bayesian_inputs,
)

#: What went: crossings, games (`solve_many`), Bayesian games (`solve_bayesian_many`),
#: sub-games folded in the port and given whole, the LPs the port reports it solved (the
#: positive control), and, for each road a game's nodes take, how many times it ran.
COUNTS = {"crossings": 0, "games": 0, "bayes": 0, "folded": 0, "whole": 0, "lps": 0, "stacked": 0,
          "nodes": 0,
          # The whole-node requests of a node with nothing hidden.
          "exact": 0,
          # The replacement nodes the port solved in one request.
          "replaced": 0,
          # The move nodes whose menus the port built.
          "menus": 0,
          # The move nodes whose menus and node were one request.
          "decided": 0,
          # The turns whose weights, draw and drawn branch were one request.
          "advanced": 0,
          # The steps whose seen identities and completions the port made.
          "hidden": 0,
          # The games whose preparation the port made.
          "prepared": 0,
          # The self-switch nodes the port chose and played on.
          "selfswitched": 0,
          # The replacement phases the port chose and played in one request.
          "phased": 0,
          # The moves the port played inside a game request.
          "played": 0,
          # The steps the port handed back to the game's own loop (a state of the game the loop
          # answers itself: a bench the sheet cannot explain, an empty menu, a turn with no
          # outcome), not faults.
          "handoffs": 0}


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
    if "fault" in answer:
        raise PortFault("lp", str(answer["fault"]))
    return answer


class PortFault(RuntimeError):
    """The port could not answer a request and said so (``fault``). The game is not played on
    in Python: the sentence names the request and what the port said happened."""

    def __init__(self, where: str, why: str) -> None:
        super().__init__(f"the port could not answer `{where}`: {why}")
        self.where = where
        self.why = why


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


def bayesian_answer(
    matrices: Sequence[np.ndarray], weights: np.ndarray, one: dict[str, Any]
) -> BayesianEquilibrium | Exception:
    """`solve_bayesian_many`'s answer for one game from the port's ``one`` (IKA-455: the
    `beliefFold` answer carries the same fields), or the error `solve_bayesian` would raise."""
    try:
        mats, w = bayesian_inputs(list(matrices), weights)
    except ValueError as exc:
        return exc
    error = _error(one)
    if error is not None:
        return error
    return assemble_bayesian(mats, w, float(one["valueRow"]), float(one["valueCol"]),
                             _unb64(one["x"]), tuple(_unb64(y) for y in one["ys"]))


def counts() -> dict[str, int]:
    return dict(COUNTS)


__all__ = ["COUNTS", "PortFault", "bayesian_answer", "counts", "solve_bayesian_many",
           "solve_many"]
