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
COUNTS = {"crossings": 0, "games": 0, "bayes": 0, "folded": 0, "whole": 0, "lps": 0, "stacked": 0,
          "nodes": 0, "faults": 0,
          # IKA-457 stage 2.1: the whole-node requests of a node with nothing hidden.
          "exact": 0,
          # IKA-457 stage 2.2: the replacement nodes the port solved in one request.
          "replaced": 0,
          # IKA-457 stage 2.3: the move nodes whose menus the port built.
          "menus": 0,
          # IKA-457 stage 2.4: the move nodes whose menus and node were one request.
          "decided": 0,
          # IKA-458 stage 3.1: the turns whose weights, draw and drawn branch were one request.
          "advanced": 0,
          # IKA-458 stage 3.4: the steps whose seen identities and completions the port made.
          "hidden": 0,
          # IKA-458 stage 3.5: the games whose preparation the port made.
          "prepared": 0,
          # IKA-458 stage 3.2: the self-switch nodes the port chose and played on.
          "selfswitched": 0,
          # IKA-458 stage 3.6a: the replacement phases the port chose and played in one request.
          "phased": 0,
          # IKA-458 stage 3.6b: the moves the port played inside a game request.
          "played": 0}


GEN_ENV = "POKEURAOU_GEN_PORT_LP"
#: IKA-455: whether generation's own LPs go to the port, off by default: the Q's games of one
#: `qrank.prefetch` in one crossing, and the Bayesian games of both sides of one
#: `search.belief_solve` in one. Apart from `ON`, which is the ladder's and also asks for span
#: blocks nobody here reads. A list so a test can turn it on in one process.
GEN = [os.environ.get(GEN_ENV, "0").strip() == "1"]


def set_gen(on: bool) -> None:
    """Turn generation's LPs on or off in this process (IKA-455)."""
    GEN[0] = bool(on)


def set_on(on: bool) -> None:
    """Turn the port's LPs on or off in this process (and the span blocks they need)."""
    ON[0] = bool(on)
    rustnode._SPAN_BLOCKS[0] = bool(on)


set_on(ON[0])

FOLD_ENV = "POKEURAOU_GEN_PORT_FOLD"
#: IKA-455 stage 1.2: whether a hidden bench's node is folded in the port too -- each
#: completion's matrix from the fills' span blocks and the leaves' values -- and its Bayesian
#: games solved in the same crossing (`beliefnode.belief_payoffs`, `rust/src/beliefnode.rs`).
#: Off by default. On includes `GEN` (the Q's games, and the games of a node not folded here),
#: and keeps the fills' span blocks, which the request is made of.
FOLD = [os.environ.get(FOLD_ENV, "0").strip() == "1"]


def set_fold(on: bool) -> None:
    """Turn stage 1.2 on or off in this process; on turns `GEN` and the span blocks on."""
    FOLD[0] = bool(on)
    if on:
        set_gen(True)
        rustnode._SPAN_BLOCKS[0] = True


set_fold(FOLD[0])

STACK_ENV = "POKEURAOU_GEN_PORT_STACK"
#: IKA-455 stage 1.3: whether a hidden bench's node is also stacked and scored in the port --
#: the port keeps each fill's arrays, patches the reference's bench rows per completion,
#: lays the batch into the leaf's shared block and asks the inference server itself, then folds
#: (`beliefnode.belief_payoffs`, `rust/src/beliefnode.rs` `beliefServed`). Off by default. On
#: includes `FOLD` (and so `GEN`). Only where the leaf is an inference server's.
STACK = [os.environ.get(STACK_ENV, "0").strip() == "1"]


def set_stack(on: bool) -> None:
    """Turn stage 1.3 on or off in this process; on turns `FOLD` on too."""
    STACK[0] = bool(on)
    if on:
        set_fold(True)


set_stack(STACK[0])

NODE_ENV = "POKEURAOU_GEN_PORT_NODE"
#: IKA-455 stage 1.4: whether a hidden bench's whole node is one request to the port -- the
#: reference fill, the plan (which cells reach a bench, which completion borrows from which),
#: each completion's own fill, the stack, the server's values, the fold and the games
#: (`beliefnode.belief_payoffs`, `rust/src/beliefplan.rs` `beliefNode`). Off by default. On
#: includes `STACK` (and so `FOLD` and `GEN`).
NODE = [os.environ.get(NODE_ENV, "0").strip() == "1"]


def set_node(on: bool) -> None:
    """Turn stage 1.4 on or off in this process; on turns `STACK` on too."""
    NODE[0] = bool(on)
    if on:
        set_stack(True)


set_node(NODE[0])

EXACT_ENV = "POKEURAOU_GEN_PORT_EXACT"
#: IKA-457 stage 2.1: whether a node with nothing hidden (`move.exact`) is the whole-node
#: request's too -- the one position's fill, the stack, the server's values, the fold and the
#: games -- where `_per_completion` filled in the port, scored in Python and solved the games
#: in another crossing. Off by default, and off in `default_on_for_generation`. On includes
#: `NODE` (and so `STACK`, `FOLD` and `GEN`).
EXACT = [os.environ.get(EXACT_ENV, "0").strip() == "1"]


def set_exact(on: bool) -> None:
    """Turn stage 2.1 on or off in this process; on turns `NODE` on too."""
    EXACT[0] = bool(on)
    if on:
        set_node(True)


set_exact(EXACT[0])

REPLACE_ENV = "POKEURAOU_GEN_PORT_REPLACE"
#: IKA-457 stage 2.2: whether a replacement node (a forced switch, `selfplay._do_replacement_node`)
#: is one request to the port -- each completion's fills, the server's values, the fold of a phase
#: that draws and the Bayesian games (`replacenode.py`, `rust/src/replacenode.rs`). Off by
#: default, and off in `default_on_for_generation`. On includes `EXACT` (and so `NODE` and below).
REPLACE = [os.environ.get(REPLACE_ENV, "0").strip() == "1"]


def set_replace(on: bool) -> None:
    """Turn stage 2.2 on or off in this process; on turns `EXACT` on too."""
    REPLACE[0] = bool(on)
    if on:
        set_exact(True)


set_replace(REPLACE[0])

MENUS_ENV = "POKEURAOU_GEN_PORT_MENUS"
#: IKA-457 stage 2.3: whether a move node's two menus by the Q (`selfplay._menus` with a
#: `q-nocover` fill) are one request to the port -- the pools, the Q's request and games, the
#: ranking and the cut at each width (`rootmenus.py`, `rust/src/qmenus.rs` `rootMenus`). Off by
#: default, and off in `default_on_for_generation`. On includes `REPLACE` (and so `EXACT` and
#: the whole-node road).
MENUS = [os.environ.get(MENUS_ENV, "0").strip() == "1"]


def set_menus(on: bool) -> None:
    """Turn stage 2.3 on or off in this process; on turns `REPLACE` on too."""
    MENUS[0] = bool(on)
    if on:
        set_replace(True)


set_menus(MENUS[0])

DECIDE_ENV = "POKEURAOU_GEN_PORT_DECIDE"
#: IKA-457 stage 2.4: whether a move node's menus and the node over them are one request to the
#: port (`moveNode`: `rootMenus` and `beliefNode` in one, `beliefnode.fused_node`). Off by default,
#: and off in `default_on_for_generation`. On includes `MENUS` (and so the stages below it).
DECIDE = [os.environ.get(DECIDE_ENV, "0").strip() == "1"]


def set_decide(on: bool) -> None:
    """Turn stage 2.4 on or off in this process; on turns `MENUS` on too."""
    DECIDE[0] = bool(on)
    if on:
        set_menus(True)


set_decide(DECIDE[0])

ADVANCE_ENV = "POKEURAOU_GEN_PORT_ADVANCE"
#: IKA-458 stage 3.1: whether a turn's weights, the draw among them (numpy's PCG64 state goes to
#: the port and comes back) and the drawn branch's position are one request
#: (`selfplay._advance_turn`, `rust/src/advance.rs` `advanceTurn`), the replacements the next
#: position owes riding along. Off by default, and off in `default_on_for_generation`. On includes
#: `DECIDE` (and so the stages below it).
ADVANCE = [os.environ.get(ADVANCE_ENV, "0").strip() == "1"]


def set_advance(on: bool) -> None:
    """Turn stage 3.1 on or off in this process; on turns `DECIDE` on too."""
    ADVANCE[0] = bool(on)
    if on:
        set_decide(True)


set_advance(ADVANCE[0])

HIDDEN_ENV = "POKEURAOU_GEN_PORT_HIDDEN"
#: IKA-458 stage 3.4: whether what each side has shown (`hidden.seen_identities`) and the
#: positions its unseen slots could hold, priced by the selection prior (`hidden.completions`,
#: `selfplay._bench_weights`), are one request to the port (`hiddenstep.step`,
#: `rust/src/hiddenbench.rs` `hiddenStep`), the completions kept there by number. Off by
#: default, and off in `default_on_for_generation`. On includes `ADVANCE` (and so the stages
#: below it).
HIDDEN = [os.environ.get(HIDDEN_ENV, "0").strip() == "1"]


def set_hidden(on: bool) -> None:
    """Turn stage 3.4 on or off in this process; on turns `ADVANCE` on too."""
    HIDDEN[0] = bool(on)
    if on:
        set_advance(True)


set_hidden(HIDDEN[0])

PREP_ENV = "POKEURAOU_GEN_PORT_PREP"
#: IKA-458 stage 3.5: whether a game's preparation -- the pair and its seats, the selection solve's
#: read, both sides' four, the bench priors and the turn-1 position with the leads' switch-ins -- is
#: one request to the port (`prepgame.prepare`, `rust/src/prep.rs` `prepGame`), the game's sheets and
#: priors defined there for `hiddenStep`. Off by default, and off in `default_on_for_generation`. On
#: includes `HIDDEN` (and so the stages below it).
PREP = [os.environ.get(PREP_ENV, "0").strip() == "1"]


def set_prep(on: bool) -> None:
    """Turn stage 3.5 on or off in this process; on turns `HIDDEN` on too."""
    PREP[0] = bool(on)
    if on:
        set_hidden(True)


set_prep(PREP[0])

SELFSWITCH_ENV = "POKEURAOU_GEN_PORT_SELFSWITCH"
#: IKA-458 stage 3.2: whether a mid-turn replacement (a self-switch) is chosen and played on in the
#: port, inside the turn's `advanceTurn` (`selfplay._advance_turn`, `rust/src/selfswitch.rs`): the
#: options, the other side's completions, the leaves' scores on the inference server, the fold and
#: the draw of what follows, to the end of the turn. Off by default, and off in
#: `default_on_for_generation`. On includes `PREP` (and so the stages below it).
SELFSWITCH = [os.environ.get(SELFSWITCH_ENV, "0").strip() == "1"]


def set_selfswitch(on: bool) -> None:
    """Turn stage 3.2 on or off in this process; on turns `PREP` on too."""
    SELFSWITCH[0] = bool(on)
    if on:
        set_prep(True)


set_selfswitch(SELFSWITCH[0])

PLAY_ENV = "POKEURAOU_GEN_PORT_PLAY"
#: IKA-458 stage 3.6: whether the game's own steps are the port's -- stage 3.6a: a replacement phase
#: (the options, the node, both draws and the phase) is one request (`replacephase`,
#: `rust/src/replacephase.rs`). Off by default, and off in `default_on_for_generation`. On includes
#: `SELFSWITCH` (and so the stages below it).
PLAY = [os.environ.get(PLAY_ENV, "0").strip() == "1"]


def set_play(on: bool) -> None:
    """Turn stage 3.6 on or off in this process; on turns `SELFSWITCH` on too."""
    PLAY[0] = bool(on)
    if on:
        set_selfswitch(True)


set_play(PLAY[0])

GAME_ENV = "POKEURAOU_GEN_PORT_GAME"
#: IKA-458 stage 3.6b: with `PLAY` on, whether a game's steps are played by the port in one request
#: (`playgame`) -- on unless this is ``0``, which leaves stage 3.6a's road (a request for each
#: step) to measure the two against each other.
GAME = [os.environ.get(GAME_ENV, "1").strip() != "0"]


def default_on_for_generation() -> None:
    """Generation's, and its board's, default: stage 1.4's whole-node road, unless
    `POKEURAOU_GEN_PORT_NODE` is set (``0`` turns it off, ``1`` is the same as the default).

    Called by the entry points of generation (`tools/selfplay.py --pool`, so also
    `generate_queue.py`) and of the board played with its arguments (`tools/pool_match.py`, so
    also `match_queue.py --pool`), and by nothing else: the library, the tests, human play and
    the reading are off unless their environment names the flag (IKA-455). The answers are the
    Python road's to the bit, so no agent is another because of it (`agent_drift.py` reads
    `play_game`'s arguments, and this is not one).

    Since IKA-457 the default is stage 2.4's road (`DECIDE`, and so every stage below it: the
    node with nothing hidden, the replacement node, the menus and the menus with the node in one
    request), each the Python road's to the bit at 21 workers. ``POKEURAOU_GEN_PORT_DECIDE=0``
    keeps the default at stage 1.4's road; ``POKEURAOU_GEN_PORT_NODE=0`` turns all of it off.

    Since IKA-458 the default is stage 3.6's road (`PLAY`, and so every stage below it: the
    turn's advance, the hidden bench's completions, the game's preparation, the self-switch,
    the replacement phase and the game's turns in one request), the stage-2 road's games to
    the bit on 4,000 board games. ``POKEURAOU_GEN_PORT_PLAY=0`` keeps the default at stage 2.4's
    road.
    """
    if os.environ.get(NODE_ENV, "").strip():
        return
    set_node(True)
    if os.environ.get(DECIDE_ENV, "").strip():
        return
    set_decide(True)
    if not os.environ.get(PLAY_ENV, "").strip():
        set_play(True)


def default_on_for_play() -> None:
    """The default of human play and the analysis (IKA-460): the roads of the hidden-bench
    node and the menus (stage 2.4's: `DECIDE`, and so everything below it), the same
    `POKEURAOU_GEN_PORT_NODE=0` (all off) and `POKEURAOU_GEN_PORT_DECIDE=0` (stage 1.4 only)
    turn off.

    The game's own loop (stage 3) is generation's: a person's game is played in
    `humanplay`, so those flags stay off here. The roads engage where the leaf is on an
    inference server (`inference.RemoteValue`); with a leaf loaded in the process they are
    not taken and the answers are Python's.
    """
    if os.environ.get(NODE_ENV, "").strip():
        return
    set_node(True)
    if os.environ.get(DECIDE_ENV, "").strip():
        return
    set_decide(True)


def echo() -> str:
    """The worker's own line saying which roads ran and how often (a worker's end)."""
    word = {True: "on", False: "off"}
    return (
        f"port roads: lp {word[GEN[0]]}, fold {word[FOLD[0]]}, stack {word[STACK[0]]}, "
        f"node {word[NODE[0]]} (POKEURAOU_GEN_PORT_NODE={os.environ.get(NODE_ENV, '(unset)')}), "
        f"exact {word[EXACT[0]]} ({EXACT_ENV}={os.environ.get(EXACT_ENV, '(unset)')}), "
        f"replace {word[REPLACE[0]]} ({REPLACE_ENV}={os.environ.get(REPLACE_ENV, '(unset)')}), "
        f"menus {word[MENUS[0]]} ({MENUS_ENV}={os.environ.get(MENUS_ENV, '(unset)')}), "
        f"decide {word[DECIDE[0]]} ({DECIDE_ENV}={os.environ.get(DECIDE_ENV, '(unset)')}), "
        f"advance {word[ADVANCE[0]]} ({ADVANCE_ENV}={os.environ.get(ADVANCE_ENV, '(unset)')}), "
        f"hidden {word[HIDDEN[0]]} ({HIDDEN_ENV}={os.environ.get(HIDDEN_ENV, '(unset)')}), "
        f"prep {word[PREP[0]]} ({PREP_ENV}={os.environ.get(PREP_ENV, '(unset)')}), "
        f"selfswitch {word[SELFSWITCH[0]]} ({SELFSWITCH_ENV}={os.environ.get(SELFSWITCH_ENV, '(unset)')}), "
        f"play {word[PLAY[0]]} ({PLAY_ENV}={os.environ.get(PLAY_ENV, '(unset)')}); "
        f"faults solved in Python: {COUNTS['faults']}; {counts()}"
    )


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
        raise PortFault(str(answer["fault"]))
    return answer


class PortFault(RuntimeError):
    """The port could not answer a request of this module's and said so (`fault`), as one
    does not restart the process for: the caller solves the same thing in Python (IKA-455)."""


#: How many faults' words are printed (the first few; the count is in `COUNTS["faults"]`).
FAULTS_SHOWN = 5


def faulted(where: str, why: object) -> None:
    """Count a fault and print its words if it is one of the first few; the caller then goes
    on in Python."""
    import sys

    COUNTS["faults"] += 1
    if COUNTS["faults"] <= FAULTS_SHOWN:
        print(f"[portlp] {where}: the port answered a fault ({why}); solved in Python",
              file=sys.stderr)


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


__all__ = ["ADVANCE", "ADVANCE_ENV", "COUNTS", "DECIDE", "DECIDE_ENV", "ENV", "EXACT",
           "PLAY", "PLAY_ENV", "set_play", "PREP", "PREP_ENV", "set_prep",
           "SELFSWITCH", "SELFSWITCH_ENV", "set_selfswitch",
           "EXACT_ENV", "FOLD", "FOLD_ENV",
           "GEN", "GEN_ENV", "NODE", "NODE_ENV", "MENUS", "MENUS_ENV", "ON", "REPLACE",
           "REPLACE_ENV", "PortFault", "STACK", "STACK_ENV", "bayesian_answer", "counts",
           "default_on_for_generation", "echo", "faulted", "set_advance", "set_decide",
           "set_exact", "set_fold",
           "set_gen", "set_menus", "set_node", "set_on", "set_replace", "set_stack",
           "solve_bayesian_many", "solve_bayesian_one", "solve_many", "solve_one", "solve_subs"]
