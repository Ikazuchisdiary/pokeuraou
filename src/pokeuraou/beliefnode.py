"""One node solved over several possible benches, resolving each turn once.

The straightforward way to price a turn when the opponent's back two are unknown is to
build a payoff matrix per completion, and that is what :func:`pokeuraou.search.belief_search`
did first. It costs six matrices a turn against one -- 2.84 completions for side 0's
opponent and 3.16 for side 1's, one solve each side -- and measured on the machine that
came to 8.2x, which is more than the answer is worth.

Most of it is not the answer. Timed at the boundary, a matrix is 44.6 ms of resolving and
encoding against 5.2 ms of forward pass, and **an unrevealed benched Pokemon takes no part
in the turn unless it is switched in**. So for the cells where none of them reaches the
field, the turn resolves identically in every completion and the leaves differ only in who
is standing on the bench. Counted over opening nodes, that is **84.3% of cells**.

So the port resolves those cells once, on the true position, and builds each completion's
leaves by overwriting the bench rows of the encoding -- every per-Pokemon array is
``(B, 2, M, ...)``, so a slot is one assignment. The true position is only a carrier: the
Pokemon it holds in those slots take no part in the cells being shared, and their rows are
overwritten before anything is scored. Nothing about them reaches a payoff.

The remaining cells are resolved per completion:

- the opponent's action switches into a hidden slot;
- our action carries a move that forces their switch;
- the turn suspends for a replacement, whose choices include the hidden slots;
- (a cell the port refuses stops the node: there is no Python resolver behind it, IKA-209).

Both sides' games share the one resolution, because their hidden slots are disjoint and a
shared cell's resolution depends on neither. That is what takes 6.0x to a projected 2.2x.

A dirty cell is dirty for a side (IKA-269). One that reaches only side 0's bench is the true
position's turn for every completion of side 1's, read off their patched blocks like a
shared cell; and one that reaches a bench only by that side's own switches is the same turn
for every completion that brings the same Pokemon into the same slot.

The whole node is scored in one call of the inference server (IKA-105): every completion's
patched rows and the port's leaves of its dirty cells are blocks of one batch, and each
completion reads its values back by where its block starts.

Since IKA-460 stage 4 the node is the port's, and only the port's: `belief_payoffs` is one
`beliefNode` request (`rust/src/beliefplan.rs`) on the inference server's leaf
(`inference.RemoteValue`), which plans the node, fills and patches the blocks, scores them on
the server, folds them and solves the Bayesian games (`rust/src/beliefnode.rs`). The Python
road that did each of those in turn, and the matrix per completion that defined the answer, are
gone; a leaf that is not the inference server's is `UnservedLeaf`, and a request the port
answers with a fault is `portlp.PortFault`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import port, timing
from .actions import MoveAction, SideAction, SwitchAction
from .budget import Budget
from .position import Position
from .regulation import Regulation
from .rustnode import PortBranch

#: Moves that put a Pokemon on the field without its owner choosing it, so a cell carrying
#: one can reach a hidden slot whatever the opponent declared. Listed rather than derived
#: because a missing entry is silent: the cell would be shared, the substitution would be
#: wrong, and the payoff would still look like a payoff.
PHAZING_MOVES = frozenset({"dragontail", "circlethrow", "whirlwind", "roar"})


class UnservedLeaf(TypeError):
    """A node was asked of a leaf that is not the inference server's (`inference.RemoteValue`).

    The port plans, scores and folds the node on the server itself; there is no leaf held in
    this process to score it with (IKA-460 stage 4)."""


@dataclass
class BeliefNode:
    """Payoff matrices per side, over the completions that side cannot see."""

    #: `matrices[s][k]` is the payoff to side 0 when side `1 - s`'s bench is completion
    #: `k`. Always oriented to side 0, because that is what `batched_payoff` produces and
    #: what every recorded position means by "own"; a caller solving for side 1 transposes.
    matrices: dict[int, list[np.ndarray]]
    unmodelled: set[str] = field(default_factory=set)
    #: Cells resolved once and shared, and cell-resolutions that had to be repeated.
    shared: int = 0
    redone: int = 0
    #: The Bayesian games the port solved with the fold, by the side whose strategy each is
    #: (`BayesianEquilibrium`, or the error `solve_bayesian` would raise).
    solved: dict = field(default_factory=dict)


def reaches_bench(
    reg: Regulation,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    hidden: dict[int, tuple[int, ...]],
    position: Position | None = None,
) -> np.ndarray:
    """(rows, cols) mask: cells where a hidden Pokemon could be put on the field.

    Conservative in the direction that costs time rather than correctness -- a cell wrongly
    marked here is resolved per completion, which is what every cell used to do.

    With `position`, a held item's path too (IKA-191): a damaging move into a Red Card drags
    the *user's* side in, so a side with a holder on the field spoils every cell where the
    other side attacks, for the attacker's hidden slots. Eject Button and Emergency Exit
    put the holder's own bench in through a mid-turn replacement, and a cell that suspends
    is already dirty (`filled.folded`), so they need no row here.
    """
    rows, cols = len(ours), len(theirs)
    mask = np.zeros((rows, cols), dtype=bool)
    ours_hidden = set(hidden.get(0, ()))
    theirs_hidden = set(hidden.get(1, ()))
    row_phazes = [_phazes(reg, action) for action in ours]
    col_phazes = [_phazes(reg, action) for action in theirs]
    carded = [_red_card_on_field(position, side) for side in (0, 1)]
    row_carded = [carded[1] and _attacks(reg, action) for action in ours]
    col_carded = [carded[0] and _attacks(reg, action) for action in theirs]
    # `switch_indices` is 1-based, matching the choice string Showdown reads; `Pokemon.slot`
    # and therefore every hidden slot is 0-based. Comparing them directly marked the wrong
    # column dirty and shared the right one, which the equality test caught as nine cells
    # in one row -- the only reason it was not a silently wrong payoff.
    row_switches = [
        any(index - 1 in ours_hidden for index in action.switch_indices) for action in ours
    ]
    col_switches = [
        any(index - 1 in theirs_hidden for index in action.switch_indices)
        for action in theirs
    ]
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = (
                row_switches[i]
                or col_switches[j]
                # Either side's phazing move can drag the *other* side's bench in, and both
                # sides' benches carry hidden slots here, so one move is enough to spoil
                # the cell whichever side played it.
                or (row_phazes[i] and theirs_hidden)
                or (col_phazes[j] and ours_hidden)
                or (row_carded[i] and ours_hidden)
                or (col_carded[j] and theirs_hidden)
            )
    return mask


def turn_in_completion(
    reference: Position,
    turn: port.PortTurn,
    position: Position,
    side: int,
    slots: tuple[int, ...],
) -> port.PortTurn | None:
    """`turn`, resolved in `reference`, as the same turn in `position` (IKA-295), or None.

    `reference` and `position` are two completions of one node: the same position but for
    `side`'s hidden `slots`. A turn in which no hidden Pokemon is brought in -- the cells
    `reaches_bench` clears, and the caller asks only for those -- is the same turn in both,
    and each branch differs only in who stands in those slots. So each branch here is the
    reference's with `position`'s Pokemon in those slots, and the side's `mega_capable_slots`
    (numbers read off the party when the side was built, which the port hands back as it was
    given) theirs.

    Checked rather than assumed, branch by branch: a hidden slot the turn changed in any
    way -- brought in, dragged in, renumbered, touched -- is not the Pokemon the reference
    had there, and then this is None and the caller resolves the turn itself. So is a turn
    that paused (a replacement could choose a hidden slot) or kept no branch.
    """
    if turn.pauses or not turn.outcomes:
        return None
    before = reference.sides[side]
    after = position.sides[side]
    for slot in slots:
        if (
            slot >= len(before.pokemon)
            or slot >= len(after.pokemon)
            or before.pokemon[slot].slot != slot
            or after.pokemon[slot].slot != slot
        ):
            return None
    outcomes: list[PortBranch] = []
    for branch in turn.outcomes:
        mine = branch.position.sides[side]
        if mine.mega_capable_slots != before.mega_capable_slots:
            return None
        if any(mine.pokemon[slot] != before.pokemon[slot] for slot in slots):
            return None
        made = branch.position.copy()
        own = made.sides[side]
        for slot in slots:
            own.pokemon[slot] = after.pokemon[slot].copy()
        own.mega_capable_slots = list(after.mega_capable_slots)
        outcomes.append(PortBranch(branch.probability, made))
    return port.PortTurn(
        branches=list(turn.branches),
        suspended=[],
        exact=turn.exact,
        unmodelled=tuple(turn.unmodelled),
        outcomes=outcomes,
        pauses=[],
    )


def _phazes(reg: Regulation, action: SideAction) -> bool:
    # By the move's id: the choice string names a move by its index ("move 2 1"), so
    # reading PHAZING_MOVES off it never matched anything (IKA-191).
    return any(isinstance(s, MoveAction) and s.move_id in PHAZING_MOVES for s in action.slots)


def _attacks(reg: Regulation, action: SideAction) -> bool:
    """Whether the action uses a damaging move -- what a Red Card answers."""
    return any(
        isinstance(s, MoveAction)
        and (move := reg.moves.get(s.move_id)) is not None
        and move.category != "Status"
        for s in action.slots
    )


def _red_card_on_field(position: Position | None, side: int) -> bool:
    if position is None:
        return False
    return any(
        mon is not None and not mon.fainted and mon.item == "redcard"
        for mon in position.sides[side].active_pokemon()
    )


def served_leaf(evaluate: object, what: str = "the hidden bench's node"):  # noqa: ANN201
    """The inference server's leaf (`inference.RemoteValue`) that ``evaluate`` is, or
    `UnservedLeaf`: the port scores a node on the server and reads no leaf held here."""
    from .inference import RemoteValue

    owner = getattr(evaluate, "__self__", evaluate)
    if type(owner) is not RemoteValue:
        raise UnservedLeaf(
            f"{what} is solved by the port on the inference server's leaf "
            f"(inference.RemoteValue), not on {type(owner).__name__}; start the Rust inference "
            "server (`inference.start_server`, or `--inference ADDRESS`) and read through it"
        )
    return owner


# Its own Python -- the request's pieces and the books -- had no stage and sat in "the rest"
# (IKA-98); what it calls is still charged to the stages it calls.
@timing.timed("belief")
@timing.labelled("matrix")
def belief_payoffs(
    reg: Regulation,
    position: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    evaluate: Callable[[list[Position]], np.ndarray],
    *,
    budget: Budget,
    spreads: dict[int, list],
    solve: Sequence[int] = (),
) -> BeliefNode:
    """Every completion's matrix, resolving the turns that do not depend on the bench once.

    One `beliefNode` request to the port (`_node_in_port`), on the inference server's leaf.
    ``solve`` names the sides whose Bayesian game (`search.belief_solve`) the port solves in
    the same request: their answers come back in `BeliefNode.solved`.

    `spreads[s]` completes side `s`'s own unseen slots, so side `1 - s`'s game is solved
    over it. A side with nothing hidden contributes a single exact completion and costs
    nothing extra: with both exact the two completions are the same position, which the
    port resolves once. A cell the port refuses stops the node (IKA-209): there is no Python
    resolver behind it, and a request the port faults on is `portlp.PortFault`.
    """
    row, col = list(ours), list(theirs)
    hidden = {
        side: (items[0].slots if items and not items[0].exact else ())
        for side, items in spreads.items()
    }
    if timing.ON:
        # Per decision, by whether any bench here is hidden (IKA-98's per-decision table).
        timing.refine("hidden" if any(hidden.values()) else "exact")
        timing.count("completions", sum(len(items) for items in spreads.values()))
    from .encode import rules_of

    leaf = served_leaf(evaluate)
    rules = rules_of(leaf)
    if rules.patch_shares_side:
        raise UnservedLeaf(
            "the pre-IKA-119 patch (`patch_shares_side`) is not the port's: the node's leaves "
            "are patched there with the rebuilt side vector"
        )
    # The leaf's own encoder and rules, for the port's arrays: a match plays two arms in one
    # process, and each arm is scored the way its leaf says (IKA-141).
    encoder = getattr(leaf, "encoder", None) or _encoder_for(reg)
    done = _node_in_port(reg, leaf, position, row, col, budget, rules, hidden, spreads, solve, encoder)
    if done is _NOT_ON_THIS_BINARY:
        raise RuntimeError(
            "the port encoded with another `can_mega` rule than the leaf asked for; the "
            "executable predates the field -- rebuild it"
        )
    return done


def _fold_answer(
    answer: dict, jobs: list, m: int, n: int, asks: list
) -> tuple[list[np.ndarray], dict]:
    """Every completion's matrix and the solved games out of a `beliefNode`'s answer."""
    import base64

    from . import portlp

    flat = np.frombuffer(base64.b64decode(answer["matrices"]), dtype="<f8")
    payoffs = [
        flat[k * m * n : (k + 1) * m * n].reshape(m, n).astype(np.float64)
        for k in range(len(jobs))
    ]
    by_side: dict[int, list[np.ndarray]] = {}
    for job, payoff in zip(jobs, payoffs, strict=True):
        by_side.setdefault(1 - job.side, []).append(payoff)
    weights = dict(asks)
    solved: dict = {}
    for one in answer["answers"]:
        side = int(one["side"])
        games = [mat if side == 0 else -mat.T for mat in by_side[side]]
        solved[side] = portlp.bayesian_answer(games, weights[side], one)
    portlp.COUNTS["folded"] += len(jobs)
    portlp.COUNTS["bayes"] += len(answer["answers"])
    return payoffs, solved


def _stack_leaf(owner, rules):  # noqa: ANN001, ANN202 - the leaf, an inference.RemoteValue
    """The inference server's leaf when the port takes this node (and the game's steps), or None."""
    if rules.patch_shares_side:
        return None
    from .inference import RemoteValue

    return owner if type(owner) is RemoteValue else None


#: `_node_in_port`'s answer when the port encoded with another `can_mega` rule than asked.
_NOT_ON_THIS_BINARY = object()


def _action_features(reg: Regulation, actions: Sequence[SideAction]) -> list[dict]:
    """What the port reads off each action (`beliefplan.rs`): whether it forces a switch
    (`_phazes`), attacks (`_attacks`) and which party members its switches bring in."""
    from .regulation import to_id

    return [
        {
            "p": _phazes(reg, action),
            "a": _attacks(reg, action),
            "s": [[s.party_index, to_id(s.species)] for s in action.slots
                  if isinstance(s, SwitchAction)],
        }
        for action in actions
    ]


def _node_in_port(  # noqa: PLR0913 - the node's pieces
    reg: Regulation,
    leaf,  # noqa: ANN001 - inference.RemoteValue
    position: Position,
    row: list[SideAction],
    col: list[SideAction],
    budget: Budget,
    rules,  # noqa: ANN001 - EncodingRules
    hidden: dict[int, tuple[int, ...]],
    spreads: dict[int, list],
    solve: Sequence[int],
    encoder,  # noqa: ANN001 - Encoder
):  # noqa: ANN202 - a BeliefNode, or `_NOT_ON_THIS_BINARY`
    """The whole node in the port (`beliefNode`, IKA-455 stage 1.4): the reference fill, the plan,
    each completion's own fill, the stack, the server's values, the fold and the games, in one
    crossing. Python sends the positions, the actions and what it alone reads off them; the
    answer is the node."""
    from types import SimpleNamespace

    from . import portlp
    from .inference import request_failed

    request, jobs, asks = node_request(reg, leaf, position, row, col, budget, rules, hidden,
                                       spreads, solve, encoder)
    started = timing.clock()
    answer = port.ask(reg, lambda node: node.belief_node(request))
    wall = timing.clock() - started
    portlp.COUNTS["crossings"] += 1
    portlp.COUNTS["lps"] += int(answer.get("lps", 0))
    if "fault" in answer:
        raise portlp.PortFault(request["kind"], str(answer["fault"]))
    if "serverError" in answer:
        raise request_failed({"error": answer["serverError"], "oom": answer.get("oom"),
                              "capGb": answer.get("capGb")}, "inference failed")
    if "fills" not in answer:
        raise RuntimeError("the port did not answer `beliefNode`; the binary predates it -- rebuild it")
    book_node(answer, leaf, encoder, wall)
    if "refused" in answer:
        port.raise_refused([(int(i), int(j), str(why)) for i, j, why in answer["refused"]])
    if rules.mega_from_slots and answer["mega"] is not True:
        # A binary that predates the field encoded with the current rule.
        return _NOT_ON_THIS_BINARY
    if "served" not in answer:
        raise RuntimeError("the port did not answer `beliefNode`; the binary predates it -- rebuild it")
    book_served(answer, leaf, encoder, any(hidden.values()))
    unmodelled = set(answer["unmodelled"])
    ordered = [SimpleNamespace(side=job["side"]) for job in jobs]
    payoffs, solved = _fold_answer(answer, ordered, len(row), len(col), asks)
    matrices: dict[int, list[np.ndarray]] = {}
    for job, payoff in zip(ordered, payoffs, strict=True):
        matrices.setdefault(1 - job.side, []).append(payoff)
    return BeliefNode(matrices, unmodelled, int(answer["shared"]), int(answer["redone"]), solved)


def node_request(  # noqa: PLR0913 - the node's pieces
    reg: Regulation,
    leaf,  # noqa: ANN001 - inference.RemoteValue
    position: Position,
    row: list[SideAction],
    col: list[SideAction],
    budget: Budget,
    rules,  # noqa: ANN001 - EncodingRules
    hidden: dict[int, tuple[int, ...]],
    spreads: dict[int, list],
    solve: Sequence[int],
    encoder,  # noqa: ANN001 - Encoder
) -> tuple[dict, list[dict], list]:
    """The `beliefNode` request `_node_in_port` sends, its jobs and the games it asks to solve.
    The port's deepening (`portdeepen`, IKA-464) starts from the same request."""
    from . import portlp
    from .encode import HP_SCALE
    from .rustnode import _candidates, _position, dump_budget  # noqa: PLC2701

    ident = _id_of
    request: dict = {"kind": "beliefNode", "position": _position(position)}
    jobs = []
    for side, items in spreads.items():
        for item in items:
            request[f"p{len(jobs)}"] = _position(item.position)
            jobs.append({
                "side": side,
                "slots": list(hidden.get(side, ())),
                "exact": bool(item.exact),
                "root": item.position is position,
                "pk": [list(p) for p in item.pk] if item.pk is not None else [
                    [ident(mon.species), ident(mon.base_species), mon.slot]
                    for mon in item.position.sides[side].pokemon
                ],
            })
    request.update({
        "budget": dump_budget(budget),
        "hidden": [list(hidden.get(0, ())), list(hidden.get(1, ()))],
        "carded": [_red_card_on_field(position, 0), _red_card_on_field(position, 1)],
        "jobs": jobs,
    })
    request.update({
        "ours": _candidates(row),
        "theirs": _candidates(col),
        "rows": len(row),
        "cols": len(col),
        "rowFeat": _action_features(reg, row),
        "colFeat": _action_features(reg, col),
    })
    if rules.mega_from_slots:
        request["encoding"] = rules.to_request()
    asks = []
    for side in solve:
        items = spreads.get(1 - side)
        if items:
            asks.append((side, np.asarray([item.weight for item in items], dtype=np.float64)))
    request["solve"] = [{"side": side, "weights": portlp._b64(w)}  # noqa: SLF001
                        for side, w in asks]
    mon_k = {name: k for k, name in enumerate(encoder.mon_names)}
    side_k = {name: k for k, name in enumerate(encoder.side_names)}
    request["server"] = {
        "address": leaf.address,
        "model": leaf.model,
        "shm": leaf._block.name,  # noqa: SLF001 - the leaf's own block, idle while the port asks
        "bytes": int(leaf._block.size),  # noqa: SLF001
        "batch": int(leaf.batch_size),
    }
    request["netScoresEnds"] = bool(leaf.encoder.rules.net_scores_ends)
    request["patch"] = {
        "can_mega": mon_k["can_mega"],
        "maxhp_scaled": mon_k["maxhp_scaled"],
        "hp_fraction": mon_k["hp_fraction"],
        "fainted": mon_k["fainted"],
        "mega_used": side_k["mega_used"],
        "mega_available": side_k["mega_available"],
        "alive_fraction": side_k["alive_fraction"],
        "team_hp_fraction": side_k["team_hp_fraction"],
        "hp_scale": float(HP_SCALE),
    }
    return request, jobs, asks


def book_node(answer: dict, leaf, encoder, wall: float) -> None:  # noqa: ANN001
    """What `fill_encoded` booked for each fill of a `beliefNode` / `moveNode` answer, booked from
    what the port says it did (IKA-455 stage 1.4; stage 3.6b books a game's nodes with it too)."""
    for fill in answer["fills"]:
        with timing.purpose(fill["purpose"]):
            what = "rust can_mega=" + {True: "slots", False: "holder", None: "unechoed"}[fill["mega"]]
            if hasattr(encoder, "note"):
                encoder.note(what, int(fill["leaves"]))
            timing.add("rust.child.resolve", float(fill["resolveUs"]) / 1e6)
            timing.add("rust.child.encode", float(fill["encodeUs"]) / 1e6)
            timing.add("rust.child.bind", float(fill["bindUs"]) / 1e6)
            timing.count("leaves.offered", int(fill.get("offered", 0)))
            timing.count("leaves.stored", int(fill["leaves"]))
            if fill["purpose"] == "dirty":
                timing.count("leaves.node", int(fill["leaves"]))
            if timing.ON:
                used = timing.current_purpose()
                timing.add(
                    f"rust.child@{used}",
                    (float(fill["resolveUs"]) + float(fill["encodeUs"])) / 1e6,
                )
                timing.count("fills")
                timing.count("fill.cells", int(fill["cells"]))
                timing.count("fill.leaves", int(fill["leaves"]))
    timing.add("rust.fill@matrix", wall)


def book_served(answer: dict, leaf, encoder, hidden: bool) -> None:  # noqa: ANN001
    """The inference server's work in a `beliefNode` / `moveNode` answer, booked on the leaf."""
    from . import portlp, rustnode

    served = answer["served"]
    waited = float(served.get("waitUs", 0.0)) / 1e6
    rustnode.PORT_WAITED[0] -= waited
    requests, count = int(served.get("requests", 0)), int(served.get("rows", 0))
    leaf.waited += waited
    leaf.copied += float(served.get("copyUs", 0.0)) / 1e6
    leaf.calls += requests
    leaf.evaluated += count
    leaf.ended += int(served.get("ended", 0))
    timing.count("leaves", count)
    timing.count("forward.passes", requests)
    timing.count("serve.requests", requests)
    portlp.COUNTS["stacked"] += 1
    portlp.COUNTS["nodes"] += 1
    if not hidden:
        portlp.COUNTS["exact"] += 1
    if hasattr(encoder, "note"):
        for rows in answer["patched"]:
            encoder.note("patched side=rebuilt", int(rows))


@functools.cache
def _id_of(name: str) -> str:
    from .regulation import to_id

    return to_id(name)


#: The arrays one batch is made of, in `Encoded`'s order.
_ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def _stacked(
    parts: Sequence[tuple[int, Callable]], like  # noqa: ANN001 - Encoded
) -> tuple[object, list[int]]:
    """The parts one after another in one `Encoded`, and where each one starts.

    Written into arrays allocated once, one part made at a time -- a patched copy is the
    size of the whole reference, so holding every completion's before copying them would
    double the node's memory. Each array keeps the reference's dtype: the port's index
    arrays are int32 and the encoder's int64, and an index lookup is the same either way.
    """
    from .encode import Encoded

    total = sum(rows for rows, _make in parts)
    out = {
        name: np.empty((total, *getattr(like, name).shape[1:]), dtype=getattr(like, name).dtype)
        for name in _ARRAYS
    }
    # Which rows are a finished battle (IKA-253); a part that does not say is "not known".
    decided = np.full(total, np.nan, dtype=np.float64)
    starts: list[int] = []
    at = 0
    for rows, make in parts:
        starts.append(at)
        if rows:
            made = make()
            assert len(made) == rows, (len(made), rows)
            for name in _ARRAYS:
                out[name][at : at + rows] = getattr(made, name)
            if made.decided is not None:
                decided[at : at + rows] = made.decided
        at += rows
    return Encoded(**out, unknown_volatiles=dict(like.unknown_volatiles), decided=decided), starts


def _patched(
    reference,  # noqa: ANN001 - Encoded, imported lazily
    side: int,
    slots: tuple[int, ...],
    item,  # noqa: ANN001 - Completion
    reg: Regulation,
    position: Position,
    encoder=None,  # noqa: ANN001 - Encoder, imported lazily
    rules=None,  # noqa: ANN001 - EncodingRules
):  # noqa: ANN202
    """The reference leaves with `side`'s hidden slots replaced by this completion's.

    The rows come from encoding the completion's own root position once. A Pokemon that
    has never been active looks the same in every leaf it was not part of -- full HP, no
    status, no boosts, nothing switched in -- so the root's row is that leaf's row, and
    the test is what says so rather than this comment.

    Two things are not a row copy (IKA-119). A bench row's `can_mega` is the root's, but the
    side may have mega evolved during the turn, so it is ANDed with the leaf's `mega_used`.
    And the side vector reads the bench -- `mega_available` (a stone anywhere on the side),
    `alive_fraction`, `team_hp_fraction` (the bench's max HP is in the denominator) -- so it
    is rebuilt per leaf from the patched rows. Until then it was the true position's, and
    every completion was scored with the true bench's side features.

    `rules.patch_shares_side` puts that back -- the true position's side vector, and the
    root's `can_mega` copied as it is -- for a match that measures the fix (IKA-141). It is
    the pre-IKA-119 body, line for line, and nothing else should ask for it.

    The port makes the same blocks for a node (`rust/src/beliefplan.rs`); this is what a
    self-switch node in Python patches its options' leaves with (`selfplay._self_switch_encoded`).
    """
    from .encode import CURRENT_RULES, HP_SCALE, Encoded

    encoder = encoder or _encoder_for(reg)
    rules = rules or CURRENT_RULES
    # Only the hidden bench rows are read from it, and a bench row's bind columns are 0
    # (IKA-429), so the port is not asked for them.
    source = encoder.encode_positions([item.position], bind=False)
    if hasattr(encoder, "note"):
        encoder.note(
            "patched side=" + ("shared" if rules.patch_shares_side else "rebuilt"),
            len(reference.species),
        )
    if rules.patch_shares_side:
        old = Encoded(
            species=reference.species.copy(),
            ability=reference.ability.copy(),
            item=reference.item.copy(),
            moves=reference.moves.copy(),
            mon=reference.mon.copy(),
            mask=reference.mask.copy(),
            side=reference.side,
            field=reference.field,
            unknown_volatiles=dict(reference.unknown_volatiles),
            decided=reference.decided,
        )
        for slot in slots:
            old.species[:, side, slot] = source.species[0, side, slot]
            old.ability[:, side, slot] = source.ability[0, side, slot]
            old.item[:, side, slot] = source.item[0, side, slot]
            old.moves[:, side, slot] = source.moves[0, side, slot]
            old.mon[:, side, slot] = source.mon[0, side, slot]
            old.mask[:, side, slot] = source.mask[0, side, slot]
        return old
    out = Encoded(
        species=reference.species.copy(),
        ability=reference.ability.copy(),
        item=reference.item.copy(),
        moves=reference.moves.copy(),
        mon=reference.mon.copy(),
        mask=reference.mask.copy(),
        side=reference.side.copy(),
        # Nothing in the field vector reads a Pokemon.
        field=reference.field,
        unknown_volatiles=dict(reference.unknown_volatiles),
        # A cell reaching no hidden slot ends the same way whatever the bench is: a side
        # with an unseen Pokemon has one standing, so only the other side can have lost.
        decided=reference.decided,
    )
    for slot in slots:
        out.species[:, side, slot] = source.species[0, side, slot]
        out.ability[:, side, slot] = source.ability[0, side, slot]
        out.item[:, side, slot] = source.item[0, side, slot]
        out.moves[:, side, slot] = source.moves[0, side, slot]
        out.mon[:, side, slot] = source.mon[0, side, slot]
        out.mask[:, side, slot] = source.mask[0, side, slot]

    mon_k = {name: k for k, name in enumerate(encoder.mon_names)}
    side_k = {name: k for k, name in enumerate(encoder.side_names)}
    spent = out.side[:, side, side_k["mega_used"]] > 0  # (B,)
    rows = out.mon[:, side]  # (B, M, F), a view
    can = mon_k["can_mega"]
    for slot in slots:
        rows[spent, slot, can] = 0.0
    # With the mega unspent no Pokemon of the side is a mega, so "holds its stone" is
    # exactly `can_mega` on every row, patched or not.
    present = out.mask[:, side] > 0  # (B, M)
    holder = ((rows[:, :, can] > 0) & present).any(axis=1)
    out.side[:, side, side_k["mega_available"]] = (holder & ~spent).astype(np.float32)
    # Integers again, so the division is the encoder's own and the float32 is identical.
    maxhp = np.rint(rows[:, :, mon_k["maxhp_scaled"]].astype(np.float64) * HP_SCALE)
    hp = np.rint(rows[:, :, mon_k["hp_fraction"]].astype(np.float64) * maxhp)
    maxhp = np.where(present, maxhp, 0.0)
    hp = np.where(present, hp, 0.0)
    count = present.sum(axis=1)
    alive = (present & ~(rows[:, :, mon_k["fainted"]] > 0)).sum(axis=1)
    out.side[:, side, side_k["alive_fraction"]] = alive / np.maximum(count, 1)
    total = maxhp.sum(axis=1)
    out.side[:, side, side_k["team_hp_fraction"]] = hp.sum(axis=1) / np.where(
        total > 0, total, 1.0
    )
    return out


#: One encoder per regulation. Building one reads the whole vocabulary, and this is called
#: once per completion per node.
_ENCODERS: dict[int, object] = {}


def _encoder_for(reg: Regulation):  # noqa: ANN202
    from .encode import Encoder

    key = id(reg)
    made = _ENCODERS.get(key)
    if made is None:
        made = Encoder(reg)
        _ENCODERS[key] = made
    return made


__all__ = [
    "PHAZING_MOVES",
    "BeliefNode",
    "UnservedLeaf",
    "belief_payoffs",
    "reaches_bench",
    "served_leaf",
    "turn_in_completion",
]
