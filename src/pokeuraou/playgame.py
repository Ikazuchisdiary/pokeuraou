"""IKA-458 stage 3.6b: a game's turns played in one request to the port (`playGame`,
`rust/src/playgame.rs`).

`selfplay.play_game`'s loop made, for each step, the hidden bench's completions
(`hiddenstep.step`), then a replacement phase (`replacephase.play`) or a move node
(`selfplay._decide`: the menus and the node over them) and the turn's advance
(`selfplay._advance_turn`) -- each a crossing to the port, the next step's request built from the
last one's answer. For the game of the shape generation plays (one leaf on both seats, the
hidden bench's node, the Q's menus), the port makes all the steps it can in one request and
Python does what only it does: it writes the steps down (`Decision`, the record's notes) and
books what the port says it did (the inference server's work on the leaf, the counts the report
reads) exactly as the loop that crossed for every step booked it.

The port stops at a step that is a state of the game the loop answers itself (a hidden bench its
sheet cannot explain, an empty menu, a draw numpy would refuse, a turn it has no outcome of)
before drawing or recording anything, and `play` hands the game back at that step: `play_game`
goes on from the same position and generator, in its own loop (`portlp.COUNTS["handoffs"]`). A
request the port answers with a fault is `portlp.PortFault` (IKA-460).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from . import port, portlp, rank_scores, rustnode, timing
from .position import Position
from .regulation import Regulation


class Setup:
    """What a game's steps share and `playGame` needs of it: the leaf, the menus' plan (which also
    books the menus' half of every node), the request's static part and the leaves the server work
    is booked on."""

    __slots__ = ("context", "encoder", "leaf", "node", "owners", "plan", "rules")

    def __init__(self) -> None:
        self.context: dict[str, Any] = {}
        self.owners: list[Any] = []
        self.node: dict[str, Any] = {}
        self.leaf: Any = None
        self.encoder: Any = None
        self.plan: Any = None
        self.rules: Any = None


class Played:
    """Where a run of steps left the game."""

    __slots__ = ("position", "seen", "steps", "stop", "turns")

    def __init__(self, answer: dict[str, Any]) -> None:
        self.position = Position.from_json(answer["position"])
        self.seen = [frozenset(ids) for ids in answer["seen"]]
        self.steps = int(answer["steps"])
        self.stop: str = answer["stop"]
        self.turns: int | None = answer["turns"]


def prepare(  # noqa: PLR0913 - the loop's settings
    reg: Regulation,
    pos: Position,
    leaves: tuple[Any, Any],
    limits: tuple[int, int],
    fill: str,
    budget: Any,  # noqa: ANN401 - budget.Budget
    game: int | None,
    seen: list[frozenset[str]],
    leads: list[frozenset[str] | None],
) -> Setup | None:
    """The game's shared part, or None where the port does not play its steps (any road
    `_decide`, `fused_node` and `_self_switch_request` would not take either)."""
    from . import beliefnode, rootmenus
    from .encode import HP_SCALE, rules_of
    from .selfplay import _HiddenBench, _self_switch_request

    if game is None or timing.DUPES or rank_scores._SINK.get() is not None:  # noqa: SLF001
        return None
    if leaves[0] is None or leaves[0] is not leaves[1]:
        return None
    evaluate = leaves[0]
    owner = getattr(evaluate, "__self__", evaluate)
    if getattr(owner, "from_encoded", None) is None:
        return None
    encoder = getattr(owner, "encoder", None) or beliefnode._encoder_for(reg)  # noqa: SLF001
    rules = rules_of(evaluate)
    leaf = beliefnode._stack_leaf(owner, rules)  # noqa: SLF001
    if leaf is None:
        return None
    ready = _self_switch_request(
        leaves, _HiddenBench((), list(seen), None, list(leads), game)
    )
    if ready is None:
        return None
    planned = rootmenus.plan(reg, pos, [pos, pos], limits, fill, server_key="qServer")
    if planned is None:
        return None
    setup = Setup()
    setup.context, setup.owners = ready
    setup.plan = planned
    setup.leaf, setup.encoder, setup.rules = leaf, encoder, rules
    mon_k = {name: k for k, name in enumerate(encoder.mon_names)}
    side_k = {name: k for k, name in enumerate(encoder.side_names)}
    node = {key: planned.request[key] for key in ("widths", "properties", "megaFromSlots", "qServer", "blas")}
    node["budget"] = rustnode.dump_budget(budget)
    node["server"] = {
        "address": leaf.address,
        "model": leaf.model,
        "shm": leaf._block.name,  # noqa: SLF001 - the leaf's own block, idle while the port asks
        "bytes": int(leaf._block.size),  # noqa: SLF001
        "batch": int(leaf.batch_size),
    }
    node["netScoresEnds"] = bool(leaf.encoder.rules.net_scores_ends)
    node["patch"] = {
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
    if rules.mega_from_slots:
        node["encoding"] = rules.to_request()
    setup.node = node
    return setup


def play(  # noqa: PLR0913 - the loop's state
    reg: Regulation,
    rng: Any,  # noqa: ANN401 - numpy.random.Generator
    pos: Position,
    record: Any,  # noqa: ANN401 - selfplay.GameRecord
    setup: Setup,
    steps: int,
    seen: list[frozenset[str]],
    leads: list[frozenset[str] | None],
    hidden: Any,  # noqa: ANN401 - hiddenstep.Game
) -> Played | None:
    """Plays the steps the port does, up to ``steps``, and writes them into ``record``; None where
    the port answered a fault before any step (nothing was written, the generator untouched)."""
    from .budget import Budget

    if type(rng.bit_generator).__name__ != "PCG64":
        return None
    before = rng.bit_generator.state
    context = dict(setup.context)
    context["hidden"] = {
        "game": hidden.number,
        "seen": [sorted(s) for s in seen],
        "leads": [None if lead is None else sorted(lead) for lead in leads],
    }
    request = {
        "kind": "playGame",
        "position": rustnode._position(pos),  # noqa: SLF001
        "rng": {
            "state": format(before["state"]["state"], "x"),
            "inc": format(before["state"]["inc"], "x"),
            "hasUint32": bool(before["has_uint32"]),
            "uinteger": int(before["uinteger"]),
        },
        "context": context,
        "node": setup.node,
        "exactBudget": rustnode.dump_budget(Budget.exact()),
        "steps": steps,
    }
    answer = port.ask(reg, lambda node: node.play_game(request, hidden))
    portlp.COUNTS["crossings"] += 1
    if "fault" in answer:
        raise portlp.PortFault("playGame", str(answer["fault"]))
    if "serverError" in answer:
        from .inference import request_failed

        raise request_failed({"error": answer["serverError"], "oom": answer.get("oom"),
                              "capGb": answer.get("capGb")}, "inference failed")
    if "rng" not in answer:
        raise RuntimeError("the port did not answer `playGame`; the binary predates it -- rebuild it")
    for event in answer["events"]:
        _write(record, event, setup)
    after = dict(before)
    after["state"] = {"state": int(answer["rng"]["state"], 16), "inc": int(answer["rng"]["inc"], 16)}
    after["has_uint32"] = int(bool(answer["rng"]["hasUint32"]))
    after["uinteger"] = int(answer["rng"]["uinteger"])
    rng.bit_generator.state = after
    if answer["stop"] == "handoff":
        # A state of the game the port leaves to the loop (see the module's words), not a fault.
        portlp.COUNTS["handoffs"] += 1
    played = Played(answer)
    if played.turns is not None:
        record.turns = played.turns
    if played.stop == "unresolved":
        record.end_reason = "unresolved"
    return played


def _write(record: Any, event: dict[str, Any], setup: Setup) -> None:  # noqa: ANN401, PLR0915
    """One step as `play_game` wrote it: the notes, the node's and the server's books, the
    `Decision`, then the turn's own notes and the self-switch nodes."""
    from . import beliefnode, replacephase
    from .selfplay import WrittenDecision, _apply_chain

    for note in event["stepNotes"]:
        if note not in record.unmodelled:
            record.unmodelled.append(note)
    portlp.COUNTS["hidden"] += 1
    timing.count("hidden.completions", int(event["completions"]))
    answer = event["answer"]
    if event["kind"] == "replacement":
        replacephase._book(setup.owners, answer)  # noqa: SLF001
        portlp.COUNTS["lps"] += int(answer.get("lps", 0))
        record.decisions.append(WrittenDecision(event["decision"]))
        timing.decided("replacement")
        record.unmodelled.extend(answer["notes"])
        portlp.COUNTS["phased"] += 1
        return

    hidden = any(len(slots) for slots in event["hiddenSlots"])
    if timing.ON:
        timing.refine("hidden" if hidden else "exact")
        timing.count("completions", int(event["completions"]))
    setup.plan.finish(answer["rootMenus"])
    beliefnode.book_node(answer, setup.leaf, setup.encoder, float(event["seconds"]))
    beliefnode.book_served(answer, setup.leaf, setup.encoder, hidden)
    portlp.COUNTS["lps"] += int(answer.get("lps", 0))
    portlp.COUNTS["decided"] += 1
    portlp.COUNTS["played"] = portlp.COUNTS.get("played", 0) + 1
    record.unmodelled.extend(event["nodeNotes"])
    seconds = float(event["seconds"])
    record.search_seconds[0] += seconds / 2
    record.search_seconds[1] += seconds / 2
    record.decisions.append(WrittenDecision(event["decision"]))
    timing.decided("move")
    record.unmodelled.extend(event["advanceNotes"])
    portlp.COUNTS["advanced"] += 1
    chain = event.get("chain")
    if chain is not None:
        _apply_chain(record, SimpleNamespace(chain=chain, position=None), setup.owners)
