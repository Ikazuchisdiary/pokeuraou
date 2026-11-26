"""IKA-458 stage 3.6b: a game's turns played in one request to the port (`playGame`,
`rust/src/playgame.rs`) -- off by default (`portlp.PLAY`).

`selfplay.play_game`'s loop made, for each step, the hidden bench's completions
(`hiddenstep.step`), then a replacement phase (`replacephase.play`) or a move node
(`selfplay._decide`: the menus and the node over them) and the turn's advance
(`selfplay._advance_turn`) -- each a crossing to the port, the next step's request built from the
last one's answer. With `portlp.PLAY` on and the game of the shape generation plays, the port makes
all the steps it can in one request and Python does what only it does: it writes the steps down
(`Decision`, the record's notes) and books what the port says it did (the inference server's work
on the leaf, the counts the report reads) exactly as the road that crossed for every step booked
it.

The port stops at a step it does not do the way Python does (a hidden bench its sheet cannot
explain, an empty menu, a draw numpy would refuse, a game it cannot resolve) before drawing or
recording anything, and `play` hands the game back at that step: `play_game` goes on from the same
position and generator, the old road.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from . import luck, port, portlp, rank_scores, rustnode, timing
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

    if not (portlp.PLAY[0] and portlp.GAME[0] and portlp.DECIDE[0] and portlp.NODE[0] and portlp.EXACT[0]):
        return None
    if game is None or luck.active() or timing.DUPES or rank_scores._SINK.get() is not None:  # noqa: SLF001
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
        leaves, _HiddenBench((), list(seen), None, list(leads), (False, False), game)
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
        "merge": bool(leaf.merge),
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
    game: int,
) -> Played | None:
    """Plays the steps the port does, up to ``steps``, and writes them into ``record``; None where
    the port answered a fault before any step (nothing was written, the generator untouched)."""
    from .budget import Budget

    if type(rng.bit_generator).__name__ != "PCG64":
        return None
    before = rng.bit_generator.state
    context = dict(setup.context)
    context["hidden"] = {
        "game": game,
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
    answer = port.ask(reg, lambda node: node.play_game(request))
    portlp.COUNTS["crossings"] += 1
    if "fault" in answer:
        portlp.faulted("playGame", str(answer["fault"]))
        return None
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
        portlp.faulted("playGame", f"a step is the caller's: {answer['why']}")
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
    from .selfplay import Decision, _apply_chain

    for note in event["stepNotes"]:
        if note not in record.unmodelled:
            record.unmodelled.append(note)
    portlp.COUNTS["hidden"] += 1
    timing.count("hidden.completions", int(event["completions"]))
    answer = event["answer"]
    if event["kind"] == "replacement":
        replacephase._book(setup.owners, answer)  # noqa: SLF001
        portlp.COUNTS["lps"] += int(answer.get("lps", 0))
        options = [list(side) for side in answer["options"]]
        policy = [[float(x) for x in side] for side in answer["policy"]]
        chosen = [int(i) for i in answer["chosen"]]
        record.decisions.append(
            Decision(
                turn=int(event["turn"]),
                kind="replacement",
                position=event["position"],
                own_actions=options[0],
                own_policy=policy[0],
                foe_actions=options[1],
                foe_policy=policy[1],
                search_value=float(answer["value"]),
                own_chosen=options[0][chosen[0]],
                foe_chosen=options[1][chosen[1]],
                shown=[list(ids) for ids in event["seen"]],
                foe_search_value=float(answer["foeValue"]),
            )
        )
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
    views = [None if got is None else [got[0], list(got[1])] for got in event["rankViews"]]
    chosen = [int(i) for i in event["chosen"]]
    record.decisions.append(
        Decision(
            turn=int(event["turn"]),
            kind="move",
            position=event["position"],
            own_actions=list(event["ours"]),
            own_policy=[float(x) for x in event["ownPolicy"]],
            foe_actions=list(event["theirs"]),
            foe_policy=[float(x) for x in event["foePolicy"]],
            search_value=float(event["value"]),
            own_chosen=event["ours"][chosen[0]],
            foe_chosen=event["theirs"][chosen[1]],
            rank_views=None if views == [None, None] else views,
            shown=[list(ids) for ids in event["seen"]],
            foe_search_value=float(event["foeValue"]),
            deepened=None,
        )
    )
    timing.decided("move")
    record.unmodelled.extend(event["advanceNotes"])
    portlp.COUNTS["advanced"] += 1
    chain = event.get("chain")
    if chain is not None:
        _apply_chain(record, SimpleNamespace(chain=chain, position=None), setup.owners)
