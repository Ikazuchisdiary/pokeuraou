"""IKA-458 stage 3.6a: a game's replacement phase chosen and played in one request to the port
(`replacementPhase`, `rust/src/replacephase.rs`).

`selfplay._do_replacement_node` made its options, asked for the node (`replacenode.solve`), drew
each side's replacement with numpy's `choice` and asked for the phase once more for each draw it
opened. The port does all of it from the position, the generator's state and where the leaves and
the hidden bench are: the same requests to the inference server, the same draws, the same
position. This module builds the request and books what comes back; the record's `Decision` is
written by the caller as before.

Where this road does not go -- a leaf that is not the inference server's, another equilibrium
selection, a game the port does not hold (the completions were Python's), luck gathered inside
the process -- `play` returns None and the caller does the node itself (`replacenode.solve` over
the same port), the generator untouched. A fault the port answers is `portlp.PortFault`.
"""

from __future__ import annotations

from typing import Any

from . import port, portlp, rustnode, timing
from .position import Position
from .regulation import Regulation


class Played:
    """What the port chose and played: the options as the record writes them, the strategies, the
    values, the indices drawn, the position the phase ended in and its notes."""

    __slots__ = ("chosen", "foe_value", "notes", "options", "policy", "position", "value")

    def __init__(self, answer: dict[str, Any]) -> None:
        self.options: list[list[str]] = [list(side) for side in answer["options"]]
        self.policy: list[list[float]] = [[float(x) for x in side] for side in answer["policy"]]
        self.value = float(answer["value"])
        self.foe_value = float(answer["foeValue"])
        self.chosen: list[int] = [int(i) for i in answer["chosen"]]
        self.position = Position.from_json(answer["position"])
        self.notes: list[str] = list(answer["notes"])


def play(  # noqa: PLR0913 - the phase's pieces
    reg: Regulation,
    rng: Any,  # noqa: ANN401 - numpy.random.Generator
    pos: Position,
    leaves: tuple[Any, Any],
    selects: tuple[str, str],
    game: int | None,
    seen: list[frozenset[str]],
    leads: list[frozenset[str] | None],
) -> Played | None:
    """The replacement phase at `pos`, or None where the port does not play it. `rng` is set to
    the state after the draws only once the answer is in."""
    from .selfplay import DEFAULT_EQ_SELECT, _HiddenBench, _self_switch_request

    if game is None or timing.DUPES:
        return None
    if type(rng.bit_generator).__name__ != "PCG64":
        return None
    if tuple(selects) != (DEFAULT_EQ_SELECT, DEFAULT_EQ_SELECT):
        return None
    ready = _self_switch_request(
        leaves,
        _HiddenBench((), list(seen), None, list(leads), game),
    )
    if ready is None:
        return None
    context, owners = ready
    before = rng.bit_generator.state
    request = {
        "kind": "replacementPhase",
        "position": rustnode._position(pos),  # noqa: SLF001
        "rng": {
            "state": format(before["state"]["state"], "x"),
            "inc": format(before["state"]["inc"], "x"),
            "hasUint32": bool(before["has_uint32"]),
            "uinteger": int(before["uinteger"]),
        },
        "context": context,
    }
    answer = port.ask(reg, lambda node: node.replacement_phase(request))
    portlp.COUNTS["crossings"] += 1
    portlp.COUNTS["lps"] += int(answer.get("lps", 0))
    if "fault" in answer:
        raise portlp.PortFault("replacementPhase", str(answer["fault"]))
    if "serverError" in answer:
        from .inference import request_failed

        raise request_failed({"error": answer["serverError"], "oom": answer.get("oom"),
                              "capGb": answer.get("capGb")}, "inference failed")
    if "rng" not in answer:
        raise RuntimeError("the port did not answer `replacementPhase`; the binary predates it -- rebuild it")
    _book(owners, answer)
    after = dict(before)
    after["state"] = {"state": int(answer["rng"]["state"], 16), "inc": int(answer["rng"]["inc"], 16)}
    after["has_uint32"] = int(bool(answer["rng"]["hasUint32"]))
    after["uinteger"] = int(answer["rng"]["uinteger"])
    rng.bit_generator.state = after
    played = Played(answer)
    portlp.COUNTS["phased"] += 1
    return played


def _book(owners: list[Any], answer: dict[str, Any]) -> None:
    """The inference server's work as `replacenode.solve` books it on each leaf."""
    for owner, served in zip(owners, answer["served"], strict=True):
        waited = float(served.get("waitUs", 0.0)) / 1e6
        rustnode.PORT_WAITED[0] -= waited
        requests, rows = int(served.get("requests", 0)), int(served.get("rows", 0))
        owner.waited += waited
        owner.copied += float(served.get("copyUs", 0.0)) / 1e6
        owner.calls += requests
        owner.evaluated += rows
        owner.ended += int(served.get("ended", 0))
        timing.count("leaves", rows)
        timing.count("forward.passes", requests)
        timing.count("serve.requests", requests)
        if int(served.get("ended", 0)):
            timing.count("leaves.ended", int(served["ended"]))
    for fill in answer["fills"]:
        encoder = getattr(owners[int(fill["leaf"])], "encoder", None)
        if encoder is not None and hasattr(encoder, "note"):
            what = "rust can_mega=" + {True: "slots", False: "holder", None: "unechoed"}[fill["mega"]]
            encoder.note(what, int(fill["leaves"]))
        timing.count("port.replacements.encoded", int(fill["pairs"]))
