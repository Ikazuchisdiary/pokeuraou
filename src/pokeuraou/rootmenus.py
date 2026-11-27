"""IKA-457 stage 2.3: a move node's two menus by the Q, built in the port.

`selfplay._menus` with a `q-nocover` rank fill built them in the worker's Python: both sides'
legal pools (`qhead.legal_pool`), `qrank.prefetch`'s Q request (the arrays, a `qfeatures`
crossing a side, `RemoteQ.matrices`), the Q's games, and `narrow(..., rank=..., cover=False)`
of each ranking (which also asked the port for both pools' damage scores, only to keep them
in a detail line nobody reads). Now one `rootMenus` request
(`rust/src/qmenus.rs`) does all of it and hands back the two menus: the same pools action for
action, the same arrays byte for byte into the Q's own block, the same `q` / `q_many` request,
the same games, numpy's own BLAS for the scores, and the cut at each side's width.

Only where the Q is the inference server's (`qrank.RemoteQ`), its encoder is `encode.Encoder`
and numpy's BLAS is an OpenBLAS the port can load; the fill is a ``q-nocover`` one; and no
wider menu is asked for. Anything else and an empty pool leave the
menus to Python (`selfplay._menus`'s own ranking: the damage order, the leaf's, a wider menu).
A fault from the port is `portlp.PortFault`. A binary that predates the request stops, as the
other requests do.

Stage 2.4 (`beliefnode.fused_node`) asks the same of the same request as the first half of
`moveNode`, whose second half is the node over those menus: `Plan` is the half both use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import port, portlp, portmenus, qrank, rustnode, timing
from .actions import SideAction
from .narrow import note_coverless_count
from .position import Position
from .regulation import Regulation


def static_request(model: Any, limits: Sequence[int], server_key: str) -> dict[str, Any]:  # noqa: ANN401
    """The menus' request that no position changes: the widths, the Q's properties and
    `can_mega` rule and its server (`Plan`, and the Rust worker's config, `rustworker`)."""
    return {
        "widths": [int(limits[0]), int(limits[1])],
        "properties": bool(model.properties),
        "megaFromSlots": bool(model.encoder.rules.mega_from_slots),
        server_key: {
            "address": model.address,
            "model": model.model,
            "shm": model._block.name,  # noqa: SLF001 - the Q's own block, idle while the port asks
            "bytes": int(model.buffer_bytes),
        },
    }


class Plan:
    """The menus' request and what is done with its answer, for `menus` and for `moveNode`."""

    def __init__(  # noqa: PLR0913 - the node's pieces
        self, rank_fill: str, model: Any,  # noqa: ANN401
        pos: Position, views: Sequence[Position], limits: Sequence[int], *, server_key: str,
    ) -> None:
        self.rank_fill = rank_fill
        self.model = model
        self.request: dict[str, Any] = {
            "position": rustnode._position(pos),  # noqa: SLF001
            "v0": rustnode._position(views[0]),  # noqa: SLF001
            "v1": rustnode._position(views[1]),  # noqa: SLF001
            "same": views[0] is views[1],
            **static_request(model, limits, server_key),
        }
        self.menus: tuple[list[SideAction], list[SideAction]] = ([], [])

    def finish(self, answer: dict[str, Any]) -> None:
        """Book what the port did, as the Python road books it, and keep the menus."""
        from .inference import request_failed

        model = self.model
        encoder = model.encoder
        served = answer.get("served") or {}
        waited = float(served.get("waitUs", 0.0)) / 1e6
        # The server's share of the port's answer is the Q's clock, as it was on the Python road.
        rustnode.PORT_WAITED[0] -= waited
        model.waited += waited
        model.calls += int(answer.get("requests", 0))
        model.trips += int(served.get("requests", 0))
        if "tooBig" in answer:
            raise RuntimeError(str(answer["tooBig"]))
        if "serverError" in answer:
            raise request_failed({"error": answer["serverError"], "oom": answer.get("oom"),
                                  "capGb": answer.get("capGb")}, "Q request failed")
        if "invalid" in answer:
            raise ValueError(str(answer["invalid"]))
        if "menus" not in answer:
            raise RuntimeError("the port did not answer `rootMenus`; the binary predates it -- rebuild it")
        asked = int(answer["requests"])
        # What `prefetch` and `q_ranking` count, from what the port says it did.
        timing.count("q.prefetch.trips")
        timing.count("q.prefetch.requests", asked)
        got = qrank.QRANKED.setdefault(self.rank_fill, {"rankings": 0, "cells": 0})
        for side in (0, 1):
            timing.count("q.prefetched")
            got["rankings"] += 1
            got["cells"] += int(answer["cells"][side])
            timing.count("q.rankings")
            timing.count("q.cells", int(answer["cells"][side]))
            note_coverless_count(self.rank_fill, int(answer["uncovered"][side]))
        if asked == 1:
            timing.count("q.solve.shared")
        # `Encoder.encode_positions`' own count, one position a request as `_pool_arrays` encodes.
        encoder.note("python can_mega=" + ("slots" if encoder.rules.mega_from_slots else "holder"), asked)
        portlp.COUNTS["menus"] += 1
        first, second = ([portmenus._action(a) for a in side] for side in answer["menus"])  # noqa: SLF001
        self.menus = (first, second)


def plan(  # noqa: PLR0913 - the node's pieces
    reg: Regulation,  # noqa: ARG001
    pos: Position,
    views: Sequence[Position],
    limits: Sequence[int],
    rank_fill: str,
    *,
    server_key: str = "server",
) -> Plan | None:
    """The menus' request where this road goes, else None (Python then builds the menus)."""
    from . import actions
    from .encode import Encoder

    if actions.ALLY_TARGETS[0] != "off":
        # IKA-181: the port's `legal.rs` lists no ally target of a `normal` move.
        return None
    model = qrank.installed(qrank.q_name(rank_fill))
    if type(model) is not qrank.RemoteQ or type(model.encoder) is not Encoder:
        return None
    return Plan(rank_fill, model, pos, views, limits, server_key=server_key)


def menus(  # noqa: PLR0913 - the node's pieces
    reg: Regulation,
    pos: Position,
    views: Sequence[Position],
    limits: Sequence[int],
    rank_fill: str,
) -> tuple[list[SideAction], list[SideAction]] | None:
    """Both sides' menus as `_menus` builds them, or None where this road does not go."""
    planned = plan(reg, pos, views, limits, rank_fill)
    if planned is None:
        return None
    request = {"kind": "rootMenus", **planned.request}
    answer = port.ask(reg, lambda node: node.root_menus(request))
    portlp.COUNTS["crossings"] += 1
    portlp.COUNTS["lps"] += int(answer.get("lps", 0))
    if "fault" in answer:
        raise portlp.PortFault("rootMenus", str(answer["fault"]))
    planned.finish(answer)
    return planned.menus
