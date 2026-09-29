"""IKA-386: a depth-2 call's sub-games filled, scored and solved in the port, which asks the
inference server itself -- off by default.

With `portlp` (IKA-381) a ladder read's sub-games crossed three times: the port filled and
encoded each node (`port.pending_payoffs`, a crossing per `search.FILL_BATCH`), Python laid
the arrays into its leaf's shared block and asked the server (`port.score_stacked` ->
`RemoteValue.from_encoded`), and Python sent the span blocks and the values back to the port
to fold and solve (`portlp.solve_subs`). Here one crossing does all three: the port fills the
same crossings (`fillsServed`), writes the same arrays into the same block and sends the
server the same requests (`rust/src/served.rs`) at the same points -- every waiting node's
rows as one batch once `search.GATHER_ROWS` wait after a crossing, and the rest at the end,
cut at `RemoteValue.batch_size` -- and folds and solves each node as `folds` does. What comes
back is each node's notes and counts and its value; the leaves never reach Python.

    POKEURAOU_LADDER_PORT_SERVED=1     # on (default off; implies `portlp`'s LPs)

Only where the leaf is an inference server's (`inference.RemoteValue`, unmerged or merged as
it asks) and the sub-games are scored stacked (`ladder.STACK`); anything else takes the old
road. The server answers a request by its rows and their count, so the same requests are the
same values: a read is the `portlp` read to the bit.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from . import portlp, rustnode, timing

ENV = "POKEURAOU_LADDER_PORT_SERVED"
#: Whether a ladder read's sub-games go to the port and on to the server. A list so a test
#: (and a ladder worker, told by its reader) can turn it on in one process.
ON = [os.environ.get(ENV, "0").strip() == "1"]
#: What went: crossings, nodes filled, the server's requests from the port and their rows
#: (the positive control: the port asked the server).
COUNTS = {"crossings": 0, "nodes": 0, "requests": 0, "rows": 0}


def set_on(on: bool) -> None:
    """Turn the road on or off in this process; on turns `portlp`'s LPs on with it."""
    ON[0] = bool(on)
    if on:
        portlp.set_on(True)


set_on(ON[0])


def leaf_of(evaluate: Any, stack: bool) -> Any:  # noqa: ANN401
    """The inference server's leaf behind ``evaluate`` when this road takes the call, or None."""
    if not ON[0] or not stack or timing.DUPES:
        return None
    from .inference import RemoteValue

    owner = getattr(evaluate, "__self__", evaluate)
    return owner if type(owner) is RemoteValue else None


def fill_scored(
    reg: Any,  # noqa: ANN401 - a Regulation
    chunks: Sequence[tuple[list, list]],
    leaf: Any,  # noqa: ANN401 - inference.RemoteValue
    *,
    budget: Any,  # noqa: ANN401 - a Budget
    gather: int,
) -> None:
    """Every sub-game of ``chunks`` (`search._fill_chunks`' crossings of `(sub, position,
    row, col)`) filled, scored and solved in one crossing: each sub's ``notes``, ``solved``
    and ``done`` as `pending_payoffs`, `score_stacked` and `portlp.solve_subs` leave them,
    or its ``error`` where the port refused cells of its node."""
    from . import port
    from .encode import rules_of
    from .inference import request_failed

    subs = [item for chunk, _links in chunks for item in chunk]
    if not subs:
        return
    links = [link for _chunk, chunk_links in chunks for link in chunk_links]
    rules = rules_of(leaf.from_encoded)
    wants_old = bool(rules.mega_from_slots)
    server = {
        "address": leaf.address,
        "model": leaf.model,
        "shm": leaf._block.name,  # noqa: SLF001 - the leaf's own block, idle while the port asks
        "bytes": int(leaf._block.size),  # noqa: SLF001
        "batch": int(leaf.batch_size),
        "merge": bool(leaf.merge),
    }
    answer = port.ask(
        reg,
        lambda node: node.fills_served(
            [(pos, list(row), list(col)) for _sub, pos, row, col in subs],
            [len(chunk) for chunk, _links in chunks],
            budget,
            server=server,
            gather=gather,
            rules=rules,
            links=links if any(link for link in links) else None,
        ),
    )
    served = answer.get("served") or {}
    waited = float(served.get("waitUs", 0.0)) / 1e6
    if "serverError" in answer:
        raise request_failed({"error": answer["serverError"], "oom": answer.get("oom"),
                              "capGb": answer.get("capGb")}, "inference failed")
    # The server's share of the port's answer is the leaf's clock, as it was on the old road.
    rustnode.PORT_WAITED[0] -= waited
    requests, rows = int(served.get("requests", 0)), int(served.get("rows", 0))
    leaf.waited += waited
    leaf.copied += float(served.get("copyUs", 0.0)) / 1e6
    leaf.calls += requests
    leaf.evaluated += rows
    leaf.ended += int(served.get("ended", 0))
    timing.count("leaves", rows)
    timing.count("forward.passes", requests)
    timing.count("serve.requests", requests)
    COUNTS["crossings"] += 1
    COUNTS["nodes"] += len(subs)
    COUNTS["requests"] += requests
    COUNTS["rows"] += rows
    portlp.COUNTS["crossings"] += 1
    portlp.COUNTS["lps"] += int(answer.get("lps", 0))
    nodes = answer["nodes"]
    if len(nodes) != len(subs):
        raise RuntimeError(f"`fillsServed` answered {len(nodes)} of {len(subs)} nodes")
    encoder = leaf.encoder
    for (sub, _pos, _row, _col), node in zip(subs, nodes, strict=True):
        echo = node.get("megaFromSlots")
        if wants_old and echo is not True:
            raise RuntimeError(
                "the Rust node was asked for the revision-1 can_mega rule and did not say it "
                f"applied it (echo {echo!r}); the binary predates IKA-141 -- rebuild it")
        timing.count("leaves.offered", int(node.get("offered", 0)))
        timing.count("leaves.stored", int(node["leaves"]))
        try:
            port.raise_refused([(int(i), int(j), str(why)) for i, j, why in node["refused"]])
        except port.PortRefused as refused:
            sub.error = refused
            continue
        # `port.note_port_rule`: the rule the port says it encoded with, per node.
        if hasattr(encoder, "note"):
            encoder.note("rust can_mega=" + {True: "slots", False: "holder", None: "unechoed"}[echo],
                         int(node["leaves"]))
        timing.count("leaves.node", int(node["leaves"]))
        sub.notes = set(node["unmodelled"])
        if "invalid" in node:
            raise ValueError(str(node["invalid"]))
        sub.solved = None if "failed" in node else float(node["value"])
        sub.done = True
        portlp.COUNTS["folded"] += 1


def counts() -> dict[str, int]:
    return dict(COUNTS)


__all__ = ["COUNTS", "ENV", "ON", "counts", "fill_scored", "leaf_of", "set_on"]
