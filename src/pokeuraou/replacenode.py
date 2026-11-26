"""IKA-457 stage 2.2: a replacement node solved in one request to the port.

`selfplay._do_replacement_node` with a hidden bench read the node one completion at a time: the
port filled every pair of replacements (`port.replacements_encoded`), Python scored the rows on
the inference server (`RemoteValue.from_encoded`), folded a phase that draws into its cell
(`replacement_matrix`'s `np.bincount`), and then solved each side's Bayesian game. The port now
does all of it in one `replacementNode` request (`rust/src/replacenode.rs`): the same fills, the
same requests to the server in the same order (side 0's completions, then side 1's), the same
settling of the ended leaves, and the same games.

The road is the inference server's leaf and nothing else (IKA-460 stage 4): any other leaf
(hp-share, a local net) is `beliefnode.UnservedLeaf`, and a fault the port answers is
`portlp.PortFault`.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import port, portlp, rustnode, timing
from .actions import SideAction
from .eqselect import reselect_bayesian
from .position import Position
from .regulation import Regulation


def solve(  # noqa: PLR0913 - the node's pieces
    reg: Regulation,
    options: Sequence[Sequence[SideAction]],
    spreads: dict[int, list],
    leaves: Sequence[Any],  # noqa: ANN401 - each side's leaf
    selects: Sequence[str],
) -> dict[int, tuple[list[float], float]]:
    """Each side's (strategy, value) for the replacement node, as `_do_replacement_node`'s
    loop gives them."""
    from .beliefnode import served_leaf
    from .encode import rules_of
    from .inference import request_failed

    owners: list[Any] = []
    served = [served_leaf(leaves[side], "the replacement node") for side in (0, 1)]
    for owner in served:
        if not any(owner is seen for seen in owners):
            owners.append(owner)

    m, n = len(options[0]), len(options[1])
    request: dict[str, Any] = {
        "kind": "replacementNode",
        "rows": m,
        "cols": n,
        "pairs": rustnode._joined(  # noqa: SLF001
            *(rustnode._candidates((a, b)) for a in options[0] for b in options[1])  # noqa: SLF001
        ),
    }
    jobs: list[dict[str, Any]] = []
    positions: list[tuple[int, Position]] = []
    for side in (0, 1):
        owner = served[side]
        index = next(k for k, seen in enumerate(owners) if seen is owner)
        for item in spreads[1 - side]:
            request[f"p{len(jobs)}"] = rustnode._position(item.position)  # noqa: SLF001
            # The job's `side` is the one whose bench the completion fills (`spreads`' key), as in
            # `beliefNode`: the game of side `s` is over the jobs of side `1 - s`.
            jobs.append({"side": 1 - side, "leaf": index})
            positions.append((side, item.position))
    request["jobs"] = jobs
    servers = []
    for owner in owners:
        one: dict[str, Any] = {
            "address": owner.address,
            "model": owner.model,
            "shm": owner._block.name,  # noqa: SLF001 - the leaf's own block, idle while the port asks
            "bytes": int(owner._block.size),  # noqa: SLF001
            "batch": int(owner.batch_size),
            "merge": bool(owner.merge),
            "netScoresEnds": bool(owner.encoder.rules.net_scores_ends),
        }
        rules = rules_of(owner)
        if rules.mega_from_slots:
            one["encoding"] = rules.to_request()
        servers.append(one)
    request["servers"] = servers
    weights = {
        side: np.asarray([item.weight for item in spreads[1 - side]], dtype=np.float64)
        for side in (0, 1)
    }
    request["solve"] = [{"side": side, "weights": portlp._b64(weights[side])}  # noqa: SLF001
                        for side in (0, 1)]

    answer = port.ask(reg, lambda node: node.replacement_node(request))
    portlp.COUNTS["crossings"] += 1
    portlp.COUNTS["lps"] += int(answer.get("lps", 0))
    if "fault" in answer:
        raise portlp.PortFault("replacementNode", str(answer["fault"]))
    if "serverError" in answer:
        raise request_failed({"error": answer["serverError"], "oom": answer.get("oom"),
                              "capGb": answer.get("capGb")}, "inference failed")
    if "served" not in answer:
        raise RuntimeError("the port did not answer `replacementNode`; the binary predates it -- rebuild it")
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
    # `port.note_port_rule` of each fill, and what `replacements_encoded` counted.
    for job, fill in zip(jobs, answer["fills"], strict=True):
        encoder = getattr(owners[job["leaf"]], "encoder", None)
        if encoder is not None and hasattr(encoder, "note"):
            what = "rust can_mega=" + {True: "slots", False: "holder", None: "unechoed"}[fill["mega"]]
            encoder.note(what, int(fill["leaves"]))
        timing.count("port.replacements.encoded", int(fill["pairs"]))

    flat = np.frombuffer(base64.b64decode(answer["matrices"]), dtype="<f8")
    by_side: dict[int, list[np.ndarray]] = {}
    for k, (side, _position) in enumerate(positions):
        by_side.setdefault(side, []).append(
            flat[k * m * n : (k + 1) * m * n].reshape(m, n).astype(np.float64)
        )
    solved: dict[int, Any] = {}
    for one in answer["answers"]:
        side = int(one["side"])
        games = [mat if side == 0 else -mat.T for mat in by_side[side]]
        solved[side] = (games, portlp.bayesian_answer(games, weights[side], one))
    portlp.COUNTS["replaced"] += 1
    out: dict[int, tuple[list[float], float]] = {}
    for side in (0, 1):
        games, got = solved[side]
        if isinstance(got, Exception):
            raise got
        strategy = reselect_bayesian(
            games, weights[side], float(got.value), got.row_strategy, selects[side]
        )
        out[side] = ([float(x) for x in strategy], float(got.value))
    return out
