"""The port answers a node the same at any cell-thread count (IKA-32).

`node --threads N` resolves a node's cells on N threads and gathers them back in the order
they were asked for; everything after the gathering -- a leaf the node already has, the
encoding, the header -- stays on one thread. So the answer is the same bytes at 1 and at N,
which is what keeps a game played on N threads the game played on one.

Each request below is sent to a 1-thread and a 4-thread process as the same line, and the
two answers are compared as bytes: the header less the child's own clocks, and the arrays
behind it. The 4-thread process is then asked how much of the work ran off the thread that
reads requests (`parallel`): an agreement where the threads never ran would be no check.
The control build (`--features ika32-control`) gathers in finishing order and fails here.
A `fills` crossing of plain nodes goes one node per thread (IKA-32 stage 2), each resolved
and encoded whole there; its body is compared the same way.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from pokeuraou import rustnode
from pokeuraou.budget import Budget

from .test_rust_node import _node, _with_parting_shot

#: The child's own clocks: how long this answer took, which is the one thing that differs.
CLOCKS = ("resolveUs", "encodeUs", "foldUs", "parseUs", "headerUs")


@pytest.fixture()
def processes():
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    reg, pos, row, col = _node(limit=10)
    # A Parting Shot that goes first pauses the turn for a replacement, so the pool's
    # resuming (`resume_all`) is on the road too, not only plain cells.
    pos.sides[0].active_pokemon()[1].boosts["spe"] = 6
    row = _with_parting_shot(reg, pos, row)
    one = rustnode.RustNode(reg, threads=1)
    many = rustnode.RustNode(reg, threads=4)
    yield reg, pos, row, col, one, many
    one.close()
    many.close()


def _ask(node: rustnode.RustNode, request: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    """One line out; the header back with the clocks taken off, and the body if any."""
    process = node._process
    process.stdin.write(json.dumps(request).encode("utf-8") + b"\n")
    process.stdin.flush()
    header = json.loads(process.stdout.readline().decode("utf-8"))
    body = b""
    if header.get("via") == "pipe":
        body = process.stdout.read(int(header["bytes"]))
    for key in CLOCKS:
        header.pop(key, None)
    for nested in header.get("nodes", []):
        for key in CLOCKS:
            nested.pop(key, None)
    return header, body


def _node_request(pos: Any, row: list, col: list, **extra: Any) -> dict[str, Any]:
    return {
        "position": pos.to_json(),
        "ours": [[rustnode.dump_action(a) for a in side.slots] for side in row],
        "theirs": [[rustnode.dump_action(a) for a in side.slots] for side in col],
        "budget": rustnode.dump_budget(Budget.matrix()),
        "objectives": ["hp-share", "faints"],
        **extra,
    }


def test_the_answers_are_the_same_bytes_at_one_and_at_four_threads(processes) -> None:
    reg, pos, row, col, one, many = processes
    cells = [[i, j] for i in range(len(row)) for j in range(len(col)) if (i * 7 + j) % 3]
    requests = [
        _node_request(pos, row, col),
        _node_request(pos, row, col, encode=True),
        _node_request(pos, row, col, encode=True, cells=cells),
        {
            "kind": "fills",
            "requests": [
                _node_request(pos, row, col, encode=True, keep=True),
                _node_request(pos, row[:4], col, encode=True),
            ],
        },
        # Plain nodes (none keeps or reads another's): one per thread, whole (stage 2).
        {
            "kind": "fills",
            "requests": [
                _node_request(pos, row[:3], col[:3], encode=True),
                _node_request(pos, row[2:], col, encode=True),
                _node_request(pos, row, col[1:5], encode=True),
                _node_request(pos, row[:1], col[:2], encode=True),
                _node_request(pos, row[3:7], col[2:], encode=True, cells=[[0, 0], [1, 2], [3, 1]]),
            ],
        },
        {
            "kind": "many",
            "requests": [
                {
                    "kind": "turn",
                    "position": pos.to_json(),
                    "actions": [
                        [rustnode.dump_action(a) for a in row[i].slots],
                        [rustnode.dump_action(a) for a in col[j].slots],
                    ],
                    "budget": rustnode.dump_budget(Budget.matrix()),
                    "full": True,
                    "select": None,
                    "events": False,
                }
                for i in range(len(row))
                for j in range(0, len(col), 3)
            ]
            + [
                {
                    "kind": "score",
                    "position": pos.to_json(),
                    "side": side,
                    "candidates": [
                        [rustnode.dump_action(a) for a in c.slots] for c in (row, col)[side]
                    ],
                }
                for side in (0, 1)
            ],
        },
    ]
    folded = 0
    for request in requests:
        alone, alone_body = _ask(one, request)
        spread, spread_body = _ask(many, request)
        assert "error" not in alone, alone
        assert spread == alone, f"{request.get('kind', 'fill')}: the headers differ"
        assert spread_body == alone_body, f"{request.get('kind', 'fill')}: the arrays differ"
        folded += len(alone.get("folded", []))
    assert folded, "no cell paused: the pool's resuming was never compared"

    counted = many.parallel()
    assert counted["threads"] == 4
    assert counted["maps"] >= len(requests)
    assert counted["offMain"] > 0, "no cell ran on a worker thread"
    assert counted["nodes"] >= 5, "no crossing's nodes went one per thread"
    alone = one.parallel()
    assert alone["threads"] == 1 and alone["maps"] == 0 and alone["nodes"] == 0, alone


def test_the_thread_count_reaches_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """`set_port_threads` and the environment both reach the child, and a count set here
    wins over the environment's (it is passed as `--threads`)."""
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    reg, *_ = _node(limit=2)
    monkeypatch.setenv(rustnode.ENV_THREADS, "3")
    rustnode.reset()
    try:
        from_env = rustnode.RustNode(reg)
        assert from_env.parallel()["threads"] == 3
        from_env.close()
        rustnode.set_port_threads(2)
        assert rustnode.port_threads() == 2
        set_here = rustnode.RustNode(reg)
        assert set_here.parallel()["threads"] == 2
        set_here.close()
    finally:
        rustnode.set_port_threads(None)
        rustnode.reset()
        os.environ.pop(rustnode.ENV_THREADS, None)
    with pytest.raises(ValueError):
        rustnode.set_port_threads(0)
