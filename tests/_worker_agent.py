"""IKA-458 stage 3.7: generation's agent and the board's, built from the same flags and compared.

`tools/agent_drift.py --check` and `tests/test_worker_drift.py` use this. It writes a small pool,
starts the tests' stand-in inference server (a leaf and a Q, no GPU), runs `tools/selfplay.py
--rust-config` and `tools/pool_match.py --rust-config` in this process with generation's flags
(both board arms on generation's leaf, Q and width), and compares the keys of the two configs that
decide which agent plays, by value. `rustworker_source` swaps in another `rustworker.py` (a
control: a copy with one line changed must make the comparison fail).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: The keys of a move node the board's worker fills per arm itself (`rustworker.BOARD_FILLS`):
#: compared through the arm's leaf, Q and width below rather than as node keys.
BOARD_FILLS = ("server", "qServer", "widths")
#: The node keys the Rust workers set or change after reading a config (`worker.rs` sets the
#: servers' addresses and blocks; `boardworker.rs` sets the servers, the widths and an arm's
#: other executable). `rust_node_writes` holds the sources to this list.
RUST_NODE_FILLS = frozenset({"server", "qServer", "widths", "binary", "regulation"})
#: Keys that differ between generation and a board on purpose: generation explores the
#: selection (epsilon 0.25), a board plays the equilibrium (0, as a rating wants).
ALLOWED_DIFFERENT = ("epsilon", "temperature")


def _tool(name: str) -> Any:  # noqa: ANN401
    """A fresh copy of tools/<name>.py, so it binds the `rustworker` in `sys.modules` now."""
    spec = importlib.util.spec_from_file_location(f"_agent_tool_{name}", ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@contextmanager
def _rustworker(source: Path | None) -> Iterator[None]:
    """`pokeuraou.rustworker` read from ``source`` while the block runs (None: the tree's)."""
    if source is None:
        yield
        return
    import pokeuraou

    real = sys.modules.get("pokeuraou.rustworker")
    spec = importlib.util.spec_from_file_location("pokeuraou.rustworker", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pokeuraou.rustworker"] = module
    pokeuraou.rustworker = module
    try:
        spec.loader.exec_module(module)
        yield
    finally:
        if real is not None:
            sys.modules["pokeuraou.rustworker"] = real
            pokeuraou.rustworker = real
        else:
            sys.modules.pop("pokeuraou.rustworker", None)


@contextmanager
def serving(pool_path: Path) -> Iterator[str]:
    """The stand-in server for the pool's regulation: leaf arm `value`, Q arm `q`."""
    from pokeuraou.encode import Encoder
    from pokeuraou.pool import load_pool

    from ._served import hp_share_model, model_of, serve
    from .test_q_rank import _ServedStub

    encoder = Encoder(load_pool(pool_path).reg)
    server, address = serve(
        {"value": model_of(hp_share_model(encoder))}, arms={"value": ["stub.pt"]},
        q_models={"q": _ServedStub(encoder.vocab.fingerprint())},
    )
    try:
        yield address
    finally:
        server.shutdown()
        server.server_close()


def write_pool(directory: Path) -> Path:
    from .test_poolplay import _variants, _write_pool

    return _write_pool(directory / "pool.json", _variants())


def build_both(
    address: str, pool_path: Path, tmp: Path, rustworker_source: Path | None = None,
    limit: int = 12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(generation's config, the board's config) for generation's flags, at generation's width
    (`tools/generate_queue.py` passes `--limit`; `agent_drift` holds the board's default to it)."""
    import json

    from pokeuraou import qrank

    # The tools' own echo goes nowhere: what is compared is the configs they write.
    with _rustworker(rustworker_source), contextlib.redirect_stderr(io.StringIO()):
        qrank._INSTALLED.clear()  # noqa: SLF001
        argv = sys.argv
        sys.argv = [
            "selfplay.py", "--pool", str(pool_path), "--inference", address, "--inference-arm",
            "value", "--q-arm", "q", "--rank-leaf", "--seed", "1", "--hide-bench",
            "--limit", str(limit),
            "--out", str(tmp / "g" / "games.jsonl"), "--rust-config", str(tmp / "g.json"),
        ]
        try:
            _tool("selfplay").main()
        finally:
            sys.argv = argv
            qrank._INSTALLED.clear()  # noqa: SLF001
        try:
            _tool("pool_match").main([
                "--pool", str(pool_path), "--inference", address, "--inference-arm", "value",
                "--baseline-inference-arm", "value", "--q-arm", "q", "--rank-leaf",
                "--baseline-rank-leaf", "--hide-bench", "--limit", str(limit),
                "--games-out", str(tmp / "b" / "games.jsonl"),
                "--rust-config", str(tmp / "b.json"),
            ])
        finally:
            qrank._INSTALLED.clear()  # noqa: SLF001
    return (json.loads((tmp / "g.json").read_bytes()), json.loads((tmp / "b.json").read_bytes()))


def agent_differences(generation: dict[str, Any], board: dict[str, Any]) -> list[str]:
    """Where a board arm of generation's leaf, Q and width plays another agent than generation:
    the move node, the budget of the exact resolution, the turn cap, the selection's notes, the
    width, the leaf and the selection store's tag, the Q. Empty when every arm plays generation's."""
    problems: list[str] = []
    for key in ("exactBudget", "maxTurns", "notes"):
        if board.get(key) != generation.get(key):
            problems.append(f"the board's {key} is {board.get(key)!r}, generation's "
                            f"{generation.get(key)!r}")
    node = {k: v for k, v in generation["node"].items() if k not in BOARD_FILLS}
    servers = board.get("servers") or []
    for k, arm in enumerate(board.get("arms") or []):
        mine = arm.get("node")
        if mine != node:
            keys = sorted(set(mine or {}) ^ set(node)
                          | {x for x in set(mine or {}) & set(node) if mine[x] != node[x]})
            problems.append(f"arm {k}'s node differs from generation's at {keys}")
        if [arm.get("limit")] * 2 != generation["node"].get("widths"):
            problems.append(f"arm {k}'s width {arm.get('limit')} is not generation's "
                            f"{generation['node'].get('widths')}")
        leaf = servers[arm["leaf"]] if isinstance(arm.get("leaf"), int) and arm["leaf"] < len(servers) else {}
        if leaf.get("model") != generation["value"]["model"] or arm.get("tag") != generation["tag"]:
            problems.append(f"arm {k}'s leaf ({leaf.get('model')}, tag {arm.get('tag')}) is not "
                            f"generation's ({generation['value']['model']}, {generation['tag']})")
        qs = board.get("qServers") or []
        q = qs[arm["q"]] if isinstance(arm.get("q"), int) and arm["q"] < len(qs) else {}
        if q.get("model") != generation["q"]["model"]:
            problems.append(f"arm {k}'s Q {q.get('model')} is not generation's {generation['q']['model']}")
    return problems


def rust_node_writes(texts: dict[str, str] | None = None) -> list[str]:
    """The node keys the Rust workers write that are not in `RUST_NODE_FILLS`, as problems.
    ``texts`` stands in for the sources (a control); None reads worker.rs and boardworker.rs."""
    import re

    if texts is None:
        texts = {name: (ROOT / "rust" / "src" / name).read_bytes().decode("utf-8")
                 for name in ("worker.rs", "boardworker.rs")}
    problems = []
    for name, text in texts.items():
        for key in sorted(set(re.findall(r'\bnode\["(\w+)"\]\s*=(?!=)', text))
                          | set(re.findall(r'&mut node\["(\w+)"\]', text))):
            if key not in RUST_NODE_FILLS:
                problems.append(f"rust/src/{name} writes node[{key!r}], not one of "
                                f"{sorted(RUST_NODE_FILLS)}")
        if re.search(r"node\.as_object_mut\(\)|node\[[^\]\"]+\]\s*=", text):
            problems.append(f"rust/src/{name} changes a node in a way this check cannot read")
    return problems


def worker_drift(rustworker_source: Path | None = None) -> list[str]:
    """Builds both configs from generation's flags and compares them; the Rust sources' node
    writes too. Empty when the board's arms play generation's agent."""
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        pool_path = write_pool(tmp)
        with serving(pool_path) as address:
            generation, board = build_both(address, pool_path, tmp, rustworker_source)
    return agent_differences(generation, board) + rust_node_writes()
