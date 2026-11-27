"""IKA-458 stage 3.7: generation's agent and the board's, built from the same flags and compared.

`tools/agent_drift.py --check` and `tests/test_worker_drift.py` use this. It writes a small pool,
starts the stand-in inference server (`pokeuraou.standin`: a leaf and a Q, no GPU), runs
`tools/selfplay.py --rust-config` and `tools/pool_match.py --rust-config` in this process with
generation's flags (both board arms on generation's leaf, Q, width, seed and store), and compares
the two configs by value: every key of both, read per board arm the way the board's worker hands it
to `playGame` (`board_view`), except the keys `ALLOWED_DIFFERENT` names with its reason. A key
either config gains is compared until someone names it there. `rustworker_source` swaps in
another `rustworker.py` (a control: a copy with one line changed must make the comparison fail).

The flags are written here by hand, not taken from the command lines `tools/generate_queue.py` and
`tools/match_queue.py` build.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .regulation import repo_root

ROOT = repo_root()

#: The keys of a server the workers fill for the run they are in (addresses, shared memory, the
#: server's role between the two inference addresses of a board): not part of the agent.
PER_RUN = ("address", "shm", "role")
#: The node keys the Rust workers set or change after reading a config (`worker.rs` sets the
#: servers' addresses and blocks; `boardworker.rs` sets the servers, the widths and an arm's
#: other executable). `rust_node_writes` looks for others in the sources, as a marker only.
RUST_NODE_FILLS = frozenset({"server", "qServer", "widths", "binary", "regulation"})
#: The keys compared nowhere, each with why. Everything else of both configs is compared.
ALLOWED_DIFFERENT = {
    # Generation explores the selection (0.25 by default); a board plays the equilibrium: its
    # epsilon must be 0 (`agent_differences` checks that), and the temperature then shapes nothing.
    "epsilon": "generation explores the selection; a board plays the equilibrium (epsilon 0)",
    "temperature": "only shapes the exploration a board does not do",
    # What each worker writes into its records, not how it plays.
    "record": "generation's record template",
    "records": "the board's record templates, one per seat",
    "label": "the board's name for its tested arm, written into the records",
    "mode": "says which worker reads the config",
    # Compared per arm (`board_view`), not as their own keys: the arm's leaf server as its
    # node.server and as the context's server of each side, its Q server as its node.qServer and q.
    "servers": "read per arm into its node.server and each side's context server",
    "qServers": "read per arm into its node.qServer and q",
    "arms": "each arm is compared as a whole (`board_view`)",
    # `context.hidden` is a placeholder both workers rebuild per game (the game's number, the
    # seen identities, the leads); `context.leaf` names a server by its place in a list the two
    # configs order differently. The servers themselves are compared per side (`_side_view`:
    # the one the side's leaf names and the others), and the rest of the context is compared.
    "context.hidden": "rebuilt by each worker per game",
    "context.leaf": "a place in a list; the server it names and the others are compared",
}


def _stub_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """A made-up Q of the request's arrays (as `tests/test_q_rank.py`'s)."""
    a0 = arrays["acts0"].reshape(len(arrays["acts0"]), -1).astype(np.float64).sum(1)
    a1 = arrays["acts1"].reshape(len(arrays["acts1"]), -1).astype(np.float64).sum(1)
    f0 = arrays["feats0"].astype(np.float64).sum(1)
    f1 = arrays["feats1"].astype(np.float64).sum(1)
    return 0.5 + 0.4 * np.sin(0.37 * a0[:, None] + 0.11 * f0[:, None] - 0.23 * a1[None, :]
                              - 0.07 * f1[None, :])


class StubQ:
    """A served Q (`qrank.served_q`'s interface) over `_stub_matrix`."""

    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint
        self.properties = True
        self.files = ["stub-q.pt"]
        self.held = 0.0
        self.calls = 0

    def __call__(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        self.calls += 1
        return _stub_matrix(arrays)


def write_pool(directory: Path) -> Path:
    """Three variants of the M-B roster in the pool file's shape (as `tests/test_poolplay.py`'s),
    under M-C's regulation, the one generation and the boards play."""
    base = json.loads((ROOT / "configs" / "teams" / "rizabanadohido.json").read_bytes())["team"]
    rotated = copy.deepcopy(base[2:] + base[:2])
    respread = copy.deepcopy(base)
    for member in respread:
        member["sp"] = {"hp": 32, "spe": 32, "atk": 2}
    data = {
        "id": "agent-pool", "name": "agent", "regulation": "gen9championsvgc2026regmc",
        "character": "the agent comparison's pool",
        "validatedAgainst": {"formatId": "gen9championsvgc2026regmc"},
        "teams": [{"id": f"t{i}", "name": f"team {i}", "team": t}
                  for i, t in enumerate((base, rotated, respread))],
    }
    path = directory / "pool.json"
    path.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    return path


@contextmanager
def serving(pool_path: Path) -> Iterator[str]:
    """The stand-in server for the pool's regulation: leaf arm `value`, Q arm `q`."""
    from .encode import Encoder
    from .pool import load_pool
    from .standin import hp_share_model, model_of, serve

    encoder = Encoder(load_pool(pool_path).reg)
    server, address = serve(
        {"value": model_of(hp_share_model(encoder))}, arms={"value": ["stub.pt"]},
        q_models={"q": StubQ(encoder.vocab.fingerprint())},
    )
    try:
        yield address
    finally:
        server.shutdown()
        server.server_close()


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
            # Not imported before: neither the module table nor the package keeps the copy (the
            # package attribute is what `from pokeuraou import rustworker` reads first).
            sys.modules.pop("pokeuraou.rustworker", None)
            if getattr(pokeuraou, "rustworker", None) is module:
                delattr(pokeuraou, "rustworker")


def build_both(
    address: str, pool_path: Path, tmp: Path, rustworker_source: Path | None = None,
    limit: int = 12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(generation's config, the board's config) for generation's flags, at generation's width
    (`tools/generate_queue.py` passes `--limit`; `agent_drift` holds the board's default to it),
    with one seed and one selection store for both."""
    from . import qrank

    common = ["--pool", str(pool_path), "--inference", address, "--inference-arm", "value",
              "--q-arm", "q", "--rank-leaf", "--seed", "1", "--hide-bench", "--limit", str(limit),
              "--selection-store", str(tmp / "store")]
    # The tools' own echo goes nowhere: what is compared is the configs they write.
    with _rustworker(rustworker_source), contextlib.redirect_stderr(io.StringIO()):
        qrank._INSTALLED.clear()  # noqa: SLF001
        argv = sys.argv
        sys.argv = ["selfplay.py", *common, "--out", str(tmp / "g" / "games.jsonl"),
                    "--rust-config", str(tmp / "g.json")]
        try:
            _tool("selfplay").main()
        finally:
            sys.argv = argv
            qrank._INSTALLED.clear()  # noqa: SLF001
        try:
            _tool("pool_match").main([
                *common, "--baseline-inference-arm", "value", "--baseline-rank-leaf",
                "--baseline-selection-store", str(tmp / "store"),
                "--games-out", str(tmp / "b" / "games.jsonl"), "--rust-config", str(tmp / "b.json"),
            ])
        finally:
            qrank._INSTALLED.clear()  # noqa: SLF001
    return (json.loads((tmp / "g.json").read_bytes()), json.loads((tmp / "b.json").read_bytes()))


def _server(server: Any) -> Any:  # noqa: ANN401
    """A server as an agent sees it: without the keys of the run it is in (`PER_RUN`)."""
    if not isinstance(server, dict):
        return server
    return {k: v for k, v in server.items() if k not in PER_RUN}


def _at(items: list[Any], index: Any) -> Any:  # noqa: ANN401
    """``items[index]`` when ``index`` names one, else None."""
    return items[index] if isinstance(index, int) and 0 <= index < len(items) else None


def _side_view(leaf_server: Any, others: list[Any], value: Any, q: Any) -> dict[str, Any]:  # noqa: ANN401
    """The servers a side plays on, as `playGame`'s context and the config's `value` and `q` hand
    them over: the leaf server whole but the keys of the run (its `netScoresEnds` and `encoding`
    are read by the replacement and self-switch nodes, `replacenode.rs` and `selfswitch.rs`), the
    context's other servers (those the side's leaf does not name), the leaf's buffer (`value`) and
    the Q's server (`q`), each whole but the keys of the run."""
    return {
        "context.servers[leaf]": _server(leaf_server),
        "context.servers (not the leaf)": [_server(s) for s in others],
        "value": _server(value),
        "q": _server(q),
    }


def generation_view(generation: dict[str, Any]) -> dict[str, Any]:
    """Generation's config without the keys compared nowhere: its node's servers as an agent sees
    them, per side the servers its context hands `playGame` with its `value` and `q`, and the rest
    of its context but `hidden`, `leaf` and `servers` (`ALLOWED_DIFFERENT`; the servers are in the
    sides)."""
    view = {k: v for k, v in generation.items()
            if k not in ALLOWED_DIFFERENT and k not in ("context", "value", "q")}
    context = generation.get("context") or {}
    view["context"] = {k: v for k, v in context.items() if k not in ("hidden", "leaf", "servers")}
    servers, leaves = context.get("servers") or [], context.get("leaf") or []
    for side in (0, 1):
        leaf = _at(leaves, side)
        others = [s for i, s in enumerate(servers) if i != leaf]
        view[f"side {side}"] = _side_view(_at(servers, leaf), others, generation.get("value"),
                                          generation.get("q"))
    node = dict(view.get("node") or {})
    for key in ("server", "qServer"):
        if key in node:
            node[key] = _server(node[key])
    view["node"] = node
    return view


def board_view(board: dict[str, Any], k: int) -> dict[str, Any]:
    """Board arm ``k`` in generation's shape: the board's keys without those compared nowhere, per
    side the servers `boardworker.rs` hands `playGame` (the config's servers, the arm's leaf; the
    leaf's buffer as `value`, the arm's Q server as `q`), the arm's node filled as the worker fills
    it (its leaf's and Q's servers, its width in both seats; the leaf server without
    `netScoresEnds` and `encoding`, which the worker drops), and the arm's selection store and tag.
    The board has no context of its own beyond the servers: its rest is empty."""
    view = {key: v for key, v in board.items() if key not in ALLOWED_DIFFERENT}
    view["context"] = {}
    arm = board["arms"][k]
    servers, qs = board.get("servers") or [], board.get("qServers") or []
    leaf, q = arm.get("leaf"), arm.get("q")
    leaf_server, q_server = _at(servers, leaf), _at(qs, q)
    value = None if not isinstance(leaf_server, dict) else {
        "model": leaf_server.get("model"), "bytes": leaf_server.get("bytes")}
    # The arm plays the same servers in either seat.
    side = _side_view(leaf_server, [s for i, s in enumerate(servers) if i != leaf], value, q_server)
    view["side 0"] = side
    view["side 1"] = side
    node = dict(arm.get("node") or {})
    server = _server(leaf_server)
    if isinstance(server, dict):
        # `boardworker.rs` drops these two from the leaf server it puts into the node.
        server = {key: v for key, v in server.items() if key not in ("netScoresEnds", "encoding")}
    node["server"] = server
    node["qServer"] = _server(q_server)
    node["widths"] = [arm.get("limit"), arm.get("limit")]
    if arm.get("binary") is not None:
        node["binary"] = arm["binary"]
    view["node"] = node
    for key in ("store", "tag", "model"):
        view[key] = arm.get(key, "<missing>")
    unread = set(arm) - {"name", "leaf", "q", "limit", "binary", "node", "store", "tag", "model"}
    if unread:
        view["unread arm keys"] = sorted(unread)
    return view


def _diff(a: Any, b: Any, path: str = "") -> list[str]:  # noqa: ANN401
    """The paths where ``a`` and ``b`` differ (a key missing on one side is a difference)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}.{key} (only in {'generation' if key in a else 'the board'})")
            else:
                out += _diff(a[key], b[key], f"{path}.{key}")
        return out
    return [] if a == b else [path or "(top)"]


def agent_differences(generation: dict[str, Any], board: dict[str, Any]) -> list[str]:
    """Where a board arm built with generation's flags plays another agent than generation, and
    where a key either config needs is missing. Empty when both arms play generation's agent."""
    problems: list[str] = []
    for key in ("node", "exactBudget", "maxTurns", "notes", "seed", "regulation", "pool"):
        if key not in generation:
            problems.append(f"generation's config has no {key}")
        if key not in board and key != "node":
            problems.append(f"the board's config has no {key}")
    if board.get("epsilon") != 0:
        problems.append(f"the board's epsilon is {board.get('epsilon')!r}: a board plays the "
                        "equilibrium (0)")
    arms = board.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        problems.append(f"the board's config has {0 if not isinstance(arms, list) else len(arms)} "
                        "arms, not 2")
        return problems
    mine = generation_view(generation)
    for k in range(2):
        for where in _diff(mine, board_view(board, k)):
            problems.append(f"arm {k} differs from generation at {where}")
    return problems


def rust_node_writes(texts: dict[str, str] | None = None) -> list[str]:
    """Node keys the Rust workers write that are not in `RUST_NODE_FILLS`, as problems. A marker,
    not a proof: it reads the source text, so a write through an alias, `get_mut`, `pointer_mut` or
    a node built elsewhere is found only where it names `node` or `nodes` as below. ``texts``
    stands in for the sources (a control); None reads worker.rs and boardworker.rs."""
    if texts is None:
        texts = {name: (ROOT / "rust" / "src" / name).read_bytes().decode("utf-8")
                 for name in ("worker.rs", "boardworker.rs")}
    problems = []
    write = r'\bnode\s*\[\s*"(\w+)"\s*\](?:\s*\[[^\]]*\])*\s*=(?!=)'
    for name, text in texts.items():
        for key in sorted(set(re.findall(write, text))
                          | set(re.findall(r'&mut\s+node\s*\[\s*"(\w+)"\s*\]', text))):
            if key not in RUST_NODE_FILLS:
                problems.append(f"rust/src/{name} writes node[{key!r}], not one of "
                                f"{sorted(RUST_NODE_FILLS)}")
        unreadable = (r'\bnode\s*\.\s*(as_object_mut|get_mut|pointer_mut)\b'
                      r'|\bnode\s*\[\s*[^\]"\s][^\]]*\](?:\s*\[[^\]]*\])*\s*=(?!=)'
                      r'|\bnodes\s*\[[^\]]*\](?:\s*\[[^\]]*\])+\s*=(?!=)')
        if re.search(unreadable, text):
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
