"""A leaf the port can ask: a stand-in inference server, and the `RemoteValue` on it.

The tests use it through `tests/_served.py`; `pokeuraou.workeragent` (`tools/agent_drift.py`)
builds the Rust workers' configs on it (IKA-458 stage 3.7: a tool must not import the tests).

Since IKA-460 the hidden bench's node, the replacement node and a game's steps are the port's, and
the port scores them on the inference server (`inference.RemoteValue`); a leaf held in the test's
process is `beliefnode.UnservedLeaf`. This module serves a test's stand-in from a thread of the
test's own process with the real protocol (a socket and shared memory, `inference.serve`), with
no torch and no GPU, so the search and the game's roads run in CI.

* `serving(reg, from_encoded)`: a context manager that serves a function of an `Encoded` batch and
  yields the `RemoteValue` on it (its `__call__` encodes positions and asks the server, as the
  real leaf does).
* `hp_share_model(encoder)`: the stand-in for `payoff.HP_SHARE` read off the encoding (remaining
  HP of each side's live members over each side's total, side 0's share of the two): what a
  Rust inference server's hp-share arm gives, and what the tests that took `HP_SHARE` as "a leaf"
  use. It agrees with `HP_SHARE` to the rounding of the encoder's HP scale.
"""

from __future__ import annotations

import contextlib
import json
import socketserver
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from .encode import HP_SCALE, Encoded, Encoder
from .inference import RemoteValue, is_out_of_memory
from .workqueue import SERVER_OOM_MARK

_ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def _views(buffer: memoryview, layout: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for item in layout:
        start = int(item["offset"])
        stop = start + int(item["nbytes"])
        out[item["name"]] = np.frombuffer(
            buffer[start:stop], dtype=np.dtype(item["dtype"])
        ).reshape(tuple(item["shape"]))
    return out


class _Handler(socketserver.StreamRequestHandler):
    """One connection of the stand-in server: the protocol of the Rust inference server
    (`pokeuraou-damage inference-server`) as a test needs it. The Python server this was cut
    from (IKA-460 step 2) is gone; its ``merge`` road and its CUDA graphs never were the Rust
    server's."""

    def handle(self) -> None:
        attached: dict[str, shared_memory.SharedMemory] = {}
        server = self.server
        try:
            for raw in self.rfile:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                request = json.loads(line)
                try:
                    reply = self._serve(request, attached)
                except Exception as error:  # noqa: BLE001 -- reported, never fatal
                    reply = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                    if is_out_of_memory(error):
                        reply["oom"] = True
                        reply["capGb"] = getattr(server, "memory_cap_gb", None)
                        server.note_oom(request.get("op"), reply["error"])  # type: ignore[attr-defined]
                self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (ConnectionError, OSError, json.JSONDecodeError):
            pass
        finally:
            for block in attached.values():
                with contextlib.suppress(OSError):
                    block.close()

    def _serve(self, request: dict[str, Any], attached: dict) -> dict[str, Any]:
        server = self.server
        op = request.get("op")
        if op == "ping":
            return {"ok": True, "models": sorted(server.models)}  # type: ignore[attr-defined]
        if op == "describe":
            return {
                "ok": True,
                "arms": {name: list(group) for name, group in server.arms.items()},  # type: ignore[attr-defined]
            }
        if op in ("q", "q_many", "q_batch", "describe_q"):
            return _serve_q(server, request, attached)
        if op not in ("score", "score_segments"):
            raise ValueError(f"unknown op {op!r}")
        name = request["model"]
        models = server.models  # type: ignore[attr-defined]
        if name not in models:
            raise KeyError(f"no model named {name!r}; have {sorted(models)}")
        handle = request["shm"]
        if handle not in attached:
            attached[handle] = shared_memory.SharedMemory(name=handle)
        buffer = attached[handle].buf
        arrays = _views(buffer, request["layout"])
        rows = int(request["rows"])
        model = models[name]
        if op == "score_segments":
            parts: list[np.ndarray] = []
            at = 0
            for size in request["segments"]:
                size = int(size)
                block = {key: value[at : at + size] for key, value in arrays.items()}
                parts.append(np.asarray(model(block, size), dtype=np.float64))
                at += size
            if at != rows:
                raise ValueError(f"segments add up to {at} rows, the request says {rows}")
            scores = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
        else:
            scores = model(arrays, rows)
        out = int(request["result_offset"])
        buffer[out : out + scores.nbytes] = scores.tobytes()
        return {"ok": True, "rows": int(scores.shape[0])}


def _serve_q(server: Any, request: dict[str, Any], attached: dict) -> dict[str, Any]:  # noqa: ANN401
    name = request["model"]
    q_models = getattr(server, "q_models", {})
    if name not in q_models:
        raise KeyError(f"no Q arm named {name!r}; have {sorted(q_models)}")
    model = q_models[name]
    if request["op"] == "describe_q":
        return {"ok": True, "files": list(model.files), "fingerprint": model.fingerprint,
                "properties": bool(model.properties),
                "sha256": list(getattr(model, "digests", []))}
    handle = request["shm"]
    if handle not in attached:
        attached[handle] = shared_memory.SharedMemory(name=handle)
    buffer = attached[handle].buf
    if request["op"] == "q_batch":
        items = request["items"]
        got = model.batch([_views(buffer, item["layout"]) for item in items])
        for item, matrix in zip(items, got, strict=True):
            if list(matrix.shape) != [int(n) for n in item["shape"]]:
                raise ValueError(f"Q answered {matrix.shape}, asked {item['shape']}")
            out = int(item["result_offset"])
            buffer[out : out + matrix.nbytes] = np.ascontiguousarray(
                matrix, dtype=np.float64
            ).tobytes()
        return {"ok": True}
    for item in request["items"] if request["op"] == "q_many" else [request]:
        matrix = model(_views(buffer, item["layout"]))
        if list(matrix.shape) != [int(n) for n in item["shape"]]:
            raise ValueError(f"Q answered {matrix.shape}, asked {item['shape']}")
        out = int(item["result_offset"])
        buffer[out : out + matrix.nbytes] = np.ascontiguousarray(
            matrix, dtype=np.float64
        ).tobytes()
    return {"ok": True}


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, models: dict, arms: dict | None = None) -> None:  # noqa: ANN001
        super().__init__(address, handler)
        self.models = models
        self.q_models: dict = {}
        self.arms = arms or {name: [] for name in models}
        self.oom_replies = 0
        self.memory_cap_gb: float | None = None
        self._lock = threading.Lock()

    def note_oom(self, op: Any, error: str) -> None:  # noqa: ANN401
        with self._lock:
            self.oom_replies += 1
            count = self.oom_replies
        cap = f"{self.memory_cap_gb:g} GB" if self.memory_cap_gb else "no cap"
        print(f"  {SERVER_OOM_MARK} #{count} to a {op!r} request (cuda memory cap: {cap}): "
              f"{' '.join(error.split())[:400]}", file=sys.stderr, flush=True)


def served_q(
    net: Any, device: Any, files: Sequence[str], digests: Sequence[str] = ()  # noqa: ANN401
) -> Callable[..., np.ndarray]:
    """A Q arm of the stand-in server on a torch net, eager (the Rust server's Q arm is
    `tests/test_rust_inference_server.py`'s): (arrays) -> the N0 x N1 matrix, with ``batch``."""
    from pokeuraou import qhead

    def answer(arrays: dict[str, np.ndarray]) -> np.ndarray:
        return qhead.q_matrix(net, arrays, device)

    def batch(requests: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
        return qhead.q_matrices(net, requests, device)

    answer.batch = batch  # type: ignore[attr-defined]
    answer.fingerprint = net.vocab_fingerprint  # type: ignore[attr-defined]
    answer.properties = bool(net.config.properties)  # type: ignore[attr-defined]
    answer.files = list(files)  # type: ignore[attr-defined]
    answer.digests = list(digests)  # type: ignore[attr-defined]
    return answer


def load_q_arms(paths: dict, device_name: str) -> dict[str, Any]:
    """Each named Q arm, loaded on ``device_name``, as `served_q`."""
    from pathlib import Path

    import torch

    from pokeuraou import qhead
    from pokeuraou.qrank import file_sha256

    device = torch.device(device_name)
    return {
        name: served_q(qhead.load_q(path, device_name), device, [Path(path).name],
                       [file_sha256(path)])
        for name, path in paths.items()
    }


def serve(
    models: dict,
    host: str = "127.0.0.1",
    port: int = 0,
    arms: dict | None = None,
    q_models: dict | None = None,
) -> tuple[_Server, str]:
    """Starts the stand-in server on a thread of this process. ``models`` maps a name to a callable
    ``(arrays, rows) -> scores``; ``arms`` the names to the files `describe` reports;
    ``q_models`` the Q arms (objects with ``files``, ``fingerprint``, ``properties``, a call on
    the arrays and ``batch``)."""
    server = _Server((host, port), _Handler, models, arms)
    server.q_models = dict(q_models or {})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    shown_host, shown_port = server.server_address[:2]
    return server, f"{shown_host}:{shown_port}"


def encoded_of(arrays: dict[str, np.ndarray]) -> Encoded:
    """The server's arrays as the `Encoded` batch a leaf scores."""
    return Encoded(**{name: arrays[name] for name in _ARRAYS}, unknown_volatiles={}, decided=None)


def model_of(from_encoded: Callable[[Encoded], np.ndarray]) -> Callable[[dict, int], np.ndarray]:
    """A server model (``(arrays, rows) -> values``) from a function of an `Encoded` batch."""

    def model(arrays: dict[str, np.ndarray], rows: int) -> np.ndarray:
        del rows
        return np.asarray(from_encoded(encoded_of(arrays)), dtype=np.float64)

    return model


@contextmanager
def serving(
    reg: Any,  # noqa: ANN401 - a Regulation
    from_encoded: Callable[[Encoded], np.ndarray] | None = None,
    *,
    model: Callable[[dict, int], np.ndarray] | None = None,
    encoder: Encoder | None = None,
    batch_size: int | None = None,
    buffer_bytes: int = 32 << 20,
) -> Iterator[RemoteValue]:
    """A `RemoteValue` on a server of this process that scores with ``from_encoded`` (or the
    server model ``model``)."""
    encoder = encoder or Encoder(reg)
    if model is None:
        assert from_encoded is not None
        model = model_of(from_encoded)
    shut, address = serve({"stub": model})
    extra = {} if batch_size is None else {"batch_size": batch_size}
    try:
        with RemoteValue(address, "stub", encoder, buffer_bytes=buffer_bytes, **extra) as leaf:
            yield leaf
    finally:
        shut.shutdown()
        shut.server_close()


_OPEN: list[Callable[[], None]] = []


def leaf_of(
    scores: Any,  # noqa: ANN401 - an object with ``from_encoded(Encoded) -> values``
    encoder: Encoder,
    *,
    batch_size: int | None = None,
    buffer_bytes: int = 32 << 20,
) -> RemoteValue:
    """A `RemoteValue` on a server of this process scoring with ``scores.from_encoded``. It is
    closed at the end of the test that made it (`close_all`, called by `conftest`); the scorer
    is ``leaf.scores``, so a test can read what the server was asked."""
    shut, address = serve({"stub": model_of(scores.from_encoded)})
    extra = {} if batch_size is None else {"batch_size": batch_size}
    leaf = RemoteValue(address, "stub", encoder, buffer_bytes=buffer_bytes, **extra)
    leaf.scores = scores

    def close() -> None:
        leaf.close()
        shut.shutdown()
        shut.server_close()

    _OPEN.append(close)
    return leaf


def close_all() -> None:
    """Every leaf `leaf_of` made, closed."""
    while _OPEN:
        _OPEN.pop()()


_HP_LEAF: list[RemoteValue] = []
_HP_CLOSE: list[Callable[[], None]] = []


def hp_share_leaf() -> RemoteValue:
    """One `RemoteValue` of this process on a server of `hp_share_model`: the stand-in, in the
    tests, for the Rust inference server's hp-share arm (`payoff.HP_SHARE` as a served leaf).
    Made on first use and closed when the process ends."""
    if not _HP_LEAF:
        import atexit

        from pokeuraou.regulation import load_regulation

        encoder = Encoder(load_regulation("gen9championsvgc2026regmc"))
        shut, address = serve({"stub": model_of(hp_share_model(encoder))})
        leaf = RemoteValue(address, "stub", encoder, buffer_bytes=64 << 20)

        def close() -> None:
            leaf.close()
            shut.shutdown()
            shut.server_close()

        atexit.register(close)
        _HP_LEAF.append(leaf)
    return _HP_LEAF[0]


_FROM: dict[int, tuple[Any, RemoteValue]] = {}


def served_from(owner: Any) -> RemoteValue:  # noqa: ANN401 - a leaf with `from_encoded`, `encoder`
    """A `RemoteValue` of this process on a server answering with ``owner.from_encoded`` (one per
    leaf, kept for the life of the process: a module fixture may hold the leaf for many tests)."""
    found = _FROM.get(id(owner))
    if found is None:
        import atexit

        model = model_of(owner.from_encoded)
        shut, address = serve({"stub": model})
        leaf = RemoteValue(address, "stub", owner.encoder, buffer_bytes=64 << 20)

        def close() -> None:
            leaf.close()
            shut.shutdown()
            shut.server_close()

        atexit.register(close)
        found = _FROM[id(owner)] = (owner, leaf)  # the owner is kept, so its id is not reused
    return found[1]


_STEEP_LEAF: list[RemoteValue] = []


def steep_leaf() -> RemoteValue:
    """A second served leaf that disagrees with `hp_share_leaf` (antisymmetric too): the HP
    share through a steeper curve. For the tests that play two arms with different leaves."""
    if not _STEEP_LEAF:
        import atexit

        from pokeuraou.regulation import load_regulation

        encoder = Encoder(load_regulation("gen9championsvgc2026regmc"))
        share = hp_share_model(encoder)

        def steep(encoded: Encoded) -> np.ndarray:
            s = np.clip(share(encoded), 1e-9, 1 - 1e-9)
            return 1.0 / (1.0 + np.exp(-np.log(s / (1.0 - s)) * 3.0))

        shut, address = serve({"stub": model_of(steep)})
        leaf = RemoteValue(address, "stub", encoder, buffer_bytes=64 << 20)

        def close() -> None:
            leaf.close()
            shut.shutdown()
            shut.server_close()

        atexit.register(close)
        _STEEP_LEAF.append(leaf)
    return _STEEP_LEAF[0]


def hp_share_model(encoder: Encoder) -> Callable[[Encoded], np.ndarray]:
    """`payoff.HP_SHARE` read off an encoded batch."""
    mon_k = {name: k for k, name in enumerate(encoder.mon_names)}

    def score(encoded: Encoded) -> np.ndarray:
        mon = encoded.mon.astype(np.float64)
        present = encoded.mask > 0
        maxhp = np.rint(mon[..., mon_k["maxhp_scaled"]] * HP_SCALE)
        hp = np.rint(mon[..., mon_k["hp_fraction"]] * maxhp)
        alive = present & ~(mon[..., mon_k["fainted"]] > 0)
        live = np.where(alive, hp, 0.0).sum(axis=-1)
        total = np.where(present, maxhp, 0.0).sum(axis=-1)
        remaining = np.where(total > 0, live / np.where(total > 0, total, 1.0), 0.0)
        denominator = remaining[:, 0] + remaining[:, 1]
        return np.where(denominator > 0, remaining[:, 0] / np.where(denominator > 0, denominator, 1.0), 0.5)

    return score


class Rows:
    """An encoded batch read back per mon, each array shaped (rows, 2, mons a side): the
    species id (``""`` where no mon), whether it is on the field, its HP and maximum HP."""

    def __init__(self, encoder: Encoder, encoded: Encoded, names: np.ndarray) -> None:
        mon_k = {name: k for k, name in enumerate(encoder.mon_names)}
        mon = encoded.mon.astype(np.float64)
        self.present = encoded.mask > 0
        self.species = np.where(self.present, names[encoded.species], "")
        self.active = (mon[..., mon_k["is_active"]] > 0) & self.present
        self.maxhp = np.where(self.present, np.rint(mon[..., mon_k["maxhp_scaled"]] * HP_SCALE), 0.0)
        self.hp = np.rint(mon[..., mon_k["hp_fraction"]] * self.maxhp)


class RowStub:
    """A stand-in leaf a test writes over `Rows` (``score(rows) -> values``), for a request
    the port scores on the inference server itself (IKA-463: `selectionSolve`): it has
    `from_encoded` and `encoder`, so the tests serve it (`served_from`). ``calls`` holds the
    rows of each request."""

    def __init__(self, reg: Any, score: Callable[[Rows], np.ndarray]) -> None:  # noqa: ANN401
        self.encoder = Encoder(reg)
        self.score = score
        self.calls: list[int] = []
        names = [""] * (max(self.encoder.vocab.species.values()) + 1)
        for name, index in self.encoder.vocab.species.items():
            names[index] = name
        self._names = np.asarray(names, dtype=object)

    def from_encoded(self, encoded: Encoded) -> np.ndarray:
        self.calls.append(len(encoded.species))
        return np.asarray(self.score(Rows(self.encoder, encoded, self._names)), dtype=np.float64)

    def __call__(self, positions: Sequence[Any]) -> np.ndarray:  # noqa: ANN401 - positions
        """The same values on positions, encoded here (a test that calls the leaf itself)."""
        if not positions:
            return np.zeros(0, dtype=np.float64)
        encoded = self.encoder.encode_positions(list(positions), bind=False)
        return np.asarray(self.score(Rows(self.encoder, encoded, self._names)), dtype=np.float64)


def subset(encoded: Encoded, rows: list[int]) -> Encoded:
    """Rows ``rows`` of an `Encoded`, in that order (what the Python road cut a lent fill with)."""
    return Encoded(
        **{name: getattr(encoded, name)[rows] for name in _ARRAYS},
        unknown_volatiles=dict(encoded.unknown_volatiles),
        decided=None if encoded.decided is None else encoded.decided[rows],
    )
