"""One process holds the models; the workers hold none.

A match worker costs 4.0 GB of commit and 1.5 GB of VRAM, and almost none of it is ours:
importing torch is 816 MB, numpy 491, a CUDA context 446, torch's arena about 2,000, and
everything this project loads -- regulation, priors, standings, roster, selection book --
is 27 MB between them. Sharing *our* data across eight workers would save 190 MB. The
weight is torch and the CUDA context, and those are one per process, which is why eight
match workers do not fit in 31.1 GB and 12.2 GB while eight generation workers do: a match
carries two leaves.

So the leaf moves out. A worker keeps the encoder, which is numpy, and sends the arrays it
already has; the server keeps the models. A worker becomes about 0.55 GB and holds no CUDA
context at all, and the number of workers stops being a property of the machine's memory.

**Requests are not merged across workers.** They could be -- that is the usual reason to
build one of these -- but the measurements say not to. A CUDA answer depends on the batch's
*size* (5.96e-08 between a row alone and the same row among eight; nothing from the batch's
contents or the row's position in it), so merging makes the answer depend on who else
happened to ask, and 1.9e-06 in a leaf is enough to move an equilibrium. It can be made
exact by padding every batch to a fixed length, which costs 1.13x of a forward pass -- but
the forward pass is 7.7% of a worker, so the whole prize for merging is a few percent of
one, and the memory it is being built for does not depend on merging at all. A request goes
through as it arrived, with the row count it arrived with, and the answers are the answers
this project was already getting.

That last property is what makes the server checkable rather than plausible: the same
games, played through the server, must produce the same decisions.

The arrays go through shared memory and only a control line goes through the socket.
Generation moves about 900 MB a second of encoded leaves, which is not a thing to put down
a socket, and this project has already lost a worker to a 64 MB pipe write.
"""

from __future__ import annotations

import contextlib
import json
import socket
import socketserver
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

#: Environment variable carrying "host:port" to a worker.
ENV_SERVER = "POKEURAOU_INFERENCE"

#: Arrays an `Encoded` carries, in the order they are laid into the buffer. `Encoded` also
#: carries `unknown_volatiles`, which is a dict of diagnostics the model never sees.
ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")

#: Default shared buffer, per worker. A batch is at most `BatchedValue.batch_size` rows by
#: the time the model sees it, but a *request* can be larger -- 21,520 rows was observed --
#: and at about 3.7 KB a row that is 80 MB.
BUFFER_BYTES = 128 * 1024 * 1024


def _plan(encoded: Any) -> tuple[list[dict[str, Any]], int]:
    """Where each array sits in the buffer, and how much of it is used."""
    layout: list[dict[str, Any]] = []
    offset = 0
    for name in ARRAYS:
        array = np.ascontiguousarray(getattr(encoded, name))
        # 64-byte aligned, so a view is never a copy on either side.
        offset = (offset + 63) & ~63
        layout.append({
            "name": name,
            "offset": offset,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "nbytes": int(array.nbytes),
        })
        offset += int(array.nbytes)
    return layout, offset


def _slice(encoded: Any, start: int, stop: int) -> Any:
    """Rows `start:stop` of an encoded batch, as an `Encoded`."""
    from .encode import Encoded

    return Encoded(
        **{name: getattr(encoded, name)[start:stop] for name in ARRAYS},
        unknown_volatiles={},
    )


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
    """One worker's connection. Dropping it frees that worker's buffer and nothing else."""

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
                self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (ConnectionError, OSError, json.JSONDecodeError):
            pass
        finally:
            # A worker that died takes its own buffer with it and leaves the server up.
            for block in attached.values():
                with contextlib.suppress(OSError):
                    block.close()
            server.note_departure()  # type: ignore[attr-defined]

    def _serve(self, request: dict[str, Any], attached: dict) -> dict[str, Any]:
        server = self.server
        if request.get("op") == "ping":
            return {"ok": True, "models": sorted(server.models)}  # type: ignore[attr-defined]
        if request.get("op") == "describe":
            # What an arm *is*, not what the caller was told to call it. A match stamps
            # the leaf into every game's provenance and a rating is fitted from that, so
            # a name the worker supplies is a name that can be wrong -- the policy's
            # position representation was exactly this mistake, and it was silent.
            return {
                "ok": True,
                "arms": {
                    name: list(group)
                    for name, group in server.arms.items()  # type: ignore[attr-defined]
                },
            }
        if request.get("op") != "score":
            raise ValueError(f"unknown op {request.get('op')!r}")

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

        scores = models[name](arrays, rows)

        out = int(request["result_offset"])
        buffer[out : out + scores.nbytes] = scores.tobytes()
        server.note_request(rows)  # type: ignore[attr-defined]
        return {"ok": True, "rows": int(scores.shape[0])}


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, models: dict, arms: dict | None = None) -> None:
        super().__init__(address, handler)
        self.models = models
        #: Each arm's model file names, for `describe`. Empty when the caller built the
        #: models itself (a test with a stub), which `describe` then reports honestly as
        #: empty rather than inventing something.
        self.arms = arms or {name: [] for name in models}
        self.requests_served = 0
        self.rows_served = 0
        self.connections = 0
        self._lock = threading.Lock()

    def note_request(self, rows: int) -> None:
        with self._lock:
            self.requests_served += 1
            self.rows_served += rows

    def note_departure(self) -> None:
        with self._lock:
            self.connections -= 1


def serve(
    models: dict,
    host: str = "127.0.0.1",
    port: int = 0,
    arms: dict | None = None,
) -> tuple[_Server, str]:
    """Starts the server. `models` maps a name to a callable (arrays, rows) -> scores.

    ``arms`` maps the same names to the model files behind them, which `describe` hands
    back so a caller can record what it is really playing instead of what it was told.
    """
    server = _Server((host, port), _Handler, models, arms)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    shown_host, shown_port = server.server_address[:2]
    return server, f"{shown_host}:{shown_port}"


@dataclass
class RemoteValue:
    """A leaf that lives in another process, with the interface the search already uses.

    `resolve` finds the scorer by `getattr(owner, "from_encoded")`, and everything else
    calls the object. So this offers the same three things `BatchedValue` does --
    `__call__`, `from_encoded`, and the `evaluated` count -- and nothing here imports
    torch, which is the whole point.
    """

    address: str
    model: str
    encoder: Any
    buffer_bytes: int = BUFFER_BYTES
    #: Rows per request. `BatchedValue.batch_size`, and it has to stay that: a CUDA answer
    #: depends on the number of rows in the call, so the two paths agree only while they
    #: cut a long batch in the same places.
    batch_size: int = 8192

    def __post_init__(self) -> None:
        self.evaluated = 0
        self._block = shared_memory.SharedMemory(create=True, size=self.buffer_bytes)
        host, _, port = self.address.rpartition(":")
        self._sock = socket.create_connection((host, int(port)))
        self._file = self._sock.makefile("rwb")

    def close(self) -> None:
        try:
            self._file.close()
            self._sock.close()
        finally:
            self._block.close()
            self._block.unlink()

    def describe(self) -> list[str]:
        """The model files behind this arm, as the server knows them.

        For provenance. A worker that names its leaf from its own command line can name
        it wrongly -- it is told which arm to use, not what that arm is -- and the record
        it writes is what a rating is fitted from.
        """
        self._file.write((json.dumps({"op": "describe"}) + "\n").encode("utf-8"))
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise RuntimeError("the inference server closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"describe failed: {reply.get('error')}")
        arms = reply.get("arms") or {}
        if self.model not in arms:
            raise RuntimeError(
                f"the server has no arm named {self.model!r}; it has {sorted(arms)}"
            )
        return list(arms[self.model])

    def __enter__(self) -> RemoteValue:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __call__(self, positions: list[Any]) -> np.ndarray:
        if not positions:
            return np.zeros(0, dtype=np.float64)
        as_json = isinstance(positions[0], dict)
        encoded = (
            self.encoder.encode(positions)
            if as_json
            else self.encoder.encode_positions(positions)
        )
        return self.from_encoded(encoded)

    def from_encoded(self, encoded: Any) -> np.ndarray:
        """Score a batch, in the same pieces `BatchedValue` would have scored it in.

        Chunked at `batch_size`, which is `BatchedValue.batch_size`. The first version
        refused to chunk at all, on the grounds that splitting changes a batch's size and
        a CUDA answer depends on that -- but the model behind the server chunks at 8,192
        regardless, so refusing did not preserve the answers, it only forbade the batches
        larger than one chunk. A self-switch node produced 1,048,576 leaves in a real
        match and killed the worker; the direct path had been evaluating that node in 128
        pieces all along, quietly.

        So the invariant is not "never split" but "split where `BatchedValue` splits", and
        the two paths now agree by construction on batches of any size.
        """
        rows = int(len(encoded.species))
        if rows == 0:
            return np.zeros(0, dtype=np.float64)
        if rows > self.batch_size:
            out = np.empty(rows, dtype=np.float64)
            for start in range(0, rows, self.batch_size):
                stop = min(start + self.batch_size, rows)
                out[start:stop] = self.from_encoded(_slice(encoded, start, stop))
            return out
        layout, used = _plan(encoded)
        result_offset = (used + 63) & ~63
        needed = result_offset + rows * 8
        if needed > self.buffer_bytes:
            raise RuntimeError(
                f"one chunk of {rows} rows needs {needed / 1e6:.0f} MB and the buffer "
                f"is {self.buffer_bytes / 1e6:.0f} MB. Longer batches are already cut at "
                f"batch_size ({self.batch_size}) to match BatchedValue; this is a single "
                f"chunk that does not fit, so raise buffer_bytes rather than cutting "
                f"smaller -- a CUDA answer depends on the number of rows in the call, and "
                f"the two paths agree only while they cut in the same places."
            )
        view = self._block.buf
        for item in layout:
            array = np.ascontiguousarray(getattr(encoded, item["name"]))
            start = int(item["offset"])
            view[start : start + array.nbytes] = array.tobytes()

        self._file.write(
            (json.dumps({
                "op": "score",
                "model": self.model,
                "shm": self._block.name,
                "rows": rows,
                "layout": layout,
                "result_offset": result_offset,
            }) + "\n").encode("utf-8")
        )
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise RuntimeError("the inference server closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"inference failed: {reply.get('error')}")
        scores = np.frombuffer(
            view[result_offset : result_offset + rows * 8], dtype=np.float64
        ).copy()
        self.evaluated += rows
        return scores


def served_model(value: Any):
    """The server's side of one arm: `BatchedValue`, called from another process.

    It delegates rather than reimplements, and that is the whole point rather than a
    convenience. The first version wrote its own averaging --
    `torch.stack([net(batch) for net in nets]).mean(0)` -- which is a second path to the
    same quantity and came out 4e-08 from the first, enough to move a mixed strategy in
    the fourth decimal. `value.py` says as much in its own comment and says not to keep
    two paths; the server had quietly made one. Copying carefully was not the fix, because
    the copy was already careful: same chunk boundary, same sigmoid, same dtype, and still
    different.

    A real `Encoded` is rebuilt rather than something shaped like one, so that a field
    `from_encoded` starts reading cannot go missing here. `unknown_volatiles` is
    diagnostics the model never sees.

    One request at a time. The server answers each connection on its own thread, but an
    ensemble goes through `torch.func.functional_call`, which swaps a module's parameters
    in place for the duration of a call -- two threads doing that to the same module race,
    and the loser sees the meta-device base the swap was supposed to fill in. Ten workers
    produced exactly that, five seconds in. Per-thread bases would remove the need for the
    lock and were measured at 419 calls/s against 403, which is 4% of a forward pass that
    is 7.7% of a worker; and a failure without the lock is not self-limiting, because a
    shared base left holding a BatchedTensor stays broken for every later caller.
    """
    from .encode import Encoded

    lock = threading.Lock()

    def score(arrays: dict[str, np.ndarray], rows: int) -> np.ndarray:
        # The arrays are already exactly `rows` long: the client's layout was planned from
        # the batch it is asking about, and `_views` reshapes to that plan.
        encoded = Encoded(**{name: arrays[name] for name in ARRAYS}, unknown_volatiles={})
        with lock:
            return value.from_encoded(encoded)

    return score


def load_models(paths: dict[str, Sequence[Path]], encoder: Any, device_name: str) -> dict:
    """Loads each named arm as a `BatchedValue`. Several files are an ensemble."""
    import torch

    from .value import BatchedValue, load_model

    device = torch.device(device_name)
    models: dict[str, Any] = {}
    for name, group in paths.items():
        nets = [load_model(path, encoder)[0].to(device).eval() for path in group]
        value = BatchedValue(nets if len(nets) > 1 else nets[0], encoder, device=device)
        # Averaging an ensemble is `BatchedValue`'s job, not this module's, and a checkout
        # whose `BatchedValue` silently kept only one net would serve an arm that is not
        # the arm it is named after. Asked of the object rather than of its signature,
        # because a signature check is a string match that stops working quietly.
        if len(getattr(value, "nets", nets)) != len(nets):
            raise SystemExit(
                f"arm {name!r} has {len(nets)} models and this checkout's BatchedValue "
                f"kept {len(value.nets)}. Averaging them here instead would be a second "
                f"path to a quantity that already has one, and the two differ by 4e-08."
            )
        models[name] = served_model(value)
    return models


__all__ = [
    "ENV_SERVER",
    "RemoteValue",
    "load_models",
    "serve",
    "served_model",
]
