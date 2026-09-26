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
exact by padding every batch to a fixed length, which costs 1.13x of a forward pass -- and
that trade is worth restating, because the share it was weighed against has moved. The
figure here was 7.7% of a worker; measured on 2026-09-20 over a 300-seat-game served
match, 24 workers over two servers (`tools/profile_stages.py match`), a worker spent
34.2% of its wall clock waiting on this server, and 94% of that wait was the server
computing rather than queueing. So padding cost about 4.4% of a worker, not 1%.

**That 34.2% was taken before IKA-52 and IKA-53 landed the same afternoon**, and those
two remove most of what made it large: a third of that match was the refused-cell tail,
where every refused cell asked this server to score a handful of rows on its own. In
generation the forward passes went 10,930 to 776. **The match has not been re-measured**,
so the honest reading of 34.2% is "an upper bound from before the tail was batched", and
the padding cost that follows from it is an upper bound too. The
decision stands on the other leg anyway -- merging makes an answer depend on who else
happened to ask -- and the memory this was built for does not depend on merging at all.
A request goes through as it arrived, with the row count it arrived with, and the answers
are the answers this project was already getting.

That last property is what makes the server checkable rather than plausible: the same
games, played through the server, must produce the same decisions.

The arrays go through shared memory and only a control line goes through the socket.
Generation moves about 900 MB a second of encoded leaves, which is not a thing to put down
a socket, and this project has already lost a worker to a 64 MB pipe write.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import socketserver
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

from . import timing

#: Environment variable carrying "host:port" to a worker.
ENV_SERVER = "POKEURAOU_INFERENCE"

#: Arrays an `Encoded` carries, in the order they are laid into the buffer. `Encoded` also
#: carries `unknown_volatiles`, which is a dict of diagnostics the model never sees.
ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")

#: Rows in one request, matching `BatchedValue.batch_size`. Not a tunable: a CUDA answer
#: depends on how many rows are in the call, so chunking anywhere other than where
#: `BatchedValue` chunks would be a different leaf wearing the same name.
CHUNK_ROWS = 8192

#: Default shared buffer, per worker. One chunk of 8,192 rows is about 30 MB at 3.7 KB a
#: row, so this is comfortable rather than tight, and 14 workers hold 0.9 GB of it. It was
#: 128 MB while a request could be a whole node -- 21,520 rows was observed and 1,048,576
#: was survived -- and requests are now chunks.
BUFFER_BYTES = 64 * 1024 * 1024


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


def _rows(blocks: Sequence[Any]) -> int:
    return sum(int(len(block.species)) for block in blocks)


def _plan_blocks(blocks: Sequence[Any]) -> tuple[list[dict[str, Any]] | None, int]:
    """`_plan` of the blocks one after another, or None if their arrays do not stack.

    Each array is the blocks' rows end to end, as `np.concatenate` would lay them, so the
    server reads block k as rows `[start_k, start_k + n_k)` of every array.
    """
    rows = _rows(blocks)
    layout: list[dict[str, Any]] = []
    offset = 0
    for name in ARRAYS:
        first = np.asarray(getattr(blocks[0], name))
        for block in blocks[1:]:
            other = np.asarray(getattr(block, name))
            if other.dtype != first.dtype or other.shape[1:] != first.shape[1:]:
                return None, 0
        offset = (offset + 63) & ~63
        shape = [rows, *first.shape[1:]]
        nbytes = int(first.dtype.itemsize * int(np.prod(shape)))
        layout.append({
            "name": name,
            "offset": offset,
            "shape": shape,
            "dtype": first.dtype.str,
            "nbytes": nbytes,
        })
        offset += nbytes
    return layout, offset


def _slice(encoded: Any, start: int, stop: int) -> Any:
    """Rows `start:stop` of an encoded batch, as an `Encoded`."""
    from .encode import Encoded

    decided = getattr(encoded, "decided", None)
    return Encoded(
        **{name: getattr(encoded, name)[start:stop] for name in ARRAYS},
        unknown_volatiles={},
        decided=None if decided is None else decided[start:stop],
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
        if request.get("op") in ("q", "describe_q"):
            return _serve_q(server, request, attached)
        if request.get("op") not in ("score", "score_segments"):
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

        if request["op"] == "score_segments":
            # IKA-291: several blocks in one round trip, each scored as its own request
            # would have been -- the same rows in a call of the same size. A model from
            # `served_model` scores a small block by replaying a CUDA graph (`block`);
            # anything else (a test's stand-in) is called once per block.
            model = models[name]
            one = getattr(model, "block", None) or model
            parts: list[np.ndarray] = []
            at = 0
            for size in request["segments"]:
                size = int(size)
                block = {key: value[at : at + size] for key, value in arrays.items()}
                parts.append(np.asarray(one(block, size), dtype=np.float64))
                server.note_request(size)  # type: ignore[attr-defined]
                at += size
            if at != rows:
                raise ValueError(f"segments add up to {at} rows, the request says {rows}")
            scores = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
            out = int(request["result_offset"])
            buffer[out : out + scores.nbytes] = scores.tobytes()
            return {"ok": True, "rows": int(scores.shape[0])}

        # IKA-107: a lone request is scored the way one block of `score_segments` is -- a
        # CUDA graph's replay when it is small enough (`block`), the eager pass otherwise.
        # Depth 1 sends only these, and four in five of them are 512 rows or fewer (600
        # M-C games: 80.0%, 42.6% at 8 or fewer), each paying the eager pass's launches.
        # Same rows, same count, same answer to the bit.
        model = models[name]
        scores = (getattr(model, "block", None) or model)(arrays, rows)

        out = int(request["result_offset"])
        buffer[out : out + scores.nbytes] = scores.tobytes()
        server.note_request(rows)  # type: ignore[attr-defined]
        return {"ok": True, "rows": int(scores.shape[0])}


def _serve_q(server: Any, request: dict[str, Any], attached: dict) -> dict[str, Any]:  # noqa: ANN401
    """A Q arm's two ops (IKA-274, `qrank`): what it is, and one position's matrix."""
    name = request["model"]
    q_models = getattr(server, "q_models", {})
    if name not in q_models:
        raise KeyError(f"no Q arm named {name!r}; have {sorted(q_models)}")
    model = q_models[name]
    if request["op"] == "describe_q":
        return {"ok": True, "files": list(model.files), "fingerprint": model.fingerprint,
                "properties": bool(model.properties)}
    handle = request["shm"]
    if handle not in attached:
        attached[handle] = shared_memory.SharedMemory(name=handle)
    buffer = attached[handle].buf
    matrix = model(_views(buffer, request["layout"]))
    if list(matrix.shape) != [int(n) for n in request["shape"]]:
        raise ValueError(f"Q answered {matrix.shape}, asked {request['shape']}")
    out = int(request["result_offset"])
    buffer[out : out + matrix.nbytes] = np.ascontiguousarray(matrix, dtype=np.float64).tobytes()
    return {"ok": True}


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, models: dict, arms: dict | None = None) -> None:
        super().__init__(address, handler)
        self.models = models
        #: Q arms (IKA-274, `qrank.served_q`), apart from the value arms: they answer
        #: another op and take no part in the value arms' reports.
        self.q_models: dict = {}
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
    q_models: dict | None = None,
) -> tuple[_Server, str]:
    """Starts the server. `models` maps a name to a callable (arrays, rows) -> scores.

    ``arms`` maps the same names to the model files behind them, which `describe` hands
    back so a caller can record what it is really playing instead of what it was told.
    ``q_models`` are Q arms (`qrank.load_q_arms`), asked by `qrank.RemoteQ`.
    """
    server = _Server((host, port), _Handler, models, arms)
    server.q_models = dict(q_models or {})
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
    batch_size: int = CHUNK_ROWS

    def __post_init__(self) -> None:
        self.evaluated = 0
        #: Ended positions among them (IKA-253), as `BatchedValue.ended`.
        self.ended = 0
        #: Seconds, split so the server's own report can be subtracted from them. `copied`
        #: is laying the arrays into the shared block; `waited` is from sending the control
        #: line to having the reply; `calls` counts requests, not batches, so a long batch
        #: cut into chunks counts once per chunk.
        self.copied = 0.0
        self.waited = 0.0
        self.calls = 0
        self._block = shared_memory.SharedMemory(create=True, size=self.buffer_bytes)
        host, _, port = self.address.rpartition(":")
        self._sock = socket.create_connection((host, int(port)))
        self._file = self._sock.makefile("rwb")
        # The report's `served` comes from here: this process's leaf is a server's.
        timing.serving()

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

    @timing.timed("forward")
    def __call__(self, positions: list[Any]) -> np.ndarray:
        if not positions:
            return np.zeros(0, dtype=np.float64)
        as_json = isinstance(positions[0], dict)
        out = np.empty(len(positions), dtype=np.float64)
        # A chunk at a time, exactly as `BatchedValue.__call__` does -- and encoding comes
        # *after* the cut, not before it. Encoding the whole list first is what killed a
        # server with no message in its log: a self-switch node makes 1,048,576 leaves,
        # which is 3.9 GB of arrays in one go, where the local path never holds more than
        # a chunk. Chunking `from_encoded` alone fixed the half that sends and left the
        # half that builds, and the test that covered it asserted the answers rather than
        # the cost, so it passed.
        for start in range(0, len(positions), self.batch_size):
            chunk = positions[start : start + self.batch_size]
            encoded = (
                self.encoder.encode(chunk)
                if as_json
                else self.encoder.encode_positions(chunk)
            )
            out[start : start + len(chunk)] = self.from_encoded(encoded)
        return out

    @timing.timed("forward")
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
        if timing.DUPES:
            note_repeated_rows(encoded, rows)
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
        before_copy = time.perf_counter()
        with timing.stage("serve.copy"):
            for item in layout:
                array = np.ascontiguousarray(getattr(encoded, item["name"]))
                start = int(item["offset"])
                # Through a view, not `tobytes()`: the latter builds the array again and
                # the assignment then copies that, so the batch crosses memory twice. At
                # the production mean of 2,104 rows it is 1.028 ms against 0.196, and the
                # forward pass those rows are going to is 2.651.
                target = np.frombuffer(
                    view, dtype=array.dtype, count=array.size, offset=start
                ).reshape(array.shape)
                np.copyto(target, array)
        self.copied += time.perf_counter() - before_copy

        sent = time.perf_counter()
        with timing.stage("serve.wait"):
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
        self.waited += time.perf_counter() - sent
        self.calls += 1
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"inference failed: {reply.get('error')}")
        scores = np.frombuffer(
            view[result_offset : result_offset + rows * 8], dtype=np.float64
        ).copy()
        # The server scores the arrays and never hears which rows ended; the result goes on
        # here, as `BatchedValue.from_encoded` puts it on (IKA-253).
        from .encode import settle

        self.ended += settle(scores, encoded, self.encoder.rules)
        self.evaluated += rows
        timing.count("leaves", rows)
        # One round trip is one pass on the server; a longer batch came through here in pieces.
        timing.count("forward.passes")
        timing.count("serve.requests")
        return scores

    @timing.timed("forward")
    def from_encoded_segments(self, segments: Sequence[Any]) -> list[np.ndarray]:
        """Score several blocks in one round trip, each as `from_encoded` would alone.

        IKA-291. The server runs each block as its own forward pass at its own size -- the
        only way a row gets the answer `from_encoded` gives it, since a CUDA answer depends
        on the number of rows in the call -- so what this saves is the round trips, not the
        passes. Blocks are laid one after another in the shared buffer and sent together
        while they fit; a block longer than `batch_size` goes through `from_encoded`, which
        cuts it where `BatchedValue` does.
        """
        outs: list[np.ndarray | None] = [None] * len(segments)
        group: list[int] = []

        def send() -> None:
            if group:
                scored = self._score_together([segments[i] for i in group])
                for index, scores in zip(group, scored, strict=True):
                    outs[index] = scores
                group.clear()

        for index, encoded in enumerate(segments):
            rows = int(len(encoded.species))
            if rows == 0:
                outs[index] = np.zeros(0, dtype=np.float64)
                continue
            if rows > self.batch_size:
                send()
                outs[index] = self.from_encoded(encoded)
                continue
            if group and not self._fits([segments[i] for i in (*group, index)]):
                send()
            group.append(index)
        send()
        return [out if out is not None else np.zeros(0, dtype=np.float64) for out in outs]

    def _fits(self, blocks: Sequence[Any]) -> bool:
        layout, used = _plan_blocks(blocks)
        return layout is not None and ((used + 63) & ~63) + 8 * _rows(blocks) <= self.buffer_bytes

    def _score_together(self, blocks: Sequence[Any]) -> list[np.ndarray]:
        layout, used = _plan_blocks(blocks)
        if layout is None or len(blocks) == 1:
            # Blocks whose arrays do not stack (a dtype or a width differs), or only one.
            return [self.from_encoded(block) for block in blocks]
        rows = _rows(blocks)
        if timing.DUPES:
            for block in blocks:
                note_repeated_rows(block, len(block.species))
        result_offset = (used + 63) & ~63
        needed = result_offset + rows * 8
        if needed > self.buffer_bytes:
            return [self.from_encoded(block) for block in blocks]
        view = self._block.buf
        before_copy = time.perf_counter()
        with timing.stage("serve.copy"):
            for item in layout:
                at = int(item["offset"])
                for block in blocks:
                    array = np.ascontiguousarray(getattr(block, item["name"]))
                    target = np.frombuffer(
                        view, dtype=array.dtype, count=array.size, offset=at
                    ).reshape(array.shape)
                    np.copyto(target, array)
                    at += int(array.nbytes)
        self.copied += time.perf_counter() - before_copy

        sent = time.perf_counter()
        with timing.stage("serve.wait"):
            self._file.write(
                (json.dumps({
                    "op": "score_segments",
                    "model": self.model,
                    "shm": self._block.name,
                    "rows": rows,
                    "segments": [int(len(block.species)) for block in blocks],
                    "layout": layout,
                    "result_offset": result_offset,
                }) + "\n").encode("utf-8")
            )
            self._file.flush()
            line = self._file.readline()
        if not line:
            raise RuntimeError("the inference server closed the connection")
        self.waited += time.perf_counter() - sent
        self.calls += 1
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"inference failed: {reply.get('error')}")
        scores = np.frombuffer(
            view[result_offset : result_offset + rows * 8], dtype=np.float64
        ).copy()
        self.evaluated += rows
        timing.count("leaves", rows)
        # One pass per block on the server, in one round trip.
        timing.count("forward.passes", len(blocks))
        timing.count("serve.requests")
        from .encode import settle

        out: list[np.ndarray] = []
        at = 0
        for block in blocks:
            n = int(len(block.species))
            values = scores[at : at + n].copy()
            # As `from_encoded`: the server never hears which rows ended (IKA-253).
            self.ended += settle(values, block, self.encoder.rules)
            out.append(values)
            at += n
        return out


def note_repeated_rows(encoded: Any, rows: int) -> None:  # noqa: ANN401 - Encoded
    """IKA-258: count leaf rows sent to the net that this decision already sent.

    A row is every array's bytes for that leaf, so two rows are the same only if the net
    would be handed the same input. Counted as `dup.leafrow.*` (`timing.repeat`); only when
    `timing.DUPES` is on.
    """
    import hashlib

    flat = [
        np.ascontiguousarray(getattr(encoded, name)).reshape(rows, -1).view(np.uint8)
        for name in ARRAYS
    ]
    joined = np.concatenate(flat, axis=1)
    where = timing.caller(4)
    inside: set[bytes] = set()
    for row in joined:
        digest = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
        timing.repeat_where("leafrow", digest, where=where)
        # IKA-264: and a row repeated inside this one request, which is what
        # merging before sending can take without touching another request.
        timing.count("dup.leafrow_request.calls")
        if digest in inside:
            timing.count("dup.leafrow_request.repeat")
        inside.add(digest)


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
    from .value import BatchedValue

    # One `BatchedValue` per serving thread, not one lock around a shared one.
    #
    # The lock came first, because an ensemble goes through `torch.func.functional_call`,
    # which swaps a module's parameters in place and cannot have two threads inside it.
    # It was kept on a measurement saying the alternative was worth 4% of a forward pass.
    # That measurement was taken where the lock was not the bottleneck. Here it was:
    #
    #    6 workers    7.44 ms waiting for the lock,  5.63 ms holding it
    #   14 workers   32.04 ms waiting for the lock,  6.36 ms holding it
    #
    # Holding barely moves while waiting grows four-fold, which is a saturated lock and
    # not a slow one -- and it is why throughput *fell* as workers were added, 0.71x at
    # six and 0.66x at fourteen against a direct run.
    #
    # The 7.7% in that last line was 2026-09-19's figure. A served match on 2026-09-20
    # put it at 34.2% of a worker, and that reading is itself dated: it was taken before
    # IKA-52 batched the refused cells, which were a third of that match. The match has
    # not been re-measured. None of this changes the reading here either way -- the lock
    # was saturated, not slow, and any share above a few percent makes that worse.
    #
    # Each thread's copy stacks the same parameter tensors and deep-copies its own base,
    # so the arithmetic is identical and only the module being reparametrised is private.
    # The base must be copied from a module `functional_call` has never touched: one that
    # has holds a BatchedTensor, and copying it fails with "Cannot access storage of
    # BatchedTensorImpl". The nets here are the real ones, so they qualify.
    local = threading.local()

    def mine() -> Any:
        instance = getattr(local, "value", None)
        if instance is None:
            instance = BatchedValue(
                value.nets if len(value.nets) > 1 else value.nets[0],
                value.encoder,
                device=value.device,
                batch_size=value.batch_size,
            )
            local.value = instance
        return instance

    def score(arrays: dict[str, np.ndarray], rows: int) -> np.ndarray:
        # The arrays are already exactly `rows` long: the client's layout was planned from
        # the batch it is asking about, and `_views` reshapes to that plan.
        encoded = Encoded(**{name: arrays[name] for name in ARRAYS}, unknown_volatiles={})
        arrived = time.perf_counter()
        instance = mine()
        entered = time.perf_counter()
        out = instance.from_encoded(encoded)
        done = time.perf_counter()
        # Kept for the same reason they were added: they are how anyone knows whether the
        # serving side is queueing. `waited` is now only the cost of finding this thread's
        # copy, so it should sit near zero and its growing again would mean something new.
        score.waited += entered - arrived
        score.held += done - entered
        score.calls += 1
        return out

    graphs = _Graphs(value) if _Graphs.usable(value) else None

    def block(arrays: dict[str, np.ndarray], rows: int) -> np.ndarray:
        """One block of a `score_segments` request (IKA-291), or a lone `score` request
        (IKA-107): a CUDA graph's replay when one fits, otherwise `score`. The two are the
        same answer to the bit."""
        if graphs is None:
            return score(arrays, rows)
        started = time.perf_counter()
        out = graphs.score(arrays, rows)
        if out is None:
            return score(arrays, rows)
        score.held += time.perf_counter() - started
        score.calls += 1
        score.replays += 1
        return out

    score.waited = 0.0
    score.held = 0.0
    score.calls = 0
    score.replays = 0
    score.block = block
    score.graphs = graphs
    return score


#: The largest request or block `_Graphs` replays; longer ones take the eager road. Depth
#: 2's sub-games are a few hundred rows (57-395 in the turn-1 probe of IKA-291) and a node's
#: own call is thousands, so this takes the first and leaves the second. 0 turns the graphs
#: off.
#:
#: Depth 1 (IKA-107, 600 M-C generation games at width 12, hidden bench): 80.0% of the
#: requests are 512 rows or fewer (82.4% with w5), over 511 sizes; 42.6% are 8 or fewer.
#: The next band, 513-1,024, is 6.8% (9.2%) of the requests but another ~500 sizes, each
#: met a few times per 600 games -- about 1.5% of the server's CPU for another 0.33 GB of
#: graphs per arm. A replay saves 0.86 ms of CPU over eager on the single net and 1.66 ms
#: on a two-net ensemble; a capture costs about one eager pass.
GRAPH_ROWS = int(os.environ.get("POKEURAOU_GRAPH_ROWS", "512"))

#: Graphs kept per arm (and index dtype), least recently replayed dropped first. Each holds
#: about 0.65 MB of host memory on one net and 0.86 MB on a two-net ensemble (IKA-107: 511
#: of them +332 / +442 MB; IKA-291 measured +361 MB for 512), and next to nothing on the
#: card (the pool is shared: +27 / +50 MB for all 511). At GRAPH_ROWS = 512 there are at
#: most 512 sizes a kind, so this cache never evicts and bounds nothing -- it is the
#: memory bound for a larger GRAPH_ROWS. 600 depth-1 games captured 1,020 graphs over
#: two servers, and none was evicted. IKA-291 at depth 2: 48 games met 860 sizes over two
#: servers, and a cache of 256 captured 1,804 times for 1,292 evictions -- 0.546 CPU s a
#: game of server against 0.426 at 512 (0.583 at 128), for 0.2 GB of peak memory.
GRAPH_CACHE = int(os.environ.get("POKEURAOU_GRAPH_CACHE", "512"))

#: The one lock every arm's `_Graphs` takes (IKA-306). torch allows one capture at a time
#: in a process: `torch.cuda.graph` captures on one process-wide side stream, synchronizes
#: the device on entry, and the caching allocator records one pool at a time. With a lock
#: per arm, a leaf-vs-leaf match (`--baseline`, two arms on one server) captured on two
#: threads at once and lost two workers a run to `operation failed due to a previous error
#: during capture`; a capture stream per arm or `relaxed` mode still failed (the entry's
#: synchronize, "beginAllocateToPool: already recording"). Replays take it too, as within
#: one arm, so no replay is enqueued inside another arm's capture.
_GRAPH_LOCK = threading.Lock()


class _Graphs:
    """One arm's forward pass as CUDA graphs, one per block size, for `score_segments` and
    for a lone `score` request (IKA-107: depth 1, the shipping road).

    IKA-291. A depth-2 pass sends dozens of sub-games of a few hundred rows each, and each
    must be scored at its own size to keep its answer (a CUDA answer moves with the number
    of rows in the call). Eagerly, such a forward pass is bound by launching its kernels --
    1.66 ms for a block whose arithmetic takes a fraction of that -- so the server's CPU
    went where the forward passes went, 10.9x at depth 2 for 3.07x the rows (IKA-111).

    A graph captured for n rows replays the kernels eager chose for n rows, on the same
    inputs, so the answer is the eager answer to the bit (checked on 113 sub-game blocks of
    a turn-1 depth-2 decision and on 280 more sizes, single net and ensemble, and asserted
    in `tests/test_inference.py`); the launches become one. It cost 4x less CPU and 1.8x
    less wall per block there.

    One set per arm, shared by the serving threads under one lock: a graph writes into
    fixed buffers, so two replays cannot overlap, and a capture must not see another
    capture. The lock is the process's, not the arm's (`_GRAPH_LOCK`, IKA-306): two arms
    capturing at once broke each other. Captured on first use of a size, in `thread_local`
    mode on a side stream, so the other threads' eager passes go on meanwhile (280 captures
    beside six eager threads: no error, no answer moved). The forward pass is its own
    `BatchedValue`, as a serving thread's is, so no thread's parameters are swapped under it.
    """

    @staticmethod
    def usable(value: Any) -> bool:  # noqa: ANN401 - BatchedValue
        device = getattr(value, "device", None)
        return GRAPH_ROWS > 0 and getattr(device, "type", None) == "cuda"

    def __init__(self, value: Any) -> None:  # noqa: ANN401 - BatchedValue
        from .value import BatchedValue

        self.value = BatchedValue(
            value.nets if len(value.nets) > 1 else value.nets[0],
            value.encoder,
            device=value.device,
            batch_size=value.batch_size,
        )
        #: Shared with every other arm in the process, not this arm's own (IKA-306).
        self.lock = _GRAPH_LOCK
        #: (dtype and trailing shape of every array) -> the input buffers, the pool, and
        #: rows -> (graph, output). The port's index arrays are int32 and the encoder's
        #: int64; each kind gets its own buffers rather than a cast.
        self.kinds: dict[tuple, tuple[dict[str, Any], Any, dict[int, tuple[Any, Any]]]] = {}
        self.captured = 0
        self.evicted = 0
        self.warm: set[int] = set()
        #: Why the graphs were given up, once a capture has failed.
        self.failed: str | None = None

    def score(self, arrays: dict[str, np.ndarray], rows: int) -> np.ndarray | None:
        """The block's values, or None when it is too long for a graph (or empty)."""
        if self.failed or rows <= 0 or rows > GRAPH_ROWS or rows > self.value.batch_size:
            return None
        import torch

        kind = tuple(
            (name, np.asarray(arrays[name]).dtype.str, tuple(np.asarray(arrays[name]).shape[1:]))
            for name in ARRAYS
        )
        # Staged in page-locked memory first, so that the copies in and out are ordinary
        # stream work: everything below is ordered on the one stream every serving thread
        # issues onto, and the lock need only cover putting it there -- not the wait for
        # the card, which a lock held across would make every thread's wait.
        staged = {
            name: torch.from_numpy(np.ascontiguousarray(arrays[name])).pin_memory()
            for name in ARRAYS
        }
        result = torch.empty(rows, dtype=torch.float64, pin_memory=True)
        done = torch.cuda.Event()
        with self.lock, torch.no_grad():
            if kind not in self.kinds:
                inputs = {
                    name: torch.zeros(
                        (GRAPH_ROWS, *shape), dtype=torch.from_numpy(np.zeros(0, dtype)).dtype,
                        device=self.value.device,
                    )
                    for name, dtype, shape in kind
                }
                self.kinds[kind] = (inputs, torch.cuda.graph_pool_handle(), OrderedDict())
            inputs, pool, graphs = self.kinds[kind]
            if rows in graphs:
                graphs.move_to_end(rows)
            else:
                if len(graphs) >= GRAPH_CACHE:
                    # Least recently replayed first. A graph holds about 0.65 MB of host
                    # memory, and 512 of them in each of two servers put a 600-game run
                    # under the machine's memory floor.
                    graphs.popitem(last=False)
                    self.evicted += 1
                    timing.count("graph.evicted")
                try:
                    graphs[rows] = self._capture(
                        {name: t[:rows] for name, t in inputs.items()}, pool
                    )
                except Exception as error:  # noqa: BLE001 - the eager road answers instead
                    # A capture that failed can leave its pool mid-recording, so no graph
                    # is trusted after one: every block from here takes the eager road,
                    # which is the same answer at the old cost.
                    self.failed = f"{type(error).__name__}: {error}"
                    timing.count("graph.failed")
                    return None
                self.captured += 1
                timing.count("graph.captured")
            timing.count("graph.replays")
            graph, out = graphs[rows]
            for name in ARRAYS:
                inputs[name][:rows].copy_(staged[name], non_blocking=True)
            graph.replay()
            result.copy_(out, non_blocking=True)
            done.record()
        done.synchronize()
        return result.numpy().copy()

    def _capture(self, inputs: dict[str, Any], pool: Any) -> tuple[Any, Any]:  # noqa: ANN401
        import torch

        if threading.get_ident() not in self.warm:
            # Warm-up off the default stream first, as capturing asks (cuBLAS and the
            # ensemble's mapped call set themselves up on first use, per thread: a
            # thread that had not warmed failed its capture with the others running).
            # Once a thread is enough: its later sizes are captured cold, a third of the
            # cost of a capture, and replay the eager answer all the same (tested).
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.value._mean_logit(inputs)
            torch.cuda.current_stream().wait_stream(side)
            self.warm.add(threading.get_ident())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
            # `BatchedValue.from_encoded`'s own expression, on the same rows.
            out = torch.sigmoid(self.value._mean_logit(inputs)).double()
        return graph, out


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


#: `CU_CTX_SCHED_BLOCKING_SYNC`: a host thread that waits on the GPU sleeps on an OS
#: primitive instead of polling. The runtime's `cudaDeviceScheduleBlockingSync` is the
#: same bit.
BLOCKING_SYNC = 0x04
#: The low three bits of a context's flags are its scheduling mode.
SCHEDULE_MASK = 0x07


def _cuda_driver() -> Any:
    import ctypes
    import sys

    return ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")


def wait_by_sleeping(driver: Any = None) -> int:
    """Make every serving thread sleep, not spin, while it waits for the GPU (IKA-106).

    CUDA's default is `cudaDeviceScheduleAuto`, which spins when the process has no more
    contexts than logical cores -- one context against sixteen here. The server answers
    each connection on its own thread, and the card is saturated by one of them (381
    calls/s at one thread, 403-419 at ten), so the rest are waiting; spinning, each one
    holds a core the whole time. On 2026-09-23, 240 shipping-condition games put the
    servers' CPU at 0.68 of the seconds they held a request.

    Two ways were tried in one process doing twenty 4096-square matmuls (CPU over wall:
    0.97 as it was, 0.21 with either): a blocking `torch.cuda.Event` synchronised before
    each `.cpu()`, or this. This one was kept because it covers every wait, not the one
    before the copy back. Every serving thread issues onto the same default stream, so a
    thread's host-to-device copy from pageable memory can wait behind another thread's
    kernels, and that wait spins too; an event before `.cpu()` does not reach it.

    It goes through the driver API because the flags of the primary context -- the one
    torch uses -- must be set before that context exists, and torch has no call for it.
    Call it before anything touches CUDA. It sets every device, since which one the
    server lands on is `CUDA_VISIBLE_DEVICES`'s business, and returns how many it set.
    Only the waiting changes: the same kernels run on the same inputs.
    """
    import ctypes

    cuda = driver if driver is not None else _cuda_driver()
    if (rc := cuda.cuInit(0)) != 0:
        raise RuntimeError(f"cuInit failed ({rc})")
    count = ctypes.c_int()
    if (rc := cuda.cuDeviceGetCount(ctypes.byref(count))) != 0:
        raise RuntimeError(f"cuDeviceGetCount failed ({rc})")
    for index in range(count.value):
        device = ctypes.c_int()
        if (rc := cuda.cuDeviceGet(ctypes.byref(device), index)) != 0:
            raise RuntimeError(f"cuDeviceGet({index}) failed ({rc})")
        # `cuDevicePrimaryCtxSetFlags` is a macro for `_v2` in cuda.h since CUDA 11.
        if (rc := cuda.cuDevicePrimaryCtxSetFlags_v2(device, BLOCKING_SYNC)) != 0:
            raise RuntimeError(
                f"cuDevicePrimaryCtxSetFlags({index}, blocking sync) failed ({rc}); "
                f"was CUDA already in use in this process?"
            )
    return count.value


def scheduling(index: int = 0, driver: Any = None) -> str:
    """How device `index`'s primary context waits, read back from the driver.

    For the server's startup lines: a flag set and then silently replaced would look
    exactly like one that worked, until someone measured the CPU again.
    """
    import ctypes

    cuda = driver if driver is not None else _cuda_driver()
    device = ctypes.c_int()
    flags = ctypes.c_uint()
    active = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(device), index) != 0 or cuda.cuDevicePrimaryCtxGetState(
        device, ctypes.byref(flags), ctypes.byref(active)
    ) != 0:
        return "unknown"
    mode = {0: "auto", 1: "spin", 2: "yield", BLOCKING_SYNC: "blocking sync"}.get(
        flags.value & SCHEDULE_MASK, f"flags {flags.value:#x}"
    )
    return f"{mode}{'' if active.value else ' (context not yet created)'}"


__all__ = [
    "BLOCKING_SYNC",
    "ENV_SERVER",
    "RemoteValue",
    "load_models",
    "scheduling",
    "serve",
    "served_model",
    "wait_by_sleeping",
]
