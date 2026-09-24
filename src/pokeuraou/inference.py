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
import socket
import socketserver
import threading
import time
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
    batch_size: int = CHUNK_ROWS

    def __post_init__(self) -> None:
        self.evaluated = 0
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
        self.evaluated += rows
        timing.count("leaves", rows)
        # One round trip is one pass on the server; a longer batch came through here in pieces.
        timing.count("forward.passes")
        return scores


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
    for row in joined:
        timing.repeat("leafrow", hashlib.blake2b(row.tobytes(), digest_size=16).digest())


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

    score.waited = 0.0
    score.held = 0.0
    score.calls = 0
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
