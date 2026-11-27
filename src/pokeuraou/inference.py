"""The client of the inference server: workers hold no models.

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

import json
import socket
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


class ServerOutOfMemory(RuntimeError):
    """The inference server ran out of CUDA memory answering this worker's request.

    IKA-336. Since IKA-334 a server caps its allocator (`--cuda-memory-gb`), so a full card
    is an OOM instead of a GPU reset -- and a worker that got it as a generic "inference
    failed" died looking like any other crash, while the queue replayed its games and the
    run's clock went quietly wrong. The name says what happened and what to change.
    """


def is_out_of_memory(error: BaseException) -> bool:
    """torch's `OutOfMemoryError` (CUDA), or an older torch's RuntimeError saying so."""
    return type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()


def request_failed(reply: dict[str, Any], what: str) -> RuntimeError:
    """The exception for a server's `ok: false` reply: `ServerOutOfMemory` for an OOM."""
    if reply.get("oom"):
        cap = reply.get("capGb")
        return ServerOutOfMemory(
            f"{what}: the inference server ran out of CUDA memory "
            f"(its cap: {f'{cap:g} GB' if cap else 'none'}, --cuda-memory-gb, IKA-334) -- "
            f"{reply.get('error')}"
        )
    return RuntimeError(f"{what}: {reply.get('error')}")


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



def start_server(
    arms: dict[str, Sequence[Path | str]],
    *,
    q_arms: dict[str, Path | str] | None = None,
    regulation: str,
    log: Path,
    env: dict[str, str] | None = None,
) -> tuple[Any, str]:
    """Start the inference server as a child process: (the process, its address).

    It is `pokeuraou-damage inference-server` (`rustnet.server_command`: the nets exported as
    `.rnet` files beside ``log``, run on the card by cudarc, IKA-461). It needs a CUDA card; the
    Python server (`tools/inference_server.py`) is gone (IKA-460). For a tool that wants one
    server for its own processes (`time_match --served`). The address is
    the server's first line of stdout; its log is ``log``. Stop it with ``process.terminate()``.
    """
    import subprocess

    from .rustnet import server_command

    command = server_command(
        regulation, {name: list(files) for name, files in arms.items()},
        {name: file for name, file in (q_arms or {}).items()}, log.parent,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    errors = log.open("w", encoding="utf-8")
    process = subprocess.Popen(  # noqa: S603
        command, env=env, stdout=subprocess.PIPE, stderr=errors, text=True,
    )
    assert process.stdout is not None
    address = process.stdout.readline().strip()
    if not address:
        process.terminate()
        raise SystemExit(f"the inference server exited before naming an address; see {log}")
    return process, address


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
            raise request_failed(reply, "inference failed")
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
            raise request_failed(reply, "inference failed")
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
