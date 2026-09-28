"""The root menu ranked by a learned Q (IKA-274): the rank-fill labels ``q`` and ``q-nocover``.

The leaf ranking (`search.leaf_ranking`, ``refs2``) scores each candidate against two
damage replies by filling pool x 2 cells with the leaf. IKA-310/311 measured what that
misses: which combination the menu needs is decided by the opponent's *mix*. A Q
(`qhead`) predicts the whole depth-1 matrix of a position -- every legal pair of both
sides -- in one forward pass and fills no cell with the leaf, so the mix can be read off a
small game on it: solve Q over both sides' whole pools, and score each of our candidates
by its value against the opponent's half of that solve (``q-full`` in `tools/q_menus.py`).

Which form: IKA-274's yardstick put ``q-w24`` (solve Q over the width-24 menus) and
``q-full`` (the whole pools) within 0.3 of an action of each other on every recall
measure, and ``q-w24`` needs the width-24 menu that only the leaf ranking builds -- the
fill this replaces. So the search ranks by ``q-full``.

What is the same as ``refs2``: the pool (`narrow`'s own, handed in; the other side's from
`qhead.legal_pool`, the list `narrow` would rank for it), the position (the view the leaf
ranking would read, `selfplay._menus.views`), the cover rule (``q`` keeps it, ``q-nocover``
does not, as IKA-323's ``-nocover``), and where it reaches: only the root's leaf-ranked
menus. Everything that sees the matrix after that is untouched, so a Q never becomes an
output -- it only chooses which cells the leaf fills (narrow.py's principles).

Where the Q lives: in the inference server, as one more arm (`inference_server.py
--q-arm`), so a served worker still holds no torch (`RemoteQ`). A worker that loads its
leaf itself loads the Q the same way (`LocalQ`). One Q per process (`install`): both arms
of a match that name a ``q`` label rank with it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

from . import timing
from .actions import SideAction
from .position import Position
from .regulation import Regulation

#: ``q`` or ``q-nocover``: rank the leaf-ranked root menu by a Q's solve (this module).
#: A ``.NAME`` suffix (stage 3) ranks by the process's Q of that name instead of its
#: default one (`install`), so two arms of one match can each rank by their own Q:
#: ``q-nocover`` against ``q-nocover.long``.
Q_RANK_FILL = re.compile(r"q(-nocover)?(?:\.([a-z0-9_]+))?")

#: Both sides' pools of one Q request: (side 0's actions, side 1's actions).
Pools = tuple[Sequence[SideAction], Sequence[SideAction]]


class QVocabularyError(ValueError, RuntimeError):
    """A Q that reads another vocabulary than the leaf's encoder (IKA-341).

    A `ValueError` where a local Q raised one and a `RuntimeError` where a served Q did,
    so a caller that caught either still does; a person's game catches this one to fall
    back with a note (`tools/play_human.py`).
    """


def is_q(label: str) -> bool:
    """Whether a rank-fill label ranks by a Q rather than by filling cells with the leaf."""
    return Q_RANK_FILL.fullmatch(label) is not None


def _parsed(label: str) -> re.Match[str]:
    got = Q_RANK_FILL.fullmatch(label)
    if got is None:
        raise ValueError(f"{label!r} is not a q rank fill")
    return got


def q_name(label: str) -> str:
    """Which of the process's Qs a q label ranks by: its ``.NAME``, or "" (the default)."""
    return _parsed(label).group(2) or ""


def q_covers(label: str) -> bool:
    """Whether a q label keeps the cover (``q``) or not (``q-nocover``)."""
    return _parsed(label).group(1) is None


#: Per rank-fill label, what the Q ranking did in this process: rankings asked, and the
#: cells of the Q matrices behind them. The positive control that a ``q`` label reached a
#: menu, read as a change by a match's echo (as `narrow.COVERLESS`).
QRANKED: dict[str, dict[str, int]] = {}

#: The process's Qs by name: "" is the default (``q``, ``q-nocover``), any other name the
#: Q of the labels that carry it (``q-nocover.NAME``).
_INSTALLED: dict[str, Any] = {}


def install(model: Any, name: str = "") -> None:  # noqa: ANN401 - LocalQ or RemoteQ
    """Makes `model` the process's Q of `name`: every ``q`` label of that name ranks with it."""
    _INSTALLED[name] = model


def installed(name: str = "") -> Any:  # noqa: ANN401
    """The process's Q of `name`, or a stop that says how to give one."""
    if name not in _INSTALLED:
        if name:
            raise RuntimeError(
                f"a q rank fill .{name} needs the Q {name!r}: --q-model-named {name} <file> "
                f"(loaded here) or --q-arm-named {name} <arm> (on the inference server)"
            )
        raise RuntimeError(
            "a q rank fill needs a Q: --q-model <file> (loaded here) or --q-arm <name> "
            "(on the inference server, which --q-model on the launcher loads)"
        )
    return _INSTALLED[name]


def _pool_arrays(
    reg: Regulation, encoder: Any, pos: Position, pools: tuple[Sequence[SideAction], Sequence[SideAction]],  # noqa: ANN401
    properties: bool,
) -> dict[str, np.ndarray]:
    """One request's arrays: the position's encoding, both pools' actions and features."""
    from . import qhead

    enc = encoder.encode_positions([pos])
    out = {name: np.ascontiguousarray(getattr(enc, name)) for name in qhead.POSITION_ARRAYS}
    for side in (0, 1):
        out[f"acts{side}"] = qhead.encode_actions(encoder.vocab, pos, side, pools[side])
    if properties:
        got = qhead.port_features(reg, pos, pools)
        if got is None:
            # A position the port refuses to score: the features are zeros, as in training
            # (`tools/q_menus.py`), and the net still reads the position and the moves.
            got = (
                np.zeros((len(pools[0]), qhead.FEATURE_WIDTH), np.float32),
                np.zeros((len(pools[1]), qhead.FEATURE_WIDTH), np.float32),
            )
        out["feats0"], out["feats1"] = (np.ascontiguousarray(f, dtype=np.float32) for f in got)
    return out


@dataclass
class LocalQ:
    """A Q loaded in this process (torch here): for a worker that loads its own leaf."""

    path: Path
    encoder: Any
    device: str = "cpu"

    def __post_init__(self) -> None:
        import torch

        from . import qhead

        self.net = qhead.load_q(self.path, self.device)
        if self.net.vocab_fingerprint != self.encoder.vocab.fingerprint():
            raise QVocabularyError(f"{self.path} reads another vocabulary than this encoder's")
        self._device = torch.device(self.device)
        self.calls = 0
        self._digests = [file_sha256(self.path)]

    def describe(self) -> list[str]:
        return [Path(self.path).name]

    def digests(self) -> list[str]:
        """The sha256 of each file `describe` names, in its order (IKA-340)."""
        return list(self._digests)

    def matrix(
        self, reg: Regulation, pos: Position, pools: tuple[Sequence[SideAction], Sequence[SideAction]]
    ) -> np.ndarray:
        from . import qhead

        arrays = _pool_arrays(reg, self.encoder, pos, pools, self.net.config.properties)
        self.calls += 1
        return qhead.q_matrix(self.net, arrays, self._device)

    def matrices(self, reg: Regulation, asks: Sequence[tuple[Position, Pools]]) -> list[np.ndarray]:
        """`matrix` of each (position, pools), in order."""
        return [self.matrix(reg, pos, pools) for pos, pools in asks]

    def batched(self, reg: Regulation, asks: Sequence[tuple[Position, Pools]]) -> list[np.ndarray]:
        """The matrix of each (position, pools) in ONE forward pass (`qhead.q_matrices`,
        IKA-307). Not `matrices`' numbers to the bit: a batch may round otherwise."""
        from . import qhead

        requests = [
            _pool_arrays(reg, self.encoder, pos, pools, self.net.config.properties)
            for pos, pools in asks
        ]
        self.calls += 1
        return qhead.q_matrices(self.net, requests, self._device)


def _plan_named(arrays: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], int]:
    """Where each named array sits in the shared block (`inference._plan`'s layout)."""
    layout: list[dict[str, Any]] = []
    offset = 0
    for name, array in arrays.items():
        offset = (offset + 63) & ~63
        layout.append({
            "name": name, "offset": offset, "shape": list(array.shape),
            "dtype": array.dtype.str, "nbytes": int(array.nbytes),
        })
        offset += int(array.nbytes)
    return layout, offset


#: One request's shared block: a position row (~4 KB), two pools of a few hundred actions
#: and their features, and an N0 x N1 float64 answer (a 400 x 400 pool pair is 1.3 MB).
Q_BUFFER_BYTES = 8 * 1024 * 1024


@dataclass
class RemoteQ:
    """A Q that lives in the inference server (`--q-arm`); this process holds no torch.

    The same shared-memory road as `inference.RemoteValue`, its own connection and block,
    and the op ``q``: one position, both pools, one matrix back. ``q_many`` (stage 3) asks
    several in one round trip -- both sides' rankings of a menu (`prefetch`) -- and the
    server answers each as its own ``q`` would.
    """

    address: str
    model: str
    encoder: Any
    properties: bool = True
    buffer_bytes: int = Q_BUFFER_BYTES
    calls: int = field(default=0, init=False)
    trips: int = field(default=0, init=False)
    waited: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._block = shared_memory.SharedMemory(create=True, size=self.buffer_bytes)
        host, _, port = self.address.rpartition(":")
        self._sock = socket.create_connection((host, int(port)))
        self._file = self._sock.makefile("rwb")
        about = self._ask({"op": "describe_q", "model": self.model})
        if about["fingerprint"] != self.encoder.vocab.fingerprint():
            raise QVocabularyError(f"the server's Q arm {self.model!r} reads another vocabulary")
        self.properties = bool(about["properties"])
        self._files = list(about["files"])
        # The server hashed the files it loaded (IKA-340); this process never sees them.
        self._digests = list(about["sha256"])

    def _ask(self, request: dict[str, Any]) -> dict[str, Any]:
        self._file.write((json.dumps(request) + "\n").encode("utf-8"))
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise RuntimeError("the inference server closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            from .inference import request_failed

            raise request_failed(reply, "Q request failed")
        return reply

    def describe(self) -> list[str]:
        return list(self._files)

    def digests(self) -> list[str]:
        """The sha256 of each file `describe` names, as the server hashed it (IKA-340)."""
        return list(self._digests)

    def close(self) -> None:
        try:
            self._file.close()
            self._sock.close()
        finally:
            self._block.close()
            self._block.unlink()

    def matrix(
        self, reg: Regulation, pos: Position, pools: tuple[Sequence[SideAction], Sequence[SideAction]]
    ) -> np.ndarray:
        return self.matrices(reg, [(pos, pools)])[0]

    def matrices(self, reg: Regulation, asks: Sequence[tuple[Position, Pools]]) -> list[np.ndarray]:
        """`matrix` of each (position, pools), in order, in one round trip."""
        return self._send(reg, asks, "q" if len(asks) == 1 else "q_many")

    def batched(self, reg: Regulation, asks: Sequence[tuple[Position, Pools]]) -> list[np.ndarray]:
        """The matrix of each (position, pools) in one round trip and ONE forward pass on
        the server (op ``q_batch``, `qhead.q_matrices`; IKA-307)."""
        return self._send(reg, asks, "q_batch")

    def _send(
        self, reg: Regulation, asks: Sequence[tuple[Position, Pools]], op: str
    ) -> list[np.ndarray]:
        """The arrays into the shared block, one request line, the matrices back."""
        view = self._block.buf
        # IKA-339: the worker's own side of a Q request -- the actions' encoding, the
        # arrays and their copy into the block -- apart from the position's encoding and
        # the port's features, which are rows of their own inside it.
        with timing.stage("q.arrays"):
            items = self._lay(reg, asks, view)
        return self._post(op, items, view)

    def _lay(
        self, reg: Regulation, asks: Sequence[tuple[Position, Pools]], view: Any  # noqa: ANN401
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        at = 0
        for pos, pools in asks:
            arrays = _pool_arrays(reg, self.encoder, pos, pools, self.properties)
            layout, used = _plan_named(arrays)
            for item in layout:
                item["offset"] = int(item["offset"]) + at
            shape = (len(pools[0]), len(pools[1]))
            result_offset = (at + used + 63) & ~63
            at = (result_offset + shape[0] * shape[1] * 8 + 63) & ~63
            if at > self.buffer_bytes:
                raise RuntimeError(
                    f"Q requests up to {shape} do not fit {self.buffer_bytes} bytes"
                )
            for item in layout:
                array = arrays[item["name"]]
                target = np.frombuffer(
                    view, dtype=array.dtype, count=array.size, offset=int(item["offset"])
                ).reshape(array.shape)
                np.copyto(target, array)
            items.append({"layout": layout, "shape": list(shape), "result_offset": result_offset})
        return items

    def _post(self, op: str, items: list[dict[str, Any]], view: Any) -> list[np.ndarray]:  # noqa: ANN401
        sent = time.perf_counter()
        with timing.stage("serve.q"):
            if op == "q":
                self._ask({"op": "q", "model": self.model, "shm": self._block.name, **items[0]})
            else:
                self._ask({
                    "op": op, "model": self.model, "shm": self._block.name, "items": items,
                })
        self.waited += time.perf_counter() - sent
        self.calls += len(items) if op != "q_batch" else 1
        self.trips += 1
        out = []
        for item in items:
            shape = tuple(item["shape"])
            start = int(item["result_offset"])
            out.append(np.frombuffer(
                view[start : start + shape[0] * shape[1] * 8], dtype=np.float64
            ).reshape(shape).copy())
        return out


#: The longest pool `QGraphs` captures for; a request with a longer pool on either side is
#: answered by the eager pass (`qhead.q_matrix`), the same answer. The 158,392 teaching
#: views of IKA-274 met pools of 2 to 230 actions (138 sizes). 0 turns the graphs off.
Q_GRAPH_POOL = int(os.environ.get("POKEURAOU_Q_GRAPH_POOL", "256"))


class QGraphs:
    """A Q arm's forward pass as CUDA graphs, cut where the shapes allow (IKA-274 stage 3).

    `qhead.q_matrix` runs the net eagerly: about 150 kernels a request, each launched from
    the server's Python under the GIL the value arms' threads also want, and a dozen
    pageable copies in that each wait for the card's queue. Served beside 24 workers that
    was 21-44 ms a request for a few ms of work (IKA-274 §15).

    A request's shape is two pool sizes, and the pair of them is rarely met twice (6,051
    pairs among the teaching views), so one graph per pair would be mostly captures. The
    forward pass is cut instead where the shapes separate:

    * the trunk and both sides' contexts read only the position, one shape: one graph;
    * each side's actions (`QNet.encode_actions`) read the trunk's rows and that side's
      pool: one graph per (side, pool size), at most 2 x 138;
    * the pair head, which reads both sizes, stays eager (about 30 kernels).

    Each piece replays the kernels eager chose for the same shapes on the same inputs, so
    the matrix is `qhead.q_matrix`'s to the bit (asserted in `tests/test_q_rank.py` on
    CUDA; exact sizes, nothing padded). The inputs go in through page-locked staging on
    the one stream, and the whole request is enqueued under the process's graph lock
    (`inference._GRAPH_LOCK`, IKA-306: one capture at a time, and no replay inside
    another's capture) and waited for outside it, as `inference._Graphs` does.
    """

    @staticmethod
    def usable(device: Any) -> bool:  # noqa: ANN401
        return Q_GRAPH_POOL > 0 and getattr(device, "type", None) == "cuda"

    def __init__(self, net: Any, device: Any) -> None:  # noqa: ANN401
        from .inference import _GRAPH_LOCK

        self.net = net
        self.device = device
        self.lock = _GRAPH_LOCK
        self.kind: tuple | None = None
        self.inputs: dict[str, Any] = {}
        #: One memory pool per piece ("trunk", "side0", "side1"), shared by that piece's
        #: graphs of every size.
        self.pools: dict[str, Any] = {}
        self.trunk: tuple[Any, tuple[Any, Any, Any]] | None = None
        #: (side, pool size) -> (graph, that side's action vectors)
        self.sides: dict[tuple[int, int], tuple[Any, Any]] = {}
        self.captured = 0
        self.replays = 0
        #: Why the graphs were given up, once a capture has failed.
        self.failed: str | None = None

    def _capture(self, what: str, run: Callable[[], Any]) -> tuple[Any, Any]:  # noqa: ANN401
        import torch

        from .inference import GRAPH_THREAD

        # One pool per piece, never one for all. A shared pool is safe only when graphs
        # replay in the order they were captured, and these replay in a request's order
        # (trunk, side 0, side 1) whatever order their sizes were first met in: with one
        # pool, a side-1 graph captured earlier wrote its scratch over a later-captured
        # side-0 graph's output before the pair read it (seen: 23 of 150 requests off by
        # up to 0.9). Within a piece only one graph replays per request and its output is
        # read before the next, so its sizes can share. A pool per graph was right too,
        # but kept a segment per graph: two Q arms of ~180 graphs each reserved 3.4-4.6 GB
        # a server, and a 4,000-pair board lost both servers to an abort after 2 minutes.
        if what not in self.pools:
            self.pools[what] = torch.cuda.graph_pool_handle()
        # Warmed up (twice, off the default stream: cuBLAS and friends set themselves up on
        # first use) and captured on the process's one graph thread and stream (IKA-377).
        graph, out = GRAPH_THREAD.run((self, what), run, run, self.pools[what])
        self.captured += 1
        timing.count("q.graph.captured")
        return graph, out

    def _trunk(self) -> tuple[Any, Any, Any]:  # noqa: ANN401
        """(rows, side 0's context, side 1's context): `QNet.forward`'s first lines."""
        import torch

        from . import qhead

        net = self.net
        batch = {name: self.inputs[name] for name in qhead.POSITION_ARRAYS}
        h, sides = net.mons(batch)
        field = batch["field"]
        ctx0 = net.context(torch.cat([sides[:, 0], sides[:, 1], field], dim=-1))
        ctx1 = net.context(torch.cat([sides[:, 1], sides[:, 0], field], dim=-1))
        return h, ctx0, ctx1

    def matrix(self, arrays: dict[str, np.ndarray]) -> np.ndarray | None:
        """The request's N0 x N1 matrix, or None when it is not for the graphs."""
        n = (len(arrays["acts0"]), len(arrays["acts1"]))
        if self.failed or min(n) <= 0 or max(n) > Q_GRAPH_POOL:
            return None
        import torch

        from . import qhead

        names = qhead.POSITION_ARRAYS
        kind = tuple((name, arrays[name].dtype.str, tuple(arrays[name].shape)) for name in names)
        if self.kind is not None and kind != self.kind:
            return None
        props = bool(self.net.config.properties)
        # `qhead.q_matrix`'s own conversions, staged in page-locked memory.
        staged = {
            name: torch.from_numpy(np.ascontiguousarray(arrays[name])).pin_memory()
            for name in names
        }
        for s in (0, 1):
            staged[f"acts{s}"] = torch.from_numpy(
                np.ascontiguousarray(arrays[f"acts{s}"]).astype(np.int64)
            )[None].pin_memory()
            if props:
                staged[f"feats{s}"] = torch.from_numpy(
                    np.ascontiguousarray(arrays[f"feats{s}"], dtype=np.float32)
                )[None].pin_memory()
        result = torch.empty(n, dtype=torch.float64, pin_memory=True)
        done = torch.cuda.Event()
        with self.lock, torch.no_grad():
            try:
                if self.kind is None:
                    self.inputs = {
                        name: torch.zeros(shape, dtype=staged[name].dtype, device=self.device)
                        for name, _dtype, shape in kind
                    }
                    for s in (0, 1):
                        self.inputs[f"acts{s}"] = torch.zeros(
                            (1, Q_GRAPH_POOL, *arrays[f"acts{s}"].shape[1:]),
                            dtype=torch.int64, device=self.device,
                        )
                        if props:
                            self.inputs[f"feats{s}"] = torch.zeros(
                                (1, Q_GRAPH_POOL, arrays[f"feats{s}"].shape[1]),
                                dtype=torch.float32, device=self.device,
                            )
                    self.kind = kind
                for name in names:
                    self.inputs[name].copy_(staged[name], non_blocking=True)
                for s in (0, 1):
                    self.inputs[f"acts{s}"][:, : n[s]].copy_(staged[f"acts{s}"], non_blocking=True)
                    if props:
                        self.inputs[f"feats{s}"][:, : n[s]].copy_(
                            staged[f"feats{s}"], non_blocking=True
                        )
                if self.trunk is None:
                    self.trunk = self._capture("trunk", self._trunk)
                h, ctx0, ctx1 = self.trunk[1]
                for s in (0, 1):
                    if (s, n[s]) not in self.sides:
                        acts = self.inputs[f"acts{s}"][:, : n[s]]
                        feats = self.inputs[f"feats{s}"][:, : n[s]] if props else None
                        ctx = ctx0 if s == 0 else ctx1

                        def run(s: int = s, acts: Any = acts, feats: Any = feats, ctx: Any = ctx) -> Any:  # noqa: ANN401
                            return self.net.encode_actions(h, ctx, s, acts, feats)

                        self.sides[(s, n[s])] = self._capture(f"side{s}", run)
            except Exception as error:  # noqa: BLE001 - the eager road answers instead
                # A capture that failed can leave the pool mid-recording, so no graph is
                # trusted after one: every request from here is answered eagerly.
                self.failed = f"{type(error).__name__}: {error}"
                timing.count("q.graph.failed")
                return None
            self.trunk[0].replay()
            self.sides[(0, n[0])][0].replay()
            self.sides[(1, n[1])][0].replay()
            u = self.sides[(0, n[0])][1]
            v = self.sides[(1, n[1])][1]
            # `QNet.forward`'s last line and `qhead.q_matrix`'s, eagerly: the only piece
            # that reads both sizes.
            logits = self.net.pair(u, v) - self.net.pair(v, u).transpose(1, 2)
            result.copy_(torch.sigmoid(logits[0]).double(), non_blocking=True)
            done.record()
        with timing.stage("server.qsync"):
            done.synchronize()
        self.replays += 1
        timing.count("q.graph.replays")
        return result.numpy().copy()


#: The most positions `QBatchGraphs` captures a ``q_batch`` pass for: `deepen.Q_MENU_CHUNK`,
#: the chunk the ladder's and the deepening's child menus ask in. A longer batch, or one
#: with a pool past `Q_GRAPH_POOL`, is answered by the eager pass (`qhead.q_matrices`),
#: the same answer. 0 turns the batched graphs off (the A/B of IKA-373).
Q_BATCH_GRAPHS = int(os.environ.get("POKEURAOU_Q_BATCH_GRAPHS", "16"))
#: Graphs kept per piece (the sides' actions, the pair head), least recently replayed
#: dropped first. A graph held 0.4 MB of the server's host memory (219 graphs, +0.09 GB);
#: a 16-thread ladder read of seven positions captured 876 (IKA-373). The cap bounds a
#: long session's memory; a dropped shape met again is captured again, the same answer.
Q_BATCH_GRAPH_CACHE = int(os.environ.get("POKEURAOU_Q_BATCH_GRAPH_CACHE", "512"))
#: The most padded cells (B x N0 x N1) a batch graph is captured for; a larger batch is
#: answered eagerly. A graph keeps its pool's card memory, which only its own captures can
#: use, and the pool grows with the shapes. Over the 3,162 passes of a 16-thread read of
#: seven hidden positions (cells p50 4,352, p90 59,577, max 166,208; IKA-373): no cap
#: graphed them all for +2,096 MiB reserved, and the server (3.5 GB cap, IKA-334) ran out
#: of memory on a value pass; 65,536 graphed 94% for +656 MiB (eager alone: +1,686 peak),
#: at 1.41 against 1.35 ms of CPU a pass (eager 3.61). The open positions' passes are all
#: under it.
Q_BATCH_GRAPH_CELLS = int(os.environ.get("POKEURAOU_Q_BATCH_GRAPH_CELLS", str(16 * 64 * 64)))


class QBatchGraphs:
    """``q_batch`` (`qhead.q_matrices`) as CUDA graphs, one set per batch shape (IKA-373).

    The eager batched pass is about 150 kernels launched from the server's Python: 3.5 ms
    of the server's CPU (and its GIL) for a few microseconds of arithmetic, since the
    ladder's child menus ask a median of three positions with pools of a dozen actions.
    That was 7/10 of the server's time under a 16-thread ladder read (IKA-370 §4).

    `qhead.q_matrices` stacks the positions and pads each side's pool to the longest in
    the batch, so a pass's shape is (B, N0, N1). The graphs are cut where `QGraphs` cuts
    them, at those exact shapes (nothing more is padded, so each piece replays the kernels
    eager chose for the same shapes on the same inputs, and the matrices are
    `q_matrices`' to the bit; `tests/test_q_rank.py` on CUDA):

    * the trunk and both contexts read the B positions: one graph per B;
    * each side's actions read the trunk's rows and the (B, N) padded pool: one graph per
      (side, B, N);
    * the pair head, with the sigmoid and the cast `q_matrices` applies, reads both: one
      graph per pair of side graphs. Unlike `QGraphs` it is captured too: a ladder read
      meets few (B, N0, N1), and its eager kernels were most of what the pieces leave.

    Each piece has its own memory pool, as `QGraphs`' do, and a request replays one graph
    of each, in order, so an output is read before any other graph of its pool replays
    (one pool for all the pieces answered 2 to 4 of the 3,836 matrices of 400 captured
    passes wrong, by up to 0.93).
    A pair graph reads its two side graphs' outputs, so dropping a side graph drops the
    pair graphs over it. The inputs are staged in page-locked memory; the padded pools go
    through flat buffers viewed at (B, N, ...), so a piece reads a contiguous tensor as
    eager does. Enqueued under the process's graph lock (`inference._GRAPH_LOCK`) and
    waited for outside it.
    """

    @staticmethod
    def usable(device: Any) -> bool:  # noqa: ANN401
        return Q_BATCH_GRAPHS > 0 and Q_GRAPH_POOL > 0 and getattr(device, "type", None) == "cuda"

    def __init__(self, net: Any, device: Any) -> None:  # noqa: ANN401
        from collections import OrderedDict

        from .inference import _GRAPH_LOCK

        self.net = net
        self.device = device
        self.lock = _GRAPH_LOCK
        self.kind: tuple | None = None
        self.inputs: dict[str, Any] = {}
        #: Each side's padded actions and features, flat: viewed at (B, N, ...) a shape.
        self.flat: dict[str, Any] = {}
        self.pools: dict[str, Any] = {}
        #: B -> (graph, (rows, side 0's context, side 1's context))
        self.trunks: dict[int, tuple[Any, Any]] = {}
        #: (side, B, N) -> (serial, graph, action vectors)
        self.sides: OrderedDict = OrderedDict()
        #: (serial of side 0's graph, serial of side 1's) -> (graph, matrix)
        self.pairs: OrderedDict = OrderedDict()
        self._serial = 0
        self.captured = 0
        self.replays = 0
        self.evicted = 0
        self.failed: str | None = None

    def _capture(self, what: str, run: Callable[[], Any]) -> tuple[Any, Any]:  # noqa: ANN401
        import torch

        from .inference import GRAPH_THREAD

        if what not in self.pools:
            self.pools[what] = torch.cuda.graph_pool_handle()
        # Warmed up once a piece and captured on the process's one graph thread and stream
        # (IKA-377), as `QGraphs`' pieces are.
        graph, out = GRAPH_THREAD.run((self, what), run, run, self.pools[what])
        self.captured += 1
        timing.count("q.batchgraph.captured")
        return graph, out

    def _side(self, s: int, b: int, n: int, view: tuple[Any, Any]) -> tuple[int, Any, Any]:  # noqa: ANN401
        key = (s, b, n)
        if key in self.sides:
            self.sides.move_to_end(key)
            return self.sides[key]
        if len(self.sides) >= Q_BATCH_GRAPH_CACHE:
            _key, (gone, _graph, _out) = self.sides.popitem(last=False)
            # The pair graphs that read the dropped graph's output go with it.
            for pair in [p for p in self.pairs if gone in p]:
                del self.pairs[pair]
            self.evicted += 1
            timing.count("q.batchgraph.evicted")
        h, ctx0, ctx1 = self.trunks[b][1]
        acts, feats = view
        ctx = ctx0 if s == 0 else ctx1
        graph, out = self._capture(
            f"side{s}", lambda: self.net.encode_actions(h, ctx, s, acts, feats)
        )
        self._serial += 1
        self.sides[key] = (self._serial, graph, out)
        return self.sides[key]

    def matrices(self, requests: list[dict[str, np.ndarray]]) -> list[np.ndarray] | None:
        """`qhead.q_matrices(requests)`, or None when the batch is not for the graphs."""
        if self.failed or not requests or len(requests) > Q_BATCH_GRAPHS:
            return None
        sizes = [(len(r["acts0"]), len(r["acts1"])) for r in requests]
        if min(min(s) for s in sizes) <= 0:
            return None
        n = (max(s[0] for s in sizes), max(s[1] for s in sizes))
        if max(n) > Q_GRAPH_POOL or len(requests) * n[0] * n[1] > Q_BATCH_GRAPH_CELLS:
            return None
        import torch

        from . import qhead

        names = qhead.POSITION_ARRAYS
        first = requests[0]
        kind = tuple((name, first[name].dtype.str, tuple(first[name].shape[1:])) for name in names)
        kind += (("acts", tuple(first["acts0"].shape[1:])),)
        if self.kind is not None and kind != self.kind:
            return None
        props = bool(self.net.config.properties)
        b = len(requests)
        tail = tuple(first["acts0"].shape[1:])
        # `q_matrices`' own arrays: the positions stacked, each side's pool padded with
        # zeros to the longest, staged in page-locked memory.
        staged = {
            name: torch.from_numpy(
                np.ascontiguousarray(np.concatenate([r[name] for r in requests], axis=0))
            ).pin_memory()
            for name in names
        }
        padded: list[tuple[Any, Any]] = []
        for s in (0, 1):
            acts = np.zeros((b, n[s], *tail), dtype=np.int64)
            feats = np.zeros((b, n[s], qhead.FEATURE_WIDTH), dtype=np.float32) if props else None
            for i, r in enumerate(requests):
                acts[i, : len(r[f"acts{s}"])] = r[f"acts{s}"]
                if feats is not None:
                    feats[i, : len(r[f"feats{s}"])] = r[f"feats{s}"]
            padded.append((
                torch.from_numpy(acts).pin_memory(),
                None if feats is None else torch.from_numpy(feats).pin_memory(),
            ))
        result = torch.empty((b, n[0], n[1]), dtype=torch.float64, pin_memory=True)
        done = torch.cuda.Event()
        with self.lock, torch.no_grad():
            try:
                if self.kind is None:
                    self.inputs = {
                        name: torch.zeros((Q_BATCH_GRAPHS, *staged[name].shape[1:]),
                                          dtype=staged[name].dtype, device=self.device)
                        for name in names
                    }
                    width = Q_BATCH_GRAPHS * Q_GRAPH_POOL
                    for s in (0, 1):
                        self.flat[f"acts{s}"] = torch.zeros(
                            width * int(np.prod(tail)), dtype=torch.int64, device=self.device)
                        if props:
                            self.flat[f"feats{s}"] = torch.zeros(
                                width * qhead.FEATURE_WIDTH, dtype=torch.float32,
                                device=self.device)
                    self.kind = kind
                for name in names:
                    self.inputs[name][:b].copy_(staged[name], non_blocking=True)
                views: list[tuple[Any, Any]] = []
                for s in (0, 1):
                    acts, feats = padded[s]
                    av = self.flat[f"acts{s}"][: acts.numel()].view(acts.shape)
                    av.copy_(acts, non_blocking=True)
                    fv = None
                    if feats is not None:
                        fv = self.flat[f"feats{s}"][: feats.numel()].view(feats.shape)
                        fv.copy_(feats, non_blocking=True)
                    views.append((av, fv))
                if b not in self.trunks:
                    def trunk(b: int = b) -> tuple[Any, Any, Any]:  # noqa: ANN401
                        batch = {name: self.inputs[name][:b] for name in names}
                        h, sides = self.net.mons(batch)
                        field = batch["field"]
                        ctx0 = self.net.context(torch.cat([sides[:, 0], sides[:, 1], field], dim=-1))
                        ctx1 = self.net.context(torch.cat([sides[:, 1], sides[:, 0], field], dim=-1))
                        return h, ctx0, ctx1

                    self.trunks[b] = self._capture("trunk", trunk)
                side0 = self._side(0, b, n[0], views[0])
                side1 = self._side(1, b, n[1], views[1])
                key = (side0[0], side1[0])
                if key in self.pairs:
                    self.pairs.move_to_end(key)
                else:
                    if len(self.pairs) >= Q_BATCH_GRAPH_CACHE:
                        self.pairs.popitem(last=False)
                        self.evicted += 1
                        timing.count("q.batchgraph.evicted")
                    u, v = side0[2], side1[2]
                    net = self.net
                    self.pairs[key] = self._capture(
                        "pair",
                        # `QNet.forward`'s last line and `q_matrices`' sigmoid and cast.
                        lambda: torch.sigmoid(net.pair(u, v) - net.pair(v, u).transpose(1, 2)).double(),
                    )
            except Exception as error:  # noqa: BLE001 - the eager road answers instead
                # A capture that failed can leave its pool mid-recording, so no graph is
                # trusted after one: every batch from here is answered eagerly.
                self.failed = f"{type(error).__name__}: {error}"
                timing.count("q.batchgraph.failed")
                return None
            self.trunks[b][0].replay()
            side0[1].replay()
            side1[1].replay()
            graph, out = self.pairs[key]
            graph.replay()
            result.copy_(out, non_blocking=True)
            done.record()
        with timing.stage("server.qbatchsync"):
            done.synchronize()
        self.replays += 1
        timing.count("q.batchgraph.replays")
        got = result.numpy()
        return [got[i, :a, :c].copy() for i, (a, c) in enumerate(sizes)]


def served_q(
    net: Any, device: Any, files: Sequence[str], digests: Sequence[str] = ()  # noqa: ANN401
) -> Callable[..., np.ndarray]:
    """The server's side of a Q arm: (arrays) -> the N0 x N1 matrix. ``digests`` are the
    sha256 of ``files``, for the records (IKA-340).

    On a card, by `QGraphs` (stage 3); a pool past `Q_GRAPH_POOL`, or a CPU server, by the
    eager pass (`qhead.q_matrix`), which is the same answer. The module is only read, never
    reparametrised, so the serving threads share it.
    """
    from . import qhead
    from .inference import eager_pass

    graphs = QGraphs(net, device) if QGraphs.usable(device) else None
    batch_graphs = QBatchGraphs(net, device) if QBatchGraphs.usable(device) else None

    def answer(arrays: dict[str, np.ndarray]) -> np.ndarray:
        started = time.perf_counter()
        # IKA-339: the graphs' road (staging, replays, the pair product, the copy back; the
        # wait on the card is `server.qsync` inside it) and the eager one, apart.
        with timing.stage("server.qgraph"):
            out = graphs.matrix(arrays) if graphs is not None else None
        if out is None:
            # Through the value arms' gate (IKA-377): a batch past the graphs' cells is
            # 0.1-0.3 GB of activations, and nothing bounded how many ran at once.
            with timing.stage("server.qeager"):
                out, _entered = eager_pass(lambda: qhead.q_matrix(net, arrays, device))
        answer.held += time.perf_counter() - started
        answer.calls += 1
        return out

    def batch(requests: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
        """Several positions' matrices in one forward pass (IKA-307, op ``q_batch``): by
        `QBatchGraphs` on a card (IKA-373), else eagerly -- the same answer."""
        started = time.perf_counter()
        with timing.stage("server.qbatchgraph"):
            out = batch_graphs.matrices(requests) if batch_graphs is not None else None
        if out is None:
            with timing.stage("server.qbatcheager"):
                out, _entered = eager_pass(lambda: qhead.q_matrices(net, requests, device))
        answer.held += time.perf_counter() - started
        answer.calls += 1
        answer.batched += len(requests)
        return out

    answer.held = 0.0
    answer.calls = 0
    answer.batched = 0
    answer.batch = batch
    answer.graphs = graphs
    answer.batch_graphs = batch_graphs
    answer.fingerprint = net.vocab_fingerprint
    answer.properties = bool(net.config.properties)
    answer.files = list(files)
    answer.digests = list(digests)
    return answer


def load_q_arms(paths: dict[str, Path], device_name: str) -> dict[str, Any]:
    """Each named Q arm, loaded on `device_name`, as `served_q`."""
    import torch

    from . import qhead

    device = torch.device(device_name)
    return {
        name: served_q(qhead.load_q(path, device_name), device, [Path(path).name],
                       [file_sha256(path)])
        for name, path in paths.items()
    }


def _choices(actions: Sequence[SideAction]) -> tuple[str, ...]:
    return tuple(action.to_choice() for action in actions)


#: What `prefetch` hands `q_ranking` for one side: (the choices of the pool it asked for,
#: the other side's pool, the matrix, and a memo shared by every side that asked the same
#: request -- where the first to solve the matrix's game leaves it, IKA-339).
Given = tuple[tuple[str, ...], list[SideAction], np.ndarray, dict[str, Any]]


def prefetch(
    reg: Regulation, pos: Position, views: tuple[Position, Position], model: Any  # noqa: ANN401
) -> dict[int, Given]:
    """Both sides' Q matrices for one agent's menus, asked in one round trip (stage 3).

    Side `s` ranks its pool on `pos` (what `narrow` hands the ranking: `qhead.legal_pool`
    of `pos`) against the other side's pool on `views[s]`, the view it ranks from -- the
    request `q_ranking` would make, made before `narrow` asks for it. Both go to the
    server together (`RemoteQ.matrices`), which answers each as its own request, so the
    matrices are the ones the two round trips would have brought. When both sides read
    one position and one pair of pools (nothing hidden), that is one request.
    A side with an empty pool on either side asks nothing, as `q_ranking` does not.
    """
    from . import qhead

    asks: list[tuple[Position, Pools]] = []
    keys: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
    plan: dict[int, tuple[tuple[str, ...], list[SideAction], int]] = {}
    for side in (0, 1):
        own = qhead.legal_pool(reg, pos, side)
        at = views[side]
        foe = qhead.legal_pool(reg, at, 1 - side)
        if not own or not foe:
            continue
        pools = (own, foe) if side == 0 else (foe, own)
        key = (id(at), _choices(pools[0]), _choices(pools[1]))
        if key not in keys:
            keys.append(key)
            asks.append((at, pools))
        plan[side] = (key[1 + side], foe, keys.index(key))
    if not asks:
        return {}
    matrices = model.matrices(reg, asks)
    timing.count("q.prefetch.trips")
    timing.count("q.prefetch.requests", len(asks))
    # IKA-339: one memo a request. Two sides that read one position and one pair of pools
    # rank from one matrix, and so from one game: its solve is the same LP on the same
    # numbers, and was solved twice (2.9 of 16 rankings a game in M-C generation).
    memos: list[dict[str, Any]] = [{} for _ in asks]
    return {
        side: (own_key, foe, matrices[index].copy(), memos[index])
        for side, (own_key, foe, index) in plan.items()
    }


_Q_SOLVED: dict[Any, Any] = {}


def _q_solve(solve: Callable[..., Any]) -> Callable[..., Any]:
    """`solve` charged to `lp.q` rather than `lp` (IKA-339); `solve` when timing is off."""
    found = _Q_SOLVED.get(solve)
    if found is None:
        found = _Q_SOLVED[solve] = timing.timed("lp.q")(getattr(solve, "__wrapped__", solve))
    return found


def q_ranking(
    reg: Regulation, pos: Position, side: int, model: Any, label: str,  # noqa: ANN401
    given: Given | None = None,
) -> Callable[..., np.ndarray]:
    """A ranking for `narrow`: each candidate's value against the foe's half of Q's solve.

    Q is asked once for both sides' whole pools on `pos` (the view this side ranks from),
    the game on it is solved, and side 0's candidates are scored by their row value
    against the column mix, side 1's by minus their column value against the row mix --
    so higher is better for the side ranking, as the leaf ranking's scores are.

    ``given`` is `prefetch`'s answer for this side: used when `narrow` hands the pool it
    was asked for, and otherwise Q is asked here as without it.
    """
    from . import qhead
    from .equilibrium import EquilibriumError, solve

    def rank(pool: list[SideAction], _scored: object = None) -> np.ndarray:
        memo: dict[str, Any] | None = None
        if given is not None and pool and given[0] == _choices(pool):
            foe, q = given[1], given[2]
            memo = given[3] if len(given) > 3 else None
            timing.count("q.prefetched")
        else:
            foe = qhead.legal_pool(reg, pos, 1 - side)
            if not pool or not foe:
                return np.zeros(len(pool))
            pools = (pool, foe) if side == 0 else (foe, pool)
            with timing.purpose("rank"):
                q = model.matrix(reg, pos, pools)
        got = QRANKED.setdefault(label, {"rankings": 0, "cells": 0})
        got["rankings"] += 1
        got["cells"] += int(q.size)
        timing.count("q.rankings")
        timing.count("q.cells", int(q.size))
        if memo is not None and "solved" in memo:
            # IKA-339: the other side solved this matrix's game already.
            e = memo["solved"]
            timing.count("q.solve.shared")
        else:
            try:
                # IKA-339: the Q's own game, as a row apart from the search's LPs (`lp`).
                # Timing off, this is `solve` itself.
                e = _q_solve(solve)(q)
            except (EquilibriumError, ValueError):
                e = None
            if memo is not None:
                memo["solved"] = e
        if e is None:
            # An LP that does not solve still leaves an order: the mean against every reply.
            return q.mean(axis=1) if side == 0 else -q.mean(axis=0)
        # A copy each: the two sides share `e` now, and neither may hand the other's array on.
        return np.array(e.row_ev if side == 0 else -np.asarray(e.col_ev), dtype=np.float64)

    return rank


def add_q_flags(ap: Any) -> None:  # noqa: ANN401 - an ArgumentParser
    """A worker's two ways to name its Q: an arm on its inference server, or a file here.

    The plain flags give the default Q (labels ``q``, ``q-nocover``); the ``-named`` ones
    give the Q of the labels ``q.NAME`` / ``q-nocover.NAME`` (stage 3).
    """
    ap.add_argument(
        "--q-arm", default=None,
        help="the Q the q rank fills rank by, as the inference server's Q arm of this name "
        "(IKA-274; the server's --q-arm). With --inference",
    )
    ap.add_argument(
        "--q-model", type=Path, default=None,
        help="the same Q loaded here instead (torch in this worker). With neither, a q "
        f"rank fill loads {DEFAULT_Q} here, and stops if it is not there (IKA-338)",
    )
    ap.add_argument(
        "--q-arm-named", nargs=2, action="append", default=[], metavar=("NAME", "ARM"),
        help="the Q of the rank fills q.NAME / q-nocover.NAME, as the server's Q arm ARM",
    )
    ap.add_argument(
        "--q-model-named", nargs=2, action="append", default=[], metavar=("NAME", "FILE"),
        help="the same loaded here",
    )


#: Where the default Q is, from the repository root: the Q of M-C's shipped leaf (IKA-274
#: stage 3). What a q fill ranks by when no Q is named, in generation, on the board and in
#: human play (IKA-338; `search.SHIPPED_RANK_FILL`).
DEFAULT_Q = "data/models/q-mc0.pt"


def default_q() -> Path:
    """`DEFAULT_Q` in this checkout (it may not exist: `missing_q` says what to do)."""
    from .regulation import repo_root

    return repo_root() / DEFAULT_Q


def missing_q(path: Path) -> str:
    """The stop for a q fill whose default Q is not there (IKA-338).

    Generation and the board never fall back to ``refs2`` on their own: a run that asked for
    the shipped player and quietly got another is the drift `agent_drift.py` exists for.
    """
    return (
        f"the rank fill q-nocover (the default for a leaf-ranked menu since IKA-338) ranks by "
        f"a Q, and there is none at {path}: name one with --q-model FILE, or play without one "
        "by naming --rank-fill refs2 (and --baseline-rank-fill refs2 for a board's other arm)"
    )


def tail_wants_default_q(tail: Sequence[str]) -> bool:
    """Whether workers handed this tail rank by the default Q, so their launcher loads it.

    The launchers (`generate_queue.py`, `match_queue.py`) pass everything after ``--`` to the
    workers untouched. Read here for what decides the Q: each arm's ``--rank-leaf`` and
    ``--rank-fill`` (``--baseline-*`` for a board's other arm), resolved as the worker
    resolves them (`search.resolve_rank_fill`), and a ``--deepen`` whose oracle or children
    rank by the default Q. A tail that names its own Q (``--q-arm`` / ``--q-model``) wants
    none from the launcher.
    """
    import argparse

    from .deepen import deepen_spec
    from .search import resolve_rank_fill

    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for arm in ("", "baseline-"):
        ap.add_argument(f"--{arm}rank-leaf", action="store_true")
        ap.add_argument(f"--{arm}rank-fill", default=None)
        ap.add_argument(f"--{arm}deepen", default=None)
    ap.add_argument("--q-arm", default=None)
    ap.add_argument("--q-model", default=None)
    got, _ = ap.parse_known_args(list(tail))
    if got.q_arm is not None or got.q_model is not None:
        return False
    for leafy, fill, deepen in ((got.rank_leaf, got.rank_fill, got.deepen),
                                (got.baseline_rank_leaf, got.baseline_rank_fill,
                                 got.baseline_deepen)):
        label = resolve_rank_fill(fill, leafy)
        if leafy and is_q(label) and not q_name(label):
            return True
        if deepen is not None:
            spec = deepen_spec(deepen)
            if spec.q_probe is not None or spec.child_q is not None:
                return True
    return False


def install_from_args(
    args: Any, encoder: Any, fills: Sequence[str], error: Callable[[str], Any]  # noqa: ANN401
) -> dict[str, Any]:
    """Installs the Qs the flags name, and stops when a q fill has none (or one has no use).

    A q fill of the default name with no Q named ranks by `default_q` (IKA-338), loaded
    here; without that file it stops (`missing_q`). Returns {name: model} ("" the default),
    empty when no fill asks for a Q and none was named.
    """
    wanted = {q_name(fill) for fill in fills if is_q(fill)}
    if args.q_arm is not None and args.q_model is not None:
        error("--q-arm and --q-model both name a Q; one of them")
    named: dict[str, tuple[str, str]] = {}
    for how, pairs in (("arm", getattr(args, "q_arm_named", None) or []),
                       ("model", getattr(args, "q_model_named", None) or [])):
        for name, what in pairs:
            if not Q_NAME.fullmatch(name):
                error(f"a Q's name is lower-case letters, digits and _: {name!r}")
            if name in named:
                error(f"the Q {name!r} is named twice")
            named[name] = (how, what)
    if args.q_arm is not None:
        named[""] = ("arm", args.q_arm)
    elif args.q_model is not None:
        named[""] = ("model", str(args.q_model))
    for name in sorted(set(named) - wanted):
        error(f"the Q {name or '(default)'} is named but no rank fill ranks by it")
    for name in sorted(wanted - set(named)):
        if name:
            error(f"a q rank fill .{name} needs --q-arm-named {name} ARM (served) or "
                  f"--q-model-named {name} FILE (loaded here)")
        path = default_q()
        if not path.exists():
            error(missing_q(path))
        named[""] = ("model", str(path))
    models: dict[str, Any] = {}
    for name, (how, what) in named.items():
        if how == "arm":
            if getattr(args, "inference", None) is None:
                error("--q-arm is an arm of the inference server: it needs --inference")
            model = RemoteQ(args.inference, what, encoder)
        else:
            model = LocalQ(Path(what), encoder, device=getattr(args, "device", None) or "cpu")
        install(model, name)
        models[name] = model
    return models


#: A Q's name in a rank-fill label (``q-nocover.NAME``).
Q_NAME = re.compile(r"[a-z0-9_]+")


def file_sha256(path: Path | str) -> str:
    """The sha256 of a Q file as it is on disk (IKA-340): what a record names it by, since
    a file name alone says nothing once the file is replaced by a retrained one."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_fields(models: dict[str, Any]) -> dict[str, list[str]]:
    """The Qs behind `install_from_args`'s models, as a record carries them (IKA-340):
    ``qModel`` the files (`describe_installed`, the form a match record already had) and
    ``qModelSha256`` each one's sha256 in the same order. Empty without a Q, so a record
    that ranked by none gains no field."""
    if not models:
        return {}
    digests: list[str] = []
    for name in sorted(models):
        digests += models[name].digests()
    return {"qModel": describe_installed(models), "qModelSha256": digests}


def describe_installed(models: dict[str, Any]) -> list[str]:
    """The files behind `install_from_args`'s Qs, for a record: the default's as they are,
    a named one's as ``NAME=file``."""
    out: list[str] = []
    for name in sorted(models):
        out += [f if not name else f"{name}={f}" for f in models[name].describe()]
    return out


__all__ = [
    "DEFAULT_Q",
    "add_q_flags",
    "default_q",
    "missing_q",
    "tail_wants_default_q",
    "install_from_args",
    "describe_installed",
    "file_sha256",
    "record_fields",
    "prefetch",
    "q_covers",
    "q_name",
    "QRANKED",
    "Q_RANK_FILL",
    "LocalQ",
    "RemoteQ",
    "install",
    "installed",
    "is_q",
    "load_q_arms",
    "q_ranking",
    "served_q",
]
