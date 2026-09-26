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

import json
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
Q_RANK_FILL = re.compile(r"q(-nocover)?")


def is_q(label: str) -> bool:
    """Whether a rank-fill label ranks by a Q rather than by filling cells with the leaf."""
    return Q_RANK_FILL.fullmatch(label) is not None


#: Per rank-fill label, what the Q ranking did in this process: rankings asked, and the
#: cells of the Q matrices behind them. The positive control that a ``q`` label reached a
#: menu, read as a change by a match's echo (as `narrow.COVERLESS`).
QRANKED: dict[str, dict[str, int]] = {}

_INSTALLED: list[Any] = []


def install(model: Any) -> None:  # noqa: ANN401 - LocalQ or RemoteQ
    """Makes `model` the process's Q: every ``q`` label ranks with it."""
    _INSTALLED[:] = [model]


def installed() -> Any:  # noqa: ANN401
    """The process's Q, or a stop that says how to give one."""
    if not _INSTALLED:
        raise RuntimeError(
            "a q rank fill needs a Q: --q-model <file> (loaded here) or --q-arm <name> "
            "(on the inference server, which --q-model on the launcher loads)"
        )
    return _INSTALLED[0]


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
            raise ValueError(f"{self.path} reads another vocabulary than this encoder's")
        self._device = torch.device(self.device)
        self.calls = 0

    def describe(self) -> list[str]:
        return [Path(self.path).name]

    def matrix(
        self, reg: Regulation, pos: Position, pools: tuple[Sequence[SideAction], Sequence[SideAction]]
    ) -> np.ndarray:
        from . import qhead

        arrays = _pool_arrays(reg, self.encoder, pos, pools, self.net.config.properties)
        self.calls += 1
        return qhead.q_matrix(self.net, arrays, self._device)


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
    and the op ``q``: one position, both pools, one matrix back.
    """

    address: str
    model: str
    encoder: Any
    properties: bool = True
    buffer_bytes: int = Q_BUFFER_BYTES
    calls: int = field(default=0, init=False)
    waited: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._block = shared_memory.SharedMemory(create=True, size=self.buffer_bytes)
        host, _, port = self.address.rpartition(":")
        self._sock = socket.create_connection((host, int(port)))
        self._file = self._sock.makefile("rwb")
        about = self._ask({"op": "describe_q", "model": self.model})
        if about["fingerprint"] != self.encoder.vocab.fingerprint():
            raise RuntimeError(f"the server's Q arm {self.model!r} reads another vocabulary")
        self.properties = bool(about["properties"])
        self._files = list(about["files"])

    def _ask(self, request: dict[str, Any]) -> dict[str, Any]:
        self._file.write((json.dumps(request) + "\n").encode("utf-8"))
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise RuntimeError("the inference server closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"Q request failed: {reply.get('error')}")
        return reply

    def describe(self) -> list[str]:
        return list(self._files)

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
        arrays = _pool_arrays(reg, self.encoder, pos, pools, self.properties)
        layout, used = _plan_named(arrays)
        shape = (len(pools[0]), len(pools[1]))
        result_offset = (used + 63) & ~63
        if result_offset + shape[0] * shape[1] * 8 > self.buffer_bytes:
            raise RuntimeError(f"a Q request of {shape} does not fit {self.buffer_bytes} bytes")
        view = self._block.buf
        for item in layout:
            array = arrays[item["name"]]
            target = np.frombuffer(
                view, dtype=array.dtype, count=array.size, offset=int(item["offset"])
            ).reshape(array.shape)
            np.copyto(target, array)
        sent = time.perf_counter()
        with timing.stage("serve.q"):
            self._ask({
                "op": "q", "model": self.model, "shm": self._block.name, "layout": layout,
                "shape": list(shape), "result_offset": result_offset,
            })
        self.waited += time.perf_counter() - sent
        self.calls += 1
        count = shape[0] * shape[1]
        return np.frombuffer(
            view[result_offset : result_offset + count * 8], dtype=np.float64
        ).reshape(shape).copy()


def served_q(net: Any, device: Any, files: Sequence[str]) -> Callable[..., np.ndarray]:  # noqa: ANN401
    """The server's side of a Q arm: (arrays) -> the N0 x N1 matrix.

    Eager, never a CUDA graph: a Q request's shape is two pool sizes, a few thousand
    combinations, and the graphs are per value arm (`inference._Graphs`). Eager passes
    already run beside captures (a value arm's long requests do, IKA-306 §1.2), so this
    takes no graph lock and holds up no replay. The module is only read, never
    reparametrised, so the serving threads share it.
    """
    from . import qhead

    def answer(arrays: dict[str, np.ndarray]) -> np.ndarray:
        started = time.perf_counter()
        out = qhead.q_matrix(net, arrays, device)
        answer.held += time.perf_counter() - started
        answer.calls += 1
        return out

    answer.held = 0.0
    answer.calls = 0
    answer.fingerprint = net.vocab_fingerprint
    answer.properties = bool(net.config.properties)
    answer.files = list(files)
    return answer


def load_q_arms(paths: dict[str, Path], device_name: str) -> dict[str, Any]:
    """Each named Q arm, loaded on `device_name`, as `served_q`."""
    import torch

    from . import qhead

    device = torch.device(device_name)
    return {
        name: served_q(qhead.load_q(path, device_name), device, [Path(path).name])
        for name, path in paths.items()
    }


def q_ranking(
    reg: Regulation, pos: Position, side: int, model: Any, label: str  # noqa: ANN401
) -> Callable[..., np.ndarray]:
    """A ranking for `narrow`: each candidate's value against the foe's half of Q's solve.

    Q is asked once for both sides' whole pools on `pos` (the view this side ranks from),
    the game on it is solved, and side 0's candidates are scored by their row value
    against the column mix, side 1's by minus their column value against the row mix --
    so higher is better for the side ranking, as the leaf ranking's scores are.
    """
    from . import qhead
    from .equilibrium import EquilibriumError, solve

    def rank(pool: list[SideAction], _scored: object = None) -> np.ndarray:
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
        try:
            e = solve(q)
        except (EquilibriumError, ValueError):
            # An LP that does not solve still leaves an order: the mean against every reply.
            return q.mean(axis=1) if side == 0 else -q.mean(axis=0)
        return np.asarray(e.row_ev if side == 0 else -np.asarray(e.col_ev), dtype=np.float64)

    return rank


def add_q_flags(ap: Any) -> None:  # noqa: ANN401 - an ArgumentParser
    """A worker's two ways to name its Q: an arm on its inference server, or a file here."""
    ap.add_argument(
        "--q-arm", default=None,
        help="the Q the q rank fills rank by, as the inference server's Q arm of this name "
        "(IKA-274; the server's --q-arm). With --inference",
    )
    ap.add_argument(
        "--q-model", type=Path, default=None,
        help="the same Q loaded here instead (torch in this worker)",
    )


def install_from_args(
    args: Any, encoder: Any, fills: Sequence[str], error: Callable[[str], Any]  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Installs the Q the flags name, and stops when a q fill has none (or one has no use).

    Returns the model, or None when no fill asks for one and none was named.
    """
    wanted = any(is_q(fill) for fill in fills)
    if args.q_arm is not None and args.q_model is not None:
        error("--q-arm and --q-model both name a Q; one of them")
    if not wanted:
        if args.q_arm is not None or args.q_model is not None:
            error("a Q is named but no rank fill is q or q-nocover")
        return None
    if args.q_arm is not None:
        if getattr(args, "inference", None) is None:
            error("--q-arm is an arm of the inference server: it needs --inference")
        model = RemoteQ(args.inference, args.q_arm, encoder)
    elif args.q_model is not None:
        model = LocalQ(args.q_model, encoder, device=getattr(args, "device", None) or "cpu")
    else:
        error("a q rank fill needs --q-arm (served) or --q-model (loaded here)")
    install(model)
    return model


__all__ = [
    "add_q_flags",
    "install_from_args",
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
