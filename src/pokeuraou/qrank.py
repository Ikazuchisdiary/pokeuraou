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

Where the Q lives: in the inference server, as one more arm (the Rust server's
`--q-arm`), so a served worker still holds no torch (`RemoteQ`). A worker that loads its
leaf itself loads the Q the same way (`LocalQ`). One Q per process (`install`): both arms
of a match that name a ``q`` label rank with it.
"""

from __future__ import annotations

import hashlib
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

from . import portlp, timing
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
    # IKA-455: the games of every request in one crossing to the port, each `solve`'s answer
    # (`portlp`); `rank` takes it. A game the port could not solve is `solve`'s error returned
    # in its place: `rank`'s None. A fault of the port is `portlp.PortFault` (IKA-460).
    with timing.stage("lp.q"):
        got = portlp.solve_many(reg, [matrices[index] for index in range(len(asks))])
    for memo, answer in zip(memos, got, strict=True):
        memo["pre"] = None if isinstance(answer, Exception) else answer
    return {
        side: (own_key, foe, matrices[index].copy(), memos[index])
        for side, (own_key, foe, index) in plan.items()
    }


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
        elif memo is not None and "pre" in memo:
            # IKA-455: `prefetch` had the port solve it, in one crossing for the request.
            e = memo["pre"]
            memo["solved"] = e
        else:
            # IKA-339: the Q's own game, as a row apart from the search's LPs (`lp`); the
            # port's HiGHS solves it (`portlp`, IKA-460: the LP has no Python road).
            with timing.stage("lp.q"):
                got = portlp.solve_many(reg, [q])[0]
            e = None if isinstance(got, Exception) else got
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
    ap.add_argument(
        "--q-inference-named", nargs=2, action="append", default=[], metavar=("NAME", "HOST:PORT"),
        help="IKA-461: the server of the named Q's arm, when it is not --inference",
    )


#: Where the default Q is, from the repository root: the Q of M-C's shipped leaf
#: (q-mc6 since IKA-438: taught by value-mc6 x2's cells, its trunk reads the state and bind
#: columns; q-mc5 from IKA-432, q-mc4bindaux from IKA-430, q-mc4st from IKA-427, q-mc4 from IKA-409,
#: q-mc3 from IKA-402, q-mc2 from IKA-400, q-mc0, IKA-274 stage 3, before it). What a
#: q fill ranks by when no Q is named, in generation, on the board and in human play
#: (IKA-338; `search.SHIPPED_RANK_FILL`).
DEFAULT_Q = "data/models/q-mc6.pt"


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
            # IKA-461: a named Q may be on a server of its own (`--q-inference-named`).
            at = dict(getattr(args, "q_inference_named", None) or []).get(name, args.inference)
            model = RemoteQ(at, what, encoder)
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
    "q_ranking",
]
