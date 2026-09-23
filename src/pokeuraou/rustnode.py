"""A warm Rust process that fills a whole node's payoff matrix.

`batched_payoffs` is already this project's node-level boundary -- one position and two
lists of actions in, one matrix out -- which makes it the one place a process boundary
costs nothing: the leaves never cross, only 576 floats do.

The port refuses cells it does not model (a self-switch that suspends the turn, an ability
outside its set), and names them. Those come back in `refused` and the caller resolves them
in Python, so a partial port is usable rather than merely measurable: a cell Rust fills is
Python's own answer, and a cell it declines costs Python's own time.

Only the parameter-free objectives cross. A learned value function cannot: its leaves are
the input, and a node's leaves are tens of megabytes of JSON. That boundary needs the
encoder ported too, and until then a node scored by a value net stays in Python.

An encoded node's arrays do not come down the pipe when they can help it. They are the
large part of a crossing -- measured on 2026-09-20, reading them was 25.3% of an analysis
run -- and a pipe hands them over a buffer at a time with each process waiting on the
other. So they go through a block of shared memory instead, and only the header still
crosses the pipe. Nobody sizes the block by guessing: a node's leaves are not counted
until its turns are resolved, so the child asks for the size it has just found and this
end makes one of exactly that. A platform without shared memory, a machine that will not
spare the commit, or a binary built before any of this all fall back to the pipe, which
is why the pipe is still here.

Opt-in and off by default::

    POKEURAOU_RUST_NODE=1                      # use it when it is available
    POKEURAOU_RUST_NODE_BIN=/path/to/binary    # defaults to rust/target/release/
    POKEURAOU_RUST_NODE_SHM_MB=512             # the largest block; 0 keeps the pipe

Nothing imports this unless the variable is set, and `available()` answers without raising
so a caller can fall back silently rather than fail a run that was working.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import timing
from .actions import MoveAction, PassAction, SideAction, SwitchAction
from .position import Position
from .regulation import Regulation, repo_root
from .resolve import Budget

#: Seconds to wait for one node before giving up on the subprocess entirely. Not a
#: latency budget: a width-48 node is tens of megabytes and a dozen workers share one
#: machine, so this only has to be longer than the slowest honest answer and shorter than
#: "the run never finishes".
NODE_TIMEOUT = float(os.environ.get("POKEURAOU_RUST_NODE_TIMEOUT", "300"))

ENV_ENABLE = "POKEURAOU_RUST_NODE"
ENV_BINARY = "POKEURAOU_RUST_NODE_BIN"
ENV_SHM_MB = "POKEURAOU_RUST_NODE_SHM_MB"

#: The largest block a process will hold for an encoded node's arrays; zero keeps the pipe.
#:
#: A ceiling and not a size. Nobody chooses the size: a node's leaves are not counted until
#: the turns are resolved, so the child asks for what this one needs and the block is made
#: to fit it. What the ceiling is for is the other end -- on Windows a block is backed by
#: the paging file and its whole length is charged to commit the moment it is created, and
#: a generation run holds one of these per worker.
SHM_MAX_BYTES = int(float(os.environ.get(ENV_SHM_MB, "512")) * 1024 * 1024)


def _release(block: Any) -> None:
    """Give a block back, both halves, without letting either failure end a run."""
    if block is None:
        return
    with contextlib.suppress(Exception):
        block.close()
    with contextlib.suppress(Exception):
        # A no-op on Windows, where the block dies with the last handle on it.
        block.unlink()


def binary_path() -> Path:
    override = os.environ.get(ENV_BINARY)
    if override:
        return Path(override)
    name = "pokeuraou-damage.exe" if sys.platform == "win32" else "pokeuraou-damage"
    return repo_root() / "rust" / "target" / "release" / name


#: Set once the bridge has failed, so a broken one is not retried for every node.
_GAVE_UP = False


def binary_fingerprint() -> dict[str, Any]:
    """What the binary *is*, for a record that has to mean the code that ran.

    A differential that only checks the binary exists can pass against a build from before
    the change it is meant to check. Hashing the sources would not help: the games are
    played by the executable, and between an edit and a `cargo build` the two disagree.
    So this hashes the executable.
    """
    path = binary_path()
    if not path.exists():
        return {"path": str(path), "present": False}
    raw = path.read_bytes()
    return {
        "path": str(path),
        "present": True,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
        "built": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def sources_newer_than_binary() -> list[str]:
    """Rust sources edited since the binary was built, newest first.

    Empty when the binary is current, when it cannot be told (`POKEURAOU_RUST_NODE_BIN`
    points somewhere of the caller's choosing), or when there is no binary at all -- the
    caller has a better message for that one.
    """
    path = binary_path()
    if os.environ.get(ENV_BINARY) or not path.exists():
        return []
    built = path.stat().st_mtime
    sources = sorted(
        (p for p in (repo_root() / "rust" / "src").glob("*.rs") if p.stat().st_mtime > built),
        key=lambda p: -p.stat().st_mtime,
    )
    manifest = repo_root() / "rust" / "Cargo.toml"
    if manifest.exists() and manifest.stat().st_mtime > built:
        sources.append(manifest)
    return [p.name for p in sources]


def require_current_binary() -> dict[str, Any]:
    """The fingerprint, refusing if the binary is older than the sources.

    For the differentials. Holding a build to a fixture and reporting "0 wrong" is a
    statement about whatever was compiled, and saying it about code that has not been
    compiled yet is the failure the differential exists to catch, wearing its face.
    """
    path = binary_path()
    if not path.exists():
        raise SystemExit(f"no Rust binary at {path}; `cd rust && cargo build --release`")
    stale = sources_newer_than_binary()
    if stale:
        listed = ", ".join(stale[:4]) + ("..." if len(stale) > 4 else "")
        raise SystemExit(
            f"{path.name} was built before {listed} changed. A differential against a "
            "stale binary says nothing about the code you edited; "
            "`cd rust && cargo build --release` first."
        )
    return binary_fingerprint()


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "") not in ("", "0", "false", "no")


def available() -> bool:
    return not _GAVE_UP and enabled() and binary_path().exists()


def dump_action(action: object) -> dict[str, Any]:
    if isinstance(action, MoveAction):
        return {
            "kind": "move",
            "slot": action.slot,
            "moveIndex": action.move_index,
            "moveId": action.move_id,
            "target": action.target,
            "mega": action.mega,
        }
    if isinstance(action, SwitchAction):
        return {
            "kind": "switch",
            "slot": action.slot,
            "partyIndex": action.party_index,
            "species": action.species,
        }
    if isinstance(action, PassAction):
        return {"kind": "pass", "slot": action.slot}
    raise TypeError(f"unknown action {action!r}")


def dump_budget(budget: Budget) -> dict[str, Any]:
    return {
        "damageRolls": budget.damage_rolls,
        "enumerateCrit": budget.enumerate_crit,
        "enumerateAccuracy": budget.enumerate_accuracy,
        "enumerateStatusChecks": budget.enumerate_status_checks,
        "enumerateSecondary": budget.enumerate_secondary,
        "enumerateSpeedTies": budget.enumerate_speed_ties,
        "pinnedPolicy": budget.pinned_policy,
        "maxBranches": budget.max_branches,
        "mergeDuplicates": budget.merge_duplicates,
    }


@dataclass
class EncodedNode:
    """One node's leaves as the encoder's arrays, plus the fold back to a matrix."""

    encoded: Any
    #: (row, column, start, weights) for a cell that is a plain weighted mean.
    #: (row, column, leaf indices, weights). Indices rather than a start, because the
    #: port shares a leaf between cells when it is the same position.
    spans: list[tuple[int, int, list[int], list[float]]]
    #: (row, column, fold tree) for a cell whose turn stopped for a replacement.
    folded: list[tuple[int, int, dict]]
    exact: list[list[bool]]
    refused: list[tuple[int, int, str]]
    unmodelled: tuple[str, ...]
    #: Microseconds the port spent resolving the turns and encoding their leaves. Two
    #: different costs with two different fixes, so they are reported apart.
    resolve_us: float = 0.0
    encode_us: float = 0.0
    #: And what the crossing itself cost over there: reading this request, and building
    #: and serialising the header that answers it. Apart for the same reason -- the fix
    #: for one is a binary request and the fix for the other is a binary header.
    parse_us: float = 0.0
    header_us: float = 0.0
    #: One value per leaf for each named objective the request asked for beside the leaves.
    leaf_values: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    @timing.timed("rust.unpack")
    def unpack(header: dict[str, Any], body: bytearray) -> EncodedNode:
        import numpy as np

        from .encode import Encoded

        n = int(header["leaves"])
        m = int(header["monsPerSide"])
        layout = [
            ("species", np.int32, (n, 2, m)),
            ("ability", np.int32, (n, 2, m)),
            ("item", np.int32, (n, 2, m)),
            ("moves", np.int32, (n, 2, m, 4)),
            ("mon", np.float32, (n, 2, m, int(header["monWidth"]))),
            ("mask", np.float32, (n, 2, m)),
            ("side", np.float32, (n, 2, int(header["sideWidth"]))),
            ("field", np.float32, (n, int(header["fieldWidth"]))),
        ]
        arrays: dict[str, Any] = {}
        offset = 0
        for name, dtype, shape in layout:
            count = 1
            for axis in shape:
                count *= axis
            flat = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
            offset += count * np.dtype(dtype).itemsize
            # No widening. `nn.Embedding` takes int32 and gives the identical answer,
            # because an index lookup does no arithmetic -- so the four index arrays stay
            # as they came and nothing is rebuilt. Widening them cost 0.058 ms a call, and
            # more than that once they had to be copied into a shared buffer at twice the
            # width on the way to an inference server.
            arrays[name] = flat.reshape(shape)
        return EncodedNode(
            encoded=Encoded(
                species=arrays["species"],
                ability=arrays["ability"],
                item=arrays["item"],
                moves=arrays["moves"],
                mon=arrays["mon"],
                mask=arrays["mask"],
                side=arrays["side"],
                field=arrays["field"],
                unknown_volatiles=dict(header.get("unknownVolatiles", {})),
            ),
            spans=[
                (int(i), int(j), list(indices), list(w))
                for i, j, indices, w in header["spans"]
            ],
            folded=[(int(i), int(j), root) for i, j, root in header["folded"]],
            exact=header["exact"],
            refused=[(int(i), int(j), str(why)) for i, j, why in header["refused"]],
            unmodelled=tuple(header["unmodelled"]),
            resolve_us=float(header.get("resolveUs", 0.0)),
            encode_us=float(header.get("encodeUs", 0.0)),
            parse_us=float(header.get("parseUs", 0.0)),
            # The fold and the text are both the header being JSON, and one number is what
            # a table has room for. The split stays in the header for anyone who needs it.
            header_us=float(header.get("foldUs", 0.0)) + float(header.get("headerUs", 0.0)),
            leaf_values={
                name: np.frombuffer(
                    body, dtype=np.float64, count=n, offset=offset + index * n * 8
                )
                for index, name in enumerate(header.get("leafObjectives", []))
            },
        )


@dataclass
class ResolvedTurn:
    """One turn's branch weights, and optionally the branch that was chosen."""

    branches: list[float]
    #: Weights of the outcomes that stopped at a mid-turn replacement. Their continuation
    #: state stays in the Rust process, so a caller that draws one resolves the turn itself.
    suspended: list[float]
    exact: bool
    unmodelled: tuple[str, ...]
    position: Position | None


@dataclass
class NodeResult:
    """One matrix per objective, plus what the port declined to fill."""

    payoffs: list[list[list[float]]]
    exact: list[list[bool]]
    #: (row, column, why) for every cell the port refused.
    refused: list[tuple[int, int, str]]
    unmodelled: tuple[str, ...]


#: One process per regulation, per interpreter. A generation worker is a process, so
#: this is one Rust process per worker, which is what the parallelism wants.
_NODES: dict[str, RustNode | None] = {}


def node_for(reg: Regulation) -> RustNode | None:
    """The warm process for this regulation, or None if it is not usable.

    A failure here is not a reason to fail a run that was working: the caller falls back
    to Python, and the reason is printed once so it cannot be silently slow instead of
    silently wrong.
    """
    format_id = reg.meta.format_id
    if format_id in _NODES:
        return _NODES[format_id]
    if not available():
        _NODES[format_id] = None
        return None
    try:
        _NODES[format_id] = RustNode(reg)
    except Exception as exc:  # noqa: BLE001 - any failure means "use Python"
        print(f"[rustnode] disabled: {exc}", file=sys.stderr)
        _NODES[format_id] = None
    return _NODES[format_id]


def reset() -> None:
    """Closes the warm processes and forgets them, so the next call decides afresh.

    For a caller that is turning the bridge on and off deliberately -- a test, or a tool
    timing both paths. It is not the same as giving up on a broken one.
    """
    global _GAVE_UP, _RESTARTS
    _GAVE_UP = False
    _RESTARTS = 0
    for key in list(_NODES):
        node = _NODES.pop(key)
        if node is None:
            continue
        with contextlib.suppress(Exception):
            node.close()


#: How many times a failure may be answered by starting a fresh process before the bridge
#: is given up on for good.
#:
#: One failure used to end it: the process fell back to Python and stayed there, at a
#: twentieth of the speed, for however many games were left. Two generation workers hit
#: that and did not finish. A node process holds no state between requests -- the
#: regulation is loaded from a file and every request carries its own position -- so
#: starting another one costs a second and loses nothing.
RESTARTS_ALLOWED = int(os.environ.get("POKEURAOU_RUST_NODE_RESTARTS", "3"))

_RESTARTS = 0


def disable(reason: str) -> None:
    """Answers a failed bridge with a fresh process, or gives up if that keeps happening.

    Giving up is still the end state -- a broken bridge must not cost a subprocess per
    node on top of Python's own time -- but it is no longer the first move.
    """
    global _GAVE_UP, _RESTARTS
    reset()
    if _RESTARTS < RESTARTS_ALLOWED:
        _RESTARTS += 1
        print(
            f"[rustnode] restarting the node ({_RESTARTS} of {RESTARTS_ALLOWED}): {reason}",
            file=sys.stderr,
        )
        return
    print(f"[rustnode] falling back to Python: {reason}", file=sys.stderr)
    _GAVE_UP = True


class RustNode:
    """A warm subprocess. One per worker; it holds the regulation in memory."""

    def __init__(self, reg: Regulation, binary: Path | None = None) -> None:
        self.format_id = reg.meta.format_id
        self.binary = binary or binary_path()
        regulation = repo_root() / "configs" / "regulations" / f"{self.format_id}.json"
        # Binary, not text: an encoded node is a JSON header line followed by the raw
        # little-endian arrays on the same pipe, and a text stream would mangle them.
        #
        # stderr goes to a temporary *file*, not a pipe. A pipe nobody drains fills at
        # about 64 KB and blocks the child inside `eprintln!`, with the parent blocked
        # reading stdout -- the two wait on each other and the worker never moves again.
        # That is not hypothetical: a generation run at width 48 had one worker do zero
        # games in ninety seconds while its eleven siblings did fourteen to twenty-seven,
        # with nothing in its log. A file never blocks, and the diagnostics survive.
        # Held for the life of the process and closed in `close`, so a context
        # manager is the wrong shape here.
        self._errors = tempfile.TemporaryFile()  # noqa: SIM115
        self._process = subprocess.Popen(
            [str(self.binary), "node", str(regulation)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._errors,
        )
        # One thread, so a read that never returns can be abandoned. Killing the child
        # closes the pipe, which is what actually unblocks it.
        self._reader = ThreadPoolExecutor(max_workers=1)
        #: The block an encoded node's arrays are written into. None until a node asks
        #: for one, and then as large as the node that asked.
        self._shm: Any = None
        #: Set when a block could not be made, so the pipe is not re-refused per node.
        self._shm_off = SHM_MAX_BYTES <= 0

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.stdin.close()
            self._process.wait(timeout=5)
        self._reader.shutdown(wait=False)
        self._errors.close()
        # After the child is gone, so nothing is reading it when the pages go away.
        _release(self._shm)
        self._shm = None

    def _stderr_text(self) -> str:
        """Whatever the child has written to stderr so far. Never blocks."""
        try:
            self._errors.seek(0)
            return self._errors.read().decode("utf-8", "replace").strip()
        except OSError:
            return ""

    def _with_deadline(self, call, what: str):  # noqa: ANN001, ANN202
        """Run a blocking read with a deadline; a child that wedges becomes a fallback.

        The deadline is generous on purpose. A width-48 node can be tens of megabytes and
        the machine may be running a dozen of these at once, so this is not a latency
        budget -- it is the difference between a run that finishes slowly and one that
        does not finish. On expiry the child is killed, which closes the pipe and releases
        the thread still blocked inside the read.
        """
        try:
            return self._reader.submit(call).result(timeout=NODE_TIMEOUT)
        except FuturesTimeout:
            self._process.kill()
            raise RuntimeError(
                f"the Rust node did not answer within {NODE_TIMEOUT:.0f}s ({what}); "
                f"killed it. {self._stderr_text()}"
            ) from None

    @timing.timed("rust.score")
    def score(
        self,
        pos: Position,
        side: int,
        candidates: list[SideAction],
    ) -> list[tuple[float, list[tuple[int, int, bool, float, bool]]]] | None:
        """The damage score of each candidate, or None when the port declines the position.

        What decides which choices reach the matrix at all. The legality of the pool is
        settled here in Python -- it is 0.2% of a run and 500 lines of rules -- and only
        the arithmetic crosses.
        """
        request = {
            "kind": "score",
            "position": pos.to_json(),
            "side": side,
            "candidates": [[dump_action(a) for a in c.slots] for c in candidates],
        }
        response = self._exchange(request)
        if response.get("refused"):
            return None
        return [
            (float(score), [(int(a), int(b), bool(c), float(d), bool(e)) for a, b, c, d, e in parts])
            for score, parts in zip(response["scores"], response["detail"], strict=True)
        ]

    @timing.timed("rust.resolve")
    def resolve(
        self,
        pos: Position,
        actions: list[SideAction],
        budget: Budget,
        select: int | None = None,
    ) -> ResolvedTurn | None:
        """One turn, for advancing a game. None when the port declines it.

        Two calls rather than one: the weights come back first so the caller can sample an
        index with its own generator -- which is what keeps a generated game identical to
        one played without this bridge -- and only then is the chosen position asked for.
        An exact budget produces a hundred-odd branches, and sending all of them would be
        two megabytes a turn against thirty kilobytes for this.
        """
        request = {
            "kind": "resolve",
            "position": pos.to_json(),
            "actions": [[dump_action(a) for a in side.slots] for side in actions],
            "budget": dump_budget(budget),
            "select": select,
        }
        response = self._exchange(request)
        if response.get("refused"):
            return None
        raw = response.get("position")
        return ResolvedTurn(
            branches=list(response["branches"]),
            suspended=list(response.get("suspended", [])),
            exact=bool(response["exact"]),
            unmodelled=tuple(response["unmodelled"]),
            position=Position.from_json(raw) if raw else None,
        )

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        """One request out, one header line back, with this end's JSON named.

        `rust.ask` and `rust.header` are this process's share of the protocol being JSON,
        as `rust.child.parse` and `rust.child.header` are the child's. Four rows for one
        question -- whether a crossing should stop being text -- because the answer is
        different at each end and the fixes are different too.
        """
        with timing.stage("rust.ask"):
            payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
        self._process.stdin.write(payload + b"\n")
        self._process.stdin.flush()
        line = self._with_deadline(self._process.stdout.readline, "header")
        if not line:
            raise RuntimeError(f"the Rust node process stopped: {self._stderr_text()}")
        with timing.stage("rust.header"):
            response = json.loads(line.decode("utf-8"))
        if "error" in response:
            raise RuntimeError(f"the Rust node refused the request: {response['error']}")
        return response

    def _grow_to(self, body_bytes: int) -> Any:
        """A block at least `body_bytes` long, or None when one cannot be had.

        Called when the child says the node it has just resolved does not fit what this
        process holds -- which for the first node of a process is always, since it holds
        nothing. A quarter of headroom on top, because a node's neighbours are near its
        size and every growth costs the child a fresh attachment, which is every page of
        the block faulted in again.
        """
        if self._shm_off or body_bytes > SHM_MAX_BYTES:
            return None
        # Imported here rather than at the top: this module is loaded by every worker and
        # a block is only ever made by one that actually fills an encoded node.
        from multiprocessing import shared_memory

        want = min(int(body_bytes * 1.25), SHM_MAX_BYTES)
        try:
            fresh = shared_memory.SharedMemory(create=True, size=want)
        except Exception as exc:  # noqa: BLE001 - the pipe is always there to fall back to
            print(
                f"[rustnode] a {want / 1e6:.0f} MB block was refused ({exc}); "
                "the arrays keep the pipe",
                file=sys.stderr,
            )
            # Once, not once a node: a machine that cannot spare the commit now will not
            # spare it in a hundred milliseconds either.
            self._shm_off = True
            return None
        # Only after the new one exists, so a failure leaves the old one in place.
        _release(self._shm)
        self._shm = fresh
        return fresh

    def _body(self, header: dict[str, Any], count: int) -> tuple[str, bytearray]:
        """The node's arrays, by whichever road the child's header named.

        Three of them. `shm` means they are already in the block this process holds and
        only have to be copied out. `grow` means the child has them and nothing here fits:
        it is waiting to be told where to put them, and this makes a block of the size it
        just named. Anything else -- `pipe`, or no `via` at all from a binary built before
        any of this -- means they are following down the pipe.
        """
        road = header.get("via")
        if road == "shm" and self._shm is not None:
            return "shm", self._copy_shared(self._shm, count)
        if road != "grow":
            return "pipe", self._read_exactly(count)
        block = self._grow_to(count)
        answer = (
            {"name": block.name, "bytes": block.size}
            if block is not None
            else {"pipe": True}
        )
        self._process.stdin.write(json.dumps(answer).encode("utf-8") + b"\n")
        self._process.stdin.flush()
        line = self._with_deadline(self._process.stdout.readline, "the block's answer")
        if not line:
            raise RuntimeError(f"the Rust node process stopped: {self._stderr_text()}")
        settled = json.loads(line.decode("utf-8")).get("via")
        if settled == "shm" and block is not None:
            return "shm", self._copy_shared(block, count)
        return "pipe", self._read_exactly(count)

    @timing.timed("rust.body")
    def _copy_shared(self, block: Any, count: int) -> bytearray:
        """The body out of the block, as the writable buffer the arrays are built over.

        One copy at memory speed, where the pipe was a round trip per buffer. A copy and
        not a view on purpose: the block is written again by the next node, and an array
        still pointing into it would be read after the child had overwritten it -- a wrong
        answer with nothing to see, which is not a trade worth one memcpy.
        """
        return bytearray(block.buf[:count])

    @timing.timed("rust.body")
    def _read_exactly(self, count: int) -> bytearray:
        """The blob that follows an encoded node's header, in full.

        A `bytearray` rather than `bytes` so the arrays built over it are writable. torch
        warns about a tensor sharing read-only memory -- writing through it is undefined --
        and the alternative is a copy of every float block on arrival.
        """
        body = bytearray()
        while len(body) < count:
            want = count - len(body)
            chunk = self._with_deadline(
                lambda n=want: self._process.stdout.read(n), f"{count} byte body"
            )
            if not chunk:
                raise RuntimeError(
                    f"the Rust node sent {len(body)} of {count} bytes: "
                    f"{self._stderr_text()}"
                )
            body += chunk
        return body

    @timing.timed("rust.fill")
    def fill_encoded(
        self,
        pos: Position,
        ours: list[SideAction],
        theirs: list[SideAction],
        budget: Budget,
        objectives: list[str] | None = None,
        cells: Sequence[tuple[int, int]] | None = None,
    ) -> EncodedNode:
        """The node's leaves, already encoded, and how to fold their values.

        For a learned leaf: its input is the leaves, so the leaves have to cross -- but as
        the encoder's arrays rather than as positions, which is 3.7 KB each instead of
        15 KB of JSON to parse and then encode anyway.
        """
        started = timing.clock()
        request = {
            "position": pos.to_json(),
            "ours": [[dump_action(a) for a in side.slots] for side in ours],
            "theirs": [[dump_action(a) for a in side.slots] for side in theirs],
            "budget": dump_budget(budget),
            # Named objectives asked for alongside: scored per leaf over there, since the
            # leaves are there already.
            "objectives": list(objectives or []),
            "encode": True,
        }
        # Only these cells, when the caller is solving rather than tabulating.
        if cells is not None:
            request["cells"] = [[int(i), int(j)] for i, j in cells]
        # What this process holds, if anything. A `shm` key with no name says "I hold
        # none, but I will make one" -- which is what the first node of every process
        # sends, and how a block ends up the size of the node that needed it.
        if not self._shm_off:
            request["shm"] = (
                {"name": self._shm.name, "bytes": self._shm.size}
                if self._shm is not None
                else {"name": None, "bytes": 0}
            )
        header = self._exchange(request)
        count = int(header["bytes"])
        road, body = self._body(header, count)
        node = EncodedNode.unpack(header, body)
        # What the child says it spent, on its own clock. From this side the two
        # are one wait on a pipe, so there is no other honest source for the split.
        timing.add("rust.child.resolve", node.resolve_us / 1e6)
        timing.add("rust.child.encode", node.encode_us / 1e6)
        timing.add("rust.child.parse", node.parse_us / 1e6)
        timing.add("rust.child.header", node.header_us / 1e6)
        # What the crossing actually carried, and by which road. The size of a body used
        # to be arrived at by multiplying a leaf count by a per-leaf figure from another
        # file, and a conversion is not a measurement.
        timing.count("body.bytes", count)
        timing.count("body.shm" if road == "shm" else "body.pipe")
        # And how much of the node the port's own leaf sharing took off the crossing. The
        # ratio has been read off `Collector.seen` and never counted; this is the count.
        timing.count("leaves.offered", int(header.get("offered", 0)))
        timing.count("leaves.stored", int(header["leaves"]))
        if timing.ON:
            # The same call cut by what it was for (IKA-98): its whole wall clock here,
            # nestings included, and the child's four clocks together.
            used = timing.current_purpose()
            timing.add(f"rust.fill@{used}", timing.clock() - started)
            timing.add(
                f"rust.child@{used}",
                (node.resolve_us + node.encode_us + node.parse_us + node.header_us) / 1e6,
            )
            timing.count("fills")
            timing.count("fill.cells", len(cells) if cells is not None else len(ours) * len(theirs))
            timing.count("fill.leaves", int(header["leaves"]))
        return node

    @timing.timed("rust.fill")
    def fill(
        self,
        pos: Position,
        ours: list[SideAction],
        theirs: list[SideAction],
        objectives: list[str],
        budget: Budget,
        cells: Sequence[tuple[int, int]] | None = None,
    ) -> NodeResult:
        started = timing.clock()
        request = {
            "position": pos.to_json(),
            "ours": [[dump_action(a) for a in side.slots] for side in ours],
            "theirs": [[dump_action(a) for a in side.slots] for side in theirs],
            "budget": dump_budget(budget),
            "objectives": objectives,
        }
        if cells is not None:
            request["cells"] = [[int(i), int(j)] for i, j in cells]
        response = self._exchange(request)
        if timing.ON:
            # By purpose, as `fill_encoded`. This answer carries no clocks of the child's,
            # so it adds to `rust.fill@` and not to `rust.child@`, nor to `fill.leaves`.
            timing.add(f"rust.fill@{timing.current_purpose()}", timing.clock() - started)
            timing.count("fills")
            timing.count("fill.cells", len(cells) if cells is not None else len(ours) * len(theirs))
        return NodeResult(
            payoffs=response["payoffs"],
            exact=response["exact"],
            refused=[(int(i), int(j), str(why)) for i, j, why in response["refused"]],
            unmodelled=tuple(response["unmodelled"]),
        )
