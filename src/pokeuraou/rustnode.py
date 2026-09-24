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

On by default since IKA-209: the port is the only resolver the production roads have,
so a missing or stale binary stops them (`require_node`) rather than falling back::

    POKEURAOU_RUST_NODE=0                      # off: every road stops (PortUnavailable)
    POKEURAOU_RUST_NODE_BIN=/path/to/binary    # defaults to rust/target/release/
    POKEURAOU_RUST_NODE_SHM_MB=512             # the largest block; 0 keeps the pipe

`available()` answers without raising (it was for resolve.py's `batched_payoffs`, which had
Python behind it until IKA-212 deleted it). `require_node` is every road's door.
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
from .budget import Budget
from .position import Position
from .regulation import Regulation, repo_root

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
    # On unless switched off (IKA-209). Before, unset meant off.
    return os.environ.get(ENV_ENABLE, "1").strip().lower() not in ("0", "false", "no")


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
    #: The encoding rule the child says it applied (`megaFromSlots`), read off its own
    #: header rather than off the request -- what a worker echoes per arm (IKA-141). None
    #: from a binary that predates the field.
    mega_from_slots: bool | None = None

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
            mega_from_slots=(
                bool(header["encoding"]["megaFromSlots"])
                if isinstance(header.get("encoding"), dict)
                and "megaFromSlots" in header["encoding"]
                else None
            ),
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


def disable(reason: str) -> bool:
    """Answers a failed bridge with a fresh process, or gives up if that keeps happening.

    Giving up is still the end state -- a broken bridge must not cost a subprocess per
    node on top of Python's own time -- but it is no longer the first move.

    True when a fresh process was started, False once the restarts have run out; the
    roads then stop (`port.ask`, IKA-209). There is no Python to fall back to (IKA-212).
    """
    global _GAVE_UP, _RESTARTS
    reset()
    if _RESTARTS < RESTARTS_ALLOWED:
        _RESTARTS += 1
        print(
            f"[rustnode] restarting the node ({_RESTARTS} of {RESTARTS_ALLOWED}): {reason}",
            file=sys.stderr,
        )
        return True
    print(f"[rustnode] giving up on the node: {reason}", file=sys.stderr)
    _GAVE_UP = True
    return False


class PortUnavailable(RuntimeError):
    """The production roads were asked for the port and there is none to ask (IKA-209)."""


def require_node(reg: Regulation) -> RustNode:
    """The warm process for this regulation, or a stop that says why there is none.

    For the production roads -- generation, the analyser, the selection solve -- which
    have no Python resolver behind them any more (IKA-209). Where `node_for` answers None
    and lets the caller fall back, this raises: switched off, no binary, a binary older
    than the sources (as `require_current_binary`), or a bridge that was given up on.
    """
    if not enabled():
        raise PortUnavailable(
            f"{ENV_ENABLE}=0, and the production roads have no Python resolver to fall "
            "back to (IKA-209); unset it"
        )
    format_id = reg.meta.format_id
    held = _NODES.get(format_id)
    if held is not None:
        return held
    if _GAVE_UP:
        raise PortUnavailable("the Rust node failed and was given up on; see the log above")
    path = binary_path()
    if not path.exists():
        raise PortUnavailable(
            f"no Rust binary at {path}; `cd rust && cargo build --release`"
        )
    stale = sources_newer_than_binary()
    if stale:
        listed = ", ".join(stale[:4]) + ("..." if len(stale) > 4 else "")
        raise PortUnavailable(
            f"{path.name} was built before {listed} changed; the port is the only "
            "resolver, so `cd rust && cargo build --release` first"
        )
    made = RustNode(reg)
    _NODES[format_id] = made
    return made


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
            self.refusal = str(response["refused"])
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
        if timing.DUPES:
            _note_repeat(request)
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

    def _exchange_many(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Many requests down the pipe while their answers come back, in order (IKA-209).

        For stateless requests only -- each answer is read as its own header line, so a
        request whose answer has a body (an encoded node) must not be sent this way. The
        child answers them one at a time either way; what this saves is the wait between
        one answer and the next request, which for a small request is most of its cost. A
        writer thread feeds the child while this one reads, so neither pipe can fill with
        both ends waiting on it.
        """
        import threading

        with timing.stage("rust.ask"):
            payloads = [
                json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n"
                for request in requests
            ]
        if timing.DUPES:
            for request in requests:
                _note_repeat(request)
        failure: list[BaseException] = []

        def feed() -> None:
            try:
                for payload in payloads:
                    self._process.stdin.write(payload)
                self._process.stdin.flush()
            except BaseException as exc:  # noqa: BLE001 - reported by the reader
                failure.append(exc)

        writer = threading.Thread(target=feed, daemon=True)
        writer.start()
        answers: list[dict[str, Any]] = []
        for _request in requests:
            line = self._with_deadline(self._process.stdout.readline, "header")
            if not line:
                raise RuntimeError(
                    f"the Rust node process stopped: {failure or ''} {self._stderr_text()}"
                )
            with timing.stage("rust.header"):
                response = json.loads(line.decode("utf-8"))
            if "error" in response:
                raise RuntimeError(f"the Rust node refused the request: {response['error']}")
            answers.append(response)
        writer.join()
        if failure:
            raise RuntimeError(f"writing to the Rust node failed: {failure[0]}")
        return answers

    @timing.timed("rust.leads")
    def apply_lead_abilities_many(self, positions: Sequence[Position]) -> list[PortPhase | None]:
        """`apply_lead_abilities` without a generator, for many positions in one go."""
        answers = self._exchange_many(
            [{"kind": "leads", "position": pos.to_json()} for pos in positions]
        )
        out: list[PortPhase | None] = []
        for response in answers:
            if response.get("refused"):
                self.refusal = str(response["refused"])
                out.append(None)
            else:
                out.append(PortPhase.read(response))
        return out

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
        *,
        rules: Any = None,  # noqa: ANN401 - EncodingRules; encode imports numpy
    ) -> EncodedNode:
        """The node's leaves, already encoded, and how to fold their values.

        For a learned leaf: its input is the leaves, so the leaves have to cross -- but as
        the encoder's arrays rather than as positions, which is 3.7 KB each instead of
        15 KB of JSON to parse and then encode anyway.

        `rules` is the asking leaf's `EncodingRules`. Per request, not per process: one
        worker plays both arms of a match through this one child (IKA-141). A child that
        does not echo the rule it was asked for is refused -- a binary built before the
        field exists would otherwise encode an old-rule arm with the new rule and say
        nothing.
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
        # Sent only for a leaf that asks for an undone fix, so every other request is the
        # bytes it always was.
        wants_old = bool(rules is not None and rules.mega_from_slots)
        if wants_old:
            request["encoding"] = rules.to_request()
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
        if wants_old and node.mega_from_slots is not True:
            raise RuntimeError(
                "the Rust node was asked for the revision-1 can_mega rule and did not say "
                f"it applied it (echo {node.mega_from_slots!r}); the binary predates "
                "IKA-141 -- rebuild it"
            )
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

    # -- What only Python's resolver answered (IKA-211). The production roads reach these
    # -- through `port` (IKA-209).

    #: Why the port declined the last request it declined -- for a caller that stops on a
    #: refusal and has to say what it was (IKA-209).
    refusal: str | None = None

    def _ask(self, request: dict[str, Any]) -> dict[str, Any] | None:
        response = self._exchange(request)
        if response.get("refused"):
            self.refusal = str(response["refused"])
            return None
        return response

    @timing.timed("rust.turn")
    def turn(
        self,
        pos: Position,
        actions: list[SideAction],
        budget: Budget,
        *,
        full: bool = False,
        select: int | None = None,
        events: bool = False,
    ) -> PortTurn | None:
        """`resolve_turn` through the port: every branch with `full`, and with `select` the
        outcome at that index of the branches followed by the pauses -- a pause included,
        which `resolve` could only give the weight of. None when the port declines.

        `events` asks for each outcome's trace, Python's `Branch.events` and `acts`
        (IKA-215). Off by default: the port then formats and keeps nothing."""
        response = self._ask(
            {
                "kind": "turn",
                "position": pos.to_json(),
                "actions": [[dump_action(a) for a in side.slots] for side in actions],
                "budget": dump_budget(budget),
                "full": full,
                "select": select,
                "events": events,
            }
        )
        return None if response is None else PortTurn.read(response)

    @timing.timed("rust.resume")
    def resume(
        self,
        pause: PortPause,
        choices: list[SideAction],
        *,
        full: bool = False,
        select: int | None = None,
        world: tuple[Position, int] | None = None,
        events: bool = False,
    ) -> PortTurn | None:
        """`resume_turn`: the rest of a paused turn with both sides' replacement choices.
        `world` is `paused_in`'s position and side: the pause resumed in that completion.
        With `events` the trace goes on from the pause's own, as Python's does."""
        response = self._ask(
            {
                "kind": "turn",
                "pause": pause.raw,
                "choices": [[dump_action(a) for a in side.slots] for side in choices],
                "full": full,
                "select": select,
                "in": _world(world),
                "events": events,
            }
        )
        return None if response is None else PortTurn.read(response)

    @timing.timed("rust.alternatives")
    def resume_alternatives(
        self,
        pause: PortPause,
        *,
        world: tuple[Position, int] | None = None,
        full: bool = True,
        events: bool = False,
    ) -> tuple[int | None, list[tuple[SideAction, PortTurn]]] | None:
        """`resume_alternatives` (and, with `world`, of `paused_in`'s pause): who chooses, and
        every replacement with the turn it produces, in Python's order."""
        response = self._ask(
            {
                "kind": "alternatives",
                "pause": pause.raw,
                "in": _world(world),
                "full": full,
                "events": events,
            }
        )
        if response is None:
            return None
        chooser = response["chooser"]
        return (
            None if chooser is None else int(chooser),
            [
                (_side_action(option), PortTurn.read(result))
                for option, result in zip(response["options"], response["results"], strict=True)
            ],
        )

    @timing.timed("rust.alternatives")
    def alternatives_encoded(
        self,
        pause: PortPause,
        *,
        world: tuple[Position, int] | None = None,
        want: Sequence[int] | None = None,
        shared: tuple[int, Sequence[int]] | None = None,
        rules: Any = None,  # noqa: ANN401 - EncodingRules
        objectives: Sequence[str] = (),
        encode: bool = True,
    ) -> EncodedAlternatives | None:
        """`resume_alternatives` with each wanted option's turn flattened over there, as
        `turn_leaves` flattens it, and every leaf encoded (IKA-209).

        What the self-switch node scores: the leaves cross as the encoder's arrays, by the
        road an encoded node's take, instead of as positions. `shared=(side, slots)` also
        answers `_shared_self_switch_plans`' question per option. `objectives` are
        scored over there per leaf; with `encode=False` no arrays are built -- an hp-share
        node wants the values alone. None when the port declines the pause.
        """
        request: dict[str, Any] = {
            "kind": "alternativesEncoded",
            "pause": pause.raw,
            "in": _world(world),
        }
        if want is not None:
            request["want"] = [int(k) for k in want]
        if objectives:
            request["objectives"] = list(objectives)
        if not encode:
            request["encode"] = False
        if shared is not None:
            request["shared"] = {"side": int(shared[0]), "slots": [int(s) for s in shared[1]]}
        wants_old = bool(rules is not None and rules.mega_from_slots)
        if wants_old:
            request["encoding"] = rules.to_request()
        if not self._shm_off:
            request["shm"] = (
                {"name": self._shm.name, "bytes": self._shm.size}
                if self._shm is not None
                else {"name": None, "bytes": 0}
            )
        header = self._exchange(request)
        if header.get("refused"):
            self.refusal = str(header["refused"])
            return None
        count = int(header["bytes"])
        _road, body = self._body(header, count)
        node = EncodedNode.unpack(header, body)
        if wants_old and node.mega_from_slots is not True:
            raise RuntimeError(
                "the Rust node was asked for the revision-1 can_mega rule and did not say "
                f"it applied it (echo {node.mega_from_slots!r}); the binary predates IKA-141"
            )
        timing.count("body.bytes", count)
        import numpy as np

        rows = int(header.get("valueRows", 0))
        names = list(header.get("leafObjectives", []))
        # The values follow the arrays, one block of `rows` per objective.
        offset = count - rows * 8 * len(names)
        values = {
            name: np.frombuffer(body, dtype=np.float64, count=rows, offset=offset + k * rows * 8)
            for k, name in enumerate(names)
        }
        chooser = header["chooser"]
        return EncodedAlternatives(
            chooser=None if chooser is None else int(chooser),
            options=[_side_action(option) for option in header["options"]],
            plans=[
                None
                if "start" not in plan
                else EncodedPlan(
                    start=int(plan["start"]),
                    count=int(plan["count"]),
                    fold=plan["fold"],
                    unmodelled=tuple(plan["unmodelled"]),
                    suspended=bool(plan["suspended"]),
                )
                for plan in header["plans"]
            ],
            untouched=[plan.get("untouched") for plan in header["plans"]],
            node=node,
            values=values,
        )

    @timing.timed("rust.replacements")
    def resolve_replacements(
        self,
        pos: Position,
        choices: list[SideAction],
        *,
        rng: Any = None,  # noqa: ANN401 - numpy.random.Generator
        events: bool = False,
    ) -> PortPhase | None:
        """`resolve_replacements`. A draw inside a switch-in is sampled from `rng` exactly as
        Python's `_draw` samples it -- same call, same order -- or is the first, noted."""
        return self._phase(
            {
                "kind": "replacements",
                "position": pos.to_json(),
                "choices": [[dump_action(a) for a in side.slots] for side in choices],
                "events": events,
            },
            rng,
        )

    @timing.timed("rust.leads")
    def apply_lead_abilities(
        self,
        pos: Position,
        *,
        rng: Any = None,  # noqa: ANN401 - numpy.random.Generator
        events: bool = False,
    ) -> PortPhase | None:
        """`apply_lead_abilities`, with its draws answered as `resolve_replacements`'."""
        return self._phase({"kind": "leads", "position": pos.to_json(), "events": events}, rng)

    @timing.timed("rust.needed")
    def replacements_needed(self, pos: Position) -> tuple[tuple[bool, ...], ...] | None:
        response = self._ask({"kind": "needed", "position": pos.to_json()})
        if response is None:
            return None
        return tuple(tuple(bool(flag) for flag in side) for side in response["needed"])

    def _phase(self, request: dict[str, Any], rng: Any) -> PortPhase | None:  # noqa: ANN401
        """Asks for a phase, and once more per draw: the port replays the choices made so far
        and hands back the weights of the next one, which is sampled here."""
        if rng is None:
            response = self._ask(request)
            return None if response is None else PortPhase.read(response)
        presets: list[int] = []
        while True:
            response = self._ask({**request, "presets": presets})
            if response is None:
                return None
            weights = response.get("draw")
            if not weights:
                return PortPhase.read(response)
            total = float(sum(weights))
            presets.append(int(rng.choice(len(weights), p=[w / total for w in weights])))


def _note_repeat(request: dict[str, Any]) -> None:
    """IKA-258: count a request whose content this decision already sent (`timing.repeat`).

    Keyed on the request less `shm`, which names this process's block and not the question.
    Only called when `timing.DUPES` is on.
    """
    kind = request.get("kind") or ("fill_encoded" if request.get("encode") else "fill")
    asked = {key: value for key, value in request.items() if key != "shm"}
    digest = hashlib.blake2b(
        json.dumps(asked, ensure_ascii=False).encode("utf-8"), digest_size=16
    ).digest()
    asker = timing.caller()
    timing.repeat_where(f"port.{kind}", digest, where=asker)
    # The position alone, whatever was asked of it: what a child that kept the last few
    # positions it parsed would not have to be sent again.
    if "position" in request:
        where = hashlib.blake2b(
            json.dumps(request["position"], ensure_ascii=False).encode("utf-8"),
            digest_size=16,
        ).digest()
        timing.repeat_where("port.position", where, where=asker)
        # And a node's cells one by one: a cell of the leaf ranking's fill that the matrix
        # fills again, on the same position with the same two actions, is the same turn.
        if "ours" in request and "theirs" in request:
            ours = [json.dumps(a, sort_keys=True) for a in request["ours"]]
            theirs = [json.dumps(a, sort_keys=True) for a in request["theirs"]]
            cells = request.get("cells") or [
                (i, j) for i in range(len(ours)) for j in range(len(theirs))
            ]
            for i, j in cells:
                timing.repeat("port.cell", (where, ours[i], theirs[j]))


def _world(world: tuple[Position, int] | None) -> dict[str, Any] | None:
    if world is None:
        return None
    position, side = world
    return {"position": position.to_json(), "side": int(side)}


def _side_action(slots: list[dict[str, Any]]) -> SideAction:
    out: list[Any] = []
    for entry in slots:
        if entry["kind"] == "switch":
            out.append(
                SwitchAction(
                    slot=int(entry["slot"]),
                    party_index=int(entry["partyIndex"]),
                    species=str(entry["species"]),
                )
            )
        else:
            out.append(PassAction(slot=int(entry["slot"])))
    return SideAction(slots=tuple(out))


@dataclass
class PortPause:
    """A turn the port stopped for a mid-turn replacement: Python's `SuspendedTurn`.

    `raw` is what the port wrote -- the position and the continuation -- and is handed back
    whole to resume it. The node keeps nothing between requests, so the pause is data here
    rather than an id into the process.
    """

    probability: float
    position: Position
    raw: dict[str, Any]
    #: The trace up to the pause, when it was asked for (IKA-215).
    events: list[str] = field(default_factory=list)
    acts: list[tuple[int, str]] = field(default_factory=list)

    @staticmethod
    def read(raw: dict[str, Any]) -> PortPause:
        return PortPause(
            probability=float(raw["probability"]),
            position=Position.from_json(raw["position"]),
            raw=raw,
            events=list(raw.get("events") or []),
            acts=_acts(raw),
        )


def _acts(raw: dict[str, Any]) -> list[tuple[int, str]]:
    return [(int(start), str(label)) for start, label in raw.get("acts") or []]


@dataclass
class EncodedPlan:
    """One option's turn as `turn_leaves` flattens it: its rows of the answer's arrays
    (`start`, `count`) and the fold over them, with leaf indices local to the plan."""

    start: int
    count: int
    fold: dict[str, Any]
    unmodelled: tuple[str, ...]
    #: Whether the resumed turn paused again (it is then not one the shared path can use).
    suspended: bool


@dataclass
class EncodedAlternatives:
    """`alternatives_encoded`'s answer: who chooses, every option, the wanted options'
    plans (None for the others), `shared`'s answer per option, and the arrays."""

    chooser: int | None
    options: list[SideAction]
    plans: list[EncodedPlan | None]
    untouched: list[bool | None]
    node: EncodedNode
    #: One value per leaf for each objective asked for, scored over there.
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortBranch:
    probability: float
    position: Position
    #: Python's `Branch.events` and `acts`, when the turn was asked for them (IKA-215).
    events: list[str] = field(default_factory=list)
    acts: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class PortTurn:
    """One turn's answer. `branches`/`suspended` are the weights; with `full` the outcomes
    themselves are in `outcomes`/`pauses`, and a `select` is in `position` or `pause`."""

    branches: list[float]
    suspended: list[float]
    exact: bool
    unmodelled: tuple[str, ...]
    outcomes: list[PortBranch] | None = None
    pauses: list[PortPause] | None = None
    position: Position | None = None
    pause: PortPause | None = None
    #: The trace of the branch `select` named, when events were asked for.
    events: list[str] = field(default_factory=list)
    acts: list[tuple[int, str]] = field(default_factory=list)

    @staticmethod
    def read(response: dict[str, Any]) -> PortTurn:
        branches = response["branches"]
        suspended = response["suspended"]
        full = bool(branches and isinstance(branches[0], dict)) or bool(
            suspended and isinstance(suspended[0], dict)
        )
        chosen = response.get("position")
        paused = response.get("pause")
        if full or (not branches and not suspended):
            outcomes = [
                PortBranch(
                    float(b["probability"]),
                    Position.from_json(b["position"]),
                    list(b.get("events") or []),
                    _acts(b),
                )
                for b in branches
            ]
            pauses = [PortPause.read(p) for p in suspended]
            return PortTurn(
                branches=[b.probability for b in outcomes],
                suspended=[p.probability for p in pauses],
                exact=bool(response["exact"]),
                unmodelled=tuple(response["unmodelled"]),
                outcomes=outcomes,
                pauses=pauses,
                position=Position.from_json(chosen) if chosen else None,
                pause=PortPause.read(paused) if paused else None,
                events=list(response.get("events") or []),
                acts=_acts(response),
            )
        return PortTurn(
            branches=[float(w) for w in branches],
            suspended=[float(w) for w in suspended],
            exact=bool(response["exact"]),
            unmodelled=tuple(response["unmodelled"]),
            position=Position.from_json(chosen) if chosen else None,
            pause=PortPause.read(paused) if paused else None,
            events=list(response.get("events") or []),
            acts=_acts(response),
        )


@dataclass
class PortPhase:
    """A replacement phase or the leads' switch-ins: Python's `ReplacementResult`. `events`
    is the phase's trace when it was asked for (IKA-215), else empty."""

    position: Position
    unmodelled: tuple[str, ...] = ()
    events: list[str] = field(default_factory=list)

    @staticmethod
    def read(response: dict[str, Any]) -> PortPhase:
        return PortPhase(
            position=Position.from_json(response["position"]),
            unmodelled=tuple(response["unmodelled"]),
            events=list(response.get("events") or []),
        )
