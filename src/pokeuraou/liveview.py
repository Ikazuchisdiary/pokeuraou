"""A screen for a game against a person (IKA-332): the agent's answer as it forms, the
board, and the person's input, in a browser.

**Three hops, three forms.**

1. Engine to host, inside one process: a callback with Python objects (`deepen.Step`, read
   by `progress.Reader` into a `progress.Snapshot` of numpy arrays). Nothing serialised.
2. Host to screen: WebSocket binary frames (`Wire`). A step -- up to ten a second, the
   big stream -- is a fixed layout of little-endian numbers: counts, the value, and per
   action a string id, its probability (f64, so the last one equals the recorded policy
   bit for bit) and its loss (f32), the opponent's per completion, and the principal
   variation as a pre-order list. Every string (an action's label, a bench, a branch's
   text) goes once, in a strings frame, the first time it is used, and is an id after.
   The events that come once a turn or less (the board, the sheets, the person's prompt,
   the turn's result, the end) are small JSON texts in an event frame: they are read once
   by the page, not streamed, and JSON keeps them readable.
3. The record (``--live-out``): the same frames, each with its length and the seconds
   since the start (`FileSink`), so a game can be shown again (`replay`) or read back in
   Python (`read_record`, `Decoder`).

**The server** is the standard library only: `http.server` serves the page
(`web/live.html`) and upgrades ``/ws`` to a WebSocket (RFC 6455, the part a browser uses:
unfragmented frames, text and binary, ping and close). A page that connects late is sent
the game so far: every event, and the steps of the decision in progress plus the last
step of each earlier one. The page sends the person's answers as text frames
(``{"line": "..."}``) and `WebPerson` hands them to the game.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import html
import json
import queue
import re
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .actions import SideAction
from .humanplay import MovePlan, Person, parse_choice, parse_selection
from .priors import SampledSet
from .progress import SLOT_SEPARATOR, PvBranch, PvNode, PvPair, Snapshot

# ----------------------------------------------------------------------------- the wire

#: Frame types (the first byte of every frame).
STRINGS = 1
EVENT = 2
STEP = 3

KINDS = ("start", "refine", "refused", "widen", "done")
READS = ("leaf", "deep", "refused")

#: The step frame's fixed head, after the type byte (see `Wire.step`).
_HEAD = struct.Struct("<BHIIIfIfIIIIIIHHddffHHH")


def _plain(value: Any) -> Any:  # noqa: ANN401
    """An event's payload as JSON-ready values (actions by their choice strings)."""
    if isinstance(value, SideAction):
        return value.to_choice()
    if isinstance(value, MovePlan):
        return value.to_json()
    if isinstance(value, np.ndarray):
        return [float(x) for x in value]
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "to_json"):
        return value.to_json()
    return value


class Wire:
    """Packs a game's events and snapshots into frames, keeping the table of strings sent."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        self._new: list[str] = []

    def intern(self, text: str) -> int:
        got = self.ids.get(text)
        if got is None:
            got = len(self.ids)
            self.ids[text] = got
            self._new.append(text)
        return got

    def _strings(self) -> list[bytes]:
        if not self._new:
            return []
        first = len(self.ids) - len(self._new)
        parts = [struct.pack("<BIH", STRINGS, first, len(self._new))]
        for text in self._new:
            raw = text.encode("utf-8")
            parts.append(struct.pack("<H", len(raw)) + raw)
        self._new = []
        return [b"".join(parts)]

    def event(self, kind: str, payload: Any) -> list[bytes]:  # noqa: ANN401
        body = json.dumps({"type": kind, **_plain(payload)}, ensure_ascii=False)
        return [bytes([EVENT]) + body.encode("utf-8")]

    def step(self, snap: Snapshot) -> list[bytes]:
        """A snapshot's frames: new strings first, then the step."""
        ours = np.array([self.intern(t) for t in snap.our_labels], dtype="<u4")
        theirs = np.array([self.intern(t) for t in snap.their_labels], dtype="<u4")
        parts = [
            bytes([STEP]),
            _HEAD.pack(
                KINDS.index(snap.kind), min(snap.depth, 0xFFFF), snap.decision, snap.turn,
                snap.step, snap.ms, snap.budget, snap.spent, snap.cells, snap.fills,
                snap.refines, snap.expanded, snap.refused, snap.probed,
                min(snap.widened, 0xFFFF), min(snap.swapped, 0xFFFF), snap.value0, snap.value,
                snap.read, snap.gap, len(ours), len(theirs), len(snap.classes),
            ),
            ours.tobytes(),
            np.asarray(snap.our_p, dtype="<f8").tobytes(),
            np.asarray(snap.our_loss, dtype="<f4").tobytes(),
            theirs.tobytes(),
            np.asarray(snap.their_p, dtype="<f4").tobytes(),
            np.asarray(snap.their_loss, dtype="<f4").tobytes(),
        ]
        for view in snap.classes:
            parts.append(struct.pack(
                "<ffII", view.weight, view.value, self.intern(SLOT_SEPARATOR.join(view.bench)),
                self.intern(SLOT_SEPARATOR.join(view.bench_ids)),
            ))
            parts.append(np.asarray(view.p, dtype="<f4").tobytes())
        self._pairs(snap.pv, parts)
        frame = b"".join(parts)
        return [*self._strings(), frame]

    def _pairs(self, pairs: Sequence[PvPair], parts: list[bytes]) -> None:
        parts.append(struct.pack("<B", len(pairs)))
        for pair in pairs:
            parts.append(struct.pack(
                "<IIhBffB", self.intern(pair.ours), self.intern(pair.theirs), pair.klass,
                READS.index(pair.read), pair.p, pair.value, len(pair.branches),
            ))
            for branch in pair.branches:
                self._branch(branch, parts)

    def _branch(self, branch: PvBranch, parts: list[bytes]) -> None:
        node = branch.node
        flags = (1 if branch.ended else 0) | (2 if node is not None else 0) | (
            4 if node is not None and node.more else 0
        )
        parts.append(struct.pack(
            "<ffIB", branch.weight, branch.value, self.intern(branch.what), flags
        ))
        if node is not None:
            parts.append(struct.pack("<fB", node.value, len(node.ours)))
            for label, p in node.ours:
                parts.append(struct.pack("<If", self.intern(label), p))
            parts.append(struct.pack("<B", len(node.theirs)))
            for label, p in node.theirs:
                parts.append(struct.pack("<If", self.intern(label), p))
            self._pairs(node.pairs, parts)


class Decoder:
    """Reads frames back (the page's reader, in Python): for tests and analysis."""

    def __init__(self) -> None:
        self.strings: list[str] = []

    def feed(self, frame: bytes) -> dict[str, Any] | None:
        kind = frame[0]
        if kind == STRINGS:
            first, count = struct.unpack_from("<IH", frame, 1)
            if first != len(self.strings):
                raise ValueError(f"strings from {first}, table has {len(self.strings)}")
            at = 7
            for _ in range(count):
                (size,) = struct.unpack_from("<H", frame, at)
                at += 2
                self.strings.append(frame[at:at + size].decode("utf-8"))
                at += size
            return None
        if kind == EVENT:
            return json.loads(frame[1:].decode("utf-8"))
        if kind == STEP:
            return self._step(frame)
        raise ValueError(f"unknown frame type {kind}")

    def _step(self, frame: bytes) -> dict[str, Any]:
        head = _HEAD.unpack_from(frame, 1)
        (kind, depth, decision, turn, step, ms, budget, spent, cells, fills, refines,
         expanded, refused, probed, widened, swapped, value0, value, read, gap, n_ours,
         n_theirs, n_classes) = head
        at = 1 + _HEAD.size

        def take(dtype: str, count: int) -> np.ndarray:
            nonlocal at
            size = np.dtype(dtype).itemsize * count
            got = np.frombuffer(frame, dtype=dtype, count=count, offset=at)
            at += size
            return got

        s = self.strings
        ours = [s[i] for i in take("<u4", n_ours)]
        our_p = take("<f8", n_ours)
        our_loss = take("<f4", n_ours)
        theirs = [s[i] for i in take("<u4", n_theirs)]
        their_p = take("<f4", n_theirs)
        their_loss = take("<f4", n_theirs)
        classes = []
        for _ in range(n_classes):
            weight, cvalue, bench, ids = struct.unpack_from("<ffII", frame, at)
            at += 16
            classes.append({"weight": weight, "value": cvalue,
                            "bench": [b for b in s[bench].split(SLOT_SEPARATOR) if b],
                            "benchIds": [b for b in s[ids].split(SLOT_SEPARATOR) if b],
                            "p": take("<f4", n_theirs).tolist()})

        def pairs() -> list[dict[str, Any]]:
            nonlocal at
            (count,) = struct.unpack_from("<B", frame, at)
            at += 1
            out = []
            for _ in range(count):
                o, t, klass, rd, p, v, nb = struct.unpack_from("<IIhBffB", frame, at)
                at += struct.calcsize("<IIhBffB")
                out.append({"ours": s[o], "theirs": s[t], "class": klass, "read": READS[rd],
                            "p": p, "value": v, "branches": [branch() for _ in range(nb)]})
            return out

        def top() -> list[tuple[str, float]]:
            nonlocal at
            (count,) = struct.unpack_from("<B", frame, at)
            at += 1
            out = []
            for _ in range(count):
                i, p = struct.unpack_from("<If", frame, at)
                at += 8
                out.append((s[i], p))
            return out

        def branch() -> dict[str, Any]:
            nonlocal at
            w, v, what, flags = struct.unpack_from("<ffIB", frame, at)
            at += struct.calcsize("<ffIB")
            got: dict[str, Any] = {"weight": w, "value": v, "what": s[what],
                                   "ended": bool(flags & 1), "more": bool(flags & 4)}
            if flags & 2:
                (nv,) = struct.unpack_from("<f", frame, at)
                at += 4
                got["node"] = {"value": nv, "ours": top(), "theirs": top(), "pairs": pairs()}
            return got

        pv = pairs()
        if at != len(frame):
            raise ValueError(f"step frame has {len(frame) - at} bytes left over")
        return {
            "type": "step", "kind": KINDS[kind], "depth": depth, "decision": decision,
            "turn": turn, "step": step, "ms": ms, "budget": budget, "spent": spent,
            "cells": cells, "fills": fills, "refines": refines, "expanded": expanded,
            "refused": refused, "probed": probed, "widened": widened, "swapped": swapped,
            "value0": value0, "value": value, "read": read, "gap": gap,
            "ours": ours, "ourP": our_p.tolist(), "ourLoss": our_loss.tolist(),
            "theirs": theirs, "theirP": their_p.tolist(), "theirLoss": their_loss.tolist(),
            "classes": classes, "pv": pv,
        }


# ----------------------------------------------------------------------------- the record


class FileSink:
    """The frames to a file: per frame ``<u32 length><f64 seconds since the start>`` and the
    frame. Written as they come, so a game stopped halfway is still a record."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.perf_counter) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("wb")
        self.clock = clock
        self.started = clock()

    def __call__(self, frame: bytes) -> None:
        self.handle.write(struct.pack("<Id", len(frame), self.clock() - self.started) + frame)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def read_record(path: Path) -> Iterator[tuple[float, bytes]]:
    """A `FileSink` file's frames, with their seconds."""
    data = path.read_bytes()
    at = 0
    while at < len(data):
        size, seconds = struct.unpack_from("<Id", data, at)
        at += 12
        yield seconds, data[at:at + size]
        at += size


# ----------------------------------------------------------------------------- the websocket

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_accept(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + _GUID).encode("ascii")).digest()).decode("ascii")


def ws_frame(payload: bytes, opcode: int = 2, *, mask: bytes | None = None) -> bytes:
    """One unfragmented frame (a server's unmasked; a client's with ``mask``)."""
    size = len(payload)
    head = bytearray([0x80 | opcode])
    bit = 0x80 if mask is not None else 0
    if size < 126:
        head.append(bit | size)
    elif size < 1 << 16:
        head.append(bit | 126)
        head += struct.pack(">H", size)
    else:
        head.append(bit | 127)
        head += struct.pack(">Q", size)
    if mask is None:
        return bytes(head) + payload
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return bytes(head) + mask + masked


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        got = sock.recv(size)
        if not got:
            raise ConnectionError("the socket closed")
        chunks.append(got)
        size -= len(got)
    return b"".join(chunks)


def ws_read(sock: socket.socket) -> tuple[int, bytes]:
    """One frame: (opcode, payload), unmasked. Fragments are not expected."""
    b0, b1 = _read_exact(sock, 2)
    opcode = b0 & 0x0F
    size = b1 & 0x7F
    if size == 126:
        (size,) = struct.unpack(">H", _read_exact(sock, 2))
    elif size == 127:
        (size,) = struct.unpack(">Q", _read_exact(sock, 8))
    mask = _read_exact(sock, 4) if b1 & 0x80 else None
    payload = _read_exact(sock, size)
    if mask is not None:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def ws_connect(host: str, port: int, path: str = "/ws", timeout: float = 30.0) -> socket.socket:
    """A client's socket after the handshake (tests and scripted persons)."""
    sock = socket.create_connection((host, port), timeout=timeout)
    key = base64.b64encode(b"pokeuraou-ika332").decode("ascii")
    sock.sendall(
        (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
         f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        .encode("ascii")
    )
    head = b""
    while b"\r\n\r\n" not in head:
        got = sock.recv(1)
        if not got:
            raise ConnectionError("no handshake")
        head += got
    if b" 101 " not in head.split(b"\r\n", 1)[0] or ws_accept(key).encode() not in head:
        raise ConnectionError(f"bad handshake: {head!r}")
    return sock


def ws_send_text(sock: socket.socket, text: str) -> None:
    sock.sendall(ws_frame(text.encode("utf-8"), 1, mask=b"\x11\x22\x33\x44"))


#: The page's images by default (the user's decision of 9/26): Showdown's sprite server,
#: asked by the browser that opens the page. Nothing is downloaded into the repository or
#: data/. ``{id}`` is Showdown's sprite id (`humanplay.sprite_id`).
SPRITE_URL = "https://play.pokemonshowdown.com/sprites/gen5/{id}.png"

#: The page's files: the structure, the look (CSS variables and layout), the data half
#: (socket, frames, strings -- no drawing) and the drawing half. A designed page replaces
#: the look and the drawing and keeps the data half.
WEB = Path(__file__).resolve().parent / "web"
PAGE = WEB / "live.html"
FILES = {
    "/": ("live.html", "text/html; charset=utf-8"),
    "/live.html": ("live.html", "text/html; charset=utf-8"),
    "/live.css": ("live.css", "text/css; charset=utf-8"),
    "/live-data.js": ("live-data.js", "text/javascript; charset=utf-8"),
    "/live-view.js": ("live-view.js", "text/javascript; charset=utf-8"),
}


class LiveServer:
    """The page, the socket, and the game's frames for every page connected.

    `listener` is what `humanplay.play` is handed: it packs each event (`Wire`), keeps it
    for pages that connect later, sends it to the ones connected, and to ``sink`` (a
    `FileSink`) if there is one. `inbox` holds what the pages send (the person's answers).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        sink: Callable[[bytes], None] | None = None,
        web: Path = WEB,
        sprite_url: str | None = None,
    ) -> None:
        self.wire = Wire()
        self.sink = sink
        self.web = web
        #: Where the page takes its images (``{id}`` is Showdown's sprite id), written into
        #: the page's ``sprite-url`` meta. None keeps the page's (`SPRITE_URL`, Showdown's
        #: server: the browser asks it, nothing is stored); "" is no images, name cards.
        self.sprite_url = sprite_url
        self.inbox: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.clients: list[tuple[socket.socket, threading.Lock]] = []
        #: What a page connecting now is sent first: strings, events, the steps kept.
        self.history: list[bytes] = []
        self._current: list[bytes] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: ANN401
                del args

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/ws"):
                    server._serve_socket(self)
                    return
                found = FILES.get(self.path.split("?", 1)[0])
                if found is not None:
                    body = (server.web / found[0]).read_bytes()
                    if found[0] == "live.html" and server.sprite_url is not None:
                        body = re.sub(
                            rb'<meta name="sprite-url" content="[^"]*">',
                            lambda _m: b'<meta name="sprite-url" content="'
                            + html.escape(server.sprite_url).encode("utf-8") + b'">',
                            body,
                        )
                    self.send_response(200)
                    self.send_header("Content-Type", found[1])
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}/"

    def start(self) -> LiveServer:
        self.thread.start()
        return self

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        with self.lock:
            for sock, _ in self.clients:
                with contextlib.suppress(OSError):
                    sock.close()
            self.clients = []

    # -- the game's side
    def listener(self, kind: str, payload: Any) -> None:  # noqa: ANN401
        if kind == "step":
            frames = self.wire.step(payload)
            self._send(frames, keep="step", done=payload.kind == "done")
            return
        frames = self.wire.event(kind, payload)
        self._send(frames, keep="event")

    def _send(self, frames: list[bytes], *, keep: str, done: bool = False) -> None:
        with self.lock:
            for frame in frames:
                if frame[0] == STRINGS or keep == "event":
                    self.history.append(frame)
                elif done:
                    # The decision's last step stays; the ones before it are dropped.
                    self._current = []
                    self.history.append(frame)
                else:
                    self._current.append(frame)
                if self.sink is not None:
                    self.sink(frame)
            clients = list(self.clients)
        for sock, lock in clients:
            try:
                with lock:
                    for frame in frames:
                        sock.sendall(ws_frame(frame))
            except OSError:
                self._drop(sock)

    def _drop(self, sock: socket.socket) -> None:
        with self.lock:
            self.clients = [(s, lk) for s, lk in self.clients if s is not sock]

    # -- a page's side
    def _serve_socket(self, handler: BaseHTTPRequestHandler) -> None:
        key = handler.headers.get("Sec-WebSocket-Key")
        if not key or "websocket" not in handler.headers.get("Upgrade", "").lower():
            handler.send_error(400, "a websocket is asked for with Upgrade")
            return
        handler.send_response(101, "Switching Protocols")
        handler.send_header("Upgrade", "websocket")
        handler.send_header("Connection", "Upgrade")
        handler.send_header("Sec-WebSocket-Accept", ws_accept(key))
        handler.end_headers()
        handler.wfile.flush()
        sock = handler.connection
        lock = threading.Lock()
        with self.lock:
            # The game so far, before anything new: the history and the steps in progress,
            # under the same lock the sender holds, so nothing is missed or sent twice.
            with lock:
                for frame in [*self.history, *self._current]:
                    sock.sendall(ws_frame(frame))
            self.clients.append((sock, lock))
        try:
            while True:
                opcode, payload = ws_read(sock)
                if opcode == 8:
                    with lock:
                        sock.sendall(ws_frame(payload[:2], 8))
                    break
                if opcode == 9:
                    with lock:
                        sock.sendall(ws_frame(payload, 10))
                    continue
                if opcode == 1:
                    try:
                        message = json.loads(payload.decode("utf-8"))
                    except ValueError:
                        continue
                    line = message.get("line") if isinstance(message, dict) else None
                    if isinstance(line, str):
                        self.inbox.put(line)
        except (ConnectionError, OSError):
            pass
        finally:
            self._drop(sock)
            handler.close_connection = True


# ----------------------------------------------------------------------------- the person


class WebPerson(Person):
    """The person at the page: each question waits for a line from the page (`LiveServer.inbox`),
    read as a script line would be; a line that does not parse is said back and asked again."""

    kind = "web"

    def __init__(self, server: LiveServer, *, timeout: float | None = None) -> None:
        self.server = server
        self.timeout = timeout

    def _line(self) -> str:
        try:
            return self.server.inbox.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("no answer from the page") from None

    def select(self, six: Sequence[SampledSet], size: int, text: str) -> tuple[int, ...]:  # noqa: ARG002
        while True:
            line = self._line()
            try:
                return parse_selection(line, len(six), size)
            except ValueError as problem:
                self.server.listener("error", {"message": str(problem)})

    def choose(self, kind: str, legal: Sequence[SideAction], text: str) -> SideAction:  # noqa: ARG002
        while True:
            line = self._line()
            try:
                return parse_choice(line.lstrip("#"), legal)
            except ValueError as problem:
                self.server.listener("error", {"message": str(problem)})


@dataclass
class Replay:
    """A recorded game's frames sent again to the pages, at ``speed`` times the pace they
    were written (0: all at once)."""

    server: LiveServer
    frames: list[tuple[float, bytes]]
    speed: float = 1.0

    def run(self) -> None:
        started = time.perf_counter()
        for seconds, frame in self.frames:
            if self.speed > 0:
                wait = seconds / self.speed - (time.perf_counter() - started)
                if wait > 0:
                    time.sleep(wait)
            if frame[0] == STRINGS:
                self.server._send([frame], keep="event")
            elif frame[0] == STEP:
                kind = frame[1]
                self.server._send([frame], keep="step", done=KINDS[kind] == "done")
            else:
                self.server._send([frame], keep="event")


__all__ = [
    "EVENT",
    "FILES",
    "PAGE",
    "WEB",
    "STEP",
    "SPRITE_URL",
    "STRINGS",
    "Decoder",
    "FileSink",
    "LiveServer",
    "PvNode",
    "Replay",
    "WebPerson",
    "Wire",
    "read_record",
    "ws_connect",
    "ws_read",
    "ws_send_text",
]
