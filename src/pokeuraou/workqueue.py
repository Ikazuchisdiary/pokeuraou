"""Handing generation's games out one at a time instead of dealing them in advance.

Generation gave every worker a fixed block of games, so a run ended when the unluckiest
worker did. That is not a small effect: a game's cost varies eight-fold, and measured on
the real logs the machine spent 1.4% of gen2345 idle, 16.4% of gen6 and 10.5% of gen7's
healthy workers -- the last worker finishing 28 minutes after the first.

Dealing in advance has a second cost that is easy to miss. When a worker dies its whole
remaining block dies with it. gen7 lost about 2,100 games that way, to two workers that
stopped after 126 and 50 of 1,098, and nothing noticed until the files were counted
afterwards. A queue turns that from a silent loss into a delay: the games the dead worker
had not started are still in the queue, and a game it had started is put back when its
connection drops.

**A game must not depend on which worker plays it.** Generation streamed one RNG through a
worker's whole block, so game seventeen was whatever the sixteen before it left behind --
fine when the block is fixed, meaningless when the order is a race. So a queue hands out
*indices*, and the caller seeds each game from its index. Then game seventeen is the same
game whoever plays it and in whatever order, a run is reproducible again, and a retried
game is the game that was lost rather than a new one.

A socket, not a file or a pipe. File locking to share a counter between processes is
fiddlier than it looks on Windows, and this project already has one 64 MB pipe failure in
its logs. A loopback socket has neither problem, and a worker that dies is a socket that
closes, which is exactly the event the queue needs to hear about.
"""

from __future__ import annotations

import os
import socket
import socketserver
import threading
from collections import deque
from collections.abc import Iterable

#: Environment variable carrying "host:port" to a worker.
ENV_QUEUE = "POKEURAOU_WORK_QUEUE"

#: How many times a game may be handed out before it is abandoned. A game that kills its
#: worker would otherwise kill every worker in turn, one at a time, and a queue that feeds
#: a crash back to the pool forever is worse than the static split it replaced.
MAX_ATTEMPTS = 3


class WorkQueue:
    """Indices to be played, handed out one at a time and returned if a worker dies."""

    def __init__(self, indices: Iterable[int]) -> None:
        self._lock = threading.Lock()
        self._pending = deque(indices)
        self._attempts: dict[int, int] = {}
        #: What each connection is holding, so a dropped connection can give it back.
        self._held: dict[int, set[int]] = {}
        self.done: list[int] = []
        self.abandoned: list[int] = []

    @property
    def remaining(self) -> int:
        """Games not yet finished, including those a worker is holding."""
        with self._lock:
            return len(self._pending) + sum(len(h) for h in self._held.values())

    @property
    def pending(self) -> int:
        """Games nobody is working on. A worker leaving while this is zero is done, not
        dead -- the difference the run's report has to draw."""
        with self._lock:
            return len(self._pending)

    def take(self, holder: int) -> int | None:
        with self._lock:
            while self._pending:
                index = self._pending.popleft()
                attempts = self._attempts.get(index, 0) + 1
                if attempts > MAX_ATTEMPTS:
                    self.abandoned.append(index)
                    continue
                self._attempts[index] = attempts
                self._held.setdefault(holder, set()).add(index)
                return index
            return None

    def finish(self, holder: int, index: int) -> None:
        with self._lock:
            self._held.get(holder, set()).discard(index)
            self.done.append(index)

    def release(self, holder: int) -> list[int]:
        """Gives back whatever a departing worker was holding. Returns what came back."""
        with self._lock:
            back = sorted(self._held.pop(holder, set()))
            # Front of the queue: a returned game is the oldest outstanding work, and
            # leaving it until last would put the straggler back at the end of the run.
            self._pending.extendleft(reversed(back))
            return back


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        holder = id(self)
        queue: WorkQueue = self.server.queue  # type: ignore[attr-defined]
        try:
            for raw in self.rfile:
                line = raw.decode("utf-8", "replace").strip()
                if line == "get":
                    index = queue.take(holder)
                    self.wfile.write(
                        b"done\n" if index is None else f"{index}\n".encode()
                    )
                    self.wfile.flush()
                elif line.startswith("ok "):
                    queue.finish(holder, int(line[3:]))
                elif not line:
                    continue
                else:
                    return
        except (ConnectionError, ValueError, OSError):
            pass
        finally:
            returned = queue.release(holder)
            if returned:
                on_return = getattr(self.server, "on_return", None)
                if on_return is not None:
                    on_return(returned)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(queue: WorkQueue, on_return=None) -> tuple[_Server, str]:
    """Starts the queue on a loopback port. Returns the server and its "host:port"."""
    server = _Server(("127.0.0.1", 0), _Handler)
    server.queue = queue  # type: ignore[attr-defined]
    server.on_return = on_return  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f"{host}:{port}"


class WorkClient:
    """A worker's end: ``for index in client:`` until the queue is empty.

    ``finish`` is separate from taking the next one on purpose. A game is only done once it
    has been written, and a worker that dies between playing and writing should have that
    game given to somebody else.
    """

    def __init__(self, address: str | None = None) -> None:
        address = address or os.environ.get(ENV_QUEUE, "")
        if not address:
            raise ValueError(f"no queue address, and {ENV_QUEUE} is not set")
        host, _, port = address.rpartition(":")
        self._sock = socket.create_connection((host, int(port)))
        self._file = self._sock.makefile("rwb")

    def take(self) -> int | None:
        self._file.write(b"get\n")
        self._file.flush()
        line = self._file.readline().decode("utf-8", "replace").strip()
        return None if line in ("", "done") else int(line)

    def finish(self, index: int) -> None:
        self._file.write(f"ok {index}\n".encode())
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._sock.close()

    def __enter__(self) -> WorkClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
