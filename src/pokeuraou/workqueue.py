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
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

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
        #: Indices nobody will play because the run was ended on purpose (`close`). Kept
        #: apart from `abandoned`, which is a failure, and from `remaining`, which a run
        #: that stopped because it had its answer must not report as work left undone.
        self.dropped: list[int] = []
        self._closed = False

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
            if self._closed:
                # Nobody is handed anything after `close`, so there is nothing to give it
                # back to: it joins what the stop dropped rather than a queue no one reads.
                self.dropped.extend(back)
                return []
            # Front of the queue: a returned game is the oldest outstanding work, and
            # leaving it until last would put the straggler back at the end of the run.
            self._pending.extendleft(reversed(back))
            return back

    def close(self) -> int:
        """Hands out nothing more. What the workers are holding finishes and is written, as
        it would at the end of any run; what nobody has started is dropped. Returns how
        many were dropped."""
        with self._lock:
            self._closed = True
            dropped = list(self._pending)
            self._pending.clear()
            self.dropped.extend(dropped)
            return len(dropped)

    def resolved(self) -> set[int]:
        """Indices that will not change again: finished, or abandoned after `MAX_ATTEMPTS`.

        A snapshot under the lock, for a monitor that has to know which results are final
        before it reads them. A match worker reports an index finished only after writing
        its record -- or after discarding the game at the turn cap, when there is nothing
        to write -- so whatever this set's games left on disk is there when it is taken.
        """
        with self._lock:
            return set(self.done) | set(self.abandoned)


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


def run_workers(
    indices: Iterable[int],
    build_command: Callable[[int, str], Sequence[str]],
    *,
    workers: int,
    out_dir: Path,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    label: str = "run",
    counts: Callable[[], int] | None = None,
    monitor: Callable[[WorkQueue], str | None] | None = None,
    poll: float = 20.0,
) -> int:
    """Serves a queue, runs `workers` processes against it, and reports what happened.

    Generation and matches had a driver each, differing only in which program they start
    and how the indices are laid out -- and the second one to be written inherited none of
    the first one's reporting. So the driver is here and the tools are the two things that
    actually differ: `build_command(worker, address)` and what an index means.

    An index means whatever the caller decides, and the decision matters. Generation makes
    it a game. A match makes it *a seat of* a game -- `i // 2` played from seat `i % 2` --
    because draining seats separately lets one of them empty the queue while the other
    starves, and the pairing between them is the whole reason for playing both.

    `monitor(queue)` is called every `poll` seconds while the workers run, and once more
    after they have all exited so the last results are seen too. Returning a reason ends
    the run early and on purpose: the queue is closed, the games in flight finish and are
    written, and the rest are recorded as dropped rather than as work left undone. It is
    how a match stops the moment a sequential test has decided (`pokeuraou.sprt`).

    Returns a process exit code: non-zero when work was left unplayed or a worker failed,
    so an incomplete run says so rather than being discovered by counting files later.
    """
    queue = WorkQueue(indices)
    returned: list[int] = []
    stopped: list[str] = []
    broken: list[BaseException] = []
    finished = threading.Event()

    def look() -> str | None:
        assert monitor is not None
        try:
            return monitor(queue)
        except Exception as error:  # noqa: BLE001 - reported, and the run says so at the end
            # A monitor that dies must not take the games with it, and must not look like
            # one that never decided either: the run goes on to its full count, and the
            # exit code says the stop it was asked for never happened.
            broken.append(error)
            print(
                f"  the monitor failed and the run goes on to its full count: {error!r}",
                file=sys.stderr,
                flush=True,
            )
            return None

    def supervise() -> None:
        while not finished.wait(poll):
            reason = look()
            if broken:
                return
            if reason:
                stopped.append(reason)
                dropped = queue.close()
                print(
                    f"  {reason}\n  stopping: {dropped} job(s) not handed out, the "
                    f"{queue.remaining} in flight finish and are written",
                    file=sys.stderr,
                    flush=True,
                )
                return

    def note_return(back: list[int]) -> None:
        returned.extend(back)
        print(
            f"  a worker went away holding {len(back)} job(s); back on the queue",
            file=sys.stderr,
            flush=True,
        )

    server, address = serve(queue, on_return=note_return)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    running = []
    for worker in range(workers):
        log = (out_dir / "logs" / f"worker{worker}.log").open("w", encoding="utf-8")
        running.append((
            worker,
            subprocess.Popen(  # noqa: S603
                list(build_command(worker, address)),
                env=env, cwd=cwd, stdout=log, stderr=log, text=True,
            ),
            log,
        ))

    failed = 0
    finished_at: dict[int, float] = {}

    def watch(worker: int, process: subprocess.Popen) -> None:
        nonlocal failed
        process.wait()
        finished_at[worker] = time.perf_counter() - started
        # Work nobody holds. A worker leaving while others still hold jobs has run out of
        # queue, which is what is supposed to happen at the end.
        left = queue.pending
        if process.returncode != 0 or left:
            print(
                f"  worker {worker} exited with {process.returncode} after "
                f"{finished_at[worker]:.0f}s, {left} job(s) still unclaimed",
                file=sys.stderr,
                flush=True,
            )
        if process.returncode != 0:
            failed += 1

    threads = [threading.Thread(target=watch, args=(w, p)) for w, p, _ in running]
    for thread in threads:
        thread.start()
    supervisor = None
    if monitor is not None:
        supervisor = threading.Thread(target=supervise, daemon=True)
        supervisor.start()
    for thread in threads:
        thread.join()
    finished.set()
    if supervisor is not None:
        # Joined before the last look, so the monitor is never called from two threads.
        supervisor.join()
        if not stopped and not broken:
            reason = look()
            if reason:
                stopped.append(reason)
                print(f"  {reason} (on the last results)", file=sys.stderr, flush=True)
    for _worker, _process, log in running:
        log.close()
    server.shutdown()

    elapsed = time.perf_counter() - started
    order = sorted(finished_at.values())
    idle = sum(order[-1] - t for t in order) if order else 0.0
    written = counts() if counts is not None else len(queue.done)
    print(
        f"{label} done in {elapsed / 60:.1f} min: {written} written, "
        f"{len(queue.done)} jobs finished"
        + (f", {queue.remaining} left unplayed" if queue.remaining else "")
        + (f", {len(queue.abandoned)} abandoned" if queue.abandoned else "")
        + (f", {len(queue.dropped)} not played because the run was stopped" if queue.dropped else "")
        + (f", {len(returned)} replayed after a worker went away" if returned else "")
        + (f", {failed} worker(s) failed" if failed else ""),
        file=sys.stderr,
    )
    if order:
        print(
            f"  workers finished {order[-1] - order[0]:.0f}s apart, "
            f"{idle / (len(order) * elapsed):.1%} of the machine idle at the end",
            file=sys.stderr,
        )
    return 1 if (queue.remaining or queue.abandoned or failed or broken) else 0


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
