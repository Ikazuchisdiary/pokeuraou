"""The queue's whole reason to exist is what happens when a worker does not come back.

Generation lost about 2,100 games to two workers that died holding a block each, so the
test that matters is not that indices come out once in the happy case -- it is that a
killed process's work is played by somebody else. That means killing a real process:
closing a socket in-process is not the same event, because `socket.makefile` keeps the
descriptor open and the server sees no EOF.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

from pokeuraou.workqueue import MAX_ATTEMPTS, WorkClient, WorkQueue, serve

TAKE_AND_DIE = """
import sys, time
sys.path.insert(0, {src!r})
from pokeuraou.workqueue import WorkClient
client = WorkClient({address!r})
taken = [client.take() for _ in range({count})]
print(" ".join(str(t) for t in taken), flush=True)
time.sleep(60)
"""


def _src_dir() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1] / "src")


def test_every_index_is_handed_out_once() -> None:
    queue = WorkQueue(range(50))
    server, address = serve(queue)
    try:
        seen: list[int] = []
        lock = threading.Lock()

        def drain() -> None:
            with WorkClient(address) as client:
                while (index := client.take()) is not None:
                    with lock:
                        seen.append(index)
                    client.finish(index)

        workers = [threading.Thread(target=drain) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        assert sorted(seen) == list(range(50))
        assert queue.remaining == 0
    finally:
        server.shutdown()


def test_a_killed_worker_s_games_are_played_by_someone_else() -> None:
    queue = WorkQueue(range(10))
    returned: list[list[int]] = []
    server, address = serve(queue, on_return=returned.append)
    try:
        victim = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    TAKE_AND_DIE.format(src=_src_dir(), address=address, count=3)
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert victim.stdout is not None
        held = [int(part) for part in victim.stdout.readline().split()]
        assert len(held) == 3
        assert queue.remaining == 10  # taken, but none of them finished

        victim.kill()
        victim.wait(timeout=30)

        deadline = time.time() + 30
        while time.time() < deadline and not returned:
            time.sleep(0.01)
        assert returned == [sorted(held)], "the queue never noticed the worker was gone"

        with WorkClient(address) as client:
            rest = []
            while (index := client.take()) is not None:
                rest.append(index)
                client.finish(index)
        assert sorted(rest) == list(range(10))
        # Returned work goes to the front: the straggler must not be left until last.
        assert rest[:3] == sorted(held)
    finally:
        server.shutdown()


def test_a_game_that_kills_every_worker_is_abandoned() -> None:
    """A crash fed back to the pool forever is worse than the split it replaced."""
    queue = WorkQueue([7])
    server, address = serve(queue)
    try:
        for _ in range(MAX_ATTEMPTS):
            client = WorkClient(address)
            assert client.take() == 7
            client.close()  # dropped without finishing, as a crash would
            deadline = time.time() + 30
            while time.time() < deadline and queue.remaining == 0:
                time.sleep(0.01)
        with WorkClient(address) as client:
            assert client.take() is None
        assert queue.abandoned == [7]
    finally:
        server.shutdown()
