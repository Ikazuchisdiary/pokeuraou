"""The OpenMP threads' wait, off for the agent a person plays (IKA-360).

torch's CPU work runs on Intel's OpenMP (``libiomp5md``). After each parallel region its
threads do not sleep: for ``KMP_BLOCKTIME`` milliseconds (200 by default) each one spins,
yielding, waiting for the next region. The agent a person plays (`humanplay`) wakes that
pool many times a move -- the CPU side of every leaf block it scores -- in its own process
and in each of the worker processes that expand the deepening's cells ahead, so seven
threads a process spun almost all the time: measured (IKA-333, 9/27) on a person's game on
the count clock at 4 threads, 175 CPU seconds for two games, 86 of them in the kernel (the
yield), on the cores the reading itself needed.

``KMP_BLOCKTIME=0`` puts the threads to sleep at the end of each region. It changes how
they wait, not what they compute: the same games to the byte, 40 CPU seconds instead of 175
and 17% less wall clock for the same reading (records/IKA-360.md). Generation is not
touched: its workers leave the leaf to the inference servers and never enter the pool
(the same A/B there moved nothing).

The runtime reads the variable once, when torch loads it, so `quiet_wait` has to run
before anything imports torch: the entry points a person starts (`tools/play_human.py`,
`tools/analyze.py`, `tools/play.py`) call it right after putting ``src`` on the path. The
worker processes are spawned with this process's environment and read it too. A value
already set (by the person, for a measurement) is kept.
"""

from __future__ import annotations

import os
import sys

#: What the entry points set: sleep at once.
BLOCKTIME_MS = "0"

#: Whether `quiet_wait` ran before torch was loaded in this process (None: not called).
before_torch: bool | None = None


def quiet_wait() -> bool:
    """Sets ``KMP_BLOCKTIME`` to `BLOCKTIME_MS` unless it is set already. Returns whether
    torch was not loaded yet -- if it was, the setting reaches only the worker processes
    started from here on."""
    global before_torch
    before_torch = "torch" not in sys.modules
    os.environ.setdefault("KMP_BLOCKTIME", BLOCKTIME_MS)
    return before_torch


__all__ = ["BLOCKTIME_MS", "before_torch", "quiet_wait"]
