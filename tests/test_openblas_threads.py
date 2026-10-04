"""IKA-431: the generation, match and Q-teaching processes load OpenBLAS with one thread.

OpenBLAS commits a buffer for every thread it may run when the library loads -- numpy's
copy and scipy's, about 0.5 GB each on a 16-thread machine -- so a worker that runs on one
core held a gigabyte it never touched (1.08 GB of a 1.2 GB generation worker). Each script
below sets `OPENBLAS_NUM_THREADS` before numpy loads, and the processes it starts inherit
it. The thread count is asked of the library numpy loaded (`portmenus.blas`), in a fresh
interpreter that has only imported the script.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "tools/generate_queue.py",
    "tools/selfplay.py",
    "tools/match_queue.py",
    "tools/pool_match.py",
    "tools/generation_match.py",
    "tools/q_teach.py",
    "tools/inference_server.py",
]

#: Imports the script named by argv[1] (not as __main__, so nothing runs), or nothing for
#: "-", then prints the thread count of the OpenBLAS numpy loaded.
PROBE = r"""
import ctypes, importlib.util, sys
if sys.argv[1] != "-":
    spec = importlib.util.spec_from_file_location("script_under_test", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
import numpy
from pokeuraou.portmenus import blas
found = blas()
if found is None:
    print("none")
    raise SystemExit(0)
lib = ctypes.CDLL(found["path"])
print(getattr(lib, found["threads"].replace("set_num_threads", "get_num_threads"))())
"""


def threads(script: str, **env: str) -> str:
    environment = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.update(env)
    done = subprocess.run(
        [sys.executable, "-c", PROBE, script if script == "-" else str(ROOT / script)],
        env=environment, cwd=str(ROOT), capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    out = done.stdout.strip().splitlines()[-1]
    if out == "none":
        pytest.skip("numpy has no OpenBLAS this test can ask")
    return out


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_loads_openblas_with_one_thread(script: str) -> None:
    assert threads(script) == "1"


def test_bare_numpy_takes_more_threads() -> None:
    """The control: without the setting, numpy's OpenBLAS takes a thread per core, so the
    test above can fail."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("one core: OpenBLAS takes one thread anyway")
    assert int(threads("-")) > 1


def test_an_explicit_setting_wins() -> None:
    """`setdefault`: a caller that names a count gets it."""
    assert threads("tools/selfplay.py", OPENBLAS_NUM_THREADS="2") == "2"
