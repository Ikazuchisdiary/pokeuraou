"""Why does a worker get heavier the longer it plays?

Measured on the board: a match worker grew from 863 MB of working set to 1,512 MB over 711
games, about 0.9 MB a game, and it did so *through the inference server* -- so it is not
torch, which that worker does not import. A generation of 12,000 games would add 10 GB at
that rate, which matters more than anything the width sweep is deciding.

Three quantities, because they answer different questions:

  working set   what the machine has to find. This is the number that killed runs.
  tracemalloc   what Python allocated and still holds. If this stays flat while the
                working set climbs, nothing is leaking in the ordinary sense and the
                allocator is simply not giving blocks back to the OS.
  gc objects    what Python is still reachable-ly holding, by type. If *this* climbs, the
                growth has a name and the name is in the table.

Run it against the server so the measurement matches the case that was seen:

    uv run --group learn python tools/inference_server.py --arm value <models> &
    uv run --group learn python tools/worker_growth.py --inference HOST:PORT --games 60
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import sys
import tracemalloc
from collections import Counter
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


class _Counters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k = ctypes.windll.kernel32
_k.GetCurrentProcess.restype = ctypes.c_void_p
_p = ctypes.windll.psapi
_p.GetProcessMemoryInfo.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(_Counters), wintypes.DWORD
]


def process_memory() -> tuple[float, float]:
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if not _p.GetProcessMemoryInfo(_k.GetCurrentProcess(), ctypes.byref(counters),
                                   counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.WorkingSetSize / 1048576, counters.PagefileUsage / 1048576


def census() -> Counter[str]:
    return Counter(type(obj).__name__ for obj in gc.get_objects())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--seed", type=int, default=808)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--inference", default=None, metavar="HOST:PORT")
    ap.add_argument("--inference-arm", default="value")
    ap.add_argument("--value", nargs="+", default=None,
                    help="load the models here instead, to compare against the served case")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    from pokeuraou.encode import Encoder

    encoder = Encoder(reg)
    if args.inference:
        from pokeuraou.inference import RemoteValue

        evaluate = RemoteValue(args.inference, args.inference_arm, encoder)
        where = f"served by {args.inference}"
    elif args.value:
        import torch

        from pokeuraou.value import BatchedValue, load_ensemble

        torch.set_num_threads(1)
        nets, _ = load_ensemble([Path(p) for p in args.value], encoder)
        device = torch.device(args.device)
        evaluate = BatchedValue([n.to(device) for n in nets], encoder, device=device)
        where = f"in this process on {args.device}"
    else:
        raise SystemExit("pass --inference or --value")

    tracemalloc.start()
    gc.collect()
    base_objects = census()
    print(f"leaf {where}, width {args.limit}, {args.games} games\n")
    print(f"  {'games':>6}{'working set':>13}{'committed':>11}{'tracemalloc':>13}"
          f"{'gc objects':>12}")

    rng = np.random.default_rng(args.seed)
    for index in range(1, args.games + 1):
        team = pool[int(rng.integers(len(pool)))]
        foe_six = sample_standings_team(rng, reg, prior, team)
        own_pick = selections[int(rng.integers(len(selections)))]
        foe_pick = selections[int(rng.integers(len(selections)))]
        play_game(
            reg, rng,
            [roster.sets[i] for i in own_pick], [foe_six[j] for j in foe_pick],
            "growth-probe",
            search_limit=args.limit, max_turns=40, evaluate=evaluate, rank_by_leaf=True,
        )
        if index % args.every == 0 or index == 1:
            gc.collect()
            ws, commit = process_memory()
            current, _peak = tracemalloc.get_traced_memory()
            objects = sum(census().values())
            print(f"  {index:>6}{ws:>10.0f} MB{commit:>8.0f} MB"
                  f"{current / 1048576:>10.0f} MB{objects:>12,}", flush=True)

    gc.collect()
    end_objects = census()
    grown = sorted(
        ((end_objects[name] - base_objects.get(name, 0), name) for name in end_objects),
        reverse=True,
    )[:12]
    print("\n  object types that grew most:")
    for delta, name in grown:
        if delta <= 0:
            continue
        print(f"    {name:<28}{delta:>+10,}  (now {end_objects[name]:,})")

    snapshot = tracemalloc.take_snapshot()
    print("\n  where Python's own allocations sit:")
    for stat in snapshot.statistics("lineno")[:8]:
        frame = stat.traceback[0]
        where = f"{Path(frame.filename).name}:{frame.lineno}"
        print(f"    {where:<34}{stat.size / 1048576:>8.1f} MB  {stat.count:,} blocks")


if __name__ == "__main__":
    main()
