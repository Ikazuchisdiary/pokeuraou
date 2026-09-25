"""What building the legal set costs, alone and inside `narrow` (IKA-319).

`narrow` is timed as one stage, and inside it the legal set (`actions.side_actions`), the
dead-move filter (`narrow.drop_dead_actions`), the damage score (a crossing to the port),
the ordering and the cover are not told apart. This times each of them *itself*, on
recorded positions, as the CPU time of this thread -- the machine is shared while this
runs, so a wall clock would read whatever else was scheduled.

The clock is `QueryThreadCycleTime` on Windows (the thread's own cycles, read per call;
`time.thread_time` there moves in 15.6 ms ticks, so it is kept only as the total that the
per-call sum is checked against) and `time.thread_time_ns` elsewhere. The port's own CPU
is read once around the whole scoring pass from the child's process times.

The stages are `narrow`'s body written out with a timer around each, and every pool's
result is checked against `narrow` itself (same kept actions, scores and uncovered
options), so what is timed is what `narrow` does. The ranking (`rank=`) is not run: it is
the leaf's fill and forward pass, which IKA-32 counts in its own parts.

    python tools/legal_moves_cost.py --games-dir data/selfplay-mc0 --positions 4000
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.actions import MoveAction, SwitchAction, locked_move, side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.narrow import (  # noqa: E402
    Candidate,
    Narrowed,
    _count_kinds,
    _slot_key,
    _slot_label,
    drop_dead_actions,
    narrow,
)
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

LIMITS = (8, 12, 48)


def _cycle_clock():  # noqa: ANN202
    """(read, ns_per_unit): the thread's own CPU counter and its unit in nanoseconds."""
    if sys.platform != "win32":
        return time.thread_time_ns, 1.0
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentThread.restype = wintypes.HANDLE
    handle = kernel32.GetCurrentThread()
    value = ctypes.c_ulonglong()
    query = kernel32.QueryThreadCycleTime
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ulonglong)]
    query.restype = wintypes.BOOL

    def read() -> int:
        query(handle, ctypes.byref(value))
        return value.value

    # Calibrate cycles against the wall clock over short busy loops; preemption only
    # lengthens the wall side, so the largest cycles-per-ns is the least disturbed.
    best = 0.0
    for _ in range(7):
        c0, t0 = read(), time.perf_counter_ns()
        while time.perf_counter_ns() - t0 < 100_000_000:
            pass
        best = max(best, (read() - c0) / (time.perf_counter_ns() - t0))
    return read, 1.0 / best


def load_positions(games_dir: Path, wanted: int, seed: int) -> list[Position]:
    """Move decisions' positions, an even share from every worker file, shuffled."""
    files = sorted(games_dir.glob("games-worker*.jsonl"))
    share = -(-wanted * 3 // (2 * len(files)))
    out: list[Position] = []
    for path in files:
        got = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for decision in record.get("decisions", ()):
                    if decision.get("kind") == "move":
                        out.append(Position.from_json(decision["position"]))
                        got += 1
                if got >= share:
                    break
    random.Random(seed).shuffle(out)
    return out[:wanted]


def depth2_children(reg, roots: list[Position], seed: int, wanted: int) -> list[Position]:  # noqa: ANN001
    """Positions a depth-2 refinement narrows at: a cell of the width-12 root menus played
    through the port, and its likeliest `SUB_BRANCHES` outcomes that are not over -- what
    `search._refine_cells` hands `narrow_many` (its cells are the root's best, these are
    drawn uniformly from the menus, so the mix of cells is not the same)."""
    from pokeuraou import port
    from pokeuraou.budget import Budget

    rng = random.Random(seed)
    out: list[Position] = []
    for pos in roots:
        rows = narrow(reg, pos, 0, limit=12).actions
        cols = narrow(reg, pos, 1, limit=12).actions
        if not rows or not cols:
            continue
        cell = [rng.choice(rows), rng.choice(cols)]
        (answer,) = port.turns(reg, [(pos, cell)], Budget.matrix(), full=True)
        if isinstance(answer, port.PortRefused) or answer.suspended or not answer.outcomes:
            continue
        for branch in sorted(answer.outcomes, key=lambda b: -b.probability)[:SUB_BRANCHES]:
            if not branch.position.ended:
                out.append(branch.position)
        if len(out) >= wanted:
            break
    return out[:wanted]


#: `search.DEFAULT_SUB_BRANCHES`.
SUB_BRANCHES = 3


def kind_of(reg, pos: Position, side: int, n_legal: int) -> dict:  # noqa: ANN001
    s = pos.sides[side]
    live_active = [m for m in s.active_pokemon() if m is not None and not m.fainted]
    bench = s.bench()
    locked = any(locked_move(reg, m) is not None for m in live_active)
    return {
        "n_legal": n_legal,
        "bench": len(bench) > 0,
        "locked": locked,
        "active": len(live_active),
    }


def legal_bucket(n: int) -> str:
    for hi in (10, 30, 60, 100, 150):
        if n <= hi:
            return f"<={hi}"
    return ">150"


def cover(reg, pool, scored, limit: int, order: list[int]):  # noqa: ANN001, ANN201
    """`narrow`'s cover and fill after its sort, as written there (checked against it)."""
    n_slots = len(pool[0].slots)
    if len(pool) <= limit:
        kept = [scored[i] for i in order]
        return Narrowed(kept=kept, considered=len(pool), for_score=len(kept))
    needed: dict[str, str] = {}
    options: dict[str, tuple[int, object]] = {}
    for action in pool:
        for index in range(n_slots):
            key = _slot_key(action, index)
            needed.setdefault(key, _slot_label(reg, action, index))
            options.setdefault(key, (index, action.slots[index]))
    kept_indices: list[int] = []
    taken: set[int] = set()
    uncovered = set(needed)
    for_coverage = 0
    while uncovered and len(kept_indices) < limit:
        best = -1
        best_gain = 0
        for i in order:
            if i in taken:
                continue
            gain = sum(1 for index in range(n_slots) if _slot_key(pool[i], index) in uncovered)
            if gain > best_gain:
                best, best_gain = i, gain
            if best_gain == n_slots:
                break
        if best < 0:
            break
        kept_indices.append(best)
        taken.add(best)
        for_coverage += 1
        for index in range(n_slots):
            uncovered.discard(_slot_key(pool[best], index))
    for i in order:
        if len(kept_indices) >= limit:
            break
        if i in taken:
            continue
        kept_indices.append(i)
        taken.add(i)
    kept = sorted((scored[i] for i in kept_indices), key=lambda c: (-c.score, c.action.to_choice()))
    return Narrowed(
        kept=kept,
        considered=len(pool),
        uncovered=tuple(sorted(needed[k] for k in uncovered)),
        uncovered_options=tuple(options[k] for k in sorted(uncovered, key=lambda k: needed[k])),
        for_coverage=for_coverage,
        for_score=len(kept) - for_coverage,
    )


def sort_only(pool, scored):  # noqa: ANN001, ANN201
    return sorted(range(len(scored)), key=lambda i: (-scored[i].score, pool[i].to_choice()))


def main() -> None:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-dir", type=Path, default=ROOT / "data" / "selfplay-mc0")
    ap.add_argument("--positions", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=319)
    ap.add_argument("--repeats", type=int, default=3, help="per-call repeats (median kept)")
    ap.add_argument("--profile-positions", type=int, default=1500)
    ap.add_argument("--out", type=Path, required=True, help="directory for the json and text")
    ap.add_argument(
        "--children", action="store_true",
        help="time the depth-2 sub-game positions one turn below the recorded ones instead",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    positions = load_positions(args.games_dir, args.positions, args.seed)
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)
    node = rustnode.node_for(reg)
    if node is None:
        raise SystemExit("the port is not available")
    if args.children:
        positions = depth2_children(reg, positions, args.seed, args.positions)
        print(f"{len(positions)} depth-2 sub-game positions", flush=True)
        node = rustnode.node_for(reg)
    read, ns = _cycle_clock()
    import psutil

    child = psutil.Process(node._process.pid)  # noqa: SLF001

    def timed(fn, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        c0 = read()
        value = fn(*a, **k)
        return value, (read() - c0) * ns / 1e6

    def timed_rep(fn, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        times = []
        value = None
        for _ in range(args.repeats):
            value, t = timed(fn, *a, **k)
            times.append(t)
        return value, statistics.median(times)

    # Warm the port and the dex caches outside the timed pass.
    for pos in positions[:20]:
        for side in (0, 1):
            narrow(reg, pos, side, limit=12)

    rows: list[dict] = []
    mismatches = 0
    refused = 0
    tt0 = time.thread_time()
    cycles_sum_ms = 0.0
    child0 = sum(child.cpu_times()[:2])
    port_wall = 0.0
    for pos_index, pos in enumerate(positions):
        for side in (0, 1):
            legal, t_legal = timed_rep(side_actions, reg, pos, side)
            pool, t_drop = timed_rep(drop_dead_actions, reg, pos, side, legal)
            row = kind_of(reg, pos, side, len(legal))
            row.update(pos=pos_index, side=side, n_pool=len(pool), t_legal=t_legal, t_drop=t_drop)
            row["switches"] = sum(
                1 for a in legal if any(isinstance(s, SwitchAction) for s in a.slots)
            )
            row["mega"] = sum(
                1 for a in legal if any(isinstance(s, MoveAction) and s.mega for s in a.slots)
            )
            cycles_sum_ms += (t_legal + t_drop) * args.repeats
            if not pool:
                rows.append(row)
                continue
            # `RustNode.score` in its four pieces, each timed: the request's dicts, its
            # JSON line, the crossing (write, the child's work, the header line read and
            # parsed) and the unpacking. The first ask of this pool, so nothing is held.
            def build(pos=pos, side=side, pool=pool):  # noqa: ANN001, ANN202
                return {
                    "kind": "score",
                    "position": rustnode._position(pos),  # noqa: SLF001
                    "side": side,
                    "candidates": [[rustnode.dump_action(a) for a in c.slots] for c in pool],
                }

            def cross(payload):  # noqa: ANN001, ANN202
                proc = node._process  # noqa: SLF001
                proc.stdin.write(payload + b"\n")
                proc.stdin.flush()
                return proc.stdout.readline()

            def unpack(response):  # noqa: ANN001, ANN202
                if response.get("refused"):
                    return None
                return [
                    (float(score), [(int(a), int(b), bool(c), float(d), bool(e)) for a, b, c, d, e in parts])
                    for score, parts in zip(response["scores"], response["detail"], strict=True)
                ]

            w0 = time.perf_counter()
            k0 = sum(child.cpu_times()[:2])
            request, t_build = timed(build)
            payload, t_dumps = timed(rustnode._payload, request)  # noqa: SLF001
            line, t_cross = timed(cross, payload)
            response, t_loads = timed(json.loads, line.decode("utf-8"))
            scored_raw, t_unpack = timed(unpack, response)
            row["child_ms"] = (sum(child.cpu_times()[:2]) - k0) * 1000
            port_wall += time.perf_counter() - w0
            t_port = t_build + t_dumps + t_cross + t_loads + t_unpack
            row.update(
                t_build=t_build, t_dumps=t_dumps, t_cross=t_cross, t_loads=t_loads,
                t_unpack=t_unpack, request_bytes=len(payload), answer_bytes=len(line),
            )
            cycles_sum_ms += t_port
            if scored_raw is None:
                refused += 1
                rows.append(row)
                continue

            def detail(pool=pool, scored_raw=scored_raw):  # noqa: ANN001, ANN202
                out = []
                for action, (total, parts) in zip(pool, scored_raw, strict=True):
                    text = tuple(
                        f"{action.slots[slot].describe(reg)} -> "
                        f"{'foe' if is_foe else 'ally'}{target_slot + 1} {signed:+.3f}"
                        f"{'' if exact else '?'}"
                        for slot, target_slot, is_foe, signed, exact in parts
                    )
                    out.append(Candidate(action=action, score=total, detail=text))
                return out

            scored, t_detail = timed(detail)
            order, t_sort = timed_rep(sort_only, pool, scored)
            row.update(t_port=t_port, t_detail=t_detail, t_sort=t_sort)
            cycles_sum_ms += t_detail + t_sort * args.repeats
            for limit in LIMITS:
                got, t_cover = timed_rep(cover, reg, pool, scored, limit, order)
                kinds, t_kinds = timed(_count_kinds, reg, got.kept)
                got = replace(got, by_kind=kinds)
                cycles_sum_ms += t_cover * args.repeats + t_kinds
                row[f"t_cover{limit}"] = t_cover
                row[f"t_kinds{limit}"] = t_kinds
                want = narrow(reg, pos, side, limit=limit)
                same = (
                    [(c.action.to_choice(), c.score, c.detail) for c in got.kept]
                    == [(c.action.to_choice(), c.score, c.detail) for c in want.kept]
                    and got.uncovered == want.uncovered
                    and got.for_coverage == want.for_coverage
                    and got.by_kind == want.by_kind
                )
                if not same:
                    mismatches += 1
            # `narrow` itself, whole, at the shipped width: a fresh request so the port
            # answers again (nothing is held between calls unless `hold_positions`).
            _n, t_whole = timed(narrow, reg, pos, side, limit=12)
            row["t_narrow12"] = t_whole
            cycles_sum_ms += t_whole
            rows.append(row)
        # The narrow checks above ask the port too; the child's CPU is read around the
        # whole pass and split per request below.
    tt_total = (time.thread_time() - tt0) * 1000
    child_cpu = (sum(child.cpu_times()[:2]) - child0) * 1000
    port_requests = sum(1 for r in rows if "t_port" in r) * (1 + len(LIMITS) + 1)

    # The clock's check: the same legal-set work as one loop, read by both clocks. Per-call
    # cycles summed must agree with the thread's tick-counted total over seconds of work.
    check_cycles = 0.0
    tt1 = time.thread_time()
    for _ in range(3):
        for pos in positions:
            for side in (0, 1):
                _v, t = timed(lambda p=pos, s=side: drop_dead_actions(reg, p, s, side_actions(reg, p, s)))
                check_cycles += t
    check_thread = (time.thread_time() - tt1) * 1000

    # The profile: side_actions and drop_dead_actions only, on the first positions.
    prof = cProfile.Profile()
    prof.enable()
    for pos in positions[: args.profile_positions]:
        for side in (0, 1):
            drop_dead_actions(reg, pos, side, side_actions(reg, pos, side))
    prof.disable()
    text = io.StringIO()
    stats = pstats.Stats(prof, stream=text)
    stats.sort_stats("tottime").print_stats(25)
    stats.sort_stats("cumulative").print_stats(30)
    (args.out / "profile.txt").write_bytes(text.getvalue().encode("utf-8"))

    summary = {
        "positions": len(positions),
        "pools": len(rows),
        "repeats": args.repeats,
        "mismatches": mismatches,
        "refused": refused,
        "thread_time_ms": tt_total,
        "check_cycles_ms": check_cycles,
        "check_thread_time_ms": check_thread,
        "cycles_sum_ms": cycles_sum_ms,
        "child_cpu_ms": child_cpu,
        "port_requests": port_requests,
        "port_wall_ms_first_requests": port_wall * 1000,
        "ns_per_cycle": ns,
        "pid": os.getpid(),
        "source": rustnode.__file__,
        "binary": str(rustnode.binary_path()),
    }
    payload = {"summary": summary, "rows": rows}
    (args.out / "rows.json").write_bytes(json.dumps(payload).encode("utf-8"))
    print(json.dumps(summary, indent=1))
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[legal_bucket(r["n_legal"])].append(r["t_legal"])
    for key, values in by.items():
        print(key, len(values), f"{statistics.mean(values):.3f} ms")


if __name__ == "__main__":
    main()
