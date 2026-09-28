"""The analysis mode (検討モード, IKA-337 stage 1): pick a position and read it with no
budget, deepening until the person stops it.

A move decision of a recorded game -- a game against a person (`tools/play_human.py`), a
board game or a generated one (`pool_match` / `poolplay` records) -- or the move in hand of
a game still being played (``play_human --current-out``) is read from one side's view: its
bench belief (the other side's completions of its unseen bench, weighted by the selection
solve as `humanplay` weighs them), its menus at a fixed width, and then the best-first
deepening (`deepen.deepen_belief` / `deepen.deepen_root`, the ``m...h`` reading) with the
root's swap oracle, as IKA-307 recommends for a long read: width first, the rest to
deepening with the oracle. The deepening is anytime, so the answer after every step is an
answer, and the screen of IKA-332 shows it as it forms.

**No budget.** The deepening is handed `ENDLESS` cells and a `StopCost` that reads the wall
clock until a `threading.Event` is set, and the budget's full value after it: the loop
stops at the top of its next step. The event is set by the page's Stop button, by
``max_seconds``, by ``max_steps`` (after that many steps -- on this reading of the clock
the same position, settings and step count give the same answer, bit for bit, whatever
the machine and its load or the number of threads), by the memory watch, or never; the
deepening also stops on its own when nothing inside the depth guard is worth a step
(``exhausted``).

**The depth guard.** Long reads meet `deepen.MAX_LEVELS` (IKA-307: 69-86% of the steps of
a 44-second move were decided by it). ``levels`` raises it (a label's ``g<L>``); the lines
that end at the guard with a cell still worth a step are counted as the read goes
(`guard_lines`, the report's ``lines`` guard count) and shown.

**Widening while it reads** (IKA-354). A wider width asked for the position being read
(the page's gear, the same position, side and guard) does not stop the read: the menus at
the new width are built with the read's own ranking and appended to the root, their cells
filled at depth 1 and the root re-solved, the tree already deepened kept (`deepen.Grow`,
`Widen`). A narrower width, or any other change, reads again from the start as before.

**Memory.** A long read grows the tree. `MemoryWatch` reads the host's free memory, this
process's resident memory with its worker processes', and the card's (``nvidia-smi``),
and stops the read with a reason before a limit (`Limits`).
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .actions import SideAction
from .budget import Budget
from .deepen import ALL_ACTIONS, GAP_FLOOR, MAX_LEVELS, Step, _Node
from .equilibrium import EquilibriumError
from .hidden import DEFAULT_BENCH_DROP, completions, seen_slots
from .humanplay import (
    BELIEF_EPSILON,
    board_view,
    sheet_view,
    solve_entry,
    solve_move,
)
from .notes_ja import notes_ja
from .payoff import HP_SHARE
from .position import Position
from .progress import Reader, Recorder
from .regulation import Regulation
from .selection_book import BenchPrior, set_from_json
from .selfplay import GameRecord, _believed, _bench_weights, _menus, _set_json
from .teams import Roster

#: The deepening's budget in the analysis mode: more cells than any read reaches (and it
#: fits the page's u32). `StopCost` reads it as spent once the read is told to stop.
ENDLESS = (1 << 31) - 1

#: IKA-307's allocation: width first -- the widest menu the human form uses, every legal
#: action on most turns -- and the rest to deepening with the root's swap oracle over
#: every legal action (``sall``).
DEFAULT_WIDTH = 64
DEFAULT_ORACLE = ALL_ACTIONS

#: Why a read stopped, as the page says it.
STOPS = {
    "person": "止めました",
    "exhausted": "読み切り（深さの上限の内に深化のステップに値するセルが無い）",
    "steps": "指定の深化のステップ数で止めました",
    "time": "指定の秒で止めました",
    "memory": "メモリの上限の手前で止めました",
    "new": "別の局面に移りました",
    "error": "止まりました（エラー）",
}


class StopCost:
    """A reading of the deepening's budget (`deepen.Cost`'s place) that does not run out
    until ``stop`` is set: the milliseconds since ``start`` (``cell`` 1, as `WallCost`),
    and `ENDLESS` -- the whole budget -- once it is. The loop reads it only at the top of a
    step, so a stop set inside a step (the progress callback, the page, a timer) ends the
    read after that step; the counts are not read."""

    cell = 1.0

    def __init__(self, stop: threading.Event, start: float | None = None) -> None:
        self.stop = stop
        self.start = time.perf_counter() if start is None else start

    def ms(  # noqa: ARG002
        self, fills: int, refines: int, cells: int, probed: int = 0, qs: int = 0
    ) -> float:
        if self.stop.is_set():
            return float(ENDLESS)
        return min((time.perf_counter() - self.start) * 1000.0, float(ENDLESS - 1))


# ----------------------------------------------------------------------------- positions


@dataclass
class Point:
    """One move decision of a game: its position (the record's canonical JSON, parsed when
    read) and what each side had shown there (`Decision.shown`: ``seen[i]`` is side i's
    identities its opponent's belief was conditioned on)."""

    decision: int
    turn: int
    position: dict[str, Any]
    seen: tuple[frozenset[str], frozenset[str]]
    #: The record's value at this decision, side 0's (``searchValue``), where it has one.
    value: float | None = None
    #: The actions each side played here (choice strings, side 0's first), where recorded.
    played: tuple[str | None, str | None] = (None, None)

    def pos(self) -> Position:
        return Position.from_json(self.position)


@dataclass
class Game:
    """A recorded game's move decisions, and what reading them from a side needs: the two
    sheets (to complete the unseen bench and to solve the selection that weighs it), the
    leads, whether the record hid the bench."""

    label: str
    points: list[Point]
    teams: tuple[Roster, Roster] | None
    leads: tuple[frozenset[str] | None, frozenset[str] | None] = (None, None)
    #: The side read by default: the person's in a game against a person, else 0.
    side: int = 0
    information: str = "hidden-bench"
    note: str = ""


def _roster(reg: Regulation, data: dict[str, Any]) -> Roster:
    sets = [set_from_json(s) for s in data["sheet"]]
    return Roster(
        id=str(data.get("id", "")), name=str(data.get("name", data.get("id", ""))), reg=reg,
        sets=sets, shown_stats=[None] * len(sets), source={},
    )


def _pool_teams(
    record: dict[str, Any], pools: dict[str, Any]
) -> tuple[tuple[Roster, Roster] | None, str]:
    """The two sheets of a record from its pool: ``teams`` (a game against a person) or
    ``pool.teams`` (board and generation records) name the pool's team ids."""
    from .pool import load_pool

    info = record.get("pool") or {}
    ids = record.get("teams") or info.get("teams")
    pool_id = info.get("id")
    if not ids or not pool_id:
        return None, "記録にチームの id が無いので、裏を全公開として読みます"
    pool = pools.get(pool_id)
    if pool is None:
        try:
            pool = load_pool(pool_id)
        except (FileNotFoundError, OSError, ValueError) as problem:
            return None, f"構築プール {pool_id} が読めない（{problem}）ので、全公開として読みます"
        pools[pool_id] = pool
    note = ""
    if info.get("sha256") and info["sha256"] != pool.sha256:
        note = f"構築プール {pool_id} の sha256 が記録と違います"
    by_id = {t.id: t for t in pool.teams}
    if ids[0] not in by_id or ids[1] not in by_id:
        return None, f"構築プール {pool_id} に {ids} が無いので、全公開として読みます"
    return (by_id[ids[0]], by_id[ids[1]]), note


def _leads(record: dict[str, Any]) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    leads = record.get("leads")
    if not leads or len(leads) != 2:
        return (None, None)
    return (frozenset(leads[0]), frozenset(leads[1]))


def _points(record: dict[str, Any]) -> list[Point]:
    out = []
    for index, decision in enumerate(record.get("decisions", [])):
        if decision.get("kind") != "move":
            continue
        shown = decision.get("shownIdentities") or decision.get("shown") or [[], []]
        value = decision.get("searchValue")
        out.append(Point(
            decision=index, turn=int(decision["turn"]), position=decision["position"],
            seen=(frozenset(shown[0]), frozenset(shown[1])),
            value=None if value is None else float(value),
            played=(decision.get("ownChosen"), decision.get("foeChosen")),
        ))
    return out


def _played(
    reg: Regulation, pos: Position, point: Point, me: int, loc: Any  # noqa: ANN401
) -> list[dict[str, Any]]:
    """The actions the record played at ``point``, the reading side's first (IKA-349): each
    side, its label's text (as the reading labels its menus, so the page finds it there) and
    parts. A side with no recorded action, or one the legal actions do not parse, is left out."""
    from .actions import side_actions
    from .humanplay import parse_choice
    from .progress import action_label, part_json

    out = []
    for side in (me, 1 - me):
        choice = point.played[side]
        if not choice:
            continue
        try:
            action = parse_choice(choice, side_actions(reg, pos, side))
        except (ValueError, KeyError, IndexError):
            continue
        label = action_label(reg, action, pos, side, loc)
        out.append({"side": side, "text": str(label), "parts": [part_json(p) for p in label.rich]})
    return out


def game_from_record(
    record: dict[str, Any], index: int, pools: dict[str, Any] | None = None
) -> Game:
    """One record line (`GameRecord.to_json`'s shape) as a `Game`."""
    pools = {} if pools is None else pools
    teams, note = _pool_teams(record, pools)
    human = record.get("human")
    side = int(human["side"]) if human else 0
    outcome = record.get("outcome")
    names = [t.name for t in teams] if teams else ["側0", "側1"]
    if human:
        you = "勝ち" if outcome is not None and (outcome > 0.5) == (side == 0) else (
            "負け" if outcome is not None else "打ち切り")
        label = f"局 {record.get('gameIndex', index)}: 人（側{side}）の{you}・{record.get('turns')} ターン"
    else:
        result = "側0 の勝ち" if outcome == 1.0 else "側1 の勝ち" if outcome == 0.0 else "打ち切り"
        label = (
            f"局 {record.get('gameIndex', index)}: {names[0]} 対 {names[1]}・{result}"
            f"・{record.get('turns')} ターン"
        )
    return Game(
        label=label, points=_points(record), teams=teams, leads=_leads(record), side=side,
        information=str(record.get("information", "open")), note=note,
    )


def load_games(
    path: Path, *, pools: dict[str, Any] | None = None, limit: int | None = None
) -> list[Game]:
    """The games of a record file (one JSON line each), the first ``limit`` of them."""
    pools = {} if pools is None else pools
    games = []
    with path.open("rb") as handle:
        for index, raw in enumerate(handle):
            if limit is not None and len(games) >= limit:
                break
            if not raw.strip():
                continue
            games.append(game_from_record(json.loads(raw), index, pools))
    return games


def point_json(
    reg: Regulation,
    pos: Position,
    side: int,
    seen: Sequence[frozenset[str]],
    leads: Sequence[Sequence[str]] | None,
    teams: tuple[Roster, Roster],
    *,
    label: str,
) -> dict[str, Any]:
    """The move in hand of a game being played, for the analysis mode to read
    (``play_human --current-out``): the position in its canonical JSON, what each side had
    shown, the leads, the two sheets whole (a roster file is not in any pool), the side to
    read (the person's)."""
    return {
        "kind": "analysis-point",
        "label": label,
        "turn": pos.turn,
        "side": side,
        "position": pos.to_json(),
        "shown": [sorted(seen[0]), sorted(seen[1])],
        "leads": [list(x) for x in leads] if leads else None,
        "teams": [
            {"id": t.id, "name": t.name, "sheet": [_set_json(reg, s) for s in t.sets]}
            for t in teams
        ],
        "information": "hidden-bench",
    }


def write_point(path: Path, payload: dict[str, Any]) -> None:
    """Writes a point file whole and then renames it over the old one, so a reader never
    sees half of it. LF, UTF-8 (bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.with_name(path.name + ".tmp")
    fresh.write_bytes((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(fresh, path)


def load_point(reg: Regulation, path: Path) -> Game:
    """A point file as a one-decision game."""
    data = json.loads(path.read_bytes().decode("utf-8"))
    teams = (_roster(reg, data["teams"][0]), _roster(reg, data["teams"][1]))
    leads = data.get("leads")
    shown = data["shown"]
    point = Point(decision=0, turn=int(data["turn"]), position=data["position"],
                  seen=(frozenset(shown[0]), frozenset(shown[1])))
    return Game(
        label=str(data.get("label", "進行中の局")), points=[point], teams=teams,
        leads=(frozenset(leads[0]), frozenset(leads[1])) if leads else (None, None),
        side=int(data.get("side", 0)), information=str(data.get("information", "hidden-bench")),
    )


# ----------------------------------------------------------------------------- the guard


def _signal_now(node: _Node) -> np.ndarray:
    """`deepen._signal`'s matrix without writing its cache (the reader writes nothing)."""
    if node.signal is not None:
        return node.signal
    eq = node.equilibrium
    gap = eq.row_ev_loss[:, None] + eq.col_ev_loss[None, :]
    return node.payoff * (1.0 - node.payoff) / (gap + GAP_FLOOR)


def guard_lines(root: Any, guard: int) -> tuple[int, int]:  # noqa: ANN401 - _Node or _BeliefRoot
    """(lines at the guard, nodes below the root): the deepened lines whose last node sits
    ``guard`` levels down with a cell still worth a step -- what the report's ``lines``
    count as ``guard`` when the read ends (`deepen._Watch`) -- and the tree's size. Reads
    the tree, writes nothing."""
    lines = 0
    nodes = 0
    stack = list(root.children.values())
    while stack:
        for _weight, child in stack.pop():
            if not isinstance(child, _Node):
                continue
            nodes += 1
            if child.children:
                stack.extend(child.children.values())
                continue
            if child.level < guard:
                continue
            signal = _signal_now(child)
            live = np.ones(signal.shape, dtype=bool)
            for cell in child.refused:
                live[cell] = False
            if live.any() and float(signal[live].max()) > 0.0:
                lines += 1
    return lines, nodes


# ----------------------------------------------------------------------------- memory


@dataclass(frozen=True)
class Limits:
    """Where the memory watch stops a read. ``rss_gb``: this process and its worker
    processes, resident. ``free_gb``: the host's available memory may not fall under it.
    ``gpu_gb``: the card's memory in use (every process on it; the house rule stops at 11
    GB of the 12). 0 turns one off.

    ``hard_free_gb``: the floor under ``free_gb`` that a person's game keeps whatever the
    cause (IKA-355, `Reading.brake`); the analysis mode's watch (`Reading.over`) does not
    read it."""

    rss_gb: float = 10.0
    free_gb: float = 2.0
    gpu_gb: float = 11.0
    #: The share of a limit from which the page warns.
    warn: float = 0.9
    hard_free_gb: float = 0.5


def host_available_gb() -> float | None:
    """The host's available physical memory, or None where it cannot be read."""
    if sys.platform == "win32":
        class _Status(ctypes.Structure):
            _fields_ = [  # noqa: RUF012
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                ("page_total", ctypes.c_ulonglong), ("page_avail", ctypes.c_ulonglong),
                ("virtual_total", ctypes.c_ulonglong), ("virtual_avail", ctypes.c_ulonglong),
                ("extended", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.length = ctypes.sizeof(_Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None
        return status.avail / 2**30
    with contextlib.suppress(OSError, ValueError):
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 2**30
    return None


def process_rss_gb(pid: int | None = None) -> float | None:
    """A process's resident memory (this one when ``pid`` is None), or None."""
    if sys.platform == "win32":
        class _Counters(ctypes.Structure):
            _fields_ = [  # noqa: RUF012
                ("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                ("peak_ws", ctypes.c_size_t), ("ws", ctypes.c_size_t),
                ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
            ]

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
        handle = (
            kernel.GetCurrentProcess() if pid is None else kernel.OpenProcess(0x1000 | 0x0010, False, pid)
        )
        if not handle:
            return None
        try:
            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None
            return counters.ws / 2**30
        finally:
            if pid is not None:
                kernel.CloseHandle(handle)
    with contextlib.suppress(OSError, ValueError):
        fields = Path(f"/proc/{pid or 'self'}/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    return None


def tree_rss_gb() -> float | None:
    """This process's resident memory with its live children's (the deepening's worker
    processes)."""
    import multiprocessing

    own = process_rss_gb()
    if own is None:
        return None
    for child in multiprocessing.active_children():
        got = process_rss_gb(child.pid)
        if got is not None:
            own += got
    return own


def gpu_used_gb() -> tuple[float, float] | None:
    """The card's memory in use and its total (GB, first card), from ``nvidia-smi``, or
    None without one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        used, total = (float(x) for x in out.splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return used / 1024, total / 1024


@dataclass
class Reading:
    """One look at the memory."""

    rss_gb: float = math.nan
    free_gb: float = math.nan
    gpu_gb: float = math.nan
    gpu_total_gb: float = math.nan

    def over(self, limits: Limits, share: float = 1.0) -> str:
        """The first limit this reading is past (at ``share`` of it), in words, or ""."""
        if limits.rss_gb > 0 and self.rss_gb >= share * limits.rss_gb:
            return f"このプロセスと補助のメモリ {self.rss_gb:.1f} GB（上限 {limits.rss_gb:g} GB）"
        if limits.free_gb > 0 and self.free_gb <= limits.free_gb / share:
            return f"ホストの空きメモリ {self.free_gb:.1f} GB（下限 {limits.free_gb:g} GB）"
        if limits.gpu_gb > 0 and self.gpu_gb >= share * limits.gpu_gb:
            return f"GPU のメモリ {self.gpu_gb:.1f} GB（上限 {limits.gpu_gb:g} GB）"
        return ""

    def brake(self, limits: Limits, base_rss: float, share: float = 1.0) -> tuple[str, str]:
        """A person's game's reading of the limits (IKA-355): what this reading asks of
        the move in hand, and why in words.

        ``("stop", why)``: stop deepening -- this process with its workers past
        ``rss_gb``, the card past ``gpu_gb``, the host's free memory under
        ``hard_free_gb`` whatever the cause, or under ``free_gb`` because this process
        grew: without its growth since ``base_rss`` (the resident memory when the move's
        deepening began) the host would be at the floor or above.
        ``("low", why)``: under ``free_gb``, but other work took the memory (the host
        would be under the floor without this move's growth too) -- read on, say so.
        ``("", "")``: nothing. ``share`` reads each limit at that share, as `over`."""
        if limits.rss_gb > 0 and self.rss_gb >= share * limits.rss_gb:
            return "stop", f"このプロセスと補助のメモリ {self.rss_gb:.1f} GB（上限 {limits.rss_gb:g} GB）"
        if limits.gpu_gb > 0 and self.gpu_gb >= share * limits.gpu_gb:
            return "stop", f"GPU のメモリ {self.gpu_gb:.1f} GB（上限 {limits.gpu_gb:g} GB）"
        if limits.hard_free_gb > 0 and self.free_gb <= limits.hard_free_gb / share:
            return "stop", f"ホストの空きメモリ {self.free_gb:.1f} GB（底 {limits.hard_free_gb:g} GB）"
        if limits.free_gb > 0 and self.free_gb <= limits.free_gb / share:
            grown = self.rss_gb - base_rss
            grown = 0.0 if math.isnan(grown) else max(0.0, grown)
            if self.free_gb + grown >= limits.free_gb / share:
                return "stop", (
                    f"ホストの空きメモリ {self.free_gb:.1f} GB（下限 {limits.free_gb:g} GB）、"
                    f"この手の読みで {grown:.1f} GB 増えた"
                )
            return "low", (
                f"ホストの空きメモリ {self.free_gb:.1f} GB（下限 {limits.free_gb:g} GB）は"
                f"ほかの処理によるもの（この手の読みの増え {grown:.1f} GB）"
            )
        return "", ""


class MemoryWatch:
    """Reads the memory every ``every`` seconds (the card every ``gpu_every``) while a read
    runs, and stops it (``halt("memory", why)``) the first time a reading is past a limit.
    ``tick`` is called after each look (the status frames). ``judge`` reads a reading into
    the reason to stop ("" for none), `Reading.over` by default; a person's game gives its
    own (IKA-355). ``lock`` is held over each look, so a caller can look in between."""

    def __init__(
        self,
        limits: Limits,
        halt: Callable[[str, str], None],
        *,
        every: float = 0.5,
        gpu_every: float = 5.0,
        tick: Callable[[Reading], None] | None = None,
        read_host: Callable[[], float | None] = host_available_gb,
        read_rss: Callable[[], float | None] = tree_rss_gb,
        read_gpu: Callable[[], tuple[float, float] | None] = gpu_used_gb,
        judge: Callable[[Reading], str] | None = None,
    ) -> None:
        self.limits = limits
        self.judge = judge
        self.lock = threading.RLock()
        self.halt = halt
        self.every = every
        self.gpu_every = gpu_every
        self.tick = tick
        self.read_host = read_host
        self.read_rss = read_rss
        self.read_gpu = read_gpu
        self.last = Reading()
        self.peak = Reading(rss_gb=0.0, free_gb=math.inf, gpu_gb=0.0)
        self._gpu_at = -math.inf
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def look(self) -> Reading:
        with self.lock:
            return self._look()

    def _look(self) -> Reading:
        now = time.perf_counter()
        reading = Reading(gpu_gb=self.last.gpu_gb, gpu_total_gb=self.last.gpu_total_gb)
        host = self.read_host()
        rss = self.read_rss()
        reading.free_gb = math.nan if host is None else host
        reading.rss_gb = math.nan if rss is None else rss
        if self.limits.gpu_gb > 0 and now - self._gpu_at >= self.gpu_every:
            self._gpu_at = now
            got = self.read_gpu()
            if got is not None:
                reading.gpu_gb, reading.gpu_total_gb = got
        self.last = reading
        if not math.isnan(reading.rss_gb):
            self.peak.rss_gb = max(self.peak.rss_gb, reading.rss_gb)
        if not math.isnan(reading.free_gb):
            self.peak.free_gb = min(self.peak.free_gb, reading.free_gb)
        if not math.isnan(reading.gpu_gb):
            self.peak.gpu_gb = max(self.peak.gpu_gb, reading.gpu_gb)
        why = reading.over(self.limits) if self.judge is None else self.judge(reading)
        if why:
            self.halt("memory", why)
        return reading

    def _run(self) -> None:
        while not self._done.is_set():
            with self.lock:
                reading = self._look()
                if self.tick is not None:
                    self.tick(reading)
            self._done.wait(self.every)

    def start(self) -> MemoryWatch:
        self._thread = threading.Thread(target=self._run, daemon=True, name="memory-watch")
        self._thread.start()
        return self

    def close(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=15)


# ----------------------------------------------------------------------------- one read


@dataclass
class Settings:
    """How a read is set up: the width of both menus, the root's oracle (a width, or
    `deepen.ALL_ACTIONS` for every legal action, or None), the depth guard, the menus'
    ranking, and whether to read the bench open."""

    width: int = DEFAULT_WIDTH
    oracle: int | None = DEFAULT_ORACLE
    levels: int = MAX_LEVELS
    #: The depth discount (``d<P>``, IKA-342), a factor a ply, or None: none.
    discount: float | None = None
    rank_fill: str = "refs2"
    rank_by_leaf: bool = True
    bench_drop: str = DEFAULT_BENCH_DROP
    open_information: bool = False
    #: Least milliseconds between two steps sent to the page.
    interval_ms: float = 100.0
    #: Least milliseconds between two walks of the tree for the guard's count; a walk that
    #: took t waits at least 10 t (a big tree is walked less often).
    guard_every_ms: float = 1000.0
    #: Read by a ladder of stages instead of the deepening (IKA-367/364: a name in
    #: `ladder.LADDERS` or stages joined by ``+``), until the person stops it or the last
    #: stage completes; its cells on the worker processes `humanplay.use_ladder_pool`
    #: started. None: the deepening.
    ladder: str | None = None

    def oracle_label(self) -> str:
        if self.oracle is None:
            return "none"
        return "sall" if self.oracle >= ALL_ACTIONS else f"s{self.oracle}"


class Widen:
    """A read's width raised while it runs (IKA-354): `request` from the page's thread,
    `deepen.Grow` on the deepening's.

    ``build(width)`` gives both sides' menus at a width, ranked as the read's own were;
    ``on_grown(width, menus)`` is told of each widening as it is handed to the deepening.
    The deepening asks at the top of every step, so a request is taken at the next one.
    `close` ends it; a width asked for and never taken (the read ended first) is returned,
    to be read from the start."""

    def __init__(
        self,
        width: int,
        build: Callable[[int], tuple[list[SideAction], list[SideAction]]],
        on_grown: Callable[[int, tuple[list[SideAction], list[SideAction]]], None] | None = None,
    ) -> None:
        self.width = width
        self.build = build
        self.on_grown = on_grown
        self.pending: int | None = None
        self.closed = False
        #: Widenings handed to the deepening.
        self.grown = 0
        self.lock = threading.Lock()

    def request(self, width: int) -> bool:
        """Asks for ``width`` (at least the width now). False once the read has ended."""
        with self.lock:
            if self.closed:
                return False
            if width > max(self.width, self.pending or 0):
                self.pending = width
            return True

    def __call__(self, spent: float) -> tuple[list[SideAction], list[SideAction]] | None:  # noqa: ARG002
        with self.lock:
            width, self.pending = self.pending, None
        if width is None:
            return None
        menus = self.build(width)
        with self.lock:
            self.width = width
            self.grown += 1
        if self.on_grown is not None:
            self.on_grown(width, menus)
        return menus

    def close(self) -> int | None:
        """Ends it; the width asked for and not taken, if any."""
        with self.lock:
            self.closed = True
            left, self.pending = self.pending, None
        return left


class Session:
    """One read's progress callback (`deepen.Progress`) and its state: the steps, the lines
    at the guard, why it stopped. Stops the read (``stop``) after ``max_steps`` steps."""

    def __init__(
        self,
        stop: threading.Event,
        *,
        guard: int,
        recorder: Callable[[Step], None] | None = None,
        max_steps: int | None = None,
        guard_every_ms: float = 1000.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.stop = stop
        self.guard = guard
        self.recorder = recorder
        self.max_steps = max_steps
        self.guard_every_ms = guard_every_ms
        self.clock = clock
        self.started = clock()
        self.steps = 0
        self.guard_count = 0
        self.nodes = 0
        self.depth = 1
        self.reason = ""
        self.why = ""
        self._walked = -math.inf
        self._walk_ms = 0.0
        self.lock = threading.Lock()
        #: The read's width, raised while it runs (IKA-354), and what it reads (`Service`
        #: compares a new request with it: source, game, decision, side, guard).
        self.widen: Widen | None = None
        self.key: dict[str, Any] = {}
        #: A width asked for too late (the read had ended), read from the start next.
        self.left: int | None = None

    def halt(self, reason: str, why: str = "") -> None:
        """Stops the read; the first reason given is the one kept."""
        with self.lock:
            if not self.reason:
                self.reason = reason
                self.why = why
        self.stop.set()

    def __call__(self, step: Step) -> None:
        if step.kind not in ("start", "done"):
            self.steps = step.index
        self.depth = step.depth
        now = self.clock()
        wait = max(self.guard_every_ms, 10.0 * self._walk_ms)
        if step.kind in ("start", "done") or (now - self._walked) * 1000.0 >= wait:
            self.guard_count, self.nodes = guard_lines(step.root, self.guard)
            self._walked = self.clock()
            self._walk_ms = (self._walked - now) * 1000.0
        if self.recorder is not None:
            self.recorder(step)
        if self.max_steps is not None and step.kind != "done" and step.index >= self.max_steps:
            self.halt("steps")

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started


@dataclass
class Result:
    """A read's answer when it stopped."""

    side: int
    turn: int
    #: The side's menu (choice strings) and its mixture; the other side's menu and the
    #: mixture this side models for it.
    ours: list[str]
    strategy: np.ndarray
    theirs: list[str]
    model: list[float]
    #: In side 0's units, and as the side's own.
    value0: float
    value: float
    steps: int
    seconds: float
    stop: str
    guard: int
    guard_lines: int
    nodes: int
    exact: bool
    classes: int
    deepened: dict[str, Any] | None
    notes: list[str] = field(default_factory=list)
    memory: Reading | None = None
    #: The width it ended at, and the widths it was widened to while it read (IKA-354).
    width: int = 0
    widened_to: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "side": self.side, "turn": self.turn, "ours": self.ours,
            "strategy": [float(x) for x in self.strategy], "theirs": self.theirs,
            "model": [float(x) for x in self.model], "value0": self.value0,
            "value": self.value, "steps": self.steps, "seconds": round(self.seconds, 3),
            "stop": self.stop, "guard": self.guard, "guardLines": self.guard_lines,
            "nodes": self.nodes, "exact": self.exact, "classes": self.classes,
            "deepened": self.deepened, "notes": self.notes,
            **({"width": self.width} if self.width else {}),
            **({"widenedTo": self.widened_to} if self.widened_to else {}),
            **({"memory": {"peakRssGb": round(self.memory.rss_gb, 3),
                           "leastFreeGb": round(self.memory.free_gb, 3),
                           "peakGpuGb": round(self.memory.gpu_gb, 3)}}
               if self.memory is not None else {}),
        }


def side_names(side: int) -> tuple[str, str]:
    """The two sides as the page names them: the side read, and its opponent."""
    return tuple("検討する側" if s == side else "相手" for s in (0, 1))  # type: ignore[return-value]


class Analyzer:
    """Reads positions with one leaf. ``evaluate`` None is hp-share (no selection solve,
    so the bench belief is uniform)."""

    def __init__(
        self,
        reg: Regulation,
        evaluate: Any,  # noqa: ANN401 - a leaf evaluator or None
        name: str,
        *,
        loc: Any = None,  # noqa: ANN401
        settings: Settings | None = None,
    ) -> None:
        self.reg = reg
        self.evaluate = evaluate
        self.name = name
        self.loc = loc
        self.settings = settings or Settings()
        self._priors: dict[tuple[str, str], tuple[BenchPrior, BenchPrior] | None] = {}

    @property
    def leaf(self) -> Any:  # noqa: ANN401
        return self.evaluate if self.evaluate is not None else HP_SHARE.batch

    def prior(self, game: Game) -> tuple[BenchPrior, BenchPrior] | None:
        """The bench prior of both sides from the two sheets' selection game, solved once
        per pair (`humanplay.play`'s: the solve's mixtures, epsilon `BELIEF_EPSILON`)."""
        if self.evaluate is None or game.teams is None:
            return None
        key = (game.teams[0].id, game.teams[1].id)
        if key not in self._priors:
            entry = solve_entry(self.reg, game.teams, self.evaluate, self.name)
            species = tuple([s.species for s in t.sets] for t in game.teams)
            self._priors[key] = tuple(  # type: ignore[assignment]
                BenchPrior.of(entry, s, species[s], epsilon=BELIEF_EPSILON, temperature=1.0)
                for s in (0, 1)
            )
        return self._priors[key]

    def spreads(
        self, game: Game, point: Point, pos: Position, settings: Settings, notes: list[str]
    ) -> dict[int, list] | None:
        """Each side's completions as the belief holds them, or None to read open."""
        if settings.open_information or game.information == "open" or game.teams is None:
            return None
        reg = self.reg
        prior = self.prior(game)
        scratch = GameRecord(own_team=[], foe_team=[], foe_archetype="analysis")
        shown = [seen_slots(pos, i, point.seen[i]) for i in (0, 1)]
        spreads = {}
        for s in (0, 1):
            try:
                weights = _bench_weights(prior, s, pos, shown[s], scratch, game.leads[s])
            except ValueError as problem:
                notes.append(f"leads: {problem}")
                weights = _bench_weights(prior, s, pos, shown[s], scratch, None)
            spreads[s] = completions(
                reg, pos, s, list(game.teams[s].sets), seen=shown[s], weights=weights
            )
        notes.extend(scratch.unmodelled)
        return _believed(spreads, (settings.bench_drop, settings.bench_drop))

    def run(
        self,
        game: Game,
        point: Point,
        side: int | None = None,
        *,
        settings: Settings | None = None,
        stop: threading.Event | None = None,
        listener: Callable[[str, Any], None] | None = None,
        run: int = 0,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        limits: Limits | None = None,
        status: Callable[[Session, Reading, str], None] | None = None,
        on_session: Callable[[Session], None] | None = None,
        tag: dict[str, Any] | None = None,
    ) -> Result:
        """Reads ``point`` from ``side`` (the game's default when None) until ``stop`` is
        set -- by the caller, after ``max_steps`` steps, after ``max_seconds``, or by the
        memory watch when ``limits`` is given -- or the deepening runs out of cells worth
        a step. ``listener`` hears what the IKA-332 page draws (``sheets``, ``board``,
        ``analysis``, ``think``, ``step``, ``answer``); ``status`` is called with the
        session and each memory reading (and a state) while it runs; ``on_session`` with
        the session as soon as it exists (to stop it from elsewhere); ``tag`` goes into the
        ``analysis`` events as it is (which source, game and decision the page picked)."""
        settings = settings or self.settings
        reg = self.reg
        me = game.side if side is None else side
        you = 1 - me
        stop = stop or threading.Event()
        started = time.perf_counter()
        pos = point.pos()
        notes: list[str] = [game.note] if game.note else []
        spreads = self.spreads(game, point, pos, settings, notes)
        exact = spreads is None or all(
            len(items) == 1 and items[0].exact for items in spreads.values()
        )
        classes = 0 if spreads is None else len(spreads[you])
        names = side_names(me)
        emit = listener or (lambda kind, payload: None)
        if game.teams is not None:
            emit("sheets", {
                "analysis": True, "agentSide": me, "personSide": you, "agent": self.name,
                "seconds": None, "clock": "analysis", "cores": None, "names": list(names),
                "teams": [sheet_view(list(t.sets), self.loc, reg) for t in game.teams],
                "statNames": {
                    k: (self.loc.stat(k, short=True) if self.loc is not None else k)
                    for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
                },
                "statusNames": {
                    k: (self.loc.status(k) if self.loc is not None else k)
                    for k in ("brn", "par", "psn", "tox", "slp", "frz")
                },
            })
        emit("board", board_view(reg, pos, me, point.seen[you], self.loc, names=names))
        budget = Budget.matrix()
        wider: dict[int, tuple[list[SideAction], list[SideAction]]] = {}
        oracle = settings.oracle
        ours, theirs = _menus(
            reg, pos, (settings.width, settings.width), self.leaf, budget,
            settings.rank_by_leaf, None, spreads, rank_fill=settings.rank_fill,
            wide=[oracle] if oracle is not None else [], wider=wider,
        )
        outside = wider.get(oracle) if oracle is not None else None
        menu_ms = (time.perf_counter() - started) * 1000.0
        recorder = None
        if listener is not None:
            recorder = Recorder(
                Reader(reg, pos, me, loc=self.loc, names=names),
                lambda snapshot: emit("step", snapshot),
                decision=run, turn=pos.turn, started=started, interval_ms=settings.interval_ms,
            )
        session = Session(
            stop, guard=settings.levels, recorder=recorder, max_steps=max_steps,
            guard_every_ms=settings.guard_every_ms,
        )
        # The menus the root holds so far, and the widths it was widened to (IKA-354).
        held = ([a.to_choice() for a in ours], [a.to_choice() for a in theirs])
        widened_to: list[int] = []

        def build(width: int) -> tuple[list[SideAction], list[SideAction]]:
            return _menus(
                reg, pos, (width, width), self.leaf, budget, settings.rank_by_leaf, None,
                spreads, rank_fill=settings.rank_fill,
            )

        def grown(width: int, menus: tuple[list[SideAction], list[SideAction]]) -> None:
            for have, menu in zip(held, menus, strict=True):
                have.extend(c for c in (a.to_choice() for a in menu) if c not in have)
            widened_to.append(width)
            state["width"] = width
            state["menu"] = [len(held[0]), len(held[1])]
            emit("analysis", {**state, "grown": True})

        session.widen = Widen(settings.width, build, grown)
        session.key = {
            **{k: v for k, v in (tag or {}).items() if k in ("source", "game", "decision")},
            "side": me, "guard": settings.levels,
        }
        if on_session is not None:
            on_session(session)
        state = {
            "played": _played(reg, pos, point, me, self.loc),
            "state": "running", "run": run, "turn": pos.turn, "side": me,
            "width": settings.width, "oracle": settings.oracle_label(), "guard": settings.levels,
            "exact": exact, "classCount": classes, "menu": [len(ours), len(theirs)],
            "maxSteps": max_steps, "maxSeconds": max_seconds,
            "notes": notes_ja(self.loc, notes), "notesRaw": list(notes),
            **(tag or {}),
        }
        emit("analysis", state)
        emit("think", {
            "decision": run, "turn": pos.turn, "analysis": True, "clock": "analysis",
            "seconds": None, "menuMs": menu_ms, "classCount": classes, "exact": exact,
            "plan": {"width": settings.width, "oracle": settings.oracle_label(),
                     "guard": settings.levels, "budgetMs": 0, "deepenMs": 0,
                     "predictedMs": 0, "nodeCells": len(ours) * len(theirs) * max(classes, 1)},
        })
        watch = None
        if limits is not None or status is not None or max_seconds is not None:
            def tick(reading: Reading) -> None:
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    session.halt("time")
                if status is not None:
                    status(session, reading, "running")

            watch = MemoryWatch(
                limits or Limits(rss_gb=0, free_gb=0, gpu_gb=0), session.halt, tick=tick,
                every=0.5 if max_seconds is None else min(0.5, max(0.01, max_seconds / 10)),
            ).start()
        rungs: list[dict[str, Any]] = []
        try:
            if settings.ladder is not None:
                from .ladder import parse_ladder

                def on_rung(rung: Any) -> None:  # noqa: ANN401 - ladder.Rung
                    # Each completed stage is the answer so far (IKA-364).
                    rungs.append(rung.to_json())
                    session.steps = len(rungs)
                    emit("analysis", {**state, "rung": rungs[-1], "steps": len(rungs)})

                solved = solve_move(
                    reg, pos, me, ours, theirs, spreads, self.leaf, budget=budget, exact=exact,
                    ladder={"stages": parse_ladder(settings.ladder), "budget_ms": None,
                            "clock": "wall", "start_ms": 0.0, "began": started, "stop": stop,
                            "on_rung": on_rung},
                )
            else:
                solved = solve_move(
                    reg, pos, me, ours, theirs, spreads, self.leaf, budget=budget, exact=exact,
                    cells=ENDLESS, cost=StopCost(stop, started), levels=settings.levels,
                    outside=outside, progress=session, discount=settings.discount,
                    grow=session.widen,
                )
        except EquilibriumError as problem:
            session.halt("error", str(problem))
            raise
        finally:
            if watch is not None:
                watch.close()
            #: A width asked for too late for this read: the caller reads it again.
            session.left = session.widen.close()
        reason = session.reason
        if not reason:
            report = solved.deepened
            reason = ("exhausted" if (report is not None and report.stop == "exhausted")
                      or (solved.ladder is not None and solved.ladder.stopped == "done")
                      else "person")
        mine = solved.ours if me == 0 else solved.theirs
        other = solved.theirs if me == 0 else solved.ours
        seconds = time.perf_counter() - started
        result = Result(
            side=me, turn=pos.turn, ours=[a.to_choice() for a in mine],
            strategy=solved.strategy, theirs=[a.to_choice() for a in other],
            model=list(solved.model), value0=solved.value,
            value=solved.value if me == 0 else 1.0 - solved.value,
            steps=session.steps, seconds=seconds, stop=reason, guard=settings.levels,
            guard_lines=session.guard_count, nodes=session.nodes, exact=exact,
            classes=classes,
            deepened=(solved.deepened.to_json() if solved.deepened is not None
                      else {"ladder": solved.ladder.to_json()} if solved.ladder is not None
                      else None),
            notes=notes + sorted(solved.unmodelled),
            memory=None if watch is None else watch.peak,
            width=int(state["width"]), widened_to=widened_to,
        )
        if status is not None:
            status(session, Reading() if watch is None else watch.last, "done:" + reason)
        emit("answer", {
            "decision": run, "turn": pos.turn, "actions": result.ours,
            "strategy": result.strategy, "value0": result.value0, "seconds": seconds,
            "deepened": result.deepened, "steps": session.steps,
            "sent": None if recorder is None else recorder.sent,
        })
        emit("analysis", {
            **state, "state": "done", "stop": reason, "stopText": STOPS.get(reason, reason),
            "why": session.why, "steps": session.steps, "seconds": seconds,
            "guardLines": session.guard_count, "nodes": session.nodes,
            # The page reads them in Japanese (IKA-349); `--out` keeps the port's own.
            "notes": notes_ja(self.loc, result.notes), "notesRaw": list(result.notes),
        })
        return result


# ----------------------------------------------------------------------------- the page


@dataclass
class Source:
    """Where the page's games come from: a record file, or a point file of a game being
    played (read again each time it is asked)."""

    name: str
    path: Path
    current: bool = False
    limit: int | None = None
    games: list[Game] = field(default_factory=list)

    def load(self, reg: Regulation, pools: dict[str, Any]) -> list[Game]:
        if self.current:
            self.games = [load_point(reg, self.path)] if self.path.exists() else []
        else:
            self.games = load_games(self.path, pools=pools, limit=self.limit)
        return self.games


#: The status frame's states (`liveview.Wire.status`).
STATES = ("idle", "running", "done")


class Service:
    """The analysis page's loop: the catalogue of games, the person's commands (analyse a
    decision from a side at a width and guard; stop; read the sources again), one read at a
    time on the calling thread, the status frames while it runs.

    ``server`` is a `liveview.LiveServer` built with ``on_command=service.command``."""

    def __init__(
        self,
        analyzer: Analyzer,
        server: Any,  # noqa: ANN401 - liveview.LiveServer
        sources: Sequence[Source],
        *,
        limits: Limits | None = None,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        threads: int = 1,
        out: Path | None = None,
        pools: dict[str, Any] | None = None,
        say: Callable[[str], None] = lambda text: None,
    ) -> None:
        self.analyzer = analyzer
        self.server = server
        self.sources = list(sources)
        self.limits = limits or Limits()
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.threads = threads
        self.out = out
        self.pools = {} if pools is None else pools
        self.say = say
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self.session: Session | None = None
        self.runs = 0
        self.results: list[Result] = []

    # -- the page's side (the socket's thread)
    def command(self, message: dict[str, Any]) -> None:
        kind = message.get("cmd")
        if kind == "stop":
            if self.session is not None:
                self.session.halt("person")
        elif kind == "analyze":
            session = self.session
            if session is not None and self._widens(session, message):
                return
            if session is not None:
                session.halt("new")
            self.commands.put(message)
        elif kind in ("refresh", "quit"):
            if kind == "quit" and self.session is not None:
                self.session.halt("person")
            self.commands.put(message)

    def _widens(self, session: Session, message: dict[str, Any]) -> bool:
        """Whether ``message`` asks the running read for a wider width and nothing else
        (the same position, side and guard), and the read took the request (IKA-354)."""
        widen, key = session.widen, session.key
        if widen is None or not key or not message.get("width"):
            return False
        guard = int(message["guard"]) if message.get("guard") else self.analyzer.settings.levels
        same = (
            all(int(message.get(k, 0)) == key.get(k) for k in ("source", "game", "decision"))
            and message.get("side") is not None and int(message["side"]) == key["side"]
            and max(1, guard) == key["guard"]
        )
        if not same or int(message["width"]) <= widen.width:
            return False
        return widen.request(int(message["width"]))

    # -- the loop's side
    def catalogue(self) -> dict[str, Any]:
        reg = self.analyzer.reg
        out = []
        for index, source in enumerate(self.sources):
            try:
                games = source.load(reg, self.pools)
                problem = ""
            except (OSError, ValueError, KeyError) as error:
                games, problem = [], str(error)
            out.append({
                "id": index, "name": source.name, "current": source.current, "problem": problem,
                "games": [
                    {
                        "index": g, "label": game.label, "side": game.side,
                        "open": game.teams is None or game.information == "open",
                        "note": game.note,
                        "decisions": [
                            {"index": k, "decision": p.decision, "turn": p.turn,
                             "value": p.value}
                            for k, p in enumerate(game.points)
                        ],
                    }
                    for g, game in enumerate(games)
                ],
            })
        settings = self.analyzer.settings
        return {
            "sources": out,
            "settings": {
                "width": settings.width, "oracle": settings.oracle_label(),
                "guard": settings.levels, "threads": self.threads,
                "agent": self.analyzer.name, "maxSteps": self.max_steps,
                "maxSeconds": self.max_seconds,
                "limits": {"rssGb": self.limits.rss_gb, "freeGb": self.limits.free_gb,
                           "gpuGb": self.limits.gpu_gb},
            },
        }

    def publish_catalogue(self) -> None:
        self.server.listener("catalogue", self.catalogue())

    def status(self, session: Session, reading: Reading, state: str) -> None:
        self.server.status(self.server.wire.status(
            state=STATES.index(state.split(":")[0]) if state.split(":")[0] in STATES else 0,
            elapsed=session.elapsed, steps=session.steps, guard_lines=session.guard_count,
            nodes=session.nodes, guard=session.guard, depth=session.depth,
            threads=self.threads, rss=reading.rss_gb, free=reading.free_gb,
            gpu=reading.gpu_gb, gpu_total=reading.gpu_total_gb,
            rss_limit=self.limits.rss_gb, free_floor=self.limits.free_gb,
            gpu_limit=self.limits.gpu_gb, warn=reading.over(self.limits, self.limits.warn) != "",
        ))

    def analyze(self, message: dict[str, Any]) -> Result | None:
        try:
            source = self.sources[int(message.get("source", 0))]
            games = (
                source.load(self.analyzer.reg, self.pools)
                if source.current or not source.games else source.games
            )
            game = games[int(message.get("game", 0))]
            point = game.points[int(message.get("decision", 0))]
        except (IndexError, ValueError, OSError, KeyError) as problem:
            self.server.listener("error", {"message": f"その局面は読めません: {problem}"})
            return None
        base = self.analyzer.settings
        settings = Settings(**{**base.__dict__})
        if message.get("width"):
            settings.width = max(1, int(message["width"]))
        if message.get("guard"):
            settings.levels = max(1, int(message["guard"]))
        side = message.get("side")
        self.server.begin_run()
        run = self.runs
        self.runs += 1

        kept: list[Session] = []

        def keep(session: Session) -> None:
            self.session = session
            kept.append(session)

        self.say(f"analysis {run}: {source.name} / {game.label} / turn {point.turn}")
        try:
            result = self.analyzer.run(
                game, point, None if side is None else int(side), settings=settings,
                listener=self.server.listener, run=run, max_steps=self.max_steps,
                max_seconds=self.max_seconds, limits=self.limits, status=self.status,
                on_session=keep,
                tag={"source": self.sources.index(source), "game": games.index(game),
                     "decision": game.points.index(point)},
            )
        except EquilibriumError as problem:
            self.server.listener("error", {"message": f"均衡が解けませんでした: {problem}"})
            return None
        finally:
            self.session = None
        left = kept[0].left if kept else None
        if left is not None:
            # A wider width asked for as the read ended: read it from the start.
            self.commands.put({**message, "width": left})
        self.results.append(result)
        self.say(
            f"analysis {run}: {result.stop} after {result.steps} steps, {result.seconds:.1f} s, "
            f"value {result.value:.4f}, lines at the guard {result.guard_lines}, nodes {result.nodes}"
        )
        if self.out is not None:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            with self.out.open("ab") as handle:
                handle.write((json.dumps({
                    "source": source.name, "game": game.label, "decision": point.decision,
                    **result.to_json(),
                }, ensure_ascii=False) + "\n").encode("utf-8"))
        return result

    def serve(self, first: dict[str, Any] | None = None, *, once: bool = False) -> None:
        """Runs commands until ``quit`` (or, with ``once``, after the first read)."""
        self.publish_catalogue()
        if first is not None:
            self.commands.put(first)
        while True:
            message = self.commands.get()
            kind = message.get("cmd")
            if kind == "quit":
                return
            if kind == "refresh":
                self.publish_catalogue()
                continue
            if kind == "analyze":
                # A newer request waiting behind this one replaces it.
                while True:
                    try:
                        later = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if later.get("cmd") == "quit":
                        return
                    if later.get("cmd") == "analyze":
                        message = later
                self.analyze(message)
                if once:
                    return


__all__ = [
    "DEFAULT_ORACLE",
    "DEFAULT_WIDTH",
    "ENDLESS",
    "STOPS",
    "Analyzer",
    "Game",
    "Limits",
    "MemoryWatch",
    "Point",
    "Reading",
    "Result",
    "Service",
    "Session",
    "Settings",
    "Source",
    "StopCost",
    "Widen",
    "game_from_record",
    "gpu_used_gb",
    "guard_lines",
    "host_available_gb",
    "load_games",
    "load_point",
    "point_json",
    "process_rss_gb",
    "side_names",
    "tree_rss_gb",
    "write_point",
]
