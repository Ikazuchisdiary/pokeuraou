"""IKA-355: a person's game stops deepening for memory only when its own reading is the
cause (`analysis.Reading.brake`, `play_human.memory_watch`, `humanplay.MemoryBrake`).

What these hold:

- **other work's low memory does not stop the agent**: the host under ``free_gb`` while
  this process has not grown in the move reads as ``low`` -- said once on the terminal and
  to the page, the brake left off; before IKA-355 it stopped every move (IKA-343 §1.2);
- **the move's own growth does**: under ``free_gb`` where the host would be at the floor
  or above without this move's growth, the brake is set with the words why;
- **the floor stays**: under ``hard_free_gb`` the brake is set whatever the cause, as are
  the resident and card limits;
- **the growth is the move's**: each move's start of deepening (`MemoryBrake.begin`)
  measures from there and lifts a brake set in the move before;
- **in a game**: readings of other work's low memory change no move (the same record to
  the byte as a game without a watch) and a move's own growth stops every deepening move,
  whose ``answer`` event carries why (the page's "メモリが少ないので読みを止めました").

The memory is not read: the watch's readers are replaced (``read_host``, ``read_rss``,
``read_gpu``).
"""

from __future__ import annotations

import json
import math

import pytest

from pokeuraou import analysis, humanplay

LIMITS = analysis.Limits(rss_gb=10.0, free_gb=2.0, gpu_gb=0, hard_free_gb=0.5)


def _tool(name: str):  # noqa: ANN202
    import importlib
    import sys

    from pokeuraou.regulation import repo_root

    tools = str(repo_root() / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module(name)


class _Memory:
    """Settable readings: the host's free memory and this process's resident memory."""

    def __init__(self, free: float, rss: float) -> None:
        self.free = free
        self.rss = rss

    def readers(self) -> dict:
        return {"read_host": lambda: self.free, "read_rss": lambda: self.rss,
                "read_gpu": lambda: None}


def _watch(memory: _Memory, brake: humanplay.MemoryBrake | None = None):  # noqa: ANN202
    said: list[str] = []
    events: list[tuple[str, dict]] = []
    brake = brake if brake is not None else humanplay.MemoryBrake()
    watch = _tool("play_human").memory_watch(
        LIMITS, brake, said.append, lambda kind, payload: events.append((kind, payload)),
        start=False, **memory.readers(),
    )
    return watch, brake, said, events


def test_the_reading_tells_the_moves_growth_from_other_work() -> None:
    def brake(free: float, rss: float, base: float) -> str:
        return analysis.Reading(rss_gb=rss, free_gb=free).brake(LIMITS, base)[0]

    # Other work took the host to 1.6 GB; this move grew 0.1 GB: read on.
    assert brake(1.6, 4.1, 4.0) == "low"
    # This move grew 0.8 GB and took the host from 2.4 to 1.6: stop.
    assert brake(1.6, 4.8, 4.0) == "stop"
    # At the floor exactly without the growth: the growth is the cause.
    assert brake(1.5, 4.5, 4.0) == "stop"
    # The floor under the floor, whatever the cause.
    assert brake(0.4, 4.0, 4.0) == "stop"
    # The resident limit, whatever the host.
    assert brake(20.0, 10.5, 4.0) == "stop"
    assert brake(3.0, 4.0, 4.0) == ""
    # An unread resident memory (NaN) is no growth.
    assert brake(1.6, math.nan, 4.0) == "low"
    # The analysis mode's reading is unchanged: under the floor stops.
    assert analysis.Reading(rss_gb=4.0, free_gb=1.6).over(LIMITS)


def test_other_works_low_memory_reads_on_and_says_so() -> None:
    memory = _Memory(free=6.0, rss=4.0)
    watch, brake, said, events = _watch(memory)
    brake.begin()
    watch.look()
    assert not brake.is_set() and not said and not events
    # Another job takes the host to 1.6 GB; this process stays where it was.
    memory.free = 1.6
    for _ in range(3):
        watch.look()
    assert not brake.is_set()
    assert len(said) == 1 and "reads on" in said[0]
    assert [e[1]["state"] for e in events] == ["low"]
    assert events[0][0] == "memory" and events[0][1]["hardFreeGb"] == 0.5
    # It grows 0.3 GB in the move: still other work's (1.3 + 0.3 < 2).
    memory.rss, memory.free = 4.3, 1.3
    watch.look()
    assert not brake.is_set()
    # Under the floor under the floor: stop, whatever the cause.
    memory.rss, memory.free = 4.6, 0.45
    watch.look()
    assert brake.is_set() and "底" in brake.why
    assert [e[1]["state"] for e in events] == ["low", "stop"]


def test_the_moves_own_growth_stops_it_and_the_next_move_measures_afresh() -> None:
    memory = _Memory(free=2.5, rss=4.0)
    watch, brake, _said, events = _watch(memory)
    brake.begin()
    # This move grows 0.8 GB and takes the host from 2.5 to 1.7.
    memory.rss, memory.free = 4.8, 1.7
    watch.look()
    assert brake.is_set() and "この手の読みで 0.8 GB" in brake.why
    # Within the move the brake holds.
    watch.look()
    assert brake.is_set()
    # The next move keeps the memory the last one took (not given back), and other work
    # holds the host at 1.7: the growth is measured from its start, so it reads on.
    brake.begin()
    assert not brake.is_set()
    watch.look()
    assert not brake.is_set()
    assert [e[1]["state"] for e in events] == ["stop", "low"]


def test_the_brake_lifts_once_every_limit_is_back_under_ninety_percent() -> None:
    memory = _Memory(free=0.4, rss=4.0)
    watch, brake, _said, events = _watch(memory)
    watch.look()
    assert brake.is_set()
    tick = watch.tick
    # Between the floor and the floor under it: still stopped in this move.
    memory.free = 1.0
    tick(watch.look())
    assert brake.is_set()
    memory.free = 3.0
    tick(watch.look())
    assert not brake.is_set()
    assert [e[1]["state"] for e in events] == ["stop", "ok"]


# ------------------------------------------------------------------------ in a game


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    """test_humanplay's: two variants of our M-B roster in the pool file's shape."""
    import copy

    from pokeuraou.damage import register_mega_stones
    from pokeuraou.pool import load_pool
    from pokeuraou.regulation import repo_root

    base = json.loads(
        (repo_root() / "configs" / "teams" / "rizabanadohido.json").read_text(encoding="utf-8")
    )["team"]
    rotated = copy.deepcopy(base[2:] + base[:2])
    data = {
        "id": "test-pool", "name": "test", "regulation": "gen9championsvgc2026regmb",
        "character": "a test pool",
        "validatedAgainst": {"formatId": "gen9championsvgc2026regmb"},
        "teams": [{"id": f"t{i}", "name": f"team {i}", "team": t}
                  for i, t in enumerate([base, rotated])],
    }
    path = tmp_path_factory.mktemp("pool") / "p.json"
    path.write_bytes(json.dumps(data).encode("utf-8"))
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture(autouse=True)
def _prices(monkeypatch):  # noqa: ANN001, ANN202
    """test_humanplay's fixed price table, so these games do not move when it is re-measured."""
    from pokeuraou.humanplay import NodeTime

    monkeypatch.setattr(humanplay, "NODE_TIME", {
        ("local", 1): NodeTime(fixed_ms=10.0, cell_ms=0.05),
        ("local", 8): NodeTime(fixed_ms=10.0, cell_ms=0.02),
    })


def _game(pool, brake=None, listener=None):  # noqa: ANN001, ANN202
    from pokeuraou.humanplay import Agent, PolicyPerson

    agent = Agent(reg=pool.reg, evaluate=None, name="hp-share", seconds=0.4, cores=1,
                  clock="count", rank_by_leaf=False, rank_fill="refs2", halt=brake)
    return humanplay.play(
        agent, PolicyPerson("random", 7), (pool.teams[0], pool.teams[1]), agent_side=1,
        seed=7, max_turns=3, listener=listener,
    )


def test_in_a_game_other_works_low_memory_changes_no_move_and_own_growth_stops(pool) -> None:  # noqa: ANN001
    plain, plain_clock, _ = _game(pool)
    deep = [row for row in plain_clock["decisions"]
            if row["kind"] == "move" and row.get("deepened", {}).get("expanded", 0) > 0]
    assert deep, "no move deepened; the brake would have nothing to stop"

    # Other work holds the host at 1.6 GB the whole game; this process does not grow.
    memory = _Memory(free=1.6, rss=4.0)
    brake = humanplay.MemoryBrake()
    watch = _tool("play_human").memory_watch(LIMITS, brake, lambda text: None,
                                             **memory.readers())
    try:
        low, low_clock, _ = _game(pool, brake)
    finally:
        watch.close()
    assert json.dumps(low, ensure_ascii=False) == json.dumps(plain, ensure_ascii=False)
    assert not any(row.get("memoryStop") for row in low_clock["decisions"])

    # The move's own growth: 0.6 GB a reading, from 1.9 GB free.
    grow = _Memory(free=1.9, rss=4.0)
    readers = grow.readers()

    def growing() -> float:
        grow.rss += 0.6
        return grow.rss

    readers["read_rss"] = growing
    brake = humanplay.MemoryBrake()
    events: list[tuple[str, dict]] = []
    # The resident limit off: the growing reading would pass it within a few readings.
    unlimited = analysis.Limits(rss_gb=0, free_gb=2.0, gpu_gb=0, hard_free_gb=0.5)
    watch = _tool("play_human").memory_watch(unlimited, brake, lambda text: None, **readers)
    try:
        _, held_clock, _ = _game(pool, brake, lambda kind, payload: events.append((kind, payload)))
    finally:
        watch.close()
    moves = [row for row in held_clock["decisions"]
             if row["kind"] == "move" and row["deepenBudget"] > 0]
    assert moves and all(row.get("memoryStop") for row in moves)
    answers = [p for kind, p in events if kind == "answer"]
    stopped = [p for p in answers if p.get("memoryStop")]
    assert len(stopped) == len(moves)
    assert all("この手の読みで" in p["memoryStop"] for p in stopped)
