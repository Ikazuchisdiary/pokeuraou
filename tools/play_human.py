"""Play one game against the agent at the terminal, on a clock (IKA-330, stage 1).

    uv run python tools/play_human.py --seconds 45 --cores 8
    uv run python tools/play_human.py --seconds 15 --human-team 3 --agent-team 17 --human-side 1
    uv run python tools/play_human.py --seconds 5 --person random --seed 4     # a stand-in person
    uv run python tools/play_human.py --person record:data/human/games.jsonl --clock count

The game, the view and the clock are `pokeuraou.humanplay`'s (its docstring says what the
person is shown and how the agent spends its seconds). This file loads the pieces:

* the teams: two of the M-C pool's (``--human-team`` / ``--agent-team``, by index or id;
  both drawn from the seed when neither is given) or a roster file each
  (``--human-team-file`` / ``--agent-team-file``, `teams.load_roster`'s shape);
* the leaf: the M-C ensemble (`DEFAULT_VALUE`) when it is there, ``--value`` to name
  another, ``--hp-share`` for none -- said at the start either way;
* the menus: ``q-nocover`` when a Q is there (``--q-model``, default `DEFAULT_Q`), the
  default fill otherwise, with a note; ``--rank-fill`` overrides. Unlike generation and the
  board, which stop without the Q (IKA-338), a person's game falls back and says so, as it
  does for a missing leaf: the note is printed at the start and the record names the fill;
* the cores: ``--cores`` prices the budget rule and spreads a move over that many threads
  (`humanplay.use_threads`: the port's cells, the deepening's cells expanded ahead, a big
  game's two LPs at once -- IKA-32); ``--threads`` sets the threads alone.

The person: ``terminal`` (you), ``first`` / ``random`` (stand-ins, for smoke runs),
``script:<file>`` (one answer per line: the selection as party numbers, then each choice
as its number in the legal list or its choice string), or ``record:<games.jsonl>[@n]``
(the inputs of the n-th game in a record: that game again, byte for byte on the count
clock).

Each game appends one line to ``--out`` and its clock to ``<out>.clock.jsonl``.

``--current-out`` writes the move in hand before each of the agent's moves, for the
analysis mode to read while the game goes on (``tools/analyze.py --current``, IKA-337).

The screen (IKA-332): ``--view`` serves a page at http://127.0.0.1:8332/ (``--view-port``)
that shows the board, the agent's answer as it forms (its mixture, the value over time, the
cells and levels read, the opponent's mixture and the bench belief, the principal variation
as a tree) and takes the person's input with ``--person web``. ``--live-out`` keeps the
page's frames (`liveview.FileSink`); ``tools/live_view.py`` shows such a file again.

    uv run python tools/play_human.py --seconds 15 --cores 8 --person web --view
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()  # before anything loads torch (IKA-360)

from pokeuraou import analysis, humanplay, liveview, qrank  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.deepen import MAX_LEVELS  # noqa: E402
from pokeuraou.hidden import DEFAULT_BENCH_DROP, parse_bench_drop  # noqa: E402
from pokeuraou.names import localiser  # noqa: E402
from pokeuraou.pool import draw_pair, load_pool  # noqa: E402
from pokeuraou.regulation import repo_root  # noqa: E402
from pokeuraou.search import DEFAULT_RANK_FILL, SHIPPED_RANK_FILL, parse_rank_fill  # noqa: E402
from pokeuraou.selfplay import MAX_TURNS  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402

#: The M-C leaf that ships (CLAUDE.md): the two-model ensemble. gen-4 with the state and
#: bind inputs and the auxiliary targets since IKA-430 (value-mc4stx2 from IKA-427,
#: value-mc4x2 from IKA-409, value-mc3x2 from IKA-402/405,
#: value-mc2x2 from IKA-400, value-mc1x2 from IKA-346, value-mc0x2 before it).
DEFAULT_VALUE = ("data/models/value-mc6.pt", "data/models/value-mc6-s1.pt")
#: Where the Q that q-nocover ranks by is looked for when --q-model is not given: the same
#: file generation and the board rank by (IKA-338). Without it the menus fall back, with a
#: note -- here, not in generation or on the board, which stop.
DEFAULT_Q = qrank.DEFAULT_Q
#: The fill the user chose on 9/26 for boards and human play (IKA-274 stage 2), and M-C
#: generation's since IKA-338.
Q_FILL = SHIPPED_RANK_FILL
#: What a person's game ranks by when it has no Q to read -- no file, no leaf's encoder
#: (hp-share), or a Q of another vocabulary than the leaf's -- with a note (IKA-330;
#: kept by the user's decision of 9/28, IKA-341). Named here: no library default.
NO_Q_FILL = DEFAULT_RANK_FILL


def install_menus(
    named: str | None, q_path: Path, encoder, evaluate, address: str | None,  # noqa: ANN001
    device: str | None, say: Callable[[str], None],
) -> tuple[str, list[str]]:
    """The fill a person's opponent (or a reading) ranks its menus by, and the Q files it
    installed (`qrank.install`).

    A named fill is played or the tool stops (a q fill with no leaf or no Q). Unnamed,
    `Q_FILL` by the Q at ``q_path`` when there is one this leaf's encoder reads, else
    `NO_Q_FILL` with a note -- as for a missing leaf -- where generation and the board stop
    (IKA-338). A Q of another vocabulary (`qrank.QVocabularyError`, e.g. a pool on another
    regulation than the Q's) is one it cannot read: a note, not a stop (IKA-341).
    """
    fill = named
    if fill is None:
        fill = Q_FILL if q_path.exists() and encoder is not None else NO_Q_FILL
        if fill != Q_FILL:
            why = "no leaf to read its encoding" if encoder is None else f"no Q at {q_path}"
            say(f"note: menus ranked by {fill}, not {Q_FILL} ({why}; --q-model names one)")
    parse_rank_fill(fill)
    if not qrank.is_q(fill):
        return fill, []
    if encoder is None:
        raise SystemExit(f"{fill} needs a leaf's encoder (not --hp-share)")
    if not q_path.exists():
        raise SystemExit(f"{fill} needs a Q: no file at {q_path}")
    try:
        model = (humanplay.served_q(address, q_path, encoder) if address and evaluate is not None
                 else qrank.LocalQ(q_path, encoder, device=device or "cpu"))
    except qrank.QVocabularyError as problem:
        if named is not None:
            raise
        say(f"note: menus ranked by {NO_Q_FILL}, not {Q_FILL} ({problem}; --q-model names one)")
        return NO_Q_FILL, []
    qrank.install(model)
    return fill, model.describe()


def leaf_name(files: list[Path]) -> str:
    """A model stem, `-sN` dropped, `xN` for an ensemble (`pool_match.leaf_name`)."""
    stem = re.sub(r"-s\d+$", "", Path(files[0]).stem)
    return stem if len(files) == 1 else f"{stem}x{len(files)}"


def _team(pool, key: str):  # noqa: ANN001, ANN202
    if key.isdigit():
        return pool.teams[int(key)]
    for team in pool.teams:
        if team.id == key:
            return team
    raise SystemExit(f"no team {key!r} in {pool.id}")


def _oracle_width(spec: str | None) -> int | None:
    """``--oracle``: s<W> or sall as a width (`deepen.ALL_ACTIONS` for every action)."""
    from pokeuraou.deepen import ALL_ACTIONS

    if spec is None or spec == "none":
        return None
    if spec == "sall":
        return ALL_ACTIONS
    if spec.startswith("s") and spec[1:].isdigit() and int(spec[1:]) > 0:
        return int(spec[1:])
    raise SystemExit(f"--oracle is s<W>, sall or none, not {spec!r}")


def _person(spec: str, reg, loc, seed: int, server=None):  # noqa: ANN001, ANN202
    if spec == "web":
        if server is None:
            raise SystemExit("--person web answers from the page: give --view")
        return liveview.WebPerson(server)
    if spec == "terminal":
        return humanplay.TerminalPerson(reg, loc)
    if spec in ("first", "random"):
        return humanplay.PolicyPerson(spec, seed)
    if spec.startswith("script:"):
        return humanplay.ScriptPerson.from_file(Path(spec[len("script:"):]))
    if spec.startswith("record:"):
        path, _, line = spec[len("record:"):].partition("@")
        return humanplay.ScriptPerson.from_record(Path(path), int(line or 0))
    raise SystemExit(
        f"--person is terminal, web, first, random, script:<file> or record:<file>[@n], not {spec!r}"
    )


def resolve_selection(args: argparse.Namespace, evaluate: Any) -> tuple[str | None, float | None]:  # noqa: ANN401
    """The reading a person's selection gets and its wall-clock seconds (IKA-392). Unset: the
    default reading for `humanplay.PLAY_SELECTION_SECONDS` on the wall clock with a leaf, else
    the leaf's solve (a count-clock game replays, a game without a leaf has nothing to read
    with). ``none``: the leaf's solve. A reading named alone on the count clock reads its
    stages, not seconds."""
    from pokeuraou import selection_deep

    given = args.selection_reading
    if given == "none" or (given is None and (args.clock != "wall" or evaluate is None)):
        return None, None
    if given is not None and evaluate is None:
        raise SystemExit("--selection-reading reads cells by a leaf: it needs one")
    reading = humanplay.PLAY_SELECTION_READING if given is None else given
    selection_deep.parse_reading(reading)
    seconds = args.selection_seconds
    if seconds is None and args.clock == "wall":
        seconds = humanplay.PLAY_SELECTION_SECONDS
    return reading, seconds


def resolve_cores(threads: int | None, cores: int | None, clock: str) -> tuple[int, int]:
    """The threads a move spreads over and the cores the budget rule prices it at (IKA-343).

    ``--cores`` alone sets both, as before. Neither: `humanplay.default_threads` threads,
    priced at that many cores on the wall clock and at 1 on the count clock, whose prices
    are measured for 1 core only -- so a count-clock game is the same game at any
    ``--threads``."""
    if threads is None:
        threads = cores if cores is not None else humanplay.default_threads()
    if cores is None:
        cores = threads if clock == "wall" else 1
    return threads, cores


def memory_watch(
    limits: analysis.Limits, halt: threading.Event, say: Callable[[str], None],
    emit: Callable[[str, dict], None] | None = None, *, start: bool = True,
    **readers: Callable,
) -> analysis.MemoryWatch:
    """IKA-337's memory watch over a person's game (IKA-343, IKA-355): twice a second it
    reads this process with its workers, the host's free memory and the card, and reads
    them by `analysis.Reading.brake`:

    * this process past ``rss_gb``, the card past ``gpu_gb``, the host under
      ``hard_free_gb``, or under ``free_gb`` because this move's deepening grew this
      process: it sets ``halt``, which makes the move in hand stop deepening at its next
      step and play the answer it has (`humanplay.HaltingCost`);
    * the host under ``free_gb`` because other work took the memory: it reads on and says so.
      Before IKA-355 this stopped every move too, and on a busy machine the agent played at
      0.08-0.19 of its budget (IKA-343 §1.2).

    A `humanplay.MemoryBrake` ``halt`` tells it when each move starts deepening: the
    growth is measured from there, and the brake is lifted and read again. Within a move the
    brake is lifted only once every limit is back under 90% (``limits.warn``).

    Each change of state (``stop``, ``low``, ``ok``) is said once on the terminal and, with
    ``emit``, sent to the page as a ``memory`` event. ``readers`` replace the watch's
    readings (``read_host``, ``read_rss``, ``read_gpu``: the tests'); ``start=False``
    leaves the thread off (the tests call ``look``)."""
    state = {"base": math.nan, "shown": ""}

    def show(kind: str, why: str) -> None:
        if kind == state["shown"]:
            return
        state["shown"] = kind
        if kind == "stop":
            say(f"note: memory near its limit, the agent stops deepening: {why}")
        elif kind == "low":
            say(f"note: host memory is low from other work; the agent reads on (it stops "
                f"under {limits.hard_free_gb:g} GB free): {why}")
        else:
            say("note: memory back under its limits")
        if emit is not None:
            emit("memory", {"state": kind or "ok", "why": why,
                            "freeGb": limits.free_gb, "hardFreeGb": limits.hard_free_gb})

    def judge(reading: analysis.Reading) -> str:
        kind, why = reading.brake(limits, state["base"])
        if kind == "low" and not halt.is_set():
            show("low", why)
        return why if kind == "stop" else ""

    def stop(reason: str, why: str) -> None:  # noqa: ARG001
        if not halt.is_set():
            halt.why = why  # type: ignore[attr-defined]
        halt.set()
        show("stop", why)

    def tick(reading: analysis.Reading) -> None:
        kind, _why = reading.brake(limits, state["base"], limits.warn)
        if kind:
            return
        halt.clear()
        show("", "")

    watch = analysis.MemoryWatch(limits, stop, tick=tick, judge=judge, **readers)
    rss = watch.read_rss()
    state["base"] = math.nan if rss is None else rss

    def begin() -> None:
        with watch.lock:
            got = watch.read_rss()
            state["base"] = math.nan if got is None else got
            halt.clear()
            watch.look()

    if isinstance(halt, humanplay.MemoryBrake):
        halt.on_begin = begin
    return watch.start() if start else watch


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="regmc-matchupweb")
    ap.add_argument("--human-team", default=None, help="the person's pool team, by index or id")
    ap.add_argument("--agent-team", default=None, help="the agent's pool team, by index or id")
    ap.add_argument("--human-team-file", type=Path, default=None, help="a roster file instead")
    ap.add_argument("--agent-team-file", type=Path, default=None, help="a roster file instead")
    ap.add_argument("--human-side", type=int, default=0, choices=(0, 1))
    ap.add_argument("--seconds", type=float, default=45.0, help="the agent's budget per move")
    ap.add_argument("--cores", type=int, default=None,
                    help="the cores the budget rule prices a move at and, unless --threads says "
                    "otherwise, the threads. Default: the threads on the wall clock, 1 on the "
                    "count clock (the only count priced, deepen.COSTS)")
    ap.add_argument("--threads", type=int, default=None,
                    help="threads a move spreads over (the port's cells, the deepening's cells "
                    "expanded ahead, a big game's two LPs at once). Default: --cores when given, "
                    f"else {humanplay.PLAY_THREADS} (IKA-32 stage 2; fewer on a smaller machine). "
                    "They change no move: a count-clock game is the same game at any number; "
                    "--threads 1 turns them off")
    ap.add_argument("--clock", default="wall", choices=humanplay.CLOCKS,
                    help="wall: stop the deepening at the budget by the clock (default); "
                    "count: spend it at measured prices, reproducible by seed")
    ap.add_argument("--width-only", action="store_true",
                    help="no deepening: the width rule alone (a baseline; how NODE_TIME is measured)")
    ap.add_argument("--max-levels", type=int, default=humanplay.PLAY_MAX_LEVELS,
                    help=f"the deepening's depth guard (IKA-307; default {humanplay.PLAY_MAX_LEVELS}, "
                    "humanplay.PLAY_MAX_LEVELS, IKA-342). Set, each move's record says why the "
                    f"deepening and its lines stopped; 0: deepen.MAX_LEVELS ({MAX_LEVELS}) unrecorded, "
                    "as before IKA-343")
    ap.add_argument("--oracle", default="sall",
                    help="the root's swap oracle while deepening: sall (every legal action, the "
                    "default: IKA-307's allocation), s<W> (the rest of a width-W menu) or none")
    ap.add_argument("--ladder", default=None,
                    help="read each move by a ladder of stages instead of the deepening (IKA-367; "
                    "a name in ladder.LADDERS such as L5, or stages joined by +), the answer the "
                    "last stage completed in the budget (L6 fills the budget on the wall clock "
                    "and its stages do not run out, IKA-376). With --threads N its cells are read by "
                    "N-1 worker processes (IKA-364; with a server, each asks it; without, at "
                    f"most {humanplay.LADDER_LOCAL_WORKERS_MAX} load the leaf and the Q). "
                    "Default: none (the deepening)")
    ap.add_argument("--selection-reading", default=None,
                    help="how the AI reads the selection (IKA-392, selection_deep): `none` (the "
                    "value function's one estimate of each cell), `default`, or `stage=d2r4b3k8,"
                    "rects=8-16,confirm=2,shift=add`. Unset: `default` on the wall clock with a "
                    "leaf (the board that decided it: +23.2 Elo over `none`), else `none`. With "
                    "--selection-seconds S the reading runs S seconds of wall time (the "
                    "rectangle widens while time is left), else exactly the stages it names")
    ap.add_argument("--selection-seconds", type=float, default=None,
                    help="the wall-clock time of the deeper selection (unset: "
                    f"{humanplay.PLAY_SELECTION_SECONDS:g} s for the default reading)")
    ap.add_argument("--selection-workers", type=int, default=None,
                    help="worker processes reading the selection's cells (default: every logical "
                    "core with an inference server, else at most "
                    f"{humanplay.LADDER_LOCAL_WORKERS_MAX}, each with a leaf of its own)")
    ap.add_argument("--child-q", type=int, default=None,
                    help="the deepening's child menus: each side's k best by the Q (IKA-307), "
                    "instead of narrow's damage-ranked 8")
    ap.add_argument("--ponder", default="on" if humanplay.PLAY_PONDER else "off", choices=("on", "off"),
                    help="on: the agent reads on while you choose (IKA-344) -- you are asked as "
                    "its move starts, and it deepens until you have chosen (its budget at least, "
                    "--ponder-seconds at most), then draws its action; yours is read after. "
                    f"Default {'on' if humanplay.PLAY_PONDER else 'off'} (humanplay.PLAY_PONDER). "
                    "A count-clock game replays from its record (record:<file>) with the stop "
                    "points it wrote")
    ap.add_argument("--ponder-seconds", type=float, default=humanplay.PLAY_PONDER_SECONDS,
                    help="with --ponder on: the longest a move reads, from its start")
    ap.add_argument("--value", type=Path, nargs="+", default=None,
                    help=f"the agent's leaf (one model or an ensemble). Default: {' '.join(DEFAULT_VALUE)}")
    ap.add_argument("--hp-share", action="store_true", help="no leaf: the hp-share proxy")
    ap.add_argument("--leaf-graphs", default="on", choices=("on", "off"),
                    help="score the leaf's small blocks by CUDA-graph replays (humanplay.GraphLeaf, "
                    "the eager answer to the bit; on a card only)")
    ap.add_argument("--rank-fill", default=None,
                    help=f"how the agent's menus are ranked. Default: {Q_FILL} when a Q is there, "
                    f"else {NO_Q_FILL} with a note")
    ap.add_argument("--q-model", type=Path, default=None, help=f"the Q. Default: {DEFAULT_Q}")
    ap.add_argument("--bench-drop", default=DEFAULT_BENCH_DROP)
    ap.add_argument("--person", default="terminal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game-index", type=int, default=0)
    ap.add_argument("--games", type=int, default=1, help="games in a row (game index counts up)")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--out", type=Path, default=Path("data/human/games.jsonl"))
    ap.add_argument("--locale", default="ja")
    ap.add_argument("--device", default=None)
    ap.add_argument("--inference", default=None, metavar="HOST:PORT",
                    help="the machine's inference server to send the forward passes to (IKA-363; "
                    "its arms value and q must be the leaf's and the Q's files). Default: "
                    "POKEURAOU_INFERENCE, else none: the leaf is loaded here. local: here")
    ap.add_argument("--merge", default="off", choices=("on", "off"),
                    help="with a server: share its forward passes with other processes' requests "
                    "(the merged road; a value moves in the last places with the timing, and "
                    "IKA-363 measured no speed from it: the processes are bound by the CPU)")
    ap.add_argument("--cuda-memory-gb", type=float, default=humanplay.PLAY_CUDA_MEMORY_GB,
                    help="cap this process's CUDA allocator and each worker's (IKA-334); 0: no cap")
    ap.add_argument("--max-rss-gb", type=float, default=analysis.Limits.rss_gb,
                    help="the memory watch (IKA-337): a move stops deepening before this process "
                    "and its workers hold this much (0: off)")
    ap.add_argument("--min-free-gb", type=float, default=analysis.Limits.free_gb,
                    help="... or before the host's free memory falls under this because the "
                    "move's own reading grew (0: off). Under it from other work, the agent "
                    "reads on and says so (IKA-355)")
    ap.add_argument("--hard-free-gb", type=float, default=analysis.Limits.hard_free_gb,
                    help="... and before the host's free memory falls under this, whatever the "
                    "cause (0: off)")
    ap.add_argument("--max-gpu-gb", type=float, default=analysis.Limits.gpu_gb,
                    help="... or before the card holds this much, all processes (0: off)")
    ap.add_argument("--quiet", action="store_true", help="no board on the terminal (stand-in persons)")
    ap.add_argument("--view", action="store_true",
                    help="serve the page that shows the game and the agent's reading (IKA-332)")
    ap.add_argument("--view-host", default="127.0.0.1")
    ap.add_argument("--view-port", type=int, default=8332)
    ap.add_argument("--open-browser", action="store_true",
                    help="with --view: open the page in the browser once it is served")
    ap.add_argument("--live-out", type=Path, default=None,
                    help="keep the page's frames in this file (tools/live_view.py shows it again)")
    ap.add_argument("--sprite-url", default=None,
                    help="the page's images, {id} = Showdown's sprite id (default: Showdown's server; "
                    "\"\" for none, name cards)")
    ap.add_argument("--analysis-url", default=None,
                    help="the analysis page's address, for the page's links to it (default: the "
                    "launcher's port 8337 on the same host)")
    ap.add_argument("--interval-ms", type=float, default=100.0,
                    help="the least time between two steps sent while the agent thinks")
    ap.add_argument("--current-out", type=Path, default=None,
                    help="before each move of the agent, write the position in hand here for the "
                    "analysis mode (tools/analyze.py --current, IKA-337)")
    args = ap.parse_args(argv)
    if args.person == "web":
        args.view = True
    parse_bench_drop(args.bench_drop)

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    loc = localiser(reg, args.locale)
    say = (lambda text: None) if args.quiet else (lambda text: print(text, file=sys.stderr))

    # The leaf.
    evaluate = None
    name = "hp-share"
    encoder = None
    device = args.device
    values = args.value
    if values is None and not args.hp_share:
        found = [repo_root() / p for p in DEFAULT_VALUE]
        if all(p.exists() for p in found):
            values = found
        else:
            say(f"note: no {DEFAULT_VALUE[0]} here, so the agent plays hp-share (--value names a leaf)")
    address = humanplay.inference_address(args.inference)
    if values and not args.hp_share and address:
        # IKA-363: the forward passes on the machine's server; no CUDA context here.
        evaluate, encoder = humanplay.served_leaf(reg, address, values,
                                                  merge=args.merge == "on")
        name = leaf_name(values)
        say(f"leaf on the inference server {address} (merged road {args.merge})")
    elif values and not args.hp_share:
        address = None
        humanplay.cap_cuda(args.cuda_memory_gb, args.device)
        evaluate, encoder, device = humanplay.load_leaf(
            reg, values, device, graphs=args.leaf_graphs == "on"
        )
        name = leaf_name(values)

    # The menus.
    q_path = args.q_model or (repo_root() / DEFAULT_Q)
    fill, q_files = install_menus(args.rank_fill, q_path, encoder, evaluate, address, device, say)

    threads, cores = resolve_cores(args.threads, args.cores, args.clock)
    if args.ladder is not None:
        # IKA-364: the ladder's cells on the workers; the port's cells and the LPs here.
        humanplay.use_threads(threads)
        spec = humanplay.ladder_spec(
            leaf=evaluate, address=address, merge=args.merge == "on", values=values,
            device=device, graphs=args.leaf_graphs == "on", cuda_memory_gb=args.cuda_memory_gb,
            q_path=q_path if qrank.is_q(fill) else None)
        workers = humanplay.use_ladder_pool(threads, reg, spec)
        say(f"ladder {args.ladder}: cells on {workers} worker process(es)")
    elif address and evaluate is not None:
        humanplay.use_threads(threads, reg, (address, "value", args.merge == "on"),
                              factory=humanplay.served_process_leaf)
    else:
        humanplay.use_threads(
            threads, reg,
            ([str(v) for v in values] if values and not args.hp_share else None,
             str(device or "cpu"), args.leaf_graphs == "on", args.cuda_memory_gb),
        )
    selection_reading, selection_seconds = resolve_selection(args, evaluate)
    if selection_reading is not None:
        from pokeuraou import selection_deep

        readers = args.selection_workers if args.selection_workers is not None else (
            (os.cpu_count() or 2) if address else min(os.cpu_count() or 2,
                                                       humanplay.LADDER_LOCAL_WORKERS_MAX))
        if readers > 1:
            # Made for each selection and closed after it: no idle workers through the moves.
            selection_deep.use_reader(selection_deep.LazyPoolReader(
                reg, readers,
                humanplay.ladder_spec(
                    leaf=evaluate, address=address, merge=args.merge == "on", values=values,
                    device=device, graphs=args.leaf_graphs == "on",
                    cuda_memory_gb=args.cuda_memory_gb,
                    q_path=q_path if qrank.is_q(fill) else None),
                rank_fill=fill))
        say(f"selection read {selection_reading} on {max(readers, 1)} process(es)"
            + (f" for {selection_seconds:g} s" if selection_seconds else " (its stages)"))
    else:
        say("selection: the leaf's one estimate of each cell")
    halt = humanplay.MemoryBrake()
    agent = humanplay.Agent(
        reg=reg, evaluate=evaluate, name=name, seconds=args.seconds, cores=cores,
        clock=args.clock, rank_fill=fill, bench_drop=args.bench_drop,
        width_only=args.width_only, max_levels=args.max_levels or None, child_q=args.child_q,
        oracle=_oracle_width(args.oracle), halt=halt,
        ponder=args.ponder == "on", ponder_seconds=args.ponder_seconds, ladder=args.ladder,
        selection_reading=selection_reading, selection_seconds=selection_seconds,
    )
    if args.child_q is not None and not qrank.is_q(fill):
        raise SystemExit("--child-q ranks the children by the Q: it needs a Q (a q rank fill)")
    say(f"agent: leaf {name} / menus {fill}" + (f" ({', '.join(q_files)})" if q_files else "")
        + f" / {args.seconds:g} s a move on {cores} core(s), {args.clock} clock, {threads} thread(s)"
        + f" / oracle {args.oracle} / guard {args.max_levels or MAX_LEVELS}"
        + (" / width only" if args.width_only else "")
        + (f" / ponder (up to {args.ponder_seconds:g} s)" if args.ponder == "on" else "")
        + " / bench hidden")

    server = None
    if args.view or args.live_out is not None:
        sink = liveview.FileSink(args.live_out) if args.live_out is not None else None
        server = liveview.LiveServer(
            args.view_host, args.view_port if args.view else 0, sink=sink,
            sprite_url=args.sprite_url,
            links={"analysis-url": args.analysis_url} if args.analysis_url else None,
        ).start()
        if args.view:
            print(f"画面: {server.url}", file=sys.stderr)
            if args.open_browser:
                webbrowser.open(server.url)
    watch = memory_watch(
        analysis.Limits(rss_gb=args.max_rss_gb, free_gb=args.min_free_gb, gpu_gb=args.max_gpu_gb,
                        hard_free_gb=args.hard_free_gb),
        halt, lambda text: print(text, file=sys.stderr),
        emit=None if server is None else server.listener,
    )
    person = _person(args.person, reg, loc, args.seed, server)
    for n in range(args.games):
        index = args.game_index + n
        if server is not None:
            # The page's end card says whether another game follows (IKA-349); its links to
            # the analysis page name the game by its index in the record (IKA-356).
            server.listener("session", {"game": n, "games": args.games, "gamesLeft": args.games - n - 1,
                                        "gameIndex": index})
        if args.human_team_file or args.agent_team_file:
            if not (args.human_team_file and args.agent_team_file):
                raise SystemExit("give both --human-team-file and --agent-team-file")
            human, mine = load_roster(args.human_team_file), load_roster(args.agent_team_file)
        elif args.human_team is not None or args.agent_team is not None:
            if args.human_team is None or args.agent_team is None:
                raise SystemExit("give both --human-team and --agent-team")
            human, mine = _team(pool, args.human_team), _team(pool, args.agent_team)
        else:
            import numpy as np

            _k, a, b = draw_pair(np.random.default_rng([args.seed, index]), pool.pairs)
            human, mine = pool.teams[a], pool.teams[b]
        agent_side = 1 - args.human_side
        teams = (mine, human) if agent_side == 0 else (human, mine)
        started = time.perf_counter()
        on_move = None
        if args.current_out is not None:
            def on_move(pos, seen, leads, index=index, you=1 - agent_side, teams=teams):  # noqa: ANN001, ANN202
                analysis.write_point(args.current_out, analysis.point_json(
                    reg, pos, you, seen, leads, teams,
                    label=f"進行中の局 {index}・ターン {pos.turn}",
                ))
        payload, clock, game = humanplay.play(
            agent, person, teams, agent_side=agent_side, seed=args.seed, game_index=index,
            max_turns=args.max_turns, loc=loc,
            out=None if args.quiet or args.person == "web" else sys.stdout,
            listener=None if server is None else server.listener, interval_ms=args.interval_ms,
            on_move=on_move,
        )
        payload["pool"] = {"id": pool.id, "sha256": pool.sha256}
        if q_files:
            payload["qModel"] = q_files
        clock["gameSeconds"] = round(time.perf_counter() - started, 3)
        humanplay.write_line(args.out, payload)
        humanplay.write_line(humanplay.clock_path(args.out), clock)
        outcome = payload["outcome"]
        you = 1 - agent_side
        result = ("引き分け・打ち切り" if outcome is None
                  else "あなたの勝ち" if (outcome > 0.5) == (you == 0) else "あなたの負け")
        moves = [d for d in clock["decisions"] if d["kind"] == "move"]
        ratios = [d["ratio"] for d in moves]
        print(
            f"game {index}: {result} ({payload['endReason']}, {payload['turns']} turns). "
            f"agent moves {len(moves)}, seconds/budget "
            + (f"mean {sum(ratios) / len(ratios):.3f} min {min(ratios):.3f} max {max(ratios):.3f}"
               if ratios else "-")
            + f"; written to {args.out}",
            file=sys.stderr,
        )
        del game
    watch.close()
    if server is not None:
        if server.sink is not None:
            server.sink.close()
        if args.view and not args.quiet:
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input("画面を閉じるには Enter（Ctrl+C）> ")
        server.close()


if __name__ == "__main__":
    main()
