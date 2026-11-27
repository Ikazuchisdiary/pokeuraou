"""The analysis mode (検討モード, IKA-337 stage 1): read a position with no time limit, on
the IKA-332 page, until you press Stop.

    uv run python tools/analyze.py --record data/human/games.jsonl --threads 4
    uv run python tools/analyze.py --record data/matches-mc/x/games-worker0.jsonl --max-levels 16
    uv run python tools/analyze.py --current data/human/current.json          # a game being played
    uv run python tools/analyze.py --record g.jsonl --game 0 --decision 2 --no-view --max-steps 500

The page (http://127.0.0.1:8337/) lists the games of each ``--record`` file (a game against a
person, a board or a generation record: one JSON line a game) and, with ``--current``, the
move in hand of a game ``tools/play_human.py --current-out`` is playing. Pick a game, a
turn and the side to read, and it deepens until you press 止める (or ``--max-seconds``,
``--max-steps``, or the memory watch stops it): the mixture, the value over time, the
opponent's mixture and the bench belief, the principal variation, the steps and the lines
that reached the depth guard, as they form.

How it reads (`pokeuraou.analysis`): IKA-307's allocation for a long read -- the menus at
``--width`` (64: every legal action on most turns), then the best-first deepening of the
side's Bayesian root (``h``) with the root's swap oracle (``--oracle``, default sall), the
depth guard at ``--max-levels`` (default `humanplay.PLAY_MAX_LEVELS`, 16, IKA-342). The
cells are expanded ahead on ``--threads`` cores (IKA-32 stage 2; 4 to 8 is the useful range).
The leaf and the menus' Q are ``play_human``'s.

``--max-steps N`` stops after N steps of the deepening: the same position and settings give
the same answer, bit for bit, at any speed and thread count. ``--out`` appends each answer.

Memory: the host's free memory, this process's and its workers', and the card's are read
twice a second; a read stops before ``--max-rss-gb`` / ``--min-free-gb`` / ``--max-gpu-gb``
and the page says which.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()  # before anything loads torch (IKA-360)

from play_human import (  # noqa: E402
    DEFAULT_Q,
    DEFAULT_VALUE,
    NO_Q_FILL,
    Q_FILL,
    _oracle_width,
    install_deepening_q,
    install_menus,
    leaf_name,
)

from pokeuraou import analysis, humanplay, liveview  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.deepen import MAX_LEVELS  # noqa: E402
from pokeuraou.hidden import DEFAULT_BENCH_DROP, parse_bench_drop  # noqa: E402
from pokeuraou.names import localiser  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402
from pokeuraou.regulation import repo_root  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", type=Path, action="append", default=[],
                    help="a record file (one game a line); may be given more than once")
    ap.add_argument("--current", type=Path, default=None,
                    help="the point file of a game being played (play_human --current-out)")
    ap.add_argument("--limit", type=int, default=500, help="games read from each record file")
    ap.add_argument("--pool", default="regmc-matchupweb",
                    help="the pool whose regulation the positions are in (and whose teams they name)")
    ap.add_argument("--game", type=int, default=None, help="start on this game (of the first source)")
    ap.add_argument("--decision", type=int, default=0,
                    help="with --game: the n-th move decision of that game (0-based)")
    ap.add_argument("--side", type=int, default=None, choices=(0, 1),
                    help="with --game: the side to read (default: the person's, else 0)")
    ap.add_argument("--width", type=int, default=analysis.DEFAULT_WIDTH,
                    help="both menus' width (IKA-307: width first; 64 is every legal action on most turns)")
    ap.add_argument("--oracle", default="sall",
                    help="the root's swap oracle while deepening: s<W>, sall (default) or none")
    ap.add_argument("--max-levels", type=int, default=humanplay.PLAY_MAX_LEVELS or MAX_LEVELS,
                    help=f"the deepening's depth guard (default {humanplay.PLAY_MAX_LEVELS or MAX_LEVELS}: "
                    "humanplay.PLAY_MAX_LEVELS, the person's game's, else deepen.MAX_LEVELS; "
                    "IKA-307: long reads meet it)")
    ap.add_argument("--depth-discount", type=float, default=None,
                    help="the deepening's depth discount a ply (a label's d<P> as P / 100, "
                    "IKA-342); default none")
    ap.add_argument("--open", action="store_true",
                    help="read with the opponent's bench open (the recorded position whole)")
    ap.add_argument("--threads", type=int, default=humanplay.default_threads(),
                    help="cores the read spreads over (IKA-32 stage 2: worker processes expand "
                    "cells ahead, a big game's two LPs at once); they change no answer")
    ap.add_argument("--value", type=Path, nargs="+", default=None,
                    help=f"the leaf (one model or an ensemble). Default: {' '.join(DEFAULT_VALUE)}")
    ap.add_argument("--hp-share", action="store_true", help="no leaf: the hp-share proxy")
    ap.add_argument("--rank-fill", default=None,
                    help=f"how the menus are ranked. Default: {Q_FILL} when a Q is there, "
                    f"else {NO_Q_FILL} with a note")
    ap.add_argument("--q-model", type=Path, default=None, help=f"the Q. Default: {DEFAULT_Q}")
    ap.add_argument("--bench-drop", default=DEFAULT_BENCH_DROP)
    ap.add_argument("--inference", default=None, metavar="HOST:PORT",
                    help="the machine's inference server to send the forward passes to (IKA-363; "
                    "its arms value and q must be the leaf's and the Q's files). Default: "
                    "POKEURAOU_INFERENCE, else the Rust server is started (it needs a CUDA card)")
    ap.add_argument("--open-browser", action="store_true",
                    help="open the page in the browser once it is served")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop each read after this many steps of the deepening (reproducible)")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop each read after this long")
    ap.add_argument("--max-rss-gb", type=float, default=analysis.Limits.rss_gb,
                    help="stop before this process and its workers hold this much (0: off)")
    ap.add_argument("--min-free-gb", type=float, default=analysis.Limits.free_gb,
                    help="stop before the host's free memory falls under this (0: off)")
    ap.add_argument("--max-gpu-gb", type=float, default=analysis.Limits.gpu_gb,
                    help="stop before the card holds this much, all processes (0: off)")
    ap.add_argument("--out", type=Path, default=None, help="append each answer here (JSON lines)")
    ap.add_argument("--no-view", action="store_true",
                    help="no page: read --game/--decision once and exit (give --max-steps or --max-seconds)")
    ap.add_argument("--view-host", default="127.0.0.1")
    ap.add_argument("--view-port", type=int, default=8337)
    ap.add_argument("--sprite-url", default=None)
    ap.add_argument("--game-url", default=None,
                    help="the game page's address, for the page's link back (default: the "
                    "launcher's port 8332 on the same host)")
    ap.add_argument("--interval-ms", type=float, default=100.0)
    ap.add_argument("--live-out", type=Path, default=None, help="keep the page's frames in this file")
    ap.add_argument("--locale", default="ja")
    args = ap.parse_args(argv)
    parse_bench_drop(args.bench_drop)
    if not args.record and args.current is None:
        raise SystemExit("give --record (a record file) or --current (a point file)")
    if args.no_view and args.game is None:
        raise SystemExit("--no-view reads --game/--decision once: give --game")
    if args.no_view and args.max_steps is None and args.max_seconds is None:
        raise SystemExit("--no-view has no Stop button: give --max-steps or --max-seconds")

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    loc = localiser(reg, args.locale)
    say = lambda text: print(text, file=sys.stderr)  # noqa: E731

    # The leaf, as play_human loads it: the inference server's.
    if args.hp_share:
        raise SystemExit("--hp-share is stopped: the hp-share leaf returns as an arm of the Rust "
                         "inference server in IKA-460 step 4")
    values = args.value
    if values is None:
        found = [repo_root() / p for p in DEFAULT_VALUE]
        if not all(p.exists() for p in found):
            raise SystemExit(f"no {DEFAULT_VALUE[0]} here: --value names the leaf")
        values = found
    address = humanplay.inference_address_or_start(
        args.inference, reg, values, args.q_model or (repo_root() / DEFAULT_Q), say)
    # IKA-363: the forward passes on the machine's server; no CUDA context here.
    evaluate, encoder = humanplay.served_leaf(reg, address, values)
    name = leaf_name(values)
    say(f"leaf on the inference server {address}")
    # The menus, as play_human picks them (IKA-341: the same fallback, one function).
    q_path = args.q_model or (repo_root() / DEFAULT_Q)
    fill, q_files = install_menus(args.rank_fill, q_path, encoder, address, say)
    install_deepening_q(q_path, encoder, address, say)
    humanplay.use_threads(args.threads)

    settings = analysis.Settings(
        width=args.width, oracle=_oracle_width(args.oracle), levels=args.max_levels,
        discount=args.depth_discount, rank_fill=fill, bench_drop=args.bench_drop, open_information=args.open,
        interval_ms=args.interval_ms,
    )
    analyzer = analysis.Analyzer(reg, evaluate, name, loc=loc, settings=settings)
    sources = [analysis.Source(path.name, path, limit=args.limit) for path in args.record]
    if args.current is not None:
        sources.append(analysis.Source("進行中の局", args.current, current=True))
    limits = analysis.Limits(rss_gb=args.max_rss_gb, free_gb=args.min_free_gb, gpu_gb=args.max_gpu_gb)
    say(f"analysis: leaf {name} / menus {fill} / width {settings.width} / oracle {settings.oracle_label()}"
        f" / guard {settings.levels}"
        + (f" / discount {settings.discount:g}" if settings.discount is not None else "")
        + f" / {args.threads} thread(s)"
        + (f" / {args.max_steps} steps" if args.max_steps is not None else "")
        + (f" / {args.max_seconds:g} s" if args.max_seconds is not None else ""))

    sink = liveview.FileSink(args.live_out) if args.live_out is not None else None
    service: analysis.Service
    server = liveview.LiveServer(
        args.view_host, 0 if args.no_view else args.view_port, sink=sink,
        sprite_url=args.sprite_url, on_command=lambda message: service.command(message),
        keep_steps=600, links={"game-url": args.game_url} if args.game_url else None,
    ).start()
    pools = {pool.id: pool}
    service = analysis.Service(
        analyzer, server, sources, limits=limits, max_steps=args.max_steps,
        max_seconds=args.max_seconds, threads=args.threads, out=args.out, pools=pools, say=say,
    )
    first = None
    if args.game is not None:
        first = {"cmd": "analyze", "source": 0, "game": args.game, "decision": args.decision,
                 "side": args.side}
    if not args.no_view:
        print(f"画面: {server.url}", file=sys.stderr)
        if args.open_browser:
            import webbrowser

            webbrowser.open(server.url)
    try:
        service.serve(first, once=args.no_view)
    except KeyboardInterrupt:
        pass
    finally:
        if service.session is not None:
            service.session.halt("person")
        if sink is not None:
            sink.close()
        with contextlib.suppress(OSError):
            server.close()
    if args.no_view and service.results:
        result = service.results[-1]
        best = sorted(zip(result.strategy, result.ours, strict=True), key=lambda t: -t[0])[:3]
        print(
            f"stop {result.stop}; steps {result.steps}; {result.seconds:.1f} s; value (side {result.side}) "
            f"{result.value:.4f}; lines at the guard {result.guard_lines}; nodes {result.nodes}; top "
            + ", ".join(f"{a} {p:.3f}" for p, a in best),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
