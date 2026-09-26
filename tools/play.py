"""Play the agent and read positions from one command (IKA-343): the game page and the
analysis page, opened in the browser.

    uv run python tools/play.py                        # 15 s a move; the game and the analysis pages
    uv run python tools/play.py --seconds 45           # a longer think
    uv run python tools/play.py --no-analysis          # the game alone (all the cores to the agent)
    uv run python tools/play.py --analysis-only        # read recorded games, no game
    uv run python tools/play.py --replay data/human/live/20260927-101500.bin   # a game's page again
    uv run python tools/play.py -- --human-team 3 --agent-team 17   # anything after -- goes to play_human

What it starts:

* **the game** (`tools/play_human.py --person web --view`, in this process): the page at
  http://127.0.0.1:8332/, the agent on its defaults (IKA-343: width first, the rest to
  deepening with the swap oracle; `humanplay.default_threads` threads; the CUDA cap and
  the memory watch). The game goes to ``--out`` (default ``data/human/games.jsonl``), the
  page's frames to ``data/human/live/<time>.bin`` (``--replay`` shows them again), and the
  move in hand to ``data/human/current.json`` for the analysis page;
* **the analysis mode** (`tools/analyze.py`, a second process, so that its reading and the
  agent's never share a process's threads): the page at http://127.0.0.1:8337/ lists the
  game in progress (「進行中の局」) and the games already in ``--out`` (「記録を読み直す」 picks
  up the ones finished since). Its threads come on top of the agent's: while it reads, the
  agent's moves have fewer cores and deepen less in the same seconds.

Both pages open in the browser (``--no-browser`` to only print the addresses). Closing: the
game ends with the last game (``--games``); Enter or Ctrl+C closes the pages and the
analysis process.
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

#: The seconds a move when --seconds is not given: long enough to deepen past the depth-1
#: answer on every move (IKA-330 §4.1: 5 s already reaches the widest menu), short enough
#: for a game of about 10 minutes of the agent's time.
DEFAULT_SECONDS = 15.0
GAME_PORT = 8332
ANALYSIS_PORT = 8337


def analysis_command(args: argparse.Namespace, current: Path) -> list[str]:
    """`tools/analyze.py` as this launcher starts it."""
    command = [
        sys.executable, str(ROOT / "tools" / "analyze.py"),
        "--record", str(args.out), "--view-port", str(args.analysis_port),
    ]
    if current is not None:
        command += ["--current", str(current)]
    if args.analysis_threads is not None:
        command += ["--threads", str(args.analysis_threads)]
    if not args.no_browser:
        command.append("--open-browser")
    return command


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="the agent's budget per move")
    ap.add_argument("--threads", type=int, default=None,
                    help="the agent's threads (default: humanplay.default_threads)")
    ap.add_argument("--analysis-threads", type=int, default=None,
                    help="the analysis mode's threads (default: analyze.py's)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="the teams' draw (default: from the clock)")
    ap.add_argument("--out", type=Path, default=Path("data/human/games.jsonl"))
    ap.add_argument("--port", type=int, default=GAME_PORT, help="the game page's port")
    ap.add_argument("--analysis-port", type=int, default=ANALYSIS_PORT)
    ap.add_argument("--no-analysis", action="store_true", help="the game alone")
    ap.add_argument("--analysis-only", action="store_true",
                    help="the analysis page alone, on --out (and any --record given after --)")
    ap.add_argument("--replay", type=Path, default=None,
                    help="show a game's page again from its frames (data/human/live/*.bin)")
    ap.add_argument("--speed", type=float, default=1.0, help="with --replay: 4 is four times as fast")
    ap.add_argument("--no-browser", action="store_true", help="print the addresses, open nothing")
    args = ap.parse_args(argv)
    args.out = args.out.resolve()

    if args.replay is not None:
        import live_view

        live_view.main([str(args.replay), "--port", str(args.port), "--speed", str(args.speed), *rest])
        return

    if args.analysis_only:
        import analyze

        analyze.main([
            "--record", str(args.out), "--view-port", str(args.analysis_port),
            *(["--threads", str(args.analysis_threads)] if args.analysis_threads is not None else []),
            *([] if args.no_browser else ["--open-browser"]), *rest,
        ])
        return

    import play_human

    stamp = time.strftime("%Y%m%d-%H%M%S")
    current = args.out.parent / "current.json"
    live = args.out.parent / "live" / f"{stamp}.bin"
    live.parent.mkdir(parents=True, exist_ok=True)
    helper = None
    if not args.no_analysis:
        helper = subprocess.Popen(analysis_command(args, current), cwd=str(ROOT))  # noqa: S603
        print(f"検討モード: http://127.0.0.1:{args.analysis_port}/（読み込み中）", file=sys.stderr)
    seed = args.seed if args.seed is not None else int(time.time()) % 1_000_000
    try:
        play_human.main([
            "--person", "web", "--view", "--view-port", str(args.port),
            "--seconds", str(args.seconds), "--games", str(args.games), "--seed", str(seed),
            "--out", str(args.out), "--live-out", str(live),
            *(["--current-out", str(current)] if helper is not None else []),
            *(["--threads", str(args.threads)] if args.threads is not None else []),
            *([] if args.no_browser else ["--open-browser"]), *rest,
        ])
    except KeyboardInterrupt:
        pass
    finally:
        if helper is not None:
            helper.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                helper.wait(timeout=15)
            if helper.poll() is None:
                helper.kill()
    print(f"対局の記録: {args.out}（画面の再生: tools/play.py --replay {live}）", file=sys.stderr)


if __name__ == "__main__":
    main()
