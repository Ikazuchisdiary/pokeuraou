"""The ranked-entry screen: a pasted team against six species, and the selection read (IKA-407).

    uv run python tools/ranked_entry.py --open-browser

A ranked match shows only the opponent's species. The screen takes your team as a Showdown
paste, the opponent's six species, fills the opponent's sets from the Baltimore Regional's most
common ones (every part labelled, every part overwritable), and reads the selection with the
same leaf, menus and reading as a person's game (`tools/play_human.py`: 90 s of the default
reading). It is for the review and practice after a match, not for use during one.

The page's second step (``/position``, IKA-408) takes the match turn by turn -- the four you
brought, who is out, HP (yours exactly, the opponent's as a per cent), what was seen -- and reads
the turn with the analysis mode's reading (`tools/analyze.py`'s: the menus at width 64, the
best-first deepening with the root's swap oracle, the depth guard of a person's game) for
``--read-seconds`` (40, a person's move). The opponent's set is narrowed by what is seen.

``--stub`` serves the screen with a cheap stand-in leaf (HP sums, no model, no reading): for
looking at the page and for tests. ``--data-dir`` names the `data/` that has the standings, the
pool and the models (a worktree has none).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()  # before anything loads torch (IKA-360)

import numpy as np  # noqa: E402

from pokeuraou import analysis, humanplay, selection_deep  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.names import localiser  # noqa: E402
from pokeuraou.rankedentry import FieldPrior, load_learned_ids  # noqa: E402
from pokeuraou.rankedweb import RankedApp, RankedServer  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


def _play_human() -> object:
    """`tools/play_human.py` as a module: its leaf and menu set-up is the one used here."""
    name = "pokeuraou_tool_play_human"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "play_human.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stub_leaf(positions):  # noqa: ANN001, ANN201
    """Antisymmetric stand-in: HP summed per side, the active pair counted twice."""
    out = np.empty(len(positions), dtype=np.float64)
    for i, pos in enumerate(positions):
        strength = [sum(m.hp * (2 if m.active_index is not None else 1) for m in side.pokemon)
                    for side in pos.sides]
        out[i] = 1.0 / (1.0 + np.exp(-(strength[0] - strength[1]) / 150.0))
    return out


def make_reader(analyzer, limits=None):  # noqa: ANN001, ANN201
    """The page's read of a typed-in position: ``analyzer.run`` for ``seconds``, as the analysis
    mode reads a position until it is stopped (here by the clock), from the person's side (0)."""
    def read(game, point, seconds, report):  # noqa: ANN001, ANN202
        report({"seconds": seconds})
        return analyzer.run(game, point, 0, max_seconds=seconds, limits=limits)
    return read


def stub_solver(reg):  # noqa: ANN001, ANN201
    def solve(mine, opponent, report):  # noqa: ANN001, ANN202
        report({"seconds": 3.0})
        time.sleep(3.0)
        return humanplay.solve_entry(reg, (mine, opponent), stub_leaf, "stub"), "stub"
    return solve


def real_solver(reg, args):  # noqa: ANN001, ANN201
    """The leaf, the menus and the reading a person's game reads its selection with."""
    ph = _play_human()
    values = args.value or [args.data_dir_root / p for p in (
        "models/value-mc4bindaux.pt", "models/value-mc4bindaux-s1.pt")]
    humanplay.cap_cuda(args.cuda_memory_gb, args.device)
    evaluate, encoder, device = humanplay.load_leaf(reg, values, args.device, graphs=True)
    name = ph.leaf_name(list(values))
    q_path = args.q_model or (args.data_dir_root / "models" / "q-mc4st.pt")

    def say(text: str) -> None:
        print(text, file=sys.stderr)

    fill, q_files = ph.install_menus(None, q_path, encoder, evaluate, None, device, say)
    threads = args.threads or humanplay.default_threads()
    humanplay.use_threads(
        threads, reg, ([str(v) for v in values], str(device or "cpu"), True, args.cuda_memory_gb))
    settings = analysis.Settings(
        width=analysis.DEFAULT_WIDTH, oracle=ph._oracle_width("sall"),
        levels=humanplay.PLAY_MAX_LEVELS or analysis.MAX_LEVELS, rank_fill=fill)
    analyzer = analysis.Analyzer(reg, evaluate, name, loc=args.loc, settings=settings)
    reader = make_reader(analyzer, analysis.Limits())
    reading, seconds = humanplay.PLAY_SELECTION_READING, humanplay.PLAY_SELECTION_SECONDS
    selection_deep.parse_reading(reading)
    readers = min(args.selection_workers or (__import__("os").cpu_count() or 2),
                  humanplay.LADDER_LOCAL_WORKERS_MAX)
    if readers > 1:
        selection_deep.use_reader(selection_deep.LazyPoolReader(
            reg, readers,
            humanplay.ladder_spec(leaf=evaluate, address=None, merge=False, values=values,
                                  device=device, graphs=True, cuda_memory_gb=args.cuda_memory_gb,
                                  q_path=q_path if fill != ph.NO_Q_FILL else None),
            rank_fill=fill))
    say(f"leaf {name} / menus {fill} {q_files} / selection read {reading} for {seconds:g} s "
        f"on {max(readers, 1)} process(es)")

    def solve(mine, opponent, report):  # noqa: ANN001, ANN202
        report({"seconds": seconds})
        reader = selection_deep.READER or selection_deep.SerialReader(
            reg, evaluate, rank_fill=fill, rank_by_leaf=True)
        deep: list = []
        entry = humanplay.solve_entry(reg, (mine, opponent), evaluate, name, reading=reading,
                                      reader=reader, seconds=seconds, report=deep)
        if hasattr(reader, "release"):
            reader.release()
        return entry, entry.model
    return solve, name, seconds, reader


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None, help="the data/ with standings, pool and models")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8338)
    ap.add_argument("--open-browser", action="store_true")
    ap.add_argument("--stub", action="store_true", help="a stand-in leaf, no model, no reading")
    ap.add_argument("--value", type=Path, nargs="+", default=None)
    ap.add_argument("--q-model", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cuda-memory-gb", type=float, default=humanplay.PLAY_CUDA_MEMORY_GB)
    ap.add_argument("--selection-workers", type=int, default=None)
    ap.add_argument("--read-seconds", type=float, default=40.0,
                    help="how long the typed-in position is read for (a person's move: 30-45)")
    ap.add_argument("--threads", type=int, default=None,
                    help=f"cores a typed-in position's read spreads over (default {humanplay.PLAY_THREADS}, "
                    "fewer on a small machine)")
    ap.add_argument("--sprite-url", default=None, help="{id} = Showdown's sprite id; \"\" for none")
    ap.add_argument("--locale", default="ja")
    args = ap.parse_args(argv)

    from pokeuraou.rankedentry import data_dir

    root = args.data_dir or data_dir()
    args.data_dir_root = root
    reg = load_regulation(FORMAT_ID)
    register_mega_stones(reg)
    prior = FieldPrior.load(reg, root)
    learned = load_learned_ids(root)
    args.loc = localiser(reg, args.locale)
    if args.stub:
        solver, name, seconds = stub_solver(reg), "stub", 3.0
        stub = analysis.Analyzer(
            reg, None, "stub", loc=args.loc,
            settings=analysis.Settings(width=8, oracle=None, levels=4, rank_fill="refs2"))
        reader, read_seconds = make_reader(stub), 3.0
    else:
        solver, name, seconds, reader = real_solver(reg, args)
        read_seconds = args.read_seconds
    app = RankedApp(reg, prior, learned, args.loc, solver, seconds=seconds, model=name,
                    reader=reader, read_seconds=read_seconds)
    server = RankedServer(app, args.host, args.port, sprite_url=args.sprite_url).start()
    print(f"画面: {server.url}", file=sys.stderr)
    if args.open_browser:
        webbrowser.open(server.url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.close()


if __name__ == "__main__":
    main()
