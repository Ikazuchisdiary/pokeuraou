"""The ranked-entry screen: a pasted team against six species, and the selection read (IKA-407).

    uv run python tools/ranked_entry.py --open-browser

A ranked match shows only the opponent's species. The screen takes your team as a Showdown
paste, the opponent's six species, fills the opponent's sets from the Baltimore Regional's most
common ones (every part labelled, every part overwritable), and reads the selection with the
same leaf, menus and reading as a person's game (`tools/play_human.py`: 90 s of the default
reading). It is for the review and practice after a match, not for use during one.

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

from pokeuraou import humanplay, selection_deep  # noqa: E402
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
        "models/value-mc3.pt", "models/value-mc3-s1.pt")]
    humanplay.cap_cuda(args.cuda_memory_gb, args.device)
    evaluate, encoder, device = humanplay.load_leaf(reg, values, args.device, graphs=True)
    name = ph.leaf_name(list(values))
    q_path = args.q_model or (args.data_dir_root / "models" / "q-mc3.pt")

    def say(text: str) -> None:
        print(text, file=sys.stderr)

    fill, q_files = ph.install_menus(None, q_path, encoder, evaluate, None, device, say)
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
    return solve, name, seconds


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
    if args.stub:
        solver, name, seconds = stub_solver(reg), "stub", 3.0
    else:
        solver, name, seconds = real_solver(reg, args)
    app = RankedApp(reg, prior, learned, localiser(reg, args.locale), solver, seconds=seconds, model=name)
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
