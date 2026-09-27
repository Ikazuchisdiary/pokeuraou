"""A verification position set: how far a reading's answer lands from a uniform depth-2 read (IKA-362).

A board answers "which is stronger" in thousands of games. Before one is spent, this asks a
cheaper question on a fixed set of recorded positions: given the same menus, how much does
a reading's mixture lose in a reference game read more deeply and *uniformly* -- every cell
of the matrix given its depth-2 value -- and how much does the reading believe it gets
beyond what the reference gives it (the optimizer's curse: a reading that picks the cells
its own errors flattered).

    python tools/position_set.py build --games-dir DATA/selfplay-mc1 --count 300 --out SET
    python tools/position_set.py reference --set SET --from 0 --to 25        # per worker
    python tools/position_set.py evaluate --set SET --arm m1:seconds=1,clock=count --from 0 --to 25
    python tools/position_set.py report --set SET

* **Positions** (`build`): one recorded move decision per game (turn 3 on, the middle one),
  read as the open game (the true position; the reading's menus are built as the agent
  builds them: q-nocover at ``--width``). Side 0 reads.
* **Reference** (`reference --name N`): the depth-1 matrix of those menus (the leaf), and
  the same matrix with every cell refined once (`search._refined_value`: the turn, its
  likeliest ``--sub-branches`` branches, each child's ``--sub-limit`` x ``--sub-limit``
  matrix by `narrow`'s damage order, solved). Written per position to
  ``SET/ref-N/<n>.npz``. A reference has to lie outside what the readings compared try --
  wider children and more branches than any of them -- or the reading most like it wins
  by resemblance (IKA-342's reversed discount); two references that order the readings
  differently leave that axis to the board.
* **Evaluate** (`evaluate`): a time-match condition (`timematch.parse_condition`; the node
  clock) reads each position with `humanplay.solve_move` on the same menus. Written per
  position to ``SET/eval/<arm>/<n>.json``: the mixture over the reference's rows (mass on
  actions the reading added from outside the menu is left out and said), the reading's
  own value, and in the reference game the value the mixture guarantees against every
  column (``got``), the reference's value (``best``), the loss ``best - got`` and the
  curse ``claimed - got``; the same against the depth-1 matrix.
* **Report**: per arm, the mean loss and curse with their standard errors, and the paired
  difference of any two arms.

The reference is itself a reading (uniform depth 2 with narrow children), not the truth:
a board decides; this set is for finding what is worth a board, and a record of how often
the two agree says how far to trust it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()  # before anything loads torch (IKA-360)

import numpy as np  # noqa: E402

from pokeuraou.regulation import repo_root  # noqa: E402

DEFAULT_VALUE = ("data/models/value-mc1.pt", "data/models/value-mc1-s1.pt")
DEFAULT_Q = "data/models/q-mc0.pt"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload) + "\n").encode("utf-8"))


def build(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    files = sorted(Path(args.games_dir).glob("games-worker*.jsonl"))
    rng.shuffle(files)
    picked = []
    for f in files:
        for line in f.read_bytes().splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            moves = [d for d in game["decisions"] if d["kind"] == "move" and d["turn"] >= 3]
            if moves:
                d = moves[len(moves) // 2]
                picked.append({"source": f.name, "turn": d["turn"], "position": d["position"]})
            if len(picked) >= args.count:
                break
        if len(picked) >= args.count:
            break
    out = Path(args.out)
    _write(out / "positions.json", {"positions": picked, "seed": args.seed,
                                     "gamesDir": str(args.games_dir), "width": args.width})
    print(f"{len(picked)} positions -> {out / 'positions.json'}", file=sys.stderr)


class _Kit:
    """The regulation, the leaf, the Q, and the positions of a set."""

    def __init__(self, args: argparse.Namespace) -> None:
        from pokeuraou import humanplay, qrank
        from pokeuraou.damage import register_mega_stones
        from pokeuraou.pool import load_pool

        pool = load_pool(args.pool)
        self.reg = pool.reg
        register_mega_stones(self.reg)
        values = args.value or [repo_root() / p for p in DEFAULT_VALUE]
        humanplay.cap_cuda(args.cuda_memory_gb, args.device)
        self.leaf, encoder, self.device = humanplay.load_leaf(
            self.reg, [Path(v) for v in values], args.device, graphs=False)
        if str(self.device) == "cpu":
            import torch

            torch.set_num_threads(1)
        qrank.install(qrank.LocalQ(Path(args.q_model or repo_root() / DEFAULT_Q), encoder,
                                   device=self.device))
        humanplay.use_threads(1, self.reg, None)
        data = json.loads((Path(args.set) / "positions.json").read_bytes())
        self.positions = data["positions"]
        self.width = data["width"]

    def menus(self, pos):  # noqa: ANN001, ANN201
        from pokeuraou.budget import Budget
        from pokeuraou.deepen import ALL_ACTIONS
        from pokeuraou.selfplay import _menus

        wider: dict = {}
        ours, theirs = _menus(self.reg, pos, (self.width, self.width), self.leaf, Budget.matrix(),
                              True, None, None, rank_fill="q-nocover", wide=[ALL_ACTIONS],
                              wider=wider)
        return ours, theirs, wider.get(ALL_ACTIONS)


def reference(args: argparse.Namespace) -> None:
    from pokeuraou.budget import Budget
    from pokeuraou.position import Position
    from pokeuraou.search import _refined_value, batched_payoff

    kit = _Kit(args)
    out = Path(args.set) / f"ref-{args.name}"
    for n in range(args.start, min(args.stop, len(kit.positions))):
        path = out / f"{n}.npz"
        if path.exists():
            continue
        began = time.perf_counter()
        pos = Position.from_json(kit.positions[n]["position"])
        ours, theirs, _outside = kit.menus(pos)
        budget = Budget.matrix()
        d1, _notes = batched_payoff(kit.reg, pos, ours, theirs, kit.leaf, budget=budget)
        d1 = np.asarray(d1, dtype=np.float64)
        d2 = d1.copy()
        refused = 0
        for i, a in enumerate(ours):
            for j, b in enumerate(theirs):
                value, _n, _s = _refined_value(kit.reg, pos, a, b, kit.leaf, budget=budget,
                                               sub_limit=args.sub_limit,
                                               sub_branches=args.sub_branches)
                if value is None:
                    refused += 1
                else:
                    d2[i, j] = value
        out.mkdir(parents=True, exist_ok=True)
        np.savez(path, d1=d1, d2=d2, rows=np.array([a.to_choice() for a in ours]),
                 cols=np.array([b.to_choice() for b in theirs]), refused=refused,
                 subLimit=args.sub_limit, subBranches=args.sub_branches)
        print(f"reference {n}: {len(ours)}x{len(theirs)}, refused {refused}, "
              f"{time.perf_counter() - began:.1f}s", file=sys.stderr, flush=True)


def _guarantee(x: np.ndarray, m: np.ndarray) -> float:
    return float((x @ m).min())


def evaluate(args: argparse.Namespace) -> None:
    from pokeuraou import humanplay, timematch
    from pokeuraou.budget import Budget
    from pokeuraou.deepen import COSTS, cells_for_seconds
    from pokeuraou.equilibrium import solve
    from pokeuraou.position import Position

    kit = _Kit(args)
    cond = timematch.parse_condition(args.arm)
    if cond.clock != "count":
        raise SystemExit("a position set reads on the node clock (clock=count)")
    out = Path(args.set) / "eval" / cond.name
    cells = 0 if cond.width_only else cells_for_seconds(cond.seconds, cond.price_cores)
    cost = COSTS["local", cond.price_cores]
    refs = sorted(d for d in Path(args.set).glob("ref-*") if d.is_dir())
    for n in range(args.start, min(args.stop, len(kit.positions))):
        path = out / f"{n}.json"
        found = {d.name[4:]: np.load(d / f"{n}.npz") for d in refs if (d / f"{n}.npz").exists()}
        if path.exists() or not found:
            continue
        ref = next(iter(found.values()))
        pos = Position.from_json(kit.positions[n]["position"])
        ours, theirs, outside = kit.menus(pos)
        if [a.to_choice() for a in ours] != list(ref["rows"]) or [
            b.to_choice() for b in theirs
        ] != list(ref["cols"]):
            raise SystemExit(f"position {n}: the menus are not the reference's (another leaf or Q?)")
        began = time.perf_counter()
        solved = humanplay.solve_move(
            kit.reg, pos, 0, list(ours), list(theirs), None, kit.leaf, budget=Budget.matrix(),
            exact=True, cells=cells, cost=cost, levels=cond.max_levels, child_q=cond.child_q,
            outside=outside if (cond.oracle is not None and not cond.restricted and cells) else None,
            sub_limit=cond.sub_limit, sub_branches=cond.sub_branches,
            restricted=cond.restricted, depth=cond.depth, refine=cond.refine,
            passes=cond.passes,
        )
        took = time.perf_counter() - began
        index = {c: i for i, c in enumerate(ref["rows"])}
        x = np.zeros(len(index), dtype=np.float64)
        outside_mass = 0.0
        for a, p in zip(solved.ours, solved.strategy, strict=True):
            k = index.get(a.to_choice())
            if k is None:
                outside_mass += float(p)
            else:
                x[k] += float(p)
        if x.sum() > 0:
            x /= x.sum()
        row = {"n": n, "seconds": round(took, 3), "claimed": solved.value, "outsideMass": outside_mass,
               "depth": None if solved.deepened is None else solved.deepened.depth,
               "expanded": None if solved.deepened is None else solved.deepened.expanded}
        for label, matrix in [("d1", ref["d1"])] + [(f"d2-{k}", r["d2"]) for k, r in found.items()]:
            m = np.asarray(matrix, dtype=np.float64)
            best = float(solve(m).value)
            got = _guarantee(x, m)
            row[label] = {"best": best, "got": got, "loss": best - got, "curse": solved.value - got}
        _write(path, row)
        print(json.dumps({k: row[k] for k in ("n", "seconds", "depth")}
                         | {k: round(v["loss"], 4) for k, v in row.items() if k.startswith("d")
                            and isinstance(v, dict)}), file=sys.stderr, flush=True)


def report(args: argparse.Namespace) -> None:
    base = Path(args.set) / "eval"
    arms = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        rows = {}
        for f in d.glob("*.json"):
            r = json.loads(f.read_bytes())
            rows[r["n"]] = r
        arms[d.name] = rows
    common = set.intersection(*(set(r) for r in arms.values())) if arms else set()
    print(f"{len(common)} positions read by every arm")
    keys = sorted({k for rows in arms.values() for r in rows.values() for k in r
                   if k.startswith("d") and isinstance(r[k], dict)})
    keys = [k for k in keys if all(k in rows[n] for rows in arms.values() for n in common)]
    for name, rows in arms.items():
        ns = sorted(common)
        for key in keys:
            loss = np.array([rows[n][key]["loss"] for n in ns])
            curse = np.array([rows[n][key]["curse"] for n in ns])
            print(f"  {name:<12} vs {key}: loss {loss.mean():.4f} (se {loss.std() / np.sqrt(len(ns)):.4f})"
                  f"  curse {curse.mean():+.4f} (se {curse.std() / np.sqrt(len(ns)):.4f})")
        secs = np.array([rows[n]["seconds"] for n in ns])
        print(f"  {name:<12} seconds mean {secs.mean():.2f}, outside mass mean "
              f"{np.mean([rows[n]['outsideMass'] for n in ns]):.3f}")
    names = list(arms)
    for key in keys:
        order = sorted(names, key=lambda a: np.mean([arms[a][n][key]["loss"] for n in common]))
        print(f"  order by loss vs {key}: {' < '.join(order)}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ns = sorted(common)
            for key in keys:
                d = np.array([arms[a][n][key]["loss"] - arms[b][n][key]["loss"] for n in ns])
                print(f"  loss {a} - {b} ({key}): {d.mean():+.4f} (se {d.std() / np.sqrt(len(ns)):.4f})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--games-dir", type=Path, required=True)
    b.add_argument("--count", type=int, default=300)
    b.add_argument("--seed", type=int, default=36200)
    b.add_argument("--width", type=int, default=64)
    b.add_argument("--out", type=Path, required=True)
    for name in ("reference", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--set", type=Path, required=True)
        s.add_argument("--from", dest="start", type=int, default=0)
        s.add_argument("--to", dest="stop", type=int, default=10**9)
        s.add_argument("--pool", default="regmc-matchupweb")
        s.add_argument("--value", type=Path, nargs="+", default=None)
        s.add_argument("--q-model", type=Path, default=None)
        s.add_argument("--device", default=None)
        s.add_argument("--cuda-memory-gb", type=float, default=0.8)
        if name == "reference":
            s.add_argument("--name", required=True, help="the reference's name: SET/ref-<name>")
            s.add_argument("--sub-limit", type=int, default=24)
            s.add_argument("--sub-branches", type=int, default=64,
                           help="branches kept per refined cell (64: all, in practice)")
        else:
            s.add_argument("--arm", required=True, help="a time-match condition on the node clock")
    r = sub.add_parser("report")
    r.add_argument("--set", type=Path, required=True)
    args = ap.parse_args(argv)
    {"build": build, "reference": reference, "evaluate": evaluate, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
