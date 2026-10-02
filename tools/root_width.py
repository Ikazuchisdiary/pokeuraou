"""IKA-418: how much a wider root candidate set is worth at the same node time, on L6.

L6 reads a rectangle of the root's depth-1 matrix, and that matrix is the candidate set the
width rule built (`humanplay.plan_move`: the widest of `WIDTHS`, 64 at most, whose node fits
half of the budget). This tool reads recorded positions (a `position_set.py` set) with L6 at
a fixed root width -- 12, 24, 36, 48, 64, every legal action -- on the node clock, and scores
each stage's answer in games whose rows are every legal action (`position_set.py reference
--all-actions`: ``F1`` at depth 1, ``F2`` with every cell refined once) as well as in the
set's own references (its width-36 menu: an answer outside it loses its mass).

    python tools/root_width.py run   --set SET --out OUT --only 0,3 --arm w12=12 --arm wall=all
    python tools/root_width.py run   --set SET --out OUT --arm S=staged:12-24-48-64-all --seconds 4,16
    python tools/root_width.py score --set SET --out OUT --refs F2,F1 --arm w12=12 --arm wall=all

* **run**: a fixed-width arm is read once, for the longest budget, with the count clock
  filling it (`ladder.COUNT_FILL`, set here): the answer at a smaller budget is the last
  stage finished by then (a filled ladder spends its budget and its stages do not depend on
  it, IKA-376), the node's own milliseconds already in each stage's clock. List the arms
  narrowest first: an arm whose menu is the one before it (every legal action already there)
  is not read again. A ``staged:`` arm (`humanplay.Agent.root_widths`) is read at each budget
  in turn, since the steps it takes depend on the budget.
* **score**: each answer's loss in each reference -- the reference's value less what the
  answer guarantees -- at each budget, per arm, with the standard error over positions; the
  paired difference of any arm from ``--base``; with ``--policies`` also three policies
  chosen from the arms by the budget (`policy_rows`): the width rule, the width rule with
  every legal action when that node fits, and the staged widening (its answer the fixed arm
  of the width it reaches, the intermediate nodes charged to the budget).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POKEURAOU_LADDER_COUNT_FILL", "1")  # before `ladder` is imported

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import position_set as ps  # noqa: E402

ALL = "all"


def _width(spec: str) -> int:
    from pokeuraou.deepen import ALL_ACTIONS

    return ALL_ACTIONS if spec == ALL else int(spec)


def _arms(args: argparse.Namespace) -> list[tuple[str, str]]:
    out = []
    for item in args.arm:
        name, eq, spec = item.partition("=")
        if not eq or not name:
            raise SystemExit(f"an arm is NAME=WIDTH or NAME=staged:12-24-..., not {item!r}")
        out.append((name, spec))
    return out


def _nested(kit, pos, widths: list[int], spreads):  # noqa: ANN001, ANN202
    """Menus at each width, narrowest first: one ranking's prefixes (`selfplay._menus`'s ``wide``)."""
    from pokeuraou.budget import Budget
    from pokeuraou.selfplay import _menus

    wider: dict = {}
    ours, theirs = _menus(kit.reg, pos, (widths[0], widths[0]), kit.leaf, Budget.matrix(), True,
                          None, spreads, rank_fill="q-nocover", wide=widths[1:], wider=wider)
    return [(ours, theirs), *(wider[w] for w in widths[1:])]


def _read(kit, n: int, spec: str, seconds: float, ladder: str):  # noqa: ANN001, ANN202
    """One L6 read of position ``n``: a fixed root width (``spec`` a number or ``all``) or a
    staged one (``staged:12-24-...``). Returns the row to write."""
    from pokeuraou import humanplay
    from pokeuraou.budget import Budget
    from pokeuraou.ladder import LADDER_COSTS, parse_ladder
    from pokeuraou.position import Position

    pos = Position.from_json(kit.positions[n]["position"])
    spreads = kit.spreads(n)
    exact = spreads is None
    classes = 1 if exact else len(spreads[1])
    price = humanplay.node_time(1)
    root = None
    if spec.startswith("staged:"):
        every = max(humanplay.legal_count(kit.reg, pos, 0), humanplay.legal_count(kit.reg, pos, 1))
        widths = sorted({min(w or every, every)
                         for w in humanplay.parse_root_widths(spec[len("staged:"):])})
        menus = _nested(kit, pos, widths, spreads)
        ours, theirs = menus[0]
        root = humanplay.RootSteps(menus, price, classes, humanplay.WIDTH_SHARE * seconds * 1000.0)
    else:
        ours, theirs, _outside = kit.menus_at(pos, _width(spec), spreads)
    node_ms = price.ms(len(ours) * len(theirs) * classes)
    began = time.perf_counter()
    solved = humanplay.solve_move(
        kit.reg, pos, 0, list(ours), list(theirs), spreads, kit.leaf, budget=Budget.matrix(),
        exact=exact,
        ladder={"stages": parse_ladder(ladder), "budget_ms": seconds * 1000.0,
                "cost": LADDER_COSTS["local", 1], "clock": "count", "start_ms": node_ms,
                **({} if root is None else {"root": root})})
    took = time.perf_counter() - began
    lad = solved.ladder
    final = [a.to_choice() for a in solved.ours]

    def answer(stage: str, spent: float, x, claimed: float) -> dict:  # noqa: ANN001
        return {"stage": stage, "spentMs": round(float(spent), 1), "claimed": float(claimed),
                "x": {c: round(float(p), 7) for c, p in zip(final, x, strict=True) if p > 1e-9}}

    x0, v0 = lad.start
    # The depth-1 answer spent the nodes read for it (a staged read's several).
    nodes_ms = node_ms if solved.root is None else sum(s["nodeMs"] for s in solved.root)
    return {"n": n, "spec": spec, "seconds": seconds, "wall": round(took, 2),
            "rows": len(solved.ours), "cols": len(solved.theirs), "classes": classes,
            "nodeMs": round(nodes_ms, 2), "stopped": lad.stopped, "work": lad.work,
            "spentMs": round(lad.spent_ms, 1),
            **({"root": solved.root} if solved.root is not None else {}),
            "answers": [answer("d1", nodes_ms, x0, v0)] + [
                answer(g.stage, g.spent_ms, g.strategy, g.value) for g in lad.rungs]}


def run(args: argparse.Namespace) -> None:
    from pokeuraou import humanplay
    from pokeuraou.position import Position

    kit = ps._Kit(args)
    out = Path(args.out)
    budgets = [float(s) for s in args.seconds.split(",")]
    for n in ps._mine(args, len(kit.positions)):
        pos = Position.from_json(kit.positions[n]["position"])
        legal = (humanplay.legal_count(kit.reg, pos, 0), humanplay.legal_count(kit.reg, pos, 1))
        previous: tuple[tuple[int, int], str] | None = None
        for name, spec in _arms(args):
            staged = spec.startswith("staged:")
            for seconds in (budgets if staged else [max(budgets)]):
                path = out / name / f"{n}@{seconds:g}.json"
                if path.exists():
                    continue
                if not staged:
                    want = _width(spec)
                    size = (min(want, legal[0]), min(want, legal[1]))
                    if previous is not None and size == previous[0]:
                        ps._write(path, {"n": n, "spec": spec, "seconds": seconds,
                                         "sameAs": previous[1], "rows": size[0], "cols": size[1]})
                        continue
                    previous = (size, name)
                row = _read(kit, n, spec, seconds, args.ladder)
                ps._write(path, row)
                print(json.dumps({"n": n, "arm": name, "seconds": seconds, "wall": row["wall"],
                                  "menu": [row["rows"], row["cols"]],
                                  "last": row["answers"][-1]["stage"],
                                  "root": row.get("root")}), file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- score


def _loss(answer: dict, ref) -> tuple[float, float, float]:  # noqa: ANN001
    """(loss, curse, mass outside the reference's rows) of an answer in a reference game."""
    index = {str(c): i for i, c in enumerate(ref["rows"])}
    x = np.zeros(len(index))
    outside = 0.0
    for choice, p in answer["x"].items():
        k = index.get(choice)
        if k is None:
            outside += p
        else:
            x[k] += p
    if x.sum() > 0:
        x = x / x.sum()
    weights = ref["weights"] if "weights" in ref.files else None
    best, got = ps._best_got(x, ref["d2"], weights)
    return best - got, answer["claimed"] - got, outside


def _answer_at(row: dict, seconds: float) -> dict | None:
    """The last stage finished within ``seconds`` (the nodes' milliseconds in its clock), or
    None: the depth-1 node alone does not fit."""
    fit = [a for a in row["answers"] if a["spentMs"] <= seconds * 1000.0]
    return fit[-1] if fit else None


def _row(out: Path, arm: str, n: int, seconds: float | None) -> dict | None:
    """An arm's read of position ``n``: a staged arm's at ``seconds``, a fixed one's (the only
    file, past a ``sameAs`` link)."""
    arm_dir = out / arm
    files = ([arm_dir / f"{n}@{seconds:g}.json"] if seconds is not None
             else sorted(arm_dir.glob(f"{n}@*.json")))
    for path in files:
        if not path.exists():
            continue
        got = json.loads(path.read_bytes())
        return _row(out, got["sameAs"], n, None) if "sameAs" in got else got
    return None


def policy_rows(rows: int, cols: int, classes: int, seconds: float) -> dict[str, tuple[str, float]]:
    """The policies at one position and budget: label -> (the fixed arm to read, milliseconds
    the policy spent before it that the arm's own clock does not hold).

    ``rule``: the width rule's menu (`humanplay.plan_move`), the arm named ``w<width>``.
    ``allfit``: every legal action (arm ``wall``) when that node fits `humanplay.WIDTH_SHARE`
    of the budget, else the rule's. ``staged``: the last of 12, 24, 48, 64, every legal action
    that fits the share with the nodes read before it (as `humanplay.RootSteps.later`), the
    nodes before it charged."""
    from pokeuraou import humanplay

    price = humanplay.node_time(1)
    share = humanplay.WIDTH_SHARE * seconds * 1000.0

    def node_ms(width: int) -> float:
        return price.ms(min(width, rows) * min(width, cols) * max(classes, 1))

    plan = humanplay.plan_move(seconds, 1, rows, cols, classes)
    every = max(rows, cols)
    out = {"rule": (f"w{plan.width}", 0.0),
           "allfit": ("wall" if node_ms(every) <= share else f"w{plan.width}", 0.0)}
    spent, last, picked, before = 0.0, None, "w12", 0.0
    for width, arm in ((12, "w12"), (24, "w24"), (48, "w48"), (64, "w64"), (every, "wall")):
        size = (min(width, rows), min(width, cols))
        if size == last:
            continue
        ms = node_ms(width)
        if last is not None and spent + ms > share:
            break
        before, picked, last = spent, arm, size
        spent += ms
    out["staged"] = (picked, before)
    return out


def score(args: argparse.Namespace) -> None:
    set_dir, out = Path(args.set), Path(args.out)
    budgets = [float(s) for s in args.seconds.split(",")]
    named = _arms(args)
    fixed = {name for name, spec in named if not spec.startswith("staged:")}
    data = json.loads((set_dir / "positions.json").read_bytes())["positions"]
    positions = ([int(n) for n in args.only.split(",")] if args.only else sorted(
        int(p.stem) for p in (set_dir / f"ref-{args.refs.split(',')[0]}").glob("*.npz")))
    legal: dict[int, tuple[int, int, int]] = {}
    if args.policies:
        from pokeuraou import humanplay
        from pokeuraou.damage import register_mega_stones
        from pokeuraou.pool import load_pool
        from pokeuraou.position import Position

        reg = load_pool(args.pool).reg
        register_mega_stones(reg)
        for n in positions:
            pos = Position.from_json(data[n]["position"])
            legal[n] = (humanplay.legal_count(reg, pos, 0), humanplay.legal_count(reg, pos, 1),
                        len(data[n]["spreads"]["1"]) if "spreads" in data[n] else 1)
    table: dict = {}
    for ref_name in args.refs.split(","):
        for n in positions:
            path = set_dir / f"ref-{ref_name}" / f"{n}.npz"
            if not path.exists():
                continue
            ref = np.load(path)
            for name, _spec in named:
                for seconds in budgets:
                    row = _row(out, name, n, None if name in fixed else seconds)
                    ans = None if row is None else _answer_at(row, seconds)
                    if ans is not None:
                        table.setdefault((ref_name, name, seconds), {})[n] = _loss(ans, ref)
            for seconds in budgets if args.policies else []:
                for label, (arm, spent) in policy_rows(*legal[n], seconds).items():
                    row = _row(out, arm, n, None)
                    ans = None if row is None else _answer_at(row, seconds - spent / 1000.0)
                    if ans is not None:
                        table.setdefault((ref_name, label, seconds), {})[n] = _loss(ans, ref)
    _print(table, args, [name for name, _ in named])


def _print(table: dict, args: argparse.Namespace, arms: list[str]) -> None:
    budgets = [float(s) for s in args.seconds.split(",")]
    labels = arms + sorted({k[1] for k in table} - set(arms))
    for ref_name in args.refs.split(","):
        print(f"\nagainst {ref_name}")
        print(f"  {'arm':10s} {'s':>5s} {'n':>3s} {'loss':>8s} {'se':>7s} {'curse':>8s} {'outside':>8s}")
        for label in labels:
            for seconds in budgets:
                got = table.get((ref_name, label, seconds))
                if not got:
                    continue
                v = np.array(list(got.values()))
                se = v[:, 0].std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
                print(f"  {label:10s} {seconds:5g} {len(v):3d} {v[:, 0].mean():8.4f} {se:7.4f} "
                      f"{v[:, 1].mean():+8.4f} {v[:, 2].mean():8.3f}")
        if args.base:
            print(f"  paired (label - {args.base}), positions in both")
            for label in labels:
                if label == args.base:
                    continue
                for seconds in budgets:
                    a = table.get((ref_name, label, seconds))
                    b = table.get((ref_name, args.base, seconds))
                    if not a or not b:
                        continue
                    d = np.array([a[n][0] - b[n][0] for n in sorted(set(a) & set(b))])
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                    print(f"  {label:10s} {seconds:5g} {len(d):3d} {d.mean():+8.4f} {se:7.4f} "
                          f"better {int((d < -5e-5).sum())} worse {int((d > 5e-5).sum())}")
    if args.json:
        Path(args.json).write_bytes((json.dumps({
            "|".join(map(str, k)): {str(n): list(v) for n, v in t.items()}
            for k, t in table.items()}) + "\n").encode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--set", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--only", default=None, metavar="N,N,...")
        s.add_argument("--from", dest="start", type=int, default=0)
        s.add_argument("--to", dest="stop", type=int, default=10**9)
        s.add_argument("--stride", type=int, default=1)
        s.add_argument("--offset", type=int, default=0)
        s.add_argument("--claim", default=None)
        s.add_argument("--arm", action="append", default=[], help="NAME=WIDTH|all|staged:W-W-all")
        s.add_argument("--seconds", default="4,16,45,128")
        s.add_argument("--pool", default="regmc-matchupweb")
        s.add_argument("--value", type=Path, nargs="+", default=None)
        s.add_argument("--q-model", type=Path, default=None)
        s.add_argument("--device", default=None)
        s.add_argument("--cuda-memory-gb", type=float, default=0.8)
        s.add_argument("--inference", default=None, metavar="HOST:PORT")
        s.add_argument("--ladder-inference", default=None)
        s.add_argument("--merge", default="off", choices=("on", "off"))
        s.add_argument("--threads", type=int, default=1)
    sub.choices["run"].add_argument("--ladder", default="L6")
    sc = sub.choices["score"]
    sc.add_argument("--refs", default="F2,F1")
    sc.add_argument("--base", default=None)
    sc.add_argument("--policies", action="store_true",
                    help="rule, allfit and staged, chosen from the fixed arms w12 w24 w48 w64 wall")
    sc.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    {"run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
