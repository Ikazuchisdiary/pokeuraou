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

IKA-367 adds the long-time band (a person's 30-45 s, the analysis view's hours):

    python tools/position_set.py subset --set SET --out SMALL --count 40
    python tools/position_set.py deep --set SMALL --base r24 --name deep --from 0 --to 5
    python tools/position_set.py sweep --set SMALL --arm lad:ladder=L1 --seconds 1,4,16,64
    python tools/position_set.py curve --set SMALL --svg curve.svg
    python tools/position_set.py fit --set SMALL

* **subset**: the first N positions of a set, with every reference they have.
* **deep** (the long reference, IKA-366 §5.3): the base reference (every cell at depth 2)
  with the cells of an 8 x 8 rectangle of its root read three plies -- the rectangle is the
  base answer's support, then the actions that do best against the other side's answer;
  each cell's turn with every branch and the knock-outs forked, each child at *every*
  legal action a side, its 6 x 6 rectangle (support, then best replies) read a ply further
  with every branch and the grandchildren at the Q's 24, the child game solved whole.
* **sweep**: each condition read at each of several budgets on the node clock, as a
  person's game reads it: the width rule's menus (at most the set's), then the
  deepening, `depth2_auto` or the ladder (`ladder.read`). Written per position to
  ``SET/sweep/<arm>@<seconds>/<n>.json`` with the wall seconds and, for a ladder, its
  stages and counted work; ``--keep-matrix`` also writes a ladder's root prices as a
  reference ``SET/ref-<arm>@<seconds>/`` (the longest read's game).
* **curve**: per condition, loss and curse against every reference at each budget, the
  paired change from one budget to the next and how many positions got worse; one table
  and an SVG figure. ``--elo`` sets loss against a recorded Elo curve (IKA-366's 0.1 -> 0.8
  s steps) to give one Elo per unit of loss.
* **fit**: least squares of a ladder's wall milliseconds on its counted work
  (`ladder.LadderCost`).
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

DEFAULT_VALUE = ("data/models/value-mc2.pt", "data/models/value-mc2-s1.pt")
DEFAULT_Q = "data/models/q-mc2.pt"


def _mine(args: argparse.Namespace, count: int):  # noqa: ANN201 - an iterator of positions
    """This process's positions: ``--from``..``--to``, every ``--stride``-th from ``--offset``,
    or with ``--claim`` each one no other process has claimed yet (`_units`)."""
    return _units(args, list(range(args.start, min(args.stop, count))))


def _units(args: argparse.Namespace, units: list):  # noqa: ANN201 - an iterator of units
    """The units of work this process takes. ``--claim DIR``: in order, each one whose claim
    file it creates first (processes pulling from one queue, so a slow unit holds up only
    its own process); else every ``--stride``-th from ``--offset``."""
    import os

    if getattr(args, "claim", None) is None:
        yield from units[args.offset::args.stride]
        return
    claims = Path(args.claim)
    claims.mkdir(parents=True, exist_ok=True)
    for unit in units:
        name = "-".join(str(u) for u in unit) if isinstance(unit, tuple) else str(unit)
        try:
            os.close(os.open(claims / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        yield unit


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload) + "\n").encode("utf-8"))


def build(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    files = sorted(Path(args.games_dir).glob("games-*worker*.jsonl"))
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


def build_hidden(args: argparse.Namespace) -> None:
    """Positions where side 0 cannot see side 1's bench (IKA-367): one move decision a
    game (turn ``--min-turn`` on, the middle of those with 2 to ``--max-completions``
    completions of side 1's bench), with both sides' completions as a person's opponent
    builds them from the public information (`hidden.completions` over the pool's sheet
    and the slots shown, uniform weights: no bench prior)."""
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.hidden import completions, seen_slots
    from pokeuraou.pool import load_pool
    from pokeuraou.position import Position

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    rng = np.random.default_rng(args.seed)
    files = sorted(Path(args.games_dir).glob("games-*worker*.jsonl"))
    rng.shuffle(files)
    picked = []

    def jsonl(items) -> list[dict]:  # noqa: ANN001
        return [{"position": c.position.to_json(), "species": list(c.species),
                 "slots": list(c.slots), "weight": c.weight, "exact": c.exact} for c in items]

    for f in files:
        for line in f.read_bytes().splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            named = game.get("pool")
            if game.get("information") != "hidden-bench" or named is None:
                continue
            if named["sha256"] != pool.sha256:
                raise SystemExit(f"{f.name}: a game from pool {named['sha256']}, not {pool.sha256}")
            teams = [next(t for t in pool.teams if t.id == named["teams"][side]) for side in (0, 1)]
            found = []
            for d in game["decisions"]:
                if d["kind"] != "move" or d["turn"] < args.min_turn:
                    continue
                pos = Position.from_json(d["position"])
                shown = d["shownIdentities"]
                spreads = {side: completions(reg, pos, side, list(teams[side].sets),
                                             seen=seen_slots(pos, side, shown[side]))
                           for side in (0, 1)}
                if 2 <= len(spreads[1]) <= args.max_completions:
                    found.append((d, spreads))
            if found:
                d, spreads = found[len(found) // 2]
                picked.append({"source": f.name, "turn": d["turn"], "position": d["position"],
                               "spreads": {str(side): jsonl(spreads[side]) for side in (0, 1)}})
            if len(picked) >= args.count:
                break
        if len(picked) >= args.count:
            break
    out = Path(args.out)
    _write(out / "positions.json", {"positions": picked, "seed": args.seed, "hidden": True,
                                     "gamesDir": str(args.games_dir), "width": args.width})
    sizes = [len(e["spreads"]["1"]) for e in picked]
    print(f"{len(picked)} hidden positions (completions {min(sizes)}-{max(sizes)}, mean "
          f"{np.mean(sizes):.1f}) -> {out / 'positions.json'}", file=sys.stderr)


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
        q_path = Path(args.q_model or repo_root() / DEFAULT_Q)
        address = humanplay.inference_address(args.inference)
        if address:
            # IKA-363: the leaf and the Q on the machine's server; no CUDA context here.
            self.leaf, encoder = humanplay.served_leaf(
                self.reg, address, values, merge=args.merge == "on", q_path=q_path)
            self.device = f"server {address}"
            #: A ladder worker's leaf and Q (IKA-364, `humanplay.ladder_process_leaf`).
            # IKA-390: with --ladder-inference, the workers' servers (a list, one per worker
            # in turn, `ladder.start_pool`); this process's own leaf stays on ``address``.
            self.spec: tuple = ("served", getattr(args, "ladder_inference", None) or address,
                                "value", args.merge == "on", str(q_path), "q")
        else:
            humanplay.cap_cuda(args.cuda_memory_gb, args.device)
            self.leaf, encoder, self.device = humanplay.load_leaf(
                self.reg, [Path(v) for v in values], args.device,
                # CUDA graphs as a person's game loads the leaf (IKA-367's timing), else off.
                graphs=bool(getattr(args, "leaf_graphs", False)))
            if str(self.device) == "cpu":
                import torch

                torch.set_num_threads(1)
            qrank.install(qrank.LocalQ(q_path, encoder, device=self.device))
            self.spec = ("local", [str(v) for v in values], str(self.device),
                         bool(getattr(args, "leaf_graphs", False)), args.cuda_memory_gb,
                         str(q_path))
        humanplay.use_threads(1, self.reg, None)
        threads = int(getattr(args, "threads", 1) or 1)
        if threads > 1:
            # IKA-364: a ladder's cells on threads - 1 workers, the port's cells here.
            humanplay.use_threads(threads)
            got = humanplay.use_ladder_pool(threads, self.reg, self.spec)
            print(f"ladder cells on {got} worker process(es)", file=sys.stderr, flush=True)
        data = json.loads((Path(args.set) / "positions.json").read_bytes())
        self.positions = data["positions"]
        self.width = data["width"]

    def spreads(self, n: int) -> dict | None:
        """A hidden position's completions by side (`build-hidden`, IKA-367), or None."""
        from pokeuraou.hidden import Completion
        from pokeuraou.position import Position

        entry = self.positions[n]
        if "spreads" not in entry:
            return None
        return {int(side): [Completion(position=Position.from_json(c["position"]),
                                       species=tuple(c["species"]), slots=tuple(c["slots"]),
                                       weight=float(c["weight"]), exact=bool(c["exact"]))
                            for c in items]
                for side, items in entry["spreads"].items()}

    def menus(self, pos, spreads=None):  # noqa: ANN001, ANN201
        return self.menus_at(pos, self.width, spreads)

    def menus_at(self, pos, width: int, spreads=None):  # noqa: ANN001, ANN201
        """The menus at a width, as the agent builds them there (IKA-367; on a hidden
        position ranked from each side's heaviest completion of the other's bench)."""
        from pokeuraou.budget import Budget
        from pokeuraou.deepen import ALL_ACTIONS
        from pokeuraou.selfplay import _menus

        wider: dict = {}
        ours, theirs = _menus(self.reg, pos, (width, width), self.leaf, Budget.matrix(), True,
                              None, spreads, rank_fill="q-nocover", wide=[ALL_ACTIONS],
                              wider=wider)
        return ours, theirs, wider.get(ALL_ACTIONS)


def reference(args: argparse.Namespace) -> None:
    from pokeuraou.budget import Budget
    from pokeuraou.position import Position
    from pokeuraou.search import _refined_value, batched_payoff

    kit = _Kit(args)
    out = Path(args.set) / f"ref-{args.name}"
    for n in _mine(args, len(kit.positions)):
        path = out / f"{n}.npz"
        if path.exists():
            continue
        began = time.perf_counter()
        pos = Position.from_json(kit.positions[n]["position"])
        spreads = kit.spreads(n)
        if getattr(args, "menus_from", None):
            # IKA-394: another leaf's game on the menus of an existing reference (the rows
            # and columns of SET/ref-<name>), so that two leaves' games share their actions.
            base = np.load(Path(args.set) / f"ref-{args.menus_from}" / f"{n}.npz")
            ours = _from_choices(kit.reg, pos, 0, base["rows"])
            theirs = _from_choices(kit.reg, pos, 1, base["cols"])
        else:
            ours, theirs, _outside = kit.menus(pos, spreads)
        budget = Budget.matrix()
        # A hidden position: the game once per completion of side 1's bench (IKA-367).
        worlds = [pos] if spreads is None else [c.position for c in spreads[1]]
        d1s, d2s = [], []
        refused = 0
        for world in worlds:
            d1, _notes = batched_payoff(kit.reg, world, ours, theirs, kit.leaf, budget=budget)
            d1 = np.asarray(d1, dtype=np.float64)
            d2 = d1.copy()
            for i, a in enumerate(ours):
                for j, b in enumerate(theirs):
                    value, _n, _s = _refined_value(kit.reg, world, a, b, kit.leaf, budget=budget,
                                                   sub_limit=args.sub_limit,
                                                   sub_branches=args.sub_branches)
                    if value is None:
                        refused += 1
                    else:
                        d2[i, j] = value
            d1s.append(d1)
            d2s.append(d2)
        out.mkdir(parents=True, exist_ok=True)
        extra = {} if spreads is None else {"weights": [c.weight for c in spreads[1]]}
        np.savez(path, d1=d1s[0] if spreads is None else np.stack(d1s),
                 d2=d2s[0] if spreads is None else np.stack(d2s),
                 rows=np.array([a.to_choice() for a in ours]),
                 cols=np.array([b.to_choice() for b in theirs]), refused=refused,
                 subLimit=args.sub_limit, subBranches=args.sub_branches, **extra)
        print(f"reference {n}: {len(ours)}x{len(theirs)}, refused {refused}, "
              f"{time.perf_counter() - began:.1f}s", file=sys.stderr, flush=True)


def _support(strategy: np.ndarray, k: int) -> list[int]:
    """The heaviest `k` actions carrying weight."""
    live = np.flatnonzero(np.asarray(strategy) > 1e-6)
    order = live[np.argsort(-np.asarray(strategy)[live], kind="stable")]
    return [int(i) for i in order[:k]]


def reference3(args: argparse.Namespace) -> None:
    """A depth-3 reference on the support: the depth-2 reference ``--base`` (every cell at
    depth 2), with the cells of its root's support rectangle (each side's ``--rect``
    heaviest) read a ply further -- each branch's child game ``--sub-limit`` wide by damage,
    the cells of *its* support rectangle refined to depth 2 (children as wide, every
    branch), the child game solved -- and the root matrix otherwise the base's. Written to
    ``SET/ref-<name>/<n>.npz`` as ``d2`` (the evaluation reads it like any reference)."""
    from pokeuraou.budget import Budget
    from pokeuraou.equilibrium import EquilibriumError, solve
    from pokeuraou.narrow import narrow
    from pokeuraou.position import Position
    from pokeuraou.search import _kept_branches, _refined_value, batched_payoff

    kit = _Kit(args)
    out = Path(args.set) / f"ref-{args.name}"
    base_dir = Path(args.set) / f"ref-{args.base}"
    budget = Budget.matrix()
    for n in _mine(args, len(kit.positions)):
        path = out / f"{n}.npz"
        if path.exists() or not (base_dir / f"{n}.npz").exists():
            continue
        began = time.perf_counter()
        base = np.load(base_dir / f"{n}.npz")
        pos = Position.from_json(kit.positions[n]["position"])
        ours, theirs, _outside = kit.menus(pos)
        if [a.to_choice() for a in ours] != list(base["rows"]):
            raise SystemExit(f"position {n}: the menus are not the base reference's")
        d3 = np.asarray(base["d2"], dtype=np.float64).copy()
        eq = solve(d3)
        rows, cols = _support(eq.row_strategy, args.rect), _support(eq.col_strategy, args.rect)
        deepened = 0
        for i in rows:
            for j in cols:
                notes: set[str] = set()
                kept = _kept_branches(kit.reg, pos, ours[i], theirs[j], budget, 64, notes)
                if kept is None:
                    continue
                branches, weights = kept
                values = []
                for branch in branches:
                    child = branch.position
                    if child.ended:
                        values.append(float(np.asarray(kit.leaf([child]))[0]))
                        continue
                    crow = narrow(kit.reg, child, 0, limit=args.sub_limit).actions
                    ccol = narrow(kit.reg, child, 1, limit=args.sub_limit).actions
                    if not crow or not ccol:
                        values = None
                        break
                    m, _notes = batched_payoff(kit.reg, child, crow, ccol, kit.leaf, budget=budget)
                    m = np.asarray(m, dtype=np.float64)
                    try:
                        ceq = solve(m)
                    except EquilibriumError:
                        values = None
                        break
                    for ci in _support(ceq.row_strategy, args.rect):
                        for cj in _support(ceq.col_strategy, args.rect):
                            v, _n, _s = _refined_value(kit.reg, child, crow[ci], ccol[cj], kit.leaf,
                                                       budget=budget, sub_limit=args.sub_limit,
                                                       sub_branches=64)
                            if v is not None:
                                m[ci, cj] = v
                    try:
                        values.append(float(solve(m).value))
                    except EquilibriumError:
                        values = None
                        break
                if values is not None:
                    d3[i, j] = float(np.array(values) @ weights)
                    deepened += 1
        out.mkdir(parents=True, exist_ok=True)
        np.savez(path, d1=base["d1"], d2=d3, rows=base["rows"], cols=base["cols"],
                 deepened=deepened, rect=args.rect, subLimit=args.sub_limit)
        print(f"reference3 {n}: rectangle {len(rows)}x{len(cols)}, deepened {deepened}, "
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
    for n in _mine(args, len(kit.positions)):
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
               "x": [round(float(v), 6) for v in x],
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


def rescore(args: argparse.Namespace) -> None:
    """Every evaluation's kept mixture scored again against every reference now present."""
    from pokeuraou.equilibrium import solve

    refs = sorted(d for d in Path(args.set).glob("ref-*") if d.is_dir())
    best: dict = {}
    for f in sorted((Path(args.set) / "eval").glob("*/*.json")):
        row = json.loads(f.read_bytes())
        if "x" not in row:
            continue
        n = row["n"]
        x = np.asarray(row["x"], dtype=np.float64)
        for d in refs:
            ref_path = d / f"{n}.npz"
            if not ref_path.exists():
                continue
            m = np.asarray(np.load(ref_path)["d2"], dtype=np.float64)
            if (d.name, n) not in best:
                best[d.name, n] = float(solve(m).value)
            got = _guarantee(x, m)
            row[f"d2-{d.name[4:]}"] = {"best": best[d.name, n], "got": got,
                                        "loss": best[d.name, n] - got,
                                        "curse": row["claimed"] - got}
        _write(f, row)
    print("rescored", file=sys.stderr)


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


# ----------------------------------------------------------------------------- IKA-367


def subset(args: argparse.Namespace) -> None:
    """The first ``--count`` positions of a set and every reference they have."""
    src, out = Path(args.set), Path(args.out)
    data = json.loads((src / "positions.json").read_bytes())
    data["positions"] = data["positions"][: args.count]
    data["subsetOf"] = str(src)
    _write(out / "positions.json", data)
    copied = 0
    for ref in sorted(d for d in src.glob("ref-*") if d.is_dir()):
        for n in range(args.count):
            f = ref / f"{n}.npz"
            if f.exists():
                (out / ref.name).mkdir(parents=True, exist_ok=True)
                (out / ref.name / f.name).write_bytes(f.read_bytes())
                copied += 1
    print(f"{len(data['positions'])} positions, {copied} reference files -> {out}", file=sys.stderr)


def _rectangle(strategy: np.ndarray, ev: np.ndarray, count: int, *, larger: bool) -> list[int]:
    """Support by weight, then the rest by how they do against the other side's answer."""
    from pokeuraou.ladder import _order

    return _order(strategy, ev, count, larger=larger)


def _deep_cell(kit, pos, a, b, budget, args, work: dict) -> float | None:  # noqa: ANN001
    """One root cell read three plies (`deep`'s docstring)."""
    from pokeuraou import qhead
    from pokeuraou.equilibrium import EquilibriumError, solve
    from pokeuraou.search import _kept_branches, _refine_cells, batched_payoff

    notes: set[str] = set()
    kept = _kept_branches(kit.reg, pos, a, b, budget, 1000, notes)
    if kept is None:
        return None
    branches, weights = kept
    work["turns"] += 1
    values = []
    for branch in branches:
        child = branch.position
        if child.ended:
            values.append(float(np.asarray(kit.leaf([child]))[0]))
            continue
        crow, ccol = qhead.legal_pool(kit.reg, child, 0), qhead.legal_pool(kit.reg, child, 1)
        if not crow or not ccol:
            return None
        m, _notes = batched_payoff(kit.reg, child, crow, ccol, kit.leaf, budget=budget)
        m = np.asarray(m, dtype=np.float64)
        work["subgames"] += 1
        work["cells"] += m.size
        try:
            eq = solve(m)
        except EquilibriumError:
            return None
        rows = _rectangle(eq.row_strategy, eq.row_ev, args.child_rect, larger=True)
        cols = _rectangle(eq.col_strategy, eq.col_ev, args.child_rect, larger=False)
        cells = [(i, j) for i in rows for j in cols]
        found = _refine_cells(kit.reg, [(child, crow[i], ccol[j]) for i, j in cells], kit.leaf,
                              budget=budget, sub_limit=args.grand, sub_branches=1000,
                              child_q=args.grand)
        for (i, j), (v, _n, _s) in zip(cells, found, strict=True):
            if v is not None:
                m[i, j] = v
        try:
            values.append(float(solve(m).value))
        except EquilibriumError:
            return None
    return float(np.asarray(values) @ weights)


def _from_choices(reg, pos, side: int, choices) -> list:  # noqa: ANN001
    """A side's menu from its recorded choice strings, in that order."""
    from pokeuraou.actions import side_actions

    legal = {a.to_choice(): a for a in side_actions(reg, pos, side)}
    missing = [c for c in choices if str(c) not in legal]
    if missing:
        raise SystemExit(f"choices not legal here: {missing[:3]}")
    return [legal[str(c)] for c in choices]


def deep(args: argparse.Namespace) -> None:  # noqa: C901 - the units and their assembly
    """The long reference (the module's docstring, IKA-367). The work is one unit per
    completion of a position (an open position is one): each writes its part, and the unit
    that finds every part of its position present writes the position's reference."""
    from dataclasses import replace

    from pokeuraou import search
    from pokeuraou.budget import Budget
    from pokeuraou.equilibrium import solve_bayesian
    from pokeuraou.position import Position

    kit = _Kit(args)
    out = Path(args.set) / f"ref-{args.name}"
    parts = out / "parts"
    base_dir = Path(args.set) / f"ref-{args.base}"
    budget = replace(Budget.matrix(), enumerate_knockouts=True)
    units = []
    for n in range(args.start, min(args.stop, len(kit.positions))):
        if (out / f"{n}.npz").exists() or not (base_dir / f"{n}.npz").exists():
            continue
        spreads = kit.positions[n].get("spreads")
        units += [(n, k) for k in range(1 if spreads is None else len(spreads["1"]))]
    for n, k in _units(args, units):
        if (parts / f"{n}-{k}.npz").exists():
            continue
        began = time.perf_counter()
        base = np.load(base_dir / f"{n}.npz")
        pos = Position.from_json(kit.positions[n]["position"])
        spreads = kit.spreads(n)
        # The base's own menus, read back from its choices: the same actions whatever device
        # this process's leaf and Q are on (a CPU process ranks a hair differently).
        ours, theirs = (_from_choices(kit.reg, pos, side, base[key])
                        for side, key in ((0, "rows"), (1, "cols")))
        # One matrix per completion (an open position is one of weight 1): the rectangle is
        # the base's Bayesian answer's -- one row set, a column set per completion.
        mats, w = _game(base["d2"], base.get("weights", None))
        world = pos if spreads is None else spreads[1][k].position
        eq = solve_bayesian(mats, w)
        rows = _rectangle(eq.row_strategy, eq.row_ev, args.rect, larger=True)
        cols = _rectangle(eq.col_strategies[k], eq.row_strategy @ mats[k], args.rect,
                          larger=False)
        work = {"turns": 0, "subgames": 0, "cells": 0, "qs": 0}
        search.WORK = work
        mask = np.zeros(mats[k].shape, dtype=bool)
        try:
            for i in rows:
                for j in cols:
                    v = _deep_cell(kit, world, ours[i], theirs[j], budget, args, work)
                    if v is not None:
                        mats[k][i, j] = v
                        mask[i, j] = True
        finally:
            search.WORK = None
        took = time.perf_counter() - began
        parts.mkdir(parents=True, exist_ok=True)
        tmp = parts / f"{n}-{k}.tmp.npz"
        np.savez(tmp, d2=mats[k], deep=mask, seconds=took, work=json.dumps(work))
        tmp.replace(parts / f"{n}-{k}.npz")
        print(f"deep {n}/{k}: rectangle {len(rows)}x{len(cols)}, deepened {int(mask.sum())}, "
              f"{took:.1f}s, work {work}", file=sys.stderr, flush=True)
        kinds = len(mats)
        if all((parts / f"{n}-{kk}.npz").exists() for kk in range(kinds)):
            got = [np.load(parts / f"{n}-{kk}.npz") for kk in range(kinds)]
            hidden = spreads is not None
            d2 = np.stack([g["d2"] for g in got]) if hidden else got[0]["d2"]
            m = np.stack([g["deep"] for g in got]) if hidden else got[0]["deep"]
            tmp = out / f"{n}.tmp.npz"
            np.savez(tmp, d1=base["d1"], d2=d2, rows=base["rows"], cols=base["cols"], deep=m,
                     rect=args.rect, childRect=args.child_rect, grand=args.grand,
                     seconds=sum(float(g["seconds"]) for g in got),
                     **({"weights": w} if hidden else {}))
            tmp.replace(out / f"{n}.npz")


def ladder_ref(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915 - the read, its stops, its files
    """A reference read to depth 4 (IKA-369): the base reference's game (``--base``, every
    cell at depth 2) read by the ladder's stages ``--stages`` with no budget -- its root
    matrix the base's, its answer the base's -- on every core (``--threads``) through the
    machine's inference server. A read is stopped at ``--minutes`` and begun again by the
    next call where it stopped: the cells read are kept per position
    (``SET/ref-<name>/parts/<n>.memo.pkl``, `ladder.read`'s ``memo``). Done, it writes each
    stage's root matrix as a reference of its own, ``SET/ref-<name>@<i>/<n>.npz`` for the
    stage ``i`` (from 1), the last also as ``SET/ref-<name>/<n>.npz``, and the stages'
    answers, values and work in ``SET/ref-<name>/parts/<n>.json``."""
    import pickle
    import threading
    from types import SimpleNamespace

    from pokeuraou import ladder
    from pokeuraou.budget import Budget
    from pokeuraou.equilibrium import solve_bayesian
    from pokeuraou.position import Position

    kit = _Kit(args)
    stages = ladder.parse_ladder(args.stages)
    out = Path(args.set) / f"ref-{args.name}"
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.set) / f"ref-{args.base}"
    deadline = time.perf_counter() + args.minutes * 60.0
    stop = threading.Event()
    timer = threading.Timer(max(0.0, args.minutes * 60.0), stop.set)
    timer.daemon = True
    timer.start()
    budget = Budget.matrix()
    for n in _mine(args, len(kit.positions)):
        if (out / f"{n}.npz").exists() or not (base_dir / f"{n}.npz").exists():
            continue
        if stop.is_set() or time.perf_counter() > deadline:
            break
        base = np.load(base_dir / f"{n}.npz")
        pos = Position.from_json(kit.positions[n]["position"])
        spreads = kit.spreads(n)
        ours, theirs = (_from_choices(kit.reg, pos, side, base[key])
                        for side, key in ((0, "rows"), (1, "cols")))
        mats, w = _game(base["d2"], base.get("weights", None))
        items = [ladder.Item(pos)] if spreads is None else list(spreads[1])
        eq = solve_bayesian(mats, w)
        start = SimpleNamespace(row_strategy=eq.row_strategy,
                                col_strategies=list(eq.col_strategies))
        memo_path = parts / f"{n}.memo.pkl"
        held = pickle.loads(memo_path.read_bytes()) if memo_path.exists() else {
            "memo": {}, "seconds": 0.0, "calls": 0, "stages": args.stages}
        # Kept cells of the same stages, or of their first ones (a reference read further: a
        # cell's key names how it was read, so the cells carry over).
        if not (args.stages == held["stages"] or args.stages.startswith(held["stages"] + "+")):
            raise SystemExit(f"position {n}: its kept cells are of the stages {held['stages']!r}")
        held["stages"] = args.stages
        memo = held["memo"]
        had = len(memo)
        began = time.perf_counter()

        def keep(memo=memo, held=held, began=began, memo_path=memo_path) -> None:  # noqa: ANN001
            # The seconds and calls up to this one, and this one's so far.
            tmp = memo_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(dict(held, memo=memo, seconds=held["seconds"] + (
                time.perf_counter() - began), calls=held["calls"] + 1)))
            tmp.replace(memo_path)

        def rung_done(rung, keep=keep, n=n, memo=memo) -> None:  # noqa: ANN001
            keep()
            print(f"ladder-ref {n}: {rung.stage} done, rectangle {rung.rows}x{list(rung.cols)}, "
                  f"fresh {rung.fresh}, value {rung.value:+.4f}, wall {rung.wall_ms / 1000:.1f}s, "
                  f"cells kept {len(memo)}", file=sys.stderr, flush=True)

        try:
            got = ladder.read(kit.reg, 0, ours, theirs, items, mats, w, start, kit.leaf,
                              budget=budget, stages=stages, budget_ms=None, stop=stop,
                              memo=memo, on_rung=rung_done)
        finally:
            keep()
        took = time.perf_counter() - began
        print(f"ladder-ref {n}: {got.stopped} at {got.depth_reached}, cells {had} -> {len(memo)}, "
              f"{took:.1f}s this call, {held['seconds'] + took:.1f}s in all, work {got.work}",
              file=sys.stderr, flush=True)
        if got.stopped != "done":
            break
        # Each stage's root matrix: the stages up to it read again from the kept cells (no
        # cell is read; the LPs give the same rectangles).
        rows_json = []
        hidden = spreads is not None
        for i in range(1, len(stages) + 1):
            again = got if i == len(stages) else ladder.read(
                kit.reg, 0, ours, theirs, items, mats, w, start, kit.leaf, budget=budget,
                stages=stages[:i], budget_ms=None, memo=dict(memo))
            if len(again.rungs) != i:
                raise SystemExit(f"position {n}: stage {i} did not complete from the kept cells")
            prices = again.prices
            stage_dir = Path(args.set) / f"ref-{args.name}@{i}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            payload = {"d1": base["d1"], "d2": np.stack(prices) if hidden else prices[0],
                       "rows": base["rows"], "cols": base["cols"],
                       **({"weights": base["weights"]} if hidden else {})}
            np.savez(stage_dir / f"{n}.npz", **payload)
            rung = again.rungs[-1]
            rows_json.append(rung.to_json() | {"x": [round(float(v), 6) for v in rung.strategy]})
        with_all = dict(payload, stages=args.stages, seconds=held["seconds"] + took)
        tmp = out / f"{n}.tmp.npz"
        np.savez(tmp, **with_all)
        tmp.replace(out / f"{n}.npz")
        _write(parts / f"{n}.json", {"n": n, "stages": args.stages, "rungs": rows_json,
                                     "seconds": round(held["seconds"] + took, 1),
                                     "cells": len(memo), "work": got.work,
                                     # Cells given up by hand (kept at their price: a cell
                                     # no worker finished inside a call), if any.
                                     "gaveUp": [list(map(str, k)) for k in held.get("gaveUp", [])]})
    timer.cancel()


def _game(matrix, weights=None) -> tuple[list[np.ndarray], np.ndarray]:  # noqa: ANN001
    """A reference game: an open position's (R, C) matrix, or a hidden one's (K, R, C) with
    the completions' weights (IKA-367)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 2:
        return [m.copy()], np.ones(1)
    w = np.asarray(weights, dtype=np.float64)
    return [mk.copy() for mk in m], w / w.sum()


def _best_got(x: np.ndarray, matrix, weights=None) -> tuple[float, float]:  # noqa: ANN001
    """The game's value (Bayesian: side 1 knows its bench) and what ``x`` guarantees in it."""
    from pokeuraou.equilibrium import solve, solve_bayesian

    mats, w = _game(matrix, weights)
    best = float(solve(mats[0]).value) if len(mats) == 1 else float(solve_bayesian(mats, w).value)
    got = float(sum(wk * float((x @ m).min()) for wk, m in zip(w, mats, strict=True)))
    return best, got


def _score(x: np.ndarray, claimed: float, refs: dict) -> dict:
    """``refs``: label -> (matrix, weights or None)."""
    out = {}
    for label, (matrix, weights) in refs.items():
        best, got = _best_got(x, matrix, weights)
        out[label] = {"best": best, "got": got, "loss": best - got, "curse": claimed - got}
    return out


def sweep(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915 - the readings of a condition
    """Each condition at each budget, read as a person's game reads it (the module's docstring)."""
    from dataclasses import replace

    from pokeuraou import humanplay, timematch
    from pokeuraou.budget import Budget
    from pokeuraou.deepen import COSTS, cells_for_seconds
    from pokeuraou.ladder import LADDER_COSTS, parse_ladder
    from pokeuraou.position import Position

    kit = _Kit(args)
    budgets = [float(s) for s in args.seconds.split(",")]
    # The budget is the sweep's; a condition names the rest (the node clock by default).
    def spelled(arm: str) -> str:
        name, _sep, rest = arm.partition(":")
        keys = [k for k in rest.split(",") if k]
        keys += [] if any(k.startswith("seconds=") for k in keys) else ["seconds=1"]
        keys += [] if any(k.startswith("clock=") for k in keys) else ["clock=count"]
        return f"{name}:{','.join(keys)}"

    conds = [timematch.parse_condition(spelled(a)) for a in args.arm]
    price = humanplay.node_time(1)
    refs_dirs = sorted(d for d in Path(args.set).glob("ref-*") if d.is_dir())
    for n in _mine(args, len(kit.positions)):
        pos = Position.from_json(kit.positions[n]["position"])
        found = {d.name[4:]: np.load(d / f"{n}.npz") for d in refs_dirs
                 if (d / f"{n}.npz").exists()}
        if not found:
            continue
        ref = found["r24"] if "r24" in found else next(iter(found.values()))
        spreads = kit.spreads(n)
        exact = spreads is None
        classes = 1 if exact else len(spreads[1])
        full = kit.menus(pos, spreads)
        if [a.to_choice() for a in full[0]] != list(ref["rows"]):
            raise SystemExit(f"position {n}: the menus are not the reference's (another leaf or Q?)")
        index = {c: i for i, c in enumerate(ref["rows"])}
        counts = (humanplay.legal_count(kit.reg, pos, 0), humanplay.legal_count(kit.reg, pos, 1),
                  classes)
        for base in conds:
            if base.clock != "count" and base.ladder is None:
                raise SystemExit("a sweep reads on the node clock (clock=count); a ladder may "
                                 "read on the wall clock (IKA-364)")
            for seconds in budgets:
                cond = replace(base, seconds=seconds)
                name = f"{cond.name}@{seconds:g}"
                path = Path(args.set) / "sweep" / name / f"{n}.json"
                if path.exists():
                    continue
                plan = humanplay.plan_move(seconds, 1, *counts, width_only=cond.width_only,
                                           width=cond.width)
                width = min(plan.width, kit.width)
                ours, theirs, outside = (full if width >= kit.width
                                         else kit.menus_at(pos, width, spreads))
                node_ms = price.ms(len(ours) * len(theirs) * classes)
                budget = (replace(Budget.matrix(), enumerate_knockouts=True) if cond.knockouts
                          else Budget.matrix())
                extra: dict = {}
                began = time.perf_counter()
                if cond.ladder is not None:
                    # The node clock, or the wall clock from here (the depth-1 node the
                    # ladder builds included, as a person's move counts it: IKA-364).
                    clocked = ({"clock": "count", "start_ms": node_ms} if cond.clock == "count"
                               else {"clock": "wall", "start_ms": 0.0, "began": began})
                    solved = humanplay.solve_move(
                        kit.reg, pos, 0, list(ours), list(theirs), spreads, kit.leaf,
                        budget=budget, exact=exact, ladder={"stages": parse_ladder(cond.ladder),
                                            "budget_ms": seconds * 1000.0,
                                            "cost": LADDER_COSTS["local", 1], **clocked})
                    extra["ladder"] = solved.ladder.to_json()
                    extra["nodeWorkMs"] = round(LADDER_COSTS["local", 1].ms(
                        solved.ladder.work), 1)
                elif cond.depth2_auto:
                    d2k = humanplay.depth2_children(seconds * 1000.0 - node_ms, classes, price)
                    solved = humanplay.solve_move(
                        kit.reg, pos, 0, list(ours), list(theirs), spreads, kit.leaf,
                        budget=budget, exact=exact, child_q=d2k if d2k is not None else None,
                        sub_limit=cond.sub_limit, sub_branches=cond.sub_branches,
                        depth=2 if d2k is not None else 1, refine=cond.refine, passes=cond.passes)
                    extra["depth2Children"] = d2k
                else:
                    cells = (0 if cond.width_only or plan.deepen_ms <= 0
                             else cells_for_seconds(plan.deepen_ms / 1000.0, 1))
                    solved = humanplay.solve_move(
                        kit.reg, pos, 0, list(ours), list(theirs), spreads, kit.leaf,
                        budget=budget, exact=exact, cells=cells,
                        cost=COSTS["local", 1] if cells else None,
                        levels=cond.max_levels if cells else None,
                        child_q=cond.child_q if cells else None,
                        outside=(outside if (cells and cond.oracle is not None
                                             and not cond.restricted) else None),
                        sub_limit=cond.sub_limit, sub_branches=cond.sub_branches,
                        restricted=cond.restricted, depth=cond.depth, refine=cond.refine,
                        passes=cond.passes)
                    if solved.deepened is not None:
                        extra["depth"] = solved.deepened.depth
                        extra["expanded"] = solved.deepened.expanded
                took = time.perf_counter() - began

                def mapped(strategy, actions=solved.ours, index=index):  # noqa: ANN001, ANN202
                    x = np.zeros(len(index), dtype=np.float64)
                    lost = 0.0
                    for a, p in zip(actions, strategy, strict=True):
                        k = index.get(a.to_choice())
                        if k is None:
                            lost += float(p)
                        else:
                            x[k] += float(p)
                    return (x / x.sum() if x.sum() > 0 else x), lost

                x, outside_mass = mapped(solved.strategy)
                refs = {"d1": (ref["d1"], ref.get("weights", None))} | {
                    f"d2-{k}": (r["d2"], r.get("weights", None))
                    for k, r in found.items()}
                if cond.ladder is not None:
                    # Each completed stage scored as the answer it was: the read's own curve.
                    # The depth-1 answer first, as a stage that spent the node.
                    x0, v0 = solved.ladder.start
                    extra["rungScores"] = [
                        {"stage": "d1", "spentMs": round(node_ms, 1), "wallMs": 0.0,
                         "x": [round(float(v), 6) for v in mapped(x0)[0]], "claimed": v0}
                        | _score(mapped(x0)[0], v0, refs)
                    ] + [
                        {"stage": g.stage, "spentMs": round(g.spent_ms, 1),
                         "wallMs": round(g.wall_ms, 1),
                         "x": [round(float(v), 6) for v in mapped(g.strategy)[0]],
                         "claimed": g.value}
                        | _score(mapped(g.strategy)[0], g.value, refs)
                        for g in solved.ladder.rungs]
                row = {"n": n, "arm": cond.name, "budget": seconds, "wall": round(took, 3),
                       "nodeMs": round(node_ms, 2), "width": width, "classes": classes,
                       "claimed": solved.value,
                       "outsideMass": outside_mass, "x": [round(float(v), 6) for v in x],
                       **extra} | _score(x, solved.value, refs)
                _write(path, row)
                if (args.keep_matrix and cond.ladder is not None and width >= kit.width
                        and seconds == max(budgets)):
                    keep = Path(args.set) / f"ref-{name}"
                    keep.mkdir(parents=True, exist_ok=True)
                    prices = solved.ladder.prices
                    np.savez(keep / f"{n}.npz", d1=ref["d1"],
                             d2=prices[0] if exact else np.stack(prices),
                             rows=ref["rows"], cols=ref["cols"],
                             **({} if exact else {"weights": ref["weights"]}))
                print(json.dumps({"n": n, "arm": name, "wall": row["wall"]}
                                 | {k: round(v["loss"], 4) for k, v in row.items()
                                    if isinstance(v, dict) and "loss" in v}),
                      file=sys.stderr, flush=True)


def _sweep_rows(set_dir: Path) -> dict:
    """{arm: {budget: {n: row}}}, the scores refreshed against every reference now present."""

    refs = sorted(d for d in set_dir.glob("ref-*") if d.is_dir())
    out: dict = {}
    for f in sorted((set_dir / "sweep").glob("*/*.json")):
        row = json.loads(f.read_bytes())
        n = row["n"]
        x = np.asarray(row["x"], dtype=np.float64)
        for d in refs:
            key = f"d2-{d.name[4:]}"
            if not (d / f"{n}.npz").exists():
                continue
            scored = [(row, x)] + [(g, np.asarray(g["x"], dtype=np.float64))
                                   for g in row.get("rungScores", [])]
            if all(key in r for r, _x in scored):
                continue
            npz = np.load(d / f"{n}.npz")
            game = (npz["d2"], npz.get("weights", None))
            for r, xr in scored:
                b, got = _best_got(xr, *game)
                r[key] = {"best": b, "got": got, "loss": b - got, "curse": r["claimed"] - got}
        out.setdefault(row["arm"], {}).setdefault(row["budget"], {})[n] = row
    return out


def _svg(series: dict, key: str, path: Path) -> None:
    """One line per condition: mean loss against ``key`` over budgets (log2 axis)."""
    width, height, pad = 640, 400, 56
    xs = sorted({b for s in series.values() for b in s})
    ys = [v for s in series.values() for v in s.values()]
    if not xs or not ys:
        return
    lx = np.log2(np.asarray(xs))
    x0, x1 = float(lx.min()), float(lx.max()) if lx.max() > lx.min() else float(lx.min()) + 1
    y0, y1 = 0.0, max(ys) * 1.1

    def px(b: float) -> float:
        return pad + (np.log2(b) - x0) / (x1 - x0) * (width - 2 * pad)

    def py(v: float) -> float:
        return height - pad - (v - y0) / (y1 - y0) * (height - 2 * pad)

    colors = ["#2a6fdb", "#d9534f", "#2e9e5b", "#a05bd6", "#e08a1e", "#555555"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'font-family="sans-serif" font-size="12"><rect width="100%" height="100%" '
             f'fill="white"/>',
             f'<text x="{pad}" y="20">loss against {key} (mean over positions), node-clock '
             f'seconds on a log2 axis</text>',
             f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
             f'stroke="black"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="black"/>']
    for b in xs:
        parts.append(f'<text x="{px(b):.1f}" y="{height - pad + 16}" text-anchor="middle">'
                     f'{b:g}</text>')
    for t in np.linspace(y0, y1, 5):
        parts.append(f'<text x="{pad - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{t:.3f}</text>')
    for c, (name, s) in enumerate(series.items()):
        color = colors[c % len(colors)]
        pts = " ".join(f"{px(b):.1f},{py(v):.1f}" for b, v in sorted(s.items()))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        for b, v in s.items():
            parts.append(f'<circle cx="{px(b):.1f}" cy="{py(v):.1f}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{width - pad + 4}" y="{pad + 16 * c}" fill="{color}">{name}</text>')
    parts.append("</svg>\n")
    path.write_bytes("\n".join(parts).encode("utf-8"))


def curve(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912 - one table per reference
    """Loss and curse by budget, per condition (the module's docstring)."""
    arms = _sweep_rows(Path(args.set))
    for spec in args.virtual or []:
        # "lad@64=1,4,16": the answers a read at 64 s had at each budget -- its last stage
        # completed within it (a read given that budget may skip a stage it predicts will
        # not fit): a condition "lad~" at those budgets.
        source, _eq, at = spec.partition("=")
        name, _at, s = source.partition("@")
        rows = arms[name][float(s)]
        for b in (float(v) for v in at.split(",")):
            for n, r in rows.items():
                done = [g for g in r.get("rungScores", []) if g["spentMs"] <= b * 1000.0]
                if not done:
                    continue
                g = done[-1]
                arms.setdefault(f"{name}~", {}).setdefault(b, {})[n] = {
                    **{k: v for k, v in g.items()}, "wall": g["wallMs"] / 1000.0, "stage": g["stage"]}
    if args.arms:
        arms = {a: arms[a] for a in args.arms.split(",") if a in arms}
    ns = None
    for budgets in arms.values():
        for rows in budgets.values():
            ns = set(rows) if ns is None else ns & set(rows)
    ns = sorted(ns or [])
    print(f"{len(ns)} positions read by every condition at every budget")
    keys = sorted({k for b in arms.values() for rows in b.values() for r in rows.values()
                   for k in r if isinstance(r[k], dict) and "loss" in r[k]})
    # A reference need not cover every position (the long ones were cut at a time): each
    # is read on the positions it has for every condition.
    key_ns = {k: [n for n in ns if all(k in rows[n] for b in arms.values()
                                       for rows in b.values())] for k in keys}
    keys = [k for k in keys if len(key_ns[k]) >= 10]
    for key in keys:
        kn = key_ns[key]
        print(f"\nagainst {key} ({len(kn)} positions):")
        print(f"  {'condition':<14}{'s':>6}{'wall s':>9}{'loss':>9}{'se':>8}{'curse':>9}"
              f"{'change':>10}{'se':>8}{'worse':>7}")
        series: dict = {}
        for arm, budgets in arms.items():
            prev = None
            for b in sorted(budgets):
                rows = budgets[b]
                loss = np.array([rows[n][key]["loss"] for n in kn])
                curse = np.array([rows[n][key]["curse"] for n in kn])
                wall = np.mean([rows[n]["wall"] for n in kn])
                change = se_c = worse = ""
                if prev is not None:
                    d = loss - prev
                    change = f"{d.mean():+.4f}"
                    se_c = f"{d.std() / np.sqrt(len(kn)):.4f}"
                    worse = f"{int((d > 1e-4).sum())}"
                print(f"  {arm:<14}{b:>6g}{wall:>9.2f}{loss.mean():>9.4f}"
                      f"{loss.std() / np.sqrt(len(kn)):>8.4f}{curse.mean():>+9.4f}"
                      f"{change:>10}{se_c:>8}{worse:>7}")
                series.setdefault(arm, {})[b] = float(loss.mean())
                prev = loss
        if args.svg:
            svg = Path(args.svg)
            _svg(series, key, svg.with_name(f"{svg.stem}-{key}{svg.suffix}"))
        for pair in args.pairs or []:
            # "A:B": A's loss minus B's at every budget both read (negative: A better).
            a, b = pair.split(":")
            for budget in sorted(set(arms[a]) & set(arms[b])):
                d = np.array([arms[a][budget][n][key]["loss"] - arms[b][budget][n][key]["loss"]
                              for n in kn])
                print(f"  {a} - {b} at {budget:g} s: {d.mean():+.4f} (se "
                      f"{d.std() / np.sqrt(len(kn)):.4f}), {a} worse on {int((d > 1e-4).sum())}, "
                      f"better on {int((d < -1e-4).sum())} of {len(kn)}")
    if args.stages:
        # One ladder read's own curve: each stage's answer, where it completed.
        name, _at, budget = args.stages.partition("@")
        rows = arms[name][float(budget)]
        for key in keys:
            print(f"\nstages of {args.stages} against {key} (positions that completed the stage; "
                  "the change and its cost are against the stage before, on those positions):")
            print(f"  {'stage':<22}{'done':>5}{'loss':>9}{'curse':>9}{'spent s':>9}{'wall s':>8}"
                  f"{'change':>9}{'se':>8}{'per s':>9}")
            labels = []
            for r in rows.values():
                for g in r.get("rungScores", []):
                    if g["stage"] not in labels:
                        labels.append(g["stage"])
            prev_label = None
            for label in labels:
                have = {n: next(g for g in r["rungScores"] if g["stage"] == label)
                        for n, r in rows.items()
                        if any(g["stage"] == label and key in g for g in r["rungScores"])}
                if not have:
                    continue
                loss = np.array([g[key]["loss"] for g in have.values()])
                curse = np.array([g[key]["curse"] for g in have.values()])
                spent = np.array([g["spentMs"] for g in have.values()]) / 1000
                wall = np.array([g["wallMs"] for g in have.values()]) / 1000
                change = se = per = ""
                if prev_label is not None:
                    before = {n: next(g for g in rows[n]["rungScores"] if g["stage"] == prev_label)
                              for n in have}
                    d = np.array([have[n][key]["loss"] - before[n][key]["loss"] for n in have])
                    cost = np.array([have[n]["spentMs"] - before[n]["spentMs"] for n in have]) / 1000
                    change, se = f"{d.mean():+.4f}", f"{d.std() / np.sqrt(len(d)):.4f}"
                    per = f"{d.mean() / max(cost.mean(), 1e-9):+.5f}"
                print(f"  {label:<22}{len(have):>5}{loss.mean():>9.4f}{curse.mean():>+9.4f}"
                      f"{spent.mean():>9.2f}{wall.mean():>8.2f}{change:>9}{se:>8}{per:>9}")
                prev_label = label
    if args.elo:
        # "0.1:0.2=15.1,0.2:0.4=25.5": a condition's loss change between two budgets and
        # the Elo a board gave the same step.
        # Or "A@s:B@s=Elo": B's Elo over A's, two conditions (IKA-362/366's matches).
        print("\nElo scale:")
        total_elo = total_loss = 0.0
        for item in args.elo.split(","):
            span, elo = item.split("=")
            ends = []
            for end in span.split(":"):
                name, _at, s = end.rpartition("@")
                ends.append((name or args.elo_arm, float(s)))
            (la, lo), (ha, hi) = ends
            for key in keys:
                kn = key_ns[key]
                d = np.array([arms[la][lo][n][key]["loss"] - arms[ha][hi][n][key]["loss"]
                              for n in kn])
                print(f"  {la}@{lo:g} -> {ha}@{hi:g}, {float(elo):+.1f} Elo: loss falls "
                      f"{d.mean():+.4f} "
                      f"(se {d.std() / np.sqrt(len(kn)):.4f}) against {key}"
                      + (f"; {float(elo) / d.mean():+.0f} Elo per unit" if d.mean() > 0 else ""))
                if key == args.elo_key:
                    total_elo += float(elo)
                    total_loss += float(d.mean())
        if total_loss > 0:
            print(f"  pooled against {args.elo_key}: {total_elo:+.1f} Elo over a loss fall of "
                  f"{total_loss:.4f}: {total_elo / total_loss:.0f} Elo per unit of loss "
                  f"({total_elo / total_loss / 1000:.2f} Elo per 0.001)")


def fit(args: argparse.Namespace) -> None:
    """Non-negative least squares of the ladder's wall milliseconds on its counted work, a
    row per stage (what it added to the work and the wall) and one for the unfinished rest."""
    from scipy.optimize import nnls

    kinds = ("turns", "subgames", "cells", "qs", "reads")
    arms = _sweep_rows(Path(args.set))
    xs, ys = [], []
    for budgets in arms.values():
        for rows in budgets.values():
            for r in rows.values():
                lad = r.get("ladder")
                if lad is None:
                    continue
                marks = [(g["work"], g["wallMs"]) for g in lad["rungs"] if "work" in g]
                marks.append((lad["work"], lad["wallMs"]))
                prev = ({k: 0 for k in kinds}, 0.0)
                for work, wall in marks:
                    xs.append([work.get(k, 0) - prev[0].get(k, 0) for k in kinds])
                    ys.append(wall - prev[1])
                    prev = (work, wall)
    if not xs:
        raise SystemExit("no ladder readings in the sweep")
    a, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    coef, _res = nnls(a, y)
    pred = a @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"{len(y)} stages: turn {coef[0]:.3f} ms, subgame {coef[1]:.3f} ms, cell "
          f"{coef[2]:.5f} ms, q {coef[3]:.3f} ms, read {coef[4]:.3f} ms; R^2 {r2:.3f}; "
          f"wall/predicted total {y.sum() / pred.sum():.3f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--games-dir", type=Path, required=True)
    b.add_argument("--count", type=int, default=300)
    b.add_argument("--seed", type=int, default=36200)
    b.add_argument("--width", type=int, default=64)
    b.add_argument("--out", type=Path, required=True)
    for name in ("reference", "reference3", "evaluate", "deep", "sweep", "ladder-ref"):
        s = sub.add_parser(name)
        s.add_argument("--set", type=Path, required=True)
        s.add_argument("--from", dest="start", type=int, default=0)
        s.add_argument("--to", dest="stop", type=int, default=10**9)
        s.add_argument("--stride", type=int, default=1, help="every k-th position (IKA-367)")
        s.add_argument("--offset", type=int, default=0, help="from the start plus this")
        s.add_argument("--claim", default=None,
                       help="a directory of claims: processes pull units from one queue")
        s.add_argument("--pool", default="regmc-matchupweb")
        s.add_argument("--value", type=Path, nargs="+", default=None)
        s.add_argument("--q-model", type=Path, default=None)
        s.add_argument("--device", default=None)
        s.add_argument("--cuda-memory-gb", type=float, default=0.8)
        s.add_argument("--inference", default=None, metavar="HOST:PORT",
                       help="the machine's inference server (IKA-363; arms value and q). "
                       "Default: POKEURAOU_INFERENCE, else the leaf is loaded here")
        s.add_argument("--ladder-inference", default=None, metavar="HOST:PORT[,HOST:PORT...]",
                       help="IKA-390: the servers the ladder's worker processes ask, worker i "
                       "the (i mod n)th (the same models as --inference; this process keeps "
                       "asking --inference). Default: all ask --inference")
        s.add_argument("--merge", default="off", choices=("on", "off"),
                       help="the server's merged road: the values move in the last places "
                       "with the timing (off: the answers a local leaf gives)")
        if name == "reference3":
            s.add_argument("--name", required=True)
            s.add_argument("--base", required=True, help="the depth-2 reference it deepens")
            s.add_argument("--rect", type=int, default=6)
            s.add_argument("--sub-limit", type=int, default=24)
        if name == "reference":
            s.add_argument("--name", required=True, help="the reference's name: SET/ref-<name>")
            s.add_argument("--menus-from", default=None, metavar="NAME",
                           help="IKA-394: the rows and columns of SET/ref-<NAME> instead of "
                           "this leaf's own menus (another leaf's game on the same actions)")
            s.add_argument("--sub-limit", type=int, default=24)
            s.add_argument("--sub-branches", type=int, default=64,
                           help="branches kept per refined cell (64: all, in practice)")
        elif name == "evaluate":
            s.add_argument("--arm", required=True, help="a time-match condition on the node clock")
        elif name == "deep":
            s.add_argument("--name", default="deep")
            s.add_argument("--base", default="r24", help="the every-cell depth-2 reference")
            s.add_argument("--rect", type=int, default=8)
            s.add_argument("--child-rect", type=int, default=6)
            s.add_argument("--grand", type=int, default=24,
                           help="the grandchildren's menus: the Q's k best")
        elif name == "sweep":
            s.add_argument("--arm", action="append", required=True,
                           help="a time-match condition (its seconds are the sweep's)")
            s.add_argument("--seconds", default="1,4,16,64")
            s.add_argument("--threads", type=int, default=1,
                           help="a ladder's cells on threads - 1 worker processes (IKA-364; "
                           "with --inference each asks the server). 1: read here")
            s.add_argument("--keep-matrix", action="store_true",
                           help="a ladder's root prices at the longest budget kept as a "
                                "reference ref-<arm>@<s>")
        elif name == "ladder-ref":
            s.add_argument("--name", required=True, help="the reference's name: SET/ref-<name>")
            s.add_argument("--base", default="r24", help="the every-cell depth-2 reference")
            s.add_argument("--stages", required=True,
                           help="a ladder: a name (ladder.LADDERS) or stages joined by +")
            s.add_argument("--minutes", type=float, default=25.0,
                           help="stop the read here; the next call goes on from its cells")
            s.add_argument("--threads", type=int, default=1,
                           help="the cells on threads - 1 worker processes (IKA-364)")
    bh = sub.add_parser("build-hidden")
    bh.add_argument("--games-dir", type=Path, required=True)
    bh.add_argument("--pool", default="regmc-matchupweb")
    bh.add_argument("--count", type=int, default=40)
    bh.add_argument("--seed", type=int, default=36700)
    bh.add_argument("--width", type=int, default=36)
    bh.add_argument("--min-turn", type=int, default=2)
    bh.add_argument("--max-completions", type=int, default=6)
    bh.add_argument("--out", type=Path, required=True)
    sb = sub.add_parser("subset")
    sb.add_argument("--set", type=Path, required=True)
    sb.add_argument("--out", type=Path, required=True)
    sb.add_argument("--count", type=int, default=40)
    c = sub.add_parser("curve")
    c.add_argument("--set", type=Path, required=True)
    c.add_argument("--arms", default=None, help="comma-separated conditions (default: all)")
    c.add_argument("--svg", default=None, help="figure path; one per reference")
    c.add_argument("--stages", default=None, help="ARM@SECONDS: a ladder read's stages")
    c.add_argument("--pairs", action="append", default=None,
                   help="A:B: A's loss minus B's at each budget both read, paired")
    c.add_argument("--virtual", action="append", default=None,
                   help="ARM@S=B1,B2,...: a ladder read's answers at smaller budgets (ARM~)")
    c.add_argument("--elo", default=None, help="lo:hi=Elo,... steps of a recorded curve")
    c.add_argument("--elo-arm", default=None)
    c.add_argument("--elo-key", default="d2-deep")
    f = sub.add_parser("fit")
    f.add_argument("--set", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--set", type=Path, required=True)
    rs = sub.add_parser("rescore")
    rs.add_argument("--set", type=Path, required=True)
    args = ap.parse_args(argv)
    {"build": build, "reference": reference, "reference3": reference3, "evaluate": evaluate,
     "rescore": rescore, "report": report, "subset": subset, "deep": deep, "sweep": sweep,
     "build-hidden": build_hidden, "ladder-ref": ladder_ref,
     "curve": curve, "fit": fit}[args.cmd](args)


if __name__ == "__main__":
    main()
