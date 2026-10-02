"""Does a deeper reading's value predict the game's result better than a shallower one? (IKA-421)

Every earlier measure of the reading (IKA-362 to IKA-418) scored an answer against a deeper
reading by the *same* evaluation model, so the model's own error never entered. This tool
scores the values against the recorded games' results instead.

    python tools/depth_outcome.py build --games-dir DATA/selfplay-mc4 \
        --merged DATA/selfplay-mc01234-encoded.npz \
        --shards DATA/selfplay-mc0-encoded.npz ... DATA/selfplay-mc4-encoded.npz \
        --store STORE --count 6000 --out SET
    python tools/depth_outcome.py read --set SET --name x --seconds 8 --value V0 V1 --q-model Q \
        --inference HOST:PORT --claim SET/claims-x
    python tools/depth_outcome.py report --set SET --name x --budgets 1,4,8 --keys d0/d1/d2,d1/d2

* **build**: the held-out games of one generation (the value model's own split: the merged
  dataset's games, `value.split_for` at ``--split-seed``/``--holdout``; a generation's games
  are its shard's ids offset as `value.concat_datasets` offsets them), ``--count`` of them at
  random, one move decision of each at random (every turn). Each position keeps the game's
  result (side 0's score), side 1's bench as side 0 sees it (the shown slots), and the weights
  M-C generation gave side 0's belief there (`belief_weights.production_weights`: the
  selection store the games were played from, the shown Pokemon and the leads). Written to
  ``SET/positions.jsonl`` (one line a position).
* **read**: each position read by side 0 with ladder ``--ladder`` (L6) on the node clock at
  ``--seconds`` (`POKEURAOU_LADDER_COUNT_FILL`: the stages fill the budget, so the answer at
  any smaller budget is the last stage completed inside it, IKA-384/418), on the menus at
  ``--width`` (q-nocover, as the agent builds them). Twice where side 1's bench is hidden from
  side 0: the open game (the true position) and the Bayesian game over side 1's completions
  at the production weights. Each read keeps the depth-0 value (the leaf on the position, or
  the completions' weighted leaf), the depth-1 node's value and every stage's value with its
  counted time. ``--d1-only`` stops after the depth-1 node (a cheap read for a weak model).
  Written to ``SET/read-<name>/<n>.json``. IKA-421 stage 2: each answer's most played row
  (``top``) and its distance from the depth-1 answer (``tv``); with ``POKEURAOU_LADDER_DIAG=1``
  each stage's other values (`ladder.DIAG`); ``--side 1`` / ``--swap`` read the open game from
  the other seat (the mirror), every value still in the recorded side 0's units.
* **report**: per value (depth 0, depth 1, the last stage of depth 2 / 3 completed within
  each of ``--budgets``), its log loss, Brier score and calibration against the results, the
  paired differences with standard errors (one position a game, so the positions are the
  games), and the same split by the position's kind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("POKEURAOU_LADDER_COUNT_FILL", "1")  # before `ladder` is imported

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()  # before anything loads torch (IKA-360)

import numpy as np  # noqa: E402

DEFAULT_POOL = "data/pool/regmc-matchupweb.json"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(payload) + "\n").encode("utf-8"))
    tmp.replace(path)


# ------------------------------------------------------------------------------ build


def held_out_games(merged: Path, shards: list[Path], gen: int, holdout: float,
                   split_seed: int) -> tuple[np.ndarray, int, int]:
    """Generation ``gen``'s held-out game ids in its own shard's numbering, the number of
    rows of those games in the merged set, and the number of the shard's games."""
    from pokeuraou.value import Dataset

    game = np.load(merged)["game"]
    offset = 0
    for shard in shards[:gen]:
        g = np.load(shard)["game"]
        offset += int(g.max()) + 1 if len(g) else 0
    own = np.load(shards[gen])["game"]
    count = int(own.max()) + 1
    # `Dataset.split_by_game` on the merged game ids: the split the model was trained on.
    probe = Dataset.__new__(Dataset)
    probe.game = game
    _train, val = probe.split_by_game(holdout, split_seed)
    vg = np.unique(game[val])
    mine = vg[(vg >= offset) & (vg < offset + count)] - offset
    rows = int(np.isin(game[val], mine + offset).sum())
    return mine.astype(np.int64), rows, count


def build(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915 - one pass over the games
    import belief_weights as bw

    from pokeuraou.damage import register_mega_stones
    from pokeuraou.hidden import completions, seen_slots
    from pokeuraou.pool import load_pool
    from pokeuraou.position import Position

    shards = [Path(s) for s in args.shards]
    val, rows, count = held_out_games(Path(args.merged), shards, args.gen, args.holdout,
                                      args.split_seed)
    print(f"generation {args.gen}: {len(val)} held-out games of {count} ({rows} rows)",
          file=sys.stderr, flush=True)
    rng = np.random.default_rng(args.seed)
    picked = set(rng.choice(val, size=min(args.count, len(val)), replace=False).tolist())
    shard = np.load(shards[args.gen])
    meta = json.loads(str(shard["meta_json"]))
    sources = [s[0] for s in meta["sources"]]
    rows_per_game = np.bincount(shard["game"], minlength=count)
    outcome_of = np.zeros(count)
    outcome_of[shard["game"]] = shard["outcome"]
    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    store = Path(args.store)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    gid = 0
    checked = 0
    for name in sources:
        with (Path(args.games_dir) / name).open("rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                here = gid
                gid += 1
                if here not in picked:
                    continue
                game = json.loads(line)
                # The shard's numbering is the files' order: the game's rows and result agree.
                if len(game["decisions"]) != rows_per_game[here] or float(
                        game["outcome"]) != float(outcome_of[here]):
                    raise SystemExit(f"{name}: game {here} is not the shard's game {here} "
                                     f"({len(game['decisions'])} decisions, {rows_per_game[here]} rows)")
                checked += 1
                moves = [d for d in game["decisions"] if d["kind"] == "move"]
                if not moves:
                    continue
                d = moves[int(rng.integers(len(moves)))]
                named = game["pool"]
                teams = [next(t for t in pool.teams if t.id == named["teams"][side]) for side in (0, 1)]
                pos = Position.from_json(d["position"])
                shown = d["shownIdentities"]
                seen1 = sorted(seen_slots(pos, 1, shown[1]))
                seen0 = sorted(seen_slots(pos, 0, shown[0]))
                comp = completions(reg, pos, 1, list(teams[1].sets), seen=frozenset(seen1))
                keys = [tuple(sorted(c.species)) for c in comp]
                weights: dict = {}
                explains = None
                if len(comp) >= 2:
                    try:
                        w, info = bw.production_weights(pool, store, game, d, pos, keys,
                                                        epsilon=args.epsilon,
                                                        temperature=args.temperature)
                        explains = bool(info.get("explains"))
                    except ValueError:
                        # The tool's rebuilt leads miss a form's spelling (a Mega of a
                        # regional / event form): uniform, said in the position.
                        w = np.full(len(keys), 1.0 / len(keys))
                        explains = "error"
                    weights = {",".join(k): float(v) for k, v in zip(keys, w, strict=True)}
                lines.append(json.dumps({
                    "game": here, "source": name, "gameIndex": game.get("gameIndex"),
                    "turn": d["turn"], "turns": game["turns"], "outcome": float(game["outcome"]),
                    "teams": named["teams"], "seen": [seen0, seen1], "classes": len(comp),
                    "weights": weights, "explains": explains,
                    "searchValue": d.get("searchValue"), "foeSearchValue": d.get("foeSearchValue"),
                    "ownChosen": d.get("ownChosen"), "foeChosen": d.get("foeChosen"),
                    "position": d["position"]}))
    if gid != count:
        raise SystemExit(f"the files hold {gid} games, the shard {count}")
    order = np.random.default_rng(args.seed + 1).permutation(len(lines))
    (out / "positions.jsonl").write_bytes(("\n".join(lines[i] for i in order) + "\n").encode("utf-8"))
    _write(out / "build.json", {"gen": args.gen, "heldOutGames": len(val), "heldOutRows": rows,
                                "shardGames": count, "picked": len(picked), "written": len(lines),
                                "checked": checked, "seed": args.seed, "split_seed": args.split_seed,
                                "holdout": args.holdout, "store": str(args.store),
                                "gamesDir": str(args.games_dir)})
    print(f"{len(lines)} positions ({checked} games checked against the shard) -> "
          f"{out / 'positions.jsonl'}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------- read


class Kit:
    def __init__(self, args: argparse.Namespace) -> None:
        from pokeuraou import humanplay
        from pokeuraou.damage import register_mega_stones
        from pokeuraou.pool import load_pool

        pool = load_pool(args.pool)
        self.pool = pool
        self.reg = pool.reg
        register_mega_stones(self.reg)
        address = humanplay.inference_address(args.inference)
        if not address:
            raise SystemExit("read asks for the machine's inference server (--inference)")
        self.leaf, _encoder = humanplay.served_leaf(self.reg, address, args.value,
                                                    q_path=Path(args.q_model))
        humanplay.use_threads(1, self.reg, None)
        self.positions = [json.loads(line) for line in
                          (Path(args.set) / "positions.jsonl").read_bytes().splitlines() if line.strip()]

    def spreads(self, entry: dict) -> dict | None:
        """Both sides' completions of the other's bench, side 1's at the production weights
        (side 0's as side 1 sees it, uniform: they only rank side 1's menu); None when side 0
        sees all of side 1."""
        from pokeuraou.hidden import completions
        from pokeuraou.position import Position

        if entry["classes"] < 2:
            return None
        pos = Position.from_json(entry["position"])
        teams = [next(t for t in self.pool.teams if t.id == entry["teams"][side]) for side in (0, 1)]
        w = {tuple(k.split(",")): v for k, v in entry["weights"].items()}
        one = completions(self.reg, pos, 1, list(teams[1].sets), seen=frozenset(entry["seen"][1]),
                          weights=w)
        zero = completions(self.reg, pos, 0, list(teams[0].sets), seen=frozenset(entry["seen"][0]))
        return {0: zero, 1: one}


def orientation(side: int, swap: bool) -> tuple[float, float]:
    """(a, b): a read's value v in the recorded side 0's units is a + b v (IKA-421's mirror).
    Side 1 reads the negated transpose (-P); the swapped position's side 0 is the recorded
    side 1 (1 - P)."""
    return {(0, False): (0.0, 1.0), (1, False): (0.0, -1.0),
            (0, True): (1.0, -1.0), (1, True): (1.0, 1.0)}[side, swap]


def _one_read(kit: Kit, pos, spreads, args: argparse.Namespace) -> dict:  # noqa: ANN001
    from pokeuraou import humanplay
    from pokeuraou.budget import Budget
    from pokeuraou.ladder import LADDER_COSTS, parse_ladder
    from pokeuraou.selfplay import _menus

    side, swap = getattr(args, "side", 0), getattr(args, "swap", False)
    exact = spreads is None
    if (side or swap) and not exact:
        raise SystemExit("--side 1 / --swap read the open game only (IKA-421's mirror)")
    if swap:
        pos = pos.swapped()
    # The leaf's value of the position as read (the swapped one's side 0 is side 1).
    leaf_a, leaf_b = (1.0, -1.0) if swap else (0.0, 1.0)
    a, b = orientation(side, swap)
    classes = 1 if exact else len(spreads[1])
    if exact:
        d0 = leaf_a + leaf_b * float(kit.leaf([pos])[0])
    else:
        vals = np.asarray(kit.leaf([c.position for c in spreads[1]]), dtype=np.float64)
        d0 = float(np.dot([c.weight for c in spreads[1]], vals))
    ours, theirs = _menus(kit.reg, pos, (args.width, args.width), kit.leaf, Budget.matrix(), True,
                          None, spreads, rank_fill="q-nocover")
    price = humanplay.node_time(1)
    node_ms = price.ms(len(ours) * len(theirs) * classes)
    began = time.perf_counter()
    stages = parse_ladder("L6@0" if args.d1_only else args.ladder)
    solved = humanplay.solve_move(
        kit.reg, pos, side, list(ours), list(theirs), spreads, kit.leaf, budget=Budget.matrix(),
        exact=exact, ladder={"stages": stages, "budget_ms": args.seconds * 1000.0,
                             "cost": LADDER_COSTS["local", 1], "clock": "count",
                             "start_ms": node_ms})
    x0, v0 = solved.ladder.start
    out = {"d0": d0, "d1": a + b * float(v0), "rows": len(ours), "cols": len(theirs),
           "classes": classes, "nodeMs": round(node_ms, 2),
           "wall": round(time.perf_counter() - began, 3),
           "stopped": solved.ladder.stopped, "unfinished": solved.ladder.unfinished,
           # IKA-421: the row each answer plays most, and how far its mixture is from the
           # depth-1 answer's (total variation: 0 the same, 1 no row in common).
           "top": int(np.argmax(x0)),
           "rungs": [{"stage": g.stage, "value": a + b * float(g.value),
                      "spentMs": round(g.spent_ms, 1), "top": int(np.argmax(g.strategy)),
                      "tv": round(0.5 * float(np.abs(np.asarray(g.strategy) - np.asarray(x0)).sum()), 6),
                      # IKA-421 (`ladder.DIAG`): in the reading side's own units (``orient``).
                      **({"diag": g.diag} if g.diag is not None else {})}
                     for g in solved.ladder.rungs]}
    if side or swap:
        out.update(side=side, swap=swap, orient=[a, b])
    return out


def read(args: argparse.Namespace) -> None:
    import position_set as ps

    from pokeuraou import ladder
    from pokeuraou.position import Position

    kit = Kit(args)
    out = Path(args.set) / f"read-{args.name}"
    args.stride, args.offset = 1, 0
    units = ([int(n) for n in args.only.split(",")] if getattr(args, "only", None)
             else list(range(args.start, min(args.stop, len(kit.positions)))))
    if (args.side or args.swap) and not args.open_only:
        raise SystemExit("--side 1 / --swap go with --open-only (the mirror is the open game's)")
    for n in ps._units(args, units):  # noqa: SLF001 - the claim queue
        path = out / f"{n}.json"
        if path.exists():
            continue
        entry = kit.positions[n]
        pos = Position.from_json(entry["position"])
        spreads = kit.spreads(entry)
        if args.hidden_only and spreads is None:
            continue
        row = {"n": n, "game": entry["game"]}
        if (ladder.VALUE != "guarantee" or ladder.ORACLE_PASSES or ladder.KEEP
                or ladder.CHILD != "guarantee"):
            # IKA-421: the reading's switches, said in the row.
            row["ladder"] = {"value": ladder.VALUE, "passes": ladder.ORACLE_PASSES,
                             "keep": ladder.KEEP, "child": ladder.CHILD}
        if not args.hidden_only:
            row["open"] = _one_read(kit, pos, None, args)
        if spreads is not None and not args.open_only:
            row["hidden"] = _one_read(kit, pos, spreads, args)
        _write(path, row)
        print(json.dumps({"n": n, "open": row.get("open", {}).get("wall"),
                          "hidden": row.get("hidden", {}).get("wall"),
                          "stages": len(row.get("open", row.get("hidden", {})).get("rungs", []))}),
              file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- report


def _depth(stage: str) -> int:
    return int(stage[1])


def values_at(read_row: dict, budget_ms: float) -> dict[str, float]:
    """The values a read held at a counted budget: depth 0, depth 1 and the last stage of
    each depth completed inside it, and the answer (the last completed stage)."""
    got = {"d0": read_row["d0"], "d1": read_row["d1"]}
    last = read_row["d1"]
    for rung in read_row["rungs"]:
        if rung["spentMs"] > budget_ms:
            break
        got[f"d{_depth(rung['stage'])}"] = rung["value"]
        # the first stage of each depth too (``d2a``: depth 2's 4 x 4 at three branches)
        got.setdefault(f"d{_depth(rung['stage'])}a", rung["value"])
        last = rung["value"]
    got["answer"] = last
    return got


PROTECTS = frozenset({"protect", "detect", "spikyshield", "kingsshield", "banefulbunker",
                      "silktrap", "burningbulwark", "obstruct"})


def _chosen_kinds(position: dict, side: int, chosen: str | None) -> set[str]:
    """``switch`` / ``protect`` / ``attack`` in a side's recorded action."""
    out: set[str] = set()
    if not chosen:
        return out
    s = position["sides"][side]
    for slot, part in enumerate(chosen.split(",")):
        words = part.split()
        if not words:
            continue
        if words[0] == "switch":
            out.add("switch")
        elif words[0] == "move" and slot < len(s["active"]) and s["active"][slot] is not None:
            mon = s["pokemon"][s["active"][slot]]
            k = int(words[1]) - 1
            move = mon["moves"][k]["id"] if 0 <= k < len(mon["moves"]) else ""
            out.add("protect" if move in PROTECTS else "attack")
    return out


def features(entry: dict) -> dict:
    p = entry["position"]
    alive, hp = [], []
    for side in p["sides"]:
        mons = side["pokemon"]
        alive.append(sum(1 for m in mons if not m["fainted"]))
        hp.append(sum(m["hp"] / m["maxhp"] for m in mons if not m["fainted"]))
    kinds = (_chosen_kinds(p, 0, entry.get("ownChosen")) | _chosen_kinds(p, 1, entry.get("foeChosen")))
    return {"turn": entry["turn"], "alive": alive, "hp": hp, "classes": entry["classes"],
            "switch": "switch" in kinds, "protect": "protect" in kinds,
            "left": entry["turns"] - entry["turn"]}


#: The position kinds the report splits by: name -> (label, predicate on features).
SPLITS: dict[str, list[tuple[str, Any]]] = {
    "turn": [("1", lambda f: f["turn"] == 1), ("2-3", lambda f: 2 <= f["turn"] <= 3),
             ("4-6", lambda f: 4 <= f["turn"] <= 6), ("7+", lambda f: f["turn"] >= 7)],
    "alive": [("8", lambda f: sum(f["alive"]) == 8), ("6-7", lambda f: 6 <= sum(f["alive"]) <= 7),
              ("4-5", lambda f: 4 <= sum(f["alive"]) <= 5), ("2-3", lambda f: sum(f["alive"]) <= 3)],
    "classes": [("1", lambda f: f["classes"] == 1), ("3-4", lambda f: f["classes"] in (3, 4)),
                ("6+", lambda f: f["classes"] >= 6)],
    "hpgap": [("<0.5", lambda f: abs(f["hp"][0] - f["hp"][1]) < 0.5),
              ("0.5-1.5", lambda f: 0.5 <= abs(f["hp"][0] - f["hp"][1]) < 1.5),
              (">=1.5", lambda f: abs(f["hp"][0] - f["hp"][1]) >= 1.5)],
    "played": [("protect", lambda f: f["protect"]), ("switch", lambda f: f["switch"]),
               ("neither", lambda f: not f["protect"] and not f["switch"])],
    "turnsleft": [("0", lambda f: f["left"] == 0), ("1-2", lambda f: 1 <= f["left"] <= 2),
                  ("3-5", lambda f: 3 <= f["left"] <= 5), ("6+", lambda f: f["left"] >= 6)],
}


def _logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _mse(v: np.ndarray) -> tuple[float, float]:
    return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """a, b of P(y) = sigmoid(a + b z) by Newton's method (two parameters)."""
    a, b = 0.0, 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(a + b * z, -30, 30)))
        g = np.array([np.sum(p - y), np.sum((p - y) * z)])
        w = p * (1 - p)
        h = np.array([[np.sum(w), np.sum(w * z)], [np.sum(w * z), np.sum(w * z * z)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(a), float(b)


def recalibrated(p: np.ndarray, y: np.ndarray, folds: int = 2, seed: int = 0) -> np.ndarray:
    """Each position's log loss after a logistic recalibration fitted on the other folds:
    what the value knows about the result once its scale and its bias are set aside."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(folds, size=len(y))
    z = _logit(p)
    out = np.zeros(len(y))
    for k in range(folds):
        a, b = _platt(z[fold != k], y[fold != k])
        q = 1 / (1 + np.exp(-np.clip(a + b * z[fold == k], -30, 30)))
        out[fold == k] = _logloss(q, y[fold == k])
    return out


def auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # ties share their mean rank
    _u, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def load_rows(set_dir: Path, name: str) -> tuple[list[dict], list[dict]]:
    entries = [json.loads(line) for line in
               (set_dir / "positions.jsonl").read_bytes().splitlines() if line.strip()]
    rows = []
    for path in sorted((set_dir / f"read-{name}").glob("*.json")):
        rows.append(json.loads(path.read_bytes()))
    return entries, rows


def collect(entries: list[dict], rows: list[dict], view: str, budget_s: float, keys: list[str],
            mask=None) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:  # noqa: ANN001
    """The results, each key's values and the positions' features, over the positions whose
    read of ``view`` holds every key at the budget (and that ``mask`` keeps)."""
    ys, vals, feats = [], {k: [] for k in keys}, []
    for r in rows:
        if view not in r:
            continue
        e = entries[r["n"]]
        f = features(e)
        if mask is not None and not mask(f):
            continue
        got = values_at(r[view], budget_s * 1000.0)
        if any(k not in got for k in keys):
            continue
        ys.append(e["outcome"])
        feats.append(f)
        for k in keys:
            vals[k].append(got[k])
    return (np.asarray(ys, dtype=np.float64), {k: np.asarray(v, dtype=np.float64)
                                                for k, v in vals.items()}, feats)


def table(entries: list[dict], rows: list[dict], view: str, budget_s: float, keys: list[str],
          mask=None, short: bool = False) -> list[str]:  # noqa: ANN001
    """One block of the report: each key's loss, Brier, recalibrated loss, AUC and bias, and
    the paired differences from the first key (standard errors over positions = games)."""
    y, vals, _f = collect(entries, rows, view, budget_s, keys, mask)
    if len(y) < 20:
        return [f"  (n {len(y)})"]
    out = [f"  n {len(y)}  mean result {y.mean():.4f}"]
    base = keys[0]
    q = vals[base]
    for k in keys:
        p = vals[k]
        ll, llse = _mse(_logloss(p, y))
        br, brse = _mse((p - y) ** 2)
        rc = recalibrated(p, y)
        miss = np.abs(p - y) > 0.9  # confident and wrong
        line = (f"  {k:6s} log {ll:.4f} ({llse:.4f}) brier {br:.4f} ({brse:.4f}) "
                f"recal {rc.mean():.4f} slope {_platt(_logit(p), y)[1]:.3f} "
                f"auc {auc(p, y):.4f} bias {p.mean() - y.mean():+.4f} "
                f"miss>0.9 {int(miss.sum())} ({_logloss(p, y)[miss].sum() / len(y):.4f})")
        if k != base:
            dl, dlse = _mse(_logloss(p, y) - _logloss(q, y))
            db, dbse = _mse((p - y) ** 2 - (q - y) ** 2)
            dr, drse = _mse(rc - recalibrated(q, y))
            # How much of the deeper value's change from the base is borne out by the result:
            # least squares of (result - base) on (value - base) with an intercept (the mean
            # shift is the bias column's). 1: the change is right in size; 0: it is noise to
            # the result.
            u, r = p - q - np.mean(p - q), y - q - np.mean(y - q)
            slope = float(u @ r / (u @ u)) if u @ u > 0 else float("nan")
            resid = r - slope * u
            slope_se = float(np.sqrt((resid @ resid) / max(len(y) - 2, 1) / (u @ u))) if u @ u > 0 else 0.0
            line += (f" | vs {base}: dlog {dl:+.4f} ({dlse:.4f}) dbrier {db:+.4f} ({dbse:.4f}) "
                     f"drecal {dr:+.4f} ({drse:.4f}) |diff| {np.mean(np.abs(p - q)):.4f} "
                     f"borne {slope:.3f} ({slope_se:.3f})")
        out.append(line)
    if not short:
        # Calibration: the results' mean in ten bins of each value.
        edges = np.linspace(0, 1, 11)
        for k in keys:
            p = vals[k]
            cells = []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
                cells.append(f"{lo:.1f}:{int(m.sum())}/{p[m].mean():.3f}/{y[m].mean():.3f}"
                             if m.any() else f"{lo:.1f}:0")
            out.append(f"  calib {k}: " + " ".join(cells))
    return out


def report(args: argparse.Namespace) -> None:
    set_dir = Path(args.set)
    entries, rows = load_rows(set_dir, args.name)
    lines = [f"set {set_dir}, read {args.name}: {len(rows)} positions read"]
    chains = [k.split("/") for k in args.keys.split(",")]
    # Three readings of the positions: the open positions (side 1's bench all seen) read as
    # they are; the hidden positions read as the agent reads them (the Bayesian game); and the
    # hidden positions read with side 1's true bench -- information the agent does not have,
    # an upper bound of what seeing the bench would give, not a reading the agent makes.
    views = [("open", "open", lambda f: f["classes"] == 1),
             ("hidden", "hidden", None),
             ("hidden-seen", "open", lambda f: f["classes"] >= 2)]
    for budget in [float(b) for b in args.budgets.split(",")]:
        for label, view, only in views:
            for keys in chains:
                lines.append(f"\n{label} @ {budget:g}s, {'/'.join(keys)}")
                lines += table(entries, rows, view, budget, keys, mask=only)
                if args.splits:
                    for split, parts in SPLITS.items():
                        for name, pred in parts:
                            both = (pred if only is None
                                    else (lambda f, p=pred, o=only: o(f) and p(f)))
                            lines.append(f"  [{split} {name}]")
                            lines += ["  " + s for s in table(entries, rows, view, budget, keys,
                                                              mask=both, short=True)]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        Path(args.out).write_bytes(text.encode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--games-dir", required=True)
    b.add_argument("--merged", required=True)
    b.add_argument("--shards", nargs="+", required=True)
    b.add_argument("--gen", type=int, default=4)
    b.add_argument("--holdout", type=float, default=0.15)
    b.add_argument("--split-seed", type=int, default=0)
    b.add_argument("--count", type=int, default=4000)
    b.add_argument("--seed", type=int, default=421)
    b.add_argument("--store", required=True, help="the selection store the games were played from")
    b.add_argument("--epsilon", type=float, default=0.25)
    b.add_argument("--temperature", type=float, default=0.5)
    b.add_argument("--pool", default=DEFAULT_POOL)
    b.add_argument("--out", required=True)
    r = sub.add_parser("read")
    r.add_argument("--set", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--pool", default=DEFAULT_POOL)
    r.add_argument("--value", nargs="+", required=True)
    r.add_argument("--q-model", required=True)
    r.add_argument("--inference", required=True)
    r.add_argument("--ladder", default="L6")
    r.add_argument("--seconds", type=float, default=16.0)
    r.add_argument("--width", type=int, default=64)
    r.add_argument("--d1-only", action="store_true")
    r.add_argument("--open-only", action="store_true")
    r.add_argument("--hidden-only", action="store_true",
                   help="IKA-421: only the Bayesian read, of the positions with a hidden bench")
    r.add_argument("--side", type=int, default=0, choices=(0, 1),
                   help="IKA-421: the side that reads (the open game only); values are written "
                   "in side 0's units")
    r.add_argument("--swap", action="store_true",
                   help="IKA-421: read the position with its seats swapped (the open game only); "
                   "values are written in the recorded side 0's units")
    r.add_argument("--only", default=None, help="IKA-421: these positions, comma-separated")
    r.add_argument("--from", dest="start", type=int, default=0)
    r.add_argument("--to", dest="stop", type=int, default=10**9)
    r.add_argument("--claim", default=None)
    p = sub.add_parser("report")
    p.add_argument("--set", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--budgets", default="4")
    p.add_argument("--keys", default="d0/d1,d0/d1/d2,d0/d1/d2/d3",
                   help="chains of values compared on the same positions, comma-separated")
    p.add_argument("--splits", action="store_true", help="also by the position's kind")
    p.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    {"build": build, "read": read, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
