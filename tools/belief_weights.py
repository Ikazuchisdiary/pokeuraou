"""Does a better weight on the hidden bench's completions cut the search's loss? (IKA-417, stage 1)

Reads a saved hidden position set (IKA-394's `hidden40`: `positions.json` and `ref-r24/<n>.npz`
with each completion's depth-2 matrix) and re-solves the Bayesian game under other weights on
the completions. No new reading: the matrices do not depend on the weights, only the answer
does. Nothing here touches the production path; the weights are built in this tool.

    python tools/belief_weights.py loss --set SET --games-dir DIR --pool POOL \
        --store SELECTION_SOLVED_DIR --standings STANDINGS.json.gz --out DIR

Loss: the win-rate loss of the answer solved under the weights `w`, scored in the game of the
TRUE bench (the opponent knows its bench): ``V_true - min_col(x_w @ M_true)``, where ``V_true``
is that game's own value (`error_decompose.hidden`'s ``trueLoss``). The same definition as
IKA-394 §2.1. Controls in the report: weight 1 on the true bench gives 0, and uniform weights
give IKA-394's 0.0214.

Weights compared (`--weights`):

* ``uniform``     what the saved set was built with (`position_set.py build-hidden`: no prior)
* ``production`` the weights M-C generation gives the belief: the selection game's solved
                  distribution (`BenchPrior`), conditioned on the shown Pokemon and the turn-1
                  lead pair (`hidden.completions` with `selfplay._bench_weights`)
* ``usage``      product of the species' share of tournament teams
* ``usageobs``   usage times the pair lift (co-occurrence on the same six / chance) between the
                  two completed members and between each of them and the shown Pokemon
* ``produsage``  production x usage, renormalised (the selection prior with the usage as a
                  second opinion; a check whether the two carry different information)
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from pokeuraou.equilibrium import solve, solve_bayesian
from pokeuraou.regulation import to_id

LEAD_CONTROL = "true bench weight 1"


def _value(mats: list[np.ndarray], w: np.ndarray) -> tuple[float, np.ndarray]:
    if len(mats) == 1:
        eq = solve(mats[0])
        return float(eq.value), eq.row_strategy
    eq = solve_bayesian(mats, w)
    return float(eq.value), eq.row_strategy


def _bench_key(pokemon: list[dict], slots: list[int]) -> str:
    return json.dumps(
        [(q["species"], q["item"], q["nature"], sorted(q["sp"].items()),
          [m["id"] for m in q["moves"]]) for q in pokemon if q["slot"] in slots],
        sort_keys=True)


def _norm(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / len(w))


# ---------------------------------------------------------------------------- tournament usage


def tournament_tables(path: Path) -> tuple[dict[str, float], dict[tuple[str, str], float], int]:
    """(species -> share of teams, unordered pair -> lift, teams) from a standings file.

    Lift is P(a,b) / (P(a) P(b)) over the teams, with half a team added to every pair so a
    pair nobody played is not zero (its weight would otherwise rule the completion out).
    Both keys and members are `to_id` of the species name on the sheet.
    """
    data = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    teams = [sorted({to_id(m["name"]) for m in e["team"]}) for e in data["standings"].values()]
    n = len(teams)
    count: dict[str, int] = {}
    pair: dict[tuple[str, str], int] = {}
    for team in teams:
        for a in team:
            count[a] = count.get(a, 0) + 1
        for a, b in itertools.combinations(team, 2):
            pair[(a, b)] = pair.get((a, b), 0) + 1
    share = {a: c / n for a, c in count.items()}
    lift: dict[tuple[str, str], float] = {}
    for a, b in itertools.combinations(sorted(count), 2):
        co = (pair.get((a, b), 0) + 0.5) / (n + 0.5)
        lift[(a, b)] = co / (share[a] * share[b])
    return share, lift, n


# ---------------------------------------------------------------------------- the game


def find_decision(games_dir: Path, source: str, turn: int, position: dict) -> tuple[dict, dict]:
    for line in (games_dir / source).read_bytes().splitlines():
        if not line.strip():
            continue
        game = json.loads(line)
        for d in game["decisions"]:
            if d["kind"] == "move" and d["turn"] == turn and d["position"] == position:
                return game, d
    raise LookupError(f"{source}: no decision at turn {turn} with that position")


def production_weights(  # noqa: PLR0913 - what the generation had in hand
    pool, store: Path, game: dict, decision: dict, position, keys: list[tuple[str, ...]], *,
    epsilon: float, temperature: float,
) -> tuple[np.ndarray, dict]:
    """The weights the searcher (side 0) held on side 1's completions in that decision.

    Rebuilt the way `poolplay` builds them: the entry of the pair read from the selection
    store (transposed when side 0's team has the higher index), `BenchPrior.of(entry, 1, ..)`,
    then `selfplay._bench_weights` with the shown identities and the turn-1 leads.
    """
    import json as _json

    from pokeuraou.hidden import seen_slots
    from pokeuraou.poolplay import transposed
    from pokeuraou.selection_book import BenchPrior, BookEntry
    from pokeuraou.selfplay import GameRecord, _bench_weights

    named = game["pool"]
    ids = [t.id for t in pool.teams]
    a, b = (ids.index(named["teams"][0]), ids.index(named["teams"][1]))
    lo, hi = min(a, b), max(a, b)
    data = _json.loads((store / f"{lo:03d}-{hi:03d}.json").read_bytes().decode("utf-8"))
    entry = BookEntry.from_json(data["entry"])
    if a > b:
        entry = transposed(entry, pool.teams[b].sets, key=f"{ids[a]}|{ids[b]}",
                           player=pool.teams[b].name)
    species1 = [s.species for s in pool.teams[b].sets]
    prior = BenchPrior.of(entry, 1, species1, epsilon=epsilon, temperature=temperature)
    shown = decision["shownIdentities"]
    seen1 = seen_slots(position, 1, shown[1])
    record = GameRecord.__new__(GameRecord)
    record.unmodelled = []
    # `record.leads` holds identities (base species); the generation passed `shown_species`
    # of the led slots, which also names the sheet entry of a regional form (Arcanine-Hisui
    # has base `arcanine`). Spelled the same way here.
    led_ids = set(game["leads"][1])
    led = frozenset(
        name
        for mon in position.sides[1].pokemon
        if to_id(mon.base_species) in led_ids
        for name in (to_id(mon.species), to_id(mon.base_species))
    )
    got = _bench_weights((None, prior), 1, position, seen1, record, led)
    if got is None:
        return np.full(len(keys), 1.0 / len(keys)), {"explains": False, "mass": 0.0}
    w = np.array([got.get(k, 0.0) for k in keys])
    return _norm(w), {"explains": True, "mass": float(w.sum()), "notes": record.unmodelled}


def shown_ids(position, seen_slots_) -> set[str]:
    side = position.sides[1]
    return {to_id(m.base_species) for m in side.pokemon if m.slot in seen_slots_}


def usage_weights(
    completions: list[dict], shown: set[str], share: dict, lift: dict, *, with_lift: bool,
) -> np.ndarray:
    out = []
    floor = min(share.values())
    for c in completions:
        members = [to_id(s) for s in c["species"]]
        w = 1.0
        for m in members:
            w *= share.get(m, floor)
        if with_lift:
            group = members + sorted(shown)
            for x, y in itertools.combinations(group, 2):
                if x in members or y in members:
                    key = (x, y) if x < y else (y, x)
                    w *= lift.get(key, 1.0)
        out.append(w)
    return _norm(np.array(out))


def mean_se(v) -> tuple[float, float]:  # noqa: ANN001
    a = np.asarray(v, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def loss_row(label: str, v, extra: str = "") -> str:  # noqa: ANN001
    a = np.asarray(v, dtype=np.float64)
    m, se = mean_se(a)
    return (f"  {label:<40} n={len(a):>3}  {m:.4f} (se {se:.4f})  median {np.median(a):.4f}  "
            f"max {a.max():.4f}  >0.005: {int((a > 0.005).sum()):>2}{extra}")


def loss_command(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915 - one pass
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.hidden import seen_slots
    from pokeuraou.pool import load_pool
    from pokeuraou.position import Position

    set_dir = Path(args.set)
    data = json.loads((set_dir / "positions.json").read_bytes())
    pool = load_pool(args.pool)
    register_mega_stones(pool.reg)
    share, lift, teams = tournament_tables(Path(args.standings))
    names = args.weights.split(",")
    rows = []
    for n, entry in enumerate(data["positions"]):
        path = set_dir / "ref-r24" / f"{n}.npz"
        if not path.exists():
            continue
        ref = np.load(path)
        d2 = np.asarray(ref["d2"], dtype=np.float64)
        k_all = d2.shape[0]
        mats = [d2[k] for k in range(k_all)]
        completions = entry["spreads"]["1"]
        slots = completions[0]["slots"]
        key = _bench_key(entry["position"]["sides"][1]["pokemon"], slots)
        keys = [_bench_key(c["position"]["sides"][1]["pokemon"], slots) for c in completions]
        true = [i for i, c in enumerate(keys) if c == key]
        if not true:
            raise SystemExit(f"position {n}: the true bench is not among the completions")
        t = true[0]
        position = Position.from_json(entry["position"])
        game, decision = find_decision(Path(args.games_dir), entry["source"], entry["turn"],
                                       entry["position"])
        seen1 = seen_slots(position, 1, decision["shownIdentities"][1])
        shown = shown_ids(position, seen1)
        v_worlds = [float(solve(m).value) for m in mats]
        weights: dict[str, np.ndarray] = {}
        info: dict = {}
        if "uniform" in names:
            weights["uniform"] = np.full(k_all, 1.0 / k_all)
        prod, info = production_weights(
            pool, Path(args.store), game, decision, position,
            [tuple(sorted(c["species"])) for c in completions],
            epsilon=args.epsilon, temperature=args.temperature)
        if "production" in names:
            weights["production"] = prod
        if "usage" in names:
            weights["usage"] = usage_weights(completions, shown, share, lift, with_lift=False)
        if "usageobs" in names:
            weights["usageobs"] = usage_weights(completions, shown, share, lift, with_lift=True)
        if "produsage" in names:
            weights["produsage"] = _norm(prod * weights.get("usage", usage_weights(
                completions, shown, share, lift, with_lift=False)))
        # Diagnostics around the production weights (not tuned: n=40 would overfit a knob).
        if "prodmix" in names:
            weights["prodmix"] = _norm(0.9 * prod + 0.1 / k_all)
        if "prodsharp" in names:
            weights["prodsharp"] = _norm(prod**2)
        rec = {"n": n, "turn": entry["turn"], "K": k_all, "true": t, "info": info,
               "loss": {}, "wTrue": {}, "weights": {}, "evpi": {}}
        for label, w in weights.items():
            if k_all > 1:
                rec["evpi"][label] = float(np.dot(w, v_worlds) - _value(mats, w)[0])
        for label, w in weights.items():
            _v, x = _value(mats, w) if k_all > 1 else _value(mats, np.ones(1))
            rec["loss"][label] = v_worlds[t] - float((x @ mats[t]).min())
            rec["wTrue"][label] = float(w[t])
            rec["weights"][label] = [float(v) for v in w]
        rec["loss"][LEAD_CONTROL] = 0.0 if k_all == 1 else (
            v_worlds[t] - float((_value([mats[t]], np.ones(1))[1] @ mats[t]).min()))
        # decisions: the answer from only the m heaviest completions of the production weights
        order = np.argsort(-prod, kind="stable")
        rec["topm"] = {}
        for m in range(1, k_all + 1):
            keep = list(order[:m])
            ws = _norm(prod[keep])
            _v, x = _value([mats[i] for i in keep], ws) if m > 1 else _value([mats[keep[0]]], np.ones(1))
            rec["topm"][m] = v_worlds[t] - float((x @ mats[t]).min())
        rows.append(rec)
        print(f"{n}: K={k_all} true={t} wTrue " +
              " ".join(f"{k}={v:.3f}" for k, v in rec["wTrue"].items()) +
              " loss " + " ".join(f"{k}={v:.4f}" for k, v in rec["loss"].items()),
              file=sys.stderr, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "belief-weights.json").write_bytes((json.dumps(rows, indent=1) + "\n").encode("utf-8"))
    lines = [f"BELIEF WEIGHTS set {set_dir}: {len(rows)} positions; {teams} tournament teams"]
    for label in names + [LEAD_CONTROL]:
        sel = [r for r in rows if label in r["loss"]]
        if not sel:
            continue
        wt = [r["wTrue"][label] for r in sel if label in r["wTrue"]]
        extra = f"  weight on true {np.mean(wt):.3f}" if wt else ""
        lines.append(loss_row(label, [r["loss"][label] for r in sel], extra))
    lines.append("value of seeing the bench under each weighting (sum_k w_k V_k - Bayesian V):")
    for label in names:
        v = [r["evpi"][label] for r in rows if label in r["evpi"]]
        if v:
            lines.append(loss_row(f"evpi, {label}", v))
    if "production" in names:
        lines.append("paired difference from production (loss of the weights - loss of production):")
        for label in names:
            if label == "production":
                continue
            diff = [r["loss"][label] - r["loss"]["production"] for r in rows if label in r["loss"]]
            if diff:
                m, se = mean_se(diff)
                lines.append(f"  {label:<20} {m:+.4f} (se {se:.4f}), better in "
                             f"{sum(d < -1e-9 for d in diff)}, worse in {sum(d > 1e-9 for d in diff)}, "
                             f"n={len(diff)}")
    ms = sorted({m for r in rows for m in r["topm"]})
    for m in ms:
        sel = [r["topm"][m] for r in rows if m in r["topm"]]
        lines.append(loss_row(f"production weights, heaviest {m} completions", sel))
    text = "\n".join(lines) + "\n"
    (out / "belief-weights.txt").write_bytes(text.encode("utf-8"))
    print(text)


def predict_command(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915 - one pass
    """Weight on the true bench of side 1, every hidden move decision of the first games.

    Needs no matrices: the candidates are the sheet members not shown, the true bench is the
    side-1 Pokemon sitting in the unshown slots. One number per decision, then per game,
    then the mean over games with a game-level se (decisions of one game are correlated).
    """
    from pokeuraou.hidden import seen_slots
    from pokeuraou.pool import load_pool
    from pokeuraou.position import Position

    pool = load_pool(args.pool)
    share, lift, _teams = tournament_tables(Path(args.standings))
    files = sorted(Path(args.games_dir).glob("games-*worker*.jsonl"))
    labels = ["uniform", "production", "usage", "usageobs"]
    per_game: list[dict] = []
    refused = 0
    for f in files:
        for line in f.read_bytes().splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            if game.get("information") != "hidden-bench" or game.get("leads") is None:
                continue
            if game["leads"][1] is None or len(per_game) >= args.games:
                continue
            ids = [t.id for t in pool.teams]
            sheet = [to_id(s.species) for s in pool.teams[ids.index(game["pool"]["teams"][1])].sets]
            got = {k: [] for k in labels}
            for d in game["decisions"]:
                if d["kind"] != "move":
                    continue
                position = Position.from_json(d["position"])
                seen1 = seen_slots(position, 1, d["shownIdentities"][1])
                hidden = [m for m in position.sides[1].pokemon if m.slot not in seen1]
                if not hidden:
                    continue
                shown = shown_ids(position, seen1)
                on_board = set(shown)
                for m in position.sides[1].pokemon:
                    if m.slot in seen1:
                        on_board.add(to_id(m.species))
                cand = [s for s in sheet if s not in on_board]
                combos = list(itertools.combinations(cand, len(hidden)))
                true_key = tuple(sorted(to_id(m.species) for m in hidden))
                try:
                    t = [tuple(sorted(c)) for c in combos].index(true_key)
                except ValueError:
                    refused += 1
                    continue
                fake = [{"species": list(c)} for c in combos]
                keys = [tuple(sorted(c)) for c in combos]
                try:
                    prod, info = production_weights(
                        pool, Path(args.store), game, d, position, keys,
                        epsilon=args.epsilon, temperature=args.temperature)
                except ValueError:
                    refused += 1
                    continue
                if not info["explains"]:
                    refused += 1
                    continue
                got["uniform"].append(1.0 / len(keys))
                got["production"].append(float(prod[t]))
                got["usage"].append(float(usage_weights(fake, shown, share, lift,
                                                        with_lift=False)[t]))
                got["usageobs"].append(float(usage_weights(fake, shown, share, lift,
                                                           with_lift=True)[t]))
            if got["uniform"]:
                per_game.append({k: v for k, v in got.items()})
        if len(per_game) >= args.games:
            break
    lines = [f"PREDICT {len(per_game)} games, {sum(len(g['uniform']) for g in per_game)} hidden "
             f"decisions (side 1); unexplained or refused: {refused}"]
    for label in labels:
        mean_w = [float(np.mean(g[label])) for g in per_game]
        mean_log = [float(np.mean(np.log(np.maximum(g[label], 1e-6)))) for g in per_game]
        low = [float(np.mean(np.asarray(g[label]) < 0.2)) for g in per_game]
        m, se = mean_se(mean_w)
        ml, sl = mean_se(mean_log)
        lo, so = mean_se(low)
        lines.append(f"  {label:<12} weight on true {m:.3f} (se {se:.3f})  mean log {ml:.3f} "
                     f"(se {sl:.3f})  share of decisions with weight < 0.2: {lo:.3f} (se {so:.3f})")
    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predict.txt").write_bytes(text.encode("utf-8"))
    print(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("loss")
    p.add_argument("--set", required=True)
    p.add_argument("--games-dir", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--standings", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights",
                   default="uniform,production,usage,usageobs,produsage,prodmix,prodsharp")
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=0.5)
    q = sub.add_parser("predict", help="how much weight each prior puts on the true bench, "
                       "over every hidden decision of many games (no matrices)")
    q.add_argument("--games-dir", required=True)
    q.add_argument("--pool", required=True)
    q.add_argument("--store", required=True)
    q.add_argument("--standings", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--games", type=int, default=400)
    q.add_argument("--epsilon", type=float, default=0.25)
    q.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args(argv)
    {"loss": loss_command, "predict": predict_command}[args.cmd](args)


if __name__ == "__main__":
    main()
