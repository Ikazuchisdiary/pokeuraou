"""IKA-419: the selection game's value per pair of builds, and how well it matches played games.

The cheap evaluation of a build is the equilibrium value of the 90 x 90 selection game of the
build against each pool build, with the production leaf (`value-mc4st`, the mean of two nets).
This tool computes it and checks it against the win rates of games already played.

Three steps, each its own subcommand. Nothing here learns a leaf.

``winrates``  reads the games of an M-C generation (one worker file per task, in parallel) and
              counts, per pair of pool builds, the games and the wins. Only `pool.teams` and
              `outcome` are read, by a regex on the line; `--check` re-reads the first lines of
              each file with the JSON parser and stops on any difference.
``solve``     solves the selection game of each pair with the leaf (`poolplay.SolvedSelections`,
              the code that generation uses, one process per worker, CPU) and writes the matrix
              of values. ``--team-file`` adds builds that are not in the pool; then only their
              pairs with the pool are solved. The solves are kept in ``--store`` (one file per
              pair), so a stopped run resumes and the answers can be read again.
``report``    joins the two: rank correlation and calibration per build and per pair, or, with
              ``--team-file``, one build against the pool.

    python tools/selection_diag.py winrates data/selfplay-mc4 --out winrates-mc4.json --jobs 4
    python tools/selection_diag.py solve --store sel-store --out values.json --jobs 8
    python tools/selection_diag.py solve --team-file examples/diag-team-ika419.json \\
        --store sel-store-user --out values-user.json --jobs 8
    python tools/selection_diag.py report --values values.json --winrates winrates-mc4.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_POOL = "regmc-matchupweb"
DEFAULT_VALUE = ("data/models/value-mc4st.pt", "data/models/value-mc4st-s1.pt")

#: The `pool` object of a record, whatever scalar keys (id, sha256, pair) come before `teams`.
_POOL_RE = re.compile(rb'"pool": \{[^{}\[\]]*?"teams": \["([0-9a-f]+)", "([0-9a-f]+)"\]')
_OUTCOME_RE = re.compile(rb'"outcome": ([0-9.]+)')


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def repo_file(path: str) -> Path:
    """A path as given, or relative to the repo root (no machine path is written here)."""
    from pokeuraou.regulation import repo_root

    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


# ---------------------------------------------------------------- winrates -------------------


def _scan_file(task: tuple[str, int]) -> dict[str, Any]:
    """Counts of one games file: {(first team, second team): [games, wins of the first seat]}."""
    path, check_lines = task
    counts: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0])
    lines = 0
    skipped = 0
    outcomes: dict[str, int] = defaultdict(int)
    with open(path, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            lines += 1
            pool_match = _POOL_RE.search(line)
            outcome_match = _OUTCOME_RE.search(line)
            if pool_match is None or outcome_match is None:
                skipped += 1
                continue
            seat0, seat1 = pool_match.group(1).decode(), pool_match.group(2).decode()
            outcome = float(outcome_match.group(1))
            if lines <= check_lines:
                record = json.loads(line)
                want = (record["pool"]["teams"][0], record["pool"]["teams"][1], float(record["outcome"]))
                if (seat0, seat1, outcome) != want:
                    raise SystemExit(
                        f"{path} line {lines}: regex read {(seat0, seat1, outcome)}, JSON {want}"
                    )
            outcomes[str(outcome)] += 1
            cell = counts[(seat0, seat1)]
            cell[0] += 1
            cell[1] += outcome
    return {"path": path, "lines": lines, "skipped": skipped, "outcomes": dict(outcomes),
            "counts": {f"{a}|{b}": v for (a, b), v in counts.items()}}


def cmd_winrates(args: argparse.Namespace) -> None:
    from pokeuraou.pool import load_pool

    pool = load_pool(args.pool)
    ids = [t.id for t in pool.teams]
    files: list[str] = []
    for directory in args.dirs:
        files += sorted(glob.glob(os.path.join(str(repo_file(directory)), "games-*.jsonl")))
    if not files:
        raise SystemExit("no games-*.jsonl files")
    started = time.time()
    tasks = [(f, args.check) for f in files]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        parts = list(ex.map(_scan_file, tasks, chunksize=1))
    index = {t: k for k, t in enumerate(ids)}
    n = len(ids)
    games = np.zeros((n, n))  # games[i][j]: games of i (either seat) against j
    wins = np.zeros((n, n))  # wins[i][j]: i's wins against j
    seat0_games = seat0_wins = 0.0
    total_lines = skipped = 0
    outcomes: dict[str, int] = defaultdict(int)
    for part in parts:
        total_lines += part["lines"]
        skipped += part["skipped"]
        for key, value in part["outcomes"].items():
            outcomes[key] += value
        for key, (count, first_wins) in part["counts"].items():
            a, b = key.split("|")
            i, j = index[a], index[b]
            seat0_games += count
            seat0_wins += first_wins
            games[i][j] += count
            games[j][i] += count
            wins[i][j] += first_wins
            wins[j][i] += count - first_wins
    out = {
        "pool": pool.id, "poolSha256": pool.sha256, "teams": ids, "dirs": args.dirs,
        "files": len(files), "lines": total_lines, "skipped": skipped, "outcomes": dict(outcomes),
        "seat0WinShare": seat0_wins / seat0_games if seat0_games else None,
        "games": games.tolist(), "wins": wins.tolist(), "seconds": time.time() - started,
    }
    Path(args.out).write_bytes(json.dumps(out).encode("utf-8"))
    if skipped:
        raise SystemExit(f"{skipped} of {total_lines} lines had no pool.teams / outcome; "
                         f"nothing written would be trustworthy ({args.out} is still written)")
    print(f"{len(files)} files, {total_lines} games, seat-0 win share "
          f"{out['seat0WinShare']:.4f}, {time.time() - started:.0f} s -> {args.out}", file=sys.stderr)


# ------------------------------------------------------------------- solve -------------------

_STATE: dict[str, Any] = {}


def _team_json(path: str) -> dict[str, Any]:
    return json.loads(repo_file(path).read_bytes().decode("utf-8"))


def _teams_for(pool: Any, team_files: Sequence[str]) -> list[Any]:
    from pokeuraou.teams import roster_from_data

    teams = list(pool.teams)
    for path in team_files:
        data = _team_json(path)
        data.setdefault("regulation", pool.reg.meta.format_id)
        teams.append(roster_from_data(data, default_id=str(data["id"]), reg=pool.reg))
    return teams


def _init_worker(pool_name: str, values: list[str], store: str, team_files: list[str]) -> None:
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    import torch

    torch.set_num_threads(1)
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder, EncodingRules
    from pokeuraou.pool import load_pool
    from pokeuraou.poolplay import SolvedSelections
    from pokeuraou.value import BatchedValue, load_ensemble

    pool = load_pool(pool_name)
    register_mega_stones(pool.reg)
    teams = _teams_for(pool, team_files)
    encoder = Encoder(pool.reg, rules=EncodingRules())
    paths = [repo_file(v) for v in values]
    nets, _ = load_ensemble(paths, encoder)
    device = torch.device("cpu")
    value = BatchedValue([n.to(device) for n in nets], encoder, device=device)
    stem = re.sub(r"-s\d+$", "", paths[0].stem)
    name = stem if len(paths) == 1 else f"{stem}x{len(paths)}"
    extra = "+".join(str(_team_json(p)["id"]) for p in team_files)
    tag = f"{pool.sha256}|value:{name}" + (f"|{extra}" if extra else "")
    _STATE["solver"] = SolvedSelections(
        pool.reg, teams, value, model=f"value:{name}", store=Path(store), tag=tag
    )
    _STATE["name"] = name


def _solve_pair(pair: tuple[int, int]) -> dict[str, Any]:
    solver = _STATE["solver"]
    lo, hi = pair
    started = time.perf_counter()
    entry = solver.canonical(lo, hi)
    return {"lo": lo, "hi": hi, "value": float(entry.value), "gap": float(entry.duality_gap),
            "antisymmetry": float(entry.antisymmetry_error), "seconds": time.perf_counter() - started,
            "loaded": solver.loaded, "solves": solver.solves}


def cmd_solve(args: argparse.Namespace) -> None:
    from pokeuraou.pool import load_pool

    pool = load_pool(args.pool)
    teams = _teams_for(pool, args.team_file)
    n_pool, n_all = len(pool.teams), len(teams)
    if args.team_file:
        pairs = [(i, k) for k in range(n_pool, n_all) for i in range(n_pool)]
    else:
        pairs = [(i, j) for i in range(n_pool) for j in range(i + 1, n_pool)]
    if args.limit:
        pairs = pairs[: args.limit]
    store = repo_file(args.store)
    store.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.jobs, initializer=_init_worker,
        initargs=(args.pool, list(args.value), str(store), list(args.team_file)),
    ) as ex:
        for k, row in enumerate(ex.map(_solve_pair, pairs, chunksize=4)):
            results.append(row)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(pairs)} pairs, {time.time() - started:.0f} s", file=sys.stderr)
    # Mirrors are 0.5 by antisymmetry; not solved.
    value = np.full((n_all, n_all), np.nan)
    for i in range(n_all):
        value[i][i] = 0.5
    for r in results:
        value[r["lo"]][r["hi"]] = r["value"]
        value[r["hi"]][r["lo"]] = 1.0 - r["value"]
    out = {
        "pool": pool.id, "poolSha256": pool.sha256, "teams": [t.id for t in teams],
        "names": [t.name for t in teams], "leaf": list(args.value), "nPool": n_pool,
        "values": [[None if math.isnan(x) else x for x in row] for row in value],
        "pairs": results, "seconds": time.time() - started, "jobs": args.jobs,
        "maxGap": max((r["gap"] for r in results), default=0.0),
        "maxAntisymmetry": max((r["antisymmetry"] for r in results), default=0.0),
        "loadedFromStore": sum(1 for r in results if r["seconds"] < 0.5),
    }
    Path(args.out).write_bytes(json.dumps(out).encode("utf-8"))
    print(f"{len(results)} pairs in {out['seconds']:.0f} s, max duality gap {out['maxGap']:.2e}, "
          f"max antisymmetry {out['maxAntisymmetry']:.2e} -> {args.out}", file=sys.stderr)


# ------------------------------------------------------------------- report ------------------


def wilson(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    # ties get their mean rank
    for v in np.unique(x):
        mask = x == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def cmd_report(args: argparse.Namespace) -> None:
    values = _load(args.values)
    if args.team_file:
        _report_team(args, values)
        return
    wr = _load(args.winrates)
    if values["teams"] != wr["teams"] or values["poolSha256"] != wr["poolSha256"]:
        raise SystemExit("values and winrates are for different pools")
    n = len(values["teams"])
    V = np.array([[np.nan if x is None else x for x in row] for row in values["values"]])
    G, W = np.array(wr["games"]), np.array(wr["wins"])
    off = ~np.eye(n, dtype=bool)
    if np.isnan(V[off]).any():
        raise SystemExit("values has unsolved pairs")
    rng = np.random.default_rng(args.seed)
    lines: list[str] = []
    out = lines.append

    # per build: mean value over the other builds vs the mean of pair win rates, and pooled rate
    mean_v = np.array([V[i][off[i]].mean() for i in range(n)])
    pair_rate = np.where(G > 0, W / np.maximum(G, 1), np.nan)
    mean_w = np.array([np.nanmean(pair_rate[i][off[i]]) for i in range(n)])
    pooled_g = np.array([G[i][off[i]].sum() for i in range(n)])
    pooled_w = np.array([W[i][off[i]].sum() for i in range(n)])
    pooled_rate = pooled_w / pooled_g
    rho = spearman(mean_v, mean_w)
    r_p = float(np.corrcoef(mean_v, mean_w)[0, 1])
    # sampling noise: resample every pair's wins binomially at its observed rate
    noise_rho = []
    for _ in range(args.boot):
        Wb = np.zeros_like(W)
        for i in range(n):
            for j in range(i + 1, n):
                if G[i][j] > 0:
                    w = rng.binomial(int(G[i][j]), pair_rate[i][j])
                    Wb[i][j], Wb[j][i] = w, G[i][j] - w
        pr = np.where(G > 0, Wb / np.maximum(G, 1), np.nan)
        noise_rho.append(spearman(mean_v, np.array([np.nanmean(pr[i][off[i]]) for i in range(n)])))
    boot_rho = []
    for _ in range(args.boot):
        pick = rng.integers(n, size=n)
        boot_rho.append(spearman(mean_v[pick], mean_w[pick]))
    out(f"## Per build ({n} builds): mean selection value against the other {n - 1} vs played win rate")
    out(f"games {int(G.sum() // 2)} over {int(off.sum() // 2)} pairs, "
        f"{G[off].min():.0f}-{G[off].max():.0f} per pair (mean {G[off].mean():.1f})")
    out(f"Spearman {rho:+.3f}  (build bootstrap 95% [{np.percentile(boot_rho, 2.5):+.3f}, "
        f"{np.percentile(boot_rho, 97.5):+.3f}]; game-sampling noise alone "
        f"[{np.percentile(noise_rho, 2.5):+.3f}, {np.percentile(noise_rho, 97.5):+.3f}]), Pearson {r_p:+.3f}")
    pooled_pearson = float(np.corrcoef(mean_v, pooled_rate)[0, 1])
    unplayed = int((G[off] == 0).sum() // 2)
    out(f"with the build's games pooled instead of averaged per opponent: Spearman "
        f"{spearman(mean_v, pooled_rate):+.3f}, Pearson {pooled_pearson:+.3f}; "
        f"builds' games {int(pooled_g.min())}-{int(pooled_g.max())}; pairs without a game {unplayed}")
    out(f"mean value range [{mean_v.min():.3f}, {mean_v.max():.3f}], win-rate range "
        f"[{mean_w.min():.3f}, {mean_w.max():.3f}], SD value {mean_v.std():.3f}, SD rate {mean_w.std():.3f}")
    slope = float(np.polyfit(mean_v, mean_w, 1)[0])
    out(f"least-squares slope of rate on value {slope:.2f} (1.00 would be an exact match in scale)")
    out("")
    out("Calibration by quintile of mean value (builds sorted by value):")
    out("| quintile | builds | mean value | played win rate (95% Wilson) | games |")
    out("|---|---|---|---|---|")
    order = np.argsort(mean_v)
    for q, idx in enumerate(np.array_split(order, 5)):
        g, w = pooled_g[idx].sum(), pooled_w[idx].sum()
        lo, hi = wilson(w, g)
        out(f"| {q + 1} | {len(idx)} | {mean_v[idx].mean():.3f} | "
            f"{w / g:.3f} [{lo:.3f}, {hi:.3f}] | {int(g)} |")
    out("")

    # per pair
    iu = np.triu_indices(n, 1)
    pv, pg, pw = V[iu], G[iu], W[iu]
    keep = pg > 0
    pv, pg, pw = pv[keep], pg[keep], pw[keep]
    pr = pw / pg
    rho_pair = spearman(pv, pr)
    se = np.sqrt(np.maximum(pv * (1 - pv), 1e-9) / pg)
    z = (pr - pv) / se
    inside = np.mean([wilson(w, g)[0] <= v <= wilson(w, g)[1] for v, w, g in zip(pv, pw, pg, strict=True)])
    out(f"## Per pair ({len(pv)} pairs, one value per pair from the first build's chair)")
    out(f"Spearman {rho_pair:+.3f}, Pearson {float(np.corrcoef(pv, pr)[0, 1]):+.3f}; "
        f"mean |rate - value| {np.abs(pr - pv).mean():.3f}; value inside the pair's 95% Wilson interval "
        f"in {inside * 100:.1f}% of pairs (95% if the value were the true rate)")
    out(f"sampling SD of a pair's rate at its value ~{np.sqrt(np.mean(se ** 2)):.3f}; observed SD of "
        f"(rate - value) {np.std(pr - pv):.3f}; mean z^2 {np.mean(z ** 2):.2f} (1.0 if value were exact)")
    out("")
    out("Calibration by value bin (pairs pooled; each pair counted once, "
        "from the side with the value above 0.5):")
    out("| value bin | pairs | mean value | played win rate (95% Wilson) | games |")
    out("|---|---|---|---|---|")
    # fold so every pair is read from the favoured side: value >= 0.5
    flip = pv < 0.5
    fv = np.where(flip, 1 - pv, pv)
    fw = np.where(flip, pg - pw, pw)
    edges = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]
    for lo_e, hi_e in zip(edges[:-1], edges[1:], strict=True):
        m = (fv >= lo_e) & (fv < hi_e)
        if not m.any():
            continue
        g, w = pg[m].sum(), fw[m].sum()
        lo, hi = wilson(w, g)
        out(f"| [{lo_e:.2f}, {min(hi_e, 1.0):.2f}) | {int(m.sum())} | {fv[m].mean():.3f} | "
            f"{w / g:.3f} [{lo:.3f}, {hi:.3f}] | {int(g)} |")
    out("")
    out("Pairs the value gets wrong by sign (value > 0.5 but the played rate below 0.5, with the "
        "95% interval wholly below 0.5):")
    bad = 0
    for v, w, g in zip(fv, fw, pg, strict=True):
        if v > 0.5 and wilson(w, g)[1] < 0.5:
            bad += 1
    out(f"{bad} of {len(pv)} pairs")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_bytes(text.encode("utf-8") + b"\n")
    if args.rows:
        rows = [{"team": values["teams"][i], "name": values["names"][i], "meanValue": float(mean_v[i]),
                 "meanRate": float(mean_w[i]), "pooledRate": float(pooled_rate[i]),
                 "games": int(pooled_g[i])} for i in range(n)]
        Path(args.rows).write_bytes(json.dumps(rows).encode("utf-8"))


def _report_team(args: argparse.Namespace, values: dict[str, Any]) -> None:
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.names import localiser
    from pokeuraou.pool import load_pool
    from pokeuraou.poolplay import SolvedSelections

    pool = load_pool(args.pool or values["pool"])
    register_mega_stones(pool.reg)
    teams = _teams_for(pool, args.team_file)
    n_pool = values["nPool"]
    V = np.array([[np.nan if x is None else x for x in row] for row in values["values"]])
    k = n_pool  # the first extra build
    vs = V[k][:n_pool]
    loc = localiser(pool.reg, "ja")

    def sp(species: str) -> str:
        return loc.species(species) if loc else species

    store = repo_file(args.store)
    first = json.loads(next(store.glob("*.json")).read_bytes())
    solver = SolvedSelections(pool.reg, teams, evaluate=None, store=store, tag=first["tag"])
    lines: list[str] = []
    out = lines.append
    out(f"build {teams[k].id}: {', '.join(sp(s.species) for s in teams[k].sets)}")
    out(f"mean selection value against the {n_pool} pool builds: {np.nanmean(vs):.4f} "
        f"(SD over opponents {np.nanstd(vs):.4f})")
    if args.pool_values:
        P = np.array([[np.nan if x is None else x for x in row]
                      for row in _load(args.pool_values)["values"]])
        means = np.array([np.mean(np.delete(P[i], i)) for i in range(len(P))])
        out(f"for scale, the pool builds' own mean value over the other 64: min {means.min():.3f}, "
            f"median {np.median(means):.3f}, max {means.max():.3f}; this build ranks "
            f"{int((means > np.nanmean(vs)).sum()) + 1} of {len(means) + 1} if put among them")
    out(f"opponents with value < 0.5: {int((vs < 0.5).sum())} of {n_pool}")
    out("")
    out(f"Weakest {args.top} opponents (value of the selection game for our build):")
    out("| rank | opponent | six | value | our selections (leads / back, equilibrium share) |")
    out("|---|---|---|---|---|")
    order = np.argsort(vs)
    for r, i in enumerate(order[: args.top]):
        entry = solver.entry(k, int(i))
        top = np.argsort(-entry.our_strategy)[:3]
        sel = "; ".join(
            f"{'+'.join(sp(teams[k].sets[x].species) for x in entry.selections[s][:2])} / "
            f"{'+'.join(sp(teams[k].sets[x].species) for x in entry.selections[s][2:])} "
            f"({entry.our_strategy[s] * 100:.0f}%)" for s in top
        )
        six = " ".join(sp(s.species) for s in teams[int(i)].sets)
        out(f"| {r + 1} | {teams[int(i)].name} | {six} | {vs[i]:.3f} | {sel} |")
    out("")
    out("Strongest 5 opponents-for-us:")
    for i in order[::-1][:5]:
        out(f"- {teams[int(i)].name}: {vs[i]:.3f}")
    out("")
    out("All opponents (value ascending):")
    for i in order:
        out(f"{vs[i]:.3f}  {teams[int(i)].name}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_bytes(text.encode("utf-8") + b"\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("winrates", help="count games and wins per pair of pool builds")
    w.add_argument("dirs", nargs="+")
    w.add_argument("--pool", default=DEFAULT_POOL)
    w.add_argument("--out", required=True)
    w.add_argument("--jobs", type=int, default=4)
    w.add_argument("--check", type=int, default=20, help="lines per file re-read with the JSON parser")
    w.set_defaults(func=cmd_winrates)
    s = sub.add_parser("solve", help="solve the selection game of each pair with the leaf")
    s.add_argument("--pool", default=DEFAULT_POOL)
    s.add_argument("--value", nargs="+", default=list(DEFAULT_VALUE))
    s.add_argument("--team-file", action="append", default=[],
                   help="a build outside the pool (pool-team JSON)")
    s.add_argument("--store", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--limit", type=int, default=0, help="solve only the first N pairs (a smoke test)")
    s.set_defaults(func=cmd_solve)
    r = sub.add_parser("report", help="rank correlation and calibration, or one build against the pool")
    r.add_argument("--values", required=True)
    r.add_argument("--winrates")
    r.add_argument("--team-file", action="append", default=[])
    r.add_argument("--store", help="with --team-file: the solve store")
    r.add_argument("--pool", help="with --team-file: the pool file (default: the id in values.json)")
    r.add_argument("--pool-values", help="with --team-file: the pool's own values.json, for scale")
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--boot", type=int, default=200)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out")
    r.add_argument("--rows")
    r.set_defaults(func=cmd_report)
    args = ap.parse_args()
    if args.cmd == "report" and not args.team_file and not args.winrates:
        ap.error("report needs --winrates (or --team-file)")
    args.func(args)


if __name__ == "__main__":
    main()
