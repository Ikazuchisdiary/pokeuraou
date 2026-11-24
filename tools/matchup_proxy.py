"""Does the selection game's value predict a pair's real results? (IKA-442)

The first step towards searching for strong teams. A team's strength is its mean score
against the pool, and playing games is the expensive way to measure it. The selection game
of a pair (`selection.solve_selection`: the 90x90 matrix of turn-1 values, one forward pass a
cell) is the cheap way. This tool asks how well the cheap number predicts the games that
were actually played, using the generation records only (no new game, no model).

    python tools/matchup_proxy.py extract --games-dir DATA/selfplay-mc6 --out ROWS.npz --jobs 8
    python tools/matchup_proxy.py report --rows ROWS.npz [--store STORE] --out REPORT.json

* **extract**: one row a finished game from every ``*.jsonl`` under ``--games-dir``: the two
  team ids, side 0's score and the selection value written with the game
  (``selectionValue``, side 0's win probability in the pair's selection game). Mirrors and
  games without a value are counted and dropped.
* **report**: every game turned to its pair's canonical seat (the team id that sorts first
  is the row), then
  - pair level: the correlation of each pair's mean score with its selection value, the
    share of the scores' spread that sampling noise explains (the reliability) and the
    correlation corrected by it, and the regression slope (1 = calibrated);
  - game level: the Brier score and log loss of the value against the coin (0.5) and
    against the other half's pair mean (the ceiling a per-pair number can reach);
  - team level (what a team search needs): each team's mean over its opponents by the
    value and by the scores, their Pearson and Spearman correlation, the split-half
    reliability of the scores' ranking, and how many of the top k agree.
  ``--store`` replaces the recorded value with the store's (``<lo>-<hi>.json`` from
  `poolplay.SolvedSelections`), so a store solved by another evaluation model can be
  scored against the same games.

The selection the games played is the equilibrium mixed with exploration (epsilon 0.25 in
generation), so a pair's mean score is not exactly the equilibrium's value even with no
noise. The slope and the ceiling are read with that in mind.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-6


# ------------------------------------------------------------------------------ extract


def rows_of_lines(lines: Iterable[bytes | str]) -> tuple[list[tuple[str, str, float, float, int]],
                                                       dict[str, int]]:
    """(team0, team1, side 0's score, selection value, game index) a finished pool game."""
    rows: list[tuple[str, str, float, float, int]] = []
    dropped = {"mirror": 0, "unfinished": 0, "no_value": 0, "no_pool": 0}
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        game = json.loads(line)
        pool = game.get("pool")
        if not pool:
            dropped["no_pool"] += 1
            continue
        if pool.get("mirror") or pool["teams"][0] == pool["teams"][1]:
            dropped["mirror"] += 1
            continue
        if game.get("outcome") is None:
            dropped["unfinished"] += 1
            continue
        value = game.get("selectionValue")
        if value is None:
            dropped["no_value"] += 1
            continue
        index = game.get("gameIndex")
        rows.append((str(pool["teams"][0]), str(pool["teams"][1]), float(game["outcome"]),
                     float(value), int(index) if index is not None else -1 - n))
    return rows, dropped


def _extract_file(path: str) -> tuple[list[tuple[str, str, float, float, int]], dict[str, int]]:
    with open(path, "rb") as fh:
        return rows_of_lines(fh)


def extract(args: argparse.Namespace) -> None:
    files = sorted(str(p) for d in args.games_dir for p in Path(d).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl under {args.games_dir}")
    rows: list[tuple[str, str, float, float, int]] = []
    dropped: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for k, (r, d) in enumerate(ex.map(_extract_file, files)):
            rows.extend(r)
            for key, v in d.items():
                dropped[key] = dropped.get(key, 0) + v
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(files)} files, {len(rows)} games", file=sys.stderr, flush=True)
    teams = sorted({t for r in rows for t in r[:2]})
    index = {t: i for i, t in enumerate(teams)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        teams=np.array(teams),
        t0=np.array([index[r[0]] for r in rows], dtype=np.int32),
        t1=np.array([index[r[1]] for r in rows], dtype=np.int32),
        outcome=np.array([r[2] for r in rows], dtype=np.float64),
        value=np.array([r[3] for r in rows], dtype=np.float64),
        game=np.array([r[4] for r in rows], dtype=np.int64),
        dropped=np.array(json.dumps(dropped)),
        source=np.array(json.dumps([str(d) for d in args.games_dir])),
    )
    print(f"{len(rows)} games, {len(teams)} teams, dropped {dropped} -> {out}", file=sys.stderr)


# ------------------------------------------------------------------------------ report


def canonical(t0: np.ndarray, t1: np.ndarray, outcome: np.ndarray, value: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each game from the seat of the pair's lower team index: (lo, hi, score, value)."""
    flip = t0 > t1
    lo = np.where(flip, t1, t0)
    hi = np.where(flip, t0, t1)
    y = np.where(flip, 1.0 - outcome, outcome)
    v = np.where(flip, 1.0 - value, value)
    return lo, hi, y, v


def store_values(store: Path, teams: list[str]) -> dict[tuple[int, int], float]:
    """The store's value of each pair, from the seat of the lower index in ``teams``."""
    index = {t: i for i, t in enumerate(teams)}
    found: dict[tuple[int, int], float] = {}
    for path in sorted(store.glob("*.json")):
        data = json.loads(path.read_bytes().decode("utf-8"))
        a, b = (index.get(t) for t in data["teams"])
        if a is None or b is None or a == b:
            continue
        value = float(data["entry"]["value"])
        found[(min(a, b), max(a, b))] = value if a < b else 1.0 - value
    return found


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def pair_table(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, v: np.ndarray, size: int
               ) -> dict[str, np.ndarray]:
    """Per pair (lo, hi): games, mean score, mean value and the value's spread."""
    key = lo.astype(np.int64) * size + hi
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    ybar = np.bincount(inv, weights=y) / n
    vbar = np.bincount(inv, weights=v) / n
    vsq = np.bincount(inv, weights=v * v) / n
    return {"lo": uniq // size, "hi": uniq % size, "n": n, "y": ybar, "v": vbar,
            "v_sd": np.sqrt(np.maximum(vsq - vbar * vbar, 0.0)), "inv": inv}


def pair_stats(p: dict[str, np.ndarray], min_games: int) -> dict[str, Any]:
    keep = p["n"] >= min_games
    n, yb, vb = p["n"][keep], p["y"][keep], p["v"][keep]
    var_obs = float(np.var(yb, ddof=1)) if len(yb) > 1 else float("nan")
    # The sampling variance of a pair's mean score (a draw scores 0.5, so this is an upper
    # bound by a hair).
    noise = float(np.mean(yb * (1 - yb) / np.maximum(n - 1, 1)))
    reliability = 1.0 - noise / var_obs if var_obs > 0 else float("nan")
    r = _pearson(vb, yb)
    slope, intercept = (np.polyfit(vb, yb, 1) if len(yb) > 2 else (float("nan"),) * 2)
    return {
        "pairs": int(keep.sum()),
        "pairsDropped": int((~keep).sum()),
        "gamesPerPair": {"min": float(n.min()), "median": float(np.median(n)),
                         "max": float(n.max())} if len(n) else {},
        "valueSpreadWithinPairMax": float(p["v_sd"][keep].max()) if len(n) else float("nan"),
        "pearson": r,
        "spearman": _spearman(vb, yb),
        "scoreVariance": var_obs,
        "noiseVariance": noise,
        "reliability": reliability,
        "pearsonCorrected": (r / math.sqrt(reliability)
                             if reliability and reliability > 0 else float("nan")),
        "slope": float(slope),
        "intercept": float(intercept),
        "meanAbsError": float(np.mean(np.abs(vb - yb))),
    }


def halves(game: np.ndarray) -> np.ndarray:
    """A fixed split of the games into two halves (a hash of the game index)."""
    h = (game.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(63)
    return h.astype(bool)


def game_stats(p: dict[str, np.ndarray], y: np.ndarray, v: np.ndarray, half: np.ndarray
               ) -> dict[str, Any]:
    """The value as a forecast of each game, against the coin and the other half's pair mean."""
    inv = p["inv"]
    m = len(p["n"])
    other = np.empty_like(y)
    for side in (False, True):
        mine = half == side
        nn = np.bincount(inv[~mine], minlength=m).astype(np.float64)
        ss = np.bincount(inv[~mine], weights=y[~mine], minlength=m)
        mean = np.where(nn > 0, ss / np.maximum(nn, 1), 0.5)
        other[mine] = mean[inv[mine]]
    coin = np.full_like(y, 0.5)
    return {
        "games": int(len(y)),
        "brier": {"value": _brier(v, y), "coin": _brier(coin, y), "otherHalf": _brier(other, y)},
        "logloss": {"value": _logloss(v, y), "coin": _logloss(coin, y),
                    "otherHalf": _logloss(other, y)},
        "calibration": calibration(v, y),
    }


def calibration(v: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.quantile(v, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        sel = which == b
        if sel.any():
            out.append({"n": int(sel.sum()), "value": float(v[sel].mean()),
                        "score": float(y[sel].mean())})
    return out


def team_strength(lo: np.ndarray, hi: np.ndarray, val: np.ndarray, size: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Each team's mean over the pairs it is in of ``val`` (its own seat), and the count."""
    total = np.bincount(lo, weights=val, minlength=size) + np.bincount(
        hi, weights=1.0 - val, minlength=size)
    count = np.bincount(lo, minlength=size) + np.bincount(hi, minlength=size)
    return total / np.maximum(count, 1), count


def team_stats(p: dict[str, np.ndarray], lo: np.ndarray, hi: np.ndarray, y: np.ndarray,
               half: np.ndarray, size: int, min_games: int, tops: Iterable[int]
               ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    keep = p["n"] >= min_games
    plo, phi = p["lo"][keep], p["hi"][keep]
    by_value, count = team_strength(plo, phi, p["v"][keep], size)
    by_score, _ = team_strength(plo, phi, p["y"][keep], size)
    # The scores' own reliability: the same strength from each half of the games.
    split = []
    for side in (False, True):
        sel = half == side
        q = pair_table(lo[sel], hi[sel], y[sel], y[sel], size)
        pk = np.isin(q["lo"] * size + q["hi"], plo.astype(np.int64) * size + phi)
        split.append(team_strength(q["lo"][pk], q["hi"][pk], q["y"][pk], size)[0])
    present = count > 0
    r_half = _pearson(split[0][present], split[1][present])
    rel = 2 * r_half / (1 + r_half) if r_half == r_half and r_half > -1 else float("nan")
    a, b = by_value[present], by_score[present]
    overlap = {}
    for k in tops:
        if k <= present.sum():
            top_v = set(np.argsort(-a)[:k])
            top_s = set(np.argsort(-b)[:k])
            overlap[str(k)] = len(top_v & top_s)
    r = _pearson(a, b)
    return ({
        "teams": int(present.sum()),
        "pearson": r,
        "spearman": _spearman(a, b),
        "scoreSplitHalfPearson": r_half,
        "scoreReliability": rel,
        "pearsonCorrected": r / math.sqrt(rel) if rel and rel > 0 else float("nan"),
        "scoreSpread": float(np.std(b, ddof=1)),
        "valueSpread": float(np.std(a, ddof=1)),
        "topOverlap": overlap,
    }, {"value": by_value, "score": by_score, "present": present})


def report(args: argparse.Namespace) -> None:
    data = np.load(args.rows)
    teams = [str(t) for t in data["teams"]]
    size = len(teams)
    lo, hi, y, v = canonical(data["t0"], data["t1"], data["outcome"], data["value"])
    source = "recorded"
    if args.store:
        found = store_values(Path(args.store), teams)
        have = np.array([(a, b) in found for a, b in zip(lo, hi, strict=True)])
        if not have.all():
            print(f"the store lacks {int((~have).sum())} games' pairs; they are dropped",
                  file=sys.stderr)
        lo, hi, y, game = lo[have], hi[have], y[have], data["game"][have]
        v = np.array([found[(a, b)] for a, b in zip(lo, hi, strict=True)])
        source = str(args.store)
    else:
        game = data["game"]
    half = halves(game)
    p = pair_table(lo, hi, y, v, size)
    teams_out, strength = team_stats(p, lo, hi, y, half, size, args.min_games, args.top)
    result = {
        "rows": str(args.rows),
        "valueSource": source,
        "dropped": json.loads(str(data["dropped"])),
        "pair": pair_stats(p, args.min_games),
        "game": game_stats(p, y, v, half),
        "team": teams_out,
        "teamTable": sorted(
            ({"team": teams[i], "value": float(strength["value"][i]),
              "score": float(strength["score"][i])}
             for i in range(size) if strength["present"][i]),
            key=lambda row: -row["score"]),
    }
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes((text + "\n").encode("utf-8"))
    print(summary(result))


def summary(result: dict[str, Any]) -> str:
    pr, gm, tm = result["pair"], result["game"], result["team"]
    lines = [
        f"value: {result['valueSource']}",
        f"pairs {pr['pairs']} (games/pair median {pr['gamesPerPair'].get('median')}), "
        f"games {gm['games']}",
        f"pair  r {pr['pearson']:.3f}  rho {pr['spearman']:.3f}  reliability "
        f"{pr['reliability']:.3f}  r corrected {pr['pearsonCorrected']:.3f}  "
        f"slope {pr['slope']:.3f}  MAE {pr['meanAbsError']:.4f}",
        f"game  brier value {gm['brier']['value']:.5f} coin {gm['brier']['coin']:.5f} "
        f"other half {gm['brier']['otherHalf']:.5f}",
        f"team  r {tm['pearson']:.3f}  rho {tm['spearman']:.3f}  score reliability "
        f"{tm['scoreReliability']:.3f}  r corrected {tm['pearsonCorrected']:.3f}  "
        f"top overlap {tm['topOverlap']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--games-dir", nargs="+", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--jobs", type=int, default=1)
    ex.set_defaults(func=extract)
    rp = sub.add_parser("report")
    rp.add_argument("--rows", required=True)
    rp.add_argument("--store", help="score a selection-solved store's values instead")
    rp.add_argument("--min-games", type=int, default=20)
    rp.add_argument("--top", type=int, nargs="+", default=[5, 10])
    rp.add_argument("--out")
    rp.set_defaults(func=report)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
