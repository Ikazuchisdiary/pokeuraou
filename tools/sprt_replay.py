"""Would a sequential test have stopped these matches -- when, and saying what?

IKA-89. Every match this project has judged by ran to a fixed count, usually 12,000
games. This replays a sequential probability ratio test (`pokeuraou.sprt`) over the pairs
each match actually played, in game-index order -- the order the queue hands games out and
the order a live test consumes them in -- and reports where it would have stopped, with
which answer, and whether that answer survives the fixed count's own interval.

One realised path is one draw of the stopping time, and a noisy one: the same effect can
stop a run at the 400th pair or the 2,000th. `--bootstrap` redraws the path from the
match's own pair distribution to say what a change of this size costs a sequential test in
general, not only what this run happened to cost. It answers a different question from
the replay, and the two are printed apart.

The reading of the records is `tools/paired_result.py`'s, field for field, so the fixed
count printed here is the number that tool reports -- which is the check that the pairs
are the right pairs.

    uv run python tools/sprt_replay.py data/matches/ika66-b-vs-a data/matches/ika73-d-vs-a
    uv run python tools/sprt_replay.py --bounds 0 10 --bounds 0 15 --bootstrap 1000 \\
        data/matches/ika66-*
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import sprt  # noqa: E402


def fixed_count(scores: list[float]) -> tuple[float, float]:
    """The paired estimate `tools/paired_result.py` prints: mean and 1.96 standard errors."""
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    return mean, 1.96 * math.sqrt(var / n)


def verdict(decision: str | None, lo: float, hi: float, elo0: float, elo1: float) -> str:
    """How the fixed count's interval reads the decision, in Elo.

    The test guards against two mistakes: passing a change worth `elo0` or less, and failing
    one worth `elo1` or more. The fixed count convicts a decision only when its whole
    interval says the mistake was made. Anything short of that is an answer the test was
    allowed to give -- including H0 on a change the interval puts above zero, which is the
    reading to watch: H0 says "not worth elo1", not "worth nothing".
    """
    if decision == "H1":
        if hi <= elo0:
            return f"CONTRADICTED: the fixed count puts it at most {elo0:+g}"
        return "agrees" if lo > elo0 else "allowed (the fixed count cannot exclude elo0)"
    if decision == "H0":
        if lo >= elo1:
            return f"CONTRADICTED: the fixed count puts it at least {elo1:+g}"
        if lo > elo0:
            return (
                f"allowed, but read it as 'not {elo1:+g}': the fixed count says it beats "
                f"{elo0:+g}"
            )
        return "agrees" if hi < elo1 else "allowed (the fixed count cannot exclude elo1)"
    return "undecided at the end of the run: the fixed count is the answer"


def bootstrap(
    counts: np.ndarray,
    elo0: float,
    elo1: float,
    *,
    alpha: float,
    beta: float,
    cap: int,
    runs: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Redraw `runs` paths of at most `cap` pairs from the run's own pair distribution."""
    stops, decisions = sprt.simulate(
        counts / counts.sum(), elo0, elo1, alpha=alpha, beta=beta, cap=cap, runs=runs, rng=rng
    )
    games = 2 * stops
    decided = games[decisions != 0]
    return {
        "h1": float((decisions == 1).mean()),
        "h0": float((decisions == -1).mean()),
        "undecided": float((decisions == 0).mean()),
        "mean_games": float(games.mean()),
        "median_games": float(np.median(decided)) if decided.size else float(2 * cap),
        "p90_games": float(np.percentile(games, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument(
        "--bounds",
        nargs=2,
        type=float,
        action="append",
        metavar=("ELO0", "ELO1"),
        help="hypotheses in logistic Elo for the tested arm; repeatable. Default: the two "
        "IKA-89 registered, (0, +10) and (0, +15).",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="redraw this many paths per match and bound, capped at the match's own pair "
        "count: what a change of this size costs, rather than what this run cost",
    )
    ap.add_argument("--seed", type=int, default=89)
    args = ap.parse_args()
    pairs_of_bounds = [tuple(b) for b in (args.bounds or [(0.0, 10.0), (0.0, 15.0)])]
    rng = np.random.default_rng(args.seed)

    summary: list[tuple[str, int, list[tuple[sprt.Stop, dict[str, float] | None]]]] = []
    for directory in args.dirs:
        tail = sprt.PairTail(directory)
        tail.poll()
        complete = tail.complete()
        print(f"\n{directory}")
        singles = sum(1 for seats in tail.seats.values() if len(seats) == 1)
        print(
            f"  {tail.records} records, {len(complete)} complete pairs"
            + (f", {singles} games waiting for their partner" if singles else "")
            + (f", {tail.unindexed} without a gameIndex (not pairable)" if tail.unindexed else "")
            + (f", {tail.ties} ties" if tail.ties else "")
        )
        if not complete:
            continue
        scores = [score for _index, score in complete]
        mean, half = fixed_count(scores)
        elo, lo, hi = (
            sprt.elo_of(mean),
            sprt.elo_of(max(mean - half, 1e-6)),
            sprt.elo_of(min(mean + half, 1 - 1e-6)),
        )
        split = sum(1 for s in scores if s == 0.5) / len(scores)
        print(
            f"  fixed count ({2 * len(scores)} games): tested arm {mean:.2%} +-{half:.2%} "
            f"paired, Elo {elo:+.1f} [{lo:+.1f}, {hi:+.1f}]; {split:.0%} of pairs split"
        )
        counts = np.bincount([sprt.slot(s) for s in scores], minlength=3).astype(np.float64)
        rows: list[tuple[sprt.Stop, dict[str, float] | None]] = []
        for elo0, elo1 in pairs_of_bounds:
            stop = sprt.replay(scores, elo0, elo1, alpha=args.alpha, beta=args.beta)
            label = f"SPRT({elo0:+g}, {elo1:+g})"
            if stop.decision is None:
                print(f"  {label}  no decision in {stop.pairs} pairs (LLR {stop.llr:+.2f})")
            else:
                print(
                    f"  {label}  {stop.decision} at pair {stop.pairs} = {2 * stop.pairs:,} games "
                    f"({stop.pairs / len(scores):.1%} of the run), LLR {stop.llr:+.2f}"
                )
            print(f"  {'':{len(label)}}  {verdict(stop.decision, lo, hi, elo0, elo1)}")
            boot = None
            if args.bootstrap:
                boot = bootstrap(
                    counts, elo0, elo1, alpha=args.alpha, beta=args.beta,
                    cap=len(scores), runs=args.bootstrap, rng=rng,
                )
                print(
                    f"  {'':{len(label)}}  a change this size, {args.bootstrap} redrawn runs: "
                    f"H1 {boot['h1']:.1%}, H0 {boot['h0']:.1%}, undecided at "
                    f"{2 * len(scores):,} games {boot['undecided']:.1%}; median "
                    f"{boot['median_games']:,.0f} games, 90% within {boot['p90_games']:,.0f}, "
                    f"mean {boot['mean_games']:,.0f}"
                )
            rows.append((stop, boot))
        summary.append((directory.name, 2 * len(scores), rows))

    if len(summary) < 2:
        return
    print("\nsummary, games played against games a registered test would have needed")
    header = f"  {'match':<34} {'fixed':>7}"
    for elo0, elo1 in pairs_of_bounds:
        header += f"   {f'({elo0:+g},{elo1:+g}) replay':>20}"
        if args.bootstrap:
            header += f" {'expected':>9}"
    print(header)
    totals = [0.0] * (1 + 2 * len(pairs_of_bounds))
    for name, games, rows in summary:
        line = f"  {name:<34} {games:>7,}"
        totals[0] += games
        for column, (stop, boot) in enumerate(rows):
            used = 2 * stop.pairs
            totals[1 + 2 * column] += used
            line += f"   {(stop.decision or '--') + f' {used:,}':>20}"
            if boot is not None:
                totals[2 + 2 * column] += boot["mean_games"]
                line += f" {boot['mean_games']:>9,.0f}"
        print(line)
    line = f"  {'total':<34} {totals[0]:>7,.0f}"
    for column in range(len(pairs_of_bounds)):
        used = totals[1 + 2 * column]
        line += f"   {f'{used:,.0f} ({totals[0] / used:.1f}x fewer)':>20}"
        if args.bootstrap:
            expected = totals[2 + 2 * column]
            line += f" {expected:>9,.0f}"
    print(line)


if __name__ == "__main__":
    main()
