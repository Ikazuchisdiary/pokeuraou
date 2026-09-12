"""One scale for every agent, fitted from every match ever recorded.

Each measurement so far has been a pair: "+2.9 [+0.5, +5.3] against value-gen234". That is
the right answer to the question it was asked and it does not compose. It cannot say how
much stronger the current model is than the one three generations back without playing
that match, it cannot put width 48 and the candidate ranking on the same axis, and every
new arm needs a baseline chosen for it in advance.

A rating is that answer generalised. The model is Bradley-Terry with a side term:

    P(side 0 wins) = sigmoid(r[agent on side 0] - r[agent on side 1] + s)

which is exactly the R-and-a decomposition every match here has done by hand, with two
players replaced by as many as have ever played. `s` is the seat advantage -- side 0 is
always our roster and side 1 always a tournament team, so it is real and has been measured
between 50.7% and 54.9% -- and fitting it rather than assuming it away is what lets games
from different matchups pool.

An *agent* is a model together with the search that ran it, because that is what played:
width 48 beat width 24 with the same model by +5.5, so a rating keyed on the model alone
would pool two things that are five points apart. `provenance.agent_name` builds the key.

## What it cannot do

A Bradley-Terry rating assumes strength is one number and that beating A implies as much
against B as it does against C. Matchup cycles break that, and this game has them -- a
three-way cycle among selections is documented in `configs/knowledge/`. The fit reports
its own residuals per pair so a reader can see where the assumption fails rather than
learn it later; a pair whose observed win rate sits far from the fitted one is a cycle
showing through, not noise.

Nothing here plays a game. It reads the games that were already played and recorded.

    uv run python tools/ratings.py
    uv run python tools/ratings.py --anchor value-gen2345.pt/w24 --logit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.provenance import agent_name

#: (side 0 agent, side 1 agent, side 0's wins, games played). A single game is one of
#: these with `played` 1, which is what lets a per-seat total and a per-game record sit in
#: the same list.
Observation = tuple[str, str, float, int]

#: Elo's scale: 400 points is a factor of ten in odds. Logits are what the model is in.
ELO_PER_LOGIT = 400.0 / math.log(10.0)

#: Zero is the parameter-free objective. It is the only competitor here that cannot drift:
#: `hp-share` has no weights, no training set and no version, so a rating measured against
#: it means the same thing next month as it does today -- which a rating anchored on
#: whichever model happens to be current does not. It also reads as the quantity that
#: matters: what the learned machinery is worth over having none of it.
ANCHOR = "hp-share/w24"


def recover_old_axes(source: dict) -> int:
    """Fill in `depths` and `rankings` for records written before they existed.

    Those two axes were once carried only in the free-text seat label -- `...@leafrank =
    side 0`, `...@d2 = side 1` -- and a record that lacks them reads as the default
    configuration. Which means the ranking match's two arms, the depth match's two arms and
    every plain match all arrive under one name, and a rating pools three things that were
    measured 2.9 and 2.1 points apart. The first fit did exactly that: five agents, and one
    of them holding 9,240 games.

    So the label is parsed, once, here, and the number of records that needed it is
    reported. This is archaeology on data already on disk, not a format: anything written
    from now on carries the fields, and `agent_name` reads only those.

    Returns 1 if the record needed repairing, 0 otherwise.
    """
    if "rankings" in source and "depths" in source:
        return 0
    label = str(source.get("seat", ""))
    arm, _, side_text = label.partition(" = side ")
    side = 0 if side_text.strip() == "0" else 1
    other = 1 - side
    depths = [1, 1]
    rankings = ["damage", "damage"]
    if "@leafrank" in arm:
        rankings[side] = "leaf"
    if "@d2" in arm:
        depths[side] = 2
        depths[other] = 1
    source.setdefault("depths", depths)
    source.setdefault("rankings", rankings)
    return 1


def read_games(root: Path) -> tuple[list[Observation], int]:
    """One observation per recorded game, and how many records needed repairing."""
    out: list[Observation] = []
    repaired = 0
    for path in sorted(root.glob("**/games-seed*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = game.get("provenance")
                outcome = game.get("outcome")
                if not source or outcome is None:
                    continue
                repaired += recover_old_axes(source)
                out.append(
                    (agent_name(source, 0), agent_name(source, 1), float(outcome), 1)
                )
    return out, repaired


def read_summaries(root: Path) -> list[Observation]:
    """Matches that kept only their per-seat totals, which is everything before yesterday.

    `--games-out` was wired into the launcher a day ago, so the earlier matches -- the
    width comparison that established 24 over 16, and the first two generation matches --
    left a row per seat and nothing else. A row is "n games, k of them won by this arm",
    which is exactly what the likelihood consumes; one game is the same row with n = 1. So
    the whole history goes on the scale, including the only games ever played at width 16.

    A directory that has game-level records is skipped entirely. Its summaries describe the
    same games, and counting a match twice would shrink its interval by root two while
    telling the reader nothing new.

    Two schemas, because the width match and the generation match were written apart:
    `wide`/`narrow`/`wide_wins` against `model`/`baseline`/`limit`/`gen2_wins`. Both say
    which side the named arm sat on in `seat`, which is the part that matters.
    """
    out: list[Observation] = []
    for directory in sorted({p.parent for p in root.glob("**/seed*.jsonl")}):
        if any(directory.glob("games-seed*.jsonl")):
            continue
        for path in sorted(directory.glob("seed*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                played = int(row.get("played", 0))
                if not played:
                    continue
                first = str(row.get("seat", "")).endswith("side 0")
                if "wide" in row and "narrow" in row:
                    model = row.get("model", "?")
                    arm = f"{model}/w{row['wide']}"
                    other = f"{model}/w{row['narrow']}"
                    wins = int(row.get("wide_wins", 0))
                else:
                    limit = row.get("limit", "?")
                    arm = f"{row.get('model', '?')}/w{limit}"
                    other = f"{row.get('baseline') or row.get('objective', '?')}/w{limit}"
                    depth = int(row.get("depth", 1))
                    other_depth = int(row.get("baselineDepth", 1))
                    if depth != 1:
                        arm += f"/d{depth}"
                    if other_depth != 1:
                        other += f"/d{other_depth}"
                    wins = int(row.get("gen2_wins", row.get("wins", 0)))
                # The stored count is the named arm's; the fit wants side 0's.
                if first:
                    out.append((arm, other, float(wins), played))
                else:
                    out.append((other, arm, float(played - wins), played))
    return out


def fit(
    games: list[Observation],
    *,
    anchor: str | None = None,
    iterations: int = 500,
    prior: float = 1.0,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Ratings in logits, the seat advantage, and a standard error per agent.

    Gradient ascent on the log likelihood rather than a solver, because the problem is
    tiny and a dependency is not worth it. The prior is a weak pull towards zero: two
    agents that only ever beat each other would otherwise run off to infinity, and an
    agent with one game would be reported with more confidence than it has earned.

    The error is of the *difference from the anchor*, which needs the whole covariance
    rather than its diagonal. Only differences are observable -- adding a constant to
    every rating changes no prediction -- so a marginal error printed beside an anchored
    rating would be answering a question nobody can ask. The anchor's own is exactly zero,
    which is what holding it there means.
    """
    names = sorted({a for a, _b, _w, _n in games} | {b for _a, b, _w, _n in games})
    index = {name: i for i, name in enumerate(names)}
    rating = np.zeros(len(names))
    seat = 0.0
    rows = np.array([index[a] for a, _b, _w, _n in games])
    cols = np.array([index[b] for _a, b, _w, _n in games])
    wins = np.array([w for _a, _b, w, _n in games], dtype=np.float64)
    count = np.array([n for _a, _b, _w, n in games], dtype=np.float64)
    total = float(count.sum())

    step = 0.05
    for _ in range(iterations):
        predicted = 1.0 / (1.0 + np.exp(-(rating[rows] - rating[cols] + seat)))
        # An aggregated row of n games with k wins contributes exactly what those n games
        # would have, which is why a summary and its games are interchangeable here -- and
        # why counting both would be counting the same games twice.
        error = wins - count * predicted
        gradient = np.zeros(len(names))
        np.add.at(gradient, rows, error)
        np.add.at(gradient, cols, -error)
        gradient -= prior * rating
        rating += step * gradient / max(total / len(names), 1.0)
        seat += step * error.sum() / total

    # Curvature at the optimum, for a standard error per agent. Each game contributes
    # p(1-p) to both of its players; the prior contributes its own weight.
    predicted = 1.0 / (1.0 + np.exp(-(rating[rows] - rating[cols] + seat)))
    weight = count * predicted * (1.0 - predicted)
    hessian = np.diag(np.full(len(names), float(prior)))
    np.add.at(hessian, (rows, rows), weight)
    np.add.at(hessian, (cols, cols), weight)
    np.add.at(hessian, (rows, cols), -weight)
    np.add.at(hessian, (cols, rows), -weight)
    covariance = np.linalg.inv(hessian)

    if anchor is None:
        errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    else:
        if anchor not in index:
            raise SystemExit(f"no agent named {anchor!r}; try one of {names}")
        a = index[anchor]
        rating = rating - rating[a]
        errors = np.sqrt(
            np.clip(
                np.diag(covariance) + covariance[a, a] - 2.0 * covariance[:, a], 0.0, None
            )
        )
    return (
        dict(zip(names, rating, strict=True)),
        seat,
        dict(zip(names, errors, strict=True)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matches", type=Path, default=Path("data/matches"),
        help="directory of recorded matches",
    )
    ap.add_argument(
        "--anchor",
        default=ANCHOR,
        help="agent held at zero. Empty string for a mean-zero scale instead.",
    )
    ap.add_argument("--logit", action="store_true", help="print logits instead of Elo")
    ap.add_argument("--min-games", type=int, default=1)
    args = ap.parse_args()

    games, repaired = read_games(args.matches)
    summaries = read_summaries(args.matches)
    games += summaries
    if not games:
        raise SystemExit(f"no recorded matches under {args.matches}")
    played: Counter[str] = Counter()
    for a, b, _w, n in games:
        played[a] += n
        played[b] += n

    rating, seat, errors = fit(games, anchor=args.anchor or None)
    scale = 1.0 if args.logit else ELO_PER_LOGIT
    unit = "logit" if args.logit else "Elo"
    names = sorted(rating, key=lambda n: -rating[n])

    total = sum(n for _a, _b, _w, n in games)
    from_rows = sum(n for _a, _b, _w, n in summaries)
    print(f"{total} games, {len(rating)} agents")
    if from_rows:
        print(
            f"  {from_rows} of them from matches that kept only per-seat totals, which is "
            "every match run before the games themselves were recorded"
        )
    if repaired:
        print(
            f"  {repaired} of them predate the per-side depth and ranking fields; "
            "their configuration was read back out of the seat label"
        )
    print(
        f"seat advantage {seat:+.3f} logit = side 0 wins "
        f"{100 / (1 + math.exp(-seat)):.1f}% between equals"
    )
    if args.anchor:
        print(
            f"anchored at {args.anchor} = 0, so a rating is what an agent is worth "
            "over the parameter-free objective"
        )
    print(f"\n  {'agent':<34} {unit:>9}  {'+-':>6}  {'games':>6}")
    for name in names:
        if played[name] < args.min_games:
            continue
        half = 1.96 * errors[name] * scale
        print(f"  {name:<34} {rating[name] * scale:>9.1f}  {half:>6.1f}  {played[name]:>6}")

    # Where the one-number assumption is failing, if it is.
    pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for a, b, w, n in games:
        pairs[(a, b)].extend([1.0] * int(round(w)) + [0.0] * int(n - round(w)))
    print("\n  fitted against observed, per matchup (a gap is a cycle, not noise)")
    print(f"  {'side 0':<26} {'side 1':<26} {'n':>5} {'seen':>6} {'fit':>6}  {'gap':>6}")
    worst = sorted(
        pairs.items(),
        key=lambda kv: -abs(
            float(np.mean(kv[1]))
            - 1.0 / (1.0 + math.exp(-(rating[kv[0][0]] - rating[kv[0][1]] + seat)))
        ),
    )
    for (a, b), results in worst[:10]:
        seen = float(np.mean(results))
        expected = 1.0 / (1.0 + math.exp(-(rating[a] - rating[b] + seat)))
        print(
            f"  {a:<26} {b:<26} {len(results):>5} {seen * 100:>5.1f}% "
            f"{expected * 100:>5.1f}% {(seen - expected) * 100:>+6.1f}"
        )


if __name__ == "__main__":
    main()
