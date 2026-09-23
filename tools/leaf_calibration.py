"""How far is a static leaf estimate from what the search does, and where.

The selection book's matrix is 8,100 cells per spread class and every cell is one leaf
evaluation of a turn-1 position -- `solve_selection` says so: "No turn is resolved." On
2026-09-19 the board disagreed with that matrix by 9.7 points in absolute terms, so the
question is which part of the leaf's job is failing.

Two hypotheses, and they call for different fixes:

  coverage   the leaf is fine where it has seen games and wrong where it has not. The
             pool's top six selections carry 70.4% of games, so most of the 8,100 PAIRS
             the matrix asks about have no game behind them at all. Fixed with data --
             forced-selection generation over the thin pairs.
  capacity   a static evaluation cannot represent what a search will do, anywhere. This
             is the worry left over from the policy head (TODO G1), where a network
             asked to imitate the search's ranking lost 4.1 points at width 16 -- though
             the record's actual finding is narrower than the memory of it: after the
             comparison was fixed the policy BEAT leaf-ranking at width 24 and above, and
             what hurt it at 16 was `narrow`'s coverage rule, not an inability to learn.

Three numbers per bucket, and the third is the one the book inherits:

    leaf vs outcome          is the leaf calibrated at all
    searchValue vs outcome   is the search calibrated, as a reference point
    leaf vs searchValue      what a cell of the matrix gives up by not searching

`searchValue` is already in the log, so the search side costs nothing; only the leaf pass
is new. Buckets are by turn, and turn 1 is bucketed again by how many games in the pool
shared that (ours, theirs) selection pair, because that is what separates the two
hypotheses.

    uv run python tools/leaf_calibration.py --value data/models/value-gen11L.pt \\
        --dir data/selfplay-gen11L --positions 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

TURN_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("turn 1", 1, 1),
    ("turn 2", 2, 2),
    ("turn 3-4", 3, 4),
    ("turn 5-8", 5, 8),
    ("turn 9+", 9, 10**6),
)
PAIR_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0 games", 0, 0),
    ("1-5", 1, 5),
    ("6-20", 6, 20),
    ("21+", 21, 10**9),
)


def bucket_of(value: int, buckets) -> str:  # noqa: ANN001
    for name, lo, hi in buckets:
        if lo <= value <= hi:
            return name
    return buckets[-1][0]


def unseen_of_foe(game: dict, decision: dict) -> int | None:
    """How many of side 1's four side 0's search had not seen at this decision.

    Read off `shownIdentities` (IKA-127), which in the open game is the whole four and so
    gives 0. None for a record written before the field, where the answer would have to
    be replayed -- `pokeuraou.selfplay.replay_shown` does that, at a position parse per
    decision, which this tool does not spend.
    """
    shown = decision.get("shownIdentities")
    if shown is None:
        return None
    return len(game["foeTeam"]) - len(shown[1])


def report(title: str, rows: dict[str, list[tuple[float, float, float]]], order) -> None:  # noqa: ANN001
    """One line per bucket: n, the three comparisons, and the leaf's Brier score."""
    print(f"\n  {title}")
    print(
        f"    {'bucket':<10} {'n':>5}  {'leaf':>6} {'search':>7} {'actual':>7}  "
        f"{'leaf-act':>9} {'srch-act':>9} {'leaf-srch':>10}  {'brier':>6} {'skill':>6}"
    )
    for name in order:
        items = rows.get(name) or []
        if not items:
            continue
        leaf = np.asarray([a for a, _s, _o in items], dtype=np.float64)
        srch = np.asarray([s for _a, s, _o in items], dtype=np.float64)
        out = np.asarray([o for _a, _s, o in items], dtype=np.float64)
        ok = ~np.isnan(srch)
        brier = float(np.mean((leaf - out) ** 2))
        # Against the only predictor that needs no model: this bucket's own base rate.
        # Bias and skill are different failures and a tool that prints one without the
        # other invites "unbiased" to be read as "useful" -- which is the reading that
        # was about to be given to a leaf scoring 0.223 where a coin scores 0.250.
        base = float(np.mean((out.mean() - out) ** 2))
        skill = 1.0 - brier / base if base > 1e-12 else float("nan")
        print(
            f"    {name:<10} {len(items):>5}  {leaf.mean():>5.1%} "
            f"{(srch[ok].mean() if ok.any() else float('nan')):>6.1%} {out.mean():>6.1%}  "
            f"{leaf.mean() - out.mean():>+8.1%} "
            f"{((srch[ok] - out[ok]).mean() if ok.any() else float('nan')):>+8.1%} "
            f"{((leaf[ok] - srch[ok]).mean() if ok.any() else float('nan')):>+9.1%}  "
            f"{brier:>6.3f} {skill:>+6.1%}"
        )


def ceiling(args, paths, leaf) -> None:  # noqa: ANN001
    """The best any function of the matchup could do, against what the leaf does."""
    groups: dict[tuple, list[tuple[dict, float]]] = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                game = json.loads(line)
                if not game.get("targetIsRealOutcome") or game.get("outcome") is None:
                    continue
                first = next(
                    (
                        d
                        for d in game["decisions"]
                        if d["kind"] == "move" and d["turn"] == 1
                    ),
                    None,
                )
                if first is None:
                    continue
                key = (
                    game.get("foeArchetype") or "?",
                    tuple(game.get("ownPick", ())),
                    tuple(game.get("foePick", ())),
                )
                groups[key].append((first, float(game["outcome"])))
    kept = {k: v for k, v in groups.items() if len(v) >= args.ceiling}
    items = [row for rows in kept.values() for row in rows]
    print(
        f"\n  ceiling: {len(kept)} matchups with >= {args.ceiling} games, "
        f"{len(items)} games, out of {len(groups)} matchups"
    )
    if len(items) < 50:
        print("    too few to say anything; lower --ceiling")
        return

    out = np.asarray([o for _d, o in items], dtype=np.float64)
    # Leave-one-out, so the oracle never sees the game it is scored on. Without it the
    # oracle's Brier falls by roughly 1/n per group and the ceiling comes out flattering.
    oracle = np.empty(len(items), dtype=np.float64)
    at = 0
    for rows in kept.values():
        ys = np.asarray([o for _d, o in rows], dtype=np.float64)
        total, n = ys.sum(), len(ys)
        oracle[at : at + n] = (total - ys) / (n - 1)
        at += n

    values = np.asarray(leaf([Position.from_json(d["position"]) for d, _o in items]))
    base = float(np.mean((out.mean() - out) ** 2))
    b_oracle = float(np.mean((oracle - out) ** 2))
    b_leaf = float(np.mean((values - out) ** 2))

    # The leave-one-out oracle is scored on an estimate of its own group's rate, so it
    # carries that estimate's variance: its expected error is p(1-p)*n/(n-1) where a
    # true oracle's is p(1-p). Left uncorrected it reported the leaf as getting 112% of
    # the ceiling, which is not a thing. k(n-k)/(n(n-1)) is the unbiased estimator of
    # p(1-p) per group, and pooling it by group size gives the ceiling the LOO number
    # is a noisy floor for.
    weighted = 0.0
    for rows in kept.values():
        ys = np.asarray([o for _d, o in rows], dtype=np.float64)
        n, k = len(ys), float(ys.sum())
        weighted += n * (k * (n - k) / (n * (n - 1)))
    b_true = weighted / len(items)

    print(f"    {'':<28} {'brier':>6} {'skill':>7}")
    print(f"    {'base rate only':<28} {base:>6.3f} {0.0:>+7.1%}")
    print(f"    {'leaf':<28} {b_leaf:>6.3f} {1 - b_leaf / base:>+7.1%}")
    print(
        f"    {'oracle, leave-one-out':<28} {b_oracle:>6.3f} "
        f"{1 - b_oracle / base:>+7.1%}   (a floor: carries its own estimate's noise)"
    )
    print(
        f"    {'oracle, unbiased':<28} {b_true:>6.3f} "
        f"{1 - b_true / base:>+7.1%}   (the ceiling)"
    )
    got = 1 - b_leaf / base
    top = 1 - b_true / base
    if top > 1e-9:
        print(f"    the leaf is getting {got / top:.0%} of what the matchup determines")
    print(
        "    The oracle is the best a turn-1 evaluation could ever be, because it knows\n"
        "    which matchup this is and nothing else -- the same information the selection\n"
        "    matrix has. Whatever it leaves on the table is play, not prediction."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--value", type=Path, nargs="+", required=True)
    ap.add_argument("--dir", type=Path, default=Path("data/selfplay-gen11L"))
    ap.add_argument("--positions", type=int, default=4000)
    ap.add_argument(
        "--ceiling",
        type=int,
        default=0,
        metavar="MIN_GAMES",
        help="estimate how much turn-1 skill is achievable AT ALL. Groups the games by "
        "(opponent archetype, our selection, their selection), keeps groups with at "
        "least this many, and scores the oracle that knows each group's own win rate -- "
        "leave-one-out, so it is not scored on the game it is predicting. A leaf cannot "
        "beat that, and if the oracle's skill is small the matchup simply does not "
        "determine the outcome and no amount of work on the leaf will help.",
    )
    ap.add_argument(
        "--only-turn",
        type=int,
        default=None,
        help="sample only this turn. Turn 1 is a tenth of the move decisions and the "
        "whole of the selection matrix, so the pair buckets need a run of their own to "
        "have any n in them.",
    )
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    paths = sorted(args.dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no *.jsonl under {args.dir}")

    # Pass one: every game's selection pair, so a position can be told how much company
    # it had. This reads the whole pool and keeps nothing but the counter.
    pairs: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    games = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                game = json.loads(line)
                if "ownPick" not in game or "foePick" not in game:
                    continue
                pairs[(tuple(game["ownPick"]), tuple(game["foePick"]))] += 1
                games += 1
    print(f"  {games} games over {len(paths)} files, {len(pairs)} distinct selection pairs")
    if games:
        top = sum(c for _k, c in pairs.most_common(6))
        print(f"  the six most common pairs carry {top / games:.1%}")

    # Pass two: sample positions. Reservoir-free -- the pool is read twice anyway, and a
    # fixed stride keeps the sample reproducible and spread over every file.
    rng = np.random.default_rng(args.seed)
    wanted = args.positions
    picked: list[tuple[dict, float, int, int, int | None]] = []
    seen = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                game = json.loads(line)
                if not game.get("targetIsRealOutcome") or game.get("outcome") is None:
                    continue
                key = (tuple(game.get("ownPick", ())), tuple(game.get("foePick", ())))
                company = pairs.get(key, 0) - 1  # not counting this game itself
                for decision in game["decisions"]:
                    if decision["kind"] != "move":
                        continue
                    if args.only_turn is not None and decision["turn"] != args.only_turn:
                        continue
                    seen += 1
                    item = (
                        decision,
                        float(game["outcome"]),
                        int(decision["turn"]),
                        company,
                        unseen_of_foe(game, decision),
                    )
                    if len(picked) < wanted:
                        picked.append(item)
                    else:
                        j = int(rng.integers(0, seen))
                        if j < wanted:
                            picked[j] = item
    print(f"  {seen} move decisions, {len(picked)} sampled")
    if not picked:
        raise SystemExit("nothing to score")

    import torch

    from pokeuraou.value import BatchedValue, load_ensemble

    reg = load_regulation(args.format)
    register_mega_stones(reg)
    encoder = Encoder(reg)
    nets, _ = load_ensemble(list(args.value), encoder)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    leaf = BatchedValue([n.to(device) for n in nets], encoder, device=device)

    positions = [Position.from_json(d["position"]) for d, _o, _t, _c, _u in picked]
    values = np.asarray(leaf(positions), dtype=np.float64)

    by_turn: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_pair: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_unseen: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    # (searchValue, foeSearchValue, outcome) where the decision carries both.
    seats: list[tuple[float, float, float]] = []
    for (decision, outcome, turn, company, unseen), predicted in zip(
        picked, values, strict=True
    ):
        search = decision.get("searchValue")
        row = (float(predicted), float(search) if search is not None else float("nan"), outcome)
        by_turn[bucket_of(turn, TURN_BUCKETS)].append(row)
        if turn == 1:
            by_pair[bucket_of(company, PAIR_BUCKETS)].append(row)
        if unseen is not None:
            by_unseen[f"{unseen} unseen"].append(row)
        foe_value = decision.get("foeSearchValue")
        if foe_value is not None and search is not None:
            seats.append((float(search), float(foe_value), outcome))

    # A sign error here would read as a broken model, so it is checked rather than
    # assumed: the search's own estimate must correlate positively with the outcome.
    every = [r for rows in by_turn.values() for r in rows]
    srch = np.asarray([s for _a, s, _o in every], dtype=np.float64)
    out = np.asarray([o for _a, _s, o in every], dtype=np.float64)
    ok = ~np.isnan(srch)
    if ok.sum() > 2:
        corr = float(np.corrcoef(srch[ok], out[ok])[0, 1])
        print(f"  searchValue against outcome: r = {corr:+.3f}")
        if corr < 0:
            raise SystemExit(
                "the recorded search value is anti-correlated with the outcome, so one "
                "of them is from the other seat's perspective; fix that before reading "
                "anything below as a calibration"
            )

    if args.ceiling:
        ceiling(args, paths, leaf)

    report("by turn", by_turn, [name for name, _lo, _hi in TURN_BUCKETS])
    report(
        "turn 1 only, by how many other games shared that selection pair",
        by_pair,
        [name for name, _lo, _hi in PAIR_BUCKETS],
    )
    if by_unseen:
        # The leaf reads the true position and `searchValue` is side 0's belief about
        # it, so leaf-srch mixes "what searching adds" with "what side 0 did not know".
        # At "0 unseen" only the first is left (IKA-127: from `shownIdentities`).
        report(
            "by how many of side 1's four side 0 had not seen",
            by_unseen,
            sorted(by_unseen),
        )
    else:
        print("\n  (no `shownIdentities` in these records -- written before IKA-127)")
    if seats:
        own = np.asarray([s for s, _f, _o in seats], dtype=np.float64)
        foe = np.asarray([f for _s, f, _o in seats], dtype=np.float64)
        out = np.asarray([o for _s, _f, o in seats], dtype=np.float64)
        print(
            f"\n  hidden bench, both seats' own values (side 0's units), n={len(seats)}:\n"
            f"    side 0 {own.mean():.1%}  side 1 {foe.mean():.1%}  actual {out.mean():.1%}"
            f"   side 0 - side 1 {(own - foe).mean():+.1%} (mean |gap| "
            f"{np.abs(own - foe).mean():.1%})\n"
            f"    brier: side 0 {np.mean((own - out) ** 2):.3f}  side 1 "
            f"{np.mean((foe - out) ** 2):.3f}  their mean "
            f"{np.mean(((own + foe) / 2 - out) ** 2):.3f}"
        )
    print(
        "\n  leaf-srch is what a cell of the selection matrix gives up by not searching.\n"
        "  If it is flat across the pair buckets the leaf's trouble is not coverage and\n"
        "  more games on thin pairs will not fix it; if it grows as the games run out,\n"
        "  it will."
    )


if __name__ == "__main__":
    main()
