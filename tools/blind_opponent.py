"""What does it cost us that the selection solve lets the opponent see our spreads?

`solve_bayesian`'s docstring says what game it solves: "Equilibrium when the column
player observes their own type and we do not." The matrices are built from `our_six`
exactly, so the column player's strategy is a best response to our real investment --
while ours is one vector over a distribution of theirs. They know both sides; we know
one. In a tournament they know neither: an open team sheet shows species, ability, item,
nature and moves, and not the spread.

The assumption is conservative, so its direction is known -- it makes the opponent
stronger than they can be, which understates our value and makes our selection more
defensive than it needs to be. What is not known is the size, and the size is what
decides whether it is worth fixing. Measured here:

    V_seen    our equilibrium value when they best-respond to our TRUE six  (today)
    V_blind   our value when they play the strategy they would compute against a
              SAMPLED version of our six -- same sheet, spreads from the usage prior --
              and we best-respond to that, with the payoff still read off the true six

V_blind >= V_seen, because a strategy optimised for the wrong team cannot do better
against ours than one optimised for it. The gap is the price of the assumption.

Not a fix. A fix means solving a two-sided incomplete-information game, where our spread
is a type they cannot see, and that multiplies the solve by the number of our classes --
40 hours over the field at eight each. This says whether that is worth buying.

    uv run python tools/blind_opponent.py --model data/models/value-gen11L.pt --teams 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.selection import SpreadClass, solve_selection  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import load_roster  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--model", type=Path, nargs="+", required=True)
    ap.add_argument("--teams", type=int, default=8, help="opponents to solve")
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    from pokeuraou.value import BatchedValue, load_ensemble

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    encoder = Encoder(reg)
    nets, _ = load_ensemble(list(args.model), encoder)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    leaf = BatchedValue([n.to(device) for n in nets], encoder, device=device)

    standings = load_standings(find_cached_standings(), reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    rng = np.random.default_rng(args.seed)
    pool = standings.pool("all")
    chosen = [pool[int(i)] for i in rng.permutation(len(pool))[: args.teams]]

    # Our six as an open sheet shows it: everything but the investment, which is redrawn
    # from the usage prior. This is what an opponent can actually condition on.
    blind_six = [
        entry
        for entry in sample_standings_team(
            np.random.default_rng([args.seed, 0]),
            reg,
            prior,
            _as_tournament_team(roster, reg),
        )
    ]
    print("  our six, as they can see it (spreads redrawn from the prior):")
    for real, seen in zip(roster.sets, blind_six, strict=True):
        same = "same" if real.sp == seen.sp else "differs"
        print(
            f"    {real.species:16s} true {dict(real.sp)}  seen {dict(seen.sp)}  {same}"
        )

    print(f"\n  {'place':>5}  {'V_seen':>7}  {'V_blind':>7}  {'gap':>7}   seconds")
    gaps = []
    for team in chosen:
        started = time.perf_counter()
        classes = [
            SpreadClass(
                weight=1.0 / args.classes,
                sets=tuple(sample_standings_team(rng, reg, prior, team)),
                label=f"class {k + 1}",
            )
            for k in range(args.classes)
        ]
        seen = solve_selection(reg, roster.sets, classes, leaf)
        blind = solve_selection(reg, blind_six, classes, leaf)

        # Their strategy from the blind solve, our best response on the TRUE matrices.
        true_mats = [np.asarray(m, dtype=np.float64) for m in seen.matrices]
        weights = np.asarray([c.weight for c in classes], dtype=np.float64)
        weights = weights / weights.sum()
        blind_cols = [
            np.asarray(y, dtype=np.float64) for y in blind.equilibrium.col_strategies
        ]
        # Our payoff against a column player who plays `blind_cols[k]` in class k, if we
        # play the single row that maximises it.
        against = sum(
            w * (mat @ y) for w, mat, y in zip(weights, true_mats, blind_cols, strict=True)
        )
        v_blind = float(np.max(against))
        v_seen = float(seen.equilibrium.value)
        gaps.append(v_blind - v_seen)
        print(
            f"  {team.place:>5}  {v_seen:7.4f}  {v_blind:7.4f}  {v_blind - v_seen:+7.4f}"
            f"   {time.perf_counter() - started:6.1f}"
        )

    arr = np.asarray(gaps)
    print(
        f"\n  {len(arr)} opponents: the assumption costs us "
        f"{arr.mean():+.4f} on average [{arr.min():+.4f}, {arr.max():+.4f}]"
    )
    print(
        "  Positive means our value is higher against an opponent who cannot see our\n"
        "  spreads, which is the opponent a tournament actually provides."
    )


def _as_tournament_team(roster, reg):  # noqa: ANN001, ANN201
    """Our roster in the shape `sample_standings_team` reads, so the redraw uses the
    same code path the 394 opponents go through rather than a second one."""
    from pokeuraou.standings import TeamMember, TournamentTeam

    return TournamentTeam(
        player="(ours)",
        place=0,
        country="--",
        wins=0,
        losses=0,
        made_cut=False,
        phase_two=False,
        members=tuple(
            TeamMember(
                species=s.species,
                ability=s.ability,
                item=s.item,
                nature=s.nature,
                moves=tuple(s.moves),
                ability_is_post_mega=False,
            )
            for s in roster.sets
        ),
    )


if __name__ == "__main__":
    main()
