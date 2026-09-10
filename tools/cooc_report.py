"""Does the pairwise team model actually reproduce the metagame it was built from?

The opponent pool was eleven hand-curated archetypes, and it was not the metagame: two of
the format's seven most-used Pokemon appeared in none of them. The replacement draws teams
from the chaos file's `Teammates` table, which is 1.27M battles worth of measured "who is
brought with whom".

But a pairwise model is an approximation -- the data records pairs, not teams -- so it
cannot be adopted on the strength of being data-driven. This tool measures the error:
sample teams, then compare the sampled species frequencies and pair frequencies against
the measured ones. If the sampler over-picks Kingambit, that shows up here as a number
before it shows up as a value function trained against the wrong metagame.

    uv run python tools/cooc_report.py --teams 20000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.priors import (
    build_cooccurrence,
    find_cached_chaos,
    load_chaos,
    sample_species_by_cooccurrence,
)
from pokeuraou.regulation import load_regulation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regulation", default="gen9championsvgc2026regmb")
    ap.add_argument("--teams", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()

    reg = load_regulation(args.regulation)
    cached = find_cached_chaos(reg.meta.format_id)
    if cached is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(cached, reg)
    print(prior.summary())
    print(
        f"\nsum(Teammates)/5/usage = {prior.weighted_teams:,.1f} weighted teams, "
        f"varying by {prior.weighted_teams_spread:.2e} across species"
    )
    print("  (a constant is the evidence the table holds co-occurrence counts)")

    cooc = build_cooccurrence(prior)
    print("\n" + cooc.summary())

    # Ranking by lift alone surfaces only noise: two species at 0.0% usage that shared one
    # team score a lift in the thousands. Rank by how often the pair is actually brought,
    # which is what the sampler is driven by.
    print("\nmost-brought pairs, and how far above chance they are")
    n = len(cooc.species)
    iu = np.triu_indices(n, k=1)
    freq = cooc.lift[iu] * cooc.usage[iu[0]] * cooc.usage[iu[1]]
    for k in np.argsort(-freq)[:14]:
        i, j = iu[0][k], iu[1][k]
        print(
            f"  {cooc.species[i]:>18} + {cooc.species[j]:<18} "
            f"together {freq[k] * 100:5.2f}% of teams   lift {cooc.lift[i, j]:5.2f}x"
        )

    rng = np.random.default_rng(args.seed)
    counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for _ in range(args.teams):
        team = sample_species_by_cooccurrence(rng, reg, cooc)
        counts.update(team)
        for a in range(len(team)):
            for b in range(a + 1, len(team)):
                pair_counts[tuple(sorted((team[a], team[b])))] += 1

    print(f"\nsampled {args.teams:,} teams; how close the marginals came back")
    sampled = np.array([counts[sid] / args.teams for sid in cooc.species])
    measured = cooc.usage
    # Total variation over the species marginals, which is the readable summary: half the
    # sum of absolute differences, i.e. the share of team slots placed on the wrong species.
    tv = 0.5 * np.abs(sampled / sampled.sum() - measured / measured.sum()).sum()
    print(f"  total variation between sampled and measured usage: {tv * 100:.1f}%")
    print(f"  {'species':>18}  {'measured':>9}  {'sampled':>9}  ratio")
    for i in np.argsort(-measured)[: args.show]:
        ratio = sampled[i] / measured[i] if measured[i] else float("nan")
        print(
            f"  {cooc.species[i]:>18}  {measured[i] * 100:8.2f}%  "
            f"{sampled[i] * 100:8.2f}%  {ratio:5.2f}x"
        )

    print("\nthe pairs the sampler gets most wrong (by absolute frequency error)")
    errors = []
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        a, b = cooc.species[i], cooc.species[j]
        want = cooc.lift[i, j] * measured[i] * measured[j]
        got = pair_counts[tuple(sorted((a, b)))] / args.teams
        errors.append((abs(got - want), a, b, want, got))
    errors.sort(reverse=True)
    for _, a, b, want, got in errors[:12]:
        print(f"  {a:>18} + {b:<18} measured {want * 100:5.2f}%  sampled {got * 100:5.2f}%")


if __name__ == "__main__":
    main()
