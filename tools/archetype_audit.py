"""Does the archetype list explain the metagame the usage data measured?

The opponent pool is built from tournament standings, because the field is a mixture of a
few archetypes and a pairwise co-occurrence model cannot represent a mixture -- it knows
which Pokemon go together, not which sixes were brought, so it will cheerfully assemble a
team nobody played. Curation is therefore the right source.

But curation has the opposite failure: a list can simply be missing an archetype, and
nothing about the list itself says so. That is what this tool measures, by inverting the
relationship. If the field really is a mixture of these archetypes with weights `w`, then

    usage[species] = sum over archetypes containing it of w[archetype]

which is a non-negative least squares problem with one equation per species. Two things
come out of solving it:

- **fitted weights**, an estimate of each archetype's share from 1.27M battles, which can
  be held up against the share observed in the standings. Two independent measurements of
  the same quantity agreeing is worth more than either alone.
- **residuals**, the usage each species has that no archetype accounts for. A large
  positive residual names a missing archetype: something is carrying that Pokemon and it
  is not in the list.

The pairwise version is also reported, since matching marginals is easy and matching pairs
is not: a list could account for all of Kingambit's usage while pairing it with the wrong
partners.

    uv run python tools/archetype_audit.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.priors import build_cooccurrence, find_cached_chaos, load_chaos
from pokeuraou.teams import load_archetypes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archetypes", default="wcs2026-regmb")
    ap.add_argument("--show", type=int, default=16)
    ap.add_argument("--observed-only", action="store_true",
                    help="fit only the archetypes with a tournament observation")
    args = ap.parse_args()

    reg, archetypes = load_archetypes(args.archetypes)
    cached = find_cached_chaos(reg.meta.format_id)
    if cached is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(cached, reg)
    cooc = build_cooccurrence(prior)

    if args.observed_only:
        archetypes = [a for a in archetypes if a.observed]
    names = [a.id for a in archetypes]
    index = cooc.index
    usage = cooc.usage
    n_species = len(cooc.species)

    # One column per archetype, one row per species: 1 where the archetype brings it.
    design = np.zeros((n_species, len(archetypes)), dtype=np.float64)
    off_list: list[str] = []
    for column, archetype in enumerate(archetypes):
        for species_id in archetype.species:
            row = index.get(species_id)
            if row is None:
                off_list.append(f"{archetype.id}/{species_id}")
                continue
            design[row, column] = 1.0

    weights, _residual = nnls(design, usage)
    fitted = design @ weights
    residual = usage - fitted

    print(f"{len(archetypes)} archetypes over {n_species} species with usage data")
    if off_list:
        print(f"  members with no usage data at all: {', '.join(off_list)}")
    explained = float(fitted.sum() / usage.sum())
    print(f"  share of total usage the list accounts for: {explained * 100:.1f}%")
    print(f"  weights sum to {weights.sum():.3f} (1.0 would mean the field is only these)")

    print("\nfitted share from usage, against the share observed in the standings")
    print(f"  {'archetype':>32}  {'fitted':>7}  {'standings':>10}")
    order = np.argsort(-weights)
    for i in order:
        archetype = archetypes[i]
        seen = (
            f"{archetype.observed_count}/{archetype.observed_of}"
            if archetype.observed
            else "--"
        )
        share = weights[i] / max(weights.sum(), 1e-12)
        print(f"  {names[i]:>32}  {share * 100:6.1f}%  {seen:>10}")

    print("\nusage no archetype accounts for (a missing archetype shows up here)")
    print(f"  {'species':>20}  {'usage':>7}  {'explained':>9}  {'left over':>9}")
    for i in np.argsort(-residual)[: args.show]:
        if residual[i] <= 0.005:
            break
        print(
            f"  {cooc.species[i]:>20}  {usage[i] * 100:6.1f}%  "
            f"{fitted[i] * 100:8.1f}%  {residual[i] * 100:8.1f}%"
        )

    over = np.argsort(residual)[:6]
    if residual[over[0]] < -0.005:
        print("\n  ...and the ones the list over-produces relative to usage")
        for i in over:
            if residual[i] >= -0.005:
                break
            print(
                f"  {cooc.species[i]:>20}  {usage[i] * 100:6.1f}%  "
                f"{fitted[i] * 100:8.1f}%  {residual[i] * 100:8.1f}%"
            )

    # Matching the marginals is the easy half. A list can account for every point of
    # Kingambit's usage while pairing it with the wrong Pokemon, so fit the pairs too.
    pairs = np.triu_indices(n_species, k=1)
    target = (cooc.lift * np.outer(usage, usage))[pairs]
    pair_design = np.zeros((len(target), len(archetypes)), dtype=np.float64)
    for column, archetype in enumerate(archetypes):
        rows = [index[s] for s in archetype.species if s in index]
        present = np.zeros(n_species, dtype=np.float64)
        present[rows] = 1.0
        pair_design[:, column] = np.outer(present, present)[pairs]
    pair_weights, _ = nnls(pair_design, target)
    pair_fitted = pair_design @ pair_weights
    print("\nthe same fit on pair frequencies rather than marginals")
    print(f"  share of pair mass accounted for: {pair_fitted.sum() / target.sum() * 100:.1f}%")
    worst = np.argsort(-(target - pair_fitted))[:10]
    print(f"  {'pair':>40}  {'measured':>9}  {'explained':>9}")
    for k in worst:
        i, j = pairs[0][k], pairs[1][k]
        if target[k] - pair_fitted[k] <= 0.005:
            break
        label = f"{cooc.species[i]} + {cooc.species[j]}"
        print(f"  {label:>40}  {target[k] * 100:8.2f}%  {pair_fitted[k] * 100:8.2f}%")


if __name__ == "__main__":
    main()
