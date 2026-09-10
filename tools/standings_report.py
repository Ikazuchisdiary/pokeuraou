"""What is actually in the tournament pool, and how it compares to ladder usage.

Two comparisons worth making before training against it:

- **the field against the ladder.** Smogon usage is the online ladder; this is 395 people
  at Worlds. Where they disagree, the tournament is the better answer for a tool aimed at
  tournament preparation -- and the size of the disagreement says how much the earlier
  ladder-based pool was off by.
- **the archetype clusters.** Grouping teams by species overlap shows how concentrated the
  field was, which is the claim that motivated using real teams at all: if the top cluster
  is 10% of the field, a curated list of eleven archetypes was never going to cover it.

    uv run python tools/standings_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.names import localiser
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.regulation import load_regulation
from pokeuraou.standings import (
    cluster_core,
    cluster_teams,
    find_cached_standings,
    load_standings,
    species_frequency,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regulation", default="gen9championsvgc2026regmb")
    ap.add_argument("--season", default="2026")
    ap.add_argument("--event", default="worlds")
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--min-shared", type=int, default=5)
    ap.add_argument("--lang", default="ja", help="ja, or 'en' for no localisation")
    args = ap.parse_args()

    reg = load_regulation(args.regulation)
    cached = find_cached_standings(args.season, args.event)
    if cached is None:
        raise SystemExit(
            f"no cached standings for {args.season}/{args.event}; run "
            f"tools/fetch_standings.py {args.season} {args.event}"
        )
    standings = load_standings(cached, reg)
    print(standings.summary())

    loc = localiser(reg, args.lang)

    def name(species_id: str) -> str:
        return loc.species(species_id) if loc else reg.species[species_id].name

    teams = standings.pool("all")
    frequency = species_frequency(teams)

    chaos = find_cached_chaos(reg.meta.format_id)
    ladder = {}
    if chaos is not None:
        prior = load_chaos(chaos, reg)
        ladder = {s.species_id: s.usage for s in prior.species.values()}

    print(f"\nspecies frequency in the field, against ladder usage ({len(teams)} teams)")
    print(f"  {'species':>22}  {'worlds':>7}  {'ladder':>7}  {'diff':>7}")
    for species_id, share in list(frequency.items())[: args.show]:
        online = ladder.get(species_id)
        if online is None:
            print(f"  {name(species_id):>22}  {share * 100:6.1f}%  {'--':>7}  {'--':>7}")
        else:
            print(
                f"  {name(species_id):>22}  {share * 100:6.1f}%  {online * 100:6.1f}%  "
                f"{(share - online) * 100:+6.1f}"
            )

    if ladder:
        # Anything the ladder rates highly and nobody brought is the clearest sign the two
        # distributions are not interchangeable.
        missing = sorted(
            ((sid, u) for sid, u in ladder.items() if frequency.get(sid, 0.0) == 0.0),
            key=lambda kv: -kv[1],
        )[:8]
        if missing:
            print("\n  used on the ladder, brought by nobody at Worlds:")
            for species_id, online in missing:
                print(f"  {name(species_id):>22}  ladder {online * 100:5.1f}%")

    groups = cluster_teams(teams, min_shared=args.min_shared)
    print(
        f"\narchetype clusters (single-link on >= {args.min_shared} shared species): "
        f"{len(groups)} groups. The core is the most common members with their share of "
        "the group, since a chained cluster has no useful intersection."
    )
    print(f"  {'n':>4}  {'share':>6}  {'best':>5}  core")
    for group in groups[:14]:
        if len(group) < 2:
            break
        core = ", ".join(
            f"{name(sid)} {share * 100:.0f}%" for sid, share in cluster_core(group, top=4)
        )
        print(
            f"  {len(group):>4}  {len(group) / len(teams) * 100:5.1f}%  "
            f"{min(t.place for t in group):>5}  {core}"
        )
    singles = sum(1 for g in groups if len(g) == 1)
    print(f"  ...and {singles} teams that cluster with nobody at this threshold")


if __name__ == "__main__":
    main()
