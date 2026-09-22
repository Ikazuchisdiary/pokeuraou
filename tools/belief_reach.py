"""Can the spread belief represent a build a real person actually brought?

`belief.py` keeps every spread the usage statistics observed, as a weighted particle, and
refuses to invent any others -- which is right, and which means the belief can only ever
converge on a spread usage saw. If a tournament player's allocation is not in that set, no
amount of evidence during the battle moves the posterior onto it. The analysis would then
be conditioning on a hypothesis space that excludes the truth, and every number downstream
of it inherits that silently.

Nothing in `data/standings/` can answer this: `standings._spread_for` fills each opponent's
spread by sampling the same usage prior, so those teams agree with the belief by
construction. The two files on disk that hold builds a person chose are
`configs/teams/*.json` and `configs/archetypes/*.json`, and this checks them.

Three questions per set, in increasing order of what actually matters:

    species    is the species in the usage data at all? `build_belief` raises without it
    nature     is that nature seen for that species? if not the belief falls back to
               re-deriving other natures' spreads, which it labels as weaker
    stats      is there a particle with the SAME level-50 stat line? this is the one that
               counts, because the game reads stats and two SP allocations that land on one
               stat line are the same Pokemon

The exact-spread row is reported beside the stat row so the gap between them is visible:
a build whose spread is absent but whose stat line is present is perfectly representable.

    uv run python tools/belief_reach.py
    uv run python tools/belief_reach.py configs/teams/rizabanadohido.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.belief import BeliefError, build_belief  # noqa: E402
from pokeuraou.names import localiser  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def sets_from(path: Path) -> list[dict]:
    """The (species, nature, sp) entries in a team file or an archetype file.

    An archetype entry carries BOTH a `species` list naming its six and a `sets` list of
    the builds that were actually reported for it, and only the second has spreads. The
    first draft of this matched on `species` and read the composition as a build, which
    produced sixteen rows saying a list of six names is not in the usage data.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for key in ("team", "sets", "archetypes", "entries"):
        got = data.get(key)
        if not (isinstance(got, list) and got and isinstance(got[0], dict)):
            continue
        for entry in got:
            inner = entry.get("sets")
            if isinstance(inner, list):
                label = entry.get("id") or entry.get("name") or path.stem
                out.extend(
                    {**s, "_group": label}
                    for s in inner
                    if isinstance(s, dict) and isinstance(s.get("species"), str)
                )
            elif isinstance(entry.get("species"), str):
                out.append(entry)
        if out:
            return out
    return []


def to_vector(sp: dict) -> np.ndarray:
    return np.array([int(sp.get(k, 0)) for k in STATS], dtype=np.int64)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [
        Path(p) for p in sorted(glob.glob("configs/teams/*.json"))
        + sorted(glob.glob("configs/archetypes/*.json"))
    ]
    if not paths:
        raise SystemExit("no team or archetype files found under configs/")

    first = json.loads(paths[0].read_text(encoding="utf-8"))
    reg = load_regulation(first.get("regulation") or "gen9championsvgc2026regmb")
    cached = find_cached_chaos(reg.meta.format_id)
    if cached is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(cached, reg)
    loc = localiser(reg, "ja")

    totals = {"sets": 0, "species": 0, "nature": 0, "spread": 0, "stats": 0}
    misses: list[tuple[str, int, int]] = []
    print(f"  {'file / set':<42} {'species':>7} {'nature':>6} {'spread':>6} {'stats':>6}"
          f" {'weight':>9}{'particles':>9}")
    for path in paths:
        entries = sets_from(path)
        if not entries:
            print(f"  {path.name}: no sets with a species field")
            continue
        for entry in entries:
            species_id = str(entry.get("species", "")).lower().replace(" ", "").replace("-", "")
            nature = str(entry.get("nature", "")) or "Serious"
            sp = entry.get("sp") or entry.get("evs") or None
            if sp is None:
                print(f"  {path.stem}/{entry.get('species')}: no spread stated, skipped")
                continue
            name = loc.species(species_id) if loc else species_id
            group = entry.get("_group")
            label = f"{group}/{name}" if group else f"{path.stem}/{name}"
            try:
                belief = build_belief(reg, prior, species_id, nature)
            except BeliefError:
                print(f"  {label:<42} {'NO':>7} {'-':>6} {'-':>6} {'-':>6} {'-':>9}")
                continue
            totals["sets"] += 1
            totals["species"] += 1
            nature_ok = "unseen" not in belief.provenance
            totals["nature"] += int(nature_ok)

            want = to_vector(sp)
            exact = np.all(np.asarray(belief.spreads) == want, axis=1)
            spread_w = float(np.asarray(belief.weights)[exact].sum())
            totals["spread"] += int(exact.any())

            # The stat line is what the game reads. Recompute ours the same way the belief
            # did, by finding a particle whose stats match -- if the spread itself is
            # present its stats trivially are, but the converse is the interesting case.
            stats = np.asarray(belief.stats)
            mine = stats[exact][0] if exact.any() else None
            if mine is None:
                from pokeuraou.stats import nature_multipliers, stats_from_sp

                base = np.array(reg.species[species_id].base_stats, dtype=np.int64)
                mine = stats_from_sp(
                    reg, base, want[None, :], nature_multipliers(reg, [nature]), level=50
                )[0]
            same_stats = np.all(stats == mine, axis=1)
            stats_w = float(np.asarray(belief.weights)[same_stats].sum())
            totals["stats"] += int(same_stats.any())
            # How far the belief has to be wrong when it cannot be right. The largest
            # single-stat error of the closest particle, which is what a damage roll or a
            # speed tie would actually feel.
            gaps = np.abs(stats - mine).max(axis=1)
            nearest = int(gaps.min())
            misses.append((label, nearest, int(np.asarray(belief.weights).size)))

            print(f"  {label:<42} {'yes':>7} {('yes' if nature_ok else 'fallback'):>6} "
                  f"{(f'{spread_w:.1e}' if exact.any() else 'NO'):>6} "
                  f"{('yes' if same_stats.any() else f'+/-{nearest}'):>6} "
                  f"{(f'{stats_w:.2e}' if same_stats.any() else '0'):>9}"
                  f"{len(belief.weights):>8}")

    n = max(totals["sets"], 1)
    print(f"\n  {totals['sets']} sets from {len(paths)} files")
    for key, label in (("species", "species in the usage data"),
                       ("nature", "nature seen for that species"),
                       ("spread", "exact SP allocation is a particle"),
                       ("stats", "SOME particle has the same stat line")):
        print(f"    {label:<34} {totals[key]:>3}/{n}  {totals[key] / n:.0%}")
    unreachable = [m for m in misses if m[1] > 0]
    if unreachable:
        print("\n  builds the belief cannot represent, and how wrong its closest particle is:")
        for label, nearest, size in sorted(unreachable, key=lambda m: -m[1]):
            print(f"    {label:<40} nearest particle off by {nearest} in some stat, "
                  f"of {size:,} particles")
    print("\n  The last row is the one that matters: a build whose stat line no particle\n"
          "  reaches cannot be found by any amount of evidence, because the hypothesis\n"
          "  space does not contain it. The weight column says how much prior mass sits on\n"
          "  the truth -- a particle carrying 1e-6 is reachable in principle and not in\n"
          "  practice.")


if __name__ == "__main__":
    main()
