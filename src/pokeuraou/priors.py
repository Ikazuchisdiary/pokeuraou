"""Metagame priors from Smogon's usage statistics.

Source: ``https://www.smogon.com/stats/<YYYY-MM>/chaos/<format>-<cutoff>.json.gz``.
The ``Spreads`` table is already in SP space -- ``"Adamant:32/32/0/0/1/1": weight`` -- so
it needs no conversion, only re-keying.

Why the whole distribution and not a handful of archetypes: the observed spreads do not
collapse. Incineroar alone has 10,154 distinct spreads at the 1760 cutoff, and taking the
top 8 exact stat lines covers under 30% of the probability mass. A fixed archetype set
would therefore leave most of the belief unrepresented, and posterior filtering would
reject every archetype in ordinary positions. So the belief keeps every observed spread as
a weighted particle, and reduction to a small set happens per position and losslessly
(see ``belief.py``).

Licensing note: the stats directory carries no explicit licence. It is a public directory
that third-party tools consume routinely. This module reads a locally cached copy; the
download is a separate, manual step (``tools/fetch_priors.py``) that runs at most once a
month, and raw files are not redistributed.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .regulation import STAT_IDS, Regulation, repo_root, to_id


def priors_dir() -> Path:
    return repo_root() / "data" / "priors"


@dataclass(slots=True)
class SpeciesPrior:
    """Everything the usage data says about one species."""

    species_id: str
    name: str
    #: Share of team slots this species appears on (0..1).
    usage: float
    raw_count: int
    #: (K,) nature names, (K, 6) SP spreads, (K,) weights summing to 1.
    natures: np.ndarray
    spreads: np.ndarray
    weights: np.ndarray
    abilities: dict[str, float]
    items: dict[str, float]
    moves: dict[str, float]
    teammates: dict[str, float]

    @property
    def n_particles(self) -> int:
        return int(self.weights.shape[0])

    def top_spreads(self, k: int) -> list[tuple[str, tuple[int, ...], float]]:
        order = np.argsort(-self.weights)[:k]
        return [
            (str(self.natures[i]), tuple(int(v) for v in self.spreads[i]), float(self.weights[i]))
            for i in order
        ]


@dataclass(slots=True)
class MetagamePrior:
    format_id: str
    cutoff: int
    battles: int
    species: dict[str, SpeciesPrior]
    #: Species that appear in the usage data but not in the target regulation.
    dropped_species: tuple[str, ...]
    #: Items/moves that appear in the data but not in the target regulation's pool.
    dropped_items: tuple[str, ...]
    dropped_moves: tuple[str, ...]
    source: str
    #: Mega forme in the data -> the species it was folded into. Usage data names the
    #: mega, but a team brings the base, so these are the same Pokemon.
    folded_megas: dict[str, str] = field(default_factory=dict)
    #: Species whose only usage came from a mega forme, so no *legal* ability was ever
    #: observed. Their ability distribution is uniform over the species' legal abilities,
    #: which is an assumption and is therefore named here.
    abilities_unobserved: tuple[str, ...] = ()
    #: Total weighted teams behind the tables, measured as ``sum(Teammates)/5/usage``,
    #: which is identical for every species when the values are co-occurrence counts.
    #: nan when the file does not satisfy that, which is the signal not to read the
    #: teammate tables as counts.
    weighted_teams: float = float("nan")
    #: How much that ratio varied across species, as a fraction of its median.
    weighted_teams_spread: float = float("nan")

    def by_usage(self) -> list[SpeciesPrior]:
        return sorted(self.species.values(), key=lambda s: -s.usage)

    def summary(self) -> str:
        top = self.by_usage()[:8]
        head = ", ".join(f"{s.name} {s.usage * 100:.1f}%" for s in top)
        lines = [
            f"{self.format_id} cutoff {self.cutoff}: {len(self.species)} species from "
            f"{self.battles:,} battles",
            f"  top: {head}",
            f"  dropped as not legal here: {len(self.dropped_species)} species, "
            f"{len(self.dropped_items)} items, {len(self.dropped_moves)} moves",
        ]
        if self.folded_megas:
            folded = ", ".join(
                f"{mega}->{base}" for mega, base in sorted(self.folded_megas.items())
            )
            lines.append(f"  megas folded into their base species: {folded}")
        if self.abilities_unobserved:
            lines.append(
                "  ability never observed on the base forme (uniform over legal): "
                + ", ".join(self.abilities_unobserved)
            )
        return "\n".join(lines)


def _normalise(table: dict[str, float], key: Any = to_id) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_key, weight in table.items():
        if weight <= 0:
            continue
        k = key(raw_key) if callable(key) else raw_key
        out[k] = out.get(k, 0.0) + float(weight)
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def load_chaos(
    path: str | Path, reg: Regulation, *, min_species_usage: float = 0.0
) -> MetagamePrior:
    """Parses a Smogon ``chaos`` JSON into per-species priors, filtered to a regulation.

    Species, items and moves that do not exist in the target regulation are dropped and
    reported rather than silently kept, because the prior is used across a regulation
    change (M-C launched on 2026-09-09, so the first M-C usage data is not published until
    early October and M-B data is the only starting point).
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        data = json.load(fh)

    info = data["info"]
    # sum(Teammates) / 5 / usage, per species, before anything is folded or normalised.
    # Identical values across species is the evidence that the table holds counts of
    # teams rather than conditional shares, and the shared value is the team count.
    team_counts = [
        sum(entry["Teammates"].values()) / 5.0 / entry["usage"]
        for entry in data["data"].values()
        if entry.get("Teammates") and entry.get("usage")
    ]
    species: dict[str, SpeciesPrior] = {}
    dropped_species: list[str] = []
    dropped_items: set[str] = set()
    dropped_moves: set[str] = set()
    folded_megas: dict[str, str] = {}

    # Fold each mega forme's entry into the species a team actually brings. Merging the
    # raw entries before parsing means the spread, item and move tables are combined
    # weighted by how often each forme was seen, which is what "Raw count" is for.
    merged: dict[str, dict[str, Any]] = {}
    #: every species id in the data -> the team-legal id it was folded into, so the
    #: teammate tables can be re-keyed the same way their owners were. Without this a
    #: teammate named `Charizard-Mega-Y` matches no team-legal species and the whole
    #: co-occurrence row loses its megas -- the same defect that dropped the mega formes
    #: from usage, on the other axis of the same table.
    fold_to: dict[str, str] = {}
    for name, entry in data["data"].items():
        species_id = to_id(name)
        target = reg.species.get(species_id)
        if target is not None and not target.team_legal and target.changes_from:
            base = to_id(target.changes_from)
            if reg.species.get(base) is not None:
                folded_megas[name] = reg.species[base].name
                # The post-mega ability is not legal on the base and is not what the team
                # sheet shows, so it is the one table that does not come along.
                entry = {k: v for k, v in entry.items() if k != "Abilities"}
                species_id = base
            else:
                dropped_species.append(name)
                continue
        elif target is None or not target.team_legal:
            dropped_species.append(name)
            continue
        fold_to[to_id(name)] = species_id
        existing = merged.get(species_id)
        merged[species_id] = entry if existing is None else _merge_entries(existing, entry)

    unobserved_abilities: list[str] = []
    for species_id, entry in merged.items():
        target = reg.species[species_id]
        name = target.name
        usage = float(entry.get("usage", 0.0))
        if usage < min_species_usage:
            continue

        natures: list[str] = []
        spreads: list[list[int]] = []
        weights: list[float] = []
        for key, weight in entry.get("Spreads", {}).items():
            if weight <= 0 or ":" not in key:
                continue
            nature, sp_text = key.split(":", 1)
            if nature not in reg.natures:
                continue
            parts = sp_text.split("/")
            if len(parts) != len(STAT_IDS):
                continue
            try:
                sp = [int(x) for x in parts]
            except ValueError:
                continue
            natures.append(nature)
            spreads.append(sp)
            weights.append(float(weight))

        if not weights:
            dropped_species.append(name)
            continue

        w = np.asarray(weights, dtype=np.float64)
        w /= w.sum()

        legal_ability_ids = {to_id(a) for a in target.abilities}
        observed_abilities = {
            k: v
            for k, v in _normalise(entry.get("Abilities", {})).items()
            if k in legal_ability_ids
        }
        if not observed_abilities:
            # Every sighting was of the mega forme, so the pre-mega ability was never in
            # the data. Uniform over what the species can legally have is an assumption,
            # and it is reported rather than passed off as measurement.
            unobserved_abilities.append(target.name)
            observed_abilities = {a: 1.0 / len(legal_ability_ids) for a in legal_ability_ids}

        items = _normalise(entry.get("Items", {}))
        legal_items = {k: v for k, v in items.items() if k in reg.items or k == "nothing"}
        dropped_items |= set(items) - set(legal_items)

        moves = _normalise(entry.get("Moves", {}))
        legal_moves = {k: v for k, v in moves.items() if k in reg.moves}
        dropped_moves |= set(moves) - set(legal_moves)

        species[species_id] = SpeciesPrior(
            species_id=species_id,
            name=target.name,
            usage=usage,
            raw_count=int(entry.get("Raw count", 0)),
            natures=np.asarray(natures, dtype=object),
            spreads=np.asarray(spreads, dtype=np.int64),
            weights=w,
            abilities=observed_abilities,
            items=_normalise(legal_items) if legal_items else {},
            moves=_normalise(legal_moves) if legal_moves else {},
            teammates=_normalise(
                entry.get("Teammates", {}), key=lambda k: fold_to.get(to_id(k), to_id(k))
            ),
        )

    if team_counts:
        median_teams = float(np.median(team_counts))
        spread = float(
            (np.max(team_counts) - np.min(team_counts)) / max(median_teams, 1e-9)
        )
    else:
        median_teams, spread = float("nan"), float("nan")

    return MetagamePrior(
        format_id=info.get("metagame", "?"),
        cutoff=int(info.get("cutoff", 0)),
        battles=int(info.get("number of battles", 0)),
        species=species,
        dropped_species=tuple(sorted(dropped_species)),
        dropped_items=tuple(sorted(dropped_items)),
        dropped_moves=tuple(sorted(dropped_moves)),
        weighted_teams=median_teams,
        weighted_teams_spread=spread,
        source=str(p),
        folded_megas=folded_megas,
        abilities_unobserved=tuple(sorted(unobserved_abilities)),
    )


def _merge_entries(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Combines two raw chaos entries for what is really one Pokemon.

    The count tables are already in units of sightings, so they add. ``usage`` is a share
    of team slots and adds for the same reason -- a team brings the Pokemon once, whether
    or not it megas.
    """
    out: dict[str, Any] = {
        "usage": float(a.get("usage", 0.0)) + float(b.get("usage", 0.0)),
        "Raw count": int(a.get("Raw count", 0)) + int(b.get("Raw count", 0)),
    }
    for table in ("Spreads", "Items", "Moves", "Abilities", "Teammates"):
        combined: dict[str, float] = dict(a.get(table, {}) or {})
        for key, weight in (b.get(table, {}) or {}).items():
            combined[key] = combined.get(key, 0.0) + float(weight)
        if combined:
            out[table] = combined
    return out


def find_cached_chaos(format_id: str, cutoff: int = 1760) -> Path | None:
    """Newest locally cached chaos file for a format, preferring an exact match.

    Falls back to another Champions VGC regulation's file, since a newly launched
    regulation has no usage data of its own for about a month.
    """
    raw = priors_dir() / "raw"
    if not raw.exists():
        return None
    exact = sorted(raw.glob(f"{format_id}-{cutoff}.json*"))
    if exact:
        return exact[-1]
    fallback = sorted(raw.glob(f"*vgc2026reg*-{cutoff}.json*"))
    return fallback[-1] if fallback else None


# ---------------------------------------------------------------------------
# Sampling metagame-realistic teams
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SampledSet:
    species: str
    ability: str
    item: str | None
    nature: str
    sp: dict[str, int]
    moves: list[str]

    def to_team_set_json(self, reg: Regulation) -> dict[str, Any]:
        species = reg.species[self.species]
        ability = reg.abilities.get(self.ability)
        item = reg.items.get(self.item or "")
        return {
            "species": species.name,
            "ability": ability.name if ability else species.abilities[0],
            "item": item.name if item else "",
            "nature": self.nature,
            "sp": dict(self.sp),
            "moves": [reg.moves[m].name for m in self.moves],
            "level": reg.meta.level,
        }


def weighted_choice(rng: np.random.Generator, table: dict[str, float]) -> str | None:
    if not table:
        return None
    keys = list(table)
    probs = np.array([table[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(np.asarray(keys, dtype=object), p=probs))


def sample_set(
    rng: np.random.Generator, reg: Regulation, prior: SpeciesPrior, *, n_moves: int = 4
) -> SampledSet:
    """One Pokemon drawn from its own usage distribution.

    Abilities, items, natures and spreads come from the marginal distributions, so a
    sampled set is realistic but not necessarily a real set. That is fine for the purpose:
    exercising mechanics that actually occur, at roughly the rate they occur.
    """
    species = reg.species[prior.species_id]

    ability = weighted_choice(rng, prior.abilities)
    legal_abilities = {to_id(a) for a in species.abilities}
    if ability not in legal_abilities:
        ability = to_id(species.abilities[0])

    item = weighted_choice(rng, prior.items)
    if item in (None, "nothing") or item not in reg.items:
        item = None
    # A mega stone only makes sense on its own species.
    if item and reg.items[item].mega_stone and reg.mega_target(species.id, item) is None:
        item = None

    idx = int(rng.choice(prior.n_particles, p=prior.weights))
    nature = str(prior.natures[idx])
    sp = {stat: int(v) for stat, v in zip(STAT_IDS, prior.spreads[idx], strict=True) if v}

    moves = _sample_moves(rng, reg, prior, n_moves)

    return SampledSet(
        species=species.id, ability=ability, item=item, nature=nature, sp=sp, moves=moves
    )


def _is_damaging(reg: Regulation, move_id: str) -> bool:
    move = reg.moves.get(move_id)
    if move is None or move.raw.get("category") == "Status":
        return False
    raw = move.raw
    return bool(raw.get("basePower") or raw.get("damageCallback") or raw.get("damage"))


def _sample_moves(
    rng: np.random.Generator, reg: Regulation, prior: SpeciesPrior, n_moves: int
) -> list[str]:
    """Four moves drawn from the marginal table, constrained to be able to attack.

    The move table is a marginal, so drawing four moves without replacement can produce a
    set no player would bring. 2.3% of sampled Pokemon came out with *no damaging move at
    all* -- Whimsicott with Charm/Encore/Protect/Tailwind was the most common -- and the
    opponent's win rate fell 12 points in the games where it happened, so the search was
    being handed a Pokemon that cannot win and the label recorded it as a loss.

    The constraint is stated rather than invented: a set has at least one move that deals
    damage. It is enforced by replacing the last drawn move, so the rest of the set is
    still the marginal draw and the damaging move is still drawn by usage.
    """
    table = dict(prior.moves)
    moves: list[str] = []
    while table and len(moves) < n_moves:
        pick = weighted_choice(rng, table)
        if pick is None:
            break
        table.pop(pick, None)
        moves.append(pick)

    if moves and not any(_is_damaging(reg, m) for m in moves):
        attacks = {k: v for k, v in prior.moves.items() if _is_damaging(reg, k)}
        replacement = weighted_choice(rng, attacks)
        if replacement is not None:
            moves[-1] = replacement
    if not moves:
        moves = ["tackle"] if "tackle" in reg.moves else [next(iter(reg.moves))]
    return moves


# ---------------------------------------------------------------------------
# The measured pairwise team structure
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Cooccurrence:
    """Which species are brought together, measured rather than curated.

    The ``Teammates`` table in the chaos file is raw weighted co-occurrence. Two checks
    establish that, and :func:`build_cooccurrence` reports both so the reading cannot rot
    silently:

    - ``sum(teammates) / 5 / usage`` is the same number for every species -- 6934.3 at the
      M-B 1760 cutoff -- which is what has to happen if the values count teams containing
      both, since every team has exactly five other members. That number is the total
      weighted team count :attr:`teams`.
    - the matrix is symmetric, which a count table is and a conditional probability is not.

    What is stored is **lift**, ``P(a,b) / (P(a)P(b))``: 1.0 means the pair is brought
    together exactly as often as chance would, above 1.0 means it is a real pairing, below
    means players avoid it. Lift is the right scale because it divides usage out, so the
    sampler multiplies by usage once rather than twice.
    """

    #: index -> species id, and the inverse.
    species: tuple[str, ...]
    index: dict[str, int]
    #: (S,) share of team slots this species appears on.
    usage: np.ndarray
    #: (S, S) lift, symmetric, diagonal zero because Species Clause forbids the pair.
    lift: np.ndarray
    #: Total weighted team count the tables were counted on.
    teams: float
    #: How far the two directions of the table disagreed, as
    #: ``|a-b| / (|a|+|b|)`` weighted by each pair's co-occurrence frequency. Weighted
    #: because an unweighted max is dominated by the rarest species, where one shared team
    #: is the whole count and the two directions round differently -- that number is 1.0
    #: and says nothing. Reported because a large value here *would* mean the
    #: co-occurrence reading above is wrong.
    asymmetry: float
    #: The same disagreement over the 2,000 pairs carrying the most mass.
    asymmetry_heavy: float
    #: Pairs the file records in one direction only.
    one_directional: int
    #: Fraction of pairs the data never recorded, which get the resolution bound instead.
    unrecorded_fraction: float

    def summary(self) -> str:
        n = len(self.species)
        off = self.lift[~np.eye(n, dtype=bool)]
        return "\n".join(
            [
                f"co-occurrence over {n} species on {self.teams:,.0f} weighted teams",
                f"  lift: median {np.median(off):.2f}, 90th pct "
                f"{np.percentile(off, 90):.2f}, max {off.max():.1f}",
                f"  pairs never recorded together: {self.unrecorded_fraction * 100:.1f}% "
                "(bounded by the table's own resolution, not zeroed)",
                f"  table asymmetry: {self.asymmetry:.1e} mass-weighted, "
                f"{self.asymmetry_heavy:.1e} over the heaviest pairs "
                "(a count table is symmetric)",
                f"  recorded in one direction only: {self.one_directional} pairs",
            ]
        )

    def top_partners(self, species_id: str, k: int = 8) -> list[tuple[str, float]]:
        i = self.index[species_id]
        order = np.argsort(-self.lift[i])[:k]
        return [(self.species[j], float(self.lift[i, j])) for j in order]


def build_cooccurrence(prior: MetagamePrior) -> Cooccurrence:
    """Turns the per-species teammate tables into one symmetric lift matrix."""
    entries = [s for s in prior.by_usage() if s.usage > 0 and s.teammates]
    if not entries:
        raise ValueError("usage data carries no teammate tables")
    ids = tuple(e.species_id for e in entries)
    index = {sid: i for i, sid in enumerate(ids)}
    usage = np.array([e.usage for e in entries], dtype=np.float64)
    n = len(ids)

    # `teammates` is normalised to sum to 1 on load, so the stored value is
    # co(a,b) / (5 * n_a), and therefore
    #   lift(a,b) = co(a,b) * T / (n_a * n_b) = 5 * teammates_a[b] / usage_b
    # which needs no absolute scale at all.
    lift = np.zeros((n, n), dtype=np.float64)
    recorded = np.zeros((n, n), dtype=bool)
    for i, entry in enumerate(entries):
        for partner, share in entry.teammates.items():
            j = index.get(partner)
            if j is None or i == j:
                continue
            lift[i, j] = 5.0 * share / usage[j]
            recorded[i, j] = True

    both = np.triu(recorded & recorded.T, k=1)
    if both.any():
        a, b = lift[both], lift.T[both]
        disagreement = np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12)
        # Weight by how often the pair actually occurs, so the summary describes the part
        # of the table the sampler reads rather than its resolution floor.
        i_idx, j_idx = np.nonzero(both)
        mass = lift[both] * usage[i_idx] * usage[j_idx]
        asymmetry = float((disagreement * mass).sum() / max(mass.sum(), 1e-12))
        heavy = np.argsort(-mass)[:2000]
        asymmetry_heavy = float(disagreement[heavy].max())
    else:
        asymmetry = asymmetry_heavy = float("nan")
    one_directional = int(np.triu(recorded ^ recorded.T, k=1).sum())

    # One direction can be missing where the other is present, when a partner sits below
    # the usage floor that built `entries`. Take whichever was recorded.
    only_fwd = recorded & ~recorded.T
    only_rev = recorded.T & ~recorded
    lift = np.where(only_fwd, lift, np.where(only_rev, lift.T, (lift + lift.T) / 2.0))
    recorded = recorded | recorded.T

    # A pair the data never recorded is not impossible; it is below the resolution of the
    # table, i.e. under half a weighted team. Bounding it that way states something about
    # the data instead of picking a constant that looks reasonable.
    teams = float(_teams_from_tables(prior))
    lift = np.where(recorded, lift, 0.5 / (teams * np.outer(usage, usage)))
    np.fill_diagonal(lift, 0.0)

    return Cooccurrence(
        species=ids,
        index=index,
        usage=usage,
        lift=lift,
        teams=teams,
        asymmetry=asymmetry,
        asymmetry_heavy=asymmetry_heavy,
        one_directional=one_directional,
        unrecorded_fraction=float((~recorded).sum() - n) / max(n * n - n, 1),
    )


def _teams_from_tables(prior: MetagamePrior) -> float:
    """The total weighted team count, read off the table's own consistency.

    Before normalisation ``sum(Teammates) == 5 * usage * T``, so ``T`` is recoverable from
    any single species. It is stored on the prior at load time; if a future stats file
    stops satisfying it, :attr:`MetagamePrior.weighted_teams` is nan and the resolution
    bound falls back to the smallest lift actually observed rather than inventing one.
    """
    if prior.weighted_teams and prior.weighted_teams > 0:
        return prior.weighted_teams
    raise ValueError(
        "this stats file does not satisfy sum(Teammates)/5/usage = const, so the "
        "co-occurrence counts cannot be put on an absolute scale"
    )


def sample_species_by_cooccurrence(
    rng: np.random.Generator,
    reg: Regulation,
    cooc: Cooccurrence,
    *,
    size: int | None = None,
) -> list[str]:
    """Six species drawn from the pairwise model, respecting Species Clause.

    Given the members already chosen, the next is drawn with probability proportional to
    its usage times its lift against each of them. That is the only model the data
    supports -- the chaos file records pairs, not teams -- and it is checkable:
    ``tools/cooc_report.py`` samples teams and compares the resulting usage and pair
    frequencies against the measured ones, so the approximation's error is a number.
    """
    size = size or reg.meta.team_size
    n = len(cooc.species)
    if n < size:
        raise ValueError(f"co-occurrence has {n} species, need {size}")

    bases = [reg.species[sid].base_species for sid in cooc.species]
    live = np.ones(n, dtype=bool)
    weights = cooc.usage.copy()
    chosen: list[str] = []
    taken: set[str] = set()
    for _ in range(size):
        w = np.where(live, weights, 0.0)
        total = float(w.sum())
        if total <= 0:
            options = np.flatnonzero(live)
            if not len(options):
                raise ValueError("Species Clause exhausted the pool")
            i = int(rng.choice(options))
        else:
            i = int(rng.choice(n, p=w / total))
        chosen.append(cooc.species[i])
        taken.add(bases[i])
        # Species Clause is on the base forme, so choosing Charizard removes every entry
        # that would bring the same Pokemon again.
        live &= np.array([b not in taken for b in bases])
        weights = weights * cooc.lift[i]
    return chosen


def sample_team(
    rng: np.random.Generator,
    reg: Regulation,
    prior: MetagamePrior,
    *,
    size: int | None = None,
) -> list[SampledSet]:
    """A team obeying Species Clause and the format's Item Clause.

    Species are drawn by usage; the item clause is enforced by resampling the item, and
    dropping it if no distinct one is available.
    """
    size = size or reg.meta.team_size
    pool = prior.by_usage()
    if not pool:
        raise ValueError("prior has no species")
    probs = np.array([s.usage for s in pool], dtype=np.float64)
    probs /= probs.sum()

    chosen: list[SampledSet] = []
    used_species: set[str] = set()
    used_items: set[str] = set()
    item_limit = reg.meta.item_clause

    guard = 0
    while len(chosen) < size and guard < size * 60:
        guard += 1
        i = int(rng.choice(len(pool), p=probs))
        sp_prior = pool[i]
        base = reg.species[sp_prior.species_id].base_species
        if base in used_species:
            continue
        candidate = sample_set(rng, reg, sp_prior)
        if item_limit is not None and candidate.item and candidate.item in used_items:
            alternatives = {
                k: v for k, v in sp_prior.items.items() if k not in used_items and k in reg.items
            }
            replacement = weighted_choice(rng, alternatives)
            candidate = SampledSet(
                species=candidate.species,
                ability=candidate.ability,
                item=replacement,
                nature=candidate.nature,
                sp=candidate.sp,
                moves=candidate.moves,
            )
        used_species.add(base)
        if candidate.item:
            used_items.add(candidate.item)
        chosen.append(candidate)

    if len(chosen) < size:
        raise ValueError(f"could only build {len(chosen)}/{size} team members")
    return chosen
