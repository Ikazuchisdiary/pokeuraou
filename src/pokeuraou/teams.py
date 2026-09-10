"""The teams self-play is played with, and where each number in them came from.

A learned value function is only calibrated on the distribution it was trained on, so the
team distribution is not a detail -- it decides what the tool is valid for. Three sources
feed it, and they are kept distinct on purpose:

- **A roster**, which is a real team recorded exactly. Ours is transcribed from the game's
  own team-sheet and status screens, so every SP value and every final stat is the real
  one. :func:`verify_shown_stats` re-derives the stats and compares, which makes the file
  a regression test on the SP formula against the *game* rather than against Showdown.
- **An archetype**, which is a real 6-species composition from a tournament report. The
  composition is cited; the sets are only there when the report gave numbers.
- **The usage prior**, which fills in every field a report left unstated.

What is deliberately absent is a fourth source: making a set up. A plausible-looking
spread invented here would flow into the value function and out again as a win
probability, with nothing to trace it back to. So :func:`sample_archetype` draws unstated
fields from measured usage, and an archetype whose species has no usage data raises.

Selection is 6 bring / 4 pick, and picking is not modelled: :func:`pick_four` draws
uniformly from the six. That is not the real distribution -- players pick for the matchup,
and a uniform draw gives leads nobody would lead with -- but it is the only choice that can
be *stated*, and it covers all ninety *ordered* selections (C(6,2) lead pairs x C(4,2)
behind them) so the value function is usable whichever four are brought and whichever two
of them lead. This said "fifteen combinations" while `pick_four` said ninety: the unordered
count is the one that stopped being right when the leads started being drawn separately.
Solving the selection is its own problem, and it has the same Bayesian shape as everything
else here.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from dataclasses import field as dataclasses_field
from pathlib import Path
from typing import Any

import numpy as np

from .oracle import TeamSet
from .priors import (
    Cooccurrence,
    MetagamePrior,
    SampledSet,
    sample_set,
    sample_species_by_cooccurrence,
    weighted_choice,
)
from .regulation import STAT_IDS, Regulation, load_regulation, repo_root, to_id
from .stats import nature_multipliers, stats_from_sp


class TeamError(ValueError):
    pass


def teams_dir() -> Path:
    return repo_root() / "configs" / "teams"


def archetypes_dir() -> Path:
    return repo_root() / "configs" / "archetypes"


@dataclass(slots=True)
class Roster:
    """A real team, recorded exactly."""

    id: str
    name: str
    reg: Regulation
    sets: list[SampledSet]
    #: Per Pokemon, the final stats the game's own status screen showed, when recorded.
    shown_stats: list[dict[str, int] | None]
    source: dict[str, Any]

    def team_sets(self) -> list[TeamSet]:
        return [TeamSet.from_json(s.to_team_set_json(self.reg)) for s in self.sets]


@dataclass(slots=True)
class Archetype:
    """A cited 6-species composition, with whatever sets the source stated."""

    id: str
    name: str
    species: tuple[str, ...]
    plan: str
    source: str
    #: species id -> the partially or fully stated set, if the source gave one.
    stated: dict[str, dict[str, Any]]
    #: species id -> the Mega Stone the archetype's own name commits it to. Naming a
    #: member "メガリザードンY" states the item, so leaving it to the prior would produce
    #: a Choice Scarf Charizard in a Mega-Charizard archetype and lose its identity.
    mega_stones: dict[str, str] = dataclasses_field(default_factory=dict)
    #: How many teams in a cited standings list brought exactly these six, and out of how
    #: many. ``None`` means the archetype comes from a build article rather than a
    #: standings list, so there is no observation to weight it by -- not that it is rare.
    observed_count: int = 0
    observed_of: int = 0
    observed_event: str = ""
    #: Where two sources disagreed about a member, kept verbatim rather than resolved.
    source_disagreement: str = ""

    @property
    def observed(self) -> bool:
        return self.observed_of > 0


def _sp(raw: dict[str, Any] | None, reg: Regulation, label: str) -> dict[str, int]:
    values = {stat: int((raw or {}).get(stat, 0)) for stat in STAT_IDS}
    if sum(values.values()) > reg.meta.sp_limit:
        raise TeamError(
            f"{label}: SP total {sum(values.values())} exceeds {reg.meta.sp_limit}"
        )
    over = [s for s, v in values.items() if v > reg.meta.sp_per_stat_max]
    if over:
        raise TeamError(
            f"{label}: {', '.join(over)} above the per-stat cap {reg.meta.sp_per_stat_max}"
        )
    return values


def _check_set(reg: Regulation, entry: dict[str, Any], label: str) -> None:
    species_id = to_id(entry["species"])
    species = reg.species.get(species_id)
    if species is None or not species.team_legal:
        raise TeamError(f"{label}: {entry['species']!r} is not team-legal in {reg.meta.format_id}")
    ability = entry.get("ability")
    if ability is not None and to_id(ability) not in {to_id(a) for a in species.abilities}:
        raise TeamError(
            f"{label}: {ability!r} is not an ability of {species.name} "
            f"({', '.join(species.abilities)}) -- a team sheet shows the *pre-mega* ability"
        )
    item = entry.get("item")
    if item and to_id(item) not in reg.items:
        raise TeamError(f"{label}: item {item!r} is not legal in {reg.meta.format_id}")
    nature = entry.get("nature")
    if nature and nature not in reg.natures:
        raise TeamError(f"{label}: nature {nature!r} is unknown")
    for move in entry.get("moves") or []:
        if to_id(move) not in reg.moves:
            raise TeamError(f"{label}: move {move!r} is not legal in {reg.meta.format_id}")


def load_roster(path: str | Path) -> Roster:
    """Reads an exactly recorded team, validating every field against the regulation."""
    file = Path(path)
    if not file.exists():
        file = teams_dir() / f"{path}.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    reg = load_regulation(str(data["regulation"]))

    sets: list[SampledSet] = []
    shown: list[dict[str, int] | None] = []
    items: set[str] = set()
    species_seen: set[str] = set()
    for index, entry in enumerate(data["team"]):
        label = f"{data.get('id', file.stem)}[{index}] {entry.get('species')}"
        _check_set(reg, entry, label)
        if entry.get("ability") is None:
            raise TeamError(f"{label}: a recorded team has to state the ability")
        species_id = to_id(entry["species"])
        base = reg.species[species_id].base_species
        if base in species_seen:
            raise TeamError(f"{label}: Species Clause -- {base} appears twice")
        species_seen.add(base)
        item_id = to_id(entry["item"]) if entry.get("item") else None
        if reg.meta.item_clause is not None and item_id and item_id in items:
            raise TeamError(f"{label}: Item Clause -- {item_id} appears twice")
        if item_id:
            items.add(item_id)
        sets.append(
            SampledSet(
                species=species_id,
                ability=to_id(entry["ability"]),
                item=item_id,
                nature=str(entry["nature"]),
                sp=_sp(entry.get("sp"), reg, label),
                moves=[to_id(m) for m in entry["moves"]],
            )
        )
        raw_shown = entry.get("shownStats")
        shown.append(
            {stat: int(raw_shown[stat]) for stat in STAT_IDS} if raw_shown else None
        )

    if len(sets) != reg.meta.team_size:
        raise TeamError(
            f"{file.name}: {len(sets)} Pokemon, but this regulation brings "
            f"{reg.meta.team_size}"
        )
    return Roster(
        id=str(data.get("id", file.stem)),
        name=str(data.get("name", file.stem)),
        reg=reg,
        sets=sets,
        shown_stats=shown,
        source=dict(data.get("source") or {}),
    )


def verify_shown_stats(roster: Roster) -> list[tuple[str, list[int], list[int]]]:
    """Recomputes each recorded Pokemon's stats and returns the mismatches.

    An empty list means our SP formula reproduces the numbers the game itself displayed --
    a check against the real thing, where every other stat test in this project compares
    against Showdown.
    """
    out: list[tuple[str, list[int], list[int]]] = []
    for entry, shown in zip(roster.sets, roster.shown_stats, strict=True):
        if shown is None:
            continue
        base = np.array(roster.reg.species[entry.species].base_stats, dtype=np.int64)
        got = stats_from_sp(
            roster.reg,
            base,
            np.array([[entry.sp[s] for s in STAT_IDS]], dtype=np.int64),
            nature_multipliers(roster.reg, [entry.nature]),
            level=roster.reg.meta.level,
        )[0].tolist()
        want = [shown[s] for s in STAT_IDS]
        if got != want:
            out.append((entry.species, got, want))
    return out


def load_archetypes(path: str | Path) -> tuple[Regulation, list[Archetype]]:
    file = Path(path)
    if not file.exists():
        file = archetypes_dir() / f"{path}.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    reg = load_regulation(str(data["regulation"]))

    out: list[Archetype] = []
    for entry in data["archetypes"]:
        label = f"{entry['id']}"
        species = tuple(to_id(s) for s in entry["species"])
        for species_id in species:
            found = reg.species.get(species_id)
            if found is None or not found.team_legal:
                raise TeamError(f"{label}: {species_id!r} is not team-legal here")
        if len(set(species)) != len(species):
            raise TeamError(f"{label}: Species Clause -- a species appears twice")
        stated: dict[str, dict[str, Any]] = {}
        for record in entry.get("sets") or []:
            _check_set(reg, record, f"{label} {record.get('species')}")
            stated[to_id(record["species"])] = record
        unknown = set(stated) - set(species)
        if unknown:
            raise TeamError(f"{label}: sets given for non-members {sorted(unknown)}")
        stones: dict[str, str] = {}
        for raw_species, raw_item in (entry.get("megaStones") or {}).items():
            member = to_id(raw_species)
            if member not in species:
                raise TeamError(f"{label}: mega stone given for non-member {member!r}")
            item_id = to_id(raw_item)
            if reg.mega_target(member, item_id) is None:
                raise TeamError(
                    f"{label}: {raw_item!r} does not mega-evolve {raw_species!r} in "
                    f"{reg.meta.format_id}"
                )
            stones[member] = item_id
        seen = entry.get("observed") or {}
        out.append(
            Archetype(
                id=str(entry["id"]),
                name=str(entry.get("name", entry["id"])),
                species=species,
                plan=str(entry.get("plan", "")),
                source=str(entry.get("source", "")),
                stated=stated,
                mega_stones=stones,
                observed_count=int(seen.get("count", 0)),
                observed_of=int(seen.get("of", 0)),
                observed_event=str(seen.get("event", "")),
                source_disagreement=str(entry.get("sourceDisagreement", "")),
            )
        )
    if not out:
        raise TeamError(f"{file.name}: no archetypes")
    return reg, out


def sample_archetype(
    rng: np.random.Generator,
    reg: Regulation,
    prior: MetagamePrior,
    archetype: Archetype,
    *,
    follow_source: bool = False,
) -> list[SampledSet]:
    """The archetype's six, with the spreads drawn from measured usage.

    What an archetype fixes is its *identity*: the six species, and the Mega Stone its own
    name commits a member to. Everything else is sampled, and that is deliberate -- the
    same composition is played with many different spreads, which is the entire reason the
    belief layer exists. Pinning one published spread per archetype would train the value
    function against one specific opponent and teach it that the opponent's investment is
    knowable, which is the opposite of true.

    ``follow_source=True`` uses the published set where the report gave one. That is for
    reproducing a specific tournament team, not for generating training data.

    Item Clause is enforced by resampling, with the fixed items reserved first so a
    sampled set cannot collide with one the archetype commits to.
    """
    used_items: set[str] = set(archetype.mega_stones.values())
    if follow_source:
        used_items |= {
            to_id(record["item"])
            for record in archetype.stated.values()
            if record.get("item")
        }
    out: list[SampledSet] = []
    for species_id in archetype.species:
        stated = archetype.stated.get(species_id, {}) if follow_source else {}
        entry = prior.species.get(species_id)
        needs_prior = any(
            stated.get(field) is None for field in ("ability", "item", "nature", "moves")
        ) or "sp" not in stated
        if entry is None and needs_prior:
            raise TeamError(
                f"{archetype.id}: {species_id} has unstated fields and no usage data to "
                "fill them; inventing a set would put an untraceable number into training"
            )

        drawn = sample_set(rng, reg, entry) if entry is not None else None
        ability = to_id(stated["ability"]) if stated.get("ability") else drawn.ability
        nature = str(stated["nature"]) if stated.get("nature") else drawn.nature
        moves = (
            [to_id(m) for m in stated["moves"]]
            if stated.get("moves")
            else list(drawn.moves)
        )
        sp = (
            _sp(stated.get("sp"), reg, f"{archetype.id} {species_id}")
            if "sp" in stated
            else dict(drawn.sp)
        )

        if stated.get("item"):
            item = to_id(stated["item"])
        elif species_id in archetype.mega_stones:
            item = archetype.mega_stones[species_id]
            used_items.add(item)
        else:
            item = drawn.item
            if reg.meta.item_clause is not None and item and item in used_items:
                alternatives = {
                    k: v
                    for k, v in (entry.items if entry else {}).items()
                    if k not in used_items and k in reg.items
                }
                item = _heaviest(rng, alternatives)
            if item:
                used_items.add(item)

        out.append(
            SampledSet(
                species=species_id,
                ability=ability,
                item=item,
                nature=nature,
                sp=sp,
                moves=moves,
            )
        )
    return out


def sample_metagame_team(
    rng: np.random.Generator,
    reg: Regulation,
    prior: MetagamePrior,
    cooc: Cooccurrence,
) -> list[SampledSet]:
    """Six drawn from the measured pairwise structure, with sets from usage.

    This is the fourth source, and it exists because the third was not enough. The eleven
    cited archetypes are real teams, but between them they use 28 distinct species out of
    220 with usage data, and they miss Sneasler (31.4%) and Sinistcha (26.1%) entirely --
    the fifth and seventh most-used Pokemon in the format. A value function trained only
    against them is calibrated against a metagame nobody plays.

    Nothing here is invented: the species come from the co-occurrence counts, the sets from
    the same marginals every other opponent uses. What it gives up is *identity* -- a
    sampled six has no plan and no author, so it is the right pool for training coverage
    and the wrong one for reproducing a specific tournament team, which is what
    :func:`sample_archetype` is still for.
    """
    species = sample_species_by_cooccurrence(rng, reg, cooc)
    used_items: set[str] = set()
    out: list[SampledSet] = []
    for species_id in species:
        entry = prior.species.get(species_id)
        if entry is None:
            raise TeamError(f"{species_id} came out of co-occurrence with no usage data")
        member = sample_set(rng, reg, entry)
        if reg.meta.item_clause is not None and member.item and member.item in used_items:
            alternatives = {
                k: v
                for k, v in entry.items.items()
                if k not in used_items and k in reg.items
            }
            member = replace(member, item=weighted_choice(rng, alternatives))
        if member.item:
            used_items.add(member.item)
        out.append(member)
    return out


def archetype_weights(archetypes: list[Archetype]) -> np.ndarray:
    """Sampling weights from observed tournament counts.

    Proportional to the observed count, with archetypes that have no observation at zero.
    No smoothing: a floor would be a number nobody measured, and the honest statement is
    "this archetype was not in the standings I have", which the caller can report.
    """
    weights = np.array([float(a.observed_count) for a in archetypes], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        raise TeamError(
            "no archetype carries an observed tournament count; use the uniform pool or "
            "add a standings source"
        )
    return weights / total


def sample_archetype_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    return int(rng.choice(len(weights), p=weights))


def usable_archetypes(
    prior: MetagamePrior, archetypes: list[Archetype]
) -> tuple[list[Archetype], list[tuple[Archetype, tuple[str, ...]]]]:
    """Splits archetypes into the ones that can be built and the ones that cannot.

    A composition is real and cited, but a member too rare to appear in the usage data
    leaves fields with nothing to fill them. Rather than dropping such an archetype
    quietly -- which would silently narrow the training distribution -- the caller gets
    both lists and can say which compositions the opponent pool is missing, and why.
    """
    usable: list[Archetype] = []
    blocked: list[tuple[Archetype, tuple[str, ...]]] = []
    for archetype in archetypes:
        missing = tuple(
            species
            for species in archetype.species
            if species not in prior.species
            and not _fully_stated(archetype.stated.get(species))
        )
        if missing:
            blocked.append((archetype, missing))
        else:
            usable.append(archetype)
    return usable, blocked


def _fully_stated(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return all(
        record.get(field) is not None
        for field in ("ability", "item", "nature", "moves", "sp")
    )


def _heaviest(rng: np.random.Generator, table: dict[str, float]) -> str | None:
    if not table:
        return None
    keys = list(table)
    weights = np.array([table[k] for k in keys], dtype=np.float64)
    weights /= weights.sum()
    return str(rng.choice(keys, p=weights))


def pick_four(
    rng: np.random.Generator, six: list[SampledSet], size: int = 4
) -> list[SampledSet]:
    """Uniformly picks an *ordered* selection: the first two are the leads.

    Ordered because the first two start on the field, which makes them a different choice
    from the two behind them. Returning the four in party order instead -- which this used
    to do -- lets the team file decide the leads: only 6 of the 15 possible lead pairs ever
    appeared, and whichever two members were listed last could never lead at all. Since the
    turn-1 position *is* the selection, that put a systematic hole in exactly the part of
    the dataset the value function is meant to improve.

    Uniform is still not the real distribution -- a player picks for the matchup -- but it
    is the only rule statable without modelling selection, and it covers all 90 ordered
    selections so the value function is usable whichever one is brought. Solving selection
    is its own game and has the same shape as everything else here: a 90x90 simultaneous
    move whose cells are win probabilities, i.e. exactly what the value function supplies.
    """
    return [six[i] for i in pick_four_indices(rng, len(six), size)]


def pick_four_indices(
    rng: np.random.Generator, available: int, size: int = 4
) -> tuple[int, ...]:
    """The ordered selection as party indices, leads first.

    Separate from :func:`pick_four` so a caller can record *which* four were brought and in
    what order. Recovering that from the returned sets would mean matching by identity or by
    species, and an inverse that can be avoided should be.
    """
    if available < size:
        raise TeamError(f"cannot pick {size} from {available}")
    # Two independent uniform draws: which `size` come, and which two of those lead.
    chosen = [int(i) for i in rng.choice(available, size=size, replace=False)]
    leads = [int(i) for i in rng.choice(size, size=2, replace=False)]
    front = [chosen[i] for i in sorted(leads)]
    back = [chosen[i] for i in range(size) if i not in leads]
    return tuple(front + back)


def all_picks(six: list[SampledSet], size: int = 4) -> list[tuple[int, ...]]:
    """Every *set* of ``size`` from six, ignoring who leads."""
    return list(itertools.combinations(range(len(six)), size))


def all_selections(team_size: int = 6, size: int = 4) -> list[tuple[int, ...]]:
    """Every ordered selection: leads first, then the bench.

    C(6,2) lead pairs x C(4,2) bench pairs = 90 for the standard bring-6-pick-4. Within a
    pair the order is not modelled: in doubles both leads are adjacent to both foes, and a
    replacement is chosen freely from the bench, so a pair is a set and this is the whole
    action space rather than a sample of it.
    """
    out: list[tuple[int, ...]] = []
    for front in itertools.combinations(range(team_size), 2):
        rest = [i for i in range(team_size) if i not in front]
        for back in itertools.combinations(rest, size - 2):
            out.append(tuple(front) + tuple(back))
    return out


def selection_from_indices(
    six: list[SampledSet], selection: tuple[int, ...]
) -> list[SampledSet]:
    """The sets for one ordered selection, leads first."""
    return [six[i] for i in selection]


__all__ = [
    "Archetype",
    "Roster",
    "TeamError",
    "all_picks",
    "all_selections",
    "archetypes_dir",
    "load_archetypes",
    "load_roster",
    "pick_four",
    "sample_archetype",
    "teams_dir",
    "usable_archetypes",
    "verify_shown_stats",
]
