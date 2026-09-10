"""Real tournament teams as the opponent pool.

This replaces two approximations at once, and it is worth being explicit about what they
were standing in for.

A **curated archetype list** is a human summary of the field. It is only as complete as the
reports it was built from, and measuring it against the usage data showed the gap: eleven
archetypes used 28 of 220 species and accounted for 36% of the pairwise co-occurrence mass.

A **pairwise co-occurrence model** is statistically faithful to the pairs but cannot
represent a mixture. The field is a mixture -- a few dozen real sixes played many times --
and a pairwise model asked for a team will happily combine two popular Pokemon that never
appeared together in one plan.

395 actual entries need neither. Sampling one uniformly *is* the tournament metagame, and
each carries the real ability, nature, item and four moves rather than independent draws
from marginals. The 2026 Worlds file validates 386 of them against Regulation M-B with no
missing fields.

What the standings do **not** carry is the SP spread, and that is correct rather than
missing: Champions open team sheets reveal the nature and blank the investment. So the
spread is drawn from usage, conditioned on the nature the sheet states -- which is the same
conditioning the belief layer applies at analysis time, and the reason it was measured
(3.2x sharpening, exact).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .priors import MetagamePrior, SampledSet, weighted_choice
from .regulation import STAT_IDS, Regulation, repo_root, to_id


class StandingsError(ValueError):
    pass


def standings_dir() -> Path:
    return repo_root() / "data" / "standings"


@dataclass(frozen=True, slots=True)
class TeamMember:
    """One Pokemon exactly as the team sheet showed it, minus the hidden investment."""

    species: str
    ability: str | None
    item: str | None
    nature: str
    moves: tuple[str, ...]
    #: True when the recorded ability belongs to the *mega* forme and so cannot be the one
    #: the sheet showed. Champions sheets show the pre-mega ability, so the real one is
    #: unknown here and is drawn from usage instead -- reported, never guessed silently.
    ability_is_post_mega: bool = False


@dataclass(frozen=True, slots=True)
class TournamentTeam:
    player: str
    place: int
    country: str
    wins: int
    losses: int
    made_cut: bool
    phase_two: bool
    members: tuple[TeamMember, ...]

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(m.species for m in self.members)


@dataclass(slots=True)
class Standings:
    """Every entry that validated, plus an account of the ones that did not."""

    event: str
    event_format: str
    player_count: int
    teams: list[TournamentTeam]
    #: player -> why the entry was not usable. Printed rather than dropped, because a
    #: pool silently missing a third of the field is the failure this module exists to fix.
    rejected: dict[str, str] = field(default_factory=dict)
    #: Members whose recorded ability was the mega forme's.
    post_mega_abilities: tuple[str, ...] = ()

    def summary(self) -> str:
        cut = sum(1 for t in self.teams if t.made_cut)
        phase2 = sum(1 for t in self.teams if t.phase_two)
        lines = [
            f"{self.event} ({self.event_format}): {len(self.teams)} usable teams "
            f"of {self.player_count} entries",
            f"  top cut {cut}, phase two {phase2}",
        ]
        if self.rejected:
            reasons: dict[str, int] = {}
            for reason in self.rejected.values():
                key = reason.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
            detail = ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items()))
            lines.append(f"  rejected {len(self.rejected)}: {detail}")
        if self.post_mega_abilities:
            lines.append(
                "  ability recorded as the mega forme's, so drawn from usage instead: "
                + ", ".join(sorted(set(self.post_mega_abilities)))
            )
        return "\n".join(lines)

    def pool(self, which: str = "all") -> list[TournamentTeam]:
        """The whole field, the players who reached day two, or the top cut.

        The default is the whole field on purpose. Restricting to the cut would train
        against the teams that won, which is a different distribution from the teams a
        player actually faces -- and the cut is 13 teams, which is not a metagame.
        """
        if which == "all":
            return list(self.teams)
        if which == "phase2":
            return [t for t in self.teams if t.phase_two]
        if which == "cut":
            return [t for t in self.teams if t.made_cut]
        raise StandingsError(f"unknown pool {which!r}; use all, phase2 or cut")


def find_cached_standings(season: str = "2026", event: str = "worlds") -> Path | None:
    path = standings_dir() / f"{season}-{event}.json.gz"
    return path if path.exists() else None


def load_standings(path: str | Path, reg: Regulation) -> Standings:
    """Parses a Reportworm standings dump, validating every field against the regulation.

    A team is kept only if all six members are team-legal with legal moves and items; a
    team with one unparseable member is rejected whole, because dropping the member would
    leave a five-Pokemon "real team" that nobody brought.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        data = json.load(handle)

    event = data.get("event") or {}
    teams: list[TournamentTeam] = []
    rejected: dict[str, str] = {}
    post_mega: list[str] = []

    for code, entry in (data.get("standings") or {}).items():
        raw_team = entry.get("team") or []
        name = str(entry.get("name") or code)
        if len(raw_team) != reg.meta.team_size:
            rejected[name] = f"team size: {len(raw_team)} members"
            continue

        members: list[TeamMember] = []
        problem: str | None = None
        for raw in raw_team:
            species_id = to_id(raw.get("code") or raw.get("name") or "")
            species = reg.species.get(species_id)
            if species is None or not species.team_legal:
                problem = f"species: {raw.get('name')!r} is not team-legal"
                break

            moves = tuple(to_id(m["name"]) for m in (raw.get("moves") or []))
            unknown = [m for m in moves if m not in reg.moves]
            if unknown:
                problem = f"move: {', '.join(unknown)}"
                break

            # "noitem" is how the API spells an empty item slot.
            item_id = to_id(raw.get("itemcode") or "")
            if item_id in ("", "noitem", "nothing"):
                item_id = None
            elif item_id not in reg.items:
                problem = f"item: {item_id}"
                break

            ability_id: str | None = to_id(raw.get("ability") or "") or None
            is_post_mega = False
            if ability_id is not None and ability_id not in {
                to_id(a) for a in species.abilities
            }:
                # Almost always the mega forme's ability (Drought on Charizard). The sheet
                # shows the pre-mega one, which this file does not carry, so it becomes
                # unknown and the prior fills it.
                if ability_id in reg.abilities:
                    is_post_mega = True
                    post_mega.append(f"{species.name}:{raw.get('ability')}")
                    ability_id = None
                else:
                    problem = f"ability: {raw.get('ability')!r}"
                    break

            nature = str(raw.get("nature") or "")
            if nature not in reg.natures:
                problem = f"nature: {nature!r}"
                break

            members.append(
                TeamMember(
                    species=species_id,
                    ability=ability_id,
                    item=item_id,
                    nature=nature,
                    moves=moves,
                    ability_is_post_mega=is_post_mega,
                )
            )

        if problem is not None:
            rejected[name] = problem
            continue

        bases = [reg.species[m.species].base_species for m in members]
        if len(set(bases)) != len(bases):
            rejected[name] = "species clause: a base species appears twice"
            continue
        items = [m.item for m in members if m.item]
        if reg.meta.item_clause is not None and len(set(items)) != len(items):
            rejected[name] = "item clause: an item appears twice"
            continue

        record = entry.get("record") or {}
        teams.append(
            TournamentTeam(
                player=name,
                place=int(entry.get("place") or 0),
                country=str(entry.get("country") or ""),
                wins=int(record.get("w") or 0),
                losses=int(record.get("l") or 0),
                made_cut=bool(entry.get("cut")),
                phase_two=bool(entry.get("p2")),
                members=tuple(members),
            )
        )

    if not teams:
        raise StandingsError(f"{p.name}: no team validated against {reg.meta.format_id}")
    return Standings(
        event=str(event.get("name") or p.stem),
        event_format=str(event.get("format") or "?"),
        player_count=int(event.get("playerCount") or len(data.get("standings") or {})),
        teams=teams,
        rejected=rejected,
        post_mega_abilities=tuple(post_mega),
    )


def _spread_for(
    rng: np.random.Generator, prior: MetagamePrior, species_id: str, nature: str
) -> dict[str, int]:
    """An SP spread from usage, conditioned on the nature the team sheet reveals.

    The prior's particles are joint over (nature, spread), so conditioning is a filter
    rather than a model: keep the particles with this nature and renormalise. That is the
    same operation the belief layer performs during analysis, where it was measured to
    sharpen the distribution 3.2x exactly.

    Falling back to the unconditioned particles when a nature was never observed is stated
    rather than hidden -- it means the sheet shows a nature nobody in the usage sample
    played, which is information, and refusing to build the team would lose a real entry.
    """
    entry = prior.species.get(species_id)
    if entry is None:
        return {}
    match = np.asarray(entry.natures) == nature
    weights = entry.weights * match
    total = float(weights.sum())
    if total <= 0:
        weights = entry.weights
        total = float(weights.sum())
        if total <= 0:
            return {}
    index = int(rng.choice(entry.n_particles, p=weights / total))
    return {
        stat: int(value)
        for stat, value in zip(STAT_IDS, entry.spreads[index], strict=True)
        if value
    }


def sample_standings_team(
    rng: np.random.Generator,
    reg: Regulation,
    prior: MetagamePrior,
    team: TournamentTeam,
) -> list[SampledSet]:
    """A real tournament six, with only the hidden investment filled in from usage."""
    out: list[SampledSet] = []
    for member in team.members:
        ability = member.ability
        if ability is None:
            entry = prior.species.get(member.species)
            legal = {to_id(a) for a in reg.species[member.species].abilities}
            table = (
                {k: v for k, v in entry.abilities.items() if k in legal} if entry else {}
            )
            ability = weighted_choice(rng, table) or to_id(
                reg.species[member.species].abilities[0]
            )
        out.append(
            SampledSet(
                species=member.species,
                ability=ability,
                item=member.item,
                nature=member.nature,
                sp=_spread_for(rng, prior, member.species, member.nature),
                moves=list(member.moves),
            )
        )
    return out


def species_frequency(teams: list[TournamentTeam]) -> dict[str, float]:
    """Share of teams containing each species, for comparing against ladder usage."""
    counts: dict[str, int] = {}
    for team in teams:
        for species_id in set(team.species):
            counts[species_id] = counts.get(species_id, 0) + 1
    return {k: v / len(teams) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def cluster_teams(
    teams: list[TournamentTeam], *, min_shared: int = 5
) -> list[list[TournamentTeam]]:
    """Groups teams sharing at least ``min_shared`` of six species.

    Only for reporting: per-archetype win rates need labels, and labels no longer need to
    exist for *sampling* now that the real teams do. Single-link clustering on species
    overlap is crude, but it is stated and it produces the groups a reader recognises --
    the eight identical BIG6 lists land together.
    """
    parent = list(range(len(teams)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sets = [set(t.species) for t in teams]
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            if len(sets[i] & sets[j]) >= min_shared:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    groups: dict[int, list[TournamentTeam]] = {}
    for i, team in enumerate(teams):
        groups.setdefault(find(i), []).append(team)
    return sorted(groups.values(), key=lambda g: (-len(g), min(t.place for t in g)))


def cluster_core(group: list[TournamentTeam], *, top: int = 3) -> list[tuple[str, float]]:
    """The species most of a cluster shares, with the share that brings each.

    Not the strict intersection. Single-link clustering chains -- A joins B and B joins C
    even when A and C share little -- so a 74-team group can have an intersection of one
    species while every member is recognisably the same deck. The modal species with their
    frequencies describe such a group honestly, and the frequency is what says whether the
    label is tight (100%) or loose (60%).
    """
    counts: dict[str, int] = {}
    for team in group:
        for species_id in set(team.species):
            counts[species_id] = counts.get(species_id, 0) + 1
    # Ties broken by id, so the same cluster always produces the same ordering. Without it
    # two clusters sharing a core get labels that differ only by permutation, which reads
    # as one archetype written twice.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [(sid, n / len(group)) for sid, n in ranked]


def label_for(group: list[TournamentTeam], reg: Regulation) -> str:
    """A readable, stable name for a cluster, from its most common members.

    The best placement in the cluster is appended as a disambiguator: two clusters can
    genuinely share their three most common Pokemon while being different decks, and a
    per-archetype win rate that silently merged them would be reporting an average of two
    matchups as though it were one.
    """
    core = cluster_core(group)
    if not core:
        return "unknown"
    names = "+".join(sorted(reg.species[sid].name for sid, _ in core))
    return f"{names} #{min(t.place for t in group)}"


__all__ = [
    "Standings",
    "StandingsError",
    "TeamMember",
    "TournamentTeam",
    "cluster_core",
    "cluster_teams",
    "find_cached_standings",
    "label_for",
    "load_standings",
    "sample_standings_team",
    "species_frequency",
    "standings_dir",
]
