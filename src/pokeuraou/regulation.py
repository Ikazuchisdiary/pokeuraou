"""Regulation config loading.

The config is generated from the pinned Showdown dex by
``packages/sim-bridge/src/cli/dump-regulation.ts``. Nothing about a regulation is
hardcoded here: legal species, the item pool, mega-capable species and the SP limits all
come from the JSON, so a new regulation is a new file rather than a code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

STAT_IDS: tuple[str, ...] = ("hp", "atk", "def", "spa", "spd", "spe")
STAT_INDEX: dict[str, int] = {s: i for i, s in enumerate(STAT_IDS)}
BOOST_IDS: tuple[str, ...] = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
BOOST_INDEX: dict[str, int] = {s: i for i, s in enumerate(BOOST_IDS)}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def regulation_dir() -> Path:
    return repo_root() / "configs" / "regulations"


@dataclass(frozen=True, slots=True)
class Meta:
    format_id: str
    format_name: str
    mod: str
    game_type: str
    generation: int
    sp_limit: int
    sp_per_stat_max: int
    fixed_iv: int
    level: int
    uses_level_clause_mod: bool
    team_size: int
    picked_team_size: int
    active_per_side: int
    open_team_sheets: bool
    team_sheet_reveals_nature: bool
    item_clause: int | None
    max_move_count: int
    tera_enabled: bool
    mega_per_side: int
    showdown_commit: str


@dataclass(frozen=True, slots=True)
class Nature:
    name: str
    plus: str | None
    minus: str | None
    exists_in_real_game: bool


@dataclass(frozen=True, slots=True)
class Species:
    id: str
    name: str
    num: int
    base_species: str
    forme: str
    types: tuple[str, ...]
    base_stats: tuple[int, ...]  # in STAT_IDS order
    abilities: tuple[str, ...]
    weightkg: float
    is_mega: bool
    required_item: str | None
    changes_from: str | None
    team_legal: bool


@dataclass(frozen=True, slots=True)
class Move:
    id: str
    name: str
    type: str
    category: str  # 'Physical' | 'Special' | 'Status'
    base_power: int
    accuracy: int | None  # None means "never misses"
    pp: int
    priority: int
    target: str
    crit_ratio: int
    flags: frozenset[str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def has_custom_code(self) -> bool:
        return bool(self.raw.get("hasCustomCode"))

    @property
    def custom_hooks(self) -> tuple[str, ...]:
        return tuple(self.raw.get("customHooks", ()))


@dataclass(frozen=True, slots=True)
class Item:
    id: str
    name: str
    is_berry: bool
    mega_stone: dict[str, str] | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass(frozen=True, slots=True)
class Ability:
    id: str
    name: str
    flags: frozenset[str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


# Move target values that take an explicit target index in a doubles choice string.
# Showdown rejects a choice that supplies a target for any other target type, and rejects
# one that omits it for these -- verified against `probe_choices`.
TARGETS_REQUIRING_FOE = frozenset({"normal", "any", "adjacentFoe"})
TARGETS_REQUIRING_ALLY = frozenset({"adjacentAlly", "adjacentAllyOrSelf"})
TARGETS_WITHOUT_CHOICE = frozenset(
    {
        "self",
        "all",
        "allAdjacent",
        "allAdjacentFoes",
        "allies",
        "allySide",
        "allyTeam",
        "foeSide",
        "randomNormal",
        "scripted",
    }
)


class Regulation:
    """Indexed view over one regulation config file."""

    def __init__(self, data: dict[str, Any]) -> None:
        m = data["meta"]
        self.meta = Meta(
            format_id=m["formatId"],
            format_name=m["formatName"],
            mod=m["mod"],
            game_type=m["gameType"],
            generation=m["generation"],
            sp_limit=m["spLimit"],
            sp_per_stat_max=m["spPerStatMax"],
            fixed_iv=m["fixedIv"],
            level=m["level"],
            uses_level_clause_mod=m["usesLevelClauseMod"],
            team_size=m["teamSize"],
            picked_team_size=m["pickedTeamSize"],
            active_per_side=m["activePerSide"],
            open_team_sheets=m["openTeamSheets"],
            team_sheet_reveals_nature=m["teamSheetRevealsNature"],
            item_clause=m["itemClause"],
            max_move_count=m["maxMoveCount"],
            tera_enabled=m["teraEnabled"],
            mega_per_side=m["megaPerSide"],
            showdown_commit=m["showdownCommit"],
        )

        self.natures: dict[str, Nature] = {}
        for n in data["natures"]:
            nat = Nature(
                name=n["name"],
                plus=n.get("plus"),
                minus=n.get("minus"),
                exists_in_real_game=n["existsInRealGame"],
            )
            self.natures[nat.name] = nat
            self.natures[to_id(nat.name)] = nat

        self.types: tuple[str, ...] = tuple(data["types"])
        self.typechart: dict[str, dict[str, float]] = data["typechart"]

        self.species: dict[str, Species] = {}
        for s in data["species"]:
            sp = Species(
                id=s["id"],
                name=s["name"],
                num=s["num"],
                base_species=s["baseSpecies"],
                forme=s["forme"],
                types=tuple(s["types"]),
                base_stats=tuple(s["baseStats"][k] for k in STAT_IDS),
                abilities=tuple(s["abilities"]),
                weightkg=s["weightkg"],
                is_mega=s["isMega"],
                required_item=s.get("requiredItem"),
                changes_from=s.get("changesFrom"),
                team_legal=s["teamLegal"],
            )
            self.species[sp.id] = sp

        self.moves: dict[str, Move] = {}
        for mv in data["moves"]:
            acc = mv["accuracy"]
            move = Move(
                id=mv["id"],
                name=mv["name"],
                type=mv["type"],
                category=mv["category"],
                base_power=mv["basePower"],
                accuracy=None if acc is True else int(acc),
                pp=mv["pp"],
                priority=mv["priority"],
                target=mv["target"],
                crit_ratio=mv.get("critRatio", 1),
                flags=frozenset(mv.get("flags", {})),
                raw=mv,
            )
            self.moves[move.id] = move

        self.items: dict[str, Item] = {}
        for it in data["items"]:
            item = Item(
                id=it["id"],
                name=it["name"],
                is_berry=it["isBerry"],
                mega_stone=it.get("megaStone"),
                raw=it,
            )
            self.items[item.id] = item

        self.abilities: dict[str, Ability] = {}
        for ab in data["abilities"]:
            ability = Ability(
                id=ab["id"], name=ab["name"], flags=frozenset(ab.get("flags", {})), raw=ab
            )
            self.abilities[ability.id] = ability

        # itemId -> {fromSpeciesId: toSpeciesId}
        self.mega_map: dict[str, dict[str, str]] = data["megaMap"]
        # (speciesId, itemId) -> megaSpeciesId, the lookup the resolver actually needs
        self.mega_by_species: dict[tuple[str, str], str] = {
            (from_id, item_id): to_id_
            for item_id, table in self.mega_map.items()
            for from_id, to_id_ in table.items()
        }

    # -- convenience ---------------------------------------------------------

    @property
    def team_legal_species(self) -> list[Species]:
        return [s for s in self.species.values() if s.team_legal]

    @property
    def real_game_natures(self) -> list[Nature]:
        seen: dict[str, Nature] = {}
        for n in self.natures.values():
            if n.exists_in_real_game:
                seen[n.name] = n
        return sorted(seen.values(), key=lambda n: n.name)

    def type_effectiveness(self, attacking: str, defending_types: tuple[str, ...]) -> float:
        row = self.typechart.get(attacking)
        if row is None:
            return 1.0
        mult = 1.0
        for t in defending_types:
            mult *= row.get(t, 1.0)
        return mult

    def mega_target(self, species_id: str, item_id: str | None) -> str | None:
        if not item_id:
            return None
        return self.mega_by_species.get((species_id, item_id))


def to_id(name: str) -> str:
    """Showdown's ID normalisation: lowercase, strip everything but [a-z0-9]."""
    return "".join(c for c in name.lower() if c.isalnum())


@lru_cache(maxsize=8)
def load_regulation(format_id: str, path: str | None = None) -> Regulation:
    p = Path(path) if path else regulation_dir() / f"{format_id}.json"
    if not p.exists():
        available = sorted(q.stem for q in regulation_dir().glob("*.json"))
        raise FileNotFoundError(
            f"No regulation config at {p}. Available: {available}. "
            "Generate with: node packages/sim-bridge/dist/cli/dump-regulation.js"
        )
    with p.open(encoding="utf-8") as fh:
        return Regulation(json.load(fh))
