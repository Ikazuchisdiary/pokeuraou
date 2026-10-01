"""A pasted team against six species: the ranked-match entry (IKA-407, stage 1a of IKA-404).

A ranked match shows only the opponent's species. The solver wants both full sheets, so the
opponent's sets are *estimated* here and every estimate says where it came from:

* **The set** is the tournament field's most common one for the species: ability, item,
  nature and four moves as one set (what a team sheet shows). The field is one event, the
  Baltimore Regional (`data/standings/2027-baltimore.json.gz`, every entry that validates in
  M-C), and nothing else: the screen and the record say so. When the commonest whole set is
  rare (fewer than `WHOLE_MIN` members hold it) the parts are taken one by one -- the commonest
  nature, item, ability and the four most played moves -- and it says "part by part".
* **The spread** (SP) is not in the standings (a team sheet blanks it). It is the commonest
  spread of the same species in the pool of pasted teams (`regmc-matchupweb`), the same
  nature first; with none, a neutral one made from the base stats under the regulation's
  own total and per-stat cap. That one is labelled provisional ("仮の配分").
* **Any field can be overwritten**, and the three commonest whole sets are offered as one-tap
  alternatives. Nothing here mixes sets (a mixture is stage 2 of IKA-404).

An id the value function never trained on (`LearnedIds`) is listed on the set: the leaf reads
its row as noise, and the screen says so rather than hiding it. A species the field never
shows is refused (`UnknownSpecies`), not guessed.

Why this is allowed to make a set up at all, when `teams.py` rules a made-up spread out: the
ranked match gives nothing else, the user decided it (10/1), and the made-up part is
labelled wherever it appears, down to the solve's own `notes`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .pool import Pool, load_pool
from .priors import SampledSet
from .regulation import STAT_IDS, Regulation, repo_root, to_id
from .selection_book import BookEntry
from .standings import Standings, TeamMember, load_standings
from .teams import Roster, TeamError, roster_from_data

#: The one event the opponent's sets come from.
EVENT_FILE = "2027-baltimore.json.gz"
#: The pool the spreads come from (the only M-C data that has them).
POOL_NAME = "regmc-matchupweb"
#: A whole set is used when at least this many members of the species hold it.
WHOLE_MIN = 3
#: A species' sets are read from the field teams sharing the most of the opponent's six, down
#: to the first tier that holds at least this many members of the species.
TIER_MIN = 10
#: How many whole sets are offered as one-tap alternatives.
ALTERNATIVES = 3
#: A species this fast is given its second spread slot as Speed, a slower one as HP.
FAST_BASE_SPEED = 60
LEARNED_CACHE = "ranked/learned-ids-regmc.json"
#: The encoded training sets the leaf learned from: the M-C games, and the M-B games its
#: warm start came from (the latter by names, through M-B's own vocabulary).
LEARNED_SOURCES = (
    ("selfplay-mc0123-encoded.npz", "gen9championsvgc2026regmc"),
    ("selfplay-gen11L-encoded.npz", "gen9championsvgc2026regmb"),
)


class RankedError(ValueError):
    """What the person can fix: the message is shown on the screen as it is."""


class UnknownSpecies(RankedError):
    """The field shows no member of this species, so there is nothing to estimate from."""


def data_dir() -> Path:
    return repo_root() / "data"


# ----------------------------------------------------------------------------- the paste


def _paste_tool() -> Any:  # noqa: ANN401
    """`tools/fetch_pastes.py` as a module: its reading of a Showdown export is the one the
    pool was built with, and the control compares against the pool."""
    name = "pokeuraou_tool_fetch_pastes"
    found = sys.modules.get(name)
    if found is not None:
        return found
    path = repo_root() / "tools" / "fetch_pastes.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass(slots=True)
class PasteResult:
    """A pasted team: the roster when it is complete and legal, else what is wrong."""

    roster: Roster | None
    #: ``{member, writtenAs, field, reason}`` as `fetch_pastes.build_team` reports them.
    problems: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


def roster_from_paste(
    reg: Regulation, text: str, *, team_id: str = "mine", name: str = "あなた"
) -> PasteResult:
    """The pasted Showdown export as a roster, by the pool importer's reading.

    A problem anywhere (a missing nature, an illegal move, five Pokemon) gives no roster, as the
    importer excludes a team whole. A paste with no SP line is read as 0 SP and says so: the
    solver reads real stats, so the person should know the spread was not given.
    """
    tool = _paste_tool()
    members, problems = tool.build_team(reg, text)
    warnings: list[str] = []
    if members and all(not m["sp"] for m in members):
        warnings.append("貼り付けに配分（EVs の行）が無いので、配分は全部 0 として読みます")
    elif any(not m["sp"] for m in members):
        names = ", ".join(m["writtenAs"] for m in members if not m["sp"])
        warnings.append(f"配分が無いので 0 として読みます: {names}")
    if problems:
        return PasteResult(None, problems, warnings)
    try:
        roster = roster_from_data(
            {"regulation": reg.meta.format_id, "id": team_id, "name": name, "team": members,
             "source": {"kind": "paste"}},
            default_id=team_id, reg=reg,
        )
    except TeamError as exc:
        return PasteResult(
            None, [{"member": None, "writtenAs": None, "field": "team", "reason": str(exc)}],
            warnings,
        )
    return PasteResult(roster, [], warnings)


# ----------------------------------------------------------------------------- learned ids


@dataclass(frozen=True, slots=True)
class LearnedIds:
    """The ids the value function saw in training (by name, for each kind)."""

    species: frozenset[str]
    moves: frozenset[str]
    items: frozenset[str]
    abilities: frozenset[str]
    sources: tuple[str, ...] = ()

    def missing(self, one: SampledSet) -> list[tuple[str, str]]:
        """(kind, id) of what in this set the leaf never trained on."""
        out: list[tuple[str, str]] = []
        if one.species not in self.species:
            out.append(("species", one.species))
        out += [("move", m) for m in one.moves if m not in self.moves]
        if one.item and one.item not in self.items:
            out.append(("item", one.item))
        if one.ability and one.ability not in self.abilities:
            out.append(("ability", one.ability))
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "species": sorted(self.species), "moves": sorted(self.moves),
            "items": sorted(self.items), "abilities": sorted(self.abilities),
            "sources": list(self.sources),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LearnedIds:
        return cls(
            species=frozenset(data["species"]), moves=frozenset(data["moves"]),
            items=frozenset(data["items"]), abilities=frozenset(data["abilities"]),
            sources=tuple(data.get("sources", ())),
        )


def _ids_in_npz(path: Path, format_id: str) -> dict[str, set[str]]:
    """The names behind the vocabulary indices that occur in one encoded training set.

    One array at a time (the largest, the moves, is ~2 GB), indices through the vocabulary the
    set was encoded with (its meta says which); 0 is "absent" and names nothing.
    """
    from .encode import build_vocabulary
    from .regulation import load_regulation

    vocab = build_vocabulary(load_regulation(format_id))
    tables = {"species": vocab.species, "ability": vocab.abilities,
              "item": vocab.items, "moves": vocab.moves}
    out: dict[str, set[str]] = {k: set() for k in tables}
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta_json"]))
        if meta.get("format_id") != format_id:
            raise RankedError(f"{path.name} was encoded for {meta.get('format_id')}, not {format_id}")
        for key, table in tables.items():
            by_index = {index: name for name, index in table.items()}
            for index in np.unique(z[key]):
                if int(index) > 0 and int(index) in by_index:
                    out[key].add(by_index[int(index)])
    return out


def build_learned_ids(root: Path | None = None) -> LearnedIds:
    """The union of the ids in `LEARNED_SOURCES` (reads the big encoded files once)."""
    root = root or data_dir()
    union: dict[str, set[str]] = {"species": set(), "ability": set(), "item": set(), "moves": set()}
    used: list[str] = []
    for filename, format_id in LEARNED_SOURCES:
        found = _ids_in_npz(root / filename, format_id)
        for key, names in found.items():
            union[key] |= names
        used.append(f"{filename} ({format_id}): " + ", ".join(f"{k} {len(v)}" for k, v in found.items()))
    return LearnedIds(
        species=frozenset(union["species"]), moves=frozenset(union["moves"]),
        items=frozenset(union["item"]), abilities=frozenset(union["ability"]),
        sources=tuple(used),
    )


def load_learned_ids(root: Path | None = None, *, source_root: Path | None = None) -> LearnedIds:
    """The cache at `LEARNED_CACHE`; built from the encoded sets (in ``source_root``, default
    ``root``) and written when absent."""
    root = root or data_dir()
    cache = root / LEARNED_CACHE
    if cache.exists():
        return LearnedIds.from_json(json.loads(cache.read_text(encoding="utf-8")))
    learned = build_learned_ids(source_root or root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(learned.to_json(), ensure_ascii=False, indent=1) + "\n"
    cache.write_bytes(text.encode("utf-8"))
    return learned


# ----------------------------------------------------------------------------- the estimate


@dataclass(frozen=True, slots=True)
class Candidate:
    """One whole set the field played, and how many members held it."""

    ability: str | None
    item: str | None
    nature: str
    moves: tuple[str, ...]
    count: int


@dataclass(frozen=True, slots=True)
class Observation:
    """What has been seen of one opponent member in play: moves it used, the item it showed
    (or consumed), the ability that activated. `SetBelief.observe` drops the sets that
    cannot have produced it (stage 1b, IKA-408, feeds these in)."""

    moves: frozenset[str] = frozenset()
    item: str | None = None
    ability: str | None = None

    def merged(self, other: Observation) -> Observation:
        return Observation(self.moves | other.moves, other.item or self.item, other.ability or self.ability)

    def consistent(self, member: TeamMember) -> bool:
        if not self.moves <= set(member.moves):
            return False
        if self.item is not None and member.item != self.item:
            return False
        # A field record whose ability is the mega forme's is unknown here: it cannot rule out.
        return self.ability is None or member.ability in (None, self.ability)


@dataclass(frozen=True, slots=True)
class SetBelief:
    """The field's sets for one species, read for one opposing six.

    ``members`` are the field members of the species that stand behind the estimate: those of
    the teams sharing the most of the six with the opponent's, down to the first tier with
    enough of them (`TIER_MIN`). ``overlap`` is that tier's number of shared species (1: every
    team with the species, the composition not used), ``tiers`` how many field teams share
    exactly k of the six, and ``seen`` what has been observed. The candidates are the whole
    sets in ``members`` with their counts; `observe` returns the belief with the members an
    observation rules out removed, so a read of the set is never made from what was seen
    not to be."""

    species: str
    members: tuple[TeamMember, ...]
    overlap: int
    tiers: tuple[tuple[int, int], ...]
    seen: Observation = Observation()

    def observe(self, obs: Observation) -> SetBelief:
        merged = self.seen.merged(obs)
        return replace(self, members=tuple(m for m in self.members if obs.consistent(m)), seen=merged)

    def candidates(self) -> list[Candidate]:
        """Every whole set among the members, commonest first (ties by name, so it is fixed)."""
        counts = Counter((m.ability, m.item, m.nature, tuple(sorted(m.moves))) for m in self.members)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return [Candidate(a, i, n, mv, c) for (a, i, n, mv), c in ranked]

    def teams_in_tier(self) -> int:
        return sum(n for k, n in self.tiers if k >= self.overlap)


@dataclass(slots=True)
class OpponentSet:
    """One opponent member as estimated, with where each part came from."""

    set: SampledSet
    #: "whole" | "part" | "candidate" | "override" | "manual"
    kind: str
    #: Members behind the estimate (the belief's tier), and how many hold the chosen whole set.
    species_n: int
    whole_n: int
    alternatives: tuple[Candidate, ...]
    #: "pool-nature" | "pool-species" | "neutral" | "override"
    sp_source: str
    #: How many pool sets the spread was the commonest of (0 for neutral and overrides).
    sp_n: int
    notes: list[str] = field(default_factory=list)
    overridden: tuple[str, ...] = ()
    #: What the estimate was read from: `FieldPrior.estimate` of it again, after an observation.
    belief: SetBelief | None = None

    @property
    def sp_provisional(self) -> bool:
        return self.sp_source == "neutral"


def neutral_spread(reg: Regulation, species_id: str) -> dict[str, int]:
    """A spread from the base stats alone, inside the regulation's total and per-stat cap.

    The stronger attacking stat gets the cap, Speed the next cap when the species is fast
    enough to want it (else HP), and what is left goes to HP (Defense when HP is taken).
    Made up and labelled so; not a claim about what anyone plays.
    """
    base = dict(zip(STAT_IDS, reg.species[species_id].base_stats, strict=True))
    cap, total = reg.meta.sp_per_stat_max, reg.meta.sp_limit
    out = dict.fromkeys(STAT_IDS, 0)
    attack = "atk" if base["atk"] >= base["spa"] else "spa"
    out[attack] = min(cap, total)
    left = total - out[attack]
    second = "spe" if base["spe"] >= FAST_BASE_SPEED else "hp"
    out[second] = min(cap, left)
    left -= out[second]
    rest = "hp" if second != "hp" else "def"
    out[rest] = min(cap, left)
    return out


def check_spread(reg: Regulation, sp: dict[str, int]) -> dict[str, int]:
    """The spread as ints in stat order, or a `RankedError` naming the rule it breaks."""
    out = {}
    for stat in STAT_IDS:
        value = sp.get(stat, 0)
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise RankedError(f"配分 {stat} は 0 以上の整数です")
        out[stat] = int(value)
    if sum(out.values()) > reg.meta.sp_limit:
        raise RankedError(f"配分の合計 {sum(out.values())} が上限 {reg.meta.sp_limit} を超えています")
    over = [s for s, v in out.items() if v > reg.meta.sp_per_stat_max]
    if over:
        raise RankedError(f"配分 {', '.join(over)} が 1 つの上限 {reg.meta.sp_per_stat_max} を超えています")
    return out


class FieldPrior:
    """What one tournament field and the pool of pasted teams say about each species."""

    def __init__(self, reg: Regulation, standings: Standings, pool: Pool | None,
                 *, event_file: str = EVENT_FILE) -> None:
        self.reg = reg
        self.standings = standings
        self.event_file = event_file
        self.members: dict[str, list[TeamMember]] = {}
        #: per species, (index of the field team, member): the composition read needs the team.
        self.entries: dict[str, list[tuple[int, TeamMember]]] = {}
        self.team_species: list[frozenset[str]] = []
        for index, team in enumerate(standings.teams):
            self.team_species.append(frozenset(team.species))
            for member in team.members:
                self.members.setdefault(member.species, []).append(member)
                self.entries.setdefault(member.species, []).append((index, member))
        self.spreads: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
        self.pool_id = pool.id if pool is not None else None
        self.pool_teams = len(pool.teams) if pool is not None else 0
        for roster in (pool.teams if pool is not None else ()):
            for one in roster.sets:
                self.spreads.setdefault(one.species, []).append(
                    (one.nature, tuple(int(one.sp.get(s, 0)) for s in STAT_IDS))
                )

    @classmethod
    def load(cls, reg: Regulation, root: Path | None = None) -> FieldPrior:
        root = root or data_dir()
        standings = load_standings(root / "standings" / EVENT_FILE, reg)
        try:
            pool = load_pool(root / "pool" / f"{POOL_NAME}.json", reg)
        except FileNotFoundError:
            pool = None
        return cls(reg, standings, pool)

    # -- what the field shows
    def species_ids(self) -> list[str]:
        return sorted(self.members)

    def source_line(self) -> str:
        s = self.standings
        return (f"{s.event}（{s.event_format}）の {len(s.teams):,} チームの実際の型から、"
                "相手の編成に近い構築ほど重く見て推定しています。ほかの大会は使っていません。")

    def short_line(self) -> str:
        """The one sentence on the estimate's band."""
        name = self.standings.event.split(" ")[0]
        return f"{name} の大会 {len(self.standings.teams):,} チームの型から推定しています"

    def spread_line(self) -> str:
        return ("配分（SP）は大会の公開シートに載らないので、"
                + (f"貼り付けで集めた {self.pool_teams} 構築（Match Up Web）にある同じ種族の配分を使い、"
                   if self.pool_teams else "")
                + "それも無い種族は「仮の配分」にします。")

    def belief(self, species_id: str, composition: Sequence[str] | None = None,
               *, tier_min: int = TIER_MIN) -> SetBelief:
        """The field's sets for the species, read for the opposing six ``composition``.

        Every field team holding the species shares some of the six with it; the sets come
        from the teams sharing the most, down to the first tier (6 shared, at least 5, ...)
        that holds ``tier_min`` members of the species, and from all of them when none does.
        Without a composition every team shares just the species: the species' marginal.
        """
        entries = self.entries.get(species_id)
        if not entries:
            raise UnknownSpecies(
                f"{self.reg.species[species_id].name if species_id in self.reg.species else species_id}"
                f" は {self.standings.event} に出ていないので、型を推定できません。型を手で入れてください"
            )
        six = frozenset(composition or ()) | {species_id}
        shared = [len(self.team_species[index] & six) for index, _m in entries]
        exact = Counter(shared)
        overlap, held = 1, 0
        for k in range(min(len(six), max(exact)), 0, -1):
            held += exact.get(k, 0)
            if held >= tier_min:
                overlap = k
                break
        members = tuple(m for (_i, m), k in zip(entries, shared, strict=True) if k >= overlap)
        return SetBelief(species_id, members, overlap, tuple(sorted(exact.items(), reverse=True)))

    def candidates(self, species_id: str, composition: Sequence[str] | None = None) -> list[Candidate]:
        return self.belief(species_id, composition).candidates()

    def _ability_mode(self, species_id: str) -> str | None:
        known = Counter(m.ability for m in self.members.get(species_id, ()) if m.ability)
        return sorted(known.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if known else None

    def _build(self, species_id: str, c: Candidate, kind: str, *, belief: SetBelief, whole_n: int,
               alternatives: tuple[Candidate, ...], notes: list[str]) -> OpponentSet:
        ability = c.ability
        if ability is None:
            # The field's record is the mega forme's ability, which a sheet does not show.
            ability = self._ability_mode(species_id) or self.reg.species[species_id].abilities[0]
            notes.append(f"特性は大会データに無い（メガ後の特性の記録）ので {to_id(ability)} にしました")
        sp, source, n = self._spread(species_id, c.nature)
        built = SampledSet(
            species=species_id, ability=to_id(ability), item=c.item, nature=c.nature, sp=sp,
            moves=list(c.moves),
        )
        return OpponentSet(built, kind, len(belief.members), whole_n, alternatives, source, n, notes,
                           belief=belief)

    def _spread(self, species_id: str, nature: str) -> tuple[dict[str, int], str, int]:
        have = self.spreads.get(species_id, [])
        for label, pick in (("pool-nature", [s for nat, s in have if nat == nature]),
                            ("pool-species", [s for _nat, s in have])):
            if pick:
                counts = Counter(pick)
                top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                return dict(zip(STAT_IDS, top, strict=True)), label, len(pick)
        return neutral_spread(self.reg, species_id), "neutral", 0

    def estimate(self, belief: SetBelief) -> OpponentSet:
        """The estimate from a belief: the commonest whole set, or part by part when rare.

        After `SetBelief.observe` the same call gives the read of what is left; a belief that
        an observation emptied has no read (`RankedError`)."""
        species_id = belief.species
        if not belief.members:
            raise RankedError("見たことと合う型が大会データにありません")
        ranked = belief.candidates()
        alternatives = tuple(ranked[:ALTERNATIVES])
        top = ranked[0]
        if top.count >= WHOLE_MIN:
            return self._build(species_id, top, "whole", belief=belief, whole_n=top.count,
                               alternatives=alternatives, notes=[])
        found = belief.members
        nature = Counter(m.nature for m in found)
        item = Counter(m.item for m in found)
        ability = Counter(m.ability for m in found if m.ability)
        moves = Counter(mv for m in found for mv in m.moves)

        def mode(counter: Counter[Any]) -> Any:  # noqa: ANN401
            return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]

        picked = sorted(moves.items(), key=lambda kv: (-kv[1], kv[0]))[: self.reg.meta.max_move_count]
        mixed = Candidate(
            mode(ability) if ability else None, mode(item), mode(nature),
            tuple(sorted(m for m, _n in picked)), 0,
        )
        return self._build(
            species_id, mixed, "part", belief=belief, whole_n=0, alternatives=alternatives,
            notes=[f"最頻の型でも {top.count} 体なので、部分ごとの最頻を組み合わせました"],
        )

    def fill(self, species_id: str, composition: Sequence[str] | None = None,
             *, tier_min: int = TIER_MIN) -> OpponentSet:
        """The estimate for one species of the opposing six ``composition``."""
        return self.estimate(self.belief(species_id, composition, tier_min=tier_min))

    def choose(self, one: OpponentSet, index: int) -> OpponentSet:
        """The index-th alternative as the set (its spread re-chosen for its nature)."""
        if not 0 <= index < len(one.alternatives) or one.belief is None:
            raise RankedError(f"候補は {len(one.alternatives)} 個です")
        return self._build(
            one.set.species, one.alternatives[index], "candidate", belief=one.belief,
            whole_n=one.alternatives[index].count, alternatives=one.alternatives, notes=[],
        )

    def fill_team(self, species_ids: Sequence[str], composition: Sequence[str] | None = None,
                  *, tier_min: int = TIER_MIN) -> tuple[list[OpponentSet], list[str]]:
        """The species' estimates and what is wrong with them as a team (clauses; the count
        of six is the caller's, which may hold some back for a hand entry).

        A second holder of an item already taken moves to its first alternative with another
        item, so the six is one the Item Clause allows; when none has, the clash is reported.
        """
        reg = self.reg
        problems: list[str] = []
        out: list[OpponentSet] = []
        taken: set[str] = set()
        six = list(composition if composition is not None else species_ids)
        for species_id in species_ids:
            one = self.fill(species_id, six, tier_min=tier_min)
            item = one.set.item
            if item and item in taken and reg.meta.item_clause is not None:
                for index, alt in enumerate(one.alternatives):
                    if alt.item and alt.item not in taken or alt.item is None:
                        was = reg.items[item].name
                        one = self.choose(one, index)
                        one.notes.append(f"持ち物 {was} が重複するので、候補 {index + 1} の型にしました")
                        break
                else:
                    problems.append(f"持ち物 {reg.items[item].name} が重複しています（上書きしてください）")
            if one.set.item:
                taken.add(one.set.item)
            out.append(one)
        bases = [reg.species[o.set.species].base_species for o in out]
        doubled = sorted({b for b in bases if bases.count(b) > 1})
        if doubled:
            problems.append(f"同じ種族が 2 体います: {', '.join(doubled)}")
        return out, problems


def blank_set(reg: Regulation, species_id: str) -> OpponentSet:
    """A species the field does not show: its first ability, no item, neutral spread, no moves.

    The person has to enter the moves, so this is not solvable until they have."""
    found = reg.species[species_id]
    one = SampledSet(
        species=species_id, ability=to_id(found.abilities[0]), item=None, nature="Hardy",
        sp=neutral_spread(reg, species_id), moves=[],
    )
    return OpponentSet(one, "manual", 0, 0, (), "neutral", 0,
                       ["大会データに無い種族です。技・持ち物・性格・配分を入れてください"])


def apply_override(reg: Regulation, one: OpponentSet, patch: dict[str, Any]) -> OpponentSet:
    """A set with the person's fields replacing the estimate's. ``patch`` has any of
    ``ability`` ``item`` (``""``: none) ``nature`` ``moves`` ``sp``; what it names is
    validated as a roster member is, and listed in ``overridden``."""
    cur = one.set
    species = reg.species[cur.species]
    ability, item, nature = cur.ability, cur.item, cur.nature
    moves, sp = list(cur.moves), dict(cur.sp)
    changed = list(one.overridden)
    sp_source, sp_n = one.sp_source, one.sp_n

    def mark(name: str) -> None:
        if name not in changed:
            changed.append(name)

    if "ability" in patch:
        ability = to_id(patch["ability"])
        if ability not in {to_id(a) for a in species.abilities}:
            raise RankedError(f"{patch['ability']} は {species.name} の特性ではありません")
        mark("ability")
    if "item" in patch:
        item = to_id(patch["item"]) or None
        if item is not None and item not in reg.items:
            raise RankedError(f"持ち物 {patch['item']} はこの規則で使えません")
        mark("item")
    if "nature" in patch:
        if patch["nature"] not in reg.natures:
            raise RankedError(f"性格 {patch['nature']} を知りません")
        nature = str(patch["nature"])
        mark("nature")
    if "moves" in patch:
        moves = [to_id(m) for m in patch["moves"] if to_id(m)]
        if not 1 <= len(moves) <= reg.meta.max_move_count:
            raise RankedError(f"技は 1〜{reg.meta.max_move_count} 個です")
        if len(set(moves)) != len(moves):
            raise RankedError("同じ技が 2 回あります")
        bad = [m for m in moves if m not in reg.moves]
        if bad:
            raise RankedError(f"技 {', '.join(bad)} はこの規則で使えません")
        mark("moves")
    if "sp" in patch:
        sp = check_spread(reg, {**cur.sp, **patch["sp"]})
        sp_source, sp_n = "override", 0
        mark("sp")
    built = SampledSet(species=cur.species, ability=ability, item=item, nature=nature, sp=sp, moves=moves)
    return replace(one, set=built, kind="override" if one.kind != "manual" else "manual",
                   sp_source=sp_source, sp_n=sp_n, overridden=tuple(changed))


def opponent_roster(reg: Regulation, sets: Sequence[SampledSet], *, name: str = "相手（推定）") -> Roster:
    """The estimated six as the roster `humanplay.solve_entry` takes."""
    if len(sets) != reg.meta.team_size:
        raise RankedError(f"{len(sets)} 体です。{reg.meta.team_size} 体が要ります")
    for index, one in enumerate(sets):
        if not one.moves:
            raise RankedError(f"{index + 1} 体目（{reg.species[one.species].name}）の技がありません")
    return Roster(
        id="ranked-opponent", name=name, reg=reg, sets=[replace(s, moves=list(s.moves)) for s in sets],
        shown_stats=[None] * len(sets), source={"kind": "estimated", "event": EVENT_FILE},
    )


# ----------------------------------------------------------------------------- the selection


def solve_ranked(
    reg: Regulation, mine: Roster, opponent: Roster, evaluate: Callable[..., Any], model: str,
    **kwargs: Any,  # noqa: ANN401
) -> BookEntry:
    """The selection of the pasted team against the estimated six: `humanplay.solve_entry`,
    unchanged, with the pasted team as the row player."""
    from .humanplay import solve_entry

    return solve_entry(reg, (mine, opponent), evaluate, model, **kwargs)


def selection_summary(entry: BookEntry, reg: Regulation, mine: Sequence[SampledSet],
                      theirs: Sequence[SampledSet], loc: Any = None, *, top: int = 8) -> dict[str, Any]:  # noqa: ANN401
    """What the screen shows of a solved selection: both sides' selections by probability, how
    often each member is brought and led, and the value (our win probability)."""

    def name(one: SampledSet) -> str:
        return loc.species(one.species) if loc is not None else reg.species[one.species].name

    def side(six: Sequence[SampledSet], strategy: np.ndarray) -> dict[str, Any]:
        probs = np.asarray(strategy, dtype=np.float64)
        order = np.argsort(-probs, kind="stable")
        bring = [0.0] * len(six)
        lead = [0.0] * len(six)
        for sel, p in zip(entry.selections, probs, strict=True):
            for position, index in enumerate(sel):
                bring[index] += float(p)
                if position < 2:
                    lead[index] += float(p)
        return {
            "members": [{"species": name(s), "id": s.species, "bring": bring[i], "lead": lead[i]}
                        for i, s in enumerate(six)],
            "selections": [
                {"leads": [name(six[i]) for i in entry.selections[j][:2]],
                 "back": [name(six[i]) for i in entry.selections[j][2:]],
                 "indices": list(entry.selections[j]), "p": float(probs[j])}
                for j in order[:top] if probs[j] > 1e-9
            ],
        }

    return {
        "value": float(entry.value),
        "mine": side(mine, entry.our_strategy),
        "theirs": side(theirs, entry.their_strategies[0]),
        "seconds": float(entry.seconds),
        "model": entry.model,
        "notes": list(entry.notes),
    }
