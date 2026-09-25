"""Playing games to a win or a loss, and recording what to learn from them.

The target is the outcome, not a proxy. That was a deliberate choice: a value function
trained to predict a hand-made evaluation learns the evaluation's biases and can never be
better than it, whereas one trained on who actually won can honestly be labelled a win
probability -- which is the label the whole tool has been refusing to print until now.

Three things follow from it, and each is a constraint rather than a preference:

- **A game has to finish.** That needs the replacement phase, which is why it exists.
- **A game that does not finish is discarded, not labelled.** Filling in a proxy for a
  timed-out game would smuggle the rejected target back in through the back door. The
  count of discards is reported.
- **The search that picks the moves is not the thing being learned.** With no value
  function yet, the leaves are scored with the `hp-share` proxy, so this first generation
  plays weakly. That is fine and it is recorded on every game: the *labels* are real
  outcomes regardless of how well the players played. Later generations replace the leaf
  score with the learned function.

The positions recorded carry both spreads. Hidden information is handled *outside* the
value function: at analysis time the belief layer already averages the matrix over spread
particles, so the function only has to answer "who wins from here", and asking it to also
represent our uncertainty would duplicate machinery that exists and is tested.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from . import port, rank_scores, timing
from .actions import SideAction, switch_actions_after_faint
from .budget import Budget
from .equilibrium import EquilibriumError, solve
from .fold import TurnLeaves
from .hidden import (
    DEFAULT_BENCH_DROP,
    completions,
    drop_light,
    identity,
    parse_bench_drop,
    seen_identities,
    seen_slots,
    shown_species,
)
from .narrow import narrow
from .payoff import HP_SHARE, Objective
from .policy import policy_ranking
from .position import Field, MoveSlot, Pokemon, Position, Side
from .priors import Cooccurrence, MetagamePrior, SampledSet
from .provenance import (
    LEGACY_BENCH_DROP,
    LEGACY_DEEPEN,
    LEGACY_RANK_FILL,
    engine_fingerprint,
)
from .regulation import STAT_IDS, Regulation, repo_root
from .rustnode import PortPause, PortTurn
from .search import (
    DEFAULT_RANK_FILL,
    belief_solve,
    believed_ranking,
    leaf_ranking,
    parse_rank_fill,
    search,
)
from .selection_book import (
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    BenchPrior,
    SelectionBook,
)
from .standings import Standings, cluster_teams, label_for, sample_standings_team
from .stats import nature_multipliers, stats_from_sp
from .teams import (
    Archetype,
    Roster,
    pick_four_indices,
    sample_archetype,
    sample_metagame_team,
)

#: How many actions per side reach the exact solver during self-play. The cost of a solved
#: position is rows x columns, so this is the dial that decides whether self-play is
#: affordable -- see tools/selfplay_budget.py (deleted in IKA-212 with Python's resolver;
#: README records what it measured). It is smaller than the analysis default
#: because a training game needs many positions, not one deep one.
SEARCH_LIMIT = 8

class LeafEvaluator(Protocol):
    """Scores a batch of positions as side 0's win probability.

    A protocol rather than a concrete type so the resolver and the self-play loop never
    import torch: :class:`pokeuraou.value.BatchedValue` satisfies it, and so does a stub in
    a test. Positions arrive as a list because the batch is the unit -- see
    :func:`pokeuraou.search.search`.
    """

    def __call__(self, positions: list[Position]) -> np.ndarray: ...


#: A game longer than this is discarded rather than labelled. Champions games are best-of
#: with a turn limit in practice; here it only has to be long enough that a real result is
#: the normal outcome.
MAX_TURNS = 40


def selfplay_dir() -> Path:
    return repo_root() / "data" / "selfplay"


@dataclass(slots=True)
class Decision:
    """One position where both players chose, with the equilibrium they chose from."""

    turn: int
    kind: str  # 'move' | 'replacement'
    position: dict[str, Any]
    own_actions: list[str]
    own_policy: list[float]
    foe_actions: list[str]
    foe_policy: list[float]
    #: The search's own value at this position, in the units of the objective used. Kept
    #: for diagnostics -- it is *not* the training target.
    search_value: float
    #: What each side actually played, as the choice string. The mixture above is what the
    #: search computed and stays the policy target; this is the draw from it, kept so a
    #: recorded game reads back as a game.
    own_chosen: str | None = None
    foe_chosen: str | None = None
    #: Which completion of the opponent's unseen slots each side's menu was ranked from,
    #: as ``[index, species]`` in `completions` order -- ``[0, []]`` when nothing of the
    #: opponent's was hidden -- or None for a side that never read a completion (the
    #: damage ranking, which never reads the bench). None altogether in the open game or
    #: when neither side did. IKA-143: the rule is `rank_view`, and this is the rule's
    #: answer at this decision.
    rank_views: list[list[Any] | None] | None = None
    #: What each side's opponent was allowed to know of it here, indexed by the side that
    #: owns the Pokemon: ``shown[i]`` is side ``i``'s identities (`hidden.identity`, the
    #: base species id -- never a party slot, which `_do_switch` renumbers; IKA-117),
    #: sorted, that side ``1 - i``'s search conditioned on. Under a hidden bench it is the
    #: carried `seen_identities` the belief was built from; in the open game it is the
    #: whole four, because the open search is handed the whole four. IKA-127: before it
    #: a reader had to replay `seen_identities` over the recorded positions
    #: (`replay_shown`) to learn what the belief was conditioned on.
    shown: list[list[str]] | None = None
    #: Side 1's own value of this decision, in side 0's units (side 0's win probability,
    #: the orientation of `search_value`) -- the equilibrium of the game side 1 solved
    #: over *its* belief about side 0's bench. Under a hidden bench the two sides solve
    #: different games, and `search_value` is side 0's alone. Written only where side 1
    #: solved a game of its own under a hidden bench (move and replacement nodes); None
    #: in the open game, where one agent's both sides solve the same matrix and this would
    #: repeat `search_value`, and at a self-switch, where only one side chooses.
    foe_search_value: float | None = None
    #: What each side's best-first deepening did here (`deepen.Deepened.to_json`: budget,
    #: cells spent, cells refined, depth reached), or None for a side that did not deepen
    #: -- the setting off, or a node with a bench still hidden (IKA-33). None altogether
    #: when neither side did, so a game played without it is written as before.
    deepened: list[dict[str, int] | None] | None = None


def _shown_record(
    pos: Position, hidden_seen: Sequence[Collection[str]] | None
) -> list[list[str]]:
    """`Decision.shown` at `pos`: the carried identities, or the whole four when open."""
    if hidden_seen is None:
        return [sorted(identity(mon) for mon in pos.sides[i].pokemon) for i in (0, 1)]
    return [sorted(hidden_seen[i]) for i in (0, 1)]


def replay_shown(record: dict[str, Any]) -> list[list[list[str]]]:
    """Each recorded decision's `shownIdentities`, rebuilt from the positions alone.

    What a reader of a record written before IKA-127 has to do, and the check on the
    field for one written after it: `seen_identities` carried over the move and
    replacement positions in order, as `play_game` carries it. A self-switch reads its
    own pause on top of the carried set without adding to it, because `play_game`'s
    carry never sees a pause. The open game is the whole four at every decision.
    """
    open_game = record.get("information", "open") != "hidden-bench"
    carried: list[frozenset[str]] = [frozenset(), frozenset()]
    out: list[list[list[str]]] = []
    for decision in record["decisions"]:
        pos = Position.from_json(decision["position"])
        if open_game:
            out.append(_shown_record(pos, None))
            continue
        here = [seen_identities(pos, i, carried[i]) for i in (0, 1)]
        if decision["kind"] != "selfswitch":
            carried = here
        out.append(_shown_record(pos, here))
    return out


#: Why a game stopped (IKA-87). "wipeout": one side has nothing left, or both do and
#: the last to empty lost (`outcome` is set). "draw": both emptied and the order could
#: not be told (`outcome` None). "turn-cap": `max_turns` ran out first (`outcome` None).
#: "unresolved": a turn had no outcome to continue from -- no branch with weight, or more
#: than five mid-turn replacements (`outcome` None; the final position is the board the
#: turn was chosen at). There is no resignation in self-play, so no reason for one.
END_REASONS = ("wipeout", "draw", "turn-cap", "unresolved")


def _close_record(record: GameRecord, pos: Position) -> None:
    """Writes the final position and the reason the game stopped, as `play_game` ends.

    `record.end_reason` already says "unresolved" when the loop broke on a turn with
    nothing to continue from; otherwise the board says which of the other three it was.
    """
    if record.end_reason is None:
        if pos.ended:
            record.end_reason = "wipeout" if pos.winner is not None else "draw"
        else:
            record.end_reason = "turn-cap"
    record.final_position = pos.to_json()


def _final_json(record: GameRecord) -> dict[str, Any]:
    """The record's `finalPosition` and `endReason`, or nothing when it has neither."""
    if record.final_position is None:
        return {}
    return {"finalPosition": record.final_position, "endReason": record.end_reason}


def final_position(record: dict[str, Any]) -> Position | None:
    """A written record's final position, or None for one from before IKA-87."""
    final = record.get("finalPosition")
    return None if final is None else Position.from_json(final)


@dataclass(slots=True)
class GameRecord:
    own_team: list[dict[str, Any]]
    foe_team: list[dict[str, Any]]
    foe_archetype: str
    decisions: list[Decision] = field(default_factory=list)
    #: 1.0 if side 0 won, 0.0 if it lost. None when the game hit the turn cap, in which
    #: case the record is discarded rather than labelled.
    outcome: float | None = None
    turns: int = 0
    #: Effects the resolver reported while the game was played. A game whose result rests
    #: on an unmodelled effect is still a real Showdown-checked result, but the reader
    #: should be able to see which ones were involved.
    unmodelled: list[str] = field(default_factory=list)
    #: The selection, when the caller knows it: each side's six species and the ordered
    #: party indices it brought. Ordered because the first two lead, which is a different
    #: decision from the other two -- 90 selections a side, not 15.
    own_six: list[str] = field(default_factory=list)
    foe_six: list[str] = field(default_factory=list)
    own_pick: list[int] = field(default_factory=list)
    foe_pick: list[int] = field(default_factory=list)
    #: Where the selection came from: "uniform" for an unweighted draw, "book" for a draw
    #: from the cached 6->4 equilibrium, "forced-lead" when the leads were chosen and only
    #: the two behind them were drawn. Recorded per game because a dataset will contain
    #: more than one and the distributions are not the same -- and because a pool built by
    #: forcing a lead describes itself in every summary a later decision about including it
    #: would read.
    selection_source: str = "uniform"
    #: What the search was allowed to see: "open" means it was handed the opponent's whole
    #: four, "hidden-bench" that it solved over the fours the sheet still allowed. Every
    #: game recorded before this field existed was open, which is what the default says.
    #: A training set that mixes the two without knowing is mixing two conditionings, and
    #: the value of a position is always conditional on the play that produced it.
    information: str = "open"
    #: Which ranking chose the candidates: "damage", "leaf" or "policy". Recorded because
    #: it is not a detail of how a game was played but a property of the agent that played
    #: it -- measured over 20 recorded positions at width 24, the damage ordering kept 58%
    #: of the full game's equilibrium mass against the leaf ordering's 93%, and gave up 2.5
    #: points on average against a best reply where the leaf ordering gave up 0.05.
    #:
    #: It was not recorded anywhere before, so which ranking produced generations 9 and 10
    #: cannot be recovered from their files or their logs.
    ranking: str = "damage"
    #: Which completion each side's leaf or policy ranking read under a hidden bench
    #: (`RANK_VIEWS`): "heaviest" since IKA-143, "first" before it. Written only for a
    #: hidden-bench game, so a record without it is either open or older than the rule.
    rank_view: list[str] = field(default_factory=lambda: ["heaviest", "heaviest"])
    #: How each side's leaf ranking filled its cells (`search.parse_rank_fill`, IKA-268).
    #: Written only when a side did not play `LEGACY_RANK_FILL`, so a record without it
    #: ranked the way every game before IKA-268 did.
    rank_fill: list[str] = field(default_factory=lambda: [LEGACY_RANK_FILL, LEGACY_RANK_FILL])
    #: Which completions each side's belief dropped (`hidden.parse_bench_drop`, IKA-283).
    #: Written only when a side did not play `LEGACY_BENCH_DROP`, so a record without it
    #: believed every completion, as every game before IKA-283 did.
    bench_drop: list[str] = field(
        default_factory=lambda: [LEGACY_BENCH_DROP, LEGACY_BENCH_DROP]
    )
    #: Each side's best-first deepening budget in cells (`deepen.best_first`, IKA-33).
    #: Written only when a side deepened, so a record without it searched at depth 1.
    deepen: list[int] = field(default_factory=lambda: [LEGACY_DEEPEN, LEGACY_DEEPEN])
    #: The equilibrium mixtures over the 90 ordered selections, when a book was used.
    #: These are the policy targets a selection head would learn -- the *solver's*
    #: recommendation, not the softened distribution the game was drawn from.
    own_selection_policy: list[float] = field(default_factory=list)
    foe_selection_policy: list[float] = field(default_factory=list)
    #: The distributions actually drawn from: equilibrium mixed with exploration. Kept
    #: separately so nobody reads a generated win rate as the equilibrium value.
    own_selection_mixture: list[float] = field(default_factory=list)
    foe_selection_mixture: list[float] = field(default_factory=list)
    #: The equilibrium value of the selection game, in win probability.
    selection_value: float | None = None
    #: Wall clock each agent spent deciding at the MOVE nodes, in seconds, indexed by
    #: side. Its own solve, plus its share of the candidate menus: a construction that
    #: served both agents is split in half, because charging it to whichever side the
    #: code happened to run first is how a cost comparison between two settings gets its
    #: answer from the order of the statements.
    #:
    #: Move nodes only. The replacement node is the same work for both agents -- no
    #: depth, no width, one matrix each at most -- so putting it here would dilute the
    #: ratio the number exists to report. What a whole game costs is the caller's
    #: `s/game`, which has always been there.
    #:
    #: Recorded per side rather than per game because the question depth-2 exists to
    #: answer -- is this worth its wall clock -- cannot be asked of a number that holds
    #: both arms. Two runs of different machine load are not comparable, and that is how
    #: the previous depth-2 cost was quoted.
    search_seconds: list[float] = field(default_factory=lambda: [0.0, 0.0])
    #: The pair each side led turn 1 with, as identities in `active` order -- what the
    #: bench belief conditions on (IKA-118) -- or None for a side whose turn 1 this game
    #: never saw (a game resumed from `start`). Per game rather than per decision because
    #: it cannot change within one (IKA-127). None altogether on a record not made by
    #: `play_game`.
    leads: list[list[str] | None] | None = None
    #: The position the game stopped at, as `Position.to_json` (IKA-87): the true board
    #: after the last turn was resolved -- who is left and on what HP. The decisions stop
    #: before the last turn, so without this nothing in the record says who survived.
    #: None on a record not made by `play_game`, and absent from every record before it.
    final_position: dict[str, Any] | None = None
    #: Why it stopped (`END_REASONS`), set with `final_position`.
    end_reason: str | None = None

    def to_json(self, *, objective: str, search_limit: int | tuple[int, int]) -> dict[str, Any]:
        return {
            "ownTeam": self.own_team,
            "foeTeam": self.foe_team,
            "foeArchetype": self.foe_archetype,
            "outcome": self.outcome,
            "turns": self.turns,
            "searchSeconds": list(self.search_seconds),
            "foeSpecies": [m["species"] for m in self.foe_team],
            "searchObjective": objective,
            "searchLimit": list(search_limit)
            if isinstance(search_limit, tuple)
            else search_limit,
            "targetIsRealOutcome": True,
            # Which engine produced this, so a fix found later can be dated against the
            # data rather than remembered. Three generations were produced with Hyper Beam
            # costing nothing and carry no field that says so.
            "engine": engine_fingerprint(),
            "ownSix": self.own_six,
            "foeSix": self.foe_six,
            "ownPick": self.own_pick,
            "foePick": self.foe_pick,
            "selectionSource": self.selection_source,
            "information": self.information,
            "ranking": self.ranking,
            **({"rankView": list(self.rank_view)} if self.information == "hidden-bench" else {}),
            **(
                {"rankFill": list(self.rank_fill)}
                if set(self.rank_fill) != {LEGACY_RANK_FILL}
                else {}
            ),
            **(
                {"benchDrop": list(self.bench_drop)}
                if set(self.bench_drop) != {LEGACY_BENCH_DROP}
                else {}
            ),
            **(
                {"deepen": list(self.deepen)}
                if set(self.deepen) != {LEGACY_DEEPEN}
                else {}
            ),
            "ownSelectionPolicy": self.own_selection_policy,
            "foeSelectionPolicy": self.foe_selection_policy,
            "ownSelectionMixture": self.own_selection_mixture,
            "foeSelectionMixture": self.foe_selection_mixture,
            "selectionValue": self.selection_value,
            "unmodelled": sorted(set(self.unmodelled)),
            **({"leads": self.leads} if self.leads is not None else {}),
            **_final_json(self),
            "decisions": [
                {
                    "turn": d.turn,
                    "kind": d.kind,
                    "position": d.position,
                    "ownActions": d.own_actions,
                    "ownPolicy": d.own_policy,
                    "foeActions": d.foe_actions,
                    "foePolicy": d.foe_policy,
                    "searchValue": d.search_value,
                    "ownChosen": d.own_chosen,
                    "foeChosen": d.foe_chosen,
                    **({"rankViews": d.rank_views} if d.rank_views is not None else {}),
                    **({"shownIdentities": d.shown} if d.shown is not None else {}),
                    **(
                        {"foeSearchValue": d.foe_search_value}
                        if d.foe_search_value is not None
                        else {}
                    ),
                    **({"deepened": d.deepened} if d.deepened is not None else {}),
                }
                for d in self.decisions
            ],
        }


def _make_pokemon(reg: Regulation, index: int, entry: SampledSet, active: int | None) -> Pokemon:
    species = reg.species[entry.species]
    spread = np.array([[entry.sp.get(stat, 0) for stat in STAT_IDS]], dtype=np.int64)
    stats = stats_from_sp(
        reg,
        np.array(species.base_stats, dtype=np.int64),
        spread,
        nature_multipliers(reg, [entry.nature]),
        level=reg.meta.level,
    )
    maxhp = int(stats[0, 0])
    return Pokemon(
        slot=index,
        species=entry.species,
        base_species=species.base_species,
        types=species.types,
        ability=entry.ability,
        nature=entry.nature,
        moves=[
            MoveSlot(id=m, pp=reg.moves[m].start_pp, maxpp=reg.moves[m].start_pp)
            for m in entry.moves
        ],
        hp=maxhp,
        maxhp=maxhp,
        item=entry.item,
        base_item=entry.item,
        sp={stat: entry.sp.get(stat, 0) for stat in STAT_IDS},
        active_index=active,
    )


def position_from_sets(
    reg: Regulation,
    own: list[SampledSet],
    foe: list[SampledSet],
    *,
    rng: np.random.Generator | None = None,
) -> Position:
    """A turn-1 position from two picked teams, both spreads known.

    Self-play knows both sides exactly because it generated them. That is not cheating:
    the value function's job is to answer "who wins from this position", and our
    uncertainty about the opponent's spread is integrated by the belief layer at query
    time, over positions of exactly this shape.

    What *was* wrong is that this returned the raw position: Showdown runs every lead's
    switch-in ability before it prints `|turn|1`, so the game starts with Intimidate
    applied, Defiant having answered it, and a lead's weather already up. Without that,
    every sun and rain team in the format was searched with no weather.
    """
    # A lead's Trace between two foes is drawn from `rng` for the game being played, and is
    # the first foe (noted and dropped) for a caller that only wants a position (IKA-203).
    return port.apply_lead_abilities(reg, _opening(reg, own, foe), rng=rng).position


def positions_from_sets(
    reg: Regulation, pairs: Sequence[tuple[list[SampledSet], list[SampledSet]]]
) -> list[Position]:
    """`position_from_sets` without a generator, for many pairs at once.

    The selection solve builds 8,100 of these a pair, and the leads' switch-ins were a
    quarter of a millisecond each in Python and are a round trip each through the port
    (IKA-209: 0.25 against 0.75 ms, nearly all of it the positions' JSON). But 8,100
    selections are 900 lead quartets, each with nine different backs, and the switch-ins
    before turn 1 are the leads' and the field's: so the port is asked once a quartet, and
    the other eight take its answer with their own back two put in.

    That is checked, not assumed: the quartet's answer must have left its own back two and
    the side's `mega_capable_slots` exactly as they went in, or the quartet's every member
    is asked for. What it cannot check is a switch-in that *reads* the bench without
    writing it (Illusion, which neither engine models) -- a regulation that brings one
    has to stop sharing. `tools/diff_generation.py` with a learned leaf holds the result to
    the per-position answer through the recorded selection policy.

    Shallow: the members of a quartet share its leads' objects, and every opening shares
    its Pokemon with the other openings that put the same set in the same slot (IKA-267:
    8,100 positions of 12 sets were 97,200 stat lines of 48 different Pokemon). These
    positions go to a leaf evaluator, which reads them.
    """
    active = reg.meta.active_per_side
    made: dict[tuple[int, int, int], Pokemon] = {}
    openings = [_opening(reg, own, foe, made) for own, foe in pairs]
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for index, (own, foe) in enumerate(pairs):
        key = (tuple(id(s) for s in own[:active]), tuple(id(s) for s in foe[:active]))
        groups.setdefault(key, []).append(index)
    firsts = [members[0] for members in groups.values()]
    answered = port.apply_lead_abilities_many(reg, [openings[i] for i in firsts])
    out: list[Position | None] = [None] * len(pairs)
    unshared: list[int] = []
    for members, phase in zip(groups.values(), answered, strict=True):
        led = phase.position
        opening = openings[members[0]]
        if not _bench_untouched(led, opening, active):
            out[members[0]] = led
            unshared.extend(members[1:])
            continue
        for index in members:
            out[index] = led if index == members[0] else _with_back(led, openings[index], active)
    if unshared:
        for index, phase in zip(
            unshared,
            port.apply_lead_abilities_many(reg, [openings[i] for i in unshared]),
            strict=True,
        ):
            out[index] = phase.position
    return [pos for pos in out if pos is not None]


def _bench_untouched(led: Position, opening: Position, active: int) -> bool:
    """Whether the leads' switch-ins left everything a member of the quartet differs in."""
    return all(
        [mon.to_json() for mon in after.pokemon[active:]]
        == [mon.to_json() for mon in before.pokemon[active:]]
        and after.mega_capable_slots == before.mega_capable_slots
        for after, before in zip(led.sides, opening.sides, strict=True)
    )


def _with_back(led: Position, opening: Position, active: int) -> Position:
    """`led` with `opening`'s back Pokemon and mega list: its own quartet, another back."""
    return replace(
        led,
        sides=[
            replace(
                side,
                pokemon=[*side.pokemon[:active], *mine.pokemon[active:]],
                mega_capable_slots=list(mine.mega_capable_slots),
            )
            for side, mine in zip(led.sides, opening.sides, strict=True)
        ],
    )


def _opening(
    reg: Regulation,
    own: list[SampledSet],
    foe: list[SampledSet],
    made: dict[tuple[int, int, int], Pokemon] | None = None,
) -> Position:
    """The turn-1 position before the leads' switch-ins.

    ``made`` shares the Pokemon between openings: a caller that passes one gets the same
    object for the same set in the same slot of the same side, and must not change it.
    """
    sides: list[Side] = []
    for side_index, sets in enumerate((own, foe)):
        mons = [_made_pokemon(reg, made, side_index, i, entry) for i, entry in enumerate(sets)]
        sides.append(
            Side(
                id=f"p{side_index + 1}",
                name=f"p{side_index + 1}",
                active=list(range(reg.meta.active_per_side)),
                pokemon=mons,
                slot_conditions=[[] for _ in range(reg.meta.active_per_side)],
                mega_capable_slots=[
                    m.slot for m in mons if reg.mega_target(m.species, m.item) is not None
                ],
            )
        )
    return Position(format=reg.meta.format_id, sides=sides, turn=1, field=Field())


def _made_pokemon(
    reg: Regulation,
    made: dict[tuple[int, int, int], Pokemon] | None,
    side_index: int,
    index: int,
    entry: SampledSet,
) -> Pokemon:
    """`_make_pokemon`, once per (side, slot, set) when `made` is given (IKA-267).

    Keyed by the set's identity: the caller holds every set for as long as `made` lives,
    so an id is not reused under it.
    """
    active = index if index < reg.meta.active_per_side else None
    if made is None:
        return _make_pokemon(reg, index, entry, active)
    key = (side_index, index, id(entry))
    found = made.get(key)
    if found is None:
        found = made[key] = _make_pokemon(reg, index, entry, active)
    return found


def _with_lead(
    rng: np.random.Generator,
    roster: Roster,
    wanted: tuple[str, ...],
    drawn: tuple[int, ...],
) -> tuple[int, ...]:
    """The same four, reordered so `wanted` leads -- or a fresh four containing them.

    Forcing the *selection* rather than a move is what makes this usable as teacher data.
    A random move that starts a slow plan is undone on the next turn, because the search
    that would have to continue it does not value it; a Pokemon that is on the field is on
    the field. The deviation lasts the whole game, and the outcome is a real outcome of
    having opened that way.

    It exists because the lead is where the plan starts and the lead is what the book
    stopped drawing. Toxapex is brought in 57.4% of the book's mass and leads 2.5% of it;
    Toxapex with Incineroar led 1.44% of generation 9 -- 46 games in 3,187 -- against 47%
    of the human repertoire.
    """
    names = [entry.species for entry in roster.sets]
    missing = [n for n in wanted if n not in names]
    if missing:
        raise ValueError(f"the roster has no {missing}; it has {names}")
    lead = [names.index(n) for n in wanted]
    rest = [i for i in drawn if i not in lead]
    while len(lead) + len(rest) < len(drawn):
        spare = [i for i in range(len(names)) if i not in lead and i not in rest]
        if not spare:
            break
        rest.append(int(spare[rng.integers(len(spare))]))
    return tuple([*lead, *rest[: max(len(drawn) - len(lead), 0)]])


def _sample_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    total = float(weights.sum())
    if total <= 0:
        return int(rng.integers(len(weights)))
    return int(rng.choice(len(weights), p=weights / total))


#: Which completion of the opponent's unseen slots a leaf or policy ranking reads
#: (`_menus.views`). "heaviest" ships; "first" is the rule before IKA-143.
RANK_VIEWS = ("heaviest", "first")


def _menus(
    reg: Regulation,
    pos: Position,
    limits: tuple[int, int],
    evaluate: LeafEvaluator,
    budget: Budget,
    rank_by_leaf: bool,
    policy: Any = None,
    spreads: dict[int, list] | None = None,
    rank_view: str = "heaviest",
    used: dict[int, tuple[int, tuple[str, ...]]] | None = None,
    rank_fill: str = DEFAULT_RANK_FILL,
) -> tuple[list[SideAction], list[SideAction]]:
    """Both sides' candidate menus, as one agent sees them.

    A menu belongs to the agent that built it, not to the position: with `rank_by_leaf`
    the candidates are ordered by what the leaf thinks of where they lead rather than by
    expected damage, and two agents ranking differently are choosing from different menus.
    That is why this returns a pair and why `play_game` calls it once per agent when the
    settings differ -- handing one agent's menu to the other would make a ranking
    comparison measure nothing.

    Three orderings, not two. ``policy`` -- a model from :func:`pokeuraou.policy.load_policy`
    -- supersedes ``rank_by_leaf`` when it is given, because both answer the same question
    and an agent asks it once. It is a supersession rather than an error so that an agent
    can be described by adding one setting to an existing pair rather than by rewriting it.

    ``rank_fill`` is how the leaf ranking fills its cells (`search.parse_rank_fill`,
    IKA-268): how many damage replies each candidate is resolved against, and whether at
    this budget or at `Budget.fast`. It changes nothing with the damage or policy ranking.
    """
    if rank_view not in RANK_VIEWS:
        raise ValueError(f"rank_view {rank_view!r} is not one of {RANK_VIEWS}")
    references, fast_fill = parse_rank_fill(rank_fill)
    rank_budget = Budget.fast() if fast_fill else budget
    if policy is None and not rank_by_leaf:
        # The damage score reads only the active Pokemon, so it has nothing to be blind
        # about and the true position costs nothing here.
        return (
            narrow(reg, pos, 0, limit=limits[0]).actions,
            narrow(reg, pos, 1, limit=limits[1]).actions,
        )

    def views(side: int) -> list[tuple[Position, float]]:
        """The positions side `side` ranks from.

        One completion, not all of them. The ranking only decides which actions reach the
        matrix, and measured over 60 recorded positions by solving every legal pair in
        full, ordering from a single consistent world gives up 0.55 points against a best
        reply where averaging over all six gives up 0.58 -- the same number, and both far
        under the damage ordering's 1.78. Six times the ordering cost bought nothing, and
        it was the difference between 82 and 135 generated games a minute.

        Leak-free either way: the completion is one the sheet allows, chosen without
        looking at the truth. Ranking from the *true* bench scored 0.44 on the same
        positions, which is the same number again and is the information nobody has.

        Deterministic -- the heaviest completion, not a sampled one -- so a rerun of a
        game is the same game. The heaviest, first on ties, not the first enumerated
        (IKA-143): the "one is as good as six" measurement above was taken when the
        weights were uniform, and every completion was then as likely as the first.
        Under the book's bench prior they are not -- on 60 recorded mid-game positions of
        the w12 pool (94 rankings) the first completion was the heaviest in 45, it carried
        a mean weight of 0.36 against the heaviest's 0.72, and ranking from the heaviest
        instead changed the width-12 menu in 33 of the 94, by 1.26 of 11.7 actions on
        average (2.41 where the two differ). Averaging over every completion would change
        it in 51 at 4.5 times the ranking's fills, which is the cost the paragraph above
        declined. `rank_view="first"` is the old rule, kept so an arm can play it on the
        board. Under uniform weights the two rules are the same completion.

        `used`, when given, gets `side -> (index, species)` of the completion side
        `side` ranked from, so the caller can record it.
        """
        if spreads is None:
            return [(pos, 1.0)]
        items = spreads[1 - side]
        if not items:
            return [(pos, 1.0)]
        # `max` keeps the first of equal maxima, so uniform weights pick index 0.
        index = (
            0
            if rank_view == "first"
            else max(range(len(items)), key=lambda i: items[i].weight)
        )
        if used is not None:
            used[side] = (index, tuple(items[index].species))
        return [(items[index].position, 1.0)]

    def ranker(side: int) -> Any:  # noqa: ANN401
        parts = [
            (
                policy_ranking(policy, at, side)
                if policy is not None
                else leaf_ranking(
                    reg, at, side, evaluate, budget=rank_budget, references=references
                ),
                weight,
            )
            for at, weight in views(side)
        ]
        if policy is not None:
            return believed_ranking(parts)
        # IKA-278: the ranking itself unless a generation worker is recording it.
        return rank_scores.watch(
            believed_ranking(parts), side, used, spreads, [w for _r, w in parts]
        )

    return (
        narrow(reg, pos, 0, limit=limits[0], rank=ranker(0)).actions,
        narrow(reg, pos, 1, limit=limits[1], rank=ranker(1)).actions,
    )


def _believed(
    spreads: dict[int, list], drops: tuple[str, str]
) -> dict[int, list]:
    """Each side's completions as the agent that is blind to them believes them.

    `spreads[s]` completes side `s`'s own bench, so it is side `1 - s`'s belief, and it
    is side `1 - s`'s rule that drops from it (IKA-283). One dict then serves both
    agents: each side's game, its menu's view and its Bayesian solve read only the
    other side's list. The dict itself when neither drops anything.
    """
    if set(drops) == {DEFAULT_BENCH_DROP}:
        return spreads
    out = {side: drop_light(items, drops[1 - side]) for side, items in spreads.items()}
    if timing.ON:
        timing.count(
            "completions.dropped",
            sum(len(spreads[side]) - len(out[side]) for side in spreads),
        )
    return out


def _rank_views(
    own_views: dict[int, tuple[int, tuple[str, ...]]],
    foe_views: dict[int, tuple[int, tuple[str, ...]]] | None,
) -> list[list[Any] | None] | None:
    """`Decision.rank_views` from each agent's own construction of its own menu.

    Side 0's menu is always side 0's construction; side 1's is its own when it built one
    and otherwise the shared one.
    """
    got = [own_views.get(0), (foe_views if foe_views is not None else own_views).get(1)]
    if got == [None, None]:
        return None
    return [None if g is None else [g[0], list(g[1])] for g in got]


def _bench_weights(
    bench_prior: tuple[BenchPrior | None, BenchPrior | None] | None,
    side: int,
    pos: Position,
    seen: frozenset[int],
    record: GameRecord,
    leads: frozenset[str] | None = None,
) -> dict[tuple[str, ...], float] | None:
    """How likely each way of filling that side's unseen slots is, or None for uniform.

    `completions` enumerates those ways and, without weights, treats them as equally
    likely -- so the belief holds back-two pairs the opponent would never bring at the
    same weight as the ones they would, and the opponent inside it is weaker than the one
    across the table. Measured, that is worth about 10 points of the search's value early
    in a hidden-bench game, against 1 point in the open game (G27/G30).

    An empty result means the distribution explains nothing on the board, which should
    not happen and is recorded rather than passed over: the fallback is the uniform
    belief this exists to replace, and a silent fallback to the thing being fixed is how
    a fix becomes invisible.

    `leads` is the pair that side led turn 1 with, as `play_game` recorded it; a
    selection counts only if it led that pair, not merely brought it (IKA-118).
    """
    if bench_prior is None or bench_prior[side] is None:
        return None
    weights = bench_prior[side].weights(shown_species(pos, side, seen), leads)
    if not weights:
        note = f"bench weights: side {side}'s selection prior explains nothing on board"
        if note not in record.unmodelled:
            record.unmodelled.append(note)
        return None
    return weights


def play_game(
    reg: Regulation,
    rng: np.random.Generator,
    own: list[SampledSet],
    foe: list[SampledSet],
    foe_archetype: str,
    *,
    objective: Objective = HP_SHARE,
    search_limit: int | tuple[int, int] = SEARCH_LIMIT,
    max_turns: int = MAX_TURNS,
    evaluate: LeafEvaluator | None | tuple[LeafEvaluator | None, LeafEvaluator | None] = None,
    selection: tuple[list[str], list[str], tuple[int, ...], tuple[int, ...]] | None = None,
    depth: int | tuple[int, int] = 1,
    rank_by_leaf: bool | tuple[bool, bool] = False,
    policy: Any | tuple[Any, Any] = None,
    solve_sparsely: bool | tuple[bool, bool] = False,
    solve_restricted: bool | tuple[bool, bool] = False,
    start: Position | None = None,
    first_action: str | None = None,
    sheets: tuple[Sequence[SampledSet], Sequence[SampledSet]] | None = None,
    open_information: bool = False,
    bench_prior: tuple[BenchPrior | None, BenchPrior | None] | None = None,
    one_agent: bool = True,
    rank_view: str | tuple[str, str] = "heaviest",
    rank_fill: str | tuple[str, str] = DEFAULT_RANK_FILL,
    bench_drop: str | tuple[str, str] = DEFAULT_BENCH_DROP,
    deepen: int | tuple[int, int] = 0,
) -> GameRecord:
    """Plays one game to a result, sampling both sides from the turn's equilibrium.

    ``search_limit`` may be a pair, giving each side its own number of candidates. Equal
    values are the training setting; an unequal pair is a diagnostic -- if one side wins
    less when searched just as widely, the imbalance was in the search and not the teams.

    ``depth`` takes a pair for the same reason, and it is how depth-2 is measured against
    depth-1: one side looks a ply further and the win rate says what that bought. Depth 1
    on both sides is the search this project has always had.

    ``start`` resumes from a position instead of building one from the two teams, which is
    how a phase that self-play rarely reaches gets sampled. Games average 9.4 turns and
    only 5.4% of them reach turn 15, so a value function sees the endgame -- where a stall
    plan finally pays -- in one position in twenty. Resuming from recorded turn-12
    positions and playing them out produces real outcomes in that phase without waiting
    for self-play to arrive there on its own. The label stays sound because the game is
    still played to a win or a loss; what changes is which positions get labelled.

    ``sheets`` are the two sides' *sixes*. Given them, neither search is shown the other
    side's unplayed bench: each solves over every four the opponent's sheet still allows,
    as the Bayesian game it is. Omitted, the search sees the opponent's whole four, which
    is what every game recorded before this did -- and what 47.4% of decisions had no
    right to. The two are different agents and `provenance` says which.

    Omitting ``sheets`` is not enough to get the open game: it also takes
    ``open_information=True``, and without either the call stops (IKA-123). The open game
    was the silent default for as long as the hidden bench was an afterthought, and a
    caller that forgot the sheets got an easier game under the same call. Passing both is
    a contradiction and stops too.

    ``one_agent`` says whether the two sides are one agent playing itself, which is what
    self-play is and what a match is not. It changes no decision; it changes only how
    `search_seconds` charges work that served both sides at once. One agent solving one
    matrix gets both of its moves out of it, so each move carries half. Two agents that
    happen to hold the same leaf each get one move out of it, and each would have paid
    all of it in a game it played alone -- so each is charged all of it, and the field
    keeps meaning "what this agent spends on a move" whether or not the caller was able
    to share the object. A match that shares its arms and leaves this True reports half
    the per-move cost of the configuration it is pricing.

    ``solve_restricted`` takes a pair and is a property of depth 2 alone: it says whether
    the refined cells are read as the restricted game they form, or left in the full
    matrix the way the search that IKA-12 played out did. At depth 1 it changes nothing,
    which is what makes it safe to carry on an arm that is not deep.

    ``solve_sparsely`` takes a pair too, and it is the one whose two values are supposed
    to be *equally correct*: both settle on an equilibrium of the same game, verified to
    an exploitability of 7.7e-08. What differs is which vertex of a degenerate optimum
    they land on, and a maximin strategy only guarantees the value -- against an opponent
    who is not playing the equilibrium, two equilibria can take different amounts. That is
    what a mismatched pair measures.

    ``rank_view`` takes a pair too: which completion of the opponent's unseen slots each
    agent's leaf or policy ranking reads under a hidden bench (`RANK_VIEWS`, IKA-143).
    "heaviest" ships; "first" is the enumeration-order rule every game before it played.
    It changes nothing without ``sheets`` or with the damage ranking.

    ``rank_fill`` takes a pair too: how each agent's leaf ranking fills its cells --
    ``refs<N>`` replies at the matrix budget, ``-fast`` for `Budget.fast` (IKA-268). It
    changes nothing with the damage or policy ranking.

    ``bench_drop`` takes a pair too: which completions of the opponent's unseen slots
    each agent's belief leaves out at a move node (`hidden.parse_bench_drop`, IKA-283).
    "none" ships. It changes nothing without ``sheets``.

    ``deepen`` takes a pair too: each agent's budget in cells for deepening its move
    decisions best first after the depth-1 solve (`deepen.best_first`, IKA-33). 0 ships
    and is the search unchanged. Under a hidden bench it applies only where neither
    side's bench is hidden -- the node is then the open game's node, and it is solved as
    one (`search`) instead of as a one-completion Bayesian game; a node with a bench
    still hidden stays at depth 1 (`belief_solve` has no depth, IKA-111). It goes with
    depth 1 and the full-matrix solve only.
    """
    # Closes the stretch since the last game's last record (IKA-98); the first one ends startup.
    timing.decided("between")
    limits = (search_limit, search_limit) if isinstance(search_limit, int) else search_limit
    depths = (depth, depth) if isinstance(depth, int) else depth
    ranked = (
        (rank_by_leaf, rank_by_leaf)
        if isinstance(rank_by_leaf, bool)
        else rank_by_leaf
    )
    sparse = (
        (solve_sparsely, solve_sparsely)
        if isinstance(solve_sparsely, bool)
        else solve_sparsely
    )
    restricted = (
        (solve_restricted, solve_restricted)
        if isinstance(solve_restricted, bool)
        else solve_restricted
    )
    policies = policy if isinstance(policy, tuple) else (policy, policy)
    leaves = evaluate if isinstance(evaluate, tuple) else (evaluate, evaluate)
    views_rule = (rank_view, rank_view) if isinstance(rank_view, str) else tuple(rank_view)
    for rule in views_rule:
        if rule not in RANK_VIEWS:
            raise ValueError(f"rank_view {rule!r} is not one of {RANK_VIEWS}")
    fills = (rank_fill, rank_fill) if isinstance(rank_fill, str) else tuple(rank_fill)
    for fill in fills:
        parse_rank_fill(fill)
    drops = (bench_drop, bench_drop) if isinstance(bench_drop, str) else tuple(bench_drop)
    for drop in drops:
        parse_bench_drop(drop)
    deepens = (deepen, deepen) if isinstance(deepen, int) else tuple(deepen)
    for side, cells in enumerate(deepens):
        if cells < 0:
            raise ValueError(f"deepen {deepens}: a budget of cells is not negative")
        if cells and (depths[side] != 1 or sparse[side]):
            raise ValueError(
                f"deepen {deepens} on side {side} goes with depth 1 and the full-matrix "
                f"solve, not depth {depths[side]} / solve_sparsely {sparse[side]}"
            )
    if sheets is None and not open_information:
        raise ValueError(
            "no `sheets`, so the search would be shown the opponent's four -- the open "
            "game. Pass both sixes as `sheets` to hide the bench (what ships), or "
            "`open_information=True` to mean the open game (a reference). Since IKA-123 "
            "it is not a default."
        )
    if sheets is not None and open_information:
        raise ValueError("`sheets` hides the bench and `open_information=True` shows it")
    if sheets is not None and (
        depths != (1, 1) or sparse != (False, False) or restricted != (False, False)
    ):
        # `belief_solve` takes neither, so under a hidden bench these were accepted,
        # recorded per side in the provenance, and then dropped. An argument that is
        # silently ignored is how every measurement defect found today was built: the
        # caller reads the flag it passed, the record repeats it, and nothing played it.
        raise ValueError(
            f"depth {depths}, solve_sparsely {sparse} and solve_restricted "
            f"{restricted} cannot be honoured with a hidden bench -- belief_solve has no "
            "parameter for any of them. Pass depth 1 with both flags False, or drop "
            "`sheets` and measure in the open game."
        )
    record = GameRecord(
        own_team=[_set_json(reg, s) for s in own],
        foe_team=[_set_json(reg, s) for s in foe],
        foe_archetype=foe_archetype,
    )
    if selection is not None:
        own_six, foe_six, own_pick, foe_pick = selection
        record.own_six = list(own_six)
        record.foe_six = list(foe_six)
        record.own_pick = list(own_pick)
        record.foe_pick = list(foe_pick)
    record.information = "open" if sheets is None else "hidden-bench"
    record.ranking = (
        "policy" if policies[0] is not None else "leaf" if ranked[0] else "damage"
    )
    record.rank_view = list(views_rule)
    record.rank_fill = list(fills)
    record.bench_drop = list(drops)
    record.deepen = list(deepens)
    pos = start if start is not None else position_from_sets(reg, own, foe, rng=rng)
    budget = Budget.matrix()
    # Who each side has shown, accumulated across turns. A Pokemon that came in and went
    # back out is still known, and the position alone stops saying so -- so this is
    # carried rather than recomputed from the board each time.
    #
    # Carried as identities, and turned into this position's slots at every decision.
    # It used to carry the slots themselves, and `_do_switch` renumbers those: a lead that
    # went back to the bench unharmed took the index of whoever came in, which nobody had
    # marked, and the next decision's belief offered worlds without it -- 15.8% of the
    # move decisions and 23.6% of the replacements in the shipping pool (IKA-117).
    seen: list[frozenset[str]] = [frozenset(), frozenset()]
    # Which two each side led with, read off the turn-1 position and carried as names for
    # the same reason `seen` is (IKA-118). The bench belief conditions on it: a selection
    # is ordered, and one that brought a lead but planned it for the back is not what this
    # side played. A game resumed from a later `start` has no turn 1 to read, and stays
    # None -- conditioned on `seen` alone, as before.
    leads: list[frozenset[str] | None] = [None, None]
    if pos.turn == 1:
        leads = [
            frozenset(
                shown_species(
                    pos,
                    i,
                    frozenset(
                        mon.slot
                        for mon in pos.sides[i].pokemon
                        if mon.active_index is not None
                    ),
                )
            )
            for i in (0, 1)
        ]
        # The same pair as identities, for the record only (IKA-127).
        record.leads = [
            [
                identity(mon)
                for mon in sorted(
                    (m for m in pos.sides[i].pokemon if m.active_index is not None),
                    key=lambda m: m.active_index,
                )
            ]
            for i in (0, 1)
        ]
    else:
        record.leads = [None, None]

    for _step in range(max_turns * 2):
        if pos.ended:
            break

        seen = [seen_identities(pos, i, seen[i]) for i in (0, 1)]
        shown = [seen_slots(pos, i, seen[i]) for i in (0, 1)]
        recorded_shown = _shown_record(pos, None if sheets is None else seen)
        owed = port.replacements_needed(reg, pos)
        if any(owed[0]) or any(owed[1]):
            pos = _do_replacement_node(
                reg, rng, pos, owed, record, leaves,
                sheets=sheets, shown=shown, bench_prior=bench_prior, leads=leads,
                recorded_shown=recorded_shown,
            )
            continue

        own_leaf = leaves[0] if leaves[0] is not None else objective.batch
        foe_leaf = leaves[1] if leaves[1] is not None else objective.batch
        # Built before the menus, because the menus are ranked from it: a leaf or a policy
        # ordering candidates from the true position would pick *which actions get a
        # number* using a bench nobody has seen, however carefully the matrix over them is
        # then solved.
        spreads = None
        if sheets is not None:
            try:
                spreads = {
                    side: completions(
                        reg, pos, side, sheets[side], seen=shown[side],
                        weights=_bench_weights(
                            bench_prior, side, pos, shown[side], record, leads[side]
                        ),
                    )
                    for side in (0, 1)
                }
            except ValueError as problem:
                record.unmodelled.append(f"hidden bench: {problem}")
                break
            spreads = _believed(spreads, drops)
        menu_started = perf_counter()
        # Side -> the completion that side's ranking read, per agent: `own_views` from
        # side 0's construction, `foe_views` from side 1's when it builds its own.
        own_views: dict[int, tuple[int, tuple[str, ...]]] = {}
        foe_views: dict[int, tuple[int, tuple[str, ...]]] | None = None
        rank_scores.at_node(len(record.decisions), pos.turn, 0)  # IKA-278
        ours, theirs = _menus(
            reg, pos, limits, own_leaf, budget, ranked[0], policies[0], spreads,
            rank_view=views_rule[0], used=own_views, rank_fill=fills[0],
        )
        menu_seconds = perf_counter() - menu_started
        if not ours or not theirs:
            break

        # Whether the two agents build the SAME menu, which is the only case where one
        # construction may serve both.
        #
        # This read `ranked[1] == ranked[0] and policies[1] is policies[0]` and left the
        # leaf out. Two arms both passing --rank-leaf with DIFFERENT models rank by
        # different numbers, so they do not have one menu -- and the comment three lines
        # below says exactly that ("a ranking that differs is a different menu") while the
        # condition could not see it. Every model-against-model match run with
        # `--rank-leaf --baseline-rank-leaf` had the column player choosing from a menu
        # the row player's value function picked; about 25,000 recorded games.
        #
        # The leaf only reaches the menu through `leaf_ranking`, so it matters when no
        # policy is given and the ranking is by leaf -- and not otherwise.
        same_menu = (
            ranked[1] == ranked[0]
            and policies[1] is policies[0]
            and views_rule[1] == views_rule[0]
            # A leaf ranking filled another way orders another menu (IKA-268).
            and (fills[1] == fills[0] or policies[0] is not None or not ranked[0])
            and (
                policies[0] is not None
                or not ranked[0]
                or leaves[1] is leaves[0]
            )
        )

        own_seconds = foe_seconds = 0.0
        foe_solved = False
        foe_search_value: float | None = None
        # What each side's best-first deepening did here (IKA-33); None where it did not.
        deepened: list[Any] = [None, None]

        if spreads is not None:
            # Each side is uncertain about a different bench, so each gets its own answer.
            # The node underneath them is resolved once: their hidden slots are disjoint
            # and a cell that reaches neither resolves the same way whatever is standing
            # on either bench.
            #
            # Unless neither is (IKA-33): then the node is the open game's, and a side
            # that deepens solves it as one, with `search`. A node with a bench still
            # hidden stays with `belief_solve`, which has no depth (IKA-111).
            exact = deepens != (0, 0) and all(
                len(items) == 1 and items[0].exact for items in spreads.values()
            )
            deep = (exact and deepens[0] > 0, exact and deepens[1] > 0)
            own_deep = foe_deep = None
            solve_started = perf_counter()
            try:
                # Two menus mean two solves, each read on one side only: this one for
                # side 0, the one over side 1's menu below for side 1. Asking each for
                # the side it is read on is what keeps the other side's node and LP from
                # being built and thrown away (IKA-282: half of the board's matrix,
                # dirty fill and LP).
                asked = tuple(
                    side for side in ((0, 1) if same_menu else (0,)) if not deep[side]
                )
                answers = (
                    belief_solve(
                        reg, pos, ours, theirs, spreads,
                        {0: own_leaf, 1: foe_leaf}, budget=budget,
                        sides=asked,
                    )
                    if asked
                    else {}
                )
                if deep[0]:
                    own_deep = search(
                        reg, pos, ours, theirs, own_leaf, budget=budget,
                        deepen=deepens[0],
                    )
                if same_menu and deep[1]:
                    # One agent on both sides reads both strategies off one solve.
                    foe_deep = (
                        own_deep
                        if own_deep is not None
                        and leaves[1] is leaves[0]
                        and deepens[1] == deepens[0]
                        else search(
                            reg, pos, ours, theirs, foe_leaf, budget=budget,
                            deepen=deepens[1],
                        )
                    )
            except EquilibriumError:
                break
            own_seconds = perf_counter() - solve_started
            record.unmodelled.extend(
                set().union(*(answer.unmodelled for answer in answers.values()))
            )
            if own_deep is None:
                own_strategy = answers[0].strategy
                search_value = answers[0].value
            if own_deep is not None:
                record.unmodelled.extend(own_deep.unmodelled)
                own_strategy = np.asarray(own_deep.equilibrium.row_strategy, dtype=np.float64)
                search_value = float(own_deep.equilibrium.value)
                deepened[0] = own_deep.deepened
            foe_theirs = theirs
            if same_menu and foe_deep is not None:
                if foe_deep is not own_deep:
                    record.unmodelled.extend(foe_deep.unmodelled)
                foe_strategy = np.asarray(foe_deep.equilibrium.col_strategy, dtype=np.float64)
                foe_search_value = float(foe_deep.equilibrium.value)
                deepened[1] = foe_deep.deepened
            elif same_menu:
                foe_strategy = answers[1].strategy
                # Side 1 solved the negated transpose, so its value is minus side 0's
                # win probability as side 1 believes it; negated back into side 0's units.
                foe_search_value = -answers[1].value
            if not same_menu:
                # The open path rebuilds the column player's game when the settings
                # differ; this one used to skip that entirely, so under a hidden bench
                # `ranked[1]` and `policies[1]` were dead arguments -- the provenance
                # recorded them per side and only side 0's were ever played. The anchor
                # of the hidden-bench scale was measured this way, with the hp-share arm
                # handed a menu ranked by the other arm's value function in one seat.
                foe_started = perf_counter()
                foe_views = {}
                rank_scores.at_node(len(record.decisions), pos.turn, 1)  # IKA-278
                foe_ours, foe_theirs = _menus(
                    reg, pos, limits, foe_leaf, budget, ranked[1], policies[1], spreads,
                    rank_view=views_rule[1], used=foe_views, rank_fill=fills[1],
                )
                if not foe_ours or not foe_theirs:
                    break
                try:
                    if deep[1]:
                        foe_deep = search(
                            reg, pos, foe_ours, foe_theirs, foe_leaf, budget=budget,
                            deepen=deepens[1],
                        )
                    else:
                        foe_answers = belief_solve(
                            reg, pos, foe_ours, foe_theirs, spreads,
                            {0: own_leaf, 1: foe_leaf}, budget=budget, sides=(1,),
                        )
                except EquilibriumError:
                    break
                if foe_deep is not None:
                    record.unmodelled.extend(foe_deep.unmodelled)
                    foe_strategy = np.asarray(
                        foe_deep.equilibrium.col_strategy, dtype=np.float64
                    )
                    foe_search_value = float(foe_deep.equilibrium.value)
                    deepened[1] = foe_deep.deepened
                else:
                    record.unmodelled.extend(foe_answers[1].unmodelled)
                    foe_strategy = foe_answers[1].strategy
                    # Side 1 played off this solve, over its own menu, so its value is
                    # this one.
                    foe_search_value = -foe_answers[1].value
                foe_seconds = perf_counter() - foe_started
                foe_solved = True
        else:
            solve_started = perf_counter()
            try:
                own_search = search(
                    reg, pos, ours, theirs, own_leaf, budget=budget, depth=depths[0],
                    solve_sparsely=sparse[0], solve_restricted=restricted[0],
                    deepen=deepens[0],
                )
            except EquilibriumError:
                break
            own_seconds = perf_counter() - solve_started
            record.unmodelled.extend(own_search.unmodelled)
            equilibrium = own_search.equilibrium

            # With a different leaf, depth or ranking the two sides are no longer solving
            # one game, so the column player's strategy has to come from *its* matrix --
            # over *its* menu, because a ranking that differs is a different menu.
            # Identical settings on both sides skip all of this and behave exactly as
            # before.
            foe_equilibrium = equilibrium
            foe_theirs = theirs
            if (
                leaves[1] is not leaves[0]
                or depths[1] != depths[0]
                or ranked[1] != ranked[0]
                or policies[1] is not policies[0]
                or sparse[1] != sparse[0]
                or restricted[1] != restricted[0]
                or (fills[1] != fills[0] and ranked[0] and policies[0] is None)
                or deepens[1] != deepens[0]
            ):
                foe_started = perf_counter()
                rank_scores.at_node(len(record.decisions), pos.turn, 1)  # IKA-278
                foe_ours, foe_theirs = (
                    (ours, theirs)
                    if same_menu
                    else _menus(
                        reg, pos, limits, foe_leaf, budget, ranked[1], policies[1],
                        spreads, rank_view=views_rule[1], rank_fill=fills[1],
                    )
                )
                if not foe_ours or not foe_theirs:
                    break
                try:
                    foe_search = search(
                        reg, pos, foe_ours, foe_theirs, foe_leaf, budget=budget,
                        depth=depths[1], solve_sparsely=sparse[1],
                        solve_restricted=restricted[1], deepen=deepens[1],
                    )
                except EquilibriumError:
                    break
                record.unmodelled.extend(foe_search.unmodelled)
                foe_equilibrium = foe_search.equilibrium
                deepened[1] = foe_search.deepened
                foe_seconds = perf_counter() - foe_started
                foe_solved = True
            own_strategy = np.asarray(equilibrium.row_strategy, dtype=np.float64)
            foe_strategy = np.asarray(foe_equilibrium.col_strategy, dtype=np.float64)
            search_value = float(equilibrium.value)
            deepened[0] = own_search.deepened
            if not foe_solved:
                deepened[1] = own_search.deepened

        if foe_solved:
            # `same_menu` is the only case where one construction served both agents;
            # otherwise side 0's menus are side 0's alone and side 1 timed its own.
            shared = menu_seconds / 2 if same_menu else 0.0
            record.search_seconds[0] += own_seconds + menu_seconds - shared
            record.search_seconds[1] += foe_seconds + shared
        elif one_agent:
            # One construction and one solve served both sides of one agent. Neither of
            # them would have spent less alone, and neither of them spent it alone.
            record.search_seconds[0] += (menu_seconds + own_seconds) / 2
            record.search_seconds[1] += (menu_seconds + own_seconds) / 2
        else:
            # Two agents, one leaf object -- a match of a configuration against itself,
            # or against something that differs only off the board. The machine did the
            # work once, and that is the whole point of letting the caller share the
            # object, but neither agent would have spent less than all of it alone and
            # this field is what one agent spends. Charging the halves here is how a
            # pricing run would come back reading half the configuration's cost, with
            # the ratio it is usually read as intact and the absolute silently doubled
            # against every figure recorded before the arms could be shared.
            record.search_seconds[0] += menu_seconds + own_seconds
            record.search_seconds[1] += menu_seconds + own_seconds

        own_index = _sample_index(rng, own_strategy)
        if first_action is not None and not record.decisions:
            # Side 0's opening move, overridden once. Everything after it is the search's
            # own, so the game measures "what happens if this is played here" rather than
            # "what happens if this is played forever".
            #
            # It exists for the one hypothesis the recorded games cannot settle. The
            # equilibrium prices the switch to the sweeper at zero in 60% of the positions
            # where it is available, and the 14% of times it is taken are the ones where
            # the equilibrium already liked it -- which say the leaf ranks *those*
            # correctly and nothing about the 60%. Forcing it there and playing on is the
            # only way to ask whether the zero is right.
            wanted = [i for i, a in enumerate(ours) if a.to_choice() == first_action]
            if not wanted:
                record.unmodelled.append(
                    f"forced opening {first_action!r} was not among this side's actions"
                )
            else:
                own_index = wanted[0]
        chosen = [
            ours[own_index],
            foe_theirs[_sample_index(rng, foe_strategy)],
        ]
        record.decisions.append(
            Decision(
                turn=pos.turn,
                kind="move",
                position=pos.to_json(),
                own_actions=[a.to_choice() for a in ours],
                own_policy=[float(x) for x in own_strategy],
                foe_actions=[a.to_choice() for a in foe_theirs],
                foe_policy=[float(x) for x in foe_strategy],
                search_value=search_value,
                own_chosen=chosen[0].to_choice(),
                foe_chosen=chosen[1].to_choice(),
                rank_views=_rank_views(own_views, foe_views),
                shown=recorded_shown,
                foe_search_value=foe_search_value,
                deepened=(
                    [None if got is None else got.to_json() for got in deepened]
                    if any(got is not None for got in deepened)
                    else None
                ),
            )
        )
        timing.decided("move")
        advanced = _advance_turn(
            reg, rng, pos, chosen, record, leaves, objective,
            hidden=(
                None
                if sheets is None
                else _HiddenBench(sheets, list(seen), bench_prior, list(leads))
            ),
        )
        if advanced is None:
            record.end_reason = "unresolved"
            break
        pos = advanced
        record.turns = pos.turn

    if pos.ended and pos.winner is not None:
        record.outcome = 1.0 if pos.winner == pos.sides[0].id else 0.0
    _close_record(record, pos)
    return record


def _advance_turn(
    reg: Regulation,
    rng: np.random.Generator,
    pos: Position,
    chosen: list[SideAction],
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None],
    objective: Objective,
    *,
    hidden: _HiddenBench | None = None,
) -> Position | None:
    """Resolves the chosen actions and samples one outcome, all through the port.

    The weights come first and are sampled here, so the generator draws once a turn as it
    always has; then the drawn branch, or the drawn pause, is asked for. A pause is a
    decision node (`_do_self_switch_node`) and is carried on with the port's own
    continuation (IKA-209: it used to be resolved again in Python, whose continuation
    could not cross the process boundary).
    """
    weights = port.weights(reg, pos, chosen, Budget.exact())
    record.unmodelled.extend(weights.unmodelled)
    counts = np.array(weights.branches + weights.suspended, dtype=np.float64)
    if not counts.size or float(counts.sum()) <= 0:
        return None
    index = _sample_index(rng, counts)
    if index < len(weights.branches):
        return port.branch(reg, pos, chosen, Budget.exact(), index)
    paused = port.turn(reg, pos, chosen, Budget.exact(), select=index).pause
    if paused is None:
        raise port.PortRefused(f"the port gave no pause at index {index} of the turn")
    return _advance(
        reg, rng, None, record, leaves, objective, first_pause=paused, hidden=hidden
    )


@dataclass(frozen=True)
class _HiddenBench:
    """What a mid-turn decision needs to price the opponent's bench it has not seen.

    The same four things `_do_replacement_node` is handed, as `play_game` carries them at
    the move node the turn started from: `seen` is identities (IKA-117) and is turned into
    the pause's own slots there, `leads` the turn-1 pair per side (IKA-118).
    """

    sheets: tuple[Sequence[SampledSet], Sequence[SampledSet]]
    seen: list[frozenset[str]]
    bench_prior: tuple[BenchPrior | None, BenchPrior | None] | None
    leads: list[frozenset[str] | None]


def _advance(
    reg: Regulation,
    rng: np.random.Generator,
    result: PortTurn | None,
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None],
    objective: Objective,
    first_pause: PortPause | None = None,
    hidden: _HiddenBench | None = None,
) -> Position | None:
    """Samples one outcome of a resolved turn, answering any mid-turn request on the way.

    Returns ``None`` when the turn produced nothing to continue from, which the caller
    treats as the end of the game.

    `result` is a full port turn (`PortTurn.outcomes` and `pauses`). ``first_pause`` is
    for a caller that has already drawn a pause -- `_advance_turn` samples the weights
    itself, so the generator is used exactly once per turn whichever way the turn goes --
    and `result` is then None.
    """
    for attempt in range(5):
        if attempt == 0 and first_pause is not None:
            pause = first_pause
        else:
            assert result is not None and result.outcomes is not None
            assert result.pauses is not None
            weights = np.array(result.branches + result.suspended, dtype=np.float64)
            if not weights.size or float(weights.sum()) <= 0:
                return None
            index = _sample_index(rng, weights)
            if index < len(result.branches):
                return result.outcomes[index].position
            pause = result.pauses[index - len(result.branches)]
        resumed = _do_self_switch_node(
            reg, pause, record, leaves, objective, hidden=hidden
        )
        if resumed is None:
            return None
        result = resumed
    record.unmodelled.append("more than five mid-turn replacements in one turn")
    return None


def _do_self_switch_node(
    reg: Regulation,
    pause: PortPause,
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None],
    objective: Objective,
    *,
    hidden: _HiddenBench | None = None,
    definition: bool = False,
) -> PortTurn | None:
    """Chooses the replacement a self-switching move demanded, and finishes the turn.

    Unlike the post-turn replacement phase this is not simultaneous: one side is asked and
    the other is on `wait`, so there is no matrix to solve. The chooser takes the option it
    values most, scored with its own value function.

    The approximation left here is an information one, and it is the same one the depth-1
    search makes everywhere: the rest of the turn is already committed in this line, so the
    choice is made against a known continuation where a real player would only have seen
    the turn up to the interrupt. It is reported rather than papered over.

    With `hidden` the opponent's unseen slots are not read (IKA-120). Every option used to
    be scored on the true pause, so the rest of the turn and its leaves held the
    opponent's real back two -- 52.9% of the self-switch decisions in `data/ika73/w12` had
    an unseen Pokemon on the other side. Now each option is scored in every completion of
    that bench, the pause rebuilt in it by `paused_in`, and the scores are averaged with
    the belief's weights. Only one side chooses, so there is no game to solve: the choice
    is the option with the best expected score -- by definition each completion scored
    plainly and then weighted. The turn that is then *played* is the true one, resumed
    with that choice.

    The definition is still what is computed, but not by resolving every completion
    (IKA-150: 680 ms a decision, 19% of generation's worker wall clock at 4% of its
    decisions). An option whose rest of the turn never reaches the opponent's unseen slots
    resolves to the same branches in every completion, and its leaves there are the true
    leaves with those slots' Pokemon swapped in -- see `_shared_self_switch_plans`, which
    also says what "never reaches" is checked against. Only the options that do reach them
    are resolved per completion. `definition=True` takes the per-completion path for every
    option; the tests and `scratchpad/ika150_selfswitch_replay.py` hold the two equal.

    A learned leaf or a ported objective takes `_self_switch_encoded` instead (IKA-209):
    the same plans, the same rows in the same order and the same fold, with the leaves
    crossing as the encoder's arrays, or as one value each, rather than as positions. This
    road is for an evaluator the port can neither encode for nor score.
    """
    done = _self_switch_encoded(
        reg, pause, record, leaves, objective, hidden=hidden, definition=definition
    )
    if done is not _BY_POSITIONS:
        return done
    with timing.stage("selfswitch.resume"):
        chooser, alternatives = port.resume_alternatives(reg, pause)
    if chooser is None or not alternatives:
        return None
    record.unmodelled.append(
        "mid-turn replacement chosen against the opponent's already-committed action"
    )
    options = [option.to_choice() for option, _resumed in alternatives]
    timing.count("selfswitch.options", len(options))

    try:
        spread = _self_switch_spread(reg, pause, chooser, record, hidden)
    except ValueError:
        return None

    # One list of plans per world the chooser cannot tell apart, and that world's weight.
    if spread is None:
        weights = [1.0]
        with timing.stage("selfswitch.leaves"):
            plans = [[port.turn_leaves(reg, resumed) for _option, resumed in alternatives]]
    else:
        timing.count("selfswitch.completions", len(spread))
        weights = [item.weight for item in spread]
        plans = _self_switch_plans(
            reg, pause, chooser, options, alternatives, spread, definition=definition
        )
    flat = [position for world in plans for plan in world for position in plan.positions]
    if not flat:
        return None
    timing.count("selfswitch.leaves", len(flat))
    evaluate = leaves[chooser]
    values = (
        evaluate(flat)
        if evaluate is not None
        else np.array([objective(position) for position in flat], dtype=np.float64)
    )
    with timing.stage("selfswitch.fold"):
        scores = np.zeros(len(alternatives), dtype=np.float64)
        offset = 0
        for weight, world in zip(weights, plans, strict=True):
            for k, plan in enumerate(world):
                count = len(plan.positions)
                scores[k] += weight * plan.value(values[offset : offset + count])
                record.unmodelled.extend(plan.unmodelled)
                offset += count

        best = _record_self_switch(record, pause, chooser, options, scores, hidden)
    return alternatives[best][1]


#: `_self_switch_encoded` saying "not mine": the chooser's leaf has no encoded form.
_BY_POSITIONS = object()


def _encoded_scoring(reg: Regulation, evaluate: Any) -> tuple[Any, Any, Any] | None:  # noqa: ANN401
    """(scorer, encoder, rules) for a leaf with an encoded form, else None."""
    if evaluate is None:
        return None
    owner = getattr(evaluate, "__self__", evaluate)
    scorer = getattr(owner, "from_encoded", None)
    if scorer is None:
        return None
    from .beliefnode import _encoder_for
    from .encode import rules_of

    encoder = (
        getattr(owner, "encoder", None)
        or getattr(getattr(scorer, "__self__", None), "encoder", None)
        or _encoder_for(reg)
    )
    return scorer, encoder, rules_of(evaluate)


def _self_switch_spread(
    reg: Regulation,
    pause: PortPause,
    chooser: int,
    record: GameRecord,
    hidden: _HiddenBench | None,
) -> list | None:
    """The completions of the other side's unseen bench, or None when nothing is hidden.
    Raises ValueError (after noting it) when the sheet cannot explain the board."""
    if hidden is None:
        return None
    other = 1 - chooser
    with timing.stage("selfswitch.complete"):
        carried = seen_identities(pause.position, other, hidden.seen[other])
        shown = seen_slots(pause.position, other, carried)
        try:
            spread = completions(
                reg, pause.position, other, hidden.sheets[other], seen=shown,
                weights=_bench_weights(
                    hidden.bench_prior, other, pause.position, shown, record,
                    hidden.leads[other],
                ),
            )
        except ValueError as problem:
            record.unmodelled.append(f"hidden bench at a self-switch: {problem}")
            raise
    if len(spread) == 1 and spread[0].exact:
        return None
    return spread


def _self_switch_encoded(
    reg: Regulation,
    pause: PortPause,
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None],
    objective: Objective,
    *,
    hidden: _HiddenBench | None,
    definition: bool,
) -> PortTurn | None | object:
    """`_do_self_switch_node` with the leaves encoded over there (IKA-209).

    The same plans in the same order as the positions road: per world, per option, each
    plan's rows as `turn_leaves` lays them out, a shared option's rows the true pause's
    with the completion's bench patched in (`beliefnode._patched`, which the positions
    road does to the positions with `_with_bench`), and the whole node scored in one
    call of the leaf, chunked where `BatchedValue` chunks. Only what crosses changed:
    arrays instead of every leaf as JSON, which was 1.5x a value-net generation run.

    With no leaf and a ported objective (hp-share, faints) the port scores each leaf and
    only the values cross. Nothing is shared then: a shared option's value would need
    the patched leaf, which is a position, so every completion is resolved over there --
    the definition, whose leaves the shared road equals (tests/test_hidden_selfswitch.py).
    """
    from .beliefnode import _patched, _stacked
    from .fold import fold_value

    with timing.stage("selfswitch.resume"):
        probe = port.alternatives_encoded(reg, pause, want=[])
    chooser = probe.chooser
    if chooser is None or not probe.options:
        return None
    road = _encoded_scoring(reg, leaves[chooser])
    named = None
    if road is None:
        name = getattr(objective, "name", None)
        if leaves[chooser] is not None or name not in port.PORTED_OBJECTIVES:
            return _BY_POSITIONS
        named = name
        scorer = encoder = rules = None
    else:
        scorer, encoder, rules = road
    # What each request asks for: arrays for the leaf, or the objective's values alone.
    asked: dict[str, Any] = (
        {"rules": rules} if named is None else {"objectives": [named], "encode": False}
    )
    record.unmodelled.append(
        "mid-turn replacement chosen against the opponent's already-committed action"
    )
    options = [option.to_choice() for option in probe.options]
    timing.count("selfswitch.options", len(options))
    try:
        spread = _self_switch_spread(reg, pause, chooser, record, hidden)
    except ValueError:
        return None
    other = 1 - chooser

    parts: list[tuple[int, Any]] = []

    def block(answer: Any, plan: Any, patch: Any = None) -> int:  # noqa: ANN401
        def make() -> Any:  # noqa: ANN401
            if named is not None:
                return answer.values[named][plan.start : plan.start + plan.count]
            rows = _rows(answer.node.encoded, plan.start, plan.count)
            return rows if patch is None else patch(rows)

        parts.append((plan.count, make))
        return len(parts) - 1

    layout: list[list[tuple[Any, int]]] = []
    answers: list[Any] = []
    with timing.stage("selfswitch.leaves"):
        if spread is None:
            weights = [1.0]
            true = port.alternatives_encoded(reg, pause, **asked)
            answers.append(true)
            layout.append([(plan, block(true, plan)) for plan in true.plans])
        else:
            timing.count("selfswitch.completions", len(spread))
            weights = [item.weight for item in spread]
            slots = spread[0].slots
            shared = [False] * len(options)
            true = None
            if not definition and named is None:
                true = port.alternatives_encoded(
                    reg, pause, shared=(other, slots), rules=rules
                )
                answers.append(true)
                shared = [bool(flag) for flag in true.untouched]
            timing.count("selfswitch.shared", sum(shared))
            for item in spread:
                found = None
                if not all(shared):
                    found = port.alternatives_encoded(
                        reg, pause, world=(item.position, other),
                        want=[k for k, flag in enumerate(shared) if not flag], **asked,
                    )
                    answers.append(found)
                    timing.count("selfswitch.resumed", len(found.options))
                    found_options = [option.to_choice() for option in found.options]
                    if found.chooser != chooser or found_options != options:
                        raise AssertionError(
                            f"a completion of side {other}'s bench changed side {chooser}'s "
                            f"self-switch options: {options} -> {found_options}"
                        )
                world: list[tuple[Any, int]] = []
                for k, flag in enumerate(shared):
                    if flag:
                        plan = true.plans[k]

                        def patch(rows: Any, item: Any = item) -> Any:  # noqa: ANN401
                            return _patched(
                                rows, other, slots, item, reg, pause.position, encoder, rules
                            )

                        world.append((plan, block(true, plan, patch)))
                    else:
                        plan = found.plans[k]
                        world.append((plan, block(found, plan)))
                layout.append(world)
    total = sum(rows for rows, _make in parts)
    if not total:
        return None
    timing.count("selfswitch.leaves", total)
    if named is not None:
        pieces = [make() for _rows_count, make in parts]
        starts = list(np.cumsum([0] + [len(piece) for piece in pieces])[:-1])
        values = np.concatenate(pieces).astype(np.float64)
    else:
        for answer in answers:
            port.note_port_rule([scorer], answer.node)
        stacked, starts = _stacked(parts, answers[0].node.encoded)
        values = np.asarray(scorer(stacked), dtype=np.float64)
        del stacked
    with timing.stage("selfswitch.fold"):
        scores = np.zeros(len(options), dtype=np.float64)
        for weight, world in zip(weights, layout, strict=True):
            for k, (plan, index) in enumerate(world):
                start = starts[index]
                scores[k] += weight * fold_value(
                    port.fold_from_json(plan.fold), values[start : start + plan.count]
                )
                record.unmodelled.extend(plan.unmodelled)
        best = _record_self_switch(record, pause, chooser, options, scores, hidden)
    from .actions import PassAction

    passes = SideAction(
        slots=tuple(
            PassAction(slot=i) for i in range(len(pause.position.sides[other].active))
        )
    )
    choice = probe.options[best]
    return port.resume(reg, pause, [choice, passes] if chooser == 0 else [passes, choice])


def _rows(encoded: Any, start: int, count: int) -> Any:  # noqa: ANN401
    """Rows `start:start+count` of an `Encoded`, as views."""
    from .beliefnode import _ARRAYS
    from .encode import Encoded

    return Encoded(
        **{name: getattr(encoded, name)[start : start + count] for name in _ARRAYS},
        unknown_volatiles=dict(encoded.unknown_volatiles),
    )


def _record_self_switch(
    record: GameRecord,
    pause: PortPause,
    chooser: int,
    options: list[str],
    scores: np.ndarray,
    hidden: _HiddenBench | None,
) -> int:
    """The chooser's best option, and the decision written down."""
    # Side 0 is the maximiser the payoff matrices are written for.
    best = int(np.argmax(scores)) if chooser == 0 else int(np.argmin(scores))
    policy = [0.0] * len(options)
    policy[best] = 1.0
    waiting = ["pass"]
    record.decisions.append(
        Decision(
            turn=pause.position.turn,
            kind="selfswitch",
            position=pause.position.to_json(),
            own_actions=options if chooser == 0 else waiting,
            own_policy=policy if chooser == 0 else [1.0],
            foe_actions=waiting if chooser == 0 else options,
            foe_policy=[1.0] if chooser == 0 else policy,
            search_value=float(scores[best]),
            own_chosen=options[best] if chooser == 0 else waiting[0],
            foe_chosen=waiting[0] if chooser == 0 else options[best],
            # Read for the record alone: the choice above used only `other`'s.
            shown=_shown_record(
                pause.position,
                None
                if hidden is None
                else [
                    seen_identities(pause.position, i, hidden.seen[i]) for i in (0, 1)
                ],
            ),
        )
    )
    timing.decided("selfswitch")
    return best


def _self_switch_plans(
    reg: Regulation,
    pause: PortPause,
    chooser: int,
    options: list[str],
    alternatives: list[tuple[SideAction, PortTurn]],
    spread: Sequence[Any],
    *,
    definition: bool = False,
) -> list[list[TurnLeaves]]:
    """Every completion's plan per option: `[completion][option]`, in `spread`'s order.

    By definition each completion's pause is rebuilt (the port's `paused_in`), every option
    resumed in it and flattened (`port.turn_leaves`). That is still done for an option the shared path
    cannot vouch for; the others are the true pause's plan with the bench swapped in.
    """
    other = 1 - chooser
    shared: list[TurnLeaves | None] = [None] * len(alternatives)
    if not definition:
        with timing.stage("selfswitch.leaves"):
            shared = _shared_self_switch_plans(
                reg, pause, other, alternatives, spread[0].slots
            )
    timing.count("selfswitch.shared", sum(plan is not None for plan in shared))
    per_world: list[list[TurnLeaves]] = []
    for item in spread:
        found: list[tuple[SideAction, PortTurn]] | None = None
        if any(plan is None for plan in shared):
            # The pause rebuilt in this completion, over there (`paused_in`).
            with timing.stage("selfswitch.resume"):
                seen_as, found = port.resume_alternatives(
                    reg, pause, world=(item.position, other)
                )
            timing.count("selfswitch.resumed", len(found))
            found_options = [option.to_choice() for option, _resumed in found]
            if seen_as != chooser or found_options != options:
                # The chooser's options are its own bench, which no completion of the
                # other side's touches. If that ever stops being true, the average
                # would be over different decisions.
                raise AssertionError(
                    f"a completion of side {other}'s bench changed side {chooser}'s "
                    f"self-switch options: {options} -> {found_options}"
                )
        world: list[TurnLeaves] = []
        with timing.stage("selfswitch.leaves"):
            for k, plan in enumerate(shared):
                if plan is not None:
                    world.append(_with_bench(plan, other, item))
                else:
                    assert found is not None
                    world.append(port.turn_leaves(reg, found[k][1]))
        per_world.append(world)
    return per_world


def _shared_self_switch_plans(
    reg: Regulation,
    pause: PortPause,
    other: int,
    alternatives: list[tuple[SideAction, PortTurn]],
    slots: tuple[int, ...],
) -> list[TurnLeaves | None]:
    """Each option's true plan where it holds in every completion, else None.

    An option's rest of the turn is the same in every completion of `other`'s unseen
    `slots` when none of those Pokemon is put on the field before the turn ends. Checked,
    per option, against what can put one there:

    * `other` has a switch still queued (`paused_in` re-aims it by slot, and it may be
      aimed at one of them) -- then no option is shared;
    * the rest of the turn pauses again (the next chooser's options are a bench; a nested
      `other` pause would offer the unseen slots themselves);
    * in some leaf an unseen slot's Pokemon is not the one the pause held, at the same
      party index, off the field -- anything the turn did to one of them.

    What the resolver reads of a bench it does not field is how many are left standing
    and their species in the branch-merge bucket (`_position_bucket`). An unseen Pokemon
    is healthy and unrevealed in every completion, and within one completion every branch
    carries the same pair, so neither reading depends on which pair it is. A mechanic that read more of
    a benched Pokemon (Beat Up, Illusion) would have to be added to the checks above; the
    resolver models none. `tests/test_hidden_selfswitch.py` holds this to the definition.
    """
    before = pause.position.sides[other].pokemon
    held = {index: mon for index, mon in enumerate(before) if mon.slot in slots}
    if port.remaining_switches(pause, other):
        return [None] * len(alternatives)
    out: list[TurnLeaves | None] = []
    for _option, resumed in alternatives:
        if resumed.suspended:
            out.append(None)
            continue
        plan = port.turn_leaves(reg, resumed)
        untouched = all(
            leaf.sides[other].pokemon[index] == mon
            and leaf.sides[other].pokemon[index].active_index is None
            for leaf in plan.positions
            for index, mon in held.items()
        )
        out.append(plan if untouched else None)
    return out


def _with_bench(plan: TurnLeaves, other: int, item: Any) -> TurnLeaves:
    """`plan` in the completion `item`: its unseen slots' Pokemon and mega list swapped in.

    What resolving the turn in `item` would have left there: the completion's own Pokemon,
    untouched (they took no part), and its side's `mega_capable_slots` (`substitute`
    recomputes it and the resolver never writes it).

    Shallow: everything but the new side and its list is the true leaf's own objects, and
    the completion's Pokemon are shared between leaves. These positions only ever go to the
    leaf evaluator, which reads them, and are dropped when the node returns; a full copy
    was 0.2 ms a leaf, 0.6 s on a 3,072-leaf decision.
    """
    completed = item.position.sides[other]
    fresh = {mon.slot: mon for mon in completed.pokemon if mon.slot in item.slots}
    capable = list(completed.mega_capable_slots)
    positions = []
    for leaf in plan.positions:
        sides = list(leaf.sides)
        sides[other] = replace(
            sides[other],
            pokemon=[fresh.get(mon.slot, mon) for mon in sides[other].pokemon],
            mega_capable_slots=capable,
        )
        positions.append(replace(leaf, sides=sides))
    return TurnLeaves(positions=positions, root=plan.root, unmodelled=plan.unmodelled)


def _do_replacement_node(
    reg: Regulation,
    rng: np.random.Generator,
    pos: Position,
    owed: tuple[tuple[bool, ...], tuple[bool, ...]],
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None] = (None, None),
    *,
    sheets: tuple[Sequence[SampledSet], Sequence[SampledSet]] | None = None,
    shown: list[frozenset[int]] | None = None,
    bench_prior: tuple[BenchPrior | None, BenchPrior | None] | None = None,
    leads: list[frozenset[str] | None] | None = None,
    recorded_shown: list[list[str]] | None = None,
) -> Position:
    """Solves and applies the replacement phase.

    It is a simultaneous-move node like any other -- neither player sees the other's
    replacement -- so it gets a matrix and an equilibrium rather than a heuristic pick.

    With `sheets` it is also the node where the *reveal* happens: the Pokemon coming in is
    the one that stops being hidden, and choosing what to send against an opponent whose
    bench is unknown is the same Bayesian game the move nodes solve. `shown` is each
    side's seen slots in *this* position -- `seen_slots` of the identities `play_game`
    carries -- and never a set of numbers carried from an earlier one (IKA-117). `leads`
    is the turn-1 lead pair per side, as `play_game` carries it (IKA-118).
    `recorded_shown` is `Decision.shown`, written as given and read by nothing here.

    Without them each side now also gets its own matrix. It did not: both strategies came
    off `matrix(pos, leaves[0])`, and this docstring called that a wart and left it,
    "because changing it would move every open-information number". Every one of those
    numbers was a comparison in which the column player's replacements were chosen by its
    opponent's evaluator, for about a third of the decisions in a game, and the seat swap
    moves that from one arm to the other rather than cancelling it.

    The options are built once from the true position because they are the same in every
    completion: a replacement names a *slot*, and which slots are free is public even when
    who is standing in them is not.
    """
    options: list[list[SideAction]] = []
    for side_index in range(2):
        must = list(owed[side_index])
        found = (
            switch_actions_after_faint(reg, pos, side_index, must) if any(must) else []
        )
        if not found:
            found = [
                SideAction(
                    slots=tuple(
                        _pass(slot) for slot in range(len(pos.sides[side_index].active))
                    )
                )
            ]
        options.append(found)

    def matrix(at: Position, evaluate: LeafEvaluator | None) -> np.ndarray:
        """Every pair of replacements resolved from `at` and scored.

        The replacement node used `hp-share` regardless of what the move nodes used, which
        would leave a third of a game's decisions scored by the thing being replaced.
        """
        resolved = [
            [port.resolve_replacements(reg, at, [a, b]).position for b in options[1]]
            for a in options[0]
        ]
        if evaluate is None:
            return np.array(
                [[HP_SHARE(after) for after in row] for row in resolved],
                dtype=np.float64,
            )
        flat = [after for row in resolved for after in row]
        return np.asarray(evaluate(flat), dtype=np.float64).reshape(
            len(options[0]), len(options[1])
        )

    foe_value: float | None = None
    if sheets is not None:
        seen = shown or [frozenset(), frozenset()]
        answers: dict[int, tuple[list[float], float]] = {}
        try:
            spreads = {
                side: completions(
                    reg, pos, side, sheets[side], seen=seen[side],
                    weights=_bench_weights(
                        bench_prior, side, pos, seen[side], record,
                        (leads or [None, None])[side],
                    ),
                )
                for side in (0, 1)
            }
        except ValueError as problem:
            record.unmodelled.append(f"hidden bench at a replacement: {problem}")
            spreads = {}
        if spreads:
            from .equilibrium import solve_bayesian

            for side in (0, 1):
                items = spreads[1 - side]
                leaf = leaves[side]
                built = [matrix(item.position, leaf) for item in items]
                weights = np.asarray([item.weight for item in items], dtype=np.float64)
                # Side 1 minimises what side 0 maximises, so its game is the transpose of
                # the negation -- the same turn read from the other end, as in the move
                # node, rather than a second matrix that could drift from this one.
                solved = solve_bayesian(
                    [m if side == 0 else -m.T for m in built], weights
                )
                answers[side] = (
                    [float(x) for x in solved.row_strategy],
                    float(solved.value),
                )
            own_policy, value = answers[0]
            foe_policy = answers[1][0]
            # Side 1's game is the negated transpose: back into side 0's units.
            foe_value = -answers[1][1]
        else:
            own_policy = [1.0 / len(options[0])] * len(options[0])
            foe_policy = [1.0 / len(options[1])] * len(options[1])
            value = 0.5
    else:
        # One matrix per side, as the hidden-bench branch above already does. This read
        # `matrix(pos, leaves[0])` and took BOTH strategies off it, so with different
        # leaves the column player's replacement was chosen by the row player's
        # evaluator. The docstring called it a wart and deferred it because changing it
        # "would move every open-information number" -- which is the reason to change it:
        # a replacement is about a third of a game's decisions, and the seat swap moves
        # the corruption from one arm to the other rather than cancelling it.
        payoff = matrix(pos, leaves[0])
        try:
            equilibrium = solve(payoff)
            own_policy = [float(x) for x in equilibrium.row_strategy]
            foe_policy = [float(x) for x in equilibrium.col_strategy]
            value = float(equilibrium.value)
        except EquilibriumError:
            own_policy = [1.0 / len(options[0])] * len(options[0])
            foe_policy = [1.0 / len(options[1])] * len(options[1])
            value = float(payoff.mean())
        if leaves[1] is not leaves[0]:
            foe_payoff = matrix(pos, leaves[1])
            try:
                foe_policy = [
                    float(x) for x in solve(foe_payoff).col_strategy
                ]
            except EquilibriumError:
                foe_policy = [1.0 / len(options[1])] * len(options[1])

    chosen = [
        options[0][_sample_index(rng, np.array(own_policy))],
        options[1][_sample_index(rng, np.array(foe_policy))],
    ]
    record.decisions.append(
        Decision(
            turn=pos.turn,
            kind="replacement",
            position=pos.to_json(),
            own_actions=[a.to_choice() for a in options[0]],
            own_policy=own_policy,
            foe_actions=[a.to_choice() for a in options[1]],
            foe_policy=foe_policy,
            search_value=value,
            own_chosen=chosen[0].to_choice(),
            foe_chosen=chosen[1].to_choice(),
            shown=recorded_shown,
            foe_search_value=foe_value,
        )
    )
    timing.decided("replacement")
    outcome = port.resolve_replacements(reg, pos, chosen, rng=rng)
    record.unmodelled.extend(outcome.unmodelled)
    return outcome.position


def _pass(slot: int) -> Any:  # noqa: ANN401
    from .actions import PassAction

    return PassAction(slot=slot)


def _set_json(reg: Regulation, entry: SampledSet) -> dict[str, Any]:
    del reg
    return {
        "species": entry.species,
        "ability": entry.ability,
        "item": entry.item,
        "nature": entry.nature,
        "moves": list(entry.moves),
        "sp": {k: v for k, v in entry.sp.items() if v},
    }


def cluster_labels(
    reg: Regulation, standings: Standings, which: str = "all"
) -> dict[str, str]:
    """player -> a readable archetype label, so per-archetype win rates stay reportable.

    The labels are for reading the results, not for sampling: real teams are sampled
    directly, and a cluster is only a name to group them under afterwards. A team that
    clusters with nobody is labelled as such rather than given a name of its own, so the
    long tail does not turn into 59 archetypes with three games each.
    """
    out: dict[str, str] = {}
    for group in cluster_teams(standings.pool(which)):
        label = label_for(group, reg) if len(group) > 1 else "unclustered"
        for team in group:
            out[team.player] = label
    return out


def generate(
    reg: Regulation,
    prior: MetagamePrior,
    roster: Roster,
    archetypes: list[Archetype],
    *,
    games: int,
    seed: int = 0,
    out: Path | None = None,
    objective: Objective = HP_SHARE,
    search_limit: int | tuple[int, int] = SEARCH_LIMIT,
    max_turns: int = MAX_TURNS,
    leaf: str | None = None,
    cooc: Cooccurrence | None = None,
    archetype_share: float = 0.0,
    standings: Standings | None = None,
    standings_pool: str = "all",
    evaluate: LeafEvaluator | None = None,
    book: SelectionBook | None = None,
    explore_epsilon: float = DEFAULT_EPSILON,
    explore_temperature: float = DEFAULT_TEMPERATURE,
    mirror_share: float = 0.0,
    depth: int | tuple[int, int] = 1,
    rank_by_leaf: bool = False,
    policy: Any = None,
    solve_sparsely: bool = False,
    solve_restricted: bool = False,
    force_lead: tuple[str, ...] | None = None,
    hide_bench: bool | None = None,
    indices: Iterable[int] | None = None,
    on_finish: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Plays games and appends one JSON line per finished game.

    Our six are fixed (this is a team-conditioned value function) and both sides pick four
    of six uniformly. The opponent comes from one of two pools:

    - the **measured metagame**, when ``cooc`` is given: six species drawn from the
      co-occurrence counts in the usage data, which is what a real opponent looks like at
      the rate real opponents look like it;
    - a **cited archetype**, for an ``archetype_share`` of the games: a real composition
      from a tournament report, which the metagame pool covers only by chance.

    ...and a third that supersedes both when it is available: ``standings``, a real
    tournament field. 394 entries from Worlds 2026 carry the actual ability, nature, item
    and moves, so drawing one uniformly is the tournament metagame with no archetype
    abstraction and no pairwise model. Only the SP spread is filled from usage, which
    matches the information a player really has: open team sheets show the nature and blank
    the investment. Games record which pool they came from in ``foeArchetype``.

    ``book`` replaces the uniform 4-of-6 draw with the solved selection equilibrium, for
    *both* sides. It is looked up by the opponent's team sheet, so nothing about the four
    they bring reaches our draw -- :mod:`pokeuraou.selection_book` lists the four ways that
    could have gone wrong. The opponent's spreads then come from the sampled class rather
    than from a fresh usage draw, because the column strategy the game uses is the strategy
    of a player holding exactly that investment. A team the book does not cover falls back
    to the uniform draw and is counted in ``book_misses``.

    ``mirror_share`` plays that fraction of games against **our own six, spreads
    included**. Two reasons, and the second was not obvious:

    - the mirror is where the only *external* knowledge about this team lives (a reported
      three-way cycle among three selections, in ``configs/knowledge/``), and until now
      self-play had never played one -- every opponent came from the tournament field, so
      every mirror judgement the value function makes is extrapolation. Checking a model
      against human knowledge on positions it was never trained on measures extrapolation
      rather than the model;
    - a true mirror is exactly antisymmetric, so its win rate **must** come out at 50%.
      That makes a mirror share a free calibration assertion on the whole pipeline --
      search, resolver, evaluator -- reported in ``mirror_wins`` / ``mirror_games``. A
      speed-tie bias in the search matrix was once caught by precisely this kind of sum.

    Mirror games draw their selection uniformly: the book is keyed on tournament sheets and
    has no entry for ourselves, and a uniform draw is what covers the 90 anyway.

    ``indices`` plays the games with those numbers instead of ``games`` in a row, which is
    how a worker draws from a shared queue. It changes where the randomness comes from, and
    it has to: a run streamed one generator through a whole block, so the seventeenth game
    was whatever the sixteen before it left behind. That is fine when the block is fixed
    and meaningless when the order is a race. With ``indices`` each game is seeded from its
    own number, so a game is the same game whichever worker draws it, in whatever order,
    and a game replayed after a worker died is the game that was lost.

    ``on_finish`` is called with an index once its game has been *written*, not once it has
    been played. The gap is the point: a worker that dies in between should have that game
    handed to somebody else.

    ``hide_bench`` has no default (IKA-123). True is what ships: both searches solve over
    the sixes, never the opponent's four. False is the open game, which is what every pool
    before the hidden bench was generated in and what this argument silently meant when
    left out -- so leaving it out now stops instead of making open teacher data.
    """
    if not 0.0 <= mirror_share <= 1.0:
        raise ValueError(f"mirror_share must be a probability, got {mirror_share}")
    if book is not None and standings is None:
        raise ValueError(
            "a selection book is keyed on tournament team sheets, so it needs the "
            "standings pool it was solved against"
        )
    if hide_bench is None:
        raise ValueError(
            "hide_bench must be given: True hides the opponent's bench (what ships), False "
            "plays the open game (reference). It defaulted to False before IKA-123."
        )
    if standings is not None:
        field = cluster_labels(reg, standings, standings_pool)
    else:
        field = {}
        # An all-mirror run needs no opponent pool at all: the opponent is us.
        if mirror_share < 1.0:
            if cooc is None and archetype_share < 1.0:
                archetype_share = 1.0
            if not archetypes and archetype_share > 0.0:
                raise ValueError("archetype_share > 0 but no archetypes were given")
    def scheduled() -> Iterator[tuple[int | None, np.random.Generator]]:
        if indices is None:
            rng = np.random.default_rng(seed)
            for _ in range(games):
                yield None, rng
        else:
            for index in indices:
                yield index, np.random.default_rng([seed, index])

    path = out or (selfplay_dir() / f"games-seed{seed}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    lead_wanted = tuple(force_lead) if force_lead else ()
    stats = {
        "games": 0,
        "book_hits": 0,
        "book_misses": 0,
        "mirror_games": 0,
        "mirror_wins": 0,
        "finished": 0,
        "discarded_unfinished": 0,
        "wins": 0,
        "decisions": 0,
        "turns": 0,
    }
    with path.open("a", encoding="utf-8") as handle:
        for index, rng in scheduled():
            drawn = None
            # Bound here because only the standings branch assigns it, and the bench
            # prior below reads it. It currently survives on `drawn is not None`
            # short-circuiting first, which is a name binding resting on the order of an
            # `and` -- true today and not a thing to rely on.
            entry = None
            forced_uniform = False
            mirror = mirror_share > 0.0 and rng.random() < mirror_share
            if mirror:
                # Our own six, the same spreads, no sampling: an approximate mirror would
                # not be antisymmetric and would lose the 50% assertion that is the point.
                label = "mirror"
                foe_six = list(roster.sets)
            elif standings is not None:
                pool = standings.pool(standings_pool)
                team = pool[int(rng.integers(len(pool)))]
                label = field.get(team.player, "worlds")
                entry = book.get(team) if book is not None else None
                if entry is None:
                    if book is not None:
                        stats["book_misses"] += 1
                    foe_six = sample_standings_team(rng, reg, prior, team)
                else:
                    stats["book_hits"] += 1
                    drawn = entry.draw(
                        rng,
                        epsilon=explore_epsilon,
                        temperature=explore_temperature,
                    )
                    foe_six = list(drawn.foe_six)
            elif cooc is not None and rng.random() >= archetype_share:
                label = "metagame"
                foe_six = sample_metagame_team(rng, reg, prior, cooc)
            else:
                archetype = archetypes[int(rng.integers(len(archetypes)))]
                label = archetype.id
                foe_six = sample_archetype(rng, reg, prior, archetype)
            if drawn is not None:
                own_pick = drawn.our_pick
                foe_pick = drawn.foe_pick
                if lead_wanted:
                    # Our lead, forced; everything else left as the book drew it. The
                    # opponent, the spreads and our other two are untouched, so the games
                    # differ from ordinary generation in one thing only.
                    own_pick = _with_lead(rng, roster, lead_wanted, own_pick)
            else:
                own_pick = pick_four_indices(
                    rng, len(roster.sets), size=reg.meta.picked_team_size
                )
                if lead_wanted:
                    own_pick = _with_lead(rng, roster, lead_wanted, own_pick)
                    # The uniform branch defaults to "uniform", which is as wrong here as
                    # "book" is above: two of the four were chosen, not drawn.
                    forced_uniform = True
                foe_pick = pick_four_indices(
                    rng, len(foe_six), size=reg.meta.picked_team_size
                )
            own_four = [roster.sets[i] for i in own_pick]
            foe_four = [foe_six[i] for i in foe_pick]

            record = play_game(
                reg, rng, own_four, foe_four, label,
                objective=objective, search_limit=search_limit, max_turns=max_turns,
                evaluate=evaluate,
                depth=depth,
                rank_by_leaf=rank_by_leaf,
                policy=policy,
                solve_sparsely=solve_sparsely,
                solve_restricted=solve_restricted,
                # Both sixes, so neither search is shown the other's unplayed bench.
                sheets=(list(roster.sets), list(foe_six)) if hide_bench else None,
                open_information=not hide_bench,
                # And what each side would have brought, so the belief over the bench is
                # the opponent's own selection equilibrium rather than a uniform draw
                # over every pair the sheet allows. Only when the book named this
                # opponent: without an entry there is nothing to condition on and
                # uniform is the honest prior, which is what `None` selects.
                # The same epsilon and temperature the draw above used, because the
                # belief has to be about the opponent this game actually has.
                bench_prior=(
                    (
                        BenchPrior.of(
                            entry, 0, [s.species for s in roster.sets],
                            epsilon=explore_epsilon, temperature=explore_temperature,
                        ),
                        BenchPrior.of(
                            entry, 1, [s.species for s in foe_six],
                            epsilon=explore_epsilon, temperature=explore_temperature,
                        ),
                    )
                    if hide_bench and drawn is not None and entry is not None
                    else None
                ),
                selection=(
                    [entry.species for entry in roster.sets],
                    [entry.species for entry in foe_six],
                    own_pick,
                    foe_pick,
                ),
            )
            if drawn is not None:
                # "book" means the book drew this four. A forced lead overrides the two
                # slots the book cared most about, so it did not -- and a pool built that
                # way would otherwise describe itself as book-selected in every summary a
                # later decision about including it would read. The mixtures below are
                # still the book's and are still worth keeping: they say what the book
                # would have played, which is the comparison the forcing exists to make.
                record.selection_source = "forced-lead" if lead_wanted else "book"
                record.own_selection_policy = [
                    float(x) for x in drawn.our_equilibrium
                ]
                record.foe_selection_policy = [
                    float(x) for x in drawn.foe_equilibrium
                ]
                record.own_selection_mixture = [float(x) for x in drawn.our_mixture]
                record.foe_selection_mixture = [float(x) for x in drawn.foe_mixture]
                record.selection_value = drawn.value
            elif forced_uniform:
                record.selection_source = "forced-lead"
            stats["games"] += 1
            if mirror:
                stats["mirror_games"] += 1
                if record.outcome is not None:
                    stats["mirror_wins"] += int(record.outcome > 0.5)
            if record.outcome is None:
                stats["discarded_unfinished"] += 1
            else:
                stats["finished"] += 1
                stats["wins"] += int(record.outcome > 0.5)
                stats["decisions"] += len(record.decisions)
                stats["turns"] += record.turns
                payload = record.to_json(
                    objective=leaf or objective.name, search_limit=search_limit
                )
                if index is not None:
                    # The queue's index, so two pools generated at one seed can be
                    # checked for pairing afterwards instead of assumed to pair.
                    #
                    # `data/selfplay-hidden2` and `data/selfplay-open2` were made at seed
                    # 2001 with one flag between them, and whether game k is the same
                    # matchup in both is exactly the question a twin design rests on --
                    # and it could not be asked, because a worker takes indices from the
                    # queue in whatever order it gets them, so file order says nothing.
                    # The same gap was closed for matches earlier today; generation kept
                    # it. `--mirror-share`'s short-circuit already desynchronised
                    # generation 11h from generation 10 once, and that was found by
                    # noticing afterwards rather than by being able to check.
                    payload["gameIndex"] = index
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
            if index is not None and on_finish is not None:
                # After the write, and after a discard too: a game that was played and
                # came to nothing has still been played, and handing it back would have
                # the queue retry it until it ran out of attempts.
                on_finish(index)
    stats["path"] = str(path)
    return stats


__all__ = [
    "MAX_TURNS",
    "SEARCH_LIMIT",
    "END_REASONS",
    "Decision",
    "GameRecord",
    "final_position",
    "generate",
    "play_game",
    "position_from_sets",
    "positions_from_sets",
    "selfplay_dir",
]
