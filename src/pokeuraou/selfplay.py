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

IKA-463 removed this module's game loop (`play_game`): generation's and the board's games are
the Rust workers' (`rust/src/worker.rs`, `boardworker.rs`). What is left is the record
(`GameRecord`, `Decision`), the turn-1 position, and the nodes a person's game asks
(`humanplay`): the menus, the belief, the self-switch and the replacement.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import port, qrank, rank_scores, rustnode, timing
from .actions import (
    SideAction,
    switch_actions_after_faint,
)
from .budget import Budget
from .eqselect import DEFAULT_EQ_SELECT
from .fold import TurnLeaves
from .hidden import (
    DEFAULT_BENCH_DROP,
    completions,
    drop_light,
    identity,
    seen_identities,
    seen_slots,
    shown_species,
)
from .narrow import narrow, note_coverless
from .payoff import HP_SHARE, Objective
from .policy import policy_ranking
from .position import Field, MoveSlot, Pokemon, Position, Side
from .priors import SampledSet
from .provenance import (
    LEGACY_BENCH_DROP,
    LEGACY_DEEPEN,
    LEGACY_RANK_FILL,
    engine_fingerprint,
)
from .qrank import is_q
from .regulation import STAT_IDS, Regulation, repo_root
from .rustnode import PortPause, PortTurn, ResumedTurn
from .search import (
    believed_ranking,
    leaf_ranking,
    parse_rank_fill,
    rank_fill_covers,
    resolve_rank_fill,
)
from .selection_book import (
    BenchPrior,
)
from .stats import nature_multipliers, stats_from_sp

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


class WrittenDecision:
    """IKA-458 stage 3.6c: a `Decision` the port wrote as the record holds it (the keys of
    `GameRecord.to_json`'s ``decisions``). Python keeps it and writes it; the attributes the
    readers of a record in memory use (`kind`, `turn`, `own_chosen`, ...) are read from it."""

    __slots__ = ("json",)

    #: `Decision`'s field -> the record's key, for the fields not named the same way.
    _KEYS = {
        "own_actions": "ownActions", "own_policy": "ownPolicy", "foe_actions": "foeActions",
        "foe_policy": "foePolicy", "search_value": "searchValue", "own_chosen": "ownChosen",
        "foe_chosen": "foeChosen", "rank_views": "rankViews", "shown": "shownIdentities",
        "foe_search_value": "foeSearchValue", "deepened": "deepened",
    }

    def __init__(self, written: dict[str, Any]) -> None:
        self.json = written

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        key = self._KEYS.get(name, name)
        if key not in ("turn", "kind", "position") and key not in self._KEYS.values():
            raise AttributeError(name)
        return self.json.get(key)


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
    #: Each side's best-first deepening label (`deepen.parse_deepen`, IKA-33). Written
    #: only when a side deepened, so a record without it searched at depth 1.
    deepen: list[str] = field(default_factory=lambda: [LEGACY_DEEPEN, LEGACY_DEEPEN])
    #: Each side's search depth and whether its depth 2 read the restricted game (IKA-111).
    #: Written only when a side searched past depth 1, so a record without them is depth 1
    #: on both sides, as every generated game before IKA-111 was.
    depth: list[int] = field(default_factory=lambda: [1, 1])
    solve_restricted: list[bool] = field(default_factory=lambda: [False, False])
    #: Whether each side's matrix budget took the knock-out branch
    #: (`Budget.enumerate_knockouts`, IKA-359). Written only when a side did.
    knockouts: list[bool] = field(default_factory=lambda: [False, False])
    #: Each side's port executable when it was not the process's own (`play_game`'s
    #: ``rust_binary``, IKA-413): its fingerprint (path, size, sha256, build time), else
    #: None. Written only when a side had one.
    rust_binary: list[dict | None] = field(default_factory=lambda: [None, None])
    #: Which ally targets of `normal` moves each side's menus listed
    #: (`actions.ALLY_TARGET_MODES`, IKA-181), and per side the move decisions whose menu
    #: held one and those where the side played one. Written only when a side was not off.
    ally_targets: list[str] = field(default_factory=lambda: ["off", "off"])
    ally_menus: list[int] = field(default_factory=lambda: [0, 0])
    ally_played: list[int] = field(default_factory=lambda: [0, 0])
    #: Which point of its optimal set each side played (`eqselect.parse_eq_select`,
    #: IKA-196). Written only when a side played another than the LP's vertex.
    eq_select: list[str] = field(default_factory=lambda: [DEFAULT_EQ_SELECT, DEFAULT_EQ_SELECT])
    #: The species each side was told to Mega Evolve at the first turn it stands on the
    #: field (`play_game`'s ``force_mega``, IKA-419f), and what the forcing did per side:
    #: None until the turn came, then {"turn", "rule", "mass"}. Written only when a side
    #: had one.
    force_mega: list[str | None] = field(default_factory=lambda: [None, None])
    forced_mega: list[dict | None] = field(default_factory=lambda: [None, None])
    #: The species each side held Baton Pass back for (`play_game`'s ``hold_baton``, IKA-419g),
    #: and per side what the hold did over the game's move nodes: ``held`` (the species on the
    #: field with its Special Attack or Special Defense below +1), ``held_legal`` (of those,
    #: nodes where the unheld pool had a Baton Pass action), ``open`` (both at +1 or more),
    #: ``open_menu`` (of those, nodes whose menu listed one), ``leaks`` (held nodes whose menu
    #: or choice still had one: always 0). Written only when a side held one.
    hold_baton: list[str | None] = field(default_factory=lambda: [None, None])
    baton_hold: list[dict | None] = field(default_factory=lambda: [None, None])
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
            **({"knockouts": list(self.knockouts)} if any(self.knockouts) else {}),
            **({"rustBinary": list(self.rust_binary)} if any(self.rust_binary) else {}),
            **(
                {"allyTargets": list(self.ally_targets), "allyMenus": list(self.ally_menus),
                 "allyPlayed": list(self.ally_played)}
                if set(self.ally_targets) != {"off"}
                else {}
            ),
            **(
                {"eqSelect": list(self.eq_select)}
                if set(self.eq_select) != {DEFAULT_EQ_SELECT}
                else {}
            ),
            **(
                {"forceMega": list(self.force_mega), "forcedMega": list(self.forced_mega)}
                if any(self.force_mega)
                else {}
            ),
            **(
                {"holdBaton": list(self.hold_baton), "batonHold": list(self.baton_hold)}
                if any(self.hold_baton)
                else {}
            ),
            **(
                {"depth": list(self.depth), "solveRestricted": list(self.solve_restricted)}
                if set(self.depth) != {1}
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
                d.json if isinstance(d, WrittenDecision) else {
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
        # The set's own species id, as `position.ts` writes `baseSpecies` -- not the dex's
        # `baseSpecies`. For Floette-Eternal and Meowstic-F the dex says "Floette" and
        # "Meowstic", which is neither the sheet member nor its Mega, so once the Mega was on
        # the board the belief forgot the sheet member had been brought (IKA-411).
        base_species=entry.species,
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
    has to stop sharing. (`tools/diff_generation.py` held the result to the per-position answer
    through the recorded selection policy until IKA-458 stage 3.7 removed it.)

    Shallow: the members of a quartet share its leads' objects, and every opening shares
    its Pokemon with the other openings that put the same set in the same slot (IKA-267:
    8,100 positions of 12 sets were 97,200 stat lines of 48 different Pokemon). These
    positions go to a leaf evaluator, which reads them.

    A draw in the leads' switch-ins (IKA-352: a Speed tie that changes where they end, a
    Trace between two foes) is its first option here; `lead_branches_from_sets` has every
    option, and its first is this position.
    """
    return [branches[0][1] for branches in lead_branches_from_sets(reg, pairs)]


def lead_branches_from_sets(
    reg: Regulation, pairs: Sequence[tuple[list[SampledSet], list[SampledSet]]]
) -> list[list[tuple[float, Position]]]:
    """`positions_from_sets` with every outcome of the leads' draws at its weight (IKA-352):
    one `(1.0, position)` for an opening that draws nothing, which is nearly all of them."""
    active = reg.meta.active_per_side
    made: dict[tuple[int, int, int], Pokemon] = {}
    openings = [_opening(reg, own, foe, made) for own, foe in pairs]
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for index, (own, foe) in enumerate(pairs):
        key = (tuple(id(s) for s in own[:active]), tuple(id(s) for s in foe[:active]))
        groups.setdefault(key, []).append(index)
    firsts = [members[0] for members in groups.values()]
    answered = port.lead_branches_many(reg, [openings[i] for i in firsts])
    out: list[list[tuple[float, Position]] | None] = [None] * len(pairs)
    unshared: list[int] = []
    for members, branches in zip(groups.values(), answered, strict=True):
        opening = openings[members[0]]
        led = [(weight, phase.position) for weight, phase in branches]
        if not all(_bench_untouched(pos, opening, active) for _weight, pos in led):
            out[members[0]] = led
            unshared.extend(members[1:])
            continue
        for index in members:
            out[index] = (
                led
                if index == members[0]
                else [(weight, _with_back(pos, openings[index], active)) for weight, pos in led]
            )
    if unshared:
        for index, branches in zip(
            unshared,
            port.lead_branches_many(reg, [openings[i] for i in unshared]),
            strict=True,
        ):
            out[index] = [(weight, phase.position) for weight, phase in branches]
    return [branches for branches in out if branches is not None]


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


def _sample_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    total = float(weights.sum())
    if total <= 0:
        return int(rng.integers(len(weights)))
    return int(rng.choice(len(weights), p=weights / total))


#: Which completion of the opponent's unseen slots a leaf or policy ranking reads
#: (`_menus.views`). "heaviest" ships; "first" is the rule before IKA-143.
RANK_VIEWS = ("heaviest", "first")


@cache
def _fingerprint_of(path: str) -> str:
    return json.dumps(rustnode.binary_fingerprint(path))


def _fingerprint(path: Path | str) -> dict:
    """The executable's fingerprint, read once per path (it is hashed, 12 MB)."""
    return json.loads(_fingerprint_of(str(path)))


def ranking_view(
    pos: Position,
    spreads: dict[int, list] | None,
    side: int,
    rank_view: str,
    used: dict[int, tuple[int, tuple[str, ...]]] | None,
) -> Position:
    """The position side `side` ranks from in `_menus`: the node itself, or the completion of
    the other side's bench that `rank_view` picks (the heaviest, first on ties; `first` is the
    first). `used`, when given, gets `side -> (index, species)` of the completion."""
    if spreads is None:
        return pos
    items = spreads[1 - side]
    if not items:
        return pos
    # `max` keeps the first of equal maxima, so uniform weights pick index 0.
    index = 0 if rank_view == "first" else max(range(len(items)), key=lambda i: items[i].weight)
    if used is not None:
        used[side] = (index, tuple(items[index].species))
    return items[index].position


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
    rank_fill: str | None = None,
    wide: Sequence[int] = (),
    wider: dict[int, tuple[list[SideAction], list[SideAction]]] | None = None,
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
    A ``-nocover`` label (IKA-323) builds the leaf-ranked menus -- the wider ones too --
    from the ranking alone, with no cover of every slot option first; what that leaves
    off is in each `Narrowed.uncovered` and tallied in `narrow.COVERLESS`. A ``q`` or
    ``q-nocover`` label (IKA-274, `qrank`) ranks by the process's Q instead of filling
    cells with the leaf: Q over both whole pools on the same view, each candidate scored
    against the other side's half of its solve. Unnamed (None), a leaf-ranked menu
    takes `search.SHIPPED_RANK_FILL` and needs the process's Q (IKA-341); ``refs2`` is
    played only when named.

    ``wide`` asks for the same agent's menus at other widths too, written into ``wider``
    by width: the candidates of the root's double oracle (IKA-293). They are ranked by
    the same scores -- the ranking is asked once per side and remembered -- so each is
    the menu this agent would build at that width. Without it nothing here changes.
    """
    if rank_view not in RANK_VIEWS:
        raise ValueError(f"rank_view {rank_view!r} is not one of {RANK_VIEWS}")
    rank_fill = resolve_rank_fill(rank_fill, rank_by_leaf and policy is None)
    references, fast_fill = parse_rank_fill(rank_fill)
    cover = rank_fill_covers(rank_fill)
    rank_budget = Budget.fast() if fast_fill else budget
    widths = sorted(set(wide))
    if widths and wider is None:
        raise ValueError("wide menus need a `wider` to be written into")
    if policy is None and not rank_by_leaf:
        # The damage score reads only the active Pokemon, so it has nothing to be blind
        # about and the true position costs nothing here.
        menus = (
            narrow(reg, pos, 0, limit=limits[0]).actions,
            narrow(reg, pos, 1, limit=limits[1]).actions,
        )
        for width in widths:
            wider[width] = (
                narrow(reg, pos, 0, limit=width).actions,
                narrow(reg, pos, 1, limit=width).actions,
            )
        return menus

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
        return [(ranking_view(pos, spreads, side, rank_view, used), 1.0)]

    if policy is None and is_q(rank_fill) and not widths and not cover:
        # IKA-457 stage 2.3: both menus in one request to the port, where its road goes.
        from . import rootmenus

        built = rootmenus.menus(
            reg, pos, (_one_view(views(0)), _one_view(views(1))), limits, rank_fill
        )
        if built is not None:
            return built

    # IKA-274 stage 3: both sides' Q requests in one round trip, before either menu asks.
    q_given = (
        qrank.prefetch(
            reg, pos, (_one_view(views(0)), _one_view(views(1))),
            qrank.installed(qrank.q_name(rank_fill)),
        )
        if policy is None and is_q(rank_fill)
        else {}
    )

    def ranker(side: int) -> Any:  # noqa: ANN401
        if policy is None and is_q(rank_fill):
            # IKA-274: a Q's solve on the same view, no cell filled by the leaf.
            return _q_ranker(reg, views(side), side, rank_fill, q_given.get(side))
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

    def menu(side: int, limit: int, rank: Any, *, tally: bool = True) -> list[SideAction]:  # noqa: ANN401
        got = narrow(reg, pos, side, limit=limit, rank=rank, cover=cover)
        if not cover and tally:
            note_coverless(rank_fill, got)
        return got.actions

    if not widths:
        # A part each (IKA-32): the two sides' menus are independent of each other.
        with timing.region("menu.side"):
            first = menu(0, limits[0], ranker(0))
        with timing.region("menu.side"):
            second = menu(1, limits[1], ranker(1))
        return first, second
    # The same calls in the same order, each side's ranking remembered for its wider menus.
    with timing.region("menu.side"):
        own_rank = _remembered(ranker(0))
        own = menu(0, limits[0], own_rank)
    with timing.region("menu.side"):
        foe_rank = _remembered(ranker(1))
        foe = menu(1, limits[1], foe_rank)
    for width in widths:
        wider[width] = (
            menu(0, width, own_rank, tally=False),
            menu(1, width, foe_rank, tally=False),
        )
    return own, foe


def _one_view(views: list[tuple[Position, float]]) -> Position:
    """The one position of `_menus.views` (a Q ranks from one view, as the leaf does)."""
    (at, _weight), = views
    return at


def _q_ranker(
    reg: Regulation, views: list[tuple[Position, float]], side: int, label: str,
    given: Any = None,  # noqa: ANN401 - qrank.Given
) -> Any:  # noqa: ANN401
    """The ``q`` rank fills' ranking (IKA-274, `qrank`) from the one view `_menus` reads,
    by the Q the label names; `given` is `qrank.prefetch`'s answer for this side."""
    return qrank.q_ranking(
        reg, _one_view(views), side, qrank.installed(qrank.q_name(label)), label, given
    )


def _remembered(rank: Any) -> Any:  # noqa: ANN401
    """`rank`, asked once per pool: a second menu from the same scores costs no fill."""
    seen: dict[tuple[str, ...], Any] = {}

    def again(pool: list[SideAction], scored: object = None) -> Any:  # noqa: ANN401
        key = tuple(action.to_choice() for action in pool)
        if key not in seen:
            seen[key] = rank(pool, scored)
        return seen[key]

    return again


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


def _self_switch_request(
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None], hidden: _HiddenBench | None
) -> tuple[dict[str, Any], list[Any]] | None:
    """IKA-458 stage 3.2: what the port needs to choose a self-switch and play the turn on --
    the inference server each side's leaf is, and where a hidden bench's completions are read
    from -- and the leaves it names; None where this road does not go (Python plays the pause)."""
    if timing.DUPES:
        return None
    if hidden is not None and hidden.game is None:
        return None
    from .encode import rules_of
    from .inference import RemoteValue

    owners: list[Any] = []
    leaf: list[int] = []
    for evaluate in leaves:
        owner = getattr(evaluate, "__self__", evaluate)
        if evaluate is None or type(owner) is not RemoteValue:
            return None
        if not any(owner is seen for seen in owners):
            owners.append(owner)
        leaf.append(next(k for k, seen in enumerate(owners) if seen is owner))
    servers = []
    for owner in owners:
        one: dict[str, Any] = {
            "address": owner.address,
            "model": owner.model,
            "shm": owner._block.name,  # noqa: SLF001 - the leaf's own block, idle while the port asks
            "bytes": int(owner._block.size),  # noqa: SLF001
            "batch": int(owner.batch_size),
            "netScoresEnds": bool(owner.encoder.rules.net_scores_ends),
        }
        rules = rules_of(owner)
        if rules.mega_from_slots:
            one["encoding"] = rules.to_request()
        servers.append(one)
    request: dict[str, Any] = {
        "servers": servers,
        "leaf": leaf,
        "hidden": None
        if hidden is None
        else {
            "game": hidden.game,
            "seen": [sorted(s) for s in hidden.seen],
            "leads": [None if lead is None else sorted(lead) for lead in hidden.leads],
        },
    }
    return request, owners


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
    #: IKA-458 stage 3.4: this game's number in the port (the removed `hiddenstep.Game`), or
    #: None where the port does not hold the game (a pause is then left to Python).
    game: int | None = None


def _do_self_switch_node(
    reg: Regulation,
    pause: PortPause,
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None],
    objective: Objective,
    *,
    hidden: _HiddenBench | None = None,
    definition: bool = False,
) -> PortTurn | ResumedTurn | None:
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
) -> PortTurn | ResumedTurn | None | object:
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
    # The weights, and the drawn outcome when `_advance` picks it (IKA-350).
    return port.resumed(reg, pause, [choice, passes] if chooser == 0 else [passes, choice])


def _rows(encoded: Any, start: int, count: int) -> Any:  # noqa: ANN401
    """Rows `start:start+count` of an `Encoded`, as views."""
    from .beliefnode import _ARRAYS
    from .encode import Encoded

    return Encoded(
        **{name: getattr(encoded, name)[start : start + count] for name in _ARRAYS},
        unknown_volatiles=dict(encoded.unknown_volatiles),
        decided=None if encoded.decided is None else encoded.decided[start : start + count],
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
    sheets: tuple[Sequence[SampledSet], Sequence[SampledSet]],
    shown: list[frozenset[int]] | None = None,
    bench_prior: tuple[BenchPrior | None, BenchPrior | None] | None = None,
    leads: list[frozenset[str] | None] | None = None,
    recorded_shown: list[list[str]] | None = None,
    selects: tuple[str, str] = (DEFAULT_EQ_SELECT, DEFAULT_EQ_SELECT),
) -> Position:
    """Solves and applies the replacement phase.

    It is a simultaneous-move node like any other -- neither player sees the other's
    replacement -- so it gets a matrix and an equilibrium rather than a heuristic pick.

    It is also the node where the *reveal* happens: the Pokemon coming in is
    the one that stops being hidden, and choosing what to send against an opponent whose
    bench is unknown is the same Bayesian game the move nodes solve. `shown` is each
    side's seen slots in *this* position -- `seen_slots` of the identities `play_game`
    carries -- and never a set of numbers carried from an earlier one (IKA-117). `leads`
    is the turn-1 lead pair per side, as `play_game` carries it (IKA-118).
    `recorded_shown` is `Decision.shown`, written as given and read by nothing here.

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

    foe_value: float | None = None
    seen = shown or [frozenset(), frozenset()]
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
        from . import replacenode

        # IKA-457 stage 2.2: the whole node in one request to the port, on the inference
        # server's leaf (IKA-460: the only road; side 1 minimises what side 0 maximises, so
        # its game is the transpose of the negation, read over there from the same matrices).
        answers = replacenode.solve(reg, options, spreads, leaves, selects)
        own_policy, value = answers[0]
        foe_policy = answers[1][0]
        # Side 1's game is the negated transpose: back into side 0's units.
        foe_value = -answers[1][1]
    else:
        own_policy = [1.0 / len(options[0])] * len(options[0])
        foe_policy = [1.0 / len(options[1])] * len(options[1])
        value = 0.5

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


def replacement_matrix(
    reg: Regulation,
    at: Position,
    options: Sequence[Sequence[SideAction]],
    evaluate: LeafEvaluator | None,
) -> np.ndarray:
    """Every pair of replacements resolved from `at` and scored, side 0's value.

    A phase that draws -- a Speed tie among the switch-ins that changes where they end, a
    Trace between two foes -- is every outcome at its weight (IKA-352), not the first one:
    the cell is their weighted mean. A phase that draws nothing is its one position, scored
    as before.
    """
    cells = [
        port.replacement_branches(reg, at, [a, b]) for a in options[0] for b in options[1]
    ]
    flat = [phase.position for cell in cells for _weight, phase in cell]
    weights = np.array([weight for cell in cells for weight, _phase in cell], dtype=np.float64)
    owner = np.repeat(np.arange(len(cells)), [len(cell) for cell in cells])
    if evaluate is None:
        values = np.array([HP_SHARE(after) for after in flat], dtype=np.float64)
    else:
        values = np.asarray(evaluate(flat), dtype=np.float64)
    if len(flat) == len(cells):
        scored = values
    else:
        scored = np.bincount(owner, weights=weights * values, minlength=len(cells))
    return scored.reshape(len(options[0]), len(options[1]))


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


__all__ = [
    "MAX_TURNS",
    "SEARCH_LIMIT",
    "END_REASONS",
    "Decision",
    "GameRecord",
    "final_position",
    "lead_branches_from_sets",
    "position_from_sets",
    "positions_from_sets",
    "replacement_matrix",
    "selfplay_dir",
]
