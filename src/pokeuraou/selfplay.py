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
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from .actions import SideAction, switch_actions_after_faint
from .equilibrium import EquilibriumError, solve
from .hidden import completions, seen_slots, shown_species
from .narrow import narrow
from .payoff import HP_SHARE, Objective
from .policy import policy_ranking
from .position import Field, MoveSlot, Pokemon, Position, Side
from .priors import Cooccurrence, MetagamePrior, SampledSet
from .provenance import engine_fingerprint
from .regulation import STAT_IDS, Regulation, repo_root
from .resolve import (
    Budget,
    SuspendedTurn,
    TurnResult,
    apply_lead_abilities,
    replacements_needed,
    resolve_replacements,
    resolve_turn,
    resume_alternatives,
    turn_leaves,
)
from .search import belief_solve, believed_ranking, leaf_ranking, search
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
#: affordable -- see tools/selfplay_budget.py. It is smaller than the analysis default
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
            "ownSelectionPolicy": self.own_selection_policy,
            "foeSelectionPolicy": self.foe_selection_policy,
            "ownSelectionMixture": self.own_selection_mixture,
            "foeSelectionMixture": self.foe_selection_mixture,
            "selectionValue": self.selection_value,
            "unmodelled": sorted(set(self.unmodelled)),
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
            MoveSlot(id=m, pp=reg.moves[m].pp, maxpp=reg.moves[m].pp) for m in entry.moves
        ],
        hp=maxhp,
        maxhp=maxhp,
        item=entry.item,
        base_item=entry.item,
        sp={stat: entry.sp.get(stat, 0) for stat in STAT_IDS},
        active_index=active,
    )


def position_from_sets(
    reg: Regulation, own: list[SampledSet], foe: list[SampledSet]
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
    sides: list[Side] = []
    for side_index, sets in enumerate((own, foe)):
        mons = [
            _make_pokemon(reg, i, entry, i if i < reg.meta.active_per_side else None)
            for i, entry in enumerate(sets)
        ]
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
    opening = Position(format=reg.meta.format_id, sides=sides, turn=1, field=Field())
    return apply_lead_abilities(reg, opening).position


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


def _menus(
    reg: Regulation,
    pos: Position,
    limits: tuple[int, int],
    evaluate: LeafEvaluator,
    budget: Budget,
    rank_by_leaf: bool,
    policy: Any = None,
    spreads: dict[int, list] | None = None,
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
    """
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

        Deterministic -- the first completion, not a sampled one -- so a rerun of a game
        is the same game.
        """
        if spreads is None:
            return [(pos, 1.0)]
        items = spreads[1 - side]
        return [(items[0].position, 1.0)] if items else [(pos, 1.0)]

    def ranker(side: int) -> Any:  # noqa: ANN401
        parts = [
            (
                policy_ranking(policy, at, side)
                if policy is not None
                else leaf_ranking(reg, at, side, evaluate, budget=budget),
                weight,
            )
            for at, weight in views(side)
        ]
        return believed_ranking(parts)

    return (
        narrow(reg, pos, 0, limit=limits[0], rank=ranker(0)).actions,
        narrow(reg, pos, 1, limit=limits[1], rank=ranker(1)).actions,
    )


def _bench_weights(
    bench_prior: tuple[BenchPrior | None, BenchPrior | None] | None,
    side: int,
    pos: Position,
    seen: frozenset[int],
    record: GameRecord,
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
    """
    if bench_prior is None or bench_prior[side] is None:
        return None
    weights = bench_prior[side].weights(shown_species(pos, side, seen))
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
    start: Position | None = None,
    first_action: str | None = None,
    sheets: tuple[Sequence[SampledSet], Sequence[SampledSet]] | None = None,
    bench_prior: tuple[BenchPrior | None, BenchPrior | None] | None = None,
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

    ``solve_sparsely`` takes a pair too, and it is the one whose two values are supposed
    to be *equally correct*: both settle on an equilibrium of the same game, verified to
    an exploitability of 7.7e-08. What differs is which vertex of a degenerate optimum
    they land on, and a maximin strategy only guarantees the value -- against an opponent
    who is not playing the equilibrium, two equilibria can take different amounts. That is
    what a mismatched pair measures.
    """
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
    policies = policy if isinstance(policy, tuple) else (policy, policy)
    leaves = evaluate if isinstance(evaluate, tuple) else (evaluate, evaluate)
    if sheets is not None and (depths != (1, 1) or sparse != (False, False)):
        # `belief_solve` takes neither, so under a hidden bench these were accepted,
        # recorded per side in the provenance, and then dropped. An argument that is
        # silently ignored is how every measurement defect found today was built: the
        # caller reads the flag it passed, the record repeats it, and nothing played it.
        raise ValueError(
            f"depth {depths} and solve_sparsely {sparse} cannot be honoured with a "
            "hidden bench -- belief_solve has no parameter for either. Pass depth 1 and "
            "solve_sparsely False, or drop `sheets` and measure in the open game."
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
    pos = start if start is not None else position_from_sets(reg, own, foe)
    budget = Budget.matrix()
    # Slots each side has shown, accumulated across turns. A Pokemon that came in and
    # went back out is still known, and the position alone stops saying so -- so this is
    # carried rather than recomputed from the board each time.
    shown: list[frozenset[int]] = [frozenset(), frozenset()]

    for _step in range(max_turns * 2):
        if pos.ended:
            break

        owed = replacements_needed(pos)
        if any(owed[0]) or any(owed[1]):
            shown = [seen_slots(pos, i, shown[i]) for i in (0, 1)]
            pos = _do_replacement_node(
                reg, rng, pos, owed, record, leaves,
                sheets=sheets, shown=shown, bench_prior=bench_prior,
            )
            continue

        own_leaf = leaves[0] if leaves[0] is not None else objective.batch
        foe_leaf = leaves[1] if leaves[1] is not None else objective.batch
        shown = [seen_slots(pos, i, shown[i]) for i in (0, 1)]
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
                        weights=_bench_weights(bench_prior, side, pos, shown[side], record),
                    )
                    for side in (0, 1)
                }
            except ValueError as problem:
                record.unmodelled.append(f"hidden bench: {problem}")
                break
        menu_started = perf_counter()
        ours, theirs = _menus(
            reg, pos, limits, own_leaf, budget, ranked[0], policies[0], spreads
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
            and (
                policies[0] is not None
                or not ranked[0]
                or leaves[1] is leaves[0]
            )
        )

        own_seconds = foe_seconds = 0.0
        foe_solved = False

        if spreads is not None:
            # Each side is uncertain about a different bench, so each gets its own answer.
            # The node underneath them is resolved once: their hidden slots are disjoint
            # and a cell that reaches neither resolves the same way whatever is standing
            # on either bench.
            solve_started = perf_counter()
            try:
                answers = belief_solve(
                    reg, pos, ours, theirs, spreads,
                    {0: own_leaf, 1: foe_leaf}, budget=budget,
                )
            except EquilibriumError:
                break
            own_seconds = perf_counter() - solve_started
            record.unmodelled.extend(
                answers[0].unmodelled | answers[1].unmodelled
            )
            own_strategy = answers[0].strategy
            foe_strategy = answers[1].strategy
            foe_theirs = theirs
            search_value = answers[0].value
            if not same_menu:
                # The open path rebuilds the column player's game when the settings
                # differ; this one used to skip that entirely, so under a hidden bench
                # `ranked[1]` and `policies[1]` were dead arguments -- the provenance
                # recorded them per side and only side 0's were ever played. The anchor
                # of the hidden-bench scale was measured this way, with the hp-share arm
                # handed a menu ranked by the other arm's value function in one seat.
                foe_started = perf_counter()
                foe_ours, foe_theirs = _menus(
                    reg, pos, limits, foe_leaf, budget, ranked[1], policies[1], spreads
                )
                if not foe_ours or not foe_theirs:
                    break
                try:
                    foe_answers = belief_solve(
                        reg, pos, foe_ours, foe_theirs, spreads,
                        {0: own_leaf, 1: foe_leaf}, budget=budget,
                    )
                except EquilibriumError:
                    break
                record.unmodelled.extend(foe_answers[1].unmodelled)
                foe_strategy = foe_answers[1].strategy
                foe_seconds = perf_counter() - foe_started
                foe_solved = True
        else:
            solve_started = perf_counter()
            try:
                own_search = search(
                    reg, pos, ours, theirs, own_leaf, budget=budget, depth=depths[0],
                    solve_sparsely=sparse[0],
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
            ):
                foe_started = perf_counter()
                foe_ours, foe_theirs = (
                    (ours, theirs)
                    if same_menu
                    else _menus(
                        reg, pos, limits, foe_leaf, budget, ranked[1], policies[1],
                        spreads,
                    )
                )
                if not foe_ours or not foe_theirs:
                    break
                try:
                    foe_search = search(
                        reg, pos, foe_ours, foe_theirs, foe_leaf, budget=budget,
                        depth=depths[1], solve_sparsely=sparse[1],
                    )
                except EquilibriumError:
                    break
                record.unmodelled.extend(foe_search.unmodelled)
                foe_equilibrium = foe_search.equilibrium
                foe_seconds = perf_counter() - foe_started
                foe_solved = True
            own_strategy = np.asarray(equilibrium.row_strategy, dtype=np.float64)
            foe_strategy = np.asarray(foe_equilibrium.col_strategy, dtype=np.float64)
            search_value = float(equilibrium.value)

        if foe_solved:
            # `same_menu` is the only case where one construction served both agents;
            # otherwise side 0's menus are side 0's alone and side 1 timed its own.
            shared = menu_seconds / 2 if same_menu else 0.0
            record.search_seconds[0] += own_seconds + menu_seconds - shared
            record.search_seconds[1] += foe_seconds + shared
        else:
            # One construction and one solve served both agents. Neither of them would
            # have spent less alone, and neither of them spent it alone.
            record.search_seconds[0] += (menu_seconds + own_seconds) / 2
            record.search_seconds[1] += (menu_seconds + own_seconds) / 2

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
            )
        )
        advanced = _advance_turn(reg, rng, pos, chosen, record, leaves, objective)
        if advanced is None:
            break
        pos = advanced
        record.turns = pos.turn

    if pos.ended and pos.winner is not None:
        record.outcome = 1.0 if pos.winner == pos.sides[0].id else 0.0
    return record


def _advance_turn(
    reg: Regulation,
    rng: np.random.Generator,
    pos: Position,
    chosen: list[SideAction],
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None],
    objective: Objective,
) -> Position | None:
    """Resolves the chosen actions and samples one outcome.

    The Rust port fills this when it is enabled and the turn does not suspend -- a
    suspension carries continuation state that cannot cross a process boundary, and the
    replacement it asks for is a decision node rather than a chance node. The sampling
    stays here either way, so the generator draws the same way and a game played through
    the bridge is the same game.
    """
    from . import rustnode

    if rustnode.available():
        node = rustnode.node_for(reg)
        if node is not None:
            try:
                weights = node.resolve(pos, chosen, Budget.exact())
                if weights is not None:
                    record.unmodelled.extend(weights.unmodelled)
                    counts = np.array(
                        weights.branches + weights.suspended, dtype=np.float64
                    )
                    if not counts.size or float(counts.sum()) <= 0:
                        return None
                    index = _sample_index(rng, counts)
                    if index < len(weights.branches):
                        picked = node.resolve(pos, chosen, Budget.exact(), select=index)
                        if picked is not None and picked.position is not None:
                            return picked.position
                    else:
                        # A replacement was drawn. Its continuation lives in the Rust
                        # process, so the turn is resolved here -- but the draw has already
                        # happened, and re-drawing would put this game on a different
                        # random stream than one played without the bridge.
                        result = resolve_turn(reg, pos, chosen, budget=Budget.exact())
                        return _advance(
                            reg, rng, result, record, leaves, objective, first_index=index
                        )
            except Exception as exc:  # noqa: BLE001 - a broken bridge must not fail a run
                rustnode.disable(str(exc))

    result = resolve_turn(reg, pos, chosen, budget=Budget.exact())
    record.unmodelled.extend(result.unmodelled)
    return _advance(reg, rng, result, record, leaves, objective)


def _advance(
    reg: Regulation,
    rng: np.random.Generator,
    result: TurnResult,
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None],
    objective: Objective,
    first_index: int | None = None,
) -> Position | None:
    """Samples one outcome of a resolved turn, answering any mid-turn request on the way.

    Returns ``None`` when the turn produced nothing to continue from, which the caller
    treats as the end of the game.

    ``first_index`` is for a caller that has already drawn the first index -- the Rust
    bridge hands back the weights and samples there, so that the generator is used exactly
    once per turn whichever path the turn takes.
    """
    for attempt in range(5):
        weights = np.array(
            [b.probability for b in result.branches]
            + [p.probability for p in result.suspended],
            dtype=np.float64,
        )
        if not weights.size or float(weights.sum()) <= 0:
            return None
        index = (
            first_index
            if attempt == 0 and first_index is not None
            else _sample_index(rng, weights)
        )
        if index < len(result.branches):
            return result.branches[index].position
        pause = result.suspended[index - len(result.branches)]
        resumed = _do_self_switch_node(reg, pause, record, leaves, objective)
        if resumed is None:
            return None
        result = resumed
    record.unmodelled.append("more than five mid-turn replacements in one turn")
    return None


def _do_self_switch_node(
    reg: Regulation,
    pause: SuspendedTurn,
    record: GameRecord,
    leaves: tuple[LeafEvaluator | None, LeafEvaluator | None],
    objective: Objective,
) -> TurnResult | None:
    """Chooses the replacement a self-switching move demanded, and finishes the turn.

    Unlike the post-turn replacement phase this is not simultaneous: one side is asked and
    the other is on `wait`, so there is no matrix to solve. The chooser takes the option it
    values most, scored with its own value function.

    The approximation left here is an information one, and it is the same one the depth-1
    search makes everywhere: the rest of the turn is already committed in this line, so the
    choice is made against a known continuation where a real player would only have seen
    the turn up to the interrupt. It is reported rather than papered over.
    """
    chooser, alternatives = resume_alternatives(reg, pause)
    if chooser is None or not alternatives:
        return None
    record.unmodelled.append(
        "mid-turn replacement chosen against the opponent's already-committed action"
    )

    plans = [turn_leaves(reg, resumed) for _option, resumed in alternatives]
    flat = [position for plan in plans for position in plan.positions]
    if not flat:
        return None
    evaluate = leaves[chooser]
    values = (
        evaluate(flat)
        if evaluate is not None
        else np.array([objective(position) for position in flat], dtype=np.float64)
    )
    scores: list[float] = []
    offset = 0
    for plan in plans:
        count = len(plan.positions)
        scores.append(plan.value(values[offset : offset + count]))
        record.unmodelled.extend(plan.unmodelled)
        offset += count

    # Side 0 is the maximiser the payoff matrices are written for.
    best = int(np.argmax(scores)) if chooser == 0 else int(np.argmin(scores))
    policy = [0.0] * len(alternatives)
    policy[best] = 1.0
    options = [option.to_choice() for option, _resumed in alternatives]
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
        )
    )
    return alternatives[best][1]


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
) -> Position:
    """Solves and applies the replacement phase.

    It is a simultaneous-move node like any other -- neither player sees the other's
    replacement -- so it gets a matrix and an equilibrium rather than a heuristic pick.

    With `sheets` it is also the node where the *reveal* happens: the Pokemon coming in is
    the one that stops being hidden, and choosing what to send against an opponent whose
    bench is unknown is the same Bayesian game the move nodes solve.

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
            [resolve_replacements(reg, at, [a, b]).position for b in options[1]]
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

    if sheets is not None:
        seen = shown or [frozenset(), frozenset()]
        answers: dict[int, tuple[list[float], float]] = {}
        try:
            spreads = {
                side: completions(
                    reg, pos, side, sheets[side], seen=seen[side],
                    weights=_bench_weights(bench_prior, side, pos, seen[side], record),
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
        )
    )
    outcome = resolve_replacements(reg, pos, chosen)
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
    force_lead: tuple[str, ...] | None = None,
    hide_bench: bool = False,
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
    """
    if not 0.0 <= mirror_share <= 1.0:
        raise ValueError(f"mirror_share must be a probability, got {mirror_share}")
    if book is not None and standings is None:
        raise ValueError(
            "a selection book is keyed on tournament team sheets, so it needs the "
            "standings pool it was solved against"
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
                # Both sixes, so neither search is shown the other's unplayed bench.
                sheets=(list(roster.sets), list(foe_six)) if hide_bench else None,
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
    "Decision",
    "GameRecord",
    "generate",
    "play_game",
    "position_from_sets",
    "selfplay_dir",
]
