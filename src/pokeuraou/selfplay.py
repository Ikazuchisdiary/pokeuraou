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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .actions import SideAction, switch_actions_after_faint
from .equilibrium import EquilibriumError, solve
from .narrow import narrow
from .payoff import HP_SHARE, Objective
from .position import Field, MoveSlot, Pokemon, Position, Side
from .priors import Cooccurrence, MetagamePrior, SampledSet
from .regulation import STAT_IDS, Regulation, repo_root
from .resolve import Budget, replacements_needed, resolve_replacements, resolve_turn
from .standings import Standings, cluster_teams, label_for, sample_standings_team
from .stats import nature_multipliers, stats_from_sp
from .teams import (
    Archetype,
    Roster,
    pick_four,
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
    :func:`_solve_matrix`.
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

    def to_json(self, *, objective: str, search_limit: int | tuple[int, int]) -> dict[str, Any]:
        return {
            "ownTeam": self.own_team,
            "foeTeam": self.foe_team,
            "foeArchetype": self.foe_archetype,
            "outcome": self.outcome,
            "turns": self.turns,
            "foeSpecies": [m["species"] for m in self.foe_team],
            "searchObjective": objective,
            "searchLimit": list(search_limit)
            if isinstance(search_limit, tuple)
            else search_limit,
            "targetIsRealOutcome": True,
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
    return Position(format=reg.meta.format_id, sides=sides, turn=1, field=Field())


def _sample_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    total = float(weights.sum())
    if total <= 0:
        return int(rng.integers(len(weights)))
    return int(rng.choice(len(weights), p=weights / total))


def _solve_matrix(
    reg: Regulation,
    pos: Position,
    ours: list[SideAction],
    theirs: list[SideAction],
    objective: Objective,
    budget: Budget,
    evaluate: LeafEvaluator | None = None,
) -> tuple[np.ndarray, set[str]]:
    """The payoff matrix, with every leaf in the node scored in one call.

    With ``evaluate`` the leaves go to the value function as a single batch spanning the
    whole matrix. Calling it per branch instead would be 256 forward passes for a 16x16
    node, and the forward pass is 3% of the per-leaf cost -- so batching across the node is
    the whole optimisation, and it also makes the GPU's fixed overhead amortise.

    Without ``evaluate`` this is the parameter-free objective applied branch by branch,
    unchanged, because the differential tests and the M1 analysis path still use it.
    """
    payoff = np.zeros((len(ours), len(theirs)), dtype=np.float64)
    unmodelled: set[str] = set()

    if evaluate is None:
        for i, a in enumerate(ours):
            for j, b in enumerate(theirs):
                result = resolve_turn(reg, pos, [a, b], budget=budget)
                payoff[i, j] = result.expected(objective)
                unmodelled.update(result.unmodelled)
        return payoff, unmodelled

    leaves: list[Position] = []
    weights: list[np.ndarray] = []
    spans: list[tuple[int, int, int, int]] = []
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            result = resolve_turn(reg, pos, [a, b], budget=budget)
            unmodelled.update(result.unmodelled)
            total = result.total_probability
            if not result.branches or total <= 0:
                # Nothing resolved: leave the cell at zero and let the caller see it.
                spans.append((i, j, len(leaves), 0))
                weights.append(np.zeros(0))
                continue
            start = len(leaves)
            leaves.extend(branch.position for branch in result.branches)
            weights.append(
                np.array([branch.probability for branch in result.branches]) / total
            )
            spans.append((i, j, start, len(result.branches)))

    values = evaluate(leaves) if leaves else np.zeros(0)
    for (i, j, start, count), w in zip(spans, weights, strict=True):
        if count:
            payoff[i, j] = float(values[start : start + count] @ w)
    return payoff, unmodelled


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
) -> GameRecord:
    """Plays one game to a result, sampling both sides from the turn's equilibrium.

    ``search_limit`` may be a pair, giving each side its own number of candidates. Equal
    values are the training setting; an unequal pair is a diagnostic -- if one side wins
    less when searched just as widely, the imbalance was in the search and not the teams.
    """
    limits = (search_limit, search_limit) if isinstance(search_limit, int) else search_limit
    leaves = evaluate if isinstance(evaluate, tuple) else (evaluate, evaluate)
    record = GameRecord(
        own_team=[_set_json(reg, s) for s in own],
        foe_team=[_set_json(reg, s) for s in foe],
        foe_archetype=foe_archetype,
    )
    pos = position_from_sets(reg, own, foe)
    budget = Budget.matrix()

    for _step in range(max_turns * 2):
        if pos.ended:
            break

        owed = replacements_needed(pos)
        if any(owed[0]) or any(owed[1]):
            pos = _do_replacement_node(reg, rng, pos, owed, record, leaves[0])
            continue

        ours = narrow(reg, pos, 0, limit=limits[0]).actions
        theirs = narrow(reg, pos, 1, limit=limits[1]).actions
        if not ours or not theirs:
            break

        payoff, unmodelled = _solve_matrix(
            reg, pos, ours, theirs, objective, budget, leaves[0]
        )
        record.unmodelled.extend(unmodelled)
        try:
            equilibrium = solve(payoff)
        except EquilibriumError:
            break

        # With different leaves the two sides are no longer solving one game, so the
        # column player's strategy has to come from *its* matrix. Same evaluator on both
        # sides skips this entirely and the behaviour is identical to before.
        foe_equilibrium = equilibrium
        if leaves[1] is not leaves[0]:
            foe_payoff, foe_unmodelled = _solve_matrix(
                reg, pos, ours, theirs, objective, budget, leaves[1]
            )
            record.unmodelled.extend(foe_unmodelled)
            try:
                foe_equilibrium = solve(foe_payoff)
            except EquilibriumError:
                break

        record.decisions.append(
            Decision(
                turn=pos.turn,
                kind="move",
                position=pos.to_json(),
                own_actions=[a.to_choice() for a in ours],
                own_policy=[float(x) for x in equilibrium.row_strategy],
                foe_actions=[a.to_choice() for a in theirs],
                foe_policy=[float(x) for x in foe_equilibrium.col_strategy],
                search_value=float(equilibrium.value),
            )
        )

        chosen = [
            ours[_sample_index(rng, equilibrium.row_strategy)],
            theirs[_sample_index(rng, foe_equilibrium.col_strategy)],
        ]
        result = resolve_turn(reg, pos, chosen, budget=Budget.exact())
        record.unmodelled.extend(result.unmodelled)
        weights = np.array([b.probability for b in result.branches], dtype=np.float64)
        pos = result.branches[_sample_index(rng, weights)].position
        record.turns = pos.turn

    if pos.ended and pos.winner is not None:
        record.outcome = 1.0 if pos.winner == pos.sides[0].id else 0.0
    return record


def _do_replacement_node(
    reg: Regulation,
    rng: np.random.Generator,
    pos: Position,
    owed: tuple[tuple[bool, ...], tuple[bool, ...]],
    record: GameRecord,
    evaluate: LeafEvaluator | None = None,
) -> Position:
    """Solves and applies the replacement phase.

    It is a simultaneous-move node like any other -- neither player sees the other's
    replacement -- so it gets a matrix and an equilibrium rather than a heuristic pick.
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

    # The replacement node used `hp-share` regardless of what the move nodes used, which
    # would leave a third of a game's decisions scored by the thing being replaced.
    payoff = np.zeros((len(options[0]), len(options[1])), dtype=np.float64)
    resolved = [
        [resolve_replacements(reg, pos, [a, b]).position for b in options[1]]
        for a in options[0]
    ]
    if evaluate is None:
        for i, row in enumerate(resolved):
            for j, after in enumerate(row):
                payoff[i, j] = HP_SHARE(after)
    else:
        flat = [after for row in resolved for after in row]
        values = evaluate(flat)
        payoff = values.reshape(len(options[0]), len(options[1]))
    try:
        equilibrium = solve(payoff)
        own_policy = [float(x) for x in equilibrium.row_strategy]
        foe_policy = [float(x) for x in equilibrium.col_strategy]
        value = float(equilibrium.value)
    except EquilibriumError:
        own_policy = [1.0 / len(options[0])] * len(options[0])
        foe_policy = [1.0 / len(options[1])] * len(options[1])
        value = float(payoff.mean())

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
        )
    )
    chosen = [
        options[0][_sample_index(rng, np.array(own_policy))],
        options[1][_sample_index(rng, np.array(foe_policy))],
    ]
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
    """
    if standings is not None:
        field = cluster_labels(reg, standings, standings_pool)
    else:
        field = {}
        if cooc is None and archetype_share < 1.0:
            archetype_share = 1.0
        if not archetypes and archetype_share > 0.0:
            raise ValueError("archetype_share > 0 but no archetypes were given")
    rng = np.random.default_rng(seed)
    path = out or (selfplay_dir() / f"games-seed{seed}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "games": 0,
        "finished": 0,
        "discarded_unfinished": 0,
        "wins": 0,
        "decisions": 0,
        "turns": 0,
    }
    with path.open("a", encoding="utf-8") as handle:
        for _ in range(games):
            if standings is not None:
                pool = standings.pool(standings_pool)
                team = pool[int(rng.integers(len(pool)))]
                label = field.get(team.player, "worlds")
                foe_six = sample_standings_team(rng, reg, prior, team)
            elif cooc is not None and rng.random() >= archetype_share:
                label = "metagame"
                foe_six = sample_metagame_team(rng, reg, prior, cooc)
            else:
                archetype = archetypes[int(rng.integers(len(archetypes)))]
                label = archetype.id
                foe_six = sample_archetype(rng, reg, prior, archetype)
            own_four = pick_four(rng, roster.sets, size=reg.meta.picked_team_size)
            foe_four = pick_four(rng, foe_six, size=reg.meta.picked_team_size)

            record = play_game(
                reg, rng, own_four, foe_four, label,
                objective=objective, search_limit=search_limit, max_turns=max_turns,
                evaluate=evaluate,
            )
            stats["games"] += 1
            if record.outcome is None:
                stats["discarded_unfinished"] += 1
                continue
            stats["finished"] += 1
            stats["wins"] += int(record.outcome > 0.5)
            stats["decisions"] += len(record.decisions)
            stats["turns"] += record.turns
            handle.write(
                json.dumps(
                    record.to_json(
                        objective=leaf or objective.name, search_limit=search_limit
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
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
