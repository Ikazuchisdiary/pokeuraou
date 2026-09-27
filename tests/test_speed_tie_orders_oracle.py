"""Speed ties the port settled one way, held to Showdown's shuffle (IKA-352).

Showdown breaks every Speed tie with `prng.shuffle` over the tied group (`speedSort`,
sim/battle.ts), so each of a group's `k!` orders is equally likely. The port branched on a
tie between two actions (IKA-270's mirror bias) but settled three others in a fixed order:

* **a group of three or more actions** kept the canonical order, unnoted (`tie_permutations`);
* **simultaneous switch-ins** -- the leads before `|turn|1`, a replacement phase -- ran
  fastest first and, at one Speed, side 0 first (`runSwitch`'s `speedSort(allActive)` is
  the shuffle there);
* **the residual phase** ran (-speed, species, slot, side) and noted the tie
  (`fieldEvent('Residual')` shuffles each tied run of handlers).

Each case is played by Showdown in every order of its tie (`keep` / `reverse` for a pair,
`shuffle_script` for a group of three), which is the fact: the orders end differently. The
port must answer with exactly those outcomes at `1/k!` each -- a branch per order for a
turn, a draw of equal weights for a phase. Before IKA-352 each case came back as one
outcome, which these tests fail (the positive control, `POKEURAOU_RUST_NODE_BIN` at a
master build).
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from pokeuraou.actions import SideAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.priors import SampledSet

from ._port import Budget, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

#: What the oracle's policy pins -- no crit, no chance-based secondary, the maximum roll --
#: with Speed ties enumerated.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)

#: Attack and bulk, no Speed: every Pokemon below at base 85 stands at one Speed.
SLOW = {"hp": 32, "atk": 32, "def": 2, "spa": 0, "spd": 0, "spe": 0}


def _mon(species: str, ability: str, moves: list[str], sp: dict[str, int] = SLOW,
         item: str | None = None) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Adamant", moves=moves, item=item, sp=dict(sp))


def _loaded(raw: dict) -> Position:
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
            mon.stats_override = None
    return pos


def _chosen(reg, pos: Position, step: list[str]) -> list[SideAction]:  # noqa: ANN001
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


def _board(pos: Position) -> tuple:
    """What these cases can change: the field, who won, every Pokemon's HP and boosts."""
    mons = tuple(
        (side.id, mon.species, mon.hp, mon.fainted, tuple(sorted((k, v) for k, v in mon.boosts.items() if v)))
        for side in pos.sides
        for mon in side.pokemon
    )
    return (pos.field.terrain, pos.field.weather, pos.winner, mons)


def _spread(outcomes: list[tuple]) -> dict[tuple, Fraction]:
    """Showdown's outcomes, one per order of the tie, as a distribution."""
    out: dict[tuple, Fraction] = {}
    for board in outcomes:
        out[board] = out.get(board, Fraction(0)) + Fraction(1, len(outcomes))
    return out


def _port_spread(result) -> dict[tuple, Fraction]:  # noqa: ANN001
    assert not result.suspended
    out: dict[tuple, Fraction] = {}
    for branch in result.branches:
        board = _board(branch.position)
        out[board] = out.get(board, Fraction(0)) + Fraction(branch.probability).limit_denominator(720)
    return out


# ---------------------------------------------------------------------------
# A group of three: three Fake Outs at one Speed. Rillaboom and Toxicroak (both base 85)
# Fake Out their Rillaboom; their Rillaboom Fake Outs Toxicroak. Whoever of the three
# moves first flinches its target: their Rillaboom first (2 orders of 6) flinches
# Toxicroak and takes one Fake Out; either of ours first (4 of 6) flinches it and it takes
# two. Incineroar protects, at another priority.

GROUP = (
    [_mon("Rillaboom", "Overgrow", ["fakeout", "protect"]),
     _mon("Toxicroak", "Anticipation", ["fakeout", "protect"])],
    [_mon("Rillaboom", "Overgrow", ["fakeout", "protect"]),
     _mon("Incineroar", "Blaze", ["protect", "knockoff"])],
)
GROUP_STEP = ["move 1 1, move 1 1", "move 1 2, move 1"]
FAKE_OUTS = ("p1a", "p1b", "p2a")


def _group(oracle: Oracle, order: tuple[int, ...]) -> tuple[Position, Position, list[str]]:
    """Showdown's turn with the group of three shuffled into ``order``: the position
    before, the one after, and the order the three acted in (a move or a flinch)."""
    policy = RandomnessPolicy(shuffle_script=(order,))
    handle = oracle.create(FORMAT_ID, *GROUP, policy=policy)
    handle.step(["team 12", "team 12"])
    before = _loaded(handle.position)
    handle.step(GROUP_STEP)
    assert handle.choice_errors == [], handle.choice_errors
    acted = []
    for line in handle.log:
        parts = line.split("|")
        if len(parts) > 2 and parts[1] in ("move", "cant"):
            who = parts[2].split(":")[0]
            if who in FAKE_OUTS and who not in acted:
                acted.append(who)
    after = Position.from_json(handle.position)
    handle.close()
    return before, after, acted


def test_showdown_plays_a_group_of_three_in_all_six_orders(oracle: Oracle) -> None:
    orders = {tuple(_group(oracle, order)[2]) for order in itertools.permutations(range(3))}
    assert orders == set(itertools.permutations(FAKE_OUTS))
    boards = {_board(_group(oracle, order)[1]) for order in itertools.permutations(range(3))}
    assert len(boards) == 2, "the order has to matter for the case to be one"


def test_the_port_branches_every_order_of_a_group_of_three(reg, oracle: Oracle) -> None:  # noqa: ANN001
    played = [_group(oracle, order) for order in itertools.permutations(range(3))]
    showdown = _spread([_board(after) for _before, after, _acted in played])
    before = played[0][0]
    result = resolve_turn(reg, before, _chosen(reg, before, GROUP_STEP), budget=BUDGET)
    assert _port_spread(result) == showdown
    assert sorted(showdown.values()) == [Fraction(1, 3), Fraction(2, 3)]


# ---------------------------------------------------------------------------
# The leads: Rillaboom's Grassy Surge and Indeedee's Psychic Surge at one Speed. The
# terrain that stays is the one set last, so the tie decides the field turn 1 starts on.
# The control is two Grassy Surges at one Speed: the same field whichever goes first, so
# the port draws nothing (a game's draws stay as they were).

RILLABOOM = _mon("Rillaboom", "Grassy Surge", ["protect", "woodhammer"])
INDEEDEE = _mon("Indeedee-F", "Psychic Surge", ["protect", "psychic"])
SNEASLER = _mon("Sneasler", "Poison Touch", ["protect", "closecombat"], {"hp": 2, "atk": 32, "spe": 32})
KINGAMBIT = _mon("Kingambit", "Defiant", ["protect", "ironhead"], {"hp": 32, "def": 32})

LEADS = {
    "grassy-against-psychic": ([RILLABOOM, SNEASLER], [INDEEDEE, KINGAMBIT]),
    "control-grassy-against-grassy": ([RILLABOOM, SNEASLER], [RILLABOOM, KINGAMBIT]),
}


def _lead(oracle: Oracle, name: str, tie: str) -> Position:
    ours, theirs = LEADS[name]
    handle = oracle.create(FORMAT_ID, ours, theirs, policy=RandomnessPolicy(speed_tie=tie))
    handle.step(["team 12", "team 12"])
    pos = Position.from_json(handle.position)
    handle.close()
    return pos


def _sampled(team: list[TeamSet]) -> list[SampledSet]:
    def ident(name: str | None) -> str | None:
        return None if not name else "".join(ch for ch in name.lower() if ch.isalnum())

    return [
        SampledSet(species=ident(t.species), ability=ident(t.ability), item=ident(t.item),
                   nature=t.nature, sp=dict(t.sp), moves=list(t.moves))
        for t in team
    ]


class _Pick:
    """A generator that answers every draw with ``index`` and keeps what it was asked."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.asked: list[tuple[int, list[float]]] = []

    def choice(self, n: int, p: list[float] | None = None) -> int:
        self.asked.append((n, list(p or [])))
        return self.index


def _field(pos: Position) -> tuple:
    return (pos.field.terrain, pos.field.terrain_duration)


@pytest.mark.parametrize("name", sorted(LEADS))
def test_showdown_starts_on_the_terrain_set_last(oracle: Oracle, name: str) -> None:
    kept, flipped = _field(_lead(oracle, name, "keep")), _field(_lead(oracle, name, "reverse"))
    if name.startswith("control"):
        assert kept == flipped == ("grassyterrain", 5)
    else:
        assert {kept, flipped} == {("grassyterrain", 5), ("psychicterrain", 5)}


@pytest.mark.parametrize("name", sorted(LEADS))
def test_the_port_draws_the_leads_tie(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """Each answer to the draw starts on one of Showdown's two fields, at even odds; the
    tie that changes nothing is not a draw at all."""
    from pokeuraou.selfplay import position_from_sets

    ours, theirs = LEADS[name]
    showdown = {_field(_lead(oracle, name, tie)) for tie in ("keep", "reverse")}
    ported = set()
    for index in (0, 1):
        pick = _Pick(index)
        ported.add(_field(position_from_sets(reg, _sampled(ours), _sampled(theirs), rng=pick)))
        if name.startswith("control"):
            assert pick.asked == []
        else:
            assert pick.asked == [(2, [0.5, 0.5])]
    assert ported == showdown


def test_the_port_notes_the_leads_tie_it_cannot_draw(reg) -> None:  # noqa: ANN001
    """Without a generator the first order is taken, and said to be."""
    from pokeuraou import port
    from pokeuraou.selfplay import _opening

    ours, theirs = LEADS["grassy-against-psychic"]
    opening = _opening(reg, _sampled(ours), _sampled(theirs))
    notes = port.apply_lead_abilities(reg, opening).unmodelled
    assert "switch-in speed tie (the first; not branched)" in notes


@pytest.mark.parametrize("name", sorted(LEADS))
def test_the_selection_solve_reads_every_outcome_of_the_leads_tie(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """A caller with no generator -- the selection solve's openings -- gets both fields at
    a half each, and `positions_from_sets` still the first of them."""
    from pokeuraou.selfplay import lead_branches_from_sets, positions_from_sets

    ours, theirs = LEADS[name]
    showdown = {_field(_lead(oracle, name, tie)) for tie in ("keep", "reverse")}
    pairs = [(_sampled(ours), _sampled(theirs))]
    [branches] = lead_branches_from_sets(reg, pairs)
    assert {_field(pos) for _w, pos in branches} == showdown
    assert sum(w for w, _pos in branches) == pytest.approx(1.0)
    assert len(branches) == len(showdown)
    [first] = positions_from_sets(reg, pairs)
    assert first.to_json() == branches[0][1].to_json()


def _replacement_case(reg):  # noqa: ANN001, ANN202
    """Both leads down, Rillaboom and Indeedee coming in at one Speed."""
    from pokeuraou.actions import switch_actions_after_faint
    from pokeuraou.selfplay import position_from_sets

    ours = [SNEASLER, KINGAMBIT, RILLABOOM]
    theirs = [SNEASLER, KINGAMBIT, INDEEDEE]
    pos = position_from_sets(reg, _sampled(ours), _sampled(theirs))
    for side in pos.sides:
        lead = side.pokemon[0]
        lead.hp = 0
        lead.fainted = True
    chosen = []
    for side in range(2):
        options = switch_actions_after_faint(reg, pos, side, [True, False])
        chosen.append(next(a for a in options if a.to_choice() == "switch 3, pass"))
    return pos, chosen


def test_the_replacement_search_reads_both_outcomes_of_the_tie(reg) -> None:  # noqa: ANN001
    """The replacement node's matrix (no generator) prices the cell at the mean of the two
    fields -- the same `runSwitch` shuffle as the leads' (the Showdown fact above)."""
    from pokeuraou import port
    from pokeuraou.selfplay import replacement_matrix

    pos, chosen = _replacement_case(reg)
    branches = port.replacement_branches(reg, pos, chosen)
    assert sorted((w, _field(phase.position)) for w, phase in branches) == [
        (0.5, ("grassyterrain", 5)), (0.5, ("psychicterrain", 5))
    ]

    def by_terrain(positions):  # noqa: ANN001, ANN202
        return np.array([1.0 if p.field.terrain == "grassyterrain" else 0.0 for p in positions])

    cell = replacement_matrix(reg, pos, [[chosen[0]], [chosen[1]]], by_terrain)
    assert cell.tolist() == [[0.5]]


# ---------------------------------------------------------------------------
# The residual phase: Perish Song takes all four, and the side whose last Pokemon faints
# last wins (`checkWin` on the last of the faint queue). The two Venusaurs are the slowest
# and tie, so the tie decides the game. Turn 1 sings; turns 2-4 wait.

VENUSAUR = _mon("Venusaur", "Overgrow", ["swordsdance", "sludgebomb"], {"hp": 32, "spa": 32})
FAST = {"hp": 2, "atk": 32, "spe": 32}
PERISH = (
    [VENUSAUR, _mon("Garchomp", "Rough Skin", ["perishsong", "swordsdance"], FAST)],
    [VENUSAUR, _mon("Weavile", "Pressure", ["swordsdance", "iceshard"], FAST)],
)
SING = ["move 1, move 1", "move 1, move 1"]
WAIT = ["move 1, move 2", "move 1, move 1"]
PERISH_STEPS = [SING, WAIT, WAIT, WAIT]


def _perish(oracle: Oracle, tie: str) -> tuple[Position, Position]:
    handle = oracle.create(FORMAT_ID, *PERISH, policy=RandomnessPolicy(speed_tie=tie))
    handle.step(["team 12", "team 12"])
    for step in PERISH_STEPS[:-1]:
        handle.step(step)
        assert handle.choice_errors == [], handle.choice_errors
    before = _loaded(handle.position)
    handle.step(PERISH_STEPS[-1])
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    handle.close()
    return before, after


def test_showdown_lets_the_residual_tie_decide_the_game(oracle: Oracle) -> None:
    winners = {_perish(oracle, tie)[1].winner for tie in ("keep", "reverse")}
    assert winners == {"p1", "p2"}


def test_the_port_branches_the_residual_tie(reg, oracle: Oracle) -> None:  # noqa: ANN001
    played = [_perish(oracle, tie) for tie in ("keep", "reverse")]
    showdown = _spread([_board(after) for _before, after in played])
    before = played[0][0]
    result = resolve_turn(reg, before, _chosen(reg, before, PERISH_STEPS[-1]), budget=BUDGET)
    assert _port_spread(result) == showdown
    assert "residual speed tie (Showdown breaks it at random)" not in result.unmodelled


def test_a_budget_without_ties_still_notes_the_residual_tie(reg, oracle: Oracle) -> None:  # noqa: ANN001
    before, _after = _perish(oracle, "keep")
    pinned = replace(BUDGET, enumerate_speed_ties=False)
    result = resolve_turn(reg, before, _chosen(reg, before, PERISH_STEPS[-1]), budget=pinned)
    assert len(result.branches) == 1
    assert "residual speed tie (Showdown breaks it at random)" in result.unmodelled
