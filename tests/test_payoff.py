"""The objectives the equilibrium is computed over.

These are short because the objectives are deliberately short: parameter-free functions
of the position. What is worth pinning is that they stay in [0, 1] so they can occupy the
slot a win probability will later occupy, that a decided battle reads 1 or 0 rather than
merely a large number, and that each one actually moves with the thing it claims to
measure -- an objective that is constant is an objective that cannot rank anything.
"""

from __future__ import annotations

from pokeuraou.oracle import TeamSet
from pokeuraou.payoff import FAINTS, HP_SHARE, OBJECTIVES
from pokeuraou.regulation import Regulation

from .test_actions import _synthetic_position


def test_a_mirror_position_is_even(reg: Regulation, team_a: list[TeamSet]) -> None:
    """The synthetic fixture gives both sides the same team, so both objectives sit at 0.5."""
    pos = _synthetic_position(reg, team_a)
    for objective in OBJECTIVES.values():
        assert objective(pos) == 0.5


def test_damage_moves_hp_share_but_not_faints(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Which is the whole reason both exist: they fail in opposite directions."""
    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[1].pokemon[pos.sides[1].active[0]]
    mon.hp = 1

    assert HP_SHARE(pos) > 0.5, "chip damage has to register somewhere"
    assert FAINTS(pos) == 0.5, "faints cannot see a Pokemon on 1 HP, by design"

    mon.hp = 0
    mon.fainted = True
    assert FAINTS(pos) > 0.5


def test_every_objective_stays_in_range(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    for mon in pos.sides[1].pokemon:
        mon.hp = 0
        mon.fainted = True
    for objective in OBJECTIVES.values():
        value = objective(pos)
        assert 0.0 <= value <= 1.0
        assert value == 1.0, "one side has nothing left; that is as good as it gets"


def test_a_decided_battle_reads_one_or_zero(reg: Regulation, team_a: list[TeamSet]) -> None:
    """A win is a win, not a large HP lead.

    Without this a position won with one Pokemon at 3 HP would score below a position
    merely ahead on damage, and the equilibrium would prefer the latter.
    """
    pos = _synthetic_position(reg, team_a)
    survivor = pos.sides[0].pokemon[0]
    survivor.hp = 1
    pos.ended = True
    pos.winner = pos.sides[0].id
    for objective in OBJECTIVES.values():
        assert objective(pos) == 1.0

    pos.winner = pos.sides[1].id
    for objective in OBJECTIVES.values():
        assert objective(pos) == 0.0


def test_objectives_declare_what_they_cannot_see(reg: Regulation) -> None:
    """The CLI prints these strings next to the numbers, so they have to exist."""
    del reg
    for objective in OBJECTIVES.values():
        assert objective.formula.strip()
        assert objective.blind_to.strip()
