"""tools/endgame_exact.py: IKA-254's game solved to the end, and its controls.

- Positive control: the Garchomp (HP 1) against Kingambit (HP 1) end of IKA-253, where
  Kingambit's Sucker Punch PP and Garchomp's Swords Dance make the game last nine
  decisions and its value is exactly 1/9. A one-turn reading of the same position is not.
- Null control: the tree cut at one turn over the search's own menu is `search(depth=1)`.
- The analysis: the solved strategies give up nothing in the solved game, a uniform pair
  does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.search import search
from pokeuraou.setup import parse_scenario

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("endgame_exact", ROOT / "tools" / "endgame_exact.py")
endgame_exact = importlib.util.module_from_spec(spec)
sys.modules["endgame_exact"] = endgame_exact
spec.loader.exec_module(endgame_exact)

REG = "gen9championsvgc2026regmc"


def _filler(species: str, ability: str, nature: str, moves: list[str]) -> dict:
    return {"species": species, "ability": ability, "nature": nature, "moves": moves,
            "sp": {"hp": 1}, "hp": 0}


def _chomp_gambit(sucker_pp: int = 8):  # noqa: ANN202
    """IKA-253's end: both at 1 HP, Garchomp (SD/EQ, faster) against Kingambit (KC/SP)."""
    chomp = {"species": "Garchomp", "ability": "roughskin", "nature": "Jolly",
             "moves": ["Swords Dance", "Earthquake"], "sp": {"atk": 32, "spe": 32},
             "active": True, "hp": 1}
    gambit = {"species": "Kingambit", "ability": "defiant", "nature": "Adamant",
              "moves": ["Kowtow Cleave", "Sucker Punch"], "sp": {"hp": 32, "atk": 32},
              "active": True, "hp": 1}
    sides = []
    for lead in (chomp, gambit):
        dead = _filler("Incineroar", "intimidate", "Impish", ["Fake Out"])
        dead["active"] = True
        sides.append([lead, dead, _filler("Sinistcha", "hospitality", "Bold", ["Protect"]),
                      _filler("Farigiraf", "armortail", "Calm", ["Protect"])])
    data = {"regulation": REG, "turn": 1,
            "sides": [{"id": "p1", "team": sides[0]},
                      {"id": "p2", "spreadsKnown": True, "team": sides[1]}],
            "field": {}}
    sc = parse_scenario(data)
    register_mega_stones(sc.reg)
    pos = sc.position
    pp = {"swordsdance": 20, "earthquake": 12, "kowtowcleave": 12, "suckerpunch": sucker_pp}
    for side in pos.sides:
        for m in side.pokemon[0].moves:
            m.pp = m.maxpp = pp[m.id]
    return sc.reg, pos


@pytest.mark.parametrize("sucker_pp", [1, 2, 8])
def test_the_chomp_gambit_end_is_one_over_pp_plus_one(sucker_pp: int) -> None:
    """Positive control: 1/(PP+1) -- 1/9 at 8 PP, as IKA-253 solved it by hand."""
    reg, pos = _chomp_gambit(sucker_pp)
    root, spent = endgame_exact.solve_to_end(
        reg, pos, Budget.matrix(), eps=1e-9, max_turns=20, max_nodes=5_000
    )
    eq, payoff, rows, cols = root.equilibrium, root.payoff, root.rows, root.cols
    assert eq.value == pytest.approx(1 / (sucker_pp + 1), abs=1e-9)
    assert root.lo == pytest.approx(root.hi, abs=1e-9)
    # The bounds meet only once the cut is past the last Sucker Punch.
    assert spent["turns"] == sucker_pp + 1
    # A cut one turn short leaves the answer open: the [0, 1] bound is doing the work.
    short = endgame_exact.Tree(reg, Budget.matrix(), cap=sucker_pp).solve_root(pos)
    assert short.hi - short.lo > 0.01 and short.lo <= eq.value <= short.hi
    # The solved strategies give up nothing in their own game; a pure pair does.
    pool = [[a.to_choice() for a in rows], [a.to_choice() for a in cols]]
    exact = {
        "value": eq.value,
        "payoff": payoff.tolist(),
        "row": endgame_exact.strategy(rows, eq.row_strategy),
        "col": endgame_exact.strategy(cols, eq.col_strategy),
    }
    assert endgame_exact.compare(exact, exact, pool)["exploit"] == pytest.approx(0, abs=1e-9)
    pure = {"value": 0.5, "row": {pool[0][0]: 1.0}, "col": {pool[1][0]: 1.0}}
    got = endgame_exact.compare(pure, exact, pool)
    assert got["exploit"] > 0.05 and got["err"] == pytest.approx(0.5 - eq.value)


def test_the_caps_stop_the_tree_rather_than_guess() -> None:
    reg, pos = _chomp_gambit(8)
    with pytest.raises(endgame_exact.Capped):
        endgame_exact.solve_to_end(reg, pos, Budget.matrix(), eps=1e-9, max_turns=3, max_nodes=5_000)
    with pytest.raises(endgame_exact.Capped):
        endgame_exact.solve_to_end(reg, pos, Budget.matrix(), eps=1e-9, max_turns=20, max_nodes=4)


def test_a_one_turn_horizon_over_the_menu_is_the_depth_one_search() -> None:
    """Null control: `Tree(horizon=1)` fills the matrix `search(depth=1)` fills."""
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import load_roster

    roster = load_roster("rizabanadohido")
    register_mega_stones(roster.reg)
    reg = roster.reg
    cases = [
        (reg, position_from_sets(reg, list(roster.sets[:4]), list(roster.sets[2:6]))),
        _chomp_gambit(3),
    ]
    for at, pos in cases:
        ours = narrow(at, pos, 0, limit=6).actions
        theirs = narrow(at, pos, 1, limit=6).actions
        want = search(at, pos, ours, theirs, HP_SHARE.batch, budget=Budget.matrix(), depth=1)
        tree = endgame_exact.Tree(at, Budget.matrix(), horizon=1, leaf=HP_SHARE.batch)
        root = tree.solve_root(pos, (ours, theirs))
        np.testing.assert_allclose(root.payoff_lo, want.payoff, rtol=0, atol=1e-12)
        assert np.array_equal(root.payoff_lo, root.payoff_hi)
        assert root.equilibrium.value == pytest.approx(want.equilibrium.value, abs=1e-12)
        assert tree.nodes == 1 and tree.leaves > 0
    # The comparison can fail: a second turn over the same menu is another matrix.
    at, pos = cases[1]
    ours, theirs = narrow(at, pos, 0, limit=6).actions, narrow(at, pos, 1, limit=6).actions
    want = search(at, pos, ours, theirs, HP_SHARE.batch, budget=Budget.matrix(), depth=1)
    two = endgame_exact.Tree(at, Budget.matrix(), horizon=2, leaf=HP_SHARE.batch)
    assert not np.allclose(two.solve_root(pos, (ours, theirs)).payoff_lo, want.payoff)


def test_the_fixed_leaf_decides_only_ended_positions() -> None:
    reg, pos = _chomp_gambit(1)
    ended = pos.copy()
    ended.ended = True
    ended.winner = pos.sides[1].id
    fixed = endgame_exact.with_ends_decided(lambda ps: np.full(len(ps), 0.49))
    assert fixed([pos, ended]).tolist() == [0.49, 0.0]


def test_pipelined_turns_are_the_ports_turns() -> None:
    from pokeuraou import port

    reg, pos = _chomp_gambit(3)
    rows = endgame_exact.legal(reg, pos, 0)
    cols = endgame_exact.legal(reg, pos, 1)
    pairs = [(r, c) for r in rows for c in cols]
    got = endgame_exact.turns(reg, pos, pairs, Budget.matrix())
    assert len(got) == len(pairs) > 1
    for (r, c), mine in zip(pairs, got, strict=True):
        want = port.turn(reg, pos, [r, c], Budget.matrix(), full=True)
        assert [b.probability for b in mine.outcomes] == [b.probability for b in want.outcomes]
        assert [b.position.to_json() for b in mine.outcomes] == [
            b.position.to_json() for b in want.outcomes
        ]
