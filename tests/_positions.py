"""Positions several port tests read. They lived in `tests/test_ladder.py`, which IKA-464
removed with the ladder, and in a test of the depth-2 read's served fills, which IKA-464
removed with that read."""

from __future__ import annotations

import numpy as np

from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.position import Position
from pokeuraou.selfplay import position_from_sets
from pokeuraou.setup import parse_scenario

from ._port import resolve_turn


def played(roster, turns: int = 4, seed: int = 3) -> list[Position]:  # noqa: ANN001
    """Positions from one game of two different fours (a mirror is flat under hp-share)."""
    reg = roster.reg
    pos = position_from_sets(reg, list(roster.sets[:4]), list(roster.sets[2:6]))
    rng = np.random.default_rng(seed)
    out: list[Position] = []
    for _ in range(turns):
        if pos.ended:
            break
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            break
        out.append(pos.copy())
        result = resolve_turn(
            reg, pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        pos = max(result.branches, key=lambda b: b.probability).position
    return out


def _filler(species: str, ability: str, nature: str, moves: list[str]) -> dict:
    return {"species": species, "ability": ability, "nature": nature, "moves": moves,
            "sp": {"hp": 1}, "hp": 0}


def chomp_gambit():  # noqa: ANN201
    """IKA-253's end (`tests/test_decided_leaf.py`, which needs torch): both at 1 HP."""
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
    data = {"regulation": "gen9championsvgc2026regmc", "turn": 1,
            "sides": [{"id": "p1", "team": sides[0]},
                      {"id": "p2", "spreadsKnown": True, "team": sides[1]}],
            "field": {}}
    sc = parse_scenario(data)
    register_mega_stones(sc.reg)
    pos = sc.position
    pp = {"swordsdance": 20, "earthquake": 12, "kowtowcleave": 12, "suckerpunch": 8}
    for side in pos.sides:
        for m in side.pokemon[0].moves:
            m.pp = m.maxpp = pp[m.id]
    return sc.reg, pos
