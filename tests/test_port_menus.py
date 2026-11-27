"""IKA-389: the pieces the port builds a child's Q menus from (`rust/src/qmenus.rs`, which the
port's deepening reads its children's menus with) against Python's -- the legal pools action
for action and in order (`side_actions`, `qhead.legal_pool`; read back by `portmenus._action`),
the Q's arrays byte for byte (`qrank._pool_arrays`), and the candidates' scores (`solve`'s
EVs, the mean where the game has no answer). No torch.

The comparison of whole menus with Python's (`portmenus.child_menus`) went with the depth-2
refine (IKA-464).
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from pokeuraou import portmenus, qhead, qrank, rustnode
from pokeuraou.actions import side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.equilibrium import EquilibriumError, solve
from pokeuraou.position import Position
from pokeuraou.teams import load_roster

from ._positions import chomp_gambit as _chomp_gambit
from ._positions import played as _played


def _variant(pos: Position, edit) -> Position:  # noqa: ANN001
    data = pos.to_json()
    edit(data)
    return Position.from_json(data)


def _active(data: dict, side: int, slot: int = 0) -> dict:
    return data["sides"][side]["pokemon"][data["sides"][side]["active"][slot]]


def _volatile(vid: str, **extra) -> dict:  # noqa: ANN003
    return {"id": vid, **extra}


def _variants(pos: Position) -> list[Position]:
    """The rules `slot_actions` and `_usable_move_slots` read, each set on the position."""
    first = pos.sides[0].pokemon[pos.sides[0].active[0]].moves[0].id
    edits = [
        lambda d: _active(d, 0)["volatiles"].append(_volatile("taunt")),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("encore", move=first)),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("disable", move=first)),
        lambda d: (_active(d, 0)["volatiles"].append(_volatile("torment")),
                   _active(d, 0).update(lastMove=first)),
        lambda d: (_active(d, 0)["volatiles"].append(_volatile("choicelock", move=first)),
                   _active(d, 0).update(item="choicescarf")),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("throatchop")),
        lambda d: _active(d, 1)["volatiles"].append(_volatile("imprison")),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("partiallytrapped")),
        lambda d: _active(d, 0).update(trapped=True),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("mustrecharge")),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("lockedmove", move=first)),
        lambda d: _active(d, 0)["volatiles"].append(
            _volatile("twoturnmove", move=first, extra={"targetLoc": 1})),
        lambda d: _active(d, 0).update(lockedMove=first),
        lambda d: [m.update(pp=0) for m in _active(d, 0)["moves"]],
        lambda d: d["sides"][0].update(megaUsed=True),
        lambda d: _active(d, 0, 1).update(fainted=True, hp=0),
        lambda d: _active(d, 1).update(ability="shadowtag"),
        lambda d: _active(d, 1).update(ability="magnetpull"),
        lambda d: (_active(d, 1).update(ability="arenatrap"),
                   d["field"]["pseudoWeather"].append({"id": "gravity"})),
        lambda d: _active(d, 0).update(item="shedshell", trapped=False),
        lambda d: _active(d, 0).update(activeMoveActions=2),
        lambda d: d["sides"][1]["active"].__setitem__(1, None),
    ]
    return [_variant(pos, edit) for edit in edits]


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _positions(roster) -> list[tuple]:  # noqa: ANN001
    reg = roster.reg
    played = _played(roster)
    out = [(reg, p) for p in played]
    out += [(reg, v) for p in played[:2] for v in _variants(p)]
    mc_reg, chomp = _chomp_gambit()
    out += [(mc_reg, chomp)] + [(mc_reg, v) for v in _variants(chomp)]
    return out


def _by_reg(items: list[tuple]) -> dict:
    grouped: dict = {}
    for reg, pos in items:
        grouped.setdefault(reg.meta.format_id, (reg, []))[1].append(pos)
    return grouped


def test_the_ports_legal_pools_are_pythons(roster) -> None:  # noqa: ANN001
    """`side_actions` and `qhead.legal_pool` of each position, action by action in order.
    The positive control: the variants move the pools (a rule the port did not read would
    leave its pool as the base position's)."""
    moved = 0
    for reg, positions in _by_reg(_positions(roster)).values():
        node = rustnode.require_node(reg)
        raw = node.port_lists("legal", positions, each={"raw": True})["pools"]
        legal = node.port_lists("legal", positions)["pools"]
        for pos, raw_pools, pools in zip(positions, raw, legal, strict=True):
            for side in (0, 1):
                assert [portmenus._action(a) for a in raw_pools[side]] == side_actions(reg, pos, side)
                assert [portmenus._action(a) for a in pools[side]] == qhead.legal_pool(reg, pos, side)
        base = [a.to_choice() for a in side_actions(reg, positions[0], 0)]
        moved += sum(1 for p in positions[1:] if [a.to_choice() for a in side_actions(reg, p, 0)] != base)
    assert moved >= 20, moved


def test_the_ports_q_arrays_are_pythons(roster) -> None:  # noqa: ANN001
    """`qrank._pool_arrays` of each position with its legal pools, byte for byte."""
    compared = 0
    for reg, positions in _by_reg(_positions(roster)).values():
        encoder = Encoder(reg)
        got = rustnode.require_node(reg).port_lists(
            "qArrays", positions, extra={"properties": True, "megaFromSlots": False})["arrays"]
        for pos, arrays in zip(positions, got, strict=True):
            pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
            if not pools[0] or not pools[1]:
                continue
            want = qrank._pool_arrays(reg, encoder, pos, pools, True)
            assert sorted(arrays) == sorted(want)
            for name, array in want.items():
                assert arrays[name]["dtype"] == array.dtype.str
                assert list(arrays[name]["shape"]) == list(array.shape)
                assert base64.b64decode(arrays[name]["data"]) == np.ascontiguousarray(array).tobytes()
            compared += 1
    assert compared >= 20


def test_the_ports_scores_are_numpys(roster) -> None:  # noqa: ANN001
    """The candidates' scores: `solve`'s `row_ev` and minus its `col_ev`, and the means
    against every reply where the game has no answer. The port's sums are plain (IKA-460),
    so they agree to rounding, not to the bit."""
    rng = np.random.default_rng(7)
    games = [rng.random((int(rng.integers(2, 90)), int(rng.integers(2, 90)))) for _ in range(150)]
    games += [np.round(g, 1) for g in games[:40]] + [rng.random((30, 1)), rng.random((1, 30))]
    request = {"kind": "qScores", "games": [
        {"rows": g.shape[0], "cols": g.shape[1],
         "data": base64.b64encode(g.astype("<f8").tobytes()).decode()} for g in games]}
    node = rustnode.require_node(roster.reg)
    solved = node._exchange(request)["answers"]  # noqa: SLF001
    means = node._exchange({**request, "mean": True})["answers"]  # noqa: SLF001

    def unpack(text: str) -> np.ndarray:
        return np.frombuffer(base64.b64decode(text), "<f8")

    for g, got, mean in zip(games, solved, means, strict=True):
        m, n = g.shape
        np.testing.assert_allclose(unpack(mean["rows"]), g.mean(axis=1), rtol=0, atol=1e-12)
        np.testing.assert_allclose(unpack(mean["cols"]), -g.mean(axis=0), rtol=0, atol=1e-12)
        try:
            e = solve(g)
            want = (np.asarray(e.row_ev), -np.asarray(e.col_ev))
        except (EquilibriumError, ValueError):
            want = (g.mean(axis=1), -g.mean(axis=0))
        # One score orders nothing, and numpy takes `dot` rather than `gemv` there.
        if m > 1:
            np.testing.assert_allclose(unpack(got["rows"]), want[0], rtol=0, atol=1e-12)
        if n > 1:
            np.testing.assert_allclose(unpack(got["cols"]), want[1], rtol=0, atol=1e-12)
