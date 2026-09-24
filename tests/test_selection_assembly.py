"""The selection solve builds each Pokemon once and encodes each once (IKA-267).

A solve is 8,100 turn-1 positions of the same twelve sets. It used to make a fresh
`Pokemon` (stats from SP included) for every slot of every position, 97,200 of them, and
encode every one of those rows again. Both are now done once per object, and the answer
must not move: the positions and the arrays are compared with the ones built and encoded
one by one, and the tests show the sharing really happens, so equality is not a
comparison between two copies of the old path.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder, EncodingRules
from pokeuraou.position import Effect
from pokeuraou.selfplay import position_from_sets, positions_from_sets
from pokeuraou.teams import all_selections, load_roster

ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


@pytest.fixture(scope="module")
def solve_positions():  # noqa: ANN201
    """The positions of a slice of one solve: our six against place1's six."""
    ours = load_roster("rizabanadohido")
    theirs = load_roster("place1")
    reg = ours.reg
    register_mega_stones(reg)
    selections = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
    # Every lead quartet of the first few selections with nine of the backs, so quartets
    # share leads and openings share back Pokemon, as in a whole solve.
    pairs = [
        ([ours.sets[i] for i in a], [theirs.sets[j] for j in b])
        for a in selections[:12]
        for b in selections[:9]
    ]
    return reg, pairs, positions_from_sets(reg, pairs)


def _encoded_equal(left, right) -> None:  # noqa: ANN001
    for name in ARRAYS:
        a, b = getattr(left, name), getattr(right, name)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert np.array_equal(a, b), name
    assert left.unknown_volatiles == right.unknown_volatiles


def test_the_shared_positions_are_the_ones_built_one_by_one(solve_positions) -> None:  # noqa: ANN001
    reg, pairs, positions = solve_positions
    alone = [position_from_sets(reg, own, foe) for own, foe in pairs]
    assert [p.to_json() for p in positions] == [p.to_json() for p in alone]


def test_the_back_pokemon_are_shared_between_openings(solve_positions) -> None:  # noqa: ANN001
    """The positive control for the test above: one object per (side, slot, set)."""
    reg, pairs, positions = solve_positions
    active = reg.meta.active_per_side
    back = {id(m) for p in positions for side in p.sides for m in side.pokemon[active:]}
    # One per (side, back slot, set), and the port's own answer for each quartet's first.
    quartets = {
        (tuple(id(s) for s in own[:active]), tuple(id(s) for s in foe[:active]))
        for own, foe in pairs
    }
    assert len(back) <= 2 * 2 * 6 + 2 * 2 * len(quartets)
    # Unshared, every back slot of every position is its own object (432 here).
    assert len(positions) * 4 > 10 * len(back), "nothing is shared; nothing was tested"


def test_encoding_shared_pokemon_equals_encoding_copies(solve_positions) -> None:  # noqa: ANN001
    reg, _pairs, positions = solve_positions
    encoder = Encoder(reg)
    _encoded_equal(
        encoder.encode_positions(positions),
        encoder.encode_positions([copy.deepcopy(p) for p in positions]),
    )


def _variants(reg, positions):  # noqa: ANN001, ANN202
    """The same Pokemon objects under sides that differ in what a row reads off the side.

    Half the positions have side 0's mega spent and a third have its slot list emptied, so
    the stone holder's one object has `can_mega` both ways; and two share a leading
    Pokemon with two volatiles, one named and one not.
    """
    out = []
    for b, position in enumerate(positions):
        sides = list(position.sides)
        if b % 2:
            sides[0] = replace(sides[0], mega_used=True)
        if b % 3 == 0:  # what revision 1 reads instead of the stone
            sides[0] = replace(sides[0], mega_capable_slots=[])
        out.append(replace(position, sides=sides))
    lead = out[3].sides[1].pokemon[0]
    marked = replace(
        lead, volatiles=[Effect(id="confusion"), Effect(id="notavolatile")]
    )
    for b in (3, 5):  # the same object twice, so a memoised row would count it once
        side = out[b].sides[1]
        out[b] = replace(
            out[b],
            sides=[out[b].sides[0], replace(side, pokemon=[marked, *side.pokemon[1:]])],
        )
    holders = {
        m.species for p in out for m in p.sides[0].pokemon
        if reg.mega_target(m.species, m.item) is not None
    }
    assert holders, "no stone holder in the fixture; the mega key is untested"
    return out


@pytest.mark.parametrize("rules", [EncodingRules(), EncodingRules(mega_from_slots=True)])
def test_a_shared_pokemon_is_encoded_for_its_own_side(solve_positions, rules) -> None:  # noqa: ANN001
    """A row reads `mega_used` (and under revision 1 the slot list) off the side, so a
    Pokemon object met under two sides is two rows. Checked against deep copies, and the
    rows are shown to differ, so a key that ignored the side would fail here."""
    reg, _pairs, positions = solve_positions
    varied = _variants(reg, positions)
    encoder = Encoder(reg, rules=rules)
    shared = encoder.encode_positions(varied)
    _encoded_equal(shared, encoder.encode_positions([copy.deepcopy(p) for p in varied]))
    can_mega = encoder.mon_names.index("can_mega")
    assert len(set(shared.mon[:, 0, :, can_mega].ravel().tolist())) == 2
    assert shared.unknown_volatiles == {"notavolatile": 2}
