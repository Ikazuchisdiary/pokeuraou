"""Sharing one resolution across completions, and the claim that lets it.

The claim is that an unrevealed benched Pokemon takes no part in a turn it is not switched
into, so the cells where none of them reaches the field resolve identically in every
completion and their leaves differ only in who is on the bench. If that is wrong -- a
mechanic that counts party members, a feature that reads the bench -- the payoffs stay
plausible and are wrong, and nothing downstream could notice.

So the test is the definition: `belief_payoffs` against a matrix per completion resolved
from scratch, every cell, exactly. Not almost: a payoff that differs in its last places is
an equilibrium that differs, and the whole point of this path is that it changes nothing
but the time.

It needs the Rust port, because the fast path is only taken when the port is there to be
compared against. Without it the test skips rather than passing vacuously -- a green run
on a machine where the code under test never executed is worse than a red one.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.beliefnode import _patched, belief_payoffs, reaches_bench
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.regulation import load_regulation, to_id
from pokeuraou.resolve import Budget, batched_payoff
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    if not rustnode.available():
        pytest.skip("the shared path is only taken with the port; nothing to compare")
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    roster = load_roster("rizabanadohido")
    sheet = list(roster.sets)[:6]
    position = position_from_sets(reg, sheet[:4], sheet[:4])
    return reg, sheet, position


class _Leaf:
    """A deterministic stand-in for the learned net, scored from the encoding.

    It has to score `Encoded` rather than positions, because that is what selects the
    shared path -- an evaluator without `from_encoded` falls back and the test would
    compare the slow path with itself. The arithmetic is arbitrary and reproducible; what
    matters is that it reads every array the net reads, so a patch that missed one shows up.

    Until IKA-119 it read the six per-Pokemon arrays and not `side` or `field`, and for as
    long as that was so `_patched` handed every completion the true bench's side vector
    without a test noticing: a leaf that does not read an array cannot see a leak in it.
    """

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def from_encoded(self, encoded) -> np.ndarray:  # noqa: ANN001
        n = len(encoded)
        parts = (
            encoded.species.reshape(n, -1).sum(axis=1) * 0.7,
            encoded.ability.reshape(n, -1).sum(axis=1) * 0.3,
            encoded.item.reshape(n, -1).sum(axis=1) * 0.11,
            encoded.moves.reshape(n, -1).sum(axis=1) * 0.013,
            encoded.mon.reshape(n, -1).sum(axis=1).astype(np.float64) * 0.017,
            encoded.mask.reshape(n, -1).sum(axis=1).astype(np.float64) * 0.5,
            # Weighted by column, so two features that move in opposite directions between
            # completions cannot cancel the way they could in a plain sum.
            encoded.side.reshape(n, -1).astype(np.float64)
            @ np.linspace(0.3, 1.7, encoded.side[0].size),
            encoded.field.reshape(n, -1).astype(np.float64)
            @ np.linspace(0.2, 1.1, encoded.field.shape[1]),
        )
        raw = sum(parts)
        return 1.0 / (1.0 + np.exp(-(raw % 7.0) + 3.0))

    def __call__(self, positions: list) -> np.ndarray:
        return self.from_encoded(self.encoder.encode_positions(positions))


def _menus(reg, position, limit):  # noqa: ANN001, ANN202
    return (
        narrow(reg, position, 0, limit=limit).actions,
        narrow(reg, position, 1, limit=limit).actions,
    )


def test_the_shared_node_equals_a_matrix_per_completion(setup) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 10)
    leaf = _Leaf(Encoder(reg))
    budget = Budget.matrix()
    spreads = {
        0: completions(reg, position, 0, sheet),
        1: completions(reg, position, 1, sheet),
    }
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=budget, spreads=spreads
    )
    assert node.shared > 0, "the fast path did not run; the comparison would be vacuous"

    for side, items in spreads.items():
        built = node.matrices[1 - side]
        assert len(built) == len(items)
        for index, item in enumerate(items):
            expected, _notes = batched_payoff(
                reg, item.position, ours, theirs, leaf, budget=budget
            )
            np.testing.assert_allclose(
                built[index], expected, rtol=0, atol=0,
                err_msg=f"side {side} completion {index} ({', '.join(item.species)})",
            )


def test_it_shares_most_of_the_node(setup) -> None:  # noqa: ANN001
    """If almost nothing is shared the path is correct and pointless, and a change that
    made it so would otherwise pass every other test here."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 10)
    leaf = _Leaf(Encoder(reg))
    spreads = {
        0: completions(reg, position, 0, sheet),
        1: completions(reg, position, 1, sheet),
    }
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads
    )
    cells = len(ours) * len(theirs)
    per_completion = sum(len(items) for items in spreads.values()) * cells
    assert node.redone < 0.4 * per_completion, (
        f"{node.redone} of {per_completion} cell-resolutions repeated; "
        "the sharing is not paying for itself"
    )


def test_a_switch_into_a_hidden_slot_is_not_shared(setup) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 24)
    hidden = {0: (2, 3), 1: (2, 3)}
    mask = reaches_bench(reg, ours, theirs, hidden)
    # 1-based in the action, 0-based in the slot. Spelt out here because comparing them
    # directly is what the first version did, and the mask was wrong in both directions.
    wanted = [
        j for j, action in enumerate(theirs)
        if any(index - 1 in hidden[1] for index in action.switch_indices)
    ]
    assert wanted, "this position offers no switch into a hidden slot to test with"
    for j in wanted:
        assert mask[:, j].all(), f"column {j} switches into a hidden slot and was shared"


def test_nothing_hidden_costs_nothing(setup) -> None:  # noqa: ANN001
    """Every slot seen means one completion a side and the old node, unchanged."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 8)
    leaf = _Leaf(Encoder(reg))
    seen = frozenset({0, 1, 2, 3})
    spreads = {
        side: completions(reg, position, side, sheet, seen=seen) for side in (0, 1)
    }
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads
    )
    expected, _notes = batched_payoff(
        reg, position, ours, theirs, leaf, budget=Budget.matrix()
    )
    for side in (0, 1):
        assert len(node.matrices[side]) == 1
        np.testing.assert_allclose(node.matrices[side][0], expected, rtol=0, atol=0)


# ------------------------------------------------------------------- IKA-119 --
#
# `_patched` overwrote the per-Pokemon rows of the hidden slots and left `side` as the true
# position's. Two side features read the bench -- `mega_available` (is there a stone on
# the side) and `team_hp_fraction` (the bench's max HP is in the denominator) -- so every
# completion was scored with the true bench's values. And the rows came from encoding the
# completion's *root*, so a bench holder's `can_mega` stayed 1 in a leaf where the side had
# mega evolved during the turn. Neither moved in the fixture above: no hidden candidate
# held a stone and nothing was damaged.

_ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def _differences(got, want) -> dict[str, int]:  # noqa: ANN001
    """Every array, counted -- a check that reads only some of them is the one that missed."""
    return {
        name: int((getattr(got, name) != getattr(want, name)).sum())
        for name in _ARRAYS
        if not np.array_equal(getattr(got, name), getattr(want, name))
    }


def _pick(sheet, *species):  # noqa: ANN001, ANN002, ANN202
    names = [to_id(entry.species) for entry in sheet]
    return [sheet[names.index(name)] for name in species]


def _with_stone(reg, sheet, species, stone):  # noqa: ANN001, ANN202
    """The sheet with a second mega stone on it, as a two-stone sheet has (G25)."""
    names = [to_id(entry.species) for entry in sheet]
    index = names.index(species)
    assert reg.mega_target(sheet[index].species, stone) is not None, (species, stone)
    out = list(sheet)
    out[index] = dataclasses.replace(out[index], item=stone)
    return out


def test_a_patched_leaf_carries_its_own_side_vector(setup) -> None:  # noqa: ANN001
    """Charizard (the only stone) and Toxapex behind a Venusaur at half HP."""
    reg, sheet, _position = setup
    position = position_from_sets(
        reg,
        _pick(sheet, "incineroar", "sylveon", "garchomp", "venusaur"),
        _pick(sheet, "venusaur", "garchomp", "charizard", "toxapex"),
    )
    lead = position.sides[1].pokemon[position.sides[1].active[0]]
    lead.hp = lead.maxhp // 2
    encoder = Encoder(reg)
    reference = encoder.encode_positions([position])
    items = completions(reg, position, 1, sheet, seen=frozenset({0, 1}))
    own = [encoder.encode_positions([item.position]) for item in items]

    # The control: in this position both features depend on who is on the bench, so a
    # patch that left them alone has something to get wrong.
    for feature in ("mega_available", "team_hp_fraction"):
        k = encoder.side_names.index(feature)
        seen_values = {float(encoded.side[0, 1, k]) for encoded in own}
        assert len(seen_values) > 1, f"{feature} does not depend on the bench here"

    for item, want in zip(items, own, strict=True):
        got = _patched(reference, 1, item.slots, item, reg, position)
        assert not _differences(got, want), (
            f"{'+'.join(item.species)}: {_differences(got, want)}"
        )


def test_a_patched_leaf_knows_the_side_mega_evolved_this_turn(setup) -> None:  # noqa: ANN001
    """Venusaurite on the bench, and the active Charizard megas in the leaf."""
    reg, sheet, _position = setup
    sheet = _with_stone(reg, sheet, "venusaur", "venusaurite")
    position = position_from_sets(
        reg,
        _pick(sheet, "incineroar", "sylveon", "garchomp", "toxapex"),
        _pick(sheet, "charizard", "garchomp", "sylveon", "venusaur"),
    )

    def after_mega(root):  # noqa: ANN001, ANN202
        leaf = root.copy()
        leaf.sides[1].mega_used = True
        leaf.sides[1].pokemon[leaf.sides[1].active[0]].is_mega = True
        return leaf

    encoder = Encoder(reg)
    can_mega = encoder.mon_names.index("can_mega")
    reference = encoder.encode_positions([after_mega(position)])
    assert not position.sides[1].mega_used, "after_mega wrote into the root"
    items = completions(reg, position, 1, sheet, seen=frozenset({0, 1}))

    # The control: a root row does say "can still mega" for a bench Venusaur, so a leaf
    # row copied from the root is wrong there.
    stale = [
        item for item in items
        if any(
            encoder.encode_positions([item.position]).mon[0, 1, slot, can_mega]
            for slot in item.slots
        )
    ]
    assert stale, "no completion's root offers a bench mega; the leaf cannot differ"

    for item in items:
        got = _patched(reference, 1, item.slots, item, reg, position)
        want = encoder.encode_positions([after_mega(item.position)])
        assert not _differences(got, want), (
            f"{'+'.join(item.species)}: {_differences(got, want)}"
        )

    # IKA-105 leans on "the true bench, patched, is the reference itself". Before IKA-119
    # it was not whenever the side mega'd during the turn: the true Venusaur's row said it
    # could still mega. `completions` may order the two bench slots differently from the
    # truth (the order is not a degree of freedom), so rows are matched by species.
    true_bench = {to_id(mon.species) for mon in position.sides[1].pokemon[2:]}
    truth = [item for item in items if set(item.species) == true_bench]
    assert truth, f"the true bench {sorted(true_bench)} is not among the completions"
    assert truth[0] in stale, "the true bench offers no stale row; nothing to show"
    patched_truth = _patched(reference, 1, truth[0].slots, truth[0], reg, position)
    np.testing.assert_array_equal(patched_truth.side, reference.side)
    for slot in truth[0].slots:
        species = patched_truth.species[0, 1, slot]
        (same,) = np.flatnonzero(reference.species[0, 1] == species)
        np.testing.assert_array_equal(
            patched_truth.mon[0, 1, slot], reference.mon[0, 1, same], err_msg=str(species)
        )


def test_the_shared_node_equals_a_matrix_per_completion_with_a_mega_and_damage(
    setup,  # noqa: ANN001
) -> None:
    """The first test again, on a position where the side vector and a mega can move.

    Side 1 leads Charizard (stone) and a damaged Garchomp; the sheet carries a second stone
    on Venusaur, so half the completions hold one in the back. The mega is declared through
    the resolver, not written into a leaf by hand.
    """
    reg, sheet, _position = setup
    sheet = _with_stone(reg, sheet, "venusaur", "venusaurite")
    position = position_from_sets(
        reg,
        _pick(sheet, "incineroar", "sylveon", "garchomp", "toxapex"),
        _pick(sheet, "charizard", "garchomp", "sylveon", "venusaur"),
    )
    lead = position.sides[1].pokemon[position.sides[1].active[1]]
    lead.hp = lead.maxhp // 2
    ours, theirs = _menus(reg, position, 10)
    assert any(action.declares_mega for action in theirs), "side 1's menu has no mega"
    leaf = _Leaf(Encoder(reg))
    budget = Budget.matrix()
    spreads = {
        0: completions(reg, position, 0, sheet),
        1: completions(reg, position, 1, sheet),
    }
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=budget, spreads=spreads
    )
    assert node.shared > 0, "the fast path did not run; the comparison would be vacuous"

    for side, items in spreads.items():
        built = node.matrices[1 - side]
        for index, item in enumerate(items):
            expected, _notes = batched_payoff(
                reg, item.position, ours, theirs, leaf, budget=budget
            )
            np.testing.assert_allclose(
                built[index], expected, rtol=0, atol=0,
                err_msg=f"side {side} completion {index} ({', '.join(item.species)})",
            )
