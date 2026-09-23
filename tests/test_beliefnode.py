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

from pokeuraou import beliefnode, rustnode
from pokeuraou.actions import side_actions
from pokeuraou.beliefnode import _patched, belief_payoffs, reaches_bench
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.regulation import load_regulation, to_id
from pokeuraou.resolve import Budget, batched_payoff, resolve_turn
from pokeuraou.search import belief_solve
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


class _Counted(_Leaf):
    """The same leaf, counting its forward passes: one per `from_encoded`, whichever
    crossing asked for it -- `__call__` goes through it too."""

    def __init__(self, encoder: Encoder) -> None:
        super().__init__(encoder)
        self.passes = 0

    def from_encoded(self, encoded) -> np.ndarray:  # noqa: ANN001
        self.passes += 1
        return super().from_encoded(encoded)


def _nothing_hidden(reg, position, sheet):  # noqa: ANN001, ANN202
    seen = frozenset({0, 1, 2, 3})
    spreads = {
        side: completions(reg, position, side, sheet, seen=seen) for side in (0, 1)
    }
    # What makes this case the one that was paid twice: each side's only completion is
    # the position itself, so both lists hold the same object.
    assert all(
        item.exact and item.position is position
        for items in spreads.values()
        for item in items
    )
    return spreads


def test_nothing_hidden_on_either_side_is_resolved_once(  # noqa: ANN001
    setup, monkeypatch
) -> None:
    """Both sides exact is one node, and it used to be resolved once per side (IKA-104).

    `_per_completion` called `batched_payoff` for each side's completion, and with nothing
    hidden on either side those are the same `Position`, the same menus and the same leaf
    -- the port's fill, the forward pass and the fold done twice for the same numbers, on
    37% of the move decisions of IKA-73's width-12 pool. The reference is the old
    definition itself: one `batched_payoff` per side, each compared to the bit.

    The menus are not square, so a matrix handed to side 1 transposed could not pass for
    the one it replaces.
    """
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 8)
    theirs = theirs[:6]
    assert len(ours) != len(theirs)
    budget = Budget.matrix()
    spreads = _nothing_hidden(reg, position, sheet)

    calls: list[object] = []
    unwrapped = beliefnode.batched_payoff

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(args[1])
        return unwrapped(*args, **kwargs)

    monkeypatch.setattr(beliefnode, "batched_payoff", counted)
    leaf = _Counted(Encoder(reg))
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=budget, spreads=spreads
    )
    assert len(calls) == 1, f"one node, resolved {len(calls)} times"
    assert calls[0] is position
    once = leaf.passes

    monkeypatch.setattr(beliefnode, "batched_payoff", unwrapped)
    reference = _Counted(Encoder(reg))
    for side in (0, 1):
        # Oriented to side 0 on both sides, as `BeliefNode.matrices` always is:
        # `belief_solve` is what turns side 1's into its own game, not this.
        expected, _notes = batched_payoff(
            reg, position, ours, theirs, reference, budget=budget
        )
        assert node.matrices[side][0].shape == (len(ours), len(theirs))
        assert np.array_equal(node.matrices[side][0], expected), f"side {side}"
    assert reference.passes == 2 * once, (
        f"{once} forward passes for the node against {reference.passes} for the two "
        "the old definition ran"
    )
    assert node.matrices[0][0] is not node.matrices[1][0], (
        "the two sides hold one array; a caller that edits its own would edit the other's"
    )
    assert node.redone == len(ours) * len(theirs)


def test_the_node_both_sides_share_gives_each_what_its_own_would(  # noqa: ANN001
    setup, monkeypatch
) -> None:
    """One level up, where the orientation is used.

    Side 1 solves the negated transpose of the matrix it is handed, so what it must be
    handed is side 0's payoff exactly as side 0 gets it. `belief_solve` takes the shared
    node only when both sides hold the SAME leaf object; with two leaves it builds a node
    per side, which is the path that never shared anything. The two have to agree to the
    bit for both sides -- strategy, value and reply -- and the calls are counted so that
    the comparison cannot quietly become the second path against itself.

    `tests/test_hidden_search.py` checks nothing-hidden against `search` with
    `HP_SHARE.batch` on both sides, which is two bound methods and so never took the
    shared path this is about.
    """
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 8)
    theirs = theirs[:6]
    budget = Budget.matrix()
    spreads = _nothing_hidden(reg, position, sheet)
    leaf = _Leaf(Encoder(reg))
    twin = _Leaf(Encoder(reg))

    calls: list[object] = []
    unwrapped = beliefnode.batched_payoff

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(args[1])
        return unwrapped(*args, **kwargs)

    monkeypatch.setattr(beliefnode, "batched_payoff", counted)
    shared = belief_solve(
        reg, position, ours, theirs, spreads, {0: leaf, 1: leaf}, budget=budget
    )
    assert len(calls) == 1, "one leaf for both sides did not take the shared node"
    apart = belief_solve(
        reg, position, ours, theirs, spreads, {0: leaf, 1: twin}, budget=budget
    )
    assert len(calls) == 3, "two leaves should have built a node per side"
    for side, width in ((0, len(ours)), (1, len(theirs))):
        together, alone = shared[side], apart[side]
        assert together.strategy.shape == (width,)
        assert np.array_equal(together.strategy, alone.strategy), f"side {side}"
        assert together.value == alone.value, f"side {side}"
        assert len(together.replies) == len(alone.replies) == 1
        assert np.array_equal(together.replies[0], alone.replies[0]), f"side {side}"
    assert shared[1].value == pytest.approx(-shared[0].value)


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


def _patched_before_ika119(reference, side, slots, item, encoder):  # noqa: ANN001, ANN202
    """`_patched` as it was at 4524ec4^, verbatim but for taking the encoder as an argument."""
    from pokeuraou.encode import Encoded

    source = encoder.encode_positions([item.position])
    out = Encoded(
        species=reference.species.copy(),
        ability=reference.ability.copy(),
        item=reference.item.copy(),
        moves=reference.moves.copy(),
        mon=reference.mon.copy(),
        mask=reference.mask.copy(),
        side=reference.side,
        field=reference.field,
        unknown_volatiles=dict(reference.unknown_volatiles),
    )
    for slot in slots:
        out.species[:, side, slot] = source.species[0, side, slot]
        out.ability[:, side, slot] = source.ability[0, side, slot]
        out.item[:, side, slot] = source.item[0, side, slot]
        out.moves[:, side, slot] = source.moves[0, side, slot]
        out.mon[:, side, slot] = source.mon[0, side, slot]
        out.mask[:, side, slot] = source.mask[0, side, slot]
    return out


def test_the_old_patch_rule_is_the_pre_ika119_body(setup) -> None:  # noqa: ANN001
    """`EncodingRules(patch_shares_side=True)` is the old `_patched`, array for array (IKA-141).

    On the two positions the IKA-119 tests were built on, where the old body is wrong -- so
    the current body differs from it there, which is the control that this equality could
    have failed.
    """
    from pokeuraou.encode import EncodingRules

    reg, sheet, _position = setup
    old_rules = EncodingRules(patch_shares_side=True)
    stoned = _with_stone(reg, sheet, "venusaur", "venusaurite")
    damaged = position_from_sets(
        reg,
        _pick(sheet, "incineroar", "sylveon", "garchomp", "venusaur"),
        _pick(sheet, "venusaur", "garchomp", "charizard", "toxapex"),
    )
    lead = damaged.sides[1].pokemon[damaged.sides[1].active[0]]
    lead.hp = lead.maxhp // 2
    mega = position_from_sets(
        reg,
        _pick(stoned, "incineroar", "sylveon", "garchomp", "toxapex"),
        _pick(stoned, "charizard", "garchomp", "sylveon", "venusaur"),
    )
    megad = mega.copy()
    megad.sides[1].mega_used = True
    megad.sides[1].pokemon[megad.sides[1].active[0]].is_mega = True

    cases = [(damaged, damaged, sheet), (mega, megad, stoned)]
    differed = 0
    for root, leaf, used_sheet in cases:
        for rules in (EncodingRules(), EncodingRules(mega_from_slots=True)):
            encoder = Encoder(reg, rules=rules)
            reference = encoder.encode_positions([leaf])
            for item in completions(reg, root, 1, used_sheet, seen=frozenset({0, 1})):
                old = _patched(
                    reference, 1, item.slots, item, reg, root, encoder, dataclasses.replace(
                        rules, patch_shares_side=True
                    )
                )
                want = _patched_before_ika119(reference, 1, item.slots, item, encoder)
                assert not _differences(old, want), (rules, item.species)
                assert old.side is reference.side, "the old body shared the side vector"
                new = _patched(reference, 1, item.slots, item, reg, root, encoder, rules)
                differed += bool(_differences(new, want))
    assert old_rules.patch_shares_side
    assert differed, "the old and new bodies agree on every case; nothing was compared"


def test_each_leaf_patches_by_its_own_rule_in_one_process(setup) -> None:  # noqa: ANN001
    """Two leaves through `belief_payoffs`, one per rule, as the two arms of a match are.

    The leaf under the current rule still equals a matrix per completion; the leaf under
    the old one does not, on the position where IKA-119 moved an answer; and each leaf's
    encoder booked only its own rule.
    """
    from pokeuraou.encode import EncodingRules

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
    budget = Budget.matrix()
    spreads = {
        0: completions(reg, position, 0, sheet),
        1: completions(reg, position, 1, sheet),
    }
    old = _Leaf(Encoder(reg, rules=EncodingRules(patch_shares_side=True)))
    new = _Leaf(Encoder(reg))
    old_node = belief_payoffs(reg, position, ours, theirs, old, budget=budget, spreads=spreads)
    new_node = belief_payoffs(reg, position, ours, theirs, new, budget=budget, spreads=spreads)
    assert old_node.shared > 0 and new_node.shared > 0

    assert "patched side=shared" in old.encoder.used, old.encoder.used
    assert "patched side=rebuilt" not in old.encoder.used, old.encoder.used
    assert "patched side=rebuilt" in new.encoder.used, new.encoder.used
    assert "patched side=shared" not in new.encoder.used, new.encoder.used

    moved = 0
    for side, items in spreads.items():
        for index, item in enumerate(items):
            expected, _notes = batched_payoff(
                reg, item.position, ours, theirs, new, budget=budget
            )
            np.testing.assert_array_equal(new_node.matrices[1 - side][index], expected)
            moved += int(not np.array_equal(old_node.matrices[1 - side][index], expected))
    assert moved, "the old rule moved no matrix here; the switch was not exercised"

# ------------------------------------------------------------------- IKA-139 --
#
# Every test above starts from a turn-1 position built by hand. On recorded mid-game
# positions (`data/ika73/w12` games 0-9, `scratchpad/fastpath_refused.py`) the fast path
# differed from `_per_completion` in 44 of 444 matrices, 1,429 cells, until IKA-121 and
# IKA-119 landed; after them in none (`scratchpad/ika139_patched_vs_scratch.py` compares
# the patched leaves with the port's own, feature by feature). The position here is built
# the way those were: a turn played through the resolver, so it is turn 2, both sides are
# damaged, side 1's slots were renumbered by a switch and one of its bench members has
# been seen and gone back out (seen by species, IKA-117), and side 0 can still mega into a
# leaf while a stone waits among its hidden candidates.


class _SlotLeaf:
    """Every element of every array with a weight of its own.

    `_Leaf` sums the per-Pokemon arrays, so two bench rows that traded places, or a feature
    that moved up in one row and down in another, pass it. Reduced row by row rather than
    by a matrix product: the fast path and the definition score different batches, and a
    product's summation order may depend on the batch (a `flat @ w` version of this leaf
    differed in the 1e-22 place on 242 cells whose inputs were identical).
    """

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self._weights: dict[tuple[str, int], np.ndarray] = {}

    def _w(self, name: str, size: int) -> np.ndarray:
        key = (name, size)
        if key not in self._weights:
            self._weights[key] = (
                np.random.default_rng([139, len(name), size]).normal(size=size) * 0.01
            )
        return self._weights[key]

    def from_encoded(self, encoded) -> np.ndarray:  # noqa: ANN001
        n = len(encoded)
        raw = np.zeros(n, dtype=np.float64)
        for name in _ARRAYS:
            flat = getattr(encoded, name).reshape(n, -1).astype(np.float64)
            raw += (flat * self._w(name, flat.shape[1])).sum(axis=1)
        return 1.0 / (1.0 + np.exp(-raw))

    def __call__(self, positions: list) -> np.ndarray:
        return self.from_encoded(self.encoder.encode_positions(positions))


@pytest.fixture(scope="module")
def midgame(setup):  # noqa: ANN001, ANN201
    reg, sheet, _position = setup
    sheet = _with_stone(reg, sheet, "venusaur", "venusaurite")
    root = position_from_sets(
        reg,
        _pick(sheet, "charizard", "garchomp", "sylveon", "venusaur"),
        _pick(sheet, "incineroar", "garchomp", "toxapex", "sylveon"),
    )
    legal = [{a.to_choice(): a for a in side_actions(reg, root, s)} for s in (0, 1)]
    # Heat Wave and Earthquake into Incineroar going out for Toxapex.
    chosen = [legal[0]["move 1, move 1"], legal[1]["switch 3, move 1"]]
    turn = resolve_turn(reg, root, chosen, budget=Budget.matrix())
    position = max(turn.branches, key=lambda branch: branch.probability).position

    shown = [
        {to_id(mon.species) for mon in side.pokemon if mon.active_index is not None}
        for side in root.sides
    ]
    seen = []
    for s, side in enumerate(position.sides):
        shown[s] |= {to_id(m.species) for m in side.pokemon if m.active_index is not None}
        seen.append(frozenset(m.slot for m in side.pokemon if to_id(m.species) in shown[s]))
    ours, theirs = _menus(reg, position, 10)
    if not any(action.declares_mega for action in ours):
        ours = [*ours, next(a for a in side_actions(reg, position, 0) if a.declares_mega)]

    # What makes it a mid-game position, asserted so that a change to the resolver or the
    # roster cannot quietly turn this back into turn 1.
    assert position.turn == 2
    back = position.sides[1].pokemon
    assert to_id(back[0].species) == "toxapex" and back[0].slot == 0, "no renumbering"
    out = next(m for m in back if to_id(m.species) == "incineroar")
    assert out.active_index is None and out.hp == out.maxhp and out.slot in seen[1], (
        "the switched-out Incineroar should be seen only by tracking, not by the position"
    )
    assert any(m.hp < m.maxhp for m in position.sides[0].pokemon)
    assert any(m.hp < m.maxhp for m in position.sides[1].pokemon)
    assert not position.sides[0].mega_used
    spreads = {s: completions(reg, position, s, sheet, seen=seen[s]) for s in (0, 1)}
    assert spreads[0][0].slots == (2, 3) and len(spreads[1][0].slots) == 1
    return reg, position, ours, theirs, spreads, sheet, seen


def _node_mismatches(reg, position, ours, theirs, spreads, leaf):  # noqa: ANN001, ANN202
    budget = Budget.matrix()
    node = belief_payoffs(reg, position, ours, theirs, leaf, budget=budget, spreads=spreads)
    assert node.shared > 0, "the fast path did not run; the comparison would be vacuous"
    wrong: dict[str, int] = {}
    for side, items in spreads.items():
        for index, item in enumerate(items):
            expected, _notes = batched_payoff(
                reg, item.position, ours, theirs, leaf, budget=budget
            )
            off = int((node.matrices[1 - side][index] != expected).sum())
            if off:
                wrong[f"side {side} completion {index} ({'+'.join(item.species)})"] = off
    return wrong


def test_a_midgame_shared_node_equals_a_matrix_per_completion(midgame) -> None:  # noqa: ANN001
    reg, position, ours, theirs, spreads, _sheet, _seen = midgame
    wrong = _node_mismatches(reg, position, ours, theirs, spreads, _SlotLeaf(Encoder(reg)))
    assert not wrong, f"cells not equal to the bit: {wrong}"


def _side_from_the_true_position(real):  # noqa: ANN001, ANN202
    def patch(reference, side, slots, item, reg, position, *rest):  # noqa: ANN001, ANN002, ANN202
        out = real(reference, side, slots, item, reg, position, *rest)
        out.side = reference.side.copy()
        return out

    return patch


def _bench_can_mega_from_the_root(real):  # noqa: ANN001, ANN202
    def patch(reference, side, slots, item, reg, position, *rest):  # noqa: ANN001, ANN002, ANN202
        # `*rest` is the leaf's encoder and rules, which `_patched` takes since IKA-141.
        out = real(reference, side, slots, item, reg, position, *rest)
        encoder = beliefnode._encoder_for(reg)
        source = encoder.encode_positions([item.position])
        k = encoder.mon_names.index("can_mega")
        for slot in slots:
            out.mon[:, side, slot, k] = source.mon[0, side, slot, k]
        return out

    return patch


@pytest.mark.parametrize(
    "fault", [_side_from_the_true_position, _bench_can_mega_from_the_root],
    ids=["side vector from the true position", "bench can_mega from the root"],
)
def test_the_midgame_node_catches_each_half_of_the_ika119_leak(  # noqa: ANN001
    midgame, monkeypatch, fault
) -> None:
    """The positive control for the test above: each half of what `_patched` did before
    IKA-119, put back on its own, has to make this position's node differ. Otherwise the
    equality above would pass on a position that does not exercise the patch."""
    reg, position, ours, theirs, spreads, _sheet, _seen = midgame
    monkeypatch.setattr(beliefnode, "_patched", fault(beliefnode._patched))
    wrong = _node_mismatches(reg, position, ours, theirs, spreads, _SlotLeaf(Encoder(reg)))
    assert wrong, "the leak was put back and no cell moved: this position cannot see it"


def test_a_midgame_patched_leaf_equals_the_ports_own_leaf(midgame) -> None:  # noqa: ANN001
    """The inputs rather than the payoffs: for every shared cell, each patched leaf against
    the port's leaf of the same cell resolved on the completion, every array, exactly."""
    reg, position, ours, theirs, spreads, _sheet, _seen = midgame
    budget = Budget.matrix()
    port = rustnode.node_for(reg)
    reference = port.fill_encoded(position, ours, theirs, budget)
    hidden = {side: items[0].slots for side, items in spreads.items()}
    dirty = reaches_bench(reg, ours, theirs, hidden)
    for i, j, _root in reference.folded:
        dirty[i, j] = True
    compared = 0
    for side, items in spreads.items():
        for item in items:
            got = _patched(reference.encoded, side, item.slots, item, reg, position)
            own = port.fill_encoded(item.position, ours, theirs, budget)
            theirs_spans = {(i, j): (ix, w) for i, j, ix, w in own.spans}
            for i, j, indices, weights in reference.spans:
                if dirty[i, j] or not weights:
                    continue
                other, other_weights = theirs_spans[(i, j)]
                assert list(other_weights) == list(weights), (i, j)
                for a, b in zip(indices, other, strict=True):
                    compared += 1
                    diff = {
                        name: int(
                            (getattr(got, name)[a] != getattr(own.encoded, name)[b]).sum()
                        )
                        for name in _ARRAYS
                    }
                    diff = {name: n for name, n in diff.items() if n}
                    assert not diff, (
                        f"side {side} {'+'.join(item.species)} cell ({i}, {j}): {diff}"
                    )
    assert compared > 0


def test_a_cell_the_port_refuses_is_resolved_per_completion(midgame) -> None:  # noqa: ANN001
    """The defect IKA-139 found on master: a refused cell was left at 0.0.

    `belief_payoffs` asks the port for the whole node on the true position and reads its
    spans and folds -- and a cell the port refuses has neither. Nothing marked it dirty, so
    it was not re-resolved either, and its payoff stayed at the 0.0 the matrix was made
    with in every completion; `_per_completion` fills the same cell in Python, as every
    other caller of the port does (`resolve._rust_encoded_payoffs`). On `data/ika73/w12`
    games 10-209 that was 3,886 cells in 179 of 10,158 completion matrices, 19 decisions,
    every one `breaksProtect move` (Feint). The turn-1 fixtures and games 0-9 had none.

    Feint is given to side 1's Toxapex here (Garchomp is locked into Earthquake by its
    Scarf), so the port refuses the cells where it is used.
    """
    reg, position, ours, theirs, spreads = _with_feint(midgame)
    wrong = _node_mismatches(reg, position, ours, theirs, spreads, _SlotLeaf(Encoder(reg)))
    assert not wrong, f"cells not equal to the bit: {wrong}"


def _with_feint(midgame):  # noqa: ANN001, ANN202
    """The mid-game node with Feint on Toxapex: shared refused cells, and dirty ones."""
    from pokeuraou.position import MoveSlot

    reg, position, ours, _theirs, _spreads, sheet, seen = midgame
    position = position.copy()
    toxapex = next(m for m in position.sides[1].pokemon if to_id(m.species) == "toxapex")
    toxapex.moves[0] = MoveSlot(id="feint", pp=16, maxpp=16)
    # The completions have to be of this position, not of the fixture's.
    spreads = {s: completions(reg, position, s, sheet, seen=seen[s]) for s in (0, 1)}
    theirs = [
        action for action in side_actions(reg, position, 1)
        if any(getattr(slot, "move_id", None) == "feint" for slot in action.slots)
    ][:4] + narrow(reg, position, 1, limit=6).actions

    port = rustnode.node_for(reg)
    filled = port.fill_encoded(position, ours, theirs, Budget.matrix())
    hidden = {side: items[0].slots for side, items in spreads.items()}
    dirty = reaches_bench(reg, ours, theirs, hidden)
    shared_refusals = [(i, j) for i, j, _why in filled.refused if not dirty[i, j]]
    assert shared_refusals, "no refused cell that the fast path would share; nothing to test"
    return reg, position, ours, theirs, spreads


# ------------------------------------------------------------------- IKA-105 --
#
# The node's leaves used to be scored a forward pass per completion for the shared cells,
# and per completion again for the dirty ones -- the port's leaves in one and the cells it
# refused, resolved in Python, in another: 17.74 passes a `move.hidden` decision in
# shipping generation (IKA-98). They are one batch now, and each completion reads its own
# rows back by where its block starts. A block read from the wrong start is a payoff that
# is some other completion's and still looks like a payoff.


class _CountedSlotLeaf(_SlotLeaf):
    def __init__(self, encoder: Encoder) -> None:
        super().__init__(encoder)
        self.passes = 0
        self.calls = 0

    def from_encoded(self, encoded) -> np.ndarray:  # noqa: ANN001
        self.passes += 1
        return super().from_encoded(encoded)

    def __call__(self, positions: list) -> np.ndarray:
        self.calls += 1
        return super().__call__(positions)


def test_a_hidden_node_is_scored_in_one_forward_pass(midgame, monkeypatch) -> None:  # noqa: ANN001
    """Every completion's shared rows, its dirty cells from the port and the cells the port
    refused all go through one `from_encoded`, and the node still equals the definition."""
    reg, position, ours, theirs, spreads = _with_feint(midgame)
    shapes: list[list[int]] = []
    real = beliefnode._stacked

    def recorded(parts, like):  # noqa: ANN001, ANN202
        shapes.append([rows for rows, _make in parts])
        return real(parts, like)

    monkeypatch.setattr(beliefnode, "_stacked", recorded)
    leaf = _CountedSlotLeaf(Encoder(reg))
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads
    )
    assert node.shared > 0 and node.redone > 0, "no shared or no dirty cell; nothing batched"
    assert leaf.passes == 1 and leaf.calls == 0, (leaf.passes, leaf.calls)
    (blocks,) = shapes
    completions_total = sum(len(items) for items in spreads.values())
    # A shared block per completion (the reference once for the exact side), a dirty block
    # per completion, and the refused cells' Python leaves per completion.
    assert len(blocks) > completions_total + 1, blocks

    monkeypatch.setattr(beliefnode, "_stacked", real)
    wrong = _node_mismatches(reg, position, ours, theirs, spreads, _SlotLeaf(Encoder(reg)))
    assert not wrong, f"cells not equal to the bit: {wrong}"


def _rotated(real):  # noqa: ANN001, ANN202
    """`_stacked`, with the starts of the reference-sized blocks handed round by one.

    The rows are where they were and every read is in bounds, so nothing fails but the
    answer: each block's owner reads the next block's values. Those blocks are every
    completion's shared rows (a patched copy of the reference, or the reference), which
    is the swap a wrong offset would make."""

    def stacked(parts, like):  # noqa: ANN001, ANN202
        out, starts = real(parts, like)
        group = [index for index, (rows, _make) in enumerate(parts) if rows == len(like)]
        moved = list(starts)
        for a, b in zip(group, group[1:] + group[:1], strict=True):
            moved[a] = starts[b]
        return out, moved

    return stacked


def test_rows_scattered_to_the_wrong_completion_are_caught(midgame, monkeypatch) -> None:  # noqa: ANN001
    """The positive control for the equality tests: hand each completion's shared rows to
    the next completion and the node must differ from the definition. Otherwise the tests
    that say it equals the definition could not see a scatter that went wrong."""
    reg, position, ours, theirs, spreads = _with_feint(midgame)
    monkeypatch.setattr(
        beliefnode, "_stacked", _rotated(beliefnode._stacked)
    )
    wrong = _node_mismatches(reg, position, ours, theirs, spreads, _SlotLeaf(Encoder(reg)))
    assert wrong, "the completions' rows were swapped and no cell moved"
    completions_total = sum(len(items) for items in spreads.values())
    assert len(wrong) == completions_total, (
        f"only {len(wrong)} of {completions_total} completion matrices moved: {wrong}"
    )
