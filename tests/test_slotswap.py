"""Exchanging the left and right Pokemon of a side (IKA-412).

What is tested, and what each test can fail on:

* the position swap is an involution, and it moves what `slotswap` says it moves;
* the array swap is the encoding of the position swap (so training on swapped arrays is
  training on the swapped position, not on something the encoder would never produce), and
  the torch batch swap is the numpy one;
* the port agrees that the swap leaves a turn alone -- with the menus carried over -- and the
  same comparison *fails* when the menus are left unswapped (the control that the comparison
  can fail), or when a swapped side's `source_slot`s are not moved;
* the network's only way to tell the two arrangements apart is the `active_slot_*` and
  `slot*_*` input columns: with those weights zeroed it answers the same, with them as
  initialised it does not;
* a position holding an Imposter is never exchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pokeuraou import slotswap
from pokeuraou.actions import MoveAction, SideAction
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.port import batched_payoff
from pokeuraou.position import Effect, Position
from pokeuraou.selfplay import position_from_sets
from pokeuraou.slotswap import SwapSlots, swap_action, swap_batch, swap_encoded, swap_positions
from pokeuraou.teams import load_roster
from pokeuraou.value import ValueConfig, build

from ._port import Budget, resolve_turn

VARIANTS = [(True, False), (False, True), (True, True)]


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _played(roster, turns: int = 7, seed: int = 5) -> list[Position]:  # noqa: ANN001
    """Positions of one game, a mirror, two switches and traps included as the game gives them."""
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
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
            reg,
            pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        weights = np.array([b.probability for b in result.branches], dtype=np.float64)
        pos = result.branches[int(rng.choice(len(weights), p=weights / weights.sum()))].position
    return out


def test_the_position_swap_twice_is_the_position(roster) -> None:  # noqa: ANN001
    positions = _played(roster)
    assert len(positions) >= 4
    for pos in positions:
        for which in VARIANTS:
            once = swap_positions(pos, which)
            assert once.to_json() != pos.to_json(), "the swap changed nothing: a vacuous check"
            assert swap_positions(once, which).to_json() == pos.to_json()
        assert swap_positions(pos, (False, False)).to_json() == pos.to_json()


def test_the_swap_moves_the_list_the_slots_and_the_sources(roster) -> None:  # noqa: ANN001
    pos = _played(roster)[0]
    pos.sides[0].slot_conditions = [[Effect(id="wish")], []]
    pos.sides[1].pokemon[0].volatiles.append(Effect(id="leechseed", source_slot="00"))
    pos.sides[1].pokemon[1].volatiles.append(Effect(id="partiallytrapped", source_slot="p1b"))
    pos.sides[1].pokemon[1].volatiles.append(Effect(id="infestation", source_slot="10"))
    swapped = swap_positions(pos, (True, False))
    own, before = swapped.sides[0], pos.sides[0]
    assert [m.species for m in own.pokemon[:2]] == [m.species for m in before.pokemon[1::-1]]
    assert [m.slot for m in own.pokemon] == [0, 1, 2, 3]
    assert own.active == [0, 1]
    assert [m.active_index for m in own.pokemon] == [0, 1, None, None]
    assert [[e.id for e in g] for g in own.slot_conditions] == [[], ["wish"]]
    # the foe's effects that name side 0 position 0 / side 0 'b' now name the other position;
    # an effect naming the foe's own side (the third) is not touched.
    foe_volatiles = {v.id: v.source_slot for m in swapped.sides[1].pokemon for v in m.volatiles}
    assert foe_volatiles == {"leechseed": "01", "partiallytrapped": "p1a", "infestation": "10"}
    # the foe swapped instead: now the third moves, the other two stay
    other = swap_positions(pos, (False, True))
    foe_volatiles = {v.id: v.source_slot for m in other.sides[1].pokemon for v in m.volatiles}
    assert foe_volatiles == {"leechseed": "00", "partiallytrapped": "p1b", "infestation": "11"}


def test_the_array_swap_is_the_encoding_of_the_position_swap(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    encoder = Encoder(reg)
    positions = _played(roster)
    base = encoder.encode_positions(positions)
    for which in VARIANTS:
        expected = encoder.encode_positions([swap_positions(p, which) for p in positions])
        flip = np.tile(np.array(which), (len(positions), 1))
        got = swap_encoded(base, flip, encoder.mon_names, encoder.side_names)
        for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
            np.testing.assert_array_equal(getattr(got, name), getattr(expected, name), err_msg=name)
        again = swap_encoded(got, flip, encoder.mon_names, encoder.side_names)
        for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
            np.testing.assert_array_equal(getattr(again, name), getattr(base, name), err_msg=name)
        # not vacuous: the swapped arrays differ from the originals
        assert not np.array_equal(got.mon, base.mon)


def test_the_torch_swap_is_the_numpy_swap(roster) -> None:  # noqa: ANN001
    encoder = Encoder(roster.reg)
    enc = encoder.encode_positions(_played(roster))
    rng = np.random.default_rng(0)
    flip = rng.random((len(enc), 2)) < 0.5
    flip[0] = (True, True)
    swap = SwapSlots.of(encoder.mon_names, encoder.side_names)
    batch = {
        "species": torch.from_numpy(enc.species), "ability": torch.from_numpy(enc.ability),
        "item": torch.from_numpy(enc.item), "moves": torch.from_numpy(enc.moves),
        "mon": torch.from_numpy(enc.mon), "mask": torch.from_numpy(enc.mask),
        "side": torch.from_numpy(enc.side), "field": torch.from_numpy(enc.field),
    }
    got = swap_batch(batch, flip, swap)
    want = swap_encoded(enc, flip, encoder.mon_names, encoder.side_names)
    for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
        np.testing.assert_array_equal(got[name].numpy(), getattr(want, name), err_msg=name)


def test_the_network_tells_the_arrangements_apart_only_by_the_slot_columns(roster) -> None:  # noqa: ANN001
    encoder = Encoder(roster.reg)
    enc = encoder.encode_positions(_played(roster))
    flip = np.ones((len(enc), 2), dtype=bool)
    swapped = swap_encoded(enc, flip, encoder.mon_names, encoder.side_names)
    cfg = ValueConfig()
    torch.manual_seed(0)
    net = build(encoder, cfg).eval()

    def logits(e):  # noqa: ANN001, ANN202
        batch = {k: torch.from_numpy(getattr(e, k)) for k in
                 ("species", "ability", "item", "moves", "mon", "mask", "side", "field")}
        with torch.no_grad():
            return net(batch).numpy()

    # control: as initialised, the two arrangements are told apart
    assert np.abs(logits(enc) - logits(swapped)).max() > 1e-4
    mon_a, mon_b = encoder.mon_names.index("active_slot_0"), encoder.mon_names.index("active_slot_1")
    offset = cfg.species_dim + cfg.ability_dim + cfg.item_dim + cfg.move_dim
    swap = SwapSlots.of(encoder.mon_names, encoder.side_names)
    with torch.no_grad():
        net.mon_mlp[0].weight[:, offset + mon_a] = 0
        net.mon_mlp[0].weight[:, offset + mon_b] = 0
        for column in (*swap.side_a, *swap.side_b):
            net.side_mlp[0].weight[:, 4 * cfg.mon_dim + column] = 0
    assert np.abs(logits(enc) - logits(swapped)).max() < 1e-5


def test_a_position_that_reads_its_position_is_never_exchanged(roster) -> None:  # noqa: ANN001
    encoder = Encoder(roster.reg)
    pos = _played(roster)[0]
    assert not slotswap.holds_position_reader(pos)
    pos.sides[1].pokemon[3].ability = "imposter"  # on the bench: it steps into the open position
    assert slotswap.holds_position_reader(pos)
    imposter = encoder.vocab.abilities["imposter"]
    enc = encoder.encode_positions([_played(roster)[0], pos])
    reader = slotswap.position_reader_rows(enc.ability, enc.mask, imposter)
    assert reader.tolist() == [False, True]
    swap = SwapSlots.of(encoder.mon_names, encoder.side_names, exclude=reader)
    rng = np.random.default_rng(1)
    drawn = np.concatenate([swap.draw(rng, np.array([0, 1])) for _ in range(200)])
    assert drawn[1::2].sum() == 0, "the Imposter row was exchanged"
    assert drawn[0::2].any(), "the other row was never exchanged: the control is vacuous"
    # a regulation with no such ability excludes nothing
    assert not slotswap.position_reader_rows(enc.ability, enc.mask, 0).any()


def _matrix(reg, pos, ours, theirs):  # noqa: ANN001, ANN202
    return batched_payoff(reg, pos, ours, theirs, HP_SHARE.batch, budget=Budget.matrix())[0]


def test_the_port_agrees_the_swap_leaves_the_turn_alone(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    worst = 0.0
    worst_without_menus = 0.0
    for pos in _played(roster)[:6]:
        ours = narrow(reg, pos, 0, limit=5).actions
        theirs = narrow(reg, pos, 1, limit=5).actions
        if not ours or not theirs:
            continue
        base = _matrix(reg, pos, ours, theirs)
        for which in VARIANTS:
            moved = swap_positions(pos, which)
            mine = [swap_action(a, which[0], which[1]) for a in ours]
            foes = [swap_action(a, which[1], which[0]) for a in theirs]
            worst = max(worst, float(np.abs(_matrix(reg, moved, mine, foes) - base).max()))
            # control: the same swapped position with the menus left as they were
            worst_without_menus = max(
                worst_without_menus,
                float(np.abs(_matrix(reg, moved, ours, theirs) - base).max()),
            )
    assert worst < 2e-3, worst
    assert worst_without_menus > 0.02, "the comparison cannot fail: the control is vacuous"


def test_a_swapped_action_is_the_same_choice_in_the_other_position() -> None:
    action = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="a", target=1),
            MoveAction(slot=1, move_index=2, move_id="b", target=-1),
        )
    )
    assert swap_action(swap_action(action, True, True), True, True) == action
    own = swap_action(action, True, False)
    assert [s.slot for s in own.slots] == [0, 1]
    assert [(s.move_id, s.target) for s in own.slots] == [("b", -2), ("a", 1)]
    foe = swap_action(action, False, True)
    assert [(s.move_id, s.target) for s in foe.slots] == [("a", 2), ("b", -1)]


def test_a_charging_move_keeps_aiming_at_the_same_pokemon(roster) -> None:  # noqa: ANN001
    """IKA-412 found it: `twoturnmove.extra.targetLoc` is a position, so a swap of the target's
    side must move it (and a swap of the holder's side moves an ally target)."""
    pos = _played(roster)[0]
    holder = pos.sides[0].pokemon[0]
    holder.volatiles.append(Effect(id="twoturnmove", move="phantomforce", extra={"targetLoc": 2}))
    ally = pos.sides[1].pokemon[0]
    ally.volatiles.append(Effect(id="twoturnmove", move="solarbeam", extra={"targetLoc": -1}))

    def locs(p: Position) -> tuple[int, int]:
        loc_a = next(v for m in p.sides[0].pokemon for v in m.volatiles if v.id == "twoturnmove")
        loc_b = next(v for m in p.sides[1].pokemon for v in m.volatiles if v.id == "twoturnmove")
        return loc_a.extra["targetLoc"], loc_b.extra["targetLoc"]

    assert locs(pos) == (2, -1)
    # side 1 swapped: side 0's foe target 2 -> 1, and side 1's own ally target -1 -> -2
    assert locs(swap_positions(pos, (False, True))) == (1, -2)
    # side 0 swapped: its holder's foe target and side 1's ally target are not on side 0 / side 1
    assert locs(swap_positions(pos, (True, False))) == (2, -1)
    assert locs(swap_positions(pos, (True, True))) == (1, -2)
