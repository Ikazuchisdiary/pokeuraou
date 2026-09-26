"""The candidate model's action encoding ties each choice to the rows it names (IKA-274)."""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import qhead
from pokeuraou.actions import MoveAction, SwitchAction
from pokeuraou.encode import Encoder


def _positions(reg, count: int):
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import all_selections, load_roster

    roster = load_roster("rizabanadohido")
    picks = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
    return [
        position_from_sets(
            reg,
            [roster.sets[i] for i in picks[k % len(picks)]],
            [roster.sets[i] for i in picks[(k + 3) % len(picks)]],
        )
        for k in range(count)
    ]


def _check(reg, pos, side, acts, pool) -> int:
    """Asserts every slot's rows; returns how many slots were checked."""
    own, foe = pos.sides[side], pos.sides[1 - side]
    vocab = Encoder(reg).vocab
    checked = 0
    for n, action in enumerate(pool):
        for t, slot_action in enumerate(action.slots):
            row = acts[n, t]
            assert row[1] == own.active[slot_action.slot]
            if isinstance(slot_action, SwitchAction):
                assert row[0] == qhead.KIND_SWITCH
                assert own.pokemon[row[6]].species == slot_action.species
                assert not own.pokemon[row[6]].is_active
            elif isinstance(slot_action, MoveAction):
                assert row[0] == qhead.KIND_MOVE
                actor = own.pokemon[row[1]]
                assert slot_action.move_id in [m.id for m in actor.moves]
                assert row[2] == vocab.moves[slot_action.move_id]
                assert row[3] == int(slot_action.mega)
                if slot_action.target is not None and slot_action.target > 0:
                    assert row[4] == qhead.TARGET_FOE
                    assert row[5] == foe.active[slot_action.target - 1]
                elif slot_action.target is not None:
                    assert row[4] == qhead.TARGET_ALLY
                    assert row[5] == own.active[-slot_action.target - 1]
                else:
                    assert row[4] == qhead.TARGET_NONE
            checked += 1
    return checked


def test_actions_name_the_rows_of_the_encoded_position(reg):
    checked = 0
    for pos in _positions(reg, 6):
        for side in (0, 1):
            pool = qhead.legal_pool(reg, pos, side)
            acts = qhead.encode_actions(Encoder(reg).vocab, pos, side, pool)
            assert acts.shape == (len(pool), 2, len(qhead.ACTION_FIELDS))
            checked += _check(reg, pos, side, acts, pool)
    assert checked > 200


def test_a_switch_names_its_pokemon_after_the_party_is_reordered(reg):
    """The row is found by party slot, not assumed to be the list position."""
    pos = _positions(reg, 1)[0]
    side = pos.sides[0]
    # Reverse the bench's list order while keeping each Pokemon's party slot.
    bench = [i for i, mon in enumerate(side.pokemon) if not mon.is_active]
    side.pokemon[bench[0]], side.pokemon[bench[1]] = side.pokemon[bench[1]], side.pokemon[bench[0]]
    pool = [a for a in qhead.legal_pool(reg, pos, 0) if a.switch_indices]
    acts = qhead.encode_actions(Encoder(reg).vocab, pos, 0, pool)
    assert pool
    assert _check(reg, pos, 0, acts, pool) > 0


def test_the_net_gives_a_matrix_over_both_pools(reg):
    torch = pytest.importorskip("torch")
    encoder = Encoder(reg)
    pos = _positions(reg, 1)[0]
    pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
    enc = encoder.encode_positions([pos])
    batch = {
        name: torch.from_numpy(np.ascontiguousarray(getattr(enc, name)))
        for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    }
    a0 = torch.from_numpy(qhead.encode_actions(encoder.vocab, pos, 0, pools[0]).astype(np.int64))
    a1 = torch.from_numpy(qhead.encode_actions(encoder.vocab, pos, 1, pools[1]).astype(np.int64))
    net = qhead.build_net(encoder, qhead.QConfig()).eval()
    with torch.no_grad():
        out = net(batch, a0[None], a1[None])
    assert out.shape == (1, len(pools[0]), len(pools[1]))
    assert torch.isfinite(out).all()


def test_the_net_is_antisymmetric_under_the_mirror(reg):
    """Q(mirror s, b, a) = 1 - Q(s, a, b), untrained: it is the net's shape, not learned."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    encoder = Encoder(reg)
    net = qhead.build_net(encoder, qhead.QConfig()).eval()
    for pos in _positions(reg, 3):
        pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
        enc = encoder.encode_positions([pos])
        arrays = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
        batch = {n: torch.from_numpy(np.ascontiguousarray(getattr(enc, n))) for n in arrays}
        mirrored = enc.mirror()
        flipped = {n: torch.from_numpy(np.ascontiguousarray(getattr(mirrored, n))) for n in arrays}
        a0 = torch.from_numpy(qhead.encode_actions(encoder.vocab, pos, 0, pools[0]).astype(np.int64))[None]
        a1 = torch.from_numpy(qhead.encode_actions(encoder.vocab, pos, 1, pools[1]).astype(np.int64))[None]
        with torch.no_grad():
            forward = net(batch, a0, a1)[0]
            back = net(flipped, a1, a0)[0]
        assert forward.abs().max() > 1e-3  # not the trivial zero
        assert torch.allclose(back, -forward.T, atol=1e-5)
