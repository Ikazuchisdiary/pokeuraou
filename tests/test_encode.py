"""The encoding, which now sits alongside legal-move generation and the LP solver.

The project's rule is that the three things whose failure makes every printed number
meaningless get tests. The encoder is a fourth. It converts a position into the only thing
the value function ever sees, so an encoder that drops Speed, mixes up the two sides, or
lets the outcome leak in produces a confident win probability with nothing behind it, and
no other test in the repository would notice.

Four properties are worth more than the rest:

- **no leakage**: the same position encodes identically no matter how the game it came
  from ended, which is the whole difference between a value function and a lookup table;
- **antisymmetry**: mirroring swaps the two sides and touches nothing else, which is what
  lets the model guarantee ``V(x) + V(mirror x) = 1``;
- **regulation pinning**: the vocabulary fingerprint is stable across rebuilds and
  different between regulations, so weights cannot be silently reused across a rotation;
- **the numbers are the real numbers**: stats come out equal to the tested SP formula
  rather than to something plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.encode import (
    VOLATILES,
    Encoder,
    build_vocabulary,
    field_feature_names,
    mon_feature_names,
    side_feature_names,
)
from pokeuraou.priors import build_cooccurrence, find_cached_chaos, load_chaos
from pokeuraou.regulation import STAT_IDS, load_regulation
from pokeuraou.selfplay import position_from_sets
from pokeuraou.stats import nature_multipliers, stats_from_sp
from pokeuraou.teams import load_roster, sample_metagame_team


@pytest.fixture(scope="module")
def encoder_and_positions():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    roster = load_roster("rizabanadohido")
    cached = find_cached_chaos(reg.meta.format_id)
    if cached is None:
        pytest.skip("no cached usage stats")
    prior = load_chaos(cached, reg)
    cooc = build_cooccurrence(prior)
    rng = np.random.default_rng(3)
    positions = []
    for _ in range(6):
        foe = sample_metagame_team(rng, reg, prior, cooc)
        pos = position_from_sets(
            reg, roster.sets[: reg.meta.picked_team_size], foe[: reg.meta.picked_team_size]
        )
        positions.append(pos.to_json())
    return Encoder(reg), positions


def test_the_feature_name_lists_match_the_array_widths(encoder_and_positions) -> None:  # noqa: ANN001
    encoder, positions = encoder_and_positions
    encoded = encoder.encode(positions)
    assert encoded.mon.shape[-1] == len(mon_feature_names(encoder.vocab))
    assert encoded.side.shape[-1] == len(side_feature_names())
    assert encoded.field.shape[-1] == len(field_feature_names())
    # Widths are derived from the name lists, not from hand-counted constants.
    assert encoder.widths == {
        "mon": encoded.mon.shape[-1],
        "side": encoded.side.shape[-1],
        "field": encoded.field.shape[-1],
    }


def test_shapes_and_no_nonsense_values(encoder_and_positions) -> None:  # noqa: ANN001
    encoder, positions = encoder_and_positions
    encoded = encoder.encode(positions)
    n, m = len(positions), encoder.mons_per_side
    assert encoded.species.shape == (n, 2, m)
    assert encoded.moves.shape == (n, 2, m, 4)
    assert len(encoded) == n
    for array in (encoded.mon, encoded.side, encoded.field):
        assert np.isfinite(array).all()
    assert encoded.mask.sum() == n * 2 * m  # a turn-1 position has every Pokemon
    # Index 0 is reserved for absent/unknown, so a real species never lands on it.
    assert (encoded.species > 0).all()


def test_the_outcome_cannot_leak_in(encoder_and_positions) -> None:  # noqa: ANN001
    """Encoding depends on the position and on nothing else.

    The self-play record holds the outcome next to the position, so the guard that matters
    is that the encoder is never given the record. Asserting it structurally would only
    restate the signature; instead the same position dict is encoded twice with unrelated
    labels attached to the surrounding object, and the arrays have to be bit-identical.
    """
    encoder, positions = encoder_and_positions
    won = {"outcome": 1.0, "decisions": [{"position": positions[0]}]}
    lost = {"outcome": 0.0, "decisions": [{"position": positions[0]}]}
    a = encoder.encode([won["decisions"][0]["position"]])
    b = encoder.encode([lost["decisions"][0]["position"]])
    for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_mirroring_swaps_the_sides_and_leaves_the_field_alone(
    encoder_and_positions,
) -> None:  # noqa: ANN001
    encoder, positions = encoder_and_positions
    encoded = encoder.encode(positions)
    mirrored = encoded.mirror()
    assert np.array_equal(mirrored.species[:, 0], encoded.species[:, 1])
    assert np.array_equal(mirrored.species[:, 1], encoded.species[:, 0])
    assert np.array_equal(mirrored.mon[:, 0], encoded.mon[:, 1])
    assert np.array_equal(mirrored.side[:, 0], encoded.side[:, 1])
    # Nothing side-specific may live in `field`, or mirroring would have to touch it and
    # the model's antisymmetry would stop being exact.
    assert np.array_equal(mirrored.field, encoded.field)


def test_mirroring_twice_is_the_identity(encoder_and_positions) -> None:  # noqa: ANN001
    encoder, positions = encoder_and_positions
    encoded = encoder.encode(positions)
    twice = encoded.mirror().mirror()
    for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
        assert np.array_equal(getattr(twice, name), getattr(encoded, name)), name


def test_encoding_a_swapped_position_equals_mirroring_it(encoder_and_positions) -> None:  # noqa: ANN001
    """The strong form of the claim: swapping the sides *in the position* is the mirror.

    This is what makes the model's guarantee meaningful. If some side-dependent quantity
    ended up in `field`, this test fails while the cheaper axis-flip test still passes.
    """
    encoder, positions = encoder_and_positions
    swapped = []
    for position in positions:
        copy = dict(position)
        copy["sides"] = [position["sides"][1], position["sides"][0]]
        swapped.append(copy)
    direct = encoder.encode(swapped)
    mirrored = encoder.encode(positions).mirror()
    for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
        assert np.array_equal(getattr(direct, name), getattr(mirrored, name)), name


def test_the_stat_features_are_the_tested_formula(encoder_and_positions) -> None:  # noqa: ANN001
    encoder, positions = encoder_and_positions
    encoded = encoder.encode(positions)
    names = encoder.mon_names
    offsets = [names.index(f"stat_{s}") for s in STAT_IDS]
    scale = 200.0

    position = positions[0]
    for s, side in enumerate(position["sides"]):
        for p, mon in enumerate(side["pokemon"][: encoder.mons_per_side]):
            base = np.array(encoder.reg.species[mon["species"]].base_stats, dtype=np.int64)
            sp = np.array([[mon["sp"][stat] for stat in STAT_IDS]], dtype=np.int64)
            want = stats_from_sp(
                encoder.reg,
                base,
                sp,
                nature_multipliers(encoder.reg, [mon["nature"]]),
                level=encoder.reg.meta.level,
            )[0]
            got = encoded.mon[0, s, p, offsets] * scale
            assert np.allclose(got, want, atol=1e-3), (mon["species"], got, want)
            # maxhp is the HP stat, so the two have to agree with each other too.
            assert abs(float(want[0]) - mon["maxhp"]) < 1e-6


def test_hp_fraction_and_the_flags_read_the_position(encoder_and_positions) -> None:  # noqa: ANN001
    encoder, positions = encoder_and_positions
    names = encoder.mon_names
    position = {**positions[0]}
    sides = [dict(s) for s in position["sides"]]
    mons = [dict(m) for m in sides[0]["pokemon"]]
    mons[2] = {**mons[2], "hp": 0, "fainted": True, "activeIndex": None, "status": "brn"}
    mons[0] = {**mons[0], "hp": mons[0]["maxhp"] // 2}
    sides[0] = {**sides[0], "pokemon": mons}
    position["sides"] = sides

    encoded = encoder.encode([position])
    hp_at = names.index("hp_fraction")
    assert encoded.mon[0, 0, 0, hp_at] == pytest.approx(
        (mons[0]["maxhp"] // 2) / mons[0]["maxhp"]
    )
    assert encoded.mon[0, 0, 2, hp_at] == 0.0
    assert encoded.mon[0, 0, 2, names.index("fainted")] == 1.0
    assert encoded.mon[0, 0, 2, names.index("is_active")] == 0.0
    assert encoded.mon[0, 0, 2, names.index("status_brn")] == 1.0
    assert encoded.mon[0, 0, 2, names.index("status_none")] == 0.0
    # A fainted Pokemon changes the side's own summary, not just its own row.
    alive_at = side_feature_names().index("alive_fraction")
    assert encoded.side[0, 0, alive_at] == pytest.approx(0.75)


def test_a_volatile_outside_the_vocabulary_is_counted_not_dropped(
    encoder_and_positions,
) -> None:  # noqa: ANN001
    """An unnamed volatile must show up as a number.

    The encoder cannot represent every effect, and that is acceptable. What is not
    acceptable is representing it as "nothing happened", so unknown ids set one shared bit
    and are tallied by name, which is how the vocabulary gets extended deliberately.
    """
    encoder, positions = encoder_and_positions
    position = {**positions[0]}
    sides = [dict(s) for s in position["sides"]]
    mons = [dict(m) for m in sides[0]["pokemon"]]
    mons[0] = {**mons[0], "volatiles": [{"id": "somethingnobodymodelled"}]}
    sides[0] = {**sides[0], "pokemon": mons}
    position["sides"] = sides

    encoded = encoder.encode([position])
    other_at = encoder.mon_names.index("volatile_other")
    assert encoded.mon[0, 0, 0, other_at] == 1.0
    assert encoded.unknown_volatiles == {"somethingnobodymodelled": 1}

    known = {**positions[0]}
    sides = [dict(s) for s in known["sides"]]
    mons = [dict(m) for m in sides[0]["pokemon"]]
    mons[0] = {**mons[0], "volatiles": [{"id": "substitute"}]}
    sides[0] = {**sides[0], "pokemon": mons}
    known["sides"] = sides
    encoded = encoder.encode([known])
    assert encoded.unknown_volatiles == {}
    assert encoded.mon[0, 0, 0, encoder.mon_names.index("volatile_substitute")] == 1.0
    assert encoded.mon[0, 0, 0, other_at] == 0.0


def test_the_vocabulary_is_stable_and_regulation_specific() -> None:
    reg_b = load_regulation("gen9championsvgc2026regmb")
    once = build_vocabulary(reg_b)
    twice = build_vocabulary(load_regulation("gen9championsvgc2026regmb"))
    assert once.fingerprint() == twice.fingerprint()
    assert once.species == twice.species

    reg_c = load_regulation("gen9championsvgc2026regmc")
    other = build_vocabulary(reg_c)
    # M-C added Pokemon, so the two vocabularies differ and so do their fingerprints. Since
    # IKA-82 M-C's order begins with M-B's, so an M-B model can load onto M-C by growing its
    # tables -- decided by M-C cut back to M-B's sizes (tests/test_vocab_order.py), not here.
    assert other.fingerprint() != once.fingerprint()


def test_an_encoder_refuses_a_vocabulary_from_another_regulation() -> None:
    reg_b = load_regulation("gen9championsvgc2026regmb")
    reg_c = load_regulation("gen9championsvgc2026regmc")
    with pytest.raises(ValueError, match="different Pokemon"):
        Encoder(reg_c, build_vocabulary(reg_b))


def test_the_volatile_vocabulary_has_no_duplicates() -> None:
    assert len(set(VOLATILES)) == len(VOLATILES)


def test_both_entry_points_agree(encoder_and_positions) -> None:  # noqa: ANN001
    """`encode` (JSON) and `encode_positions` (dataclass) must produce identical arrays.

    There is only one implementation -- `encode` converts and defers -- so this is really a
    round-trip check on `Position.to_json` / `from_json`: a field that serialises lossily
    would make the training set disagree with what the search sees, and only the search
    path takes the direct route.

    The direct route exists because `to_json` was 64% of the per-leaf cost in the search
    (0.228 ms against 0.117 ms of encoding and 0.012 ms of forward pass), and the search
    already holds Position objects.
    """
    from pokeuraou.position import Position

    encoder, positions = encoder_and_positions
    from_json = encoder.encode(positions)
    direct = encoder.encode_positions([Position.from_json(p) for p in positions])
    for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
        assert np.array_equal(getattr(from_json, name), getattr(direct, name)), name
    assert from_json.unknown_volatiles == direct.unknown_volatiles


def _after_the_stone_holder_switches_in():  # noqa: ANN202
    """Charizard, the side's only stone, starts on the bench at party slot 2 and comes in.

    The turn is resolved, not assembled: `_do_switch` is what renumbers `Pokemon.slot`, and
    a position built by hand with Charizard already in front never goes through it.
    Everyone else protects, so the one thing the turn does is the switch.
    """
    from pokeuraou.actions import MoveAction, SwitchAction, side_actions

    from ._port import Budget, resolve_turn

    reg = load_regulation("gen9championsvgc2026regmb")
    sheet = {entry.species: entry for entry in load_roster("rizabanadohido").sets}
    own = [sheet[n] for n in ("venusaur", "sylveon", "charizard", "garchomp")]
    foe = [sheet[n] for n in ("incineroar", "toxapex", "garchomp", "venusaur")]
    before = position_from_sets(reg, own, foe)
    assert before.sides[0].mega_capable_slots == [2], "the fixture's premise"

    ours = next(
        action
        for action in side_actions(reg, before, 0)
        if isinstance(action.slots[0], SwitchAction)
        and action.slots[0].species == "charizard"
        and isinstance(action.slots[1], MoveAction)
        and action.slots[1].move_id == "detect"
    )
    theirs = next(
        action
        for action in side_actions(reg, before, 1)
        if all(
            isinstance(one, MoveAction) and one.move_id in ("protect", "banefulbunker")
            for one in action.slots
        )
    )
    turn = resolve_turn(reg, before, [ours, theirs], budget=Budget.exact())
    assert not turn.suspended and turn.branches
    return reg, [branch.position for branch in turn.branches]


def test_can_mega_follows_the_stone_holder_through_a_switch() -> None:
    """A party slot number is not an identity (IKA-121).

    `side.mega_capable_slots` numbers the stone holders once, when the side is built, and
    `_do_switch` renumbers `Pokemon.slot` on every switch. `can_mega` used to read the
    number, so once the holder came in from the bench the flag stood on whoever took its
    old number and not on the holder -- 24.9% of the (decision, side) pairs with the mega
    unspent in `data/ika73/w12`. The legal moves were right all along; they ask the
    Pokemon, and so does the encoder now.
    """
    reg, positions = _after_the_stone_holder_switches_in()
    encoder = Encoder(reg)
    encoded = encoder.encode_positions(positions)
    can_mega = encoder.mon_names.index("can_mega")
    available = encoder.side_names.index("mega_available")
    for b, after in enumerate(positions):
        side = after.sides[0]
        holder = next(m for m in side.pokemon if m.species == "charizard")
        took_its_number = next(m for m in side.pokemon if m.slot == 2)
        # The premise: the switch really moved the holder off the number the side kept.
        assert holder.slot == 0 and holder.active_index == 0
        assert side.mega_capable_slots == [2] and took_its_number.species == "venusaur"

        flags = {m.species: float(encoded.mon[b, 0, m.slot, can_mega]) for m in side.pokemon}
        assert flags == {"charizard": 1.0, "sylveon": 0.0, "venusaur": 0.0, "garchomp": 0.0}
        assert encoded.side[b, 0, available] == 1.0


def _revision_one(reg, positions):  # noqa: ANN001, ANN202
    """`can_mega` and `mega_available` exactly as encode.py wrote them at be3b896.

    Copied from that tree's two expressions rather than derived from the switch under test,
    so the comparison below is with the old code and not with the new code's idea of it.
    """
    encoder = Encoder(reg)
    encoded = encoder.encode_positions(positions)
    can_mega = encoder.mon_names.index("can_mega")
    available = encoder.side_names.index("mega_available")
    for b, position in enumerate(positions):
        for s, side in enumerate(position.sides):
            encoded.side[b, s, available] = (
                1.0 if not side.mega_used and side.mega_capable_slots else 0.0
            )
            for p, mon in enumerate(side.pokemon[: encoder.mons_per_side]):
                encoded.mon[b, s, p, can_mega] = (
                    1.0
                    if mon.slot in side.mega_capable_slots
                    and not side.mega_used
                    and not mon.is_mega
                    else 0.0
                )
    return encoded


def test_the_old_can_mega_switch_is_revision_one() -> None:
    """`EncodingRules(mega_from_slots=True)` is the pre-IKA-121 encoder, bit for bit (IKA-141).

    On the position where the holder has just switched in, so the two rules disagree -- the
    flag lands on Venusaur, which took Charizard's old number, exactly the failure IKA-121
    recorded -- and the current rule is shown to differ on the same arrays, so equality is
    not a comparison that cannot fail.
    """
    from pokeuraou.encode import EncodingRules

    reg, positions = _after_the_stone_holder_switches_in()
    old = Encoder(reg, rules=EncodingRules(mega_from_slots=True)).encode_positions(positions)
    want = _revision_one(reg, positions)
    for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
        assert np.array_equal(getattr(old, name), getattr(want, name)), name

    encoder = Encoder(reg)
    can_mega = encoder.mon_names.index("can_mega")
    for b, after in enumerate(positions):
        side = after.sides[0]
        flags = {m.species: float(old.mon[b, 0, m.slot, can_mega]) for m in side.pokemon}
        assert flags == {"charizard": 0.0, "sylveon": 0.0, "venusaur": 1.0, "garchomp": 0.0}

    new = encoder.encode_positions(positions)
    assert not np.array_equal(new.mon, old.mon), "the two rules agree here; nothing tested"
    # Only the can_mega column moves between the rules.
    moved = np.argwhere(new.mon != old.mon)
    assert set(moved[:, -1].tolist()) == {can_mega}
    assert np.array_equal(new.side, old.side)


def test_the_rules_default_to_the_current_encoding() -> None:
    from pokeuraou.encode import CURRENT_RULES, EncodingRules, rules_of

    reg, positions = _after_the_stone_holder_switches_in()
    assert Encoder(reg).rules == EncodingRules() == CURRENT_RULES
    assert CURRENT_RULES.label() == "new"
    both = EncodingRules(mega_from_slots=True, patch_shares_side=True)
    assert both.label() == "old-can-mega+old-patch"

    class Leaf:
        def __init__(self, encoder: Encoder) -> None:
            self.encoder = encoder

        def from_encoded(self, encoded):  # noqa: ANN001, ANN202
            return encoded

    leaf = Leaf(Encoder(reg, rules=both))
    # However the leaf is handed over: itself, its bound scorer, or a wrapper around that.
    assert rules_of(leaf) == both
    assert rules_of(leaf.from_encoded) == both

    class Wrapper:
        from_encoded = leaf.from_encoded

    assert rules_of(Wrapper()) == both
    assert rules_of(object()) == CURRENT_RULES
