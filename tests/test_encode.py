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
    # M-C added Pokemon, so the same integer means a different species. Weights trained on
    # one regulation must not load against the other, and the fingerprint is the guard.
    assert other.fingerprint() != once.fingerprint()


def test_an_encoder_refuses_a_vocabulary_from_another_regulation() -> None:
    reg_b = load_regulation("gen9championsvgc2026regmb")
    reg_c = load_regulation("gen9championsvgc2026regmc")
    with pytest.raises(ValueError, match="different Pokemon"):
        Encoder(reg_c, build_vocabulary(reg_b))


def test_the_volatile_vocabulary_has_no_duplicates() -> None:
    assert len(set(VOLATILES)) == len(VOLATILES)
