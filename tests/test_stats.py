"""SP -> stat computation, checked against Showdown.

This is one of the three components whose failure would make every number the tool prints
meaningless, so it is tested exhaustively rather than by sampling: every team-legal
species, every nature, and a spread set that covers the interesting boundaries.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pokeuraou.oracle import Oracle
from pokeuraou.regulation import STAT_IDS, Regulation
from pokeuraou.stats import (
    apply_boost,
    nature_multipliers,
    sp_is_legal,
    sp_legal_mask,
    stats_for,
    stats_from_sp,
)

from .conftest import FORMAT_ID


def boundary_spreads(sp_limit: int, per_stat: int) -> list[dict[str, int]]:
    """Spreads chosen to exercise every branch of the formula.

    Includes the all-zero case, single-stat maxima, the 32/32/2 spread that the project
    brief believed Showdown could not represent, and asymmetric leftovers that catch
    off-by-one truncation.
    """
    out: list[dict[str, int]] = [dict.fromkeys(STAT_IDS, 0)]
    for stat in STAT_IDS:
        out.append({stat: per_stat})
        out.append({stat: 1})
        out.append({stat: per_stat - 1})
    out.extend(
        [
            {"hp": 32, "atk": 32, "spe": 2},  # 66 total across three stats
            {"hp": 32, "atk": 32, "spd": 1, "spe": 1},
            {"hp": 32, "def": 20, "spd": 12, "spe": 2},
            {"hp": 22, "def": 20, "spa": 11, "spe": 13},
            {"hp": 11, "atk": 11, "def": 11, "spa": 11, "spd": 11, "spe": 11},
            {"hp": 1, "atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1},
            {"hp": 31, "atk": 3, "def": 32},
            {"spe": 32, "atk": 32, "def": 2},
        ]
    )
    filled = [{s: d.get(s, 0) for s in STAT_IDS} for d in out]
    assert all(sum(d.values()) <= sp_limit for d in filled)
    assert all(all(0 <= v <= per_stat for v in d.values()) for d in filled)
    # de-duplicate, keep order
    seen: dict[tuple[int, ...], dict[str, int]] = {}
    for d in filled:
        seen.setdefault(tuple(d[s] for s in STAT_IDS), d)
    return list(seen.values())


def test_regulation_meta_matches_champions_rules(reg: Regulation) -> None:
    assert reg.meta.sp_limit == 66
    assert reg.meta.sp_per_stat_max == 32
    assert reg.meta.fixed_iv == 31
    assert reg.meta.level == 50
    assert reg.meta.active_per_side == 2
    assert reg.meta.picked_team_size == 4
    assert reg.meta.team_size == 6
    assert reg.meta.tera_enabled is False
    assert reg.meta.mega_per_side == 1
    # Open Team Sheets in Champions reveals nature; only the SP spread stays hidden.
    assert reg.meta.open_team_sheets is True
    assert reg.meta.team_sheet_reveals_nature is True


def test_only_serious_is_a_real_neutral_nature(reg: Regulation) -> None:
    """Showdown keeps all five neutral natures; the real game has only Serious.

    Rejecting the other four is our responsibility, not Showdown's.
    """
    neutral_in_showdown = {n.name for n in reg.natures.values() if not n.plus and not n.minus}
    assert neutral_in_showdown == {"Hardy", "Docile", "Bashful", "Quirky", "Serious"}
    real = {n.name for n in reg.real_game_natures}
    assert real & neutral_in_showdown == {"Serious"}
    assert len(real) == 21


@pytest.mark.parametrize(
    ("species", "nature", "sp", "expected"),
    [
        # Verified by hand against the game's formula:
        # HP  = (100*2 + 31 + 32*2) * 50/100 + 60 = 207
        # Atk = ((135*2 + 31 + 32*2) * 50/100 + 5) * 1.1 -> trunc 205
        ("kingambit", "Adamant", {"hp": 32, "atk": 32, "spd": 1, "spe": 1},
         {"hp": 207, "atk": 205, "def": 140, "spa": 72, "spd": 106, "spe": 71}),
        ("incineroar", "Impish", {"hp": 32, "def": 20, "spd": 12, "spe": 2},
         {"hp": 202, "atk": 135, "def": 143, "spa": 90, "spd": 122, "spe": 82}),
    ],
)
def test_known_stat_lines(
    reg: Regulation, species: str, nature: str, sp: dict[str, int], expected: dict[str, int]
) -> None:
    assert stats_for(reg, species, nature, sp) == expected


def test_zero_sp_gives_the_bare_formula(reg: Regulation) -> None:
    """With no investment, HP is base + 75 and other stats base + 20 before nature."""
    got = stats_for(reg, "kingambit", "Serious", {})
    base = dict(zip(STAT_IDS, reg.species["kingambit"].base_stats, strict=True))
    assert got["hp"] == base["hp"] + 75
    for stat in ("atk", "def", "spa", "spd", "spe"):
        assert got[stat] == base[stat] + 20


def test_nature_multipliers_shape_and_values(reg: Regulation) -> None:
    num = nature_multipliers(reg, ["Adamant", "Serious", "Timid"])
    assert num.shape == (3, 6)
    assert num[0].tolist() == [100, 110, 100, 90, 100, 100]  # +Atk -SpA
    assert num[1].tolist() == [100] * 6
    assert num[2].tolist() == [100, 90, 100, 100, 100, 110]  # -Atk +Spe


def test_sp_legality(reg: Regulation) -> None:
    assert sp_is_legal(reg, {"hp": 32, "atk": 32, "spe": 2})  # exactly 66
    assert not sp_is_legal(reg, {"hp": 32, "atk": 32, "spe": 3})  # 67
    assert not sp_is_legal(reg, {"hp": 33})  # over the per-stat cap
    assert not sp_is_legal(reg, {"hp": -1})
    assert sp_is_legal(reg, {})

    grid = np.array([[32, 32, 2, 0, 0, 0], [32, 32, 3, 0, 0, 0], [33, 0, 0, 0, 0, 0]])
    assert sp_legal_mask(reg, grid).tolist() == [True, False, False]


def test_apply_boost_matches_showdown_ratios() -> None:
    stat = np.array([200])
    assert apply_boost(stat, np.array([0])).tolist() == [200]
    assert apply_boost(stat, np.array([1])).tolist() == [300]
    assert apply_boost(stat, np.array([2])).tolist() == [400]
    assert apply_boost(stat, np.array([6])).tolist() == [800]
    assert apply_boost(stat, np.array([-1])).tolist() == [133]  # trunc(200 * 2/3)
    assert apply_boost(stat, np.array([-6])).tolist() == [50]
    # clamps beyond +/-6
    assert apply_boost(stat, np.array([9])).tolist() == [800]


def test_level_other_than_50_is_rejected_not_guessed(reg: Regulation) -> None:
    """The champions formula without Level Clause Mod is a level-50 closed form.

    Silently reusing it at another level would produce numbers with no basis, so it
    raises instead.
    """
    assert reg.meta.uses_level_clause_mod is False
    with pytest.raises(ValueError, match="level-50 closed form"):
        stats_for(reg, "kingambit", "Adamant", {"hp": 32}, level=100)


@pytest.mark.oracle
def test_matches_showdown_for_metagame_relevant_species(reg: Regulation, oracle: Oracle) -> None:
    """A fast check over the species most likely to appear, all natures, all spreads."""
    species = [
        "kingambit", "incineroar", "garchomp", "basculegion", "sneasler", "charizardmegay",
        "sinistcha", "whimsicott", "farigiraf", "sylveon", "absolmegaz", "garchompmegaz",
        "lucariomegaz", "delphoxmega", "floettemega", "staraptormega", "raichumegay",
    ]
    natures = [n.name for n in reg.real_game_natures]
    spreads = boundary_spreads(reg.meta.sp_limit, reg.meta.sp_per_stat_max)

    rows = [
        (sp_id, nature, spread)
        for sp_id, nature, spread in itertools.product(species, natures, spreads)
        if sp_id in reg.species
    ]
    assert len(rows) > 5000
    _assert_rows_match(reg, oracle, rows)


@pytest.mark.oracle
@pytest.mark.slow
def test_matches_showdown_for_every_reachable_species(reg: Regulation, oracle: Oracle) -> None:
    """Exhaustive: every species reachable in the format x every nature x every spread.

    Roughly 400 species * 21 natures * 27 spreads = ~230k rows. Marked slow; it is the
    check that has to pass before any printed number can be trusted.
    """
    natures = [n.name for n in reg.real_game_natures]
    spreads = boundary_spreads(reg.meta.sp_limit, reg.meta.sp_per_stat_max)
    rows = [
        (s.id, nature, spread)
        for s in reg.species.values()
        for nature, spread in itertools.product(natures, spreads)
    ]
    assert len(rows) > 200_000
    _assert_rows_match(reg, oracle, rows)


def _assert_rows_match(
    reg: Regulation, oracle: Oracle, rows: list[tuple[str, str, dict[str, int]]]
) -> None:
    expected = oracle.spread_stats(FORMAT_ID, rows)

    base = np.array([reg.species[s].base_stats for s, _, _ in rows], dtype=np.int64)
    sp = np.array([[spread[k] for k in STAT_IDS] for _, _, spread in rows], dtype=np.int64)
    num = nature_multipliers(reg, [n for _, n, _ in rows])
    got = stats_from_sp(reg, base, sp, num)

    want = np.array([[e[k] for k in STAT_IDS] for e in expected], dtype=np.int64)
    if not np.array_equal(got, want):
        bad = np.flatnonzero((got != want).any(axis=1))
        lines = []
        for i in bad[:20]:
            species, nature, spread = rows[i]
            lines.append(
                f"  {species} {nature} {spread}\n"
                f"    ours     {dict(zip(STAT_IDS, got[i].tolist(), strict=True))}\n"
                f"    showdown {dict(zip(STAT_IDS, want[i].tolist(), strict=True))}"
            )
        raise AssertionError(
            f"{len(bad)} of {len(rows)} stat lines diverge from Showdown:\n" + "\n".join(lines)
        )
