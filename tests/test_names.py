"""Japanese names, and the rule that a missing translation is reported rather than faked.

Output has to be readable in Japanese: that is how the game presents itself and how every
build article is written, so an English-only tool cannot be checked against the sources it
was built from. The risk is not that a name is missing -- it is that a missing name gets
filled with something plausible, since a wrong Japanese name reads as authoritative in a
way an English one does not.

So the tests are mostly about the fallbacks:

- a translation absent from Showdown's data falls through to the English name, never to a
  guess, and :meth:`Names.coverage` counts how often;
- a forme is composed from base plus suffix, and an untranslated suffix keeps the English
  part rather than collapsing onto a sibling forme -- "ルガルガン (Dusk)" and not bare
  "ルガルガン", which would be a different Pokemon with different Speed;
- the mega prefix is applied because the cited Japanese tournament reports use it
  throughout, which is a convention with a source, not an invention.
"""

from __future__ import annotations

import pytest

from pokeuraou.names import load_names, localiser, names_dir
from pokeuraou.regulation import load_regulation

REGULATION = "gen9championsvgc2026regmb"


@pytest.fixture(scope="module")
def bundle():  # noqa: ANN201
    if not (names_dir() / "ja.json").exists():
        pytest.skip("no ja names; run node packages/sim-bridge/dist/cli/dump-names.js")
    reg = load_regulation(REGULATION)
    loc = localiser(reg, "ja")
    assert loc is not None
    return reg, load_names("ja"), loc


def test_the_pieces_that_have_to_be_complete_are(bundle) -> None:  # noqa: ANN001
    """Moves, abilities, natures and stats are what a result line is made of.

    A partially translated move list would produce output mixing "シャドーボール" with
    "Shadow Ball" in the same table, so these are asserted at 100% rather than measured.
    """
    reg, names, _loc = bundle
    coverage = {c.kind: c for c in names.coverage(reg)}
    for kind in ("moves", "abilities", "natures", "stats"):
        assert coverage[kind].fraction == 1.0, str(coverage[kind])


def test_species_coverage_is_high_and_the_gaps_are_named(bundle) -> None:  # noqa: ANN001
    reg, names, _loc = bundle
    coverage = {c.kind: c for c in names.coverage(reg)}
    species = coverage["species"]
    # 97.2% measured; the remainder are Alcremie cosmetic formes and Floette, which
    # Showdown's ja data does not carry. The point of the assertion is that a regression
    # in the forme composition (which was 68.9% before it existed) would trip it.
    assert species.fraction > 0.95, str(species)
    assert species.missing, "every gap should be listed, not summarised away"


def test_a_missing_translation_falls_back_to_english(bundle) -> None:  # noqa: ANN001
    reg, names, loc = bundle
    coverage = {c.kind: c for c in names.coverage(reg)}
    missing_items = [i for i in coverage["items"].missing if i in reg.items]
    assert missing_items, "expected the mega stones to be untranslated in Showdown's ja"
    for item_id in missing_items[:20]:
        # The English name, not the raw id and not a guess.
        assert loc.item(item_id) == reg.items[item_id].name
    assert names.item_name(None) == "なし"


def test_formes_compose_and_stay_distinguishable(bundle) -> None:  # noqa: ANN001
    _reg, _names, loc = bundle
    # Full name in Showdown's data.
    assert loc.species("sinistcha") == "ヤバソチャ"
    assert loc.species("charizard") == "リザードン"
    # Base translated, forme suffix not: the suffix stays English so the two Lycanroc
    # formes cannot be confused. They have different base Speed, so merging them would
    # change turn order.
    dusk = loc.species("lycanrocdusk")
    assert dusk.startswith("ルガルガン") and "Dusk" in dusk
    assert loc.species("lycanroc") == "ルガルガン"
    assert loc.species("lycanrocdusk") != loc.species("lycanroc")
    # Base and forme both translated.
    assert loc.species("taurospaldeaaqua").startswith("ケンタロス")
    assert loc.species("taurospaldeaaqua") != loc.species("tauros")


def test_mega_formes_use_the_convention_the_sources_use(bundle) -> None:  # noqa: ANN001
    """メガ + base, which is what the cited Japanese tournament reports write."""
    _reg, _names, loc = bundle
    assert loc.species("charizardmegay") == "メガリザードンＹ"  # from Showdown's own data
    assert loc.species("charizardmegax") == "メガリザードンＸ"
    # Champions originals, which Showdown has not translated: the prefix is applied.
    assert loc.species("scovillainmega") == "メガスコヴィラン"
    assert loc.species("froslassmega") == "メガユキメノコ"
    assert loc.species("dragonitemega") == "メガカイリュー"
    assert loc.species("aerodactylmega") == "メガプテラ"
    assert loc.species("delphoxmega") == "メガマフォクシー"


def test_the_rest_of_the_display_vocabulary(bundle) -> None:  # noqa: ANN001
    _reg, _names, loc = bundle
    assert loc.move("matchagotcha") == "シャカシャカほう"
    assert loc.move("fakeout") == "ねこだまし"
    assert loc.item("focussash") == "きあいのタスキ"
    assert loc.ability("intimidate") == "いかく"
    assert loc.ability("spicyspray") == "とびだすハバネロ"
    assert loc.nature("Adamant") == "いじっぱり"
    assert loc.nature("Serious") == "まじめ"  # the only neutral nature in Champions
    assert [loc.stat(s, short=True) for s in ("hp", "atk", "spe")] == ["Ｈ", "Ａ", "Ｓ"]
    assert loc.status("brn") == "やけど"
    assert loc.status(None) == "-"
    assert loc.spread({"hp": 32, "def": 32, "spd": 2}) == "32-0-32-0-2-0"


def test_asking_for_no_localisation_gives_none() -> None:
    reg = load_regulation(REGULATION)
    assert localiser(reg, "en") is None
    assert localiser(reg, None) is None


def test_an_unknown_id_is_returned_unchanged(bundle) -> None:  # noqa: ANN001
    _reg, names, loc = bundle
    assert loc.species("notapokemon") == "notapokemon"
    assert names.move_name("notamove") == "notamove"
    assert names.ability_name("notanability", "Fallback") == "Fallback"
