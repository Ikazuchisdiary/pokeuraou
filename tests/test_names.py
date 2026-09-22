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

import json
import sys
from pathlib import Path

import pytest

from pokeuraou.names import load_names, localiser, names_dir
from pokeuraou.regulation import load_regulation

from ._harness import load_tool

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


def _skeleton(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict) -> dict:
    """Runs ``tools/names_report.py --skeleton`` over ``extra`` and returns what it wrote.

    Only the file it rewrites is moved: the regulation and the generated names are the real
    ones, so which kinds still have gaps is what the tool itself sees.
    """
    tool = load_tool("names_report")
    path = tmp_path / "ja-extra.json"
    path.write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(tool, "names_dir", lambda: tmp_path)
    # The usage ranking reads data/, which the skeleton does not use.
    monkeypatch.setattr(tool, "find_cached_chaos", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["names_report.py", "--locale", "ja", "--skeleton"])
    tool.main()
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_skeleton_keeps_a_name_in_a_kind_with_nothing_missing(
    bundle,  # noqa: ANN001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--skeleton`` rewrites the file names are typed into, so it must not lose one.

    It lost ``moves.recharge`` (IKA-115). ``recharge`` is the fake move Showdown substitutes
    on a recharge turn: it has no dex entry, so it is in no regulation's move list and the
    dump does not name it -- the name can only come from this file. Every move the
    regulation does list is translated, so ``moves`` has no gap, and the rewrite skipped
    every kind with no gap; the hand-typed name went with it.
    """
    reg, names, _loc = bundle
    coverage = {c.kind: c for c in names.coverage(reg)}
    # Without this the shape named above is not in the file, and the test would pass
    # without having looked at it.
    assert not coverage["moves"].missing, "moves has a gap now; move this to a kind without"
    item = coverage["items"].missing[0]

    after = _skeleton(
        tmp_path,
        monkeypatch,
        {"moves": {"recharge": "反動で動けない"}, "items": {item: "（手で補った名前）"}},
    )
    assert after.get("moves") == {"recharge": "反動で動けない"}, list(after)
    # A kind with gaps kept its typed names before the fix too; it has to stay that way.
    assert after["items"][item] == "（手で補った名前）"


def test_the_skeleton_keeps_the_files_order_of_kinds(
    bundle,  # noqa: ANN001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun's diff should be what changed, not the kinds trading places.

    The checked-in file holds ``items`` before ``species``. The coverage lists species
    first, and rewriting in that order moved the whole species block to the top of the
    file on a rerun with nothing new in it.
    """
    reg, names, _loc = bundle
    coverage = {c.kind: c for c in names.coverage(reg)}
    item, species = coverage["items"].missing[0], coverage["species"].missing[0]

    after = _skeleton(
        tmp_path,
        monkeypatch,
        {"items": {item: "（手で補った名前）"}, "species": {species: "（手で補った名前）"}},
    )
    assert list(after)[:3] == ["_note", "items", "species"], list(after)
