"""The paste importer's reading and normalisation, on fixed strings -- no network.

`tools/fetch_pastes.py` turns the Match Up Web sheet into an M-C pool whose one reason to
exist is the SP spread, so the pieces that decide what a spread and a species *are* sit
here, each against the failure it prevents:

- **SP is read from the line labelled EVs.** In the Champions format that is where the
  numbers go, and a label this cannot read is a problem, not a zero.
- **a mega listing becomes base species + stone.** 114 of the sheet's 390 members are
  written as the mega forme; the stone held is what names the base, and a forme the stone
  does not make is reported rather than resolved some other way.
- **a missing field excludes the team and is not filled in.** One real paste leaves out a
  nature; guessing it is the fourth source `teams.py` keeps out. The one way in is a
  transcription (IKA-138): a user-approved field from the same team's open team sheet,
  checked against the standings file and recorded as `supplied`; one that does not match
  the sheet or the paste stops the run.
- **the output is a function of the cached bytes.** Rerunning from the cache has to write
  the same file, or "reproducible" means nothing.

The fixture team is made up for the test (every field legal in the M-C dump); no real
paste is copied here, since the fetched ones are not redistributed.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from pokeuraou.regulation import Regulation, to_id
from pokeuraou.teams import load_roster

from ._harness import load_tool

fp = load_tool("fetch_pastes")

TEAM = """\
Salamence-Mega @ Salamencite
Ability: Intimidate
Level: 50
EVs: 32 Atk / 2 SpD / 32 Spe
Adamant Nature
- Protect
- Dragon Claw
- Double-Edge
- Tailwind

Floette-Mega (F) @ Floettite
Ability: Flower Veil
EVs: 4 HP / 30 SpA / 32 Spe
Timid Nature
- Moonblast
- Dazzling Gleam
- Light of Ruin
- Protect

Incineroar (M) @ Sitrus Berry
Ability: Intimidate
EVs: 32 HP / 17 Def / 17 SpD
Careful Nature
- Fake Out
- Parting Shot
- Flare Blitz
- Darkest Lariat

Bob (Garchomp) (F) @ Life Orb
Ability: Rough Skin
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Indeedee-F (F) @ Psychic Seed
Ability: Psychic Surge
Level: 51
EVs: 32 HP / 32 Def / 2 SpD
Bold Nature
- Follow Me
- Helping Hand
- Psychic
- Protect

Sinistcha-Masterpiece @ Rocky Helmet
Ability: Hospitality
Shiny: Yes
EVs: 32 HP / 20 Def / 14 SpD
Relaxed Nature
- Matcha Gotcha
- Rage Powder
- Trick Room
- Protect
"""


def one(text: str) -> object:
    mons = fp.parse_paste(text)
    assert len(mons) == 1
    return mons[0]


def test_the_head_line_gives_species_nickname_gender_and_item() -> None:
    assert fp.parse_head("Salamence-Mega @ Salamencite") == ("Salamence-Mega", None, None, "Salamencite")
    assert fp.parse_head("Indeedee-F (F) @ Colbur Berry") == ("Indeedee-F", None, "F", "Colbur Berry")
    assert fp.parse_head("Bob (Garchomp) (M) @ Choice Scarf") == ("Garchomp", "Bob", "M", "Choice Scarf")
    assert fp.parse_head("Ditto") == ("Ditto", None, None, None)


def test_sp_is_read_from_the_line_labelled_evs() -> None:
    mon = one(
        "Salamence @ Life Orb\nAbility: Intimidate\nEVs: 19 Atk / 15 SpA / 32 Spe\nNaive Nature\n- Protect"
    )
    assert mon.sp == {"atk": 19, "spa": 15, "spe": 32}
    assert mon.problems == []


def test_an_unreadable_sp_label_is_a_problem_not_a_zero() -> None:
    mon = one("Salamence @ Life Orb\nAbility: Intimidate\nEVs: 19 Atk / 15 Sp. Atk\nNaive Nature\n- Protect")
    assert mon.sp == {"atk": 19}
    assert [f for f, _ in mon.problems] == ["sp"]


def test_the_team_reads_whole_and_clean(reg: Regulation) -> None:
    members, problems = fp.build_team(reg, TEAM)
    assert problems == []
    assert [m["species"] for m in members] == [
        "Salamence",
        "Floette-Eternal",
        "Incineroar",
        "Garchomp",
        "Indeedee-F",
        "Sinistcha-Masterpiece",
    ]
    # The spread keeps the export's numbers, in the repository's stat order, zeros dropped.
    assert members[0]["sp"] == {"atk": 32, "spd": 2, "spe": 32}
    assert list(members[1]["sp"]) == ["hp", "spa", "spe"]
    assert all(sum(m["sp"].values()) <= reg.meta.sp_limit for m in members)


def test_mega_listings_become_base_species_and_stone(reg: Regulation) -> None:
    members, _ = fp.build_team(reg, TEAM)
    salamence, floette = members[0], members[1]
    assert (salamence["species"], salamence["item"], salamence["writtenAs"]) == (
        "Salamence",
        "Salamencite",
        "Salamence-Mega",
    )
    # Floette-Mega's base is the Eternal Flower forme; only the stone's table says so.
    assert (floette["species"], floette["writtenAs"]) == ("Floette-Eternal", "Floette-Mega")
    # And the forme it becomes is the one the paste wrote, read forwards again.
    assert reg.mega_target(to_id(floette["species"]), to_id(floette["item"])) == "floettemega"

    member, problems = fp.normalise(reg, one("Raichu-Mega-Y @ Raichunite Y\nAbility: Lightning Rod"))
    assert member["species"] == "Raichu"
    assert "species" not in [f for f, _ in problems]


def test_a_forme_the_stone_does_not_make_is_a_problem(reg: Regulation) -> None:
    _, problems = fp.normalise(reg, one("Charizard-Mega-Y @ Charizardite X\nAbility: Blaze"))
    assert ("species", "Charizard-Mega-Y is not what Charizardite X makes") in problems


def test_the_ability_has_to_be_the_pre_mega_one(reg: Regulation) -> None:
    _, problems = fp.normalise(reg, one("Salamence-Mega @ Salamencite\nAbility: Aerilate"))
    assert [f for f, _ in problems if f == "ability"] == ["ability"]


def test_a_missing_nature_excludes_the_team_and_is_not_filled_in(reg: Regulation) -> None:
    text = TEAM.replace("Adamant Nature\n", "", 1)
    members, problems = fp.build_team(reg, text)
    assert members[0]["nature"] is None
    assert [(p["member"], p["writtenAs"], p["field"]) for p in problems] == [(0, "Salamence-Mega", "nature")]


def test_the_clauses_are_checked_at_team_level(reg: Regulation) -> None:
    text = TEAM.replace("@ Psychic Seed", "@ Sitrus Berry")
    _, problems = fp.build_team(reg, text)
    assert [(p["field"], p["reason"]) for p in problems] == [("team", "Item Clause: sitrusberry")]


def test_what_the_format_does_not_keep_is_noted_not_dropped(reg: Regulation) -> None:
    members, _ = fp.build_team(reg, TEAM)
    assert members[3]["notes"] == ["nickname 'Bob'"]
    assert members[4]["notes"] == ["Level: 51（形式は 50 固定なので使わない）"]
    assert members[5]["notes"] == ["Shiny: Yes"]
    assert "notes" not in members[0]  # Level: 50 is the format's own level


def test_line_endings_do_not_matter() -> None:
    assert fp.parse_paste(TEAM.replace("\n", "  \r\n")) == fp.parse_paste(TEAM)


SHEET = (
    ',"Slot 1\nSlot 4","Slot 2\nSlot 5",Pokepastes (EVs Included),,"Main\nPlan"\r\n'
    'LIST OF MATCHUPS,,,,Vital Aspects,"Opponent\'s\nPlans"\r\n'
    "Your Team,,,https://pokepast.es/00000000aaaaaaaa ,,\r\n"
    ",,,,,\r\n"
    'Sand,,,https://pokepast.es/0123456789abcdef ,"\n",\r\n'
    ",,,,,\r\n"
    "Rain ,,,https://pokepast.es/fedcba9876543210,,\r\n"
)


def test_the_sheet_is_read_as_csv_and_the_own_row_is_not_an_opponent() -> None:
    entries, own = fp.parse_sheet(SHEET)
    assert entries == [
        fp.SheetEntry(row=5, name="Sand", paste_id="0123456789abcdef"),
        fp.SheetEntry(row=7, name="Rain", paste_id="fedcba9876543210"),
    ]
    assert own == [fp.SheetEntry(row=3, name="Your Team", paste_id="00000000aaaaaaaa")]


def test_a_paste_listed_twice_stops_the_import() -> None:
    with pytest.raises(SystemExit):
        fp.parse_sheet(SHEET.replace("fedcba9876543210", "0123456789abcdef"))


def _pool(reg: Regulation) -> dict:
    entries, own = fp.parse_sheet(SHEET)
    record = {"fetchedAt": "2026-09-23T00:00:00Z"}
    raw = {
        "0123456789abcdef": TEAM.encode(),
        "fedcba9876543210": TEAM.replace("Adamant Nature\n", "", 1).encode(),
    }
    pastes = [(e, raw[e.paste_id], {**record, "sha256": fp.sha256(raw[e.paste_id])}) for e in entries]
    sheet = SHEET.encode()
    return fp.build_pool(reg, (sheet, {**record, "sha256": fp.sha256(sheet)}), pastes, own)


def test_the_pool_file_is_a_function_of_the_cached_bytes(reg: Regulation) -> None:
    first, second = fp.encode(_pool(reg)), fp.encode(_pool(reg))
    assert first == second
    assert b"\r" not in first and first.endswith(b"\n")
    pool = json.loads(first)
    assert "prep list であって field ではない" in pool["character"]
    assert pool["counts"]["teams"] == {"listed": 2, "kept": 1, "excluded": 1}
    assert pool["excluded"][0]["problems"][0]["field"] == "nature"
    assert pool["teams"][0]["source"]["sheetRow"] == 5
    assert pool["ownRowsSkipped"] == [
        {"sheetRow": 3, "name": "Your Team", "paste": "https://pokepast.es/00000000aaaaaaaa"}
    ]


# -- transcriptions from an open team sheet (IKA-138) ---------------------------------------

MISSING = "fedcba9876543210"  # the fixture paste without a nature (see _pool)


def _supplement(**change: object) -> object:
    base = {
        "paste_id": MISSING,
        "member": 0,
        "written_as": "Salamence-Mega",
        "field": "nature",
        "value": "Adamant",
        "event": "Test Regional",
        "player": "Test Player",
        "place": 7,
        "standings_file": "test-event.json.gz",
        "player_code": "test-player",
        "team_index": 1,
        "approved": "test",
    }
    return fp.Supplement(**{**base, **change})


def _standings(tmp_path: Path, nature: str = "Adamant", moves: tuple[str, ...] = ()) -> Path:
    """A made-up standings file whose second sheet member is the fixture Salamence."""
    salamence = {
        "name": "Salamence",
        "ability": "Intimidate",
        "nature": nature,
        "item": "Salamencite",
        "moves": [{"name": m} for m in moves or ("Protect", "Dragon Claw", "Double-Edge", "Tailwind")],
    }
    other = {"name": "Incineroar", "ability": "Intimidate", "nature": "Careful", "item": "", "moves": []}
    doc = {
        "event": {"name": "Test Regional"},
        "standings": {"test-player": {"name": "Test Player", "place": 7, "team": [other, salamence]}},
    }
    (tmp_path / "test-event.json.gz").write_bytes(gzip.compress(json.dumps(doc).encode("utf-8")))
    return tmp_path


def _pool_with(reg: Regulation, supplements: tuple, standings_dir: Path | None) -> dict:
    entries, own = fp.parse_sheet(SHEET)
    record = {"fetchedAt": "2026-09-23T00:00:00Z"}
    raw = {
        "0123456789abcdef": TEAM.encode(),
        MISSING: TEAM.replace("Adamant Nature\n", "", 1).encode(),
    }
    pastes = [(e, raw[e.paste_id], {**record, "sha256": fp.sha256(raw[e.paste_id])}) for e in entries]
    sheet = SHEET.encode()
    return fp.build_pool(
        reg, (sheet, {**record, "sha256": fp.sha256(sheet)}), pastes, own, supplements, standings_dir
    )


def test_without_a_supplement_the_team_is_still_excluded(reg: Regulation) -> None:
    pool = _pool_with(reg, (), None)
    assert pool["counts"]["teams"] == {"listed": 2, "kept": 1, "excluded": 1}
    assert pool["excluded"][0]["id"] == MISSING
    assert pool["counts"]["suppliedFields"] == 0
    assert all("supplied" not in t for t in pool["teams"] + pool["excluded"])
    # The shipped supplements name real pastes only, so the fixtures are untouched by them.
    assert fp.encode(_pool(reg)) == fp.encode(_pool_with(reg, (), None))


def test_a_checked_supplement_fills_the_field_and_says_where_from(reg: Regulation, tmp_path: Path) -> None:
    pool = _pool_with(reg, (_supplement(),), _standings(tmp_path))
    assert pool["counts"]["teams"] == {"listed": 2, "kept": 2, "excluded": 0}
    team = next(t for t in pool["teams"] if t["id"] == MISSING)
    salamence = team["team"][0]
    assert (salamence["nature"], salamence["supplied"]) == ("Adamant", ["nature"])
    assert all("supplied" not in m for m in team["team"][1:])
    (record,) = team["supplied"]
    assert (record["member"], record["field"], record["value"]) == (0, "nature", "Adamant")
    source = record["source"]
    assert (source["event"], source["player"], source["place"]) == ("Test Regional", "Test Player", 7)
    assert source["standings"] == "data/standings/test-event.json.gz"
    assert source["pointer"] == "standings.test-player.team[1].nature"
    assert source["checkedAgainst"]["matched"] == ["ability", "item", "moves", "species"]
    assert pool["counts"]["suppliedFields"] == 1


def test_a_supplement_without_its_file_is_applied_but_marked_unchecked(
    reg: Regulation, tmp_path: Path
) -> None:
    pool = _pool_with(reg, (_supplement(),), tmp_path)  # the directory has no standings file
    team = next(t for t in pool["teams"] if t["id"] == MISSING)
    assert team["supplied"][0]["source"]["checkedAgainst"] is None


@pytest.mark.parametrize(
    ("supplement", "standings"),
    [
        ({"value": "Jolly"}, {}),  # the sheet says Adamant
        ({}, {"moves": ("Protect", "Draco Meteor", "Double-Edge", "Tailwind")}),  # another individual
        ({"player": "Someone Else"}, {}),
        ({"written_as": "Floette-Mega"}, {}),  # the paste has Salamence there
        ({"member": 1, "written_as": "Floette-Mega"}, {}),  # the paste writes Timid: no override
        ({"field": "sp"}, {}),  # the sheet blanks SP
    ],
)
def test_a_supplement_that_does_not_match_stops_the_run(
    reg: Regulation, tmp_path: Path, supplement: dict, standings: dict
) -> None:
    with pytest.raises(SystemExit):
        _pool_with(reg, (_supplement(**supplement),), _standings(tmp_path, **standings))


def test_the_shipped_supplement_is_the_approved_one() -> None:
    (s,) = fp.SUPPLEMENTS
    assert (s.paste_id, s.member, s.written_as, s.field, s.value) == (
        "9de5ee0a9fd58d26",
        0,
        "Salamence-Mega",
        "nature",
        "Naive",
    )
    assert (s.event, s.player, s.place, s.standings_file) == (
        "Baltimore Regional",
        "Wolfe Glick",
        15,
        "2027-baltimore.json.gz",
    )


def test_a_kept_team_reads_through_the_roster_loader(reg: Regulation, tmp_path: Path) -> None:
    team = _pool(reg)["teams"][0]
    path = tmp_path / "roster.json"
    doc = {"id": team["id"], "name": team["name"], "regulation": reg.meta.format_id, "team": team["team"]}
    path.write_bytes(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    roster = load_roster(path)
    assert [s.species for s in roster.sets][:2] == ["salamence", "floetteeternal"]
    assert roster.sets[0].sp == {"hp": 0, "atk": 32, "def": 0, "spa": 0, "spd": 2, "spe": 32}
