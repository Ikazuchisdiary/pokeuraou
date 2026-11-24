"""IKA-448: tools/field_pool.py groups field teams, picks spreads, and lists unseen ids."""

from __future__ import annotations

from types import SimpleNamespace

from pokeuraou.regulation import load_regulation
from pokeuraou.standings import TeamMember, TournamentTeam
from tests._harness import load_tool

fp = load_tool("field_pool")
REG = "gen9championsvgc2026regmc"


def _team(species: tuple[str, ...], place: int, day2: bool, item: str | None = None) -> TournamentTeam:
    members = tuple(
        TeamMember(s, "intimidate", item if i == 0 else None, "Adamant", ("protect",))
        for i, s in enumerate(species))
    return TournamentTeam("p", place, "us", 0, 0, False, day2, members)


SIX_A = ("incineroar", "rillaboom", "sneasler", "kingambit", "garchomp", "sylveon")
SIX_B = ("charizard", "venusaur", "garchomp", "incineroar", "sylveon", "toxapex")


def test_groups_are_ordered_by_day_two_then_entries() -> None:
    events = [("e1", [_team(SIX_A, 50, False), _team(SIX_A, 60, False), _team(SIX_A, 70, False),
                      _team(SIX_B, 9, True)]),
              ("e2", [_team(SIX_B[::-1], 3, True)])]
    groups = fp.group_teams(events)
    assert [g["species"] for g in groups] == [frozenset(SIX_B), frozenset(SIX_A)]
    assert (groups[0]["day2"], groups[0]["entries"]) == (2, 2)
    event, team = fp.best_team(groups[0])
    assert (event, team.place) == ("e2", 3)  # the order of the six does not split a group


def test_spread_falls_back_from_nature_to_species_to_neutral() -> None:
    reg = load_regulation(REG)
    sets = [SimpleNamespace(species="garchomp", nature="Jolly", sp={"atk": 32, "spe": 32, "hp": 2}),
            SimpleNamespace(species="garchomp", nature="Adamant", sp={"hp": 32, "atk": 32, "def": 2})]
    spreads = fp.Spreads(SimpleNamespace(teams=[SimpleNamespace(sets=sets)]), reg)
    assert spreads.pick("garchomp", "Jolly") == ({"hp": 2, "atk": 32, "spe": 32}, "pool-nature")
    assert spreads.pick("garchomp", "Modest")[1] == "pool-species"
    sp, label = spreads.pick("toxapex", "Bold")
    assert label == "neutral" and sum(sp.values()) > 0


def test_unseen_ids_are_the_ones_no_pool_set_has() -> None:
    reg = load_regulation(REG)
    known = {"species": set(SIX_A), "moves": {"protect"}, "items": set(), "abilities": {"intimidate"}}
    spreads = fp.Spreads(SimpleNamespace(teams=[]), reg)
    team = _team(SIX_B, 1, True, item="garchompitez")
    entry = fp.to_entry("e", team, reg, spreads, {}, known, [frozenset(SIX_A)])
    assert entry["ika448"]["overlap"] == 3
    assert entry["ika448"]["unseen"]["species"] == ["charizard", "toxapex", "venusaur"]
    assert entry["ika448"]["unseen"]["items"] == ["garchompitez"]
    assert entry["ika448"]["unseen"]["moves"] == []
