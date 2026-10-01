"""The ranked-match entry (IKA-407): a pasted team, six species, the estimate, the read.

What has to hold, each against the failure it prevents:

- **the estimate is the field's commonest set and says so.** A species the field plays one way
  gets that set with its counts; one it plays a dozen ways, none of them 3 times, is taken part
  by part and says so; one it never shows is refused, not made up.
- **the spread has a stated source and obeys the regulation.** The pool's same-nature spread,
  then its same-species one, then a neutral one labelled provisional; the neutral one stays
  inside the regulation's total and per-stat cap for every species.
- **the six is a legal team.** An item two members would share moves the second to another
  candidate; when none exists the clash is reported.
- **an override is checked as a roster member is.** And only what it names is marked.
- **a paste reads as the pool's pastes do, and a bad one gives no roster.**
- **the read is `solve_entry`, unchanged.** True sets in, the same entry out bit for bit; and
  the comparison can fail: one changed set moves the answer.
- **the ids the leaf never trained on are named.** From the encoded sets, through the
  vocabulary each was encoded with; "absent" (0) names nothing.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou import humanplay, rankedentry
from pokeuraou.damage import register_mega_stones
from pokeuraou.priors import SampledSet
from pokeuraou.rankedentry import (
    FieldPrior,
    LearnedIds,
    RankedError,
    UnknownSpecies,
    apply_override,
    blank_set,
    neutral_spread,
    opponent_roster,
    roster_from_paste,
)
from pokeuraou.rankedweb import RankedApp, RankedServer
from pokeuraou.regulation import STAT_IDS, to_id
from pokeuraou.selection_book import BookEntry
from pokeuraou.standings import Standings, TeamMember, TournamentTeam
from pokeuraou.teams import Roster, all_selections

ITEMS = ["leftovers", "sitrusberry", "focussash", "lumberry", "choicescarf", "scopelens",
         "lifeorb", "mentalherb", "widelens", "shellbell", "whiteherb", "kingsrock"]
MOVES = ["protect", "fakeout", "earthquake", "closecombat", "icywind", "tailwind", "flamethrower",
         "surf", "thunderbolt", "icebeam"]


def _species(reg, count: int) -> list[str]:  # noqa: ANN001
    """Distinct-base, non-mega, team-legal species, in a fixed order."""
    out, bases = [], set()
    for found in sorted(reg.team_legal_species, key=lambda s: s.id):
        if found.is_mega or found.base_species in bases:
            continue
        bases.add(found.base_species)
        out.append(found.id)
        if len(out) == count:
            break
    return out


def _member(reg, species: str, item: str | None, moves: list[str], nature: str = "Adamant") -> TeamMember:  # noqa: ANN001
    return TeamMember(species=species, ability=to_id(reg.species[species].abilities[0]), item=item,
                      nature=nature, moves=tuple(moves))


def _standings(teams: list[list[TeamMember]]) -> Standings:
    return Standings(
        event="Test Open", event_format="M-C", player_count=len(teams),
        teams=[TournamentTeam(player=f"p{i}", place=i, country="", wins=0, losses=0, made_cut=False,
                              phase_two=False, members=tuple(t)) for i, t in enumerate(teams)],
    )


def _pool(reg, sets: list[SampledSet]) -> SimpleNamespace:  # noqa: ANN001
    return SimpleNamespace(id="test-pool", teams=[SimpleNamespace(sets=[s]) for s in sets])


@pytest.fixture(scope="module")
def world(reg):  # noqa: ANN001, ANN201
    """Species A: one set 4 times and another once. B: four different sets (no whole set 3
    times). C: the commonest item is A's. D: absent from the field."""
    a, b, c, d = _species(reg, 4)
    x = (ITEMS[0], MOVES[:4])
    y = (ITEMS[1], MOVES[4:8])
    teams = []
    for _ in range(4):
        teams.append([_member(reg, a, *x)])
    teams.append([_member(reg, a, *y)])
    for i in range(4):
        teams.append([_member(reg, b, ITEMS[2 + i], [MOVES[i], MOVES[i + 1], MOVES[i + 2], MOVES[9]],
                              nature=("Adamant", "Jolly", "Bold", "Timid")[i])])
    for _ in range(3):
        teams.append([_member(reg, c, ITEMS[0], MOVES[1:5])])  # the same item as A's commonest
    teams.append([_member(reg, c, ITEMS[6], MOVES[2:6])])
    pool_sets = [
        SampledSet(species=a, ability=to_id(reg.species[a].abilities[0]), item=None, nature="Adamant",
                   sp={"atk": 32, "spe": 32, "hp": 2}, moves=["protect"]),
        SampledSet(species=a, ability=to_id(reg.species[a].abilities[0]), item=None, nature="Adamant",
                   sp={"atk": 32, "spe": 32, "hp": 2}, moves=["protect"]),
        SampledSet(species=a, ability=to_id(reg.species[a].abilities[0]), item=None, nature="Modest",
                   sp={"spa": 32, "hp": 32, "def": 2}, moves=["protect"]),
        SampledSet(species=c, ability=to_id(reg.species[c].abilities[0]), item=None, nature="Bold",
                   sp={"hp": 32, "def": 32, "spd": 2}, moves=["protect"]),
    ]
    prior = FieldPrior(reg, _standings(teams), _pool(reg, pool_sets))
    return SimpleNamespace(reg=reg, prior=prior, a=a, b=b, c=c, d=d, x=x, y=y)


# ------------------------------------------------------------------------------ the estimate


def test_the_commonest_whole_set_is_used_with_its_counts(world) -> None:  # noqa: ANN001
    one = world.prior.fill(world.a)
    assert one.kind == "whole" and (one.species_n, one.whole_n) == (5, 4)
    assert one.set.item == world.x[0] and sorted(one.set.moves) == sorted(world.x[1])
    assert [c.count for c in one.alternatives] == [4, 1]


def test_a_rare_commonest_set_is_taken_part_by_part_and_says_so(world) -> None:  # noqa: ANN001
    one = world.prior.fill(world.b)
    assert one.kind == "part" and one.whole_n == 0 and one.species_n == 4
    assert one.alternatives[0].count == 1 < rankedentry.WHOLE_MIN
    assert "部分ごと" in one.notes[0]
    # MOVES[9] is in every member's set: it must be among the four commonest moves.
    assert MOVES[9] in one.set.moves and len(one.set.moves) == 4


def test_a_species_the_field_never_shows_is_refused(world) -> None:  # noqa: ANN001
    with pytest.raises(UnknownSpecies, match="Test Open"):
        world.prior.fill(world.d)


def test_the_spread_comes_from_the_pool_same_nature_first(world) -> None:  # noqa: ANN001
    a = world.prior.fill(world.a)
    assert (a.sp_source, a.sp_n) == ("pool-nature", 2)  # the two Adamant ones, not the Modest one
    assert a.set.sp == {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
    # Same species, a nature the pool has none of: the species' commonest spread, labelled so.
    c = world.prior.fill(world.c)
    assert c.sp_source == "pool-species" and c.sp_n == 1  # Adamant, but the pool's one is Bold
    b = world.prior.fill(world.b)
    assert b.sp_source == "neutral" and b.sp_provisional and b.sp_n == 0


def test_the_neutral_spread_obeys_the_regulation_for_every_species(reg) -> None:  # noqa: ANN001
    for found in reg.team_legal_species:
        sp = neutral_spread(reg, found.id)
        assert set(sp) == set(STAT_IDS)
        assert sum(sp.values()) <= reg.meta.sp_limit, found.id
        assert max(sp.values()) <= reg.meta.sp_per_stat_max, found.id
        assert sum(sp.values()) == reg.meta.sp_limit  # the whole total is spent
    with pytest.raises(RankedError, match="上限"):
        rankedentry.check_spread(reg, {"hp": reg.meta.sp_per_stat_max + 1})
    with pytest.raises(RankedError, match="合計"):
        rankedentry.check_spread(reg, dict.fromkeys(STAT_IDS, reg.meta.sp_limit // 6 + 1))


def test_an_item_two_members_would_share_moves_the_second_to_another_candidate(world) -> None:  # noqa: ANN001
    sets, problems = world.prior.fill_team([world.a, world.c])
    assert sets[0].set.item == world.x[0]
    assert sets[1].set.item == ITEMS[6] and any("重複" in n for n in sets[1].notes)
    assert problems == []
    # The control: with no other item to move to, the clash is reported rather than hidden.
    clash = ([[_member(world.reg, world.a, ITEMS[0], MOVES[:4])] for _ in range(3)]
             + [[_member(world.reg, world.c, ITEMS[0], MOVES[1:5])] for _ in range(3)])
    only = FieldPrior(world.reg, _standings(clash), None)
    _sets, problems = only.fill_team([world.a, world.c])
    assert any("重複" in p for p in problems)
    # Two of one species is a Species Clause problem.
    assert any("同じ種族" in p for p in world.prior.fill_team([world.a, world.a])[1])


def test_choosing_an_alternative_changes_the_set_and_the_label(world) -> None:  # noqa: ANN001
    one = world.prior.fill(world.a)
    other = world.prior.choose(one, 1)
    assert other.kind == "candidate" and other.set.item == world.y[0]
    assert sorted(other.set.moves) == sorted(world.y[1])
    with pytest.raises(RankedError):
        world.prior.choose(one, 5)


# ------------------------------------------------------------------------------ the composition


@pytest.fixture(scope="module")
def duo(reg):  # noqa: ANN001, ANN201
    """A is played with B as set X (4 teams) and with C as set Y (6 teams): alone, Y is the
    commonest; read for a six holding B, X is."""
    a, b, c = _species(reg, 3)
    x, y = (ITEMS[0], MOVES[:4]), (ITEMS[1], MOVES[4:8])
    teams = ([[_member(reg, a, *x), _member(reg, b, ITEMS[2], MOVES[:4])] for _ in range(4)]
             + [[_member(reg, a, *y), _member(reg, c, ITEMS[3], MOVES[:4])] for _ in range(6)])
    return SimpleNamespace(prior=FieldPrior(reg, _standings(teams), None), a=a, b=b, c=c, x=x, y=y)


def test_the_set_is_read_from_the_teams_that_share_the_most_of_the_six(duo) -> None:  # noqa: ANN001
    alone = duo.prior.belief(duo.a, tier_min=3)
    assert alone.overlap == 1 and len(alone.members) == 10
    assert duo.prior.estimate(alone).set.item == duo.y[0]  # the species' marginal: Y, 6 of 10
    with_b = duo.prior.belief(duo.a, [duo.a, duo.b], tier_min=3)
    assert with_b.overlap == 2 and len(with_b.members) == 4 and with_b.teams_in_tier() == 4
    assert with_b.tiers == ((2, 4), (1, 6))
    one = duo.prior.estimate(with_b)
    assert one.set.item == duo.x[0] and one.species_n == 4  # the composition moved the answer
    # The control: a six that holds C reads as the marginal does (the same teams).
    with_c = duo.prior.belief(duo.a, [duo.a, duo.c], tier_min=3)
    assert with_c.overlap == 2 and duo.prior.estimate(with_c).set.item == duo.y[0]
    # Too few teams at the top tier: the next one down is used (here, every team).
    assert duo.prior.belief(duo.a, [duo.a, duo.b], tier_min=5).overlap == 1
    # fill_team reads every member for the whole six.
    sets, _problems = duo.prior.fill_team([duo.a], [duo.a, duo.b])
    assert sets[0].belief is not None and sets[0].belief.overlap in (1, 2)


def test_an_observation_removes_the_sets_that_cannot_have_produced_it(duo) -> None:  # noqa: ANN001
    base = duo.prior.belief(duo.a, tier_min=3)
    seen_move = base.observe(rankedentry.Observation(moves=frozenset({MOVES[5]})))  # only Y has it
    assert len(seen_move.members) == 6 and {m.item for m in seen_move.members} == {duo.y[0]}
    assert duo.prior.estimate(seen_move).set.item == duo.y[0]
    seen_item = base.observe(rankedentry.Observation(item=duo.x[0]))
    assert len(seen_item.members) == 4 and duo.prior.estimate(seen_item).set.item == duo.x[0]
    both = seen_move.observe(rankedentry.Observation(item=duo.x[0]))
    assert both.members == () and both.seen.item == duo.x[0] and MOVES[5] in both.seen.moves
    with pytest.raises(RankedError, match="見たこと"):
        duo.prior.estimate(both)
    assert len(base.members) == 10  # observing leaves the belief it came from alone
    # Moves that belong to different sets together fit none.
    assert len(base.observe(rankedentry.Observation(moves=frozenset({MOVES[0], MOVES[4]}))).members) == 0


# ------------------------------------------------------------------------------ overrides


def test_an_override_is_validated_and_marks_only_what_it_names(world) -> None:  # noqa: ANN001
    reg = world.reg
    one = world.prior.fill(world.a)
    new = apply_override(reg, one, {"item": "focussash", "moves": ["protect", "fakeout"]})
    assert new.set.item == "focussash" and new.set.moves == ["protect", "fakeout"]
    assert new.overridden == ("item", "moves") and new.kind == "override"
    assert new.set.nature == one.set.nature and new.set.sp == one.set.sp  # untouched
    for bad, text in (({"ability": "notanability"}, "特性"), ({"item": "notanitem"}, "持ち物"),
                      ({"nature": "Grumpy"}, "性格"), ({"moves": ["protect", "protect"]}, "同じ技"),
                      ({"moves": ["notamove"]}, "技"), ({"moves": []}, "1〜4"),
                      ({"sp": {"atk": 33}}, "上限")):
        with pytest.raises(RankedError, match=text):
            apply_override(reg, one, bad)
    spread = apply_override(reg, one, {"sp": {"hp": 2, "atk": 32, "spe": 32}})
    assert spread.sp_source == "override" and not spread.sp_provisional


def test_a_species_with_no_field_data_is_a_blank_to_fill_by_hand(world) -> None:  # noqa: ANN001
    reg = world.reg
    blank = blank_set(reg, world.d)
    assert blank.kind == "manual" and blank.set.moves == [] and blank.sp_provisional
    with pytest.raises(RankedError, match="技がありません"):
        opponent_roster(reg, [blank.set] * reg.meta.team_size)
    filled = apply_override(reg, blank, {"moves": MOVES[:4]})
    assert filled.kind == "manual" and filled.overridden == ("moves",)


# ------------------------------------------------------------------------------ the paste


def _paste(reg, species: list[str], *, nature: bool = True, sp: bool = True) -> str:  # noqa: ANN001
    blocks = []
    for i, sid in enumerate(species):
        lines = [f"{reg.species[sid].name} @ {reg.items[ITEMS[i]].name}",
                 f"Ability: {reg.species[sid].abilities[0]}"]
        if sp:
            lines.append("EVs: 32 HP / 32 Atk / 2 Def")
        if nature:
            lines.append("Adamant Nature")
        lines += [f"- {reg.moves[m].name}" for m in MOVES[:4]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def test_a_paste_reads_as_a_roster_and_a_bad_one_gives_none(reg) -> None:  # noqa: ANN001
    six = _species(reg, 6)
    good = roster_from_paste(reg, _paste(reg, six))
    assert good.roster is not None and not good.problems and not good.warnings
    assert [s.species for s in good.roster.sets] == six
    assert good.roster.sets[0].sp["atk"] == 32 and good.roster.sets[0].nature == "Adamant"
    for text, field in ((_paste(reg, six, nature=False), "nature"), (_paste(reg, six[:5]), "team")):
        bad = roster_from_paste(reg, text)
        assert bad.roster is None and any(p["field"] == field for p in bad.problems)
    spless = roster_from_paste(reg, _paste(reg, six, sp=False))
    assert spless.roster is not None and "配分" in spless.warnings[0]


# ------------------------------------------------------------------------------ the read


def _stub(positions):  # noqa: ANN001, ANN202
    out = np.empty(len(positions), dtype=np.float64)
    for i, pos in enumerate(positions):
        strength = [sum(m.hp * (2 if m.active_index is not None else 1) for m in side.pokemon)
                    for side in pos.sides]
        out[i] = 1.0 / (1.0 + np.exp(-(strength[0] - strength[1]) / 150.0))
    return out


def _rosters(reg):  # noqa: ANN001, ANN202
    species = _species(reg, 12)

    def roster(name: str, names: list[str], offset: int) -> Roster:
        sets = [SampledSet(species=s, ability=to_id(reg.species[s].abilities[0]), item=ITEMS[offset + i],
                           nature="Adamant", sp={"hp": 2, "atk": 32, "spe": 32},
                           moves=list(MOVES[i % 3: i % 3 + 4])) for i, s in enumerate(names)]
        return Roster(id=name, name=name, reg=reg, sets=sets, shown_stats=[None] * 6, source={})
    return roster("mine", species[:6], 0), roster("them", species[6:], 6)


def test_true_sets_in_give_solve_entry_bit_for_bit_and_one_changed_set_moves_it(reg, port) -> None:  # noqa: ANN001
    register_mega_stones(reg)
    mine, them = _rosters(reg)
    direct = humanplay.solve_entry(reg, (mine, them), _stub, "stub")
    via = rankedentry.solve_ranked(reg, mine, opponent_roster(reg, them.sets), _stub, "stub")
    assert np.array_equal(direct.our_strategy, via.our_strategy)
    assert direct.value == via.value and direct.selections == via.selections
    assert np.array_equal(direct.their_strategies[0], via.their_strategies[0])
    # The comparison can fail: the same six with one set changed is a different solve.
    changed = list(them.sets)
    # (the stub reads HP, so the change is a spread: a leaf that reads moves would move on those too)
    changed[0] = apply_override(reg, rankedentry.OpponentSet(changed[0], "manual", 0, 0, (), "override", 0),
                                {"sp": {"hp": 32, "atk": 2, "spe": 32}}).set
    other = rankedentry.solve_ranked(reg, mine, opponent_roster(reg, changed), _stub, "stub")
    assert (not np.array_equal(direct.our_strategy, other.our_strategy)) or direct.value != other.value


# ------------------------------------------------------------------------------ learned ids


def test_learned_ids_come_through_the_vocabulary_each_set_was_encoded_with(  # noqa: ANN001
    reg, tmp_path, monkeypatch,
) -> None:
    from pokeuraou.encode import build_vocabulary

    vocab = build_vocabulary(reg)
    inv = {k: {v: n for n, v in t.items()} for k, t in
           (("species", vocab.species), ("ability", vocab.abilities), ("item", vocab.items),
            ("moves", vocab.moves))}
    arrays = {
        "species": np.array([[[0, 3, 5, 0]]]), "ability": np.array([[[2, 0, 0, 0]]]),
        "item": np.array([[[0, 0, 4, 0]]]), "moves": np.array([[[[1, 0, 7, 0]] * 4]]),
    }
    path = tmp_path / "enc.npz"
    np.savez(path, meta_json=np.array(json.dumps({"format_id": reg.meta.format_id})), **arrays)
    got = rankedentry._ids_in_npz(path, reg.meta.format_id)
    assert got["species"] == {inv["species"][3], inv["species"][5]}  # 0 names nothing
    assert got["moves"] == {inv["moves"][1], inv["moves"][7]}
    with pytest.raises(RankedError, match="encoded for"):
        rankedentry._ids_in_npz(path, "gen9championsvgc2026regmb")
    monkeypatch.setattr(rankedentry, "LEARNED_SOURCES", (("enc.npz", reg.meta.format_id),))
    built = rankedentry.load_learned_ids(tmp_path)  # builds and writes the cache
    assert (tmp_path / rankedentry.LEARNED_CACHE).exists()
    monkeypatch.setattr(rankedentry, "LEARNED_SOURCES", ())
    assert rankedentry.load_learned_ids(tmp_path) == built  # the second read is the cache
    one = SampledSet(species=inv["species"][3], ability=inv["ability"][2], item=inv["item"][4],
                     nature="Adamant", sp={}, moves=[inv["moves"][1], inv["moves"][2]])
    assert built.missing(one) == [("move", inv["moves"][2])]


# ------------------------------------------------------------------------------ the app


def _entry(n: int = 6) -> BookEntry:
    sel = tuple(all_selections(n, 4))
    mine = np.zeros(len(sel))
    mine[:2] = (0.75, 0.25)
    theirs = np.zeros(len(sel))
    theirs[2] = 1.0
    return BookEntry(key="k", player="p", place=0, selections=sel, our_strategy=mine,
                     our_ev_loss=np.zeros(len(sel)), class_weights=np.ones(1), class_sets=((),),
                     their_strategies=(theirs,), their_ev_loss=(np.zeros(len(sel)),), value=0.6,
                     seconds=0.1, model="m", notes=("n",))


def test_the_app_walks_paste_species_override_and_read(world) -> None:  # noqa: ANN001
    reg = world.reg
    calls: list[str] = []

    def solver(mine, opp, report):  # noqa: ANN001, ANN202
        report({"seconds": 1.0})
        calls.append(opp.id)
        time.sleep(0.2)
        return _entry(), "stub"

    learned = LearnedIds(species=frozenset({world.a}), moves=frozenset(MOVES[:4]), items=frozenset(ITEMS),
                         abilities=frozenset(to_id(a) for s in reg.species.values() for a in s.abilities))
    app = RankedApp(reg, world.prior, learned, None, solver, seconds=1.0, model="stub")
    assert app.solvable() == "自分の構築を読み込んでください"
    six = _species(reg, 6)
    mine = app.set_mine(_paste(reg, six))
    assert mine["ok"] and len(mine["team"]) == 6
    assert app.set_mine(_paste(reg, six, nature=False))["ok"] is False and app.mine is None
    app.set_mine(_paste(reg, six))
    names = [reg.species[s].name for s in (world.a, world.b, world.c, world.d)]
    state = app.set_opponent(names + [reg.species[six[4]].name, reg.species[six[5]].name])
    kinds = [o["kind"] for o in state["opponent"]]
    # C's commonest item is A's, so C was moved to its other candidate; D is not in the field.
    assert kinds[:4] == ["whole", "part", "candidate", "manual"]
    assert app.solvable() is not None
    with pytest.raises(RankedError, match="わかりません"):
        app.set_opponent(["notaspecies"] * 6)
    # An unlearned id on the screen: B and D were never in the leaf's data.
    assert [u["id"] for u in state["opponent"][1]["unlearned"]][:1] == [world.b]
    # Fill the by-hand one, and the species the field lacks (E, F) the same.
    for i in (3, 4, 5):
        if state["opponent"][i]["kind"] == "manual":
            app.override(i, {"moves": MOVES[:4], "item": ITEMS[5 + i]})
    state = app.state()
    assert app.solvable() is None, app.team_problems
    started = app.start()
    assert started["state"] == "running"
    with pytest.raises(RankedError, match="最中"):
        app.start()
    for _ in range(100):
        job = app.job()
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done" and calls == ["ranked-opponent"]
    result = job["result"]
    assert result["value"] == 0.6 and result["mine"]["selections"][0]["p"] == 0.75
    assert len(result["estimated"]) == 6 and result["estimated"][1]["unlearned"]
    # What was read no longer belongs to the six once one of them changes.
    app.choose(0, 1)
    assert app.job() == {"state": "idle"}


def test_the_server_answers_json_and_refuses_bad_input(world) -> None:  # noqa: ANN001
    app = RankedApp(world.reg, world.prior, None, None, lambda m, o, r: (_entry(), "stub"), seconds=1.0)
    server = RankedServer(app, "127.0.0.1", 0).start()
    try:
        def call(path, body=None):  # noqa: ANN001, ANN202
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(server.url.rstrip("/") + path, data=data)
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as err:
                return err.code, json.loads(err.read())

        code, meta = call("/api/meta")
        assert code == 200 and meta["spLimit"] == world.reg.meta.sp_limit and "Test Open" in meta["event"]
        assert call("/api/solve", {})[0] == 400  # nothing loaded yet
        assert call("/api/override", {"index": 0})[0] == 400
        with urllib.request.urlopen(server.url) as page:
            html = page.read().decode("utf-8")
        assert "ranked.js" in html and 'name="sprite-url"' in html
        with urllib.request.urlopen(server.url + "ranked.js") as js:
            assert b"/api/opponent" in js.read()
    finally:
        server.close()
