"""The typed-in position of the ranked screen (IKA-408).

What has to hold, each against the failure it prevents:

- **a form is a position the analysis can read.** The people's actives come first on each side, the
  opponent's HP per cent is one HP inside the band the display rule allows (the ends are
  reachable for measuring), unseen members of the opponent's four are not "shown", and an
  invalid form is refused in words rather than built.
- **the next turn is the last one, copied.** The turn number rises, the events are cleared, timed
  things lose a turn, and nothing else moves.
- **what was seen narrows the opponent's set, what was not seen does not.** A move seen removes the
  field's sets without it; seeing nothing changes nothing (the comparison can fail: a seen move
  does change it); a person's own choice survives an observation it is consistent with; when no
  field set matches, the seen things are put in and the card says so.
- **who moved first and the damage taken narrow the opponent's spread, and the truth survives.** The
  attacker's true spread is among the values left, the range is narrower than the whole, a
  different truth leaves a different range, and an order or a damage no spread explains is
  reported and applied to nothing.
- **the session walks form to read.** Start, save, next turn, read: the read is the analysis mode's
  (`Reader`), its answer is summarised per Pokemon, and a changed input drops a finished read.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou import rankedposition as rp
from pokeuraou.analysis import Result
from pokeuraou.observe import _rolls
from pokeuraou.priors import SampledSet
from pokeuraou.rankedentry import FieldPrior, Observation, RankedError
from pokeuraou.rankedweb import RankedApp
from pokeuraou.regulation import to_id
from pokeuraou.standings import Standings, TeamMember, TournamentTeam
from pokeuraou.teams import Roster

MINE = [
    ("garchomp", "Jolly", {"hp": 2, "atk": 32, "spe": 32},
     ["earthquake", "dragonclaw", "protect", "rockslide"], "yacheberry"),
    ("incineroar", "Careful", {"hp": 32, "def": 20, "spd": 14},
     ["fakeout", "flareblitz", "knockoff", "partingshot"], "sitrusberry"),
    ("rillaboom", "Adamant", {"hp": 32, "atk": 32, "spe": 2},
     ["grassyglide", "woodhammer", "fakeout", "uturn"], "assaultvest"),
    ("milotic", "Bold", {"hp": 32, "def": 32, "spd": 2}, ["scald", "icebeam", "recover", "protect"],
     "leftovers"),
    ("sneasler", "Jolly", {"hp": 2, "atk": 32, "spe": 32}, ["closecombat", "direclaw", "protect", "fakeout"],
     "focussash"),
    ("gholdengo", "Modest", {"hp": 32, "spa": 32, "spe": 2}, ["makeitrain", "shadowball", "protect", "trick"],
     "choicespecs"),
]
OPP = ["tyranitar", "salamence", "talonflame", "ninetales", "kingambit", "sylveon"]
OTHER_ITEMS = ["sitrusberry", "focussash", "lumberry", "scopelens", "mentalherb"]
TYRANITAR = [
    ("leftovers", "Careful", ["stoneedge", "knockoff", "protect", "taunt"], 6),
    ("lifeorb", "Adamant", ["rockslide", "crunch", "protect", "firepunch"], 4),
    ("choicescarf", "Adamant", ["rockslide", "crunch", "protect", "icywind"], 2),
]


def _member(reg, species: str, item: str | None, moves: list[str], nature: str = "Adamant") -> TeamMember:  # noqa: ANN001
    return TeamMember(species=species, ability=to_id(reg.species[species].abilities[0]), item=item,
                      nature=nature, moves=tuple(moves))


@pytest.fixture(scope="module")
def world(reg):  # noqa: ANN001, ANN201
    mine = Roster(
        id="mine", name="自分", reg=reg, shown_stats=[None] * 6, source={"kind": "test"},
        sets=[SampledSet(species=s, ability=to_id(reg.species[s].abilities[0]), item=item, nature=n, sp=sp,
                         moves=mv) for s, n, sp, mv, item in MINE])
    mine.sets[1].ability = "intimidate"
    teams = []
    for item, nature, moves, count in TYRANITAR:
        for _ in range(count):
            team = [_member(reg, "tyranitar", item, moves, nature)]
            team += [_member(reg, s, item, ["protect", "tailwind", "flamethrower", "icywind"])
                     for s, item in zip(OPP[1:], OTHER_ITEMS, strict=True)]
            teams.append(team)
    standings = Standings(
        event="Test Open", event_format="M-C", player_count=len(teams),
        teams=[TournamentTeam(player=f"p{i}", place=i, country="", wins=0, losses=0, made_cut=False,
                              phase_two=False, members=tuple(t)) for i, t in enumerate(teams)])
    prior = FieldPrior(reg, standings, None)
    base = {s: prior.fill(s, OPP) for s in OPP}
    return SimpleNamespace(reg=reg, mine=mine, prior=prior, base=base)


def _board(world, turns: int = 1, brought=(0, 1, 2, 3)) -> rp.Board:  # noqa: ANN001
    board = rp.Board(world.reg, world.mine, list(brought), list(OPP), OPP[:2], [])
    board.check()
    sets = {s: o.set for s, o in world.base.items()}
    board.turns = [rp.normalize(world.reg, board, rp.blank_form(world.reg, board))]
    for _ in range(turns - 1):
        board.turns.append(rp.normalize(world.reg, board, rp.next_form(board.turns[-1])))
    del sets
    return board


def _edit(board: rp.Board, t: int, fn) -> None:  # noqa: ANN001
    form = rp.normalize(board.reg, board, board.turns[t])
    fn(form)
    board.turns[t] = rp.normalize(board.reg, board, form)


# ------------------------------------------------------------------------------ the position


def test_a_form_builds_a_position_with_the_actives_first_and_the_foe_hp_in_its_band(world) -> None:  # noqa: ANN001
    reg = world.reg
    board = _board(world)

    def edit(form):  # noqa: ANN001, ANN202
        form["mineActive"] = [2, 0]                  # slots 0 and 1 are Rillaboom, Garchomp
        form["theirs"]["tyranitar"]["pct"] = 62
        form["mine"]["0"]["hp"] = 100
    _edit(board, 0, edit)
    sets = {s: o.set for s, o in world.base.items()}
    built = rp.build_position(reg, board, board.turns[0], sets)
    pos = built.position
    assert built.problems == []
    assert pos.sides[0].active == [0, 1]
    assert [pos.sides[0].pokemon[i].species for i in (0, 1)] == ["rillaboom", "garchomp"]
    assert pos.sides[0].pokemon[1].hp == 100 and built.mine_slots == {2: 0, 0: 1, 1: 2, 3: 3}
    # The opponent: the seen two first, then two unseen ones from its six -- none of them "shown".
    assert [m.species for m in pos.sides[1].pokemon[:2]] == ["tyranitar", "salamence"]
    assert len(pos.sides[1].pokemon) == 4 and built.opp_slots["tyranitar"] == 0
    ty = pos.sides[1].pokemon[0]
    low = rp.build_position(reg, board, board.turns[0], sets, hp_mode="low").position.sides[1].pokemon[0].hp
    high = rp.build_position(reg, board, board.turns[0], sets, hp_mode="high").position.sides[1].pokemon[0].hp
    assert 100 * low // ty.maxhp == 62 and 100 * high // ty.maxhp == 62     # the floor rule's band
    assert low <= ty.hp <= high and (low, high) != (ty.hp, ty.hp) or low == high   # mid is inside
    assert not any(m.hp != m.maxhp or m.active_index is not None for m in pos.sides[1].pokemon[2:])


def test_an_invalid_form_is_refused_in_words(world) -> None:  # noqa: ANN001
    reg = world.reg
    board = _board(world)
    good = board.turns[0]

    def bad(edit, match):  # noqa: ANN001, ANN202
        form = rp.normalize(reg, board, good)
        edit(form)
        with pytest.raises(RankedError, match=match):
            rp.normalize(reg, board, form)

    bad(lambda f: f["mine"]["0"].update(hp=9999), "HP")
    bad(lambda f: f["mine"]["0"].update(hp=0), "倒れている")
    bad(lambda f: f["mine"]["0"]["boosts"].update(atk=7), "能力変化")
    bad(lambda f: f["theirs"]["tyranitar"].update(pct=0), "HP")
    bad(lambda f: f["theirs"]["tyranitar"].update(status="zzz"), "状態異常")
    bad(lambda f: f["theirs"]["tyranitar"].update(itemGone=True, item=None), "持ち物")
    bad(lambda f: f["theirs"]["tyranitar"].update(
        moves=["rockslide", "crunch", "protect", "icywind", "tailwind"]), "見えた技")
    bad(lambda f: f["mineActive"].__setitem__(1, 0), "2 回")
    bad(lambda f: f["field"].update(weather="rainbow"), "天気")
    bad(lambda f: f["events"]["damage"].append(
        {"attacker": "tyranitar", "move": "crunch", "target": 5, "amount": 10}), "4 体の中")
    # Someone outside the opponent's six cannot be on the field; one of the six who has not been
    # seen yet is, by standing there, seen.
    with pytest.raises(RankedError, match="6 種族"):
        f = rp.normalize(reg, board, good)
        f["theirActive"][0] = "pikachu"
        rp.normalize(reg, board, f)
    f = rp.normalize(reg, board, good)
    f["theirActive"][0] = "kingambit"
    assert "kingambit" in rp.normalize(reg, board, f)["theirs"]


def test_the_next_turn_is_the_last_one_copied_with_the_events_cleared(world) -> None:  # noqa: ANN001
    board = _board(world)

    def edit(form):  # noqa: ANN001, ANN202
        form["mine"]["1"]["hp"] = 77
        form["field"].update(weather="raindance", weatherTurns=4, trickRoom=3)
        form["sides"][0] = {"reflect": 3, "stealthrock": 1}
        form["events"]["order"].append({"first": "m:0", "second": "t:tyranitar"})
    _edit(board, 0, edit)
    nxt = rp.normalize(world.reg, board, rp.next_form(board.turns[0]))
    assert nxt["turn"] == 2 and nxt["events"] == {"order": [], "damage": []}
    assert nxt["mine"]["1"]["hp"] == 77
    assert (nxt["field"]["weatherTurns"], nxt["field"]["trickRoom"], nxt["sides"][0]["reflect"]) == (3, 2, 2)
    assert nxt["sides"][0]["stealthrock"] == 1            # a hazard has no turns to lose
    # The copy is a copy: editing it leaves the turn before alone.
    nxt["mine"]["1"]["hp"] = 1
    assert board.turns[0]["mine"]["1"]["hp"] == 77


def test_the_first_turn_has_the_leads_switch_in_abilities_put_in(world, port) -> None:  # noqa: ANN001
    reg = world.reg
    board = _board(world, brought=(1, 0, 2, 3))      # Incineroar leads: Intimidate
    sets = {s: o.set for s, o in world.base.items()}
    form = rp.initial_form(reg, board, sets)
    assert form["theirs"]["tyranitar"]["boosts"] == {"atk": -1}
    assert form["theirs"]["salamence"]["boosts"] == {"atk": -1}
    assert form["field"]["weather"] == "sandstorm"     # Tyranitar's Sand Stream
    # The comparison can fail: Incineroar not leading, nothing of the opponent's is lowered (its own
    # Salamence still takes the person's Garchomp down).
    calm = rp.initial_form(reg, _board(world, brought=(0, 2, 1, 3)), sets)
    assert calm["theirs"]["tyranitar"]["boosts"] == {} and calm["theirs"]["salamence"]["boosts"] == {}
    assert calm["mine"]["0"]["boosts"] == {"atk": -1}


# ------------------------------------------------------------------------------ the opponent's set


def test_a_seen_move_narrows_the_field_sets_and_nothing_seen_changes_nothing(world) -> None:  # noqa: ANN001
    base = world.base["tyranitar"]
    assert base.set.item == "leftovers" and base.belief is not None and len(base.belief.members) == 12
    # Nothing seen: nothing changes -- the estimate is the very same object.
    quiet = rp.refine_opponent(world.prior, base, Observation())
    assert quiet.one is base and quiet.members is None
    # A move only the Life Orb set has: the estimate becomes that set.
    seen = rp.refine_opponent(world.prior, base, Observation(moves=frozenset({"firepunch"})))
    assert seen.one.set.item == "lifeorb" and "firepunch" in seen.one.set.moves
    assert seen.members == (12, 4) and not seen.unmatched
    # A move all three share removes none of them: seeing is not the same as ruling out.
    shared = rp.refine_opponent(world.prior, base, Observation(moves=frozenset({"protect"})))
    assert shared.members == (12, 12) and shared.one.set.item == "leftovers"
    # An item seen (spent or shown) selects by item too.
    item = rp.refine_opponent(world.prior, base, Observation(item="choicescarf"))
    assert item.one.set.item == "choicescarf" and item.members == (12, 2)


def test_a_chosen_candidate_survives_a_consistent_observation_and_none_matching_is_said(world) -> None:  # noqa: ANN001
    base = world.base["tyranitar"]
    chosen = world.prior.choose(base, 1)                        # the person picked the Life Orb set
    kept = rp.refine_opponent(world.prior, chosen, Observation(moves=frozenset({"protect"})))
    assert kept.one.set.item == "lifeorb"
    moved = rp.refine_opponent(world.prior, chosen, Observation(moves=frozenset({"taunt"})))
    assert moved.one.set.item == "leftovers" and "taunt" in moved.one.set.moves   # inconsistent: re-read
    none = rp.refine_opponent(world.prior, base, Observation(moves=frozenset({"surf"}), ability="sandstream"))
    assert none.unmatched and "surf" in none.one.set.moves and "大会データに無い" in none.notes[0]
    # The seen move displaced the commonest others, not the other way round.
    assert len(none.one.set.moves) == 4


# ------------------------------------------------------------------------------ the spread


def _two_turns(world, mine_active=(0, 1)) -> rp.Board:  # noqa: ANN001
    board = _board(world, 2)
    _edit(board, 0, lambda f: f.update(mineActive=list(mine_active)))
    board.turns[1] = rp.normalize(world.reg, board, rp.next_form(board.turns[0]))
    return board


def _derive(world, board, **kw):  # noqa: ANN001, ANN202
    return rp.derive(world.reg, board, world.prior, world.base, **kw)


def _speed_note(world, order: dict, mine_active=(2, 0)):  # noqa: ANN001, ANN202
    board = _two_turns(world, mine_active=mine_active)
    board.turns[1]["events"]["order"].append({"firstMove": None, "secondMove": None, **order})
    derived = _derive(world, board)
    return derived, [n for n in derived.spread_notes if n.species == "tyranitar" and n.stat == "spe"]


def test_who_moved_first_narrows_the_speed_spread_and_moves_the_estimate(world) -> None:  # noqa: ANN001
    reg = world.reg
    top = reg.meta.sp_per_stat_max
    # Rillaboom (Adamant, 2 Spe SP: 107). Tyranitar (base 61, neutral Speed) makes 107 with 26 SP, so
    # "Rillaboom moved first" leaves 0..26 (26 is the tie), "Tyranitar moved first" leaves 26..32.
    derived, slow = _speed_note(world, {"first": "m:2", "second": "t:tyranitar"})
    assert len(slow) == 1 and (slow[0].low, slow[0].high) == (0, 26) and slow[0].stat_high == 107
    # The estimate (a neutral spread with 32 Speed) was moved to the nearest value still possible.
    assert (slow[0].before, slow[0].after) == (top, 26) and derived.sets["tyranitar"].sp["spe"] == 26
    # The comparison can fail: the other order leaves the other end, and the estimate stays.
    derived, fast = _speed_note(world, {"first": "t:tyranitar", "second": "m:2"})
    assert len(fast) == 1 and (fast[0].low, fast[0].high) == (26, top) and fast[0].after == top
    assert derived.sets["tyranitar"].sp["spe"] == top
    # No event: nothing is narrowed and the estimate is untouched.
    quiet = _derive(world, _two_turns(world))
    assert quiet.spread_notes == [] and quiet.sets["tyranitar"].sp == world.base["tyranitar"].set.sp


def test_a_priority_move_says_nothing_about_speed_and_a_contradiction_is_reported(world) -> None:  # noqa: ANN001
    # Fake Out against Rock Slide: the order is not a Speed fact.
    derived, notes = _speed_note(world, {"first": "m:1", "second": "t:tyranitar", "firstMove": "fakeout",
                                         "secondMove": "rockslide"}, mine_active=(1, 0))
    assert notes == [] and any("手がかりになりません" in line for line in derived.lines)
    # Garchomp (169, or more with +6) is said to have moved second, against a Tyranitar that cannot
    # reach 169: no spread explains it, so nothing is applied and it is said.
    derived, notes = _speed_note(world, {"first": "t:tyranitar", "second": "m:0"}, mine_active=(0, 1))
    assert notes == [] and any("どの配分でも起こりません" in line for line in derived.lines)


def test_the_damage_taken_narrows_the_attack_spread_and_the_truth_survives(world) -> None:  # noqa: ANN001
    reg = world.reg
    sets = {s: o.set for s, o in world.base.items()}
    truth = 20

    def amount_for(attack_sp: int) -> int:
        board = _board(world)
        one = sets["tyranitar"]
        grid = rp._grid(reg, one, ["atk"], {"atk": [attack_sp]})
        built = rp.build_position(reg, board, board.turns[0], sets)
        from pokeuraou.belief import battler_for
        from pokeuraou.view import battler
        attacker = battler_for(reg, built.position, (1, 0), grid)
        defender = battler(reg, built.position.sides[0].pokemon[built.mine_slots[0]])
        rolls = _rolls(reg, built.position, attacker, defender, "crunch", defender_side=0, crit=False,
                       spread=False)
        return int(rolls[0][5])

    amount = amount_for(truth)
    assert amount != amount_for(0), "the attack spread has to move the damage for this to mean anything"
    board = _two_turns(world)
    board.turns[1]["events"]["damage"].append({"attacker": "tyranitar", "move": "crunch", "target": 0,
                                               "amount": amount, "crit": False})
    notes = [n for n in _derive(world, board).spread_notes if n.stat == "atk"]
    assert len(notes) == 1 and notes[0].species == "tyranitar"
    assert notes[0].low <= truth <= notes[0].high and notes[0].high - notes[0].low < reg.meta.sp_per_stat_max
    # A different true spread leaves a different range -- the comparison can fail.
    other = _two_turns(world)
    other.turns[1]["events"]["damage"].append({"attacker": "tyranitar", "move": "crunch", "target": 0,
                                               "amount": amount_for(0), "crit": False})
    far = [n for n in _derive(world, other).spread_notes if n.stat == "atk"]
    assert len(far) == 1 and (far[0].low, far[0].high) != (notes[0].low, notes[0].high)
    assert far[0].low == 0
    # A damage that no Attack can make (1 more than the most) is reported, not applied.
    silly = _two_turns(world)
    silly.turns[1]["events"]["damage"].append({"attacker": "tyranitar", "move": "crunch", "target": 0,
                                               "amount": 400, "crit": False})
    derived = _derive(world, silly)
    assert [n for n in derived.spread_notes if n.stat == "atk"] == []
    assert any("どの配分でも起こりません" in line for line in derived.lines)


def test_the_total_stays_within_the_limit_after_a_spread_is_moved(world) -> None:  # noqa: ANN001
    reg = world.reg
    board = _two_turns(world)
    board.turns[1]["events"]["order"].append({"first": "t:tyranitar", "second": "m:0", "firstMove": None,
                                              "secondMove": None})
    sp = _derive(world, board).sets["tyranitar"].sp
    assert sum(sp.values()) <= reg.meta.sp_limit and max(sp.values()) <= reg.meta.sp_per_stat_max


# ------------------------------------------------------------------------------ the session


def _result(point, value=0.57) -> Result:
    """A read's answer over the first two legal actions of each side at ``point``."""
    from pokeuraou.actions import side_actions
    from pokeuraou.position import Position

    pos = Position.from_json(point.position)
    reg = pos.sides  # noqa: F841 - the regulation is the module's
    ours = [a.to_choice() for a in side_actions(_REG[0], pos, 0)[:2]]
    theirs = [a.to_choice() for a in side_actions(_REG[0], pos, 1)[:1]]
    return Result(
        side=0, turn=point.turn, ours=ours, strategy=np.array([0.75, 0.25]), theirs=theirs, model=[1.0],
        value0=value, value=value, steps=3, seconds=0.2, stop="time", guard=16, guard_lines=0, nodes=4,
        exact=False, classes=1, deepened=None)


_REG: list = []


def _app(world, reader):  # noqa: ANN001, ANN202
    app = RankedApp(world.reg, world.prior, None, None, lambda m, o, r: (None, "stub"), seconds=1.0,
                    reader=reader, read_seconds=0.5)
    app.mine = world.mine
    app.opponent = [world.base[s] for s in OPP]
    app._recheck()
    return app


def test_the_session_walks_start_save_next_and_read(world) -> None:  # noqa: ANN001
    seen: dict = {}

    def reader(game, point, seconds, report):  # noqa: ANN001, ANN202
        report({"seconds": seconds})
        seen.update(label=game.label, seconds=seconds, turn=point.turn, teams=[t.id for t in game.teams],
                    leads=game.leads, shown=point.seen)
        time.sleep(0.1)
        return _result(point)

    _REG[:] = [world.reg]
    app = _app(world, reader)
    board = app.board
    assert board.state()["started"] is False
    with pytest.raises(RankedError, match="4 体"):
        board.start([0, 1, 2], OPP[:2], [])
    with pytest.raises(RankedError, match="先発"):
        board.start([0, 1, 2, 3], ["tyranitar", "tyranitar"], [])
    state = board.start([0, 1, 2, 3], OPP[:2], [])
    assert state["started"] and len(state["turns"]) == 1 and state["turns"][0]["readable"]
    # Save an edit: the foe's HP, and a move seen; the estimate answers with the narrowed set.
    form = state["turns"][0]["form"]
    form["theirs"]["tyranitar"].update(pct=40, moves=["firepunch"])
    saved = board.save(0, form)
    assert saved["opp"]["tyranitar"]["item"]["id"] == "lifeorb"
    assert saved["opp"]["tyranitar"]["refine"]["members"] == (12, 4)
    assert saved["turns"][0]["oppHp"]["tyranitar"]["max"] > 0
    nxt = board.next_turn()
    assert len(nxt["turns"]) == 2 and nxt["turns"][1]["form"]["turn"] == 2
    assert nxt["turns"][1]["form"]["theirs"]["tyranitar"]["moves"] == ["firepunch"]   # carried over
    with pytest.raises(RankedError, match="ターン"):
        board.start_read(5)
    started = board.start_read(1)
    assert started["state"] == "running"
    with pytest.raises(RankedError, match="最中"):
        board.start_read(0)
    for _ in range(100):
        job = board.job()
        if job["state"] != "running":
            break
        time.sleep(0.05)
    assert job["state"] == "done" and job["turn"] == 1
    assert seen["turn"] == 2 and seen["seconds"] == 0.5 and seen["teams"][0] == "mine"
    assert seen["leads"] == (frozenset({"garchomp", "incineroar"}), frozenset({"tyranitar", "salamence"}))
    res = job["result"]
    assert res["value"] == 0.57 and len(res["ours"]["slots"]) == 2
    for slot in res["ours"]["slots"]:
        assert sum(a["p"] for a in slot["actions"]) == pytest.approx(1.0)      # a marginal of the mixture
    assert res["ours"]["joint"][0]["p"] == 0.75 and res["ours"]["joint"][1]["p"] == 0.25
    assert res["ours"]["slots"][0]["name"] == "garchomp" and len(res["theirs"]["slots"]) >= 1
    # An edit drops a finished read.
    form = board.state()["turns"][1]["form"]
    form["mine"]["0"]["hp"] = 50
    assert board.save(1, form)["read"] == {"state": "idle"}


def test_a_read_that_fails_is_shown_and_the_worker_lives_on(world) -> None:  # noqa: ANN001
    def reader(game, point, seconds, report):  # noqa: ANN001, ANN202, ARG001
        raise RuntimeError("no leaf")

    app = _app(world, reader)
    app.board.start([0, 1, 2, 3], OPP[:2], [])
    app.board.start_read(0)
    for _ in range(100):
        job = app.board.job()
        if job["state"] != "running":
            break
        time.sleep(0.05)
    assert job["state"] == "error" and "no leaf" in job["error"]
    app.board.start_read(0)                      # the worker is still there to take another
    time.sleep(0.3)
    assert app.board.job()["state"] == "error"
    no_reader = _app(world, None)
    no_reader.board.start([0, 1, 2, 3], OPP[:2], [])
    with pytest.raises(RankedError, match="読みの設定"):
        no_reader.board.start_read(0)


def test_a_pick_among_the_candidates_left_is_kept_until_what_is_seen_rules_it_out(world) -> None:  # noqa: ANN001
    app = _app(world, None)
    board = app.board
    state = board.start([0, 1, 2, 3], OPP[:2], [])
    assert state["opp"]["tyranitar"]["item"]["id"] == "leftovers"
    picked = board.choose("tyranitar", 2)              # the third of the field's sets: Choice Scarf
    got = picked["opp"]["tyranitar"]
    assert got["item"]["id"] == "choicescarf" and got["chosen"] == 2
    form = picked["turns"][0]["form"]
    form["theirs"]["tyranitar"]["moves"] = ["crunch"]  # both the Life Orb and the Scarf set have it
    kept = board.save(0, form)
    assert kept["opp"]["tyranitar"]["item"]["id"] == "choicescarf"
    form["theirs"]["tyranitar"]["moves"] = ["crunch", "firepunch"]   # only the Life Orb set has this
    moved = board.save(0, form)
    assert moved["opp"]["tyranitar"]["item"]["id"] == "lifeorb"
    with pytest.raises(RankedError, match="候補"):
        board.choose("tyranitar", 9)


def test_a_reads_notes_are_plain_and_mega_stones_have_japanese_names(world) -> None:  # noqa: ANN001
    from types import SimpleNamespace as NS

    from pokeuraou.rankedboard import plain_notes

    notes = plain_notes(["選択的延長のセルで、確率の高い分岐 3 つだけを深さ 2 で読んだ",
                         "選択的延長のセルで、確率の高い分岐 3 つだけを深さ 2 で読んだ", "別の注記"])
    assert notes == ["一部の局面は、確率の高い 3 通りの手だけを先まで読みました", "別の注記"]
    # A Mega Stone the names table lacks is named by its Pokemon, not shown in English.
    from pokeuraou.names import localiser

    loc = localiser(world.reg, "ja")
    app = RankedApp(world.reg, world.prior, None, loc, lambda m, o, r: (None, "stub"), seconds=1.0)
    assert app._name("item", "tyranitarite") == "バンギラスのメガストーン"
    assert app._name("item", "charizarditey") == "リザードンのメガストーン（Y）"
    assert app._name("item", "leftovers") == loc.item("leftovers")      # a plain item is the table's own
    del NS


def test_a_pokemon_that_stayed_on_the_field_cannot_fake_out_again(world) -> None:  # noqa: ANN001
    from pokeuraou.actions import MoveAction, side_actions

    reg = world.reg
    sets = {s: o.set for s, o in world.base.items()}

    def fakeout_for_slot0(board, t):  # noqa: ANN001, ANN202
        built = rp.build_position(reg, board, board.turns[t], sets, prev=board.turns[t - 1] if t else None)
        return any(isinstance(a.slots[0], MoveAction) and a.slots[0].move_id == "fakeout"
                   for a in side_actions(reg, built.position, 0))

    stayed = _two_turns(world, mine_active=(1, 0))           # Incineroar (Fake Out) leads and stays
    assert fakeout_for_slot0(stayed, 0) and not fakeout_for_slot0(stayed, 1)
    # The comparison can fail: Rillaboom (also Fake Out) coming in on turn 2 can use it.
    _edit(stayed, 1, lambda f: f.update(mineActive=[2, 0]))
    assert fakeout_for_slot0(stayed, 1)


def test_the_server_routes_walk_the_board_and_serve_the_page(world) -> None:  # noqa: ANN001
    import json
    import urllib.error
    import urllib.request

    from pokeuraou.rankedweb import RankedServer

    def reader(game, point, seconds, report):  # noqa: ANN001, ANN202
        return _result(point)

    _REG[:] = [world.reg]
    app = _app(world, reader)
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

        assert call("/api/board") == (200, {"started": False, "read": {"state": "idle"}})
        assert call("/api/board/start", {"brought": [0, 1, 2], "leads": OPP[:2]})[0] == 400
        code, state = call("/api/board/start", {"brought": [0, 1, 2, 3], "leads": OPP[:2]})
        assert code == 200 and state["started"] and state["options"]["status"]
        form = state["turns"][0]["form"]
        form["theirs"]["tyranitar"]["pct"] = 77
        assert call("/api/board/save", {"index": 0, "form": form})[0] == 200
        form["theirs"]["tyranitar"]["pct"] = 0
        code, bad = call("/api/board/save", {"index": 0, "form": form})
        assert code == 400 and "HP" in bad["error"]
        assert call("/api/board/next", {})[1]["turns"][1]["form"]["theirs"]["tyranitar"]["pct"] == 77
        assert call("/api/board/choose", {"species": "tyranitar", "alternative": 1})[0] == 200
        assert call("/api/board/read", {"index": 1})[1]["state"] == "running"
        for _ in range(100):
            job = call("/api/board/job")[1]
            if job["state"] != "running":
                break
            time.sleep(0.05)
        assert job["state"] == "done" and job["result"]["value"] == 0.57
        assert call("/api/board/drop", {})[1]["turns"].__len__() == 1
        assert call("/api/board/reset", {})[1]["started"] is False
        for page, marker in (("position", b"ranked-position.js"), ("ranked-position.js", b"/api/board/save"),
                             ("ranked-position.css", b".ps-mon"), ("ranked.html", b"ranked-position.html")):
            with urllib.request.urlopen(f"{server.url}{page}") as resp:
                assert marker in resp.read()
        with urllib.request.urlopen(server.url + "position") as resp:
            assert b'name="sprite-url"' in resp.read()
    finally:
        server.close()


def test_what_a_pokemon_carries_over_turns_is_in_the_position(world) -> None:  # noqa: ANN001
    reg = world.reg
    sets = {s: o.set for s, o in world.base.items()}
    board = _two_turns(world, mine_active=(0, 1))

    def edit(form):  # noqa: ANN001, ANN202
        form["mine"]["0"]["protect"] = 2                   # Garchomp protected twice in a row
        form["theirs"]["tyranitar"]["protect"] = 1
        form["theirs"]["tyranitar"]["item"] = "choicescarf"
        form["theirs"]["tyranitar"]["locked"] = "crunch"  # locked into a move it used: so a seen move
    _edit(board, 0, edit)
    built = rp.build_position(reg, board, board.turns[0], sets).position
    chomp = built.sides[0].pokemon[0]
    assert [(e.id, e.duration, e.counter) for e in chomp.volatiles] == [("stall", 1, 9)]
    ty = built.sides[1].pokemon[0]
    assert {(e.id, e.counter, e.move) for e in ty.volatiles} == {
        ("stall", 3, None), ("choicelock", None, "crunch")}
    assert "crunch" in board.turns[0]["theirs"]["tyranitar"]["moves"]
    # Unburden follows the ability and the spent item. The control: the same Pokemon with its item
    # still in hand has none.
    _edit(board, 0, lambda f: f["mine"]["1"].update(itemGone=True))
    sets["tyranitar"] = world.base["tyranitar"].set
    mine_sets = world.mine.sets
    old = mine_sets[1].ability
    mine_sets[1].ability = "unburden"
    try:
        spent = rp.build_position(reg, board, board.turns[0], sets).position.sides[0].pokemon[1]
        assert [e.id for e in spent.volatiles] == ["unburden"] and spent.item is None
        _edit(board, 0, lambda f: f["mine"]["1"].update(itemGone=False))
        held = rp.build_position(reg, board, board.turns[0], sets).position.sides[0].pokemon[1]
        assert held.volatiles == []
    finally:
        mine_sets[1].ability = old
    # The next turn forgets last turn's Protect (it is typed again) and keeps the lock.
    nxt = rp.next_form(board.turns[0])
    assert nxt["mine"]["0"]["protect"] == 0 and nxt["theirs"]["tyranitar"]["locked"] == "crunch"
    # A move a Pokemon of the person's is locked into must be one of its moves.
    bad = rp.normalize(reg, board, board.turns[0])
    bad["mine"]["0"]["locked"] = "surf"
    with pytest.raises(RankedError, match="持っていません"):
        rp.normalize(reg, board, bad)


def test_seeing_a_pokemon_mega_evolve_shows_its_item(world) -> None:  # noqa: ANN001
    reg = world.reg
    board = _board(world)
    _edit(board, 0, lambda f: f["theirs"]["tyranitar"].update(mega=True))
    derived = rp.derive(reg, board, world.prior, world.base, observe_spread=False)
    assert derived.sets["tyranitar"].item == "tyranitarite"
    mon = derived.built[0].position.sides[1].pokemon[0]
    assert mon.is_mega and mon.species == "tyranitarmega" and mon.item == "tyranitarite"
    # The comparison can fail: not Mega Evolved, nothing is said of the item.
    plain = rp.derive(reg, _board(world), world.prior, world.base, observe_spread=False)
    assert plain.sets["tyranitar"].item == "leftovers"
    # A species with two stones takes the one the field holds most (Charizard: X and Y).
    members = [SimpleNamespace(item="charizarditey")] * 3 + [SimpleNamespace(item="charizarditex")]
    assert rp.mega_stone(reg, "charizard", members) == "charizarditey"
    assert rp.mega_stone(reg, "garchomp", []) != "" and rp.mega_stone(reg, "incineroar", []) is None


def test_an_ability_the_leads_copied_is_the_one_in_effect(world, port) -> None:  # noqa: ANN001
    reg = world.reg
    sets = {s: o.set for s, o in world.base.items()}
    came_with = world.mine.sets[0].ability
    world.mine.sets[0].ability = "trace"             # Garchomp stands in for a Trace user
    try:
        form = rp.initial_form(reg, _board(world), sets)
        # Trace copies one of the opposing leads' abilities (Tyranitar's Sand Stream or Salamence's
        # Intimidate): whichever it copied is the one in effect, and the position says so.
        assert form["mine"]["0"]["abilityNow"] in ("sandstream", "intimidate")
        board = _board(world)
        board.turns = [rp.normalize(reg, board, form)]
        built = rp.build_position(reg, board, board.turns[0], sets).position
        assert built.sides[0].pokemon[0].ability == form["mine"]["0"]["abilityNow"]
        # The comparison can fail: with another ability nothing is copied and nothing is said.
        world.mine.sets[0].ability = came_with
        plain = rp.initial_form(reg, _board(world), sets)
        assert plain["mine"]["0"]["abilityNow"] is None
    finally:
        world.mine.sets[0].ability = came_with
