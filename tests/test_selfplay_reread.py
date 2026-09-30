"""`show_game.py --html --reread`: a generation game's reads, solved again and checked.

`tools/reread.py` solves each move decision of a hidden-bench record again the way `play_game`
solved it and keeps a decision's re-solve only when both seats' mixtures and values are the
record's. These tests play a short hidden-bench game, re-read it, and hold the page to that: the
re-solve matches every decision (with a hidden bench in the belief, the path the claim is about),
the page shows the other side's mixture and the moves' values only where it matched, and a
decision whose record is changed is not matched and not shown (the control that the comparison
can fail).
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import HP_SHARE
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import play_game
from pokeuraou.teams import load_roster

from ._harness import load_tool

AGAIN = "解き直した読み（記録の混合と一致を確かめた）"
NOT_MATCHED = "解き直しが記録と一致しなかった"


@pytest.fixture(scope="module")
def tools():  # noqa: ANN201
    show_game = load_tool("show_game")
    import game_page
    import reread

    return show_game, game_page, reread


@pytest.fixture(scope="module")
def played():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    own = sheet[:4]
    foe = [sheet[i] for i in (1, 0, 4, 5)]
    record = play_game(
        reg, np.random.default_rng(3), own, foe, "test",
        search_limit=3, max_turns=4, sheets=(sheet, sheet),
    )
    payload = json.loads(json.dumps(record.to_json(objective="hp-share", search_limit=3)))
    payload["ownSix"] = payload["foeSix"] = [s.species for s in sheet]
    return reg, payload, (sheet, sheet), rustnode.RustNode(reg)


@pytest.fixture(scope="module")
def reread_of(tools, played):  # noqa: ANN001, ANN201
    _show_game, _game_page, reread = tools
    reg, record, sheets, _node = played
    return reread.reread_game(reg, record, sheets=sheets, evaluate=HP_SHARE.batch)


def test_the_resolve_is_the_records(tools, played, reread_of) -> None:  # noqa: ANN001
    _show_game, _game_page, reread = tools
    _reg, record, _sheets, _node = played
    moves = [i for i, d in enumerate(record["decisions"]) if d["kind"] == "move"]
    assert sorted(reread_of) == moves and len(moves) >= 2
    for i in moves:
        got = reread_of[i]
        assert got.matched, (i, got.reason)
        assert got.worst <= reread.TOLERANCE
        d = record["decisions"][i]
        for side, names, policy in ((0, "ownActions", "ownPolicy"), (1, "foeActions", "foePolicy")):
            read = got.reads[side]
            # The seat's own mixture on the page is the record's (to the fourth decimal).
            want = dict(zip(d[names], d[policy], strict=True))
            for choice, p in read["supp"]:
                assert p == pytest.approx(want[choice], abs=1e-4)
            # What the record lacks and the re-solve adds: the other side's modelled mixture and
            # the drawn move's value against each of the other side's moves.
            assert sum(read["q"]) == pytest.approx(1.0, abs=1e-3)
            assert read["cols"] and len(read["vs"]) == len(read["cols"])
            assert 0.0 <= read["eq"] <= 1.0
            # The seat's own value from the kept matrices is the recorded one (seat 1 in its units).
            value0 = d["searchValue"] if side == 0 else d["foeSearchValue"]
            assert read["eq"] == pytest.approx(value0 if side == 0 else 1.0 - value0, abs=1e-4)
    # The belief path ran: some decision averaged over more than one completion of a bench.
    assert max(reread_of[i].reads[s]["classes"] for i in moves for s in (0, 1)) > 1


def test_the_page_shows_the_reads_that_matched(tools, played, reread_of) -> None:  # noqa: ANN001
    show_game, game_page, _reread = tools
    reg, record, _sheets, node = played
    loc = show_game.Localiser(reg, show_game.load_names("ja"))
    page = game_page.render_html(reg, loc, game_page.selfplay_line(reg, loc, record, node, reread_of))
    plain = game_page.render_html(reg, loc, game_page.selfplay_line(reg, loc, record, node))
    sections = page.split('<section class="turn"')[1:]
    assert sections and all(AGAIN in s and "<h5>相手の読み" in s for s in sections)
    assert "手ごとの値は記録にありません" not in page and NOT_MATCHED not in page
    moves = sum(1 for d in record["decisions"] if d["kind"] == "move")
    assert f"{moves} / {moves} 手番" in page
    # Control: without the re-read the page has none of it.
    assert AGAIN not in plain and "<h5>相手の読み" not in plain


def test_a_decision_that_does_not_match_is_not_shown(tools, played) -> None:  # noqa: ANN001
    show_game, game_page, reread = tools
    reg, record, sheets, node = played
    changed = copy.deepcopy(record)
    moves = [i for i, d in enumerate(changed["decisions"]) if d["kind"] == "move"]
    target = moves[0]
    d = changed["decisions"][target]
    # Move 0.01 of seat 0's mixture from its heaviest move to another: the fourth decimal fails.
    heavy = int(np.argmax(d["ownPolicy"]))
    other = (heavy + 1) % len(d["ownPolicy"])
    d["ownPolicy"][heavy] -= 0.01
    d["ownPolicy"][other] += 0.01
    got = reread.reread_game(reg, changed, sheets=sheets, evaluate=HP_SHARE.batch)
    assert not got[target].matched and got[target].reads == {}
    assert "混合か値が記録と" in got[target].reason
    assert all(got[i].matched for i in moves if i != target)
    loc = show_game.Localiser(reg, show_game.load_names("ja"))
    page = game_page.render_html(reg, loc, game_page.selfplay_line(reg, loc, changed, node, got))
    sections = page.split('<section class="turn"')[1:]
    first = sections[moves.index(target)]
    assert NOT_MATCHED in first and AGAIN not in first and "<h5>相手の読み" not in first
    assert all(AGAIN in s for k, s in enumerate(sections) if moves[k] != target)
    assert f"{len(moves) - 1} / {len(moves)} 手番" in page


def test_settings_it_cannot_resolve_are_refused(tools, played) -> None:  # noqa: ANN001
    _show_game, _game_page, reread = tools
    reg, record, sheets, _node = played
    for key, value in (("information", "open"), ("deepen", ["m64", "m64"]), ("depth", [2, 2])):
        cut = copy.deepcopy(record)
        cut[key] = value
        assert reread.refused(cut)
        with pytest.raises(ValueError):
            reread.reread_game(reg, cut, sheets=sheets, evaluate=HP_SHARE.batch)
    assert reread.refused(record) is None
    # Sheets that are not the record's sixes stop it.
    with pytest.raises(ValueError):
        reread.reread_game(reg, record, sheets=(sheets[0][::-1], sheets[1]), evaluate=HP_SHARE.batch)
