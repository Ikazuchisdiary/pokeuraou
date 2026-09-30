"""`show_game.py --file ... --html`: a generation game as the same page a time match gets.

The page is built from the record by `game_page.selfplay_line`; what happened is the port's
re-resolution (`show_game.turn_trace`), the same the text prints. These tests hold the page to
the text's numbers (win rates, mixtures, the trace) and to its disclosures (what a generation
record does not carry is a dash, not a made-up value), with a control that the comparisons fail
when the record changes.
"""

from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.damage import register_mega_stones
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import play_game
from pokeuraou.teams import load_roster

from ._harness import load_tool


@pytest.fixture(scope="module")
def tools():  # noqa: ANN201
    show_game = load_tool("show_game")
    import game_page

    return show_game, game_page


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
    payload = json.loads(json.dumps(record.to_json(objective="hp-share", search_limit=4)))
    return reg, payload, rustnode.RustNode(reg)


def _sections(page: str) -> list[str]:
    return page.split('<section class="turn"')[1:]


def test_the_page_carries_the_texts_numbers(tools, played) -> None:  # noqa: ANN001
    show_game, game_page = tools
    reg, record, node = played
    loc = show_game.Localiser(reg, show_game.load_names("ja"))
    line = game_page.selfplay_line(reg, loc, record, node)
    page = game_page.render_html(reg, loc, line)
    text = show_game.render(reg, loc, record, 5, node)

    moves = [d for d in record["decisions"] if d["kind"] == "move"]
    sections = _sections(page)
    assert len(moves) >= 2 and len(sections) == len(moves)
    # The text's "探索値（自分の勝率）" of each decision, in order, and the page's seat 0 read.
    text_values = [float(v) for v in re.findall(r"探索値（自分の勝率）: ([0-9.]+)", text)]
    kinds = [d["kind"] for d in record["decisions"]]
    by_kind = [v for v, k in zip(text_values, kinds, strict=True) if k == "move"]
    for d, section, shown in zip(moves, sections, by_kind, strict=True):
        assert shown == pytest.approx(d["searchValue"], abs=5e-4)
        assert f'<b class="n c0">{d["searchValue"] * 100:.0f}%</b>' in section
        assert f'<b class="n c1">{d["foeSearchValue"] * 100:.0f}%</b>' in section
        # Both seats' drawn moves, with the probability the text prints for them.
        for own, key, actions, policy in (
            (0, "ownChosen", "ownActions", "ownPolicy"),
            (1, "foeChosen", "foeActions", "foePolicy"),
        ):
            p = d[policy][d[actions].index(d[key])]
            if p >= game_page.SUPPORT_MIN:
                assert f'均衡で打つ確率 <b class="n">{p * 100:.0f}%</b>' in section, (own, p)

    # What happened: the page's trace is the port's trace of the same turn, and the text's
    # groups are that trace's (the text path is turn_trace + group_events, unchanged).
    for i, d in enumerate(record["decisions"]):
        if d["kind"] != "move":
            continue
        traced = show_game.turn_trace(reg, d, record["decisions"][i + 1 :], record.get("outcome"), node)
        assert traced.lines and line["transcript"]["decisions"][i]["events"]["lines"] == traced.lines
        groups = show_game.turn_events(reg, loc, d, record["decisions"][i + 1 :], record.get("outcome"), node)
        assert groups, "the text has something to say about every move turn"
    assert "起きたこと" in text and "起きたこと" in page


def test_what_the_record_does_not_carry_is_said(tools, played) -> None:  # noqa: ANN001
    show_game, game_page = tools
    reg, record, node = played
    loc = show_game.Localiser(reg, show_game.load_names("ja"))
    line = game_page.selfplay_line(reg, loc, record, node)
    page = game_page.render_html(reg, loc, line)
    assert "生成の 1 局" in page and "対戦評価の 1 局" not in page
    assert "記録にない" in page and "条件 —" in page
    assert "乱数の種" not in page and "深さ 1" not in page and "読みの段:" not in page
    # A generation record has no per-move values: said, not drawn as an empty table.
    assert "手ごとの値は記録にありません" in page

    # A hidden-bench record without the seat-1 value has no seat-1 read, not a copy of seat 0's.
    cut = copy.deepcopy(record)
    for d in cut["decisions"]:
        d.pop("foeSearchValue", None)
    assert cut["information"] == "hidden-bench"
    dash = game_page.render_html(reg, loc, game_page.selfplay_line(reg, loc, cut, node))
    assert '<b class="n c1">' not in dash and "<p>" in dash and "の読み —" in dash
    # Control: the same comparison fails on the intact record.
    assert '<b class="n c1">' in page


def test_the_page_follows_the_record(tools, played) -> None:  # noqa: ANN001
    """A changed value changes the page (the comparisons above are not vacuous)."""
    show_game, game_page = tools
    reg, record, node = played
    loc = show_game.Localiser(reg, show_game.load_names("ja"))
    page = game_page.render_html(reg, loc, game_page.selfplay_line(reg, loc, record, node))
    other = copy.deepcopy(record)
    first = next(d for d in other["decisions"] if d["kind"] == "move")
    first["searchValue"] = 1.0 - first["searchValue"] if abs(first["searchValue"] - 0.5) > 0.05 else 0.93
    changed = game_page.render_html(reg, loc, game_page.selfplay_line(reg, loc, other, node))
    assert changed != page


def test_the_text_path_is_the_traces_groups(tools, played) -> None:  # noqa: ANN001
    """`turn_events` is `turn_trace` read into groups: the refactor moved no words."""
    show_game, _game_page = tools
    reg, record, node = played
    loc = show_game.Localiser(reg, show_game.load_names("ja"))
    for i, d in enumerate(record["decisions"]):
        if d["kind"] != "move":
            continue
        rest = record["decisions"][i + 1 :]
        traced = show_game.turn_trace(reg, d, rest, record.get("outcome"), node)
        direct = show_game.group_events(loc, traced.lines, traced.acts, traced.pos)
        got = show_game.turn_events(reg, loc, d, rest, record.get("outcome"), node)
        assert traced.cut is None
        if traced.notes:
            assert got[0] == (None, traced.notes) and got[1:] == direct
        else:
            assert got == direct
