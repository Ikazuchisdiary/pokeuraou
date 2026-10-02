"""The ladder: a reading that gets more exact the longer it runs (IKA-367, `pokeuraou.ladder`).

What these hold:

- **a stage label reads back** as it was written, and a label that names no stage stops;
- **every answer is a completed stage**: read with no budget, every stage completes; read
  on a count budget that ends inside the second stage, the answer is the first stage's to
  the bit (a stage does not depend on the budget it runs under), and the result says which
  stage did not complete. The control that the budget bit: the full read's second stage
  moved the answer;
- **a cell read the same way is not read again**: two stages of one kind, the second with a
  wider rectangle, read only the cells the first did not;
- **one completion of weight 1 is the open game**: the Bayesian root with nothing hidden
  gives the open game's answer, for both sides;
- **a depth-3 stage** reads its cells a ply further and moves the value (hp-share is not
  flat there);
- **the agent plays a ladder and the count clock replays it**; the condition key parses.

Played on `rizabanadohido` under hp-share with damage-ordered children (``n``): these tests
have no Q.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou import humanplay, ladder, timematch
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import Position
from pokeuraou.search import search
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from ._port import resolve_turn

LEAF = HP_SHARE.batch
SEEN_ALL = frozenset({0, 1, 2, 3})


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _played(roster, turns: int = 4, seed: int = 3) -> list[Position]:  # noqa: ANN001
    """Positions from one game of two different fours (a mirror is flat under hp-share)."""
    reg = roster.reg
    pos = position_from_sets(reg, list(roster.sets[:4]), list(roster.sets[2:6]))
    rng = np.random.default_rng(seed)
    out: list[Position] = []
    for _ in range(turns):
        if pos.ended:
            break
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            break
        out.append(pos.copy())
        result = resolve_turn(
            reg, pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        pos = max(result.branches, key=lambda b: b.probability).position
    return out


def _named(stages, fills):  # noqa: ANN001, ANN202
    parsed = ladder.parse_ladder(stages)
    parsed.fills = fills
    return parsed


def _open_read(reg, pos, stages, budget_ms=None, *, side=0, width=8, fills=False):  # noqa: ANN001, ANN202
    ours = narrow(reg, pos, 0, limit=width).actions
    theirs = narrow(reg, pos, 1, limit=width).actions
    got = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
    eq = got.equilibrium
    m = np.asarray(got.payoff, dtype=np.float64)
    start = SimpleNamespace(row_strategy=eq.row_strategy if side == 0 else eq.col_strategy,
                            col_strategies=[eq.col_strategy if side == 0 else eq.row_strategy])
    return ladder.read(reg, side, ours, theirs, [ladder.Item(pos)], [m if side == 0 else -m.T],
                       [1.0], start, LEAF, budget=Budget.matrix(),
                       stages=_named(stages, fills), budget_ms=budget_ms)


def test_a_stage_label_reads_back() -> None:
    for label in ("d2r4b3k8", "d2r8bak24x", "d3r4b3k24x/r3b3k16", "d2r2b3n4", "d3r2ban4/r2b2n4",
                  "d4r4bak24x/r4bak24/r3b3n16"):
        assert ladder.parse_stage(label).label == label
    assert [s.label for s in ladder.parse_ladder("d2r2b3n4+d2r3ban4x")] == ["d2r2b3n4",
                                                                           "d2r3ban4x"]
    assert len(ladder.parse_ladder("L1")) == len(ladder.LADDERS["L1"])
    for bad in ("d2r4b3", "d3r4b3k8", "d2r4b3k8/r2b2k8", "d4r4b3k8", "L9"):
        with pytest.raises(ValueError):
            ladder.parse_ladder(bad)


def test_the_rule_writes_stages_past_l5() -> None:
    """IKA-376: the rule (`ladder.unending`) begins with L5's two depth-4 stages, so L6 is L5
    and then the rule; every stage it writes reads back, one depth at a time up to 9, and
    L6 alone fills a wall-clock budget."""
    rule = ladder.unending(4)
    assert rule[:2] == ladder.LADDERS["L5"][-2:]
    l6 = ladder.parse_ladder("L6")
    assert [s.label for s in l6[:len(ladder.LADDERS["L5"])]] == list(ladder.LADDERS["L5"])
    assert [s.label for s in l6[len(ladder.LADDERS["L5"]):]] == list(rule[2:])
    assert rule[4] == "d4r8bak24x/r6bak24/r4bak24"  # IKA-369's S2
    assert rule[5] == "d5r4b3k24x/r3b3k24/r3b3k24/r3b3k16"
    depths = [s.depth for s in l6]
    assert depths == sorted(depths) and depths[-1] == 9
    assert len({s.label for s in l6}) == len(l6)
    assert l6.fills and not ladder.parse_ladder("L5").fills
    assert not ladder.parse_ladder("d2r2b3n4").fills


def test_a_read_that_stops_at_a_stage(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-393: ``<name>@<n>`` is the named ladder's first n stages and reads them all with no
    budget (`stopped == "done"`); n = 0 is no stage (the depth-1 answer). L6@n is L6's first n
    stages and never fills a budget. The hidden-bench ladder is chosen by the completions."""
    l6 = ladder.LADDERS["L6"]
    assert [s.label for s in ladder.parse_ladder("L6@7")] == list(l6[:7])
    assert len(ladder.parse_ladder("L6@0")) == 0
    assert not ladder.parse_ladder("L6@7").fills and ladder.parse_ladder("L6").fills
    for bad in ("L6@", "L6@x", "L6@41", "d2r4b3k8@1", "L9@1"):
        with pytest.raises(ValueError):
            ladder.parse_ladder(bad)
    cond = timematch.parse_condition("a:seconds=1000000,clock=count,ladder=L6@7,hidden_ladder=L6@4")
    assert (cond.ladder, cond.hidden_ladder) == ("L6@7", "L6@4")
    assert timematch.parse_condition("a:seconds=1,clock=count").hidden_ladder is None
    agent = SimpleNamespace(ladder="L6@7", hidden_ladder="L6@4")
    assert humanplay.ladder_spec_for(agent, 1) == "L6@7"
    assert humanplay.ladder_spec_for(agent, 3) == "L6@4"
    assert humanplay.ladder_spec_for(SimpleNamespace(ladder="L6@7", hidden_ladder=None), 3) == "L6@7"
    monkeypatch.setitem(ladder.LADDERS, "T2", ("d2r2b3n4", "d2r4ban6x"))
    reg = roster.reg
    pos = _played(roster)[1]
    two = _open_read(reg, pos, "T2")
    one = _open_read(reg, pos, "T2@1")
    none = _open_read(reg, pos, "T2@0")
    assert two.stopped == "done" and len(two.rungs) == 2
    assert one.stopped == "done" and [r.stage for r in one.rungs] == ["d2r2b3n4"]
    assert one.rungs[0].value == two.rungs[0].value  # the same first stage
    assert none.stopped == "done" and none.rungs == []


def test_every_answer_is_a_completed_stage(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    stages = "d2r2b3n4+d2r4ban6x"
    checked = 0
    for pos in _played(roster):
        full = _open_read(reg, pos, stages)
        assert full.stopped == "done" and len(full.rungs) == 2
        first, second = full.rungs
        if first.spent_ms >= second.spent_ms or abs(first.value - second.value) < 1e-9:
            continue
        # A budget that ends inside the second stage: the first stage's answer, to the bit.
        cut = _open_read(reg, pos, stages, budget_ms=(first.spent_ms + second.spent_ms) / 2)
        assert [r.stage for r in cut.rungs] == ["d2r2b3n4"]
        assert cut.stopped == "budget" and cut.unfinished == "d2r4ban6x"
        np.testing.assert_array_equal(cut.strategy, first.strategy)
        assert cut.value == first.value
        # A budget below the first stage: the depth-1 answer, no rung.
        none = _open_read(reg, pos, stages, budget_ms=first.spent_ms / 4)
        assert none.rungs == [] and none.unfinished == "d2r2b3n4"
        checked += 1
    assert checked >= 1, "no position where the second stage moved the answer"


def test_a_cell_read_the_same_way_is_not_read_again(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    for pos in _played(roster):
        got = _open_read(reg, pos, "d2r2ban4+d2r4ban4")
        if len(got.rungs) < 2:
            continue
        first, second = got.rungs
        assert first.fresh == first.rows * first.cols[0]
        # The second rectangle holds some of the first's cells (the first's oracle may
        # have grown it by a line the second's ordering leaves out): those are not read.
        whole = second.rows * second.cols[0]
        assert whole - first.fresh <= second.fresh < whole
        return
    pytest.fail("no position read both stages")


def test_one_completion_is_the_open_ladder(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    stages = ladder.parse_ladder("d2r2b3n4+d2r3ban4")
    checked = 0
    for pos in _played(roster):
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        spreads = {side: completions(reg, pos, side, sheet, seen=SEEN_ALL) for side in (0, 1)}
        for me in (0, 1):
            how = {"stages": stages, "budget_ms": None}
            opened = humanplay.solve_move(reg, pos, me, ours, theirs, None, LEAF,
                                          budget=Budget.matrix(), exact=True, ladder=dict(how))
            hidden = humanplay.solve_move(reg, pos, me, ours, theirs, spreads, LEAF,
                                          budget=Budget.matrix(), exact=False, ladder=dict(how))
            assert [r.stage for r in opened.ladder.rungs] == [r.stage for r in hidden.ladder.rungs]
            np.testing.assert_allclose(hidden.strategy, opened.strategy, atol=1e-9)
            assert hidden.value == pytest.approx(opened.value, abs=1e-9)
            checked += 1
    assert checked >= 4


def test_a_depth_three_stage_reads_a_ply_further(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    moved = four = 0
    for pos in _played(roster):
        got = _open_read(reg, pos, "d2r2ban4+d3r2ban4/r2ban4+d4r1b2n3/r1b2n3/r1b2n3", width=6)
        if len(got.rungs) < 2:
            continue
        assert got.rungs[1].stage == "d3r2ban4/r2ban4"
        assert got.work["subgames"] > 0 and got.rungs[1].fresh > 0
        # The children's own reads add their work to the cell's.
        assert got.rungs[1].work["subgames"] > got.rungs[0].work["subgames"]
        moved += abs(got.rungs[1].value - got.rungs[0].value) > 1e-9
        four += len(got.rungs) == 3
    assert moved >= 1, "depth 3 moved no value"
    assert four >= 1, "no depth-4 stage completed"


# ----------------------------------------------------------------------- the agent


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    import copy

    from pokeuraou.pool import load_pool
    from pokeuraou.regulation import repo_root

    base = json.loads(
        (repo_root() / "configs" / "teams" / "rizabanadohido.json").read_text(encoding="utf-8")
    )["team"]
    rotated = copy.deepcopy(base[2:] + base[:2])
    data = {
        "id": "test-pool", "name": "test", "regulation": "gen9championsvgc2026regmb",
        "character": "a test pool",
        "validatedAgainst": {"formatId": "gen9championsvgc2026regmb"},
        "teams": [{"id": f"t{i}", "name": f"team {i}", "team": t}
                  for i, t in enumerate([base, rotated])],
    }
    path = tmp_path_factory.mktemp("pool") / "p.json"
    path.write_bytes(json.dumps(data).encode("utf-8"))
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _timeless(value):  # noqa: ANN001, ANN202
    # IKA-370: "here" is the read's own waits and CPU (milliseconds).
    times = ("seconds", "ratio", "menuSeconds", "selectionSeconds", "wallMs", "here")
    if isinstance(value, dict):
        return {k: _timeless(v) for k, v in value.items() if k not in times}
    if isinstance(value, list):
        return [_timeless(v) for v in value]
    return value


def test_the_agent_plays_a_ladder_and_the_count_clock_replays_it(pool, monkeypatch) -> None:  # noqa: ANN001
    from pokeuraou.hidden import DEFAULT_BENCH_DROP
    from pokeuraou.humanplay import NodeTime

    monkeypatch.setattr(humanplay, "NODE_TIME", {("local", 1): NodeTime(10.0, 0.05)})
    spec = "lad:seconds=0.5,clock=count,ladder=d2r2b3n4+d2r4ban6x"
    lad = timematch.parse_condition(spec)
    assert lad.ladder == "d2r2b3n4+d2r4ban6x" and lad.threads == 1
    with pytest.raises(ValueError):
        timematch.parse_condition("bad:seconds=1,clock=count,ladder=d9")
    flat = timematch.Condition(name="flat", seconds=0.5, threads=1, clock="count",
                               width_only=True)

    def match(seed: int) -> timematch.Match:
        return timematch.Match(
            reg=pool.reg, evaluate=None, leaf_name="hp-share", rank_fill="refs2",
            bench_drop=DEFAULT_BENCH_DROP, tested=lad, other=flat, seed=seed, max_turns=3,
            rank_by_leaf=False)

    teams = (pool.teams[0], pool.teams[1])
    first = timematch.play_pair(match(3), 1, teams)
    again = timematch.play_pair(match(3), 1, teams)
    assert json.dumps(_timeless(first)) == json.dumps(_timeless(again))
    rows = [r for ln in first for r in ln["moves"] if r["condition"] == "lad"]
    assert rows and all("ladder" in r for r in rows)
    assert any(r["ladder"]["rungs"] for r in rows), "no move completed a stage"
    assert all("ladder" not in r for ln in first for r in ln["moves"] if r["condition"] == "flat")
    other = timematch.play_pair(match(4), 1, teams)
    assert json.dumps(_timeless(other)) != json.dumps(_timeless(first))


def test_the_count_clock_fills_a_budget_only_when_asked(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-384: ``COUNT_FILL`` with a ladder in `FILLS` begins the stage the prediction says
    will not fit and spends the counted budget on it (the answer is still the last completed
    stage, to the bit); off, or for a ladder not in `FILLS`, that stage is not begun."""
    reg = roster.reg
    stages = "d2r2b3n4+d2r4ban6x"
    checked = 0
    for pos in _played(roster):
        full = _open_read(reg, pos, stages)
        first, second = full.rungs
        if first.spent_ms >= second.spent_ms or abs(first.value - second.value) < 1e-9:
            continue
        budget = first.spent_ms + 0.05 * (second.spent_ms - first.spent_ms)
        monkeypatch.setattr(ladder, "COUNT_FILL", False)
        off = _open_read(reg, pos, stages, budget, fills=True)
        if off.cut != (0, 0):
            continue  # the prediction let the stage begin: no difference to show here
        assert off.unfinished == "d2r4ban6x"
        monkeypatch.setattr(ladder, "COUNT_FILL", True)
        unnamed = _open_read(reg, pos, stages, budget, fills=False)
        assert unnamed.cut == (0, 0) and unnamed.work == off.work
        filled = _open_read(reg, pos, stages, budget, fills=True)
        assert filled.cut[0] > 0, "the flag did not begin the stage"
        assert [r.stage for r in filled.rungs] == ["d2r2b3n4"]
        np.testing.assert_array_equal(filled.strategy, first.strategy)
        assert filled.unfinished == "d2r4ban6x"
        assert filled.spent_ms >= budget
        checked += 1
    assert checked >= 1, "no position where the prediction refused the second stage"


def test_a_rung_records_the_row_it_plays_most_only_when_asked(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-384: ``RECORD_TOP`` adds ``top`` (the strategy's most played row) to each rung's JSON;
    off, the JSON has no such key."""
    pos = _played(roster)[0]
    got = _open_read(roster.reg, pos, "d2r2b3n4+d2r4ban6x")
    monkeypatch.setattr(ladder, "RECORD_TOP", False)
    assert all("top" not in r.to_json() for r in got.rungs)
    monkeypatch.setattr(ladder, "RECORD_TOP", True)
    tops = [r.to_json()["top"] for r in got.rungs]
    assert tops == [int(np.argmax(r.strategy)) for r in got.rungs] and len(tops) == 2


# ----------------------------------------------------------------------- IKA-418: the root widened


def test_root_widths_spell_and_order() -> None:
    assert humanplay.parse_root_widths("12-24-48-64-all") == (12, 24, 48, 64, 0)
    assert humanplay.parse_root_widths("all") == (0,)
    for bad in ("", "24-12", "12-12", "all-12", "12-x", "0-12", "12--24"):
        with pytest.raises(ValueError):
            humanplay.parse_root_widths(bad)
    cond = timematch.parse_condition("a:seconds=1,clock=count,ladder=L6,root_widths=12-all")
    assert cond.root_widths == "12-all"
    assert timematch.parse_condition("a:seconds=1,clock=count,ladder=L6").root_widths is None
    with pytest.raises(ValueError):
        timematch.parse_condition("a:seconds=1,clock=count,ladder=L6,root_widths=64-12")
    # A root widened over a move is a ladder's.
    with pytest.raises(ValueError):
        humanplay.Agent(reg=None, evaluate=None, name="x", seconds=1.0, cores=1, clock="wall",
                        root_widths=(12, 0))


def test_the_steps_taken_fit_the_share_and_widen() -> None:
    price = humanplay.NodeTime(fixed_ms=0.0, cell_ms=1.0)

    def menu(rows: int, cols: int):  # noqa: ANN202
        return [object()] * rows, [object()] * cols

    steps = humanplay.RootSteps([menu(2, 2), menu(4, 4), menu(4, 4), menu(8, 8)], price, 1, 30.0)
    first = steps.node_ms(*steps.menus[0])
    assert first == 4.0
    # 4 + 16 fit 30; the repeated menu is left out; 8 x 8 (64) does not fit.
    later = steps.later(first, (2, 2))
    assert [(len(o), len(t), ms) for o, t, ms in later] == [(4, 4, 16.0)]
    # A wider share takes it too (the control: the steps' count answers to the share).
    steps.share_ms = 100.0
    assert [(len(o), ms) for o, _t, ms in steps.later(first, (2, 2))] == [(4, 16.0), (8, 64.0)]
    # Classes multiply the cells.
    steps.classes, steps.share_ms = 3, 30.0
    assert steps.later(steps.node_ms(*steps.menus[0]), (2, 2)) == []


def _node_ms(price, ours, theirs):  # noqa: ANN001, ANN202
    return price.ms(len(ours) * len(theirs))


def test_a_widened_root_reads_the_wider_menu_as_a_fixed_one_does(roster) -> None:  # noqa: ANN001
    """The staged read ends where a read on the widest menu it reached ends -- the same answer
    to the bit and the same clock, the nodes before it charged -- and a share too small for the
    later step leaves the read on the first menu (the control that the share bites)."""
    reg = roster.reg
    price = humanplay.NodeTime(fixed_ms=10.0, cell_ms=0.05)
    stages = ladder.parse_ladder("d2r2b3n4+d2r3ban4")
    checked = 0
    for pos in _played(roster):
        narrow_menus = tuple(narrow(reg, pos, s, limit=3).actions for s in (0, 1))
        wide_menus = tuple(narrow(reg, pos, s, limit=6).actions for s in (0, 1))
        if tuple(map(len, narrow_menus)) == tuple(map(len, wide_menus)):
            continue
        first = _node_ms(price, *narrow_menus)
        second = _node_ms(price, *wide_menus)

        def staged(share: float, narrow_menus=narrow_menus, wide_menus=wide_menus, first=first,
                   pos=pos):  # noqa: ANN202
            how = {"stages": stages, "budget_ms": None, "clock": "count", "start_ms": first,
                   "root": humanplay.RootSteps([narrow_menus, wide_menus], price, 1, share)}
            return humanplay.solve_move(reg, pos, 0, list(narrow_menus[0]), list(narrow_menus[1]),
                                        None, LEAF, budget=Budget.matrix(), exact=True,
                                        ladder=how)

        def fixed(menus, start, pos=pos):  # noqa: ANN001, ANN202
            how = {"stages": stages, "budget_ms": None, "clock": "count", "start_ms": start}
            return humanplay.solve_move(reg, pos, 0, list(menus[0]), list(menus[1]), None, LEAF,
                                        budget=Budget.matrix(), exact=True, ladder=how)

        wide = staged(first + second + 1.0)
        assert [s["rows"] for s in wide.root] == [len(narrow_menus[0]), len(wide_menus[0])]
        direct = fixed(wide_menus, first + second)
        np.testing.assert_array_equal(wide.strategy, direct.strategy)
        assert [(r.stage, r.spent_ms) for r in wide.ladder.rungs] == [
            (r.stage, r.spent_ms) for r in direct.ladder.rungs]
        assert len(wide.strategy) == len(wide_menus[0])
        short = staged(first + 0.5 * second)
        assert len(short.root) == 1 and len(short.strategy) == len(narrow_menus[0])
        np.testing.assert_array_equal(short.strategy, fixed(narrow_menus, first).strategy)
        checked += 1
    assert checked >= 1, "no position where the wide menu is wider than the narrow"


def test_the_agent_widens_its_root_and_says_so(pool, monkeypatch) -> None:  # noqa: ANN001
    from pokeuraou.hidden import DEFAULT_BENCH_DROP

    monkeypatch.setattr(humanplay, "NODE_TIME", {("local", 1): humanplay.NodeTime(10.0, 0.05)})
    steps = timematch.parse_condition(
        "lad:seconds=0.5,clock=count,ladder=d2r2b3n4,root_widths=3-6")
    plain = timematch.parse_condition("lad:seconds=0.5,clock=count,ladder=d2r2b3n4,width=3")
    flat = timematch.Condition(name="flat", seconds=0.5, threads=1, clock="count", width_only=True)

    def moves(tested):  # noqa: ANN001, ANN202
        match = timematch.Match(
            reg=pool.reg, evaluate=None, leaf_name="hp-share", rank_fill="refs2",
            bench_drop=DEFAULT_BENCH_DROP, tested=tested, other=flat, seed=3, max_turns=3,
            rank_by_leaf=False)
        lines = timematch.play_pair(match, 1, (pool.teams[0], pool.teams[1]))
        return [r for ln in lines for r in ln["moves"] if r["condition"] == "lad"]

    widened, fixed = moves(steps), moves(plain)
    assert widened and all("rootSteps" in r for r in widened)
    assert any(len(r["rootSteps"]) == 2 for r in widened), "no move took the second step"
    assert all("rootSteps" not in r for r in fixed)
    # The move is read on the last step's menus (the row's sides may be swapped by the seat).
    for r in widened:
        last = r["rootSteps"][-1]
        assert sorted((r["rows"], r["cols"])) == sorted((last["rows"], last["cols"]))
        assert r["nodeCells"] == last["rows"] * last["cols"] * max(r["classes"], 1)
