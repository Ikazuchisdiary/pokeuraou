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


# ----------------------------------------------------------------------- IKA-421: a stage's values


IKA421_STAGES = "d2r2b3n4+d2r4ban6x"


def _hidden_read(roster, pos, stages=IKA421_STAGES):  # noqa: ANN001, ANN202
    """Side 0's Bayesian read over 6 completions of side 1's bench (two of it seen)."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    ours = narrow(reg, pos, 0, limit=6).actions
    theirs = narrow(reg, pos, 1, limit=6).actions
    spreads = {side: completions(reg, pos, side, sheet, seen=frozenset({0, 1})) for side in (0, 1)}
    assert len(spreads[1]) >= 2
    return humanplay.solve_move(reg, pos, 0, ours, theirs, spreads, LEAF, budget=Budget.matrix(),
                                exact=False, ladder={"stages": ladder.parse_ladder(stages),
                                                     "budget_ms": None}).ladder


def _reads(roster, monkeypatch, stages=IKA421_STAGES, **flags):  # noqa: ANN001, ANN202
    """The open and the Bayesian read of every played position under ``flags``."""
    for name, value in {"DIAG": False, "VALUE": "guarantee", "ORACLE_PASSES": 0, "KEEP": False,
                        **flags}.items():
        monkeypatch.setattr(ladder, name, value)
    out = []
    for pos in _played(roster):
        out.append(_open_read(roster.reg, pos, stages))
        out.append(_hidden_read(roster, pos, stages))
    return out


def _same_answers(a, b) -> None:  # noqa: ANN001
    assert [r.stage for r in a.rungs] == [r.stage for r in b.rungs]
    for x, y in zip(a.rungs, b.rungs, strict=True):
        np.testing.assert_array_equal(x.strategy, y.strategy)
        assert (x.spent_ms, x.rows, x.cols, x.fresh) == (y.spent_ms, y.rows, y.cols, y.fresh)
    np.testing.assert_array_equal(a.strategy, b.strategy)


def test_the_diagnosis_changes_no_answer_and_its_bounds_hold(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-421 (`ladder.DIAG`): with the switch on, every rung carries the stage's other values
    and the answers, values and clock are the same to the bit; off, no rung has them. The
    guarantee is the lower bound of the whole matrices' game at the stage's prices, the side's
    best row against the other side's answer the upper; the first stage's game before it is
    the depth-1 game (its value the read's start). The control that the bounds bite: some
    stage's guarantee lies below the whole game's value (the pessimism IKA-421 measured)."""
    off = _reads(roster, monkeypatch)
    on = _reads(roster, monkeypatch, DIAG=True)
    below = 0
    for a, b in zip(off, on, strict=True):
        _same_answers(a, b)
        assert [r.value for r in a.rungs] == [r.value for r in b.rungs]
        assert all(r.diag is None and "diag" not in r.to_json() for r in a.rungs)
        for n, r in enumerate(b.rungs):
            d = r.to_json()["diag"]
            assert d["lower"] == r.value and d["rect"] == pytest.approx(r.value + r.optimism)
            assert d["lower"] <= d["full"] + 1e-9 <= d["upper"] + 2e-9
            assert 0.0 <= d["changed"] <= 1.0 + 1e-9
            if n == 0:
                assert d["fullBefore"] == pytest.approx(b.start[1], abs=1e-9)
                assert d["cellStart"] == pytest.approx(d["cellPrev"], abs=1e-12)
            below += d["full"] - d["lower"] > 1e-6
    assert below >= 1, "no stage's guarantee fell below the whole game's value"


def test_the_value_switch_moves_only_the_value(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-421 (`ladder.VALUE`): ``full`` / ``rect`` / ``mid`` report the whole game's value,
    the rectangle's, or half way between the bounds; the strategies and the clock are the
    guarantee's to the bit. A value it does not know is refused."""
    base = _reads(roster, monkeypatch, DIAG=True)
    moved = 0
    for name, of in (("full", lambda d: d["full"]), ("rect", lambda d: d["rect"]),
                     ("mid", lambda d: (d["lower"] + d["upper"]) / 2)):
        got = _reads(roster, monkeypatch, VALUE=name)
        for a, b in zip(base, got, strict=True):
            _same_answers(a, b)
            for x, y in zip(a.rungs, b.rungs, strict=True):
                assert y.value == pytest.approx(of(x.diag), abs=1e-12)
                moved += abs(y.value - x.value) > 1e-6
            assert b.value == (b.rungs[-1].value if b.rungs else b.start[1])
    assert moved >= 1, "no switch moved a value"
    monkeypatch.setattr(ladder, "VALUE", "best")
    with pytest.raises(ValueError, match="POKEURAOU_LADDER_VALUE"):
        _open_read(roster.reg, _played(roster)[0], IKA421_STAGES)


def test_more_oracle_passes_grow_the_rectangle(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-421 (`ladder.ORACLE_PASSES`): 0 is every stage's own one pass, to the bit; more passes let
    the oracle add rows and columns again, so some stage's rectangle is wider, and its
    guarantee is never below the whole game's lower bound it bounds (still a guarantee)."""
    base = _reads(roster, monkeypatch)
    again = _reads(roster, monkeypatch, ORACLE_PASSES=0)
    for a, b in zip(base, again, strict=True):
        _same_answers(a, b)
    more = _reads(roster, monkeypatch, ORACLE_PASSES=4, DIAG=True)
    wider = 0
    for a, b in zip(base, more, strict=True):
        for x, y in zip(a.rungs, b.rungs, strict=False):  # more passes may cost a stage
            wider += y.rows > x.rows or any(c > d for c, d in zip(y.cols, x.cols, strict=True))
        for r in b.rungs:
            assert r.diag["lower"] <= r.diag["full"] + 1e-9
    assert wider >= 1, "more passes grew no rectangle"


def test_a_stage_that_guarantees_less_keeps_the_answer_before(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-421 (`ladder.KEEP`): with the switch on, no stage's answer guarantees less at its
    own prices than the answer before it -- where the rectangle's answer would, the answer
    before is kept (its strategy, its guarantee). The control that it bites: some stage of the
    plain read guarantees less than the answer before it, and there the switch keeps it."""
    # A rectangle of one row first: narrower than the depth-1 answer's support.
    stages = "d2r1b3n4+d2r2ban6x"
    plain = _reads(roster, monkeypatch, stages, DIAG=True)
    keep = _reads(roster, monkeypatch, stages, DIAG=True, KEEP=True)
    worse = kept = 0
    for a, b in zip(plain, keep, strict=True):
        worse += sum(r.diag["prevGuarantee"] > r.diag["lower"] + 1e-6 for r in a.rungs)
        before = b.start[0]
        for r in b.rungs:
            assert r.diag["prevGuarantee"] <= r.diag["lower"] + 1e-6
            if r.diag["kept"]:
                np.testing.assert_array_equal(r.strategy, before)
                kept += 1
            before = r.strategy
    assert worse >= 1, "no stage guaranteed less than the answer before it"
    assert kept >= 1, "the switch kept no answer"


def test_a_seat_swapped_read_is_the_other_sides_read(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-421's mirror (`tools/depth_outcome.py --swap` / ``--side 1``): side 0 reading the
    position with its seats swapped is side 1 reading the position -- the same strategies, the
    values one apart (1 - P against -P) -- under hp-share, whose value is the mirror's. And the
    two sides' readings bound the same depth-1 game from either side at the start."""
    import importlib

    from pokeuraou.regulation import repo_root

    for name, value in {"DIAG": False, "VALUE": "guarantee", "ORACLE_PASSES": 0, "KEEP": False}.items():
        monkeypatch.setattr(ladder, name, value)
    # The tool sets this default for its own reads at import; here it is undone after the test.
    monkeypatch.setenv("POKEURAOU_LADDER_COUNT_FILL", "0")
    monkeypatch.syspath_prepend(str(repo_root() / "tools"))
    orientation = importlib.import_module("depth_outcome").orientation
    reg = roster.reg

    def by_action(menu, strategy):  # noqa: ANN001, ANN202 - the menus' orders may differ
        return {a.to_choice(): round(float(p), 9) for a, p in zip(menu, strategy, strict=True)
                if p > 1e-9}

    checked = same = rungs = 0
    for pos in _played(roster):
        own = _open_read(reg, pos, IKA421_STAGES, side=0)
        other = _open_read(reg, pos, IKA421_STAGES, side=1)
        flipped = pos.swapped()
        swapped = _open_read(reg, flipped, IKA421_STAGES, side=0)
        assert own.start[1] == pytest.approx(-other.start[1], abs=1e-9)
        assert [r.stage for r in swapped.rungs] == [r.stage for r in other.rungs]
        assert swapped.start[1] == pytest.approx(1 + other.start[1], abs=1e-9)
        # The tool writes every read in the recorded side 0's units.
        for (side, swap), read in (((1, False), other), ((0, True), swapped), ((0, False), own)):
            a, b = orientation(side, swap)
            assert a + b * read.start[1] == pytest.approx(own.start[1], abs=1e-9)
        menu_swapped = narrow(reg, flipped, 0, limit=8).actions
        menu_other = narrow(reg, pos, 1, limit=8).actions
        assert sorted(a.to_choice() for a in menu_swapped) == sorted(a.to_choice() for a in menu_other)
        for x, y in zip(swapped.rungs, other.rungs, strict=True):
            assert x.value == pytest.approx(1 + y.value, abs=1e-9)
            # The strategies may be two optima of a tied game (the LP's vertex moves with the
            # constant one between the two matrices); the values are the same.
            same += by_action(menu_swapped, x.strategy) == by_action(menu_other, y.strategy)
            rungs += 1
        checked += 1
    assert checked >= 2
    assert same >= rungs // 2, f"the two reads agreed in {same} of {rungs} stages' strategies"


def test_a_depth_three_childs_value_follows_the_switch(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-421 (`ladder.CHILD`): ``guarantee`` (the reference before IKA-422) reads a depth-3
    cell's children as side 0, so side 1 reading a position and side 0 reading it with the
    seats swapped differ at depth 3 (the control: some position moves). ``seat`` (the
    default since IKA-422) reads each child as the root's reader -- the two then agree at
    every stage, and side 0's own read is the reference's to the bit; ``full`` takes each
    child's whole game -- the two agree as well. A child setting it does not know is
    refused."""
    reg = roster.reg
    stages = "d2r2ban4+d3r2ban4/r2ban4"
    for name, value in {"DIAG": False, "VALUE": "guarantee", "ORACLE_PASSES": 0,
                        "KEEP": False}.items():
        monkeypatch.setattr(ladder, name, value)

    def three(child):  # noqa: ANN001, ANN202
        monkeypatch.setattr(ladder, "CHILD", child)
        out = []
        for pos in _played(roster, turns=6):
            out.append((_open_read(reg, pos, stages, side=0, width=6),
                        _open_read(reg, pos, stages, side=1, width=6),
                        _open_read(reg, pos.swapped(), stages, side=0, width=6)))
        return out

    plain, seat, full = three("guarantee"), three("seat"), three("full")
    moved = 0
    for (own, other, swapped), (own_s, other_s, swapped_s), (_o, other_f, swapped_f) in zip(
            plain, seat, full, strict=True):
        deep = [n for n, r in enumerate(other.rungs) if r.stage.startswith("d3")]
        moved += any(abs((1 - swapped.rungs[n].value) - (-other.rungs[n].value)) > 1e-6
                     for n in deep if n < len(swapped.rungs))
        assert [r.value for r in own_s.rungs] == [r.value for r in own.rungs]
        for got_other, got_swapped in ((other_s, swapped_s), (other_f, swapped_f)):
            assert [r.stage for r in got_other.rungs] == [r.stage for r in got_swapped.rungs]
            for x, y in zip(got_swapped.rungs, got_other.rungs, strict=True):
                assert x.value == pytest.approx(1 + y.value, abs=1e-9)
    assert moved >= 1, "no depth-3 stage read differently from the two seats"
    monkeypatch.setattr(ladder, "CHILD", "best")
    with pytest.raises(ValueError, match="POKEURAOU_LADDER_CHILD"):
        _open_read(reg, _played(roster)[0], stages)


def test_the_mirror_holds_at_depth_four_on_both_children_roads(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-422: with the children read as the reader (`CHILD` ``seat``, the default), side 1
    reading a position is side 0 reading the seat-swapped one at every stage to depth 4 --
    a depth-3 cell's children (read together, `_children_at_once`) and a depth-4 cell's (read
    one at a time, `_deep_children`); and the children read together are the children read
    one at a time. The controls: the reference (``guarantee``) does not hold the mirror at
    depth 3 or 4, and the together road ran (`_children_at_once` is called) where it was
    asked for and not where it was not."""
    reg = roster.reg
    stages = "d2r2ban4+d3r2ban4/r2ban4+d4r2ban3/r2ban3/r2ban3"
    for name, value in {"DIAG": False, "VALUE": "guarantee", "ORACLE_PASSES": 0,
                        "KEEP": False}.items():
        monkeypatch.setattr(ladder, name, value)
    calls = {"together": 0}
    together = ladder._children_at_once  # noqa: SLF001

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls["together"] += 1
        return together(*args, **kwargs)

    monkeypatch.setattr(ladder, "_children_at_once", counted)

    def mirror(child, batch):  # noqa: ANN001, ANN202
        monkeypatch.setattr(ladder, "CHILD", child)
        monkeypatch.setattr(ladder, "BATCH_CHILDREN", batch)
        got = []
        for pos in _played(roster, turns=5):
            got.append((_open_read(reg, pos, stages, side=1, width=6),
                        _open_read(reg, pos.swapped(), stages, side=0, width=6)))
        return got

    seat = mirror("seat", True)
    assert calls["together"] > 0, "the children were not read together"
    asked = calls["together"]
    one_at_a_time = mirror("seat", False)
    assert calls["together"] == asked, "the one-at-a-time road read children together"
    reference = mirror("guarantee", True)
    deepest = moved = 0
    for (other, swapped), (other_1, _swapped_1), (other_g, swapped_g) in zip(
            seat, one_at_a_time, reference, strict=True):
        assert [r.stage for r in other.rungs] == [r.stage for r in swapped.rungs]
        deepest = max(deepest, max((int(r.stage[1]) for r in other.rungs), default=0))
        for x, y, z in zip(swapped.rungs, other.rungs, other_1.rungs, strict=True):
            assert x.value == pytest.approx(1 + y.value, abs=1e-9), x.stage
            assert z.value == pytest.approx(y.value, abs=1e-9), x.stage
        for x, y in zip(swapped_g.rungs, other_g.rungs, strict=False):
            moved += x.stage[1] in "34" and abs(x.value - (1 + y.value)) > 1e-6
    assert deepest == 4, "no position read to depth 4"
    assert moved >= 1, "the reference held the mirror at depth 3 and 4 as well"


def test_the_ika421_switches_read_their_own_environment() -> None:
    """IKA-421: each switch is read from its own variable at import, and none takes another
    module setting's name (``PASSES`` was IKA-380's before this one shadowed it); unset, each is
    off."""
    import subprocess
    import sys

    from pokeuraou.regulation import repo_root

    probe = ("from pokeuraou import ladder as L; "
             "print(L.DIAG, L.VALUE, L.ORACLE_PASSES, L.KEEP, L.CHILD, L.PASSES)")
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("POKEURAOU_LADDER_")}
    env["PYTHONPATH"] = str(repo_root() / "src")

    def ask(**flags: str) -> str:
        return subprocess.run([sys.executable, "-c", probe], env={**env, **flags},
                              capture_output=True, text=True, check=True).stdout.split()

    # IKA-422: the children are read as the reader by default.
    assert ask() == ["False", "guarantee", "0", "False", "seat", "True"]
    assert ask(POKEURAOU_LADDER_DIAG="1", POKEURAOU_LADDER_VALUE="full",
               POKEURAOU_LADDER_ORACLE_PASSES="4", POKEURAOU_LADDER_KEEP="1",
               POKEURAOU_LADDER_CHILD="guarantee") == ["True", "full", "4", "True", "guarantee",
                                                       "True"]
