"""IKA-392: the selection read deeper than the leaf (`pokeuraou.selection_deep`).

What these hold:

- **the stages find what the leaf misses**, in a game whose truth is known: a matrix with a
  known truth, a leaf that is the truth plus a constant bias, row and column effects and
  noise, and a reader that answers with the truth. The deep answer's exploitability in the
  truth is a fraction of the leaf's;
- **the correction of the unread cells matters** (the measured systematic shift of a read
  from the leaf): with it off the same reads find a worse answer and read more cells -- the
  control that the shift is not decoration;
- **the confirmation matters**: a reading with no confirmation pass is worse on the same
  instances;
- **a stage not completed is not the answer**: a reader that gives back half of a
  rectangle leaves the leaf's answer;
- **the reading of a cell is the person's move reading of the turn-1 position**, in the
  port (`read_cell` on a pool team pair, both the stage and a repeat);
- **nothing changes when the flag is off**: `solve_entry` with no reading is the leaf's
  solve to the bit, a pair whose two conditions name no selection hands the one solve to
  both seats and no beliefs of its own (the call `play_pair` made before), and generation's
  store (`poolplay.SolvedSelections`) never holds a deeper solve.
"""

from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pytest

from pokeuraou import humanplay, poolplay, selection_deep, timematch
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve, solve_bayesian
from pokeuraou.hidden import DEFAULT_BENCH_DROP
from pokeuraou.payoff import HP_SHARE
from pokeuraou.pool import load_pool
from pokeuraou.regulation import repo_root
from pokeuraou.selection import SelectionAnalysis
from pokeuraou.selection_book import BenchPrior
from pokeuraou.selection_deep import Reading, fit_shift, parse_reading, prices, solve_selection_deep
from pokeuraou.teams import all_selections
from pokeuraou.timematch import Match, parse_condition

# ------------------------------------------------------------------------ a known truth


class TruthReader:
    """Answers a cell with the truth; keeps what it was asked."""

    workers = 1

    def __init__(self, truth: np.ndarray, *, give: float = 1.0) -> None:
        self.truth, self.asked, self.give = truth, [], give

    def read(self, reading, our, their, selections, cells, deadline=None):  # noqa: ANN001, ANN201
        self.asked += list(cells)
        cells = list(cells)[: int(len(cells) * self.give)]
        return {c: float(self.truth[c]) for c in cells}


def _instance(seed: int, bias: float = 0.06, noise: float = 0.05, effect: float = 0.05):  # noqa: ANN202
    rng = np.random.default_rng(seed)
    n = 90
    u, v = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    truth = np.clip(0.5 + 0.09 * (u @ v.T) / 1.7, 0.05, 0.95)
    leaf = (truth + bias + rng.normal(0, effect, (n, 1)) + rng.normal(0, effect, (1, n))
            + rng.normal(0, noise, (n, n)))
    return truth, np.clip(leaf, 0.0, 1.0)


def _base(leaf: np.ndarray) -> SelectionAnalysis:
    selections = tuple(all_selections(6, 4))
    return SelectionAnalysis(
        reg=None, our_six=(), their_six=(), ours=selections, theirs=selections,  # type: ignore[arg-type]
        matrices=(leaf,), classes=(), equilibrium=solve_bayesian([leaf], np.ones(1)), notes=(),
    )


def _exploitability(truth: np.ndarray, x: np.ndarray) -> float:
    return float(solve(truth).value - (np.asarray(x) @ truth).min())


def _deep(seed: int, spec: str, **kwargs):  # noqa: ANN001, ANN202
    truth, leaf = _instance(seed)
    reader = TruthReader(truth, **kwargs)
    analysis, report = solve_selection_deep(
        None, (), (), None, reader, parse_reading(spec), base=_base(leaf))
    return truth, leaf, reader, analysis, report


SEEDS = range(8)


def test_a_reading_is_written_and_refused_as_written() -> None:
    assert parse_reading("default") == Reading()
    got = parse_reading("stage=d2r8b3k8;rects=4-8-12,confirm=1,shift=const,width=24,fill=off")
    assert (got.stage, got.rects, got.confirm, got.shift, got.width, got.fill) == (
        "d2r8b3k8", (4, 8, 12), 1, "const", 24, False)
    for bad in ("shift=other", "unknown=1", "rects=x"):
        with pytest.raises(ValueError):
            parse_reading(bad)


def test_a_cells_value_is_the_read_side_symmetric_one_unless_the_guarantee_is_asked() -> None:
    """The ladder's own value is one side's guarantee: read from each side, a position sums to
    less than 1 (measured -0.10 on 30 cells). The cell's default is the value of the priced matrix."""
    from types import SimpleNamespace

    from pokeuraou.selection_deep import _cell_value

    rung = SimpleNamespace(value=0.40, optimism=0.07)
    read = SimpleNamespace(value=0.40, rungs=[rung], prices=[np.array([[0.6, 0.2], [0.3, 0.5]])])
    assert _cell_value(read, "guarantee") == 0.40
    assert abs(_cell_value(read, "rect") - 0.47) < 1e-12
    assert abs(_cell_value(read, "priced") - solve(read.prices[0]).value) < 1e-12
    assert parse_reading("default").value == "priced"
    with pytest.raises(ValueError):
        parse_reading("value=other")
    assert _cell_value(SimpleNamespace(value=0.3, rungs=[], prices=[]), "rect") == 0.3


def test_the_shift_fits_what_it_says() -> None:
    n = 90
    rng = np.random.default_rng(0)
    leaf = rng.uniform(0.3, 0.7, (n, n))
    a, b = rng.normal(0, 0.05, n), rng.normal(0, 0.05, n)
    truth = leaf + 0.04 + a[:, None] + b[None, :]
    mask = rng.random((n, n)) < 0.4  # every row and column has reads
    read = {(int(i), int(j)): float(truth[i, j]) for i, j in zip(*np.nonzero(mask), strict=True)}
    assert not fit_shift(read, leaf, "none").any()
    d = np.array([read[c] - leaf[c] for c in read])
    assert np.allclose(fit_shift(read, leaf, "const"), d.mean())
    got = fit_shift(read, leaf, "add", ridge=0.01)
    assert np.abs(got - (truth - leaf)).max() < 5e-3  # the additive shift is recovered
    # ... and the prices keep a cell read at its value, whatever the shift says.
    p = prices(leaf, read, "add")
    assert all(p[c] == v for c, v in read.items())
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_the_stages_find_what_the_leaf_misses() -> None:
    leaf_loss, deep_loss, cells = [], [], []
    for seed in SEEDS:
        truth, leaf, reader, analysis, report = _deep(seed, "rects=8-16,confirm=2,shift=add")
        leaf_loss.append(_exploitability(truth, _base(leaf).equilibrium.row_strategy))
        deep_loss.append(_exploitability(truth, analysis.equilibrium.row_strategy))
        cells.append(report.cells)
        # every cell is read once, and the count says what was read (a positive control)
        assert len(reader.asked) == len(set(reader.asked)) == report.cells
        assert report.completed.startswith("stage 2")
    assert np.mean(deep_loss) < 0.2 * np.mean(leaf_loss)
    assert max(cells) < 0.1 * 8100  # a rectangle of the game, never the whole matrix
    # the control that the comparison can fail: no reading is the leaf's answer to the bit
    truth, leaf = _instance(0)
    analysis, report = solve_selection_deep(
        None, (), (), None, TruthReader(truth), parse_reading("rects=8"), base=_base(leaf),
        deadline=0.0)
    assert report.completed == "leaf" and report.cells == 0
    assert np.array_equal(analysis.equilibrium.row_strategy, _base(leaf).equilibrium.row_strategy)


def test_the_same_reading_is_the_same_answer() -> None:
    first = _deep(3, "rects=8-16,confirm=2,shift=add")[3].equilibrium
    again = _deep(3, "rects=8-16,confirm=2,shift=add")[3].equilibrium
    other = _deep(4, "rects=8-16,confirm=2,shift=add")[3].equilibrium
    assert np.array_equal(first.row_strategy, again.row_strategy)
    assert not np.array_equal(first.row_strategy, other.row_strategy)


def test_the_shift_of_the_unread_cells_is_not_decoration() -> None:
    loss = {}
    cells = {}
    for mode in ("none", "const", "add"):
        loss[mode], cells[mode] = [], []
        for seed in SEEDS:
            truth, _leaf, _reader, analysis, report = _deep(seed, f"rects=8-16,confirm=2,shift={mode}")
            loss[mode].append(_exploitability(truth, analysis.equilibrium.row_strategy))
            cells[mode].append(report.cells)
    # fault injection: with the shift off the answer is worse and more cells are read
    assert np.mean(loss["none"]) > 2 * np.mean(loss["add"])
    assert np.mean(cells["none"]) > np.mean(cells["const"]) > 0.9 * np.mean(cells["add"])


def test_the_confirmation_finds_the_answer_outside_the_rectangle() -> None:
    with_confirm, without = [], []
    for seed in SEEDS:
        truth, *_rest, analysis, _report = _deep(seed, "rects=8,confirm=2,shift=const")
        with_confirm.append(_exploitability(truth, analysis.equilibrium.row_strategy))
        truth, *_rest, analysis, _report = _deep(seed, "rects=8,confirm=0,shift=const")
        without.append(_exploitability(truth, analysis.equilibrium.row_strategy))
    assert np.mean(without) > 1.5 * np.mean(with_confirm)


def test_a_stage_not_completed_is_not_the_answer() -> None:
    truth, leaf, reader, analysis, report = _deep(2, "rects=8-16", give=0.5)
    assert report.completed == "leaf" and report.steps == []
    assert np.array_equal(analysis.equilibrium.row_strategy, _base(leaf).equilibrium.row_strategy)
    assert reader.asked  # the reading was tried (the control that nothing was skipped)


# ------------------------------------------------------------------------ in the port


_stub = HP_SHARE.batch  # scored in the port: a stub leaf in Python costs seconds a node


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
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


READING = "stage=d2r4b3n8;rects=2;confirm=0;width=8"


def test_a_cell_is_read_as_a_move_reads_a_turn_one_position(pool) -> None:  # noqa: ANN001
    reg = pool.reg
    reader = selection_deep.SerialReader(reg, _stub, rank_fill="refs2", rank_by_leaf=False)
    sels = all_selections(6, 4)
    ours, theirs = pool.teams[0].sets, pool.teams[1].sets
    got = reader.read(parse_reading(READING), ours, theirs, sels, [(0, 0), (0, 1), (1, 0)])
    assert set(got) == {(0, 0), (0, 1), (1, 0)} and reader.cells == 3
    assert all(0.0 <= v <= 1.0 for v in got.values())
    again = reader.read(parse_reading(READING), ours, theirs, sels, [(0, 1)])
    assert again[(0, 1)] == got[(0, 1)]  # a cell read twice is the same value
    assert len(set(got.values())) > 1  # the cells differ (the control that they are not all one)


def test_a_deep_entry_is_the_leafs_solve_read_deeper(pool) -> None:  # noqa: ANN001
    reg, teams = pool.reg, (pool.teams[0], pool.teams[1])
    leaf = humanplay.solve_entry(reg, teams, _stub, "stub")
    reader = selection_deep.SerialReader(reg, _stub, rank_fill="refs2", rank_by_leaf=False)
    report: list = []
    deep = humanplay.solve_entry(reg, teams, _stub, "stub", reading=READING, reader=reader,
                                 report=report)
    assert leaf.model == "stub" and deep.model.startswith("stub+selection[")
    assert report[0].cells > 0 and reader.cells == report[0].cells
    assert deep.selections == leaf.selections
    assert len(deep.our_strategy) == 90 and abs(float(np.sum(deep.our_strategy)) - 1.0) < 1e-9
    # the flag off is the solve it always was, to the bit
    from pokeuraou.selection import SpreadClass, book_entry, solve_selection

    direct = book_entry(
        solve_selection(reg, teams[0].sets,
                        [SpreadClass(weight=1.0, sets=tuple(teams[1].sets), label="sheet")], _stub),
        key=f"{teams[0].id}|{teams[1].id}", player=teams[1].name, model="stub")
    once = {k: v for k, v in leaf.to_json().items() if k != "seconds"}
    assert once == {k: v for k, v in direct.to_json().items() if k != "seconds"}
    # ... and it is not the deep solve (the control that the comparison can fail)
    assert once != {k: v for k, v in deep.to_json().items() if k != "seconds"}


def test_generations_store_never_holds_a_deeper_solve(pool, tmp_path) -> None:  # noqa: ANN001
    assert "selection_deep" not in inspect.getsource(poolplay)
    store = tmp_path / "solved"
    solved = poolplay.SolvedSelections(
        pool.reg, pool.teams, _stub, model="stub", store=store, tag="t")
    solved.entry(0, 1)
    written = [p.read_text(encoding="utf-8") for p in store.glob("*.json")]
    assert written and all("selection[" not in text for text in written)


def _match(pool, tested, other, *, seed: int = 3):  # noqa: ANN001, ANN202
    return Match(
        reg=pool.reg, evaluate=_stub, leaf_name="stub", rank_fill="refs2",
        bench_drop=DEFAULT_BENCH_DROP, tested=tested, other=other, seed=seed, max_turns=1,
        rank_by_leaf=False,
    )


class _Seen(Exception):  # noqa: N818 - a signal
    pass


def _calls(monkeypatch, match, pool):  # noqa: ANN001, ANN202
    """`play_pair`'s first `humanplay.play` call (its entry and priors), without playing: the
    tested condition on side 0. A caller reads the other side's arrangement by swapping."""
    calls = []

    def play(agent, person, teams, **kwargs):  # noqa: ANN001, ANN202
        calls.append((agent, person, kwargs))
        raise _Seen

    monkeypatch.setattr(humanplay, "play", play)
    with pytest.raises(_Seen):
        timematch.play_pair(match, 0, (pool.teams[0], pool.teams[1]))
    return calls


def test_two_conditions_on_one_selection_play_as_they_always_did(monkeypatch, pool) -> None:  # noqa: ANN001
    tested, other = parse_condition("a:seconds=1,clock=count"), parse_condition("b:seconds=2,clock=count")
    assert tested.selection is None and other.selection is None
    calls = _calls(monkeypatch, _match(pool, tested, other), pool)
    swapped = _calls(monkeypatch, _match(pool, other, tested), pool)
    for _agent, person, kwargs in calls + swapped:
        assert kwargs["priors"] is None  # the play's own beliefs, from its one solve
        assert person.entry is kwargs["entry"]  # both seats draw from the one solve
        assert "selection[" not in kwargs["entry"].model


def test_each_seat_plays_and_believes_from_its_own_selection(monkeypatch, pool) -> None:  # noqa: ANN001
    deep = parse_condition(f"deep:seconds=1,clock=count,selection={READING}")
    leaf = parse_condition("leaf:seconds=1,clock=count")
    species = tuple(tuple(s.species for s in t.sets) for t in (pool.teams[0], pool.teams[1]))
    distinct = []
    # the first game of pair 0: the tested condition on side 0; then swapped, the deep
    # condition on side 1 (the agent's seat is side 0 in both)
    for deep_side, match in ((0, _match(pool, deep, leaf)), (1, _match(pool, leaf, deep))):
        agent, person, kwargs = _calls(monkeypatch, match, pool)[0]
        by_seat = {kwargs["agent_side"]: kwargs["entry"], 1 - kwargs["agent_side"]: person.entry}
        deep_entry, leaf_entry = by_seat[deep_side], by_seat[1 - deep_side]
        assert "selection[" in deep_entry.model and "selection[" not in leaf_entry.model
        # the belief about side s is held by seat 1 - s and comes from that seat's own solve
        for side in (0, 1):
            expected = BenchPrior.of(by_seat[1 - side], side, species[side],
                                     epsilon=humanplay.BELIEF_EPSILON, temperature=1.0)
            assert kwargs["priors"][side] == expected
        assert agent.selection_reading == (READING if kwargs["agent_side"] == deep_side else None)
        distinct.append(not np.array_equal(deep_entry.our_strategy, leaf_entry.our_strategy))
    # the control that the seats can differ: the two solves are not one
    assert any(distinct)


def test_a_condition_names_its_selection() -> None:
    got = parse_condition("x:seconds=1,selection=stage=d2r4b3k8;rects=8")
    assert got.selection == "stage=d2r4b3k8;rects=8" and "selection read" in got.describe()
    with pytest.raises(ValueError):
        parse_condition("x:seconds=1,selection=shift=bad")


# ------------------------------------------------------------------------ a person's game


def _play_human_tool():  # noqa: ANN202
    from tests._harness import load_tool

    return load_tool("play_human")


def _args(**kwargs):  # noqa: ANN003, ANN202
    from types import SimpleNamespace

    base = {"selection_reading": None, "selection_seconds": None, "clock": "wall"}
    return SimpleNamespace(**{**base, **kwargs})


def test_a_persons_game_reads_the_selection_deeper_unless_told_not_to() -> None:
    tool = _play_human_tool()
    leaf = _stub
    assert tool.resolve_selection(_args(), leaf) == ("default", humanplay.PLAY_SELECTION_SECONDS)
    assert humanplay.PLAY_SELECTION_SECONDS == 90.0
    # the leaf's solve: told so, on the count clock (a replay), and without a leaf
    assert tool.resolve_selection(_args(selection_reading="none"), leaf) == (None, None)
    assert tool.resolve_selection(_args(clock="count"), leaf) == (None, None)
    assert tool.resolve_selection(_args(), None) == (None, None)
    # named: its own seconds, or its stages alone on the count clock; a leaf is required
    assert tool.resolve_selection(_args(selection_reading="rects=4", selection_seconds=30.0),
                                  leaf) == ("rects=4", 30.0)
    assert tool.resolve_selection(_args(selection_reading="rects=4", clock="count"),
                                  leaf) == ("rects=4", None)
    with pytest.raises(SystemExit):
        tool.resolve_selection(_args(selection_reading="default"), None)
    with pytest.raises(ValueError):
        tool.resolve_selection(_args(selection_reading="shift=bad"), leaf)


def _agent(pool, **kwargs):  # noqa: ANN001, ANN003, ANN202
    settings = {"seconds": 0.4, "cores": 1, "clock": "count", "rank_by_leaf": False,
                "rank_fill": "refs2"}
    settings.update(kwargs)
    return humanplay.Agent(reg=pool.reg, evaluate=_stub, name="hp-share", **settings)


def _events(pool, agent):  # noqa: ANN001, ANN202
    seen: list[str] = []
    payload, clock, _game = humanplay.play(
        agent, humanplay.PolicyPerson("first"), (pool.teams[0], pool.teams[1]), agent_side=1,
        seed=2, max_turns=1, listener=lambda kind, _data: seen.append(kind))
    return seen, payload, clock


def test_the_screen_is_told_the_selection_is_being_read(pool) -> None:  # noqa: ANN001
    seen, _payload, clock = _events(pool, _agent(pool))
    assert seen[:2] == ["sheets", "select"] and "selecting" not in seen
    assert "selectionRead" not in clock
    seen, payload, clock = _events(pool, _agent(pool, selection_reading=READING))
    # the teams first, then the reading (so the page is not blank), then the person's turn
    # and the person's turn at once: both choose in the same time (the reading ends with
    # `selected`, which does not say what the AI chose)
    assert seen[:4] == ["sheets", "selecting", "select", "selected"] and seen.count("sheets") == 1
    read = clock["selectionRead"]
    assert read["cells"] > 0 and read["reading"] == READING and read["steps"]
    assert payload["picks"]  # a game was played from the deeper solve


def test_the_person_picks_while_the_ai_reads_the_selection(pool) -> None:  # noqa: ANN001
    import threading

    done = threading.Event()
    asked: dict[str, bool] = {}

    class Waits(humanplay.PolicyPerson):
        def select(self, six, size, text):  # noqa: ANN001, ANN202
            # sequential play would ask only after the reading, when `selected` had been heard
            asked["before_done"] = not done.is_set()
            asked["reading_finished_while_asking"] = done.wait(30)
            return super().select(six, size, text)

    def listen(kind, _data):  # noqa: ANN001, ANN202
        if kind == "selected":
            done.set()

    agent = _agent(pool, selection_reading=READING)
    payload, _clock, _game = humanplay.play(
        agent, Waits("first"), (pool.teams[0], pool.teams[1]), agent_side=1, seed=2,
        max_turns=1, listener=listen)
    assert asked == {"before_done": True, "reading_finished_while_asking": True}
    assert payload["human"]["inputs"][0] == "1 2 3 4"


def test_a_failing_person_stops_the_game_after_the_reading(pool) -> None:  # noqa: ANN001
    class Broken(humanplay.PolicyPerson):
        def select(self, six, size, text):  # noqa: ANN001, ANN202
            raise ValueError("no such selection")

    with pytest.raises(ValueError, match="no such selection"):
        humanplay.play(
            _agent(pool, selection_reading=READING), Broken("first"),
            (pool.teams[0], pool.teams[1]), agent_side=1, seed=2, max_turns=1,
            listener=lambda *_a: None)


def test_the_selection_workers_are_made_for_the_selection_and_closed_after_it(  # noqa: ANN201
    pool, monkeypatch,  # noqa: ANN001
):
    log: list[str] = []

    class Fake:
        def __init__(self, reg, count, spec, *, rank_fill, rank_by_leaf):  # noqa: ANN001
            log.append(f"made {count}")
            self.inner = selection_deep.SerialReader(reg, _stub, rank_fill=rank_fill,
                                                     rank_by_leaf=rank_by_leaf)
            self.cells = self.seconds = 0

        def read(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return self.inner.read(*args, **kwargs)

        def close(self):  # noqa: ANN202
            log.append("closed")

    monkeypatch.setattr(selection_deep, "PoolReader", Fake)
    lazy = selection_deep.LazyPoolReader(pool.reg, 3, (), rank_fill="refs2", rank_by_leaf=False)
    monkeypatch.setattr(selection_deep, "READER", lazy)
    seen: list[str] = []

    def listener(kind, _data):  # noqa: ANN001, ANN202
        seen.append(kind)
        if kind == "select":
            log.append("person asked")
        if kind == "selected":
            log.append("reading done")

    humanplay.play(_agent(pool, selection_reading=READING), humanplay.PolicyPerson("first"),
                   (pool.teams[0], pool.teams[1]), agent_side=1, seed=2, max_turns=1,
                   listener=listener)
    # the person is asked first (the reading runs beside them); the workers are made when the
    # first cell is read and closed before the reading is announced done (so before any move)
    assert log == ["person asked", "made 3", "closed", "reading done"]
    assert lazy.inner is None
