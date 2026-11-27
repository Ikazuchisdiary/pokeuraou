"""IKA-464: the port's deepening (`portdeepen`, `rust/src/deepen.rs`).

A move read on counted cells in the port: the root is the depth-1 node the port already
answers, the read spends its budget, the answer moves off the depth-1 answer, the same
budget reads the same answer twice, and the swap oracle adds actions from outside the menu.
The leaf and the Q are stand-ins on the test server (no GPU), built so their numbers are the
same bits on every machine (`tests/_exact_standins.py`): a read's steps follow every last bit
of them, and CI on Linux took other steps with stand-ins read through ``sin`` and BLAS."""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import humanplay, portdeepen, qrank
from pokeuraou.budget import Budget
from pokeuraou.encode import Encoder
from pokeuraou.narrow import narrow
from pokeuraou.selfplay import position_from_sets

from ._exact_standins import ExactQStub, exact_leaf
from .test_humanplay import pool  # noqa: F401 - the fixture


@pytest.fixture
def served(pool):  # noqa: ANN001, ANN201, F811
    from pokeuraou.inference import RemoteValue
    from pokeuraou.standin import serve

    encoder = Encoder(pool.reg)
    stub = ExactQStub(encoder.vocab.fingerprint())
    server, address = serve({"value": exact_leaf}, arms={"value": ["exact.pt"]},
                            q_models={"qstub": stub})
    leaf = RemoteValue(address, "value", Encoder(pool.reg))
    model = qrank.RemoteQ(address, "qstub", encoder)
    saved = dict(qrank._INSTALLED)  # noqa: SLF001
    qrank.install(model)
    yield leaf
    qrank._INSTALLED.clear()  # noqa: SLF001
    qrank._INSTALLED.update(saved)  # noqa: SLF001
    model.close()
    leaf.close()
    server.shutdown()
    server.server_close()


def _positions(pool, count: int, seed: int = 5) -> list:  # noqa: ANN001, F811
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        four = [list(pool.teams[s].sets)[:4] for s in (0, 1)]
        pos = position_from_sets(pool.reg, four[0], four[1], rng=rng)
        for side in pos.sides:
            for mon in side.pokemon:
                if mon.active_index is not None:
                    mon.hp = max(1, int(mon.maxhp * rng.uniform(0.3, 1.0)))
        out.append(pos)
    return out


def _read(pool, pos, me, leaf, cells, **kwargs):  # noqa: ANN001, ANN003, ANN202, F811
    ours = narrow(pool.reg, pos, 0, limit=6).actions
    theirs = narrow(pool.reg, pos, 1, limit=6).actions
    return humanplay.solve_move(pool.reg, pos, me, list(ours), list(theirs), None, leaf,
                                budget=Budget.matrix(), exact=True, cells=cells,
                                **kwargs)


def test_the_port_reads_a_move_deeper_on_counted_cells(pool, served) -> None:  # noqa: ANN001, F811
    reads = portdeepen.COUNTS["reads"]
    moved = 0
    for pos in _positions(pool, 3):
        for me in (0, 1):
            depth1 = _read(pool, pos, me, served, 0)
            deep = _read(pool, pos, me, served, 400, levels=4)
            again = _read(pool, pos, me, served, 400, levels=4)
            report = deep.deepened
            assert report.cells >= 400 and report.expanded > 0 and report.depth >= 2
            assert report.child_passes > 0
            np.testing.assert_array_equal(deep.strategy, again.strategy)
            assert deep.value == again.value
            moved += abs(deep.value - depth1.value) > 1e-6
    assert portdeepen.COUNTS["reads"] == reads + 12
    assert moved >= 3, "the deepened value never left the depth-1 value"


def test_the_swap_oracle_brings_in_actions_from_outside(pool, served) -> None:  # noqa: ANN001, F811
    from pokeuraou.actions import side_actions

    widened = 0
    for pos in _positions(pool, 3, seed=9):
        outside = (side_actions(pool.reg, pos, 0), side_actions(pool.reg, pos, 1))
        for me in (0, 1):
            got = _read(pool, pos, me, served, 300, levels=4, outside=outside)
            assert got.deepened.oracle and got.deepened.probed > 0
            widened += got.deepened.widened
    assert widened > 0, "the oracle never added an action"


def _python_deepening() -> dict:
    """Python's deepening's answers on these positions, frozen at 887df85 before IKA-464 removed
    it (`tests/fixtures/ika464-python-deepening.json`; its children by the Q's 8, ``c8``)."""
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "ika464-python-deepening.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_port_s_tree_is_the_python_deepening_with_the_q_s_children(pool, served) -> None:  # noqa: ANN001, F811
    """The port's read is `deepen.py`'s with the children by the Q's 8 (``c8``): the same steps,
    the same depth, the same value and the same mixture as Python's deepening gave on the same
    reads before IKA-464 removed it (the frozen answers). Without the oracle, so the steps are
    the deepening's alone; at both seats, on mixed answers."""
    want = {(r["position"], r["me"]): r for r in _python_deepening()["tree"]}
    mixed = 0
    for p, pos in enumerate(_positions(pool, 4, seed=11)):
        for me in (0, 1):
            port = _read(pool, pos, me, served, 600, levels=4)
            python = want[p, me]
            assert port.deepened.expanded == python["expanded"] > 0, (p, me)
            assert port.deepened.depth == python["depth"]
            assert port.value == pytest.approx(python["value"], abs=1e-9)
            np.testing.assert_allclose(port.strategy, python["strategy"], atol=1e-7)
            mixed += float(np.max(port.strategy)) < 1 - 1e-6
    assert len(want) == 8 and mixed >= 1, "no mixed answer"


def test_a_read_widened_mid_way_keeps_its_tree(pool, served) -> None:  # noqa: ANN001, F811
    """IKA-354's `grow` on the port's road (the analysis mode widens the root while it reads):
    the actions asked for join the root once, the read goes on, and the report counts them."""
    from pokeuraou.actions import side_actions

    pos = _positions(pool, 1, seed=13)[0]
    ours = list(narrow(pool.reg, pos, 0, limit=4).actions)
    theirs = list(narrow(pool.reg, pos, 1, limit=4).actions)
    extra = [a for a in side_actions(pool.reg, pos, 0) if a not in ours][:2]
    asked: list[float] = []

    def grow(spent: float):  # noqa: ANN202
        asked.append(spent)
        return (extra, []) if len(asked) == 2 else None

    got = humanplay.solve_move(pool.reg, pos, 0, ours, theirs, None, served, budget=Budget.matrix(),
                               exact=True, cells=3000, levels=4, grow=grow)
    assert len(asked) >= 2 and got.deepened.grown == len(extra) == 2
    assert got.deepened.grown_cells == 2 * len(theirs)
    assert len(got.ours) == len(ours) + 2 and got.deepened.expanded > 0
    plain = humanplay.solve_move(pool.reg, pos, 0, ours, theirs, None, served,
                                 budget=Budget.matrix(), exact=True, cells=3000, levels=4)
    assert plain.deepened.grown == 0 and len(plain.ours) == len(ours)


def _move(pool, pos, served, cells, cost, **kwargs):  # noqa: ANN001, ANN003, ANN202, F811
    ours = list(narrow(pool.reg, pos, 0, limit=6).actions)
    theirs = list(narrow(pool.reg, pos, 1, limit=6).actions)
    return humanplay.solve_move(pool.reg, pos, 0, ours, theirs, None, served,
                                budget=Budget.matrix(), exact=True, cells=cells, cost=cost,
                                levels=4, **kwargs)


def test_a_priced_budget_is_read_as_the_python_deepening_reads_it(pool, served) -> None:  # noqa: ANN001, F811
    """A count clock's budget priced by `deepen.COSTS` (a person's count-clock game, the tests'
    agents): the port reads the cost's units before every step, the levels and the root's cells
    at each refinement included, as `deepen.py` read them (``c8``, the frozen answers). After
    every step the two reads share, the cells and the reading of the budget are the same to the
    bit and the root's value is the same to rounding; only the last step passes the budget.

    The frozen answers also hold, for each of Python's steps, the priorities of the cell it
    refined and of the best cell after it. Two cells can tie to the last bit (position 1, step
    19: two cells of level 1 at 233.47166917005143, both refinements of 193 cells), and which
    of the two each side takes is the last bit of a sum in Rust or in numpy. So the two reads
    may part, at any step, only where Python's best two were tied there; the step is not fixed.
    A read that parts at a step with no tie, or that reads the budget otherwise, fails here."""
    from pokeuraou.deepen import COSTS, Cost

    cost = COSTS["local", 1]
    assert cost.level and cost.root, "the prices this test is about"
    budget = int(400.0 / cost.cell)  # 400 ms of priced work, in cells
    want = {r["position"]: r for r in _python_deepening()["priced"]}
    compared = 0
    for p, pos in enumerate(_positions(pool, 3, seed=17)):
        steps: list = []
        port = _move(pool, pos, served, budget, cost, progress=steps.append)
        got = [(s.cells, s.spent, s.root.equilibrium.value) for s in steps if s.kind == "refine"]
        spent = [v for _cells, v, _value in got]
        assert spent and spent[-1] >= budget and all(v < budget for v in spent[:-1])
        python = want[p]["steps"]
        same = 0
        while same < min(len(got), len(python)):
            cells, units, value = got[same]
            if cells != python[same][0] or abs(value - python[same][2]) > 1e-9:
                break
            assert units == python[same][1], (p, same)
            same += 1
        compared += same
        if same == len(got) == len(python):
            assert port.deepened.expanded == want[p]["expanded"] > 0, p
            continue
        # Parted at step `same + 1`: only where Python's best two cells were tied.
        assert same < len(python), (p, same, "the port read on past Python's last step")
        assert same < len(got), (p, same, "the port stopped before Python's last step")
        first, second = python[same][3], python[same][4]
        assert first == second, (p, same + 1, first, second)
    assert compared >= 60, compared
    # The control that the port reads the level and root prices at all: a hundred times
    # dearer, the same budget buys fewer steps.
    dear = Cost(fill=cost.fill, refine=cost.refine, cell=cost.cell, level=cost.level * 100,
                root=cost.root * 100)
    pos = _positions(pool, 1, seed=17)[0]
    priced = _move(pool, pos, served, budget, cost)
    dearer = _move(pool, pos, served, budget, dear)
    assert 0 < dearer.deepened.expanded < priced.deepened.expanded


def test_a_read_is_let_go_however_it_ends(pool, served) -> None:  # noqa: ANN001, F811
    """`deepenEnd` follows every `deepenStart`: after a read, and after a progress callback that
    raised mid-read or at the read's first report (``start``, before any slice), the port holds
    no read (`deepenCounts.held`)."""
    from pokeuraou import port as port_mod

    def counts() -> dict:
        return port_mod.ask(pool.reg, lambda node: node._exchange({"kind": "deepenCounts"}))  # noqa: SLF001

    pos = _positions(pool, 1, seed=19)[0]
    _read(pool, pos, 0, served, 300, levels=4)
    assert int(counts()["held"]) == 0

    for kind in ("refine", "start"):
        started = int(counts()["started"])

        def boom(step, kind=kind) -> None:  # noqa: ANN001
            if step.kind == kind:
                raise RuntimeError("the page went away")

        with pytest.raises(RuntimeError, match="went away"):
            _read(pool, pos, 0, served, 300, levels=4, progress=boom)
        after = counts()
        assert int(after["started"]) == started + 1, kind
        assert int(after["held"]) == 0, f"a read raised at {kind!r} was held"


def test_the_lines_are_walked_now_and_then_and_at_the_end(pool, served, monkeypatch) -> None:  # noqa: ANN001, F811
    """The port walks its tree for the lines (the guard's count) at most every
    `LINES_EVERY_MS`; a read whose last slice did not walk asks for one walk at the end, so
    the report's lines are the tree's as it ended: the same lines as a read that walked at
    every slice."""
    pos = _positions(pool, 1, seed=23)[0]

    def read(every: float) -> tuple:
        monkeypatch.setattr(portdeepen, "LINES_EVERY_MS", every)
        fresh: list = []
        walks = portdeepen.COUNTS["walks"]
        got = _read(pool, pos, 0, served, 600, levels=2,
                    progress=lambda s: fresh.append(s.root.port.get("linesFresh")))
        return got.deepened, fresh, portdeepen.COUNTS["walks"] - walks

    seldom, fresh, walks = read(1e9)
    always, fresh_always, walks_always = read(0.0)
    # The positive control: slices answered the last walk's lines, and the end walked once.
    assert False in fresh and walks == 1
    assert all(f is not False for f in fresh_always) and walks_always == 0
    assert seldom.lines == always.lines and sum(seldom.lines) > 0
    assert seldom.expanded == always.expanded and seldom.cells == always.cells
