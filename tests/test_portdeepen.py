"""IKA-464: the port's deepening (`portdeepen`, `rust/src/deepen.rs`).

A move read on counted cells in the port: the root is the depth-1 node the port already
answers, the read spends its budget, the answer moves off the depth-1 answer, the same
budget reads the same answer twice, and the swap oracle adds actions from outside the menu.
The leaf and the Q are stand-ins on the test server (no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import humanplay, portdeepen, qrank
from pokeuraou.budget import Budget
from pokeuraou.encode import Encoder
from pokeuraou.narrow import narrow
from pokeuraou.selfplay import position_from_sets

from .test_humanplay import _noisy, pool  # noqa: F401 - the fixture
from .test_port_menus import QStub


@pytest.fixture
def served(pool):  # noqa: ANN001, ANN201, F811
    from pokeuraou.inference import RemoteValue
    from pokeuraou.standin import serve

    encoder = Encoder(pool.reg)
    stub = QStub(encoder.vocab.fingerprint())
    server, address = serve({"value": _noisy}, arms={"value": ["noisy.pt"]},
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
                                budget=Budget.matrix(), exact=True, cells=cells, road="port",
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


def test_the_port_s_tree_is_the_python_deepening_with_the_q_s_children(pool, served) -> None:  # noqa: ANN001, F811
    """The port's read is `deepen.py`'s with the children by the Q's 8 (``c8``): the same steps,
    the same value and the same mixture where both took the same number of steps. Without the
    oracle, so the steps are the deepening's alone; at both seats, on mixed answers."""
    same = 0
    mixed = 0
    for pos in _positions(pool, 4, seed=11):
        for me in (0, 1):
            port = _read(pool, pos, me, served, 600, levels=4)
            python = humanplay.solve_move(
                pool.reg, pos, me, list(narrow(pool.reg, pos, 0, limit=6).actions),
                list(narrow(pool.reg, pos, 1, limit=6).actions), None, served,
                budget=Budget.matrix(), exact=True, cells=600, levels=4, child_q=8, road="python")
            assert port.deepened.expanded > 0
            if port.deepened.expanded != python.deepened.expanded:
                continue
            same += 1
            assert port.deepened.depth == python.deepened.depth
            assert port.value == pytest.approx(python.value, abs=1e-9)
            np.testing.assert_allclose(port.strategy, python.strategy, atol=1e-7)
            mixed += float(np.max(port.strategy)) < 1 - 1e-6
    assert same == 8, f"only {same} of 8 reads took the same steps"
    assert mixed >= 1, "no mixed answer"


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
                               exact=True, cells=3000, levels=4, road="port", grow=grow)
    assert len(asked) >= 2 and got.deepened.grown == len(extra) == 2
    assert got.deepened.grown_cells == 2 * len(theirs)
    assert len(got.ours) == len(ours) + 2 and got.deepened.expanded > 0
    plain = humanplay.solve_move(pool.reg, pos, 0, ours, theirs, None, served,
                                 budget=Budget.matrix(), exact=True, cells=3000, levels=4, road="port")
    assert plain.deepened.grown == 0 and len(plain.ours) == len(ours)
