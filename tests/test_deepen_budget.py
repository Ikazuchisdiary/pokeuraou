"""The deepening's budget (`deepen.Cost`, `COSTS`, `cells_for_seconds`, `_Meter`): the
pieces the port's deepening reads its budget with (`portdeepen`). Kept from the tests of
Python's deepening, which IKA-464 removed (test_deepen, test_deepen_discount,
test_deepen_qprobe)."""

from __future__ import annotations

import threading

import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import humanplay


def test_seconds_turn_into_a_budget_at_measured_prices() -> None:
    local = deepen_mod.COSTS["local", 1]
    assert deepen_mod.cells_for_seconds(0.1) == int(100.0 / local.cell)
    served = deepen_mod.COSTS["served", 1]
    assert deepen_mod.cells_for_seconds(2, form="served") == int(2000.0 / served.cell)
    assert deepen_mod.cells_for_seconds(-1) == 0
    # A price is measured at its core count, never scaled from one core (IKA-32).
    with pytest.raises(ValueError, match="core"):
        deepen_mod.cells_for_seconds(45, cores=2)
    assert local.ms(2, 3, 100) == pytest.approx(2 * local.fill + 3 * local.refine + 100 * local.cell)


def test_a_wrapped_count_clock_keeps_the_level_and_root_prices() -> None:
    local = deepen_mod.COSTS["local", 1]
    bare = deepen_mod._Meter(local)  # noqa: SLF001
    wrapped = deepen_mod._Meter(humanplay.HaltingCost(local, threading.Event(), 10**6))  # noqa: SLF001
    for meter in (bare, wrapped):
        meter.refined(12, 2, 5, 3000)
        meter.refined(3, 1, 11, 3000)
    # The memory brake's wrapper charges what the bare clock charges (IKA-344 found the
    # prices read as 0 through it), and the prices are really there.
    assert wrapped.spent == bare.spent
    assert bare.spent > local.ms(3, 2, 15) / local.cell
    # A wall clock has neither price, wrapped or not.
    wall = deepen_mod._Meter(humanplay.HaltingCost(humanplay.WallCost(0.0), threading.Event(), 5))  # noqa: SLF001
    assert getattr(wall.cost, "level", 0.0) == 0.0 and getattr(wall.cost, "root", 0.0) == 0.0


def test_the_meter_prices_a_refined_cell_by_its_level() -> None:
    plain = deepen_mod.Cost(fill=5.0, refine=1.5, cell=0.1)
    leveled = deepen_mod.Cost(fill=5.0, refine=1.5, cell=0.1, level=0.4, root=0.01)
    a, b = deepen_mod._Meter(plain), deepen_mod._Meter(leveled)  # noqa: SLF001
    for meter in (a, b):
        meter.refined(10, 1, 0, 300)
        meter.refined(4, 1, 7, 300)
        meter.refined(2, 1, 12, 320)
    assert a.levels == b.levels == 19
    assert a.roots == b.roots == 920
    # Without the prices a level and a root cell cost nothing: the old reading, to the bit.
    assert a.spent == plain.ms(3, 3, 16) / 0.1
    assert b.spent == pytest.approx((plain.ms(3, 3, 16) + 0.4 * 19 + 0.01 * 920) / 0.1)
    # The deepening charges both: a root's cells at each refinement, the cell's level.
    local = deepen_mod.COSTS["local", 1]
    assert local.level > 0 and local.root > 0

    # A clock that stands in for a Cost (no `level`) reads as before.
    class Clock:
        cell = 1.0

        def ms(self, fills, refines, cells, probed=0, qs=0):  # noqa: ANN001, ANN202, ARG002
            return 42.0

    c = deepen_mod._Meter(Clock())  # noqa: SLF001
    c.refined(3, 1, 9)
    assert c.spent == 42.0


def test_a_cost_prices_the_probe_apart() -> None:
    plain = deepen_mod.Cost(fill=5.0, refine=1.5, cell=0.1)
    # Without the probe's price a probed cell is any cell and a Q is free: as before.
    assert plain.ms(3, 2, 100, probed=40, qs=7) == plain.ms(3, 2, 100)
    priced = deepen_mod.Cost(fill=5.0, refine=1.5, cell=0.1, probe=0.05, q=5.2)
    assert priced.ms(3, 2, 100, probed=40, qs=7) == pytest.approx(
        3 * 5.0 + 2 * 1.5 + 60 * 0.1 + 40 * 0.05 + 7 * 5.2
    )
    meter = deepen_mod._Meter(priced)  # noqa: SLF001
    meter.fills, meter.cells, meter.probed = 2, 100, 40
    assert meter.spent == pytest.approx(priced.ms(2, 0, 100, 40, 0) / 0.1)
