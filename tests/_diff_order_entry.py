"""Imports the turn-order differential harness from tools/ for the test suite."""

from __future__ import annotations

from ._harness import load_tool

_MODULE = load_tool("diff_order")

Report = _MODULE.Report
run = _MODULE.run
actual_order = _MODULE.actual_order
desplit = _MODULE.desplit
predicted_order = _MODULE.predicted_order
