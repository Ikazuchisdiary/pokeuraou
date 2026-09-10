"""Imports the damage differential harness from tools/ for the test suite."""

from __future__ import annotations

from ._harness import load_tool

_MODULE = load_tool("diff_damage")

Report = _MODULE.Report
run = _MODULE.run
compare_turn = _MODULE.compare_turn
desplit = _MODULE.desplit
first_move_hits = _MODULE.first_move_hits
