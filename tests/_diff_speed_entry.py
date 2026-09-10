"""Imports the Speed differential harness from tools/ for the test suite."""

from __future__ import annotations

from ._harness import load_tool

_MODULE = load_tool("diff_speed")

Report = _MODULE.Report
run = _MODULE.run
