"""`tools/diverge_report.py` carries paused turns on and plays seeds side by side (IKA-227).

Until IKA-227 a turn the port and Showdown both stopped inside at a mid-turn replacement
was set aside unscored, so a port that got the rest of such a turn wrong ranked the same as
one that got it right (the `ika211-control` build, which drops the last action of a resumed
turn, printed a report identical to the real one at 20 seeds x 20 battles). Now each such
turn is resumed with the replacement Showdown's run sent in (`diff_turn.follow_port`) and
scored at its end.

Measured when written, 3 seeds x 4 battles: 5 stops, all 5 carried on and scored, 1 of them
silently divergent (seed 1, a Parting Shot turn with Throat Chop and Solar Beam: hp 89 vs
Showdown's 76, a port divergence of its own). The control build diverges on 3.
"""

from __future__ import annotations

import pytest

from pokeuraou import rustnode
from pokeuraou.oracle import ORACLE_JS

from ._harness import load_tool

diverge_report = load_tool("diverge_report")

pytestmark = pytest.mark.oracle


@pytest.fixture(scope="module")
def one_process():  # noqa: ANN201
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    return diverge_report.run(seeds=3, battles=4, roll=8, max_turns=10, jobs=1)


def test_paused_turns_are_carried_on_and_scored(one_process) -> None:  # noqa: ANN001
    agg = one_process
    assert agg.paused >= 3, agg.render(10, 3)
    assert agg.carried_on == agg.paused, agg.render(10, 3)
    assert not any("stopped at a mid-turn replacement" in k for k in agg.skipped)
    # The one known divergence; the control build (`--features ika211-control`) gives 3.
    assert agg.carried_on_silent + agg.carried_on_flagged <= 1, agg.render(10, 3)


def test_the_report_does_not_depend_on_jobs(one_process) -> None:  # noqa: ANN001
    """Null control for `--jobs`: seeds 1 and 3 in one worker, 2 in the other, merged in
    seed order, give one process's report."""
    several = diverge_report.run(seeds=3, battles=4, roll=8, max_turns=10, jobs=2)
    assert several.render(25, 3) == one_process.render(25, 3)
    assert several.compared == one_process.compared
