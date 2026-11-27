"""A match's bench belief belongs to the arm, in both seats (IKA-122).

Generation has handed `play_game` a `bench_prior` since IKA-5: the belief over the hidden
bench is the opponent's selection equilibrium, not a uniform draw over every pair the sheet
allows. `generation_match` never did, so every `--hide-bench` match measured an agent
generation does not ship. IKA-460 removed `generation_match` (M-B's board); what is left
here is what does not need it: `agent_drift` holds `bench_prior` to account, and a record
written before IKA-122 still reads as the uniform belief. The pool match's own bench
belief is the Rust board worker's (`tests/test_rust_board_worker.py`).
"""

from __future__ import annotations

from pokeuraou.provenance import agent_name
from tests._harness import load_tool


def test_agent_drift_counts_the_bench_prior_as_part_of_the_agent() -> None:
    """Generation believes the bench from the solve, so `bench_prior` has to be among the
    arguments a `play_game` call is held to; and with it there, `--check` has to pass on the
    tree. (Generation's own games are the Rust worker's since IKA-458 stage 3.7.)"""
    drift = load_tool("agent_drift")
    assert "bench_prior" in drift.shipping_args()
    assert drift.main(["--check"]) == 0


def test_old_hidden_records_read_as_the_uniform_belief() -> None:
    """Every hidden-bench match before IKA-122 played the uniform belief, so it is named so."""
    source = {
        "leaves": ["a", "b"],
        "limits": [24, 24],
        "books": ["x", "x"],
        "information": ["hidden-bench", "hidden-bench"],
    }
    assert agent_name(source, 0) == "a/w24/book:x/hidden-bench"
    # An open game has no bench to believe anything about, whatever the field says.
    source["information"] = ["open", "open"]
    source["beliefs"] = ["book", "book"]
    assert agent_name(source, 0) == "a/w24/book:x"
