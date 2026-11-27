"""A match's bench belief belongs to the arm, in both seats (IKA-122).

Generation has handed `play_game` a `bench_prior` since IKA-5: the belief over the hidden
bench is the opponent's selection equilibrium, not a uniform draw over every pair the sheet
allows. `generation_match` never did, so every `--hide-bench` match measured an agent
generation does not ship. IKA-460 removed `generation_match` (M-B's board); what is left
here is that a record written before IKA-122 still reads as the uniform belief (the check
that `agent_drift` held `play_game`'s `bench_prior` to account went with `play_game`,
IKA-463). The pool match's own bench belief is the Rust board worker's
(`tests/test_rust_board_worker.py`).
"""

from __future__ import annotations

from pokeuraou.provenance import agent_name


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
