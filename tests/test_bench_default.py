"""No information condition is a default any more (IKA-123).

The hidden bench was added after the open game, as `--hide-bench`, so the unmarked command
played the open game -- the easier one, and not the one that ships. Every place that
defaulted to open now asks to be told:

* `play_game` without `sheets` wants `open_information=True`
* `generate` wants `hide_bench`
* every tool that can play both wants `--hide-bench` or `--open-bench`
* `resume_generate.py`, which can only play open, wants `--open-bench`
* `provenance` wants `information`

and the rating's zero moved to the hidden `hp-share`, while every recorded agent keeps the
name it had. Each test below failed, or could not be written, before the change: a call
without the statement ran the open game.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.benchflags import MISSING, add_bench_flags, bench_argv, require_bench
from pokeuraou.provenance import agent_name, provenance
from pokeuraou.selfplay import generate, play_game
from tests._harness import load_tool


def test_play_game_without_sheets_must_say_it_means_the_open_game() -> None:
    # Raised before the teams are read, so nothing here needs a regulation.
    with pytest.raises(ValueError, match="open_information=True"):
        play_game(None, np.random.default_rng(0), [], [], "x")  # type: ignore[arg-type]


def test_sheets_and_open_information_contradict() -> None:
    with pytest.raises(ValueError, match="shows it"):
        play_game(
            None,  # type: ignore[arg-type]
            np.random.default_rng(0), [], [], "x",
            sheets=([], []), open_information=True,
        )


def test_generate_wants_the_bench_named() -> None:
    with pytest.raises(ValueError, match="hide_bench must be given"):
        generate(None, None, None, [], games=1)  # type: ignore[arg-type]


def test_provenance_has_no_default_information() -> None:
    with pytest.raises(TypeError, match="information"):
        provenance("x", seat="s", leaves=("a", "b"), limits=(12, 12))  # type: ignore[call-arg]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_bench_flags(ap)
    return ap


def test_the_flag_pair() -> None:
    assert require_bench(_parser().parse_args(["--hide-bench"])) is True
    assert require_bench(_parser().parse_args(["--open-bench"])) is False
    with pytest.raises(SystemExit) as stopped:
        require_bench(_parser().parse_args([]))
    assert stopped.value.code == MISSING
    # Both at once is argparse's own refusal.
    with pytest.raises(SystemExit):
        _parser().parse_args(["--hide-bench", "--open-bench"])
    assert bench_argv(True) == ["--hide-bench"]
    assert bench_argv(False) == ["--open-bench"]


@pytest.mark.parametrize(
    ("tool", "argv"),
    [
        # The acceptance line of IKA-123: the two queue drivers started with nothing added.
        ("generate_queue", ["--out", "unused", "--games", "1"]),
        ("match_queue", ["--out", "unused", "--games", "1", "--value", "v.pt"]),
        ("selfplay", []),
        ("generation_match", []),
        ("selection_check", []),
        # `branch_dedup` stood here until IKA-212 deleted it with Python's resolver.
        ("profile_stages", ["generation"]),
    ],
)
def test_a_tool_given_neither_flag_stops_and_says_what_to_add(
    monkeypatch: pytest.MonkeyPatch, tool: str, argv: list[str]
) -> None:
    module = load_tool(tool)
    monkeypatch.setattr(sys, "argv", [f"{tool}.py", *argv])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == MISSING


def test_resume_generate_cannot_make_open_teacher_data_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    module = load_tool("resume_generate")
    monkeypatch.setattr(sys, "argv", ["resume_generate.py", "--dir", "nowhere"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert "--open-bench" in str(stopped.value.code)
    # With the statement it goes on, and stops on the next thing: no games under `nowhere`.
    monkeypatch.setattr(sys, "argv", ["resume_generate.py", "--dir", "nowhere", "--open-bench"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert "no recorded games" in str(stopped.value.code)


def test_agent_drift_finds_a_call_that_does_not_say_which_game(tmp_path: Path) -> None:
    drift = load_tool("agent_drift")
    silent = tmp_path / "silent.py"
    silent.write_text(
        "play_game(reg, rng, a, b, 'x', search_limit=12)\n"
        "play_game(reg, rng, a, b, 'x', open_information=True)\n"
        "play_game(reg, rng, a, b, 'x', sheets=s)\n"
        "play_game(reg, rng, a, b, 'x', **kwargs)\n",
        encoding="utf-8",
    )
    assert drift.unstated(silent) == [1]
    # And the tree itself has none: `--check` passes.
    assert drift.main(["--check"]) == 0


def test_the_rating_zero_is_the_hidden_hp_share() -> None:
    ratings = load_tool("ratings")
    assert ratings.ANCHOR == "hp-share/w24/hidden-bench"
    assert ratings.OPEN_ANCHOR == "hp-share/w24"
    assert ratings.is_hidden("value-gen11L/w24/leaf/book:x/hidden-bench")
    # `endswith` read this one as open.
    assert ratings.is_hidden("value-gen11L/w24/leaf/book:x/hidden-bench/belief:book")
    assert not ratings.is_hidden("value-gen11L/w24/leaf/book:x")


@pytest.mark.parametrize(
    ("source", "name"),
    [
        # Written before `information` existed: open, unmarked.
        ({"leaves": ["value-gen9.pt", "hp-share"], "limits": [24, 24]}, "value-gen9/w24"),
        # Open, written with the field.
        ({"leaves": ["hp-share", "b"], "limits": [24, 24], "information": ["open", "open"]},
         "hp-share/w24"),
        # Hidden, uniform belief (every hidden match before IKA-122).
        ({"leaves": ["hp-share", "b"], "limits": [24, 24],
          "information": ["hidden-bench", "hidden-bench"]},
         "hp-share/w24/hidden-bench"),
        # Hidden, book belief.
        ({"leaves": ["value-gen11L", "b"], "limits": [24, 24], "rankings": ["leaf", "leaf"],
          "books": ["riza-value-gen11L", "x"], "information": ["hidden-bench", "open"],
          "beliefs": ["book", "uniform"]},
         "value-gen11L/w24/leaf/book:riza-value-gen11L/hidden-bench/belief:book"),
    ],
)
def test_recorded_names_do_not_move(source: dict, name: str) -> None:
    """Open is still the unmarked name: renaming it would split every recorded agent from
    its own games, the way `.pt` and `uniform-against-` once did."""
    assert agent_name(source, 0) == name
