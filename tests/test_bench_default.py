"""The bench is hidden in every game (IKA-123, IKA-460).

The hidden bench was added after the open game, as `--hide-bench`, so for months the unmarked
command played the open game -- the easier one, and not the one that ships. IKA-123 made the
caller name the condition. IKA-460 removed the open game, so there is nothing left to name:

* `play_game` requires `sheets` (the two sixes) and has no `open_information`
* `--hide-bench` is the default of every tool that took it, and `--open-bench` stops
* `provenance` wants `information` (recorded open games are still read)

and the rating's zero is the hidden `hp-share`, while every recorded agent keeps the name it
had, so an old open record still reads under the name it was written with.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pytest

from pokeuraou.benchflags import OPEN_REMOVED, add_bench_flags, bench_argv, require_bench
from pokeuraou.provenance import agent_name, provenance
from pokeuraou.selfplay import play_game
from tests._harness import load_tool


def test_play_game_has_no_open_game() -> None:
    parameters = inspect.signature(play_game).parameters
    assert "open_information" not in parameters
    assert parameters["sheets"].default is inspect.Parameter.empty


def test_provenance_has_no_default_information() -> None:
    with pytest.raises(TypeError, match="information"):
        provenance("x", seat="s", leaves=("a", "b"), limits=(12, 12))  # type: ignore[call-arg]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_bench_flags(ap)
    return ap


def test_the_bench_is_hidden_by_default_and_open_stops(capsys: pytest.CaptureFixture[str]) -> None:
    assert require_bench(_parser().parse_args([])) is True
    assert _parser().parse_args(["--hide-bench"]).hide_bench is True
    assert _parser().parse_args([]).hide_bench is True
    assert bench_argv(True) == ["--hide-bench"]
    with pytest.raises(SystemExit):
        _parser().parse_args(["--open-bench"])
    assert OPEN_REMOVED.split(":")[0] in capsys.readouterr().err


def test_agent_drift_finds_a_call_that_does_not_name_the_sheets(tmp_path: Path) -> None:
    drift = load_tool("agent_drift")
    silent = tmp_path / "silent.py"
    silent.write_text(
        "play_game(reg, rng, a, b, 'x', search_limit=12)\n"
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
