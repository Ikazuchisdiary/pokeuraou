"""IKA-424: `tools/depth_outcome.py next` finds the move decision after a position."""

from __future__ import annotations

import importlib

import pytest

from pokeuraou.regulation import repo_root


@pytest.fixture
def tool(monkeypatch):  # noqa: ANN001, ANN201
    # The tool sets this default for its own reads at import; here it is undone after the test.
    monkeypatch.setenv("POKEURAOU_LADDER_COUNT_FILL", "0")
    monkeypatch.syspath_prepend(str(repo_root() / "tools"))
    return importlib.import_module("depth_outcome")


def _decision(kind: str, turn: int) -> dict:
    return {"kind": kind, "turn": turn, "position": {"turn": turn, "kind": kind}}


def test_the_next_move_decision_skips_replacements_and_ends_with_the_game(tool) -> None:  # noqa: ANN001
    game = [_decision("move", 1), _decision("replacement", 1), _decision("move", 2),
            _decision("move", 3), _decision("replacement", 3)]
    at, nxt = tool.following(game, game[0]["position"])
    assert at == 0
    assert nxt is game[2]  # the replacement between is not a move
    at, nxt = tool.following(game, game[2]["position"])
    assert (at, nxt) == (2, game[3])
    # The game's last move decision has none after it (only a replacement follows).
    assert tool.following(game, game[3]["position"]) == (3, None)
    # A position that is not one of the game's move decisions is refused, and a replacement's
    # position is not a move decision.
    with pytest.raises(ValueError):
        tool.following(game, {"turn": 9})
    with pytest.raises(ValueError):
        tool.following(game, game[1]["position"])


def test_the_game_index_is_read_from_the_lines_text(tool) -> None:  # noqa: ANN001
    line = b'{"ownTeam": [], "gameIndex": 23472, "decisions": []}'
    assert int(tool.GAME_INDEX.search(line).group(1)) == 23472
