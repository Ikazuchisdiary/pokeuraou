"""IKA-442: `tools/matchup_proxy.py` scores the selection value against played pairs."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from pokeuraou.regulation import repo_root


@pytest.fixture
def tool(monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.syspath_prepend(str(repo_root() / "tools"))
    return importlib.import_module("matchup_proxy")


def _line(t0: str, t1: str, outcome: float | None, value: float | None, index: int,
          mirror: bool = False) -> str:
    return json.dumps({"outcome": outcome, "selectionValue": value, "gameIndex": index,
                       "decisions": [], "pool": {"teams": [t0, t1], "mirror": mirror}})


def _synthetic(rng: np.random.Generator, teams: int, games: int, noise: float
               ) -> tuple[list[str], np.ndarray]:
    """Games between teams of known strength; the recorded value is the true chance plus
    ``noise``, and the seats are dealt both ways."""
    strength = rng.normal(0, 0.5, teams)
    lines = []
    for g in range(games):
        a, b = rng.choice(teams, 2, replace=False)
        p = 1 / (1 + np.exp(-(strength[a] - strength[b])))
        v = float(np.clip(p + rng.normal(0, noise), 0.01, 0.99))
        y = float(rng.random() < p)
        lines.append(_line(f"t{a:02d}", f"t{b:02d}", y, v, g))
    return lines, strength


def _report(tool, tmp_path, lines: list[str], *extra: str) -> dict:  # noqa: ANN001
    games = tmp_path / "games"
    games.mkdir(exist_ok=True)
    (games / "a.jsonl").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    rows = tmp_path / "rows.npz"
    tool.main(["extract", "--games-dir", str(games), "--out", str(rows)])
    out = tmp_path / "report.json"
    tool.main(["report", "--rows", str(rows), "--out", str(out), "--min-games", "5", *extra])
    return json.loads(out.read_bytes())


def test_mirrors_unfinished_and_valueless_games_are_dropped(tool) -> None:  # noqa: ANN001
    rows, dropped = tool.rows_of_lines([
        _line("a", "b", 1.0, 0.6, 0),
        _line("a", "a", 1.0, 0.5, 1, mirror=True),
        _line("a", "b", None, 0.6, 2),
        _line("b", "a", 0.0, None, 3),
        "",
    ])
    assert rows == [("a", "b", 1.0, 0.6, 0)]
    assert dropped == {"mirror": 1, "unfinished": 1, "no_value": 1, "no_pool": 0}


def test_a_game_seen_from_the_other_seat_is_the_same_game(tool) -> None:  # noqa: ANN001
    lo, hi, y, v = tool.canonical(np.array([0, 1]), np.array([1, 0]),
                                  np.array([1.0, 0.0]), np.array([0.7, 0.3]))
    assert lo.tolist() == [0, 0] and hi.tolist() == [1, 1]
    assert y.tolist() == [1.0, 1.0] and v.tolist() == pytest.approx([0.7, 0.7])


def test_an_exact_value_reads_as_calibrated_and_noise_explains_the_rest(tool, tmp_path) -> None:  # noqa: ANN001
    lines, strength = _synthetic(np.random.default_rng(0), teams=12, games=30_000, noise=0.0)
    result = _report(tool, tmp_path, lines)
    pair = result["pair"]
    # The value is the true chance, so all of the scores' true spread is predicted: the
    # corrected correlation is ~1 and the slope ~1.
    assert pair["pearsonCorrected"] == pytest.approx(1.0, abs=0.08)
    assert pair["slope"] == pytest.approx(1.0, abs=0.1)
    # The forecast beats the coin, and is no worse than the other half's pair means.
    brier = result["game"]["brier"]
    assert brier["value"] < brier["coin"]
    assert brier["value"] <= brier["otherHalf"] + 1e-3
    team = result["team"]
    assert team["spearman"] > 0.9
    assert team["topOverlap"]["5"] >= 4
    # The team table follows the true strengths.
    order = [int(row["team"][1:]) for row in result["teamTable"]]
    assert order[0] == int(np.argmax(strength))


def test_an_uninformative_value_predicts_nothing(tool, tmp_path) -> None:  # noqa: ANN001
    rng = np.random.default_rng(1)
    lines, _ = _synthetic(rng, teams=12, games=20_000, noise=0.0)
    shuffled = []
    for line in lines:
        game = json.loads(line)
        game["selectionValue"] = float(rng.uniform(0.2, 0.8))
        shuffled.append(json.dumps(game))
    result = _report(tool, tmp_path, shuffled)
    assert abs(result["pair"]["pearson"]) < 0.25
    assert result["game"]["brier"]["value"] > result["game"]["brier"]["otherHalf"]


def test_a_store_replaces_the_recorded_value(tool, tmp_path) -> None:  # noqa: ANN001
    lines = [_line("a", "b", 1.0, 0.5, i) for i in range(10)]
    lines += [_line("b", "a", 0.0, 0.5, 10 + i) for i in range(10)]
    store = tmp_path / "store"
    store.mkdir()
    # The store names the pair as (b, a): its value is b's, so a's is 1 - 0.2.
    (store / "000-001.json").write_bytes(json.dumps(
        {"pair": [0, 1], "teams": ["b", "a"], "entry": {"value": 0.2}}).encode("utf-8"))
    result = _report(tool, tmp_path, lines, "--store", str(store))
    assert result["valueSource"] == str(store)
    assert result["game"]["brier"]["value"] == pytest.approx(0.2 ** 2)
