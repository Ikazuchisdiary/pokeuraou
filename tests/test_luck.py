"""IKA-193 stage 3: the worker's own AIVAT terms (`pokeuraou.luck`).

Three things are checked against something other than themselves. The worker's chance
term is `tools/aivat.py`'s after-the-fact one, game for game, and its action term is the
same after-the-fact computation priced at the matrix budget (what one leaf's one matrix
holds on the true position). The games a worker plays with the ledger on are the games it
plays with it off -- the hooks draw nothing. And the corrected test fed pair by pair is the
path `aivat.llr_normal_path` gives the whole sequence, burn-in included.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokeuraou import luck
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import HP_SHARE
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import play_game
from pokeuraou.teams import load_roster
from tests._harness import load_tool

aivat = load_tool("aivat")


@pytest.mark.parametrize("seed", range(10))
def test_the_action_term_averages_to_zero_over_the_two_mixtures(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = [f"a{i}" for i in range(int(rng.integers(2, 6)))]
    cols = [f"b{j}" for j in range(int(rng.integers(2, 6)))]
    own = rng.random(len(rows)) * (rng.random(len(rows)) < 0.7)
    foe = rng.random(len(cols)) * (rng.random(len(cols)) < 0.7)
    own[0] = foe[0] = 0.3  # a support on both sides
    # Two matrices, one of them missing a column: Q is their mean where both hold a pair.
    m0 = rng.random((len(rows), len(cols)))
    m1 = rng.random((len(rows), len(cols) - 1))
    matrices = [(rows, cols, m0), (rows, cols[:-1], m1)]
    total = weight = 0.0
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            if own[i] * foe[j] <= 0:
                continue
            status, c, _ = luck.action_correction(rows, own, cols, foe, (a, b), matrices)
            if status == "pure":
                return
            assert status == "corrected"
            total += own[i] * foe[j] * c
            weight += own[i] * foe[j]
    assert abs(total / weight) < 1e-12


def test_a_pair_no_matrix_holds_leaves_the_decision_uncorrected() -> None:
    rows, cols = ["a", "b"], ["x", "y"]
    status, c, pairs = luck.action_correction(
        rows, [0.5, 0.5], cols, [0.5, 0.5], ("a", "x"), [(rows, ["x"], np.ones((2, 1)))]
    )
    assert (status, c, pairs) == ("uncovered", 0.0, 4)
    status, c, _ = luck.action_correction(
        rows, [1.0, 0.0], cols, [0.0, 1.0], ("a", "y"), [(rows, cols, np.eye(2))]
    )
    assert (status, c) == ("pure", 0.0)


def test_the_mean_of_two_matrices_is_what_is_read() -> None:
    rows, cols = ["a", "b"], ["x", "y"]
    m0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    m1 = np.array([[0.0, 0.0], [0.0, 0.0]])
    _, c, _ = luck.action_correction(
        rows, [0.5, 0.5], cols, [0.5, 0.5], ("a", "x"), [(rows, cols, m0), (rows, cols, m1)]
    )
    # Q(a, x) = 0.5, the rest 0; the mixtures' mean of Q is 0.125.
    assert c == pytest.approx(0.5 - 0.125)


def test_the_corrected_test_fed_pair_by_pair_is_the_whole_paths() -> None:
    rng = np.random.default_rng(3)
    scores = np.clip(rng.normal(0.52, 0.2, size=200), 0, 1)
    path = aivat.llr_normal_path(scores, 0.0, 10.0, burn=luck.BURN_IN)
    test = luck.NormalTest(0.0, 10.0)
    for k, x in enumerate(scores):
        test.add(float(x))
        assert test.llr == pytest.approx(path[k], rel=1e-9, abs=1e-12)
    assert np.all(path[: luck.BURN_IN] == 0.0)
    assert np.any(path[luck.BURN_IN :] != 0.0)


def _line(index: int, seat: int, outcome: float, chance: float, action: float) -> bytes:
    game = {
        "outcome": outcome,
        "decisions": [],
        "provenance": {"seat": f"arm = side {seat}"},
        "gameIndex": index,
        "seatIndex": seat,
        "aivat": {"C": chance, "A": action, "seconds": 0.1, "gameSeconds": 1.0},
    }
    return json.dumps(game).encode("utf-8")


def test_one_game_in_both_seats_reads_a_half_after_both_terms(tmp_path) -> None:  # noqa: ANN001
    lines = [_line(0, 0, 1.0, 0.3, -0.1), _line(0, 1, 1.0, 0.3, -0.1), _line(1, 0, 0.0, 0.2, 0.05)]
    (tmp_path / "games-worker0.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    pairs = luck.CorrectedPairs(tmp_path)
    assert pairs.poll() == 3
    indices, raw, corrected = pairs.complete()
    assert list(indices) == [0]
    assert raw[0] == 0.5
    assert corrected[0] == pytest.approx(0.5, abs=1e-15)
    got = luck.seat_of_line(_line(4, 1, 1.0, 0.3, -0.1))
    assert got is not None and got[:4] == (4, 1, 0.0, pytest.approx(0.2))


@pytest.fixture(scope="module")
def ledgered():  # noqa: ANN201
    """Two games in the open game and one under a hidden bench, each played with the ledger
    and again without it. The open games give both seats one leaf object, so one matrix."""
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    leaf = HP_SHARE.batch
    out = []
    # Incineroar's Parting Shot on both sides: seeds 0 and 13 stop a turn mid-way.
    for seed, hidden in ((0, False), (13, False), (0, True)):
        def play(seed: int = seed, hidden: bool = hidden):  # noqa: ANN202
            return play_game(
                reg, np.random.default_rng(seed), [sheet[i] for i in (5, 0, 1, 2)],
                [sheet[i] for i in (5, 1, 3, 4)],
                "test", search_limit=4, max_turns=40, evaluate=leaf,
                sheets=(sheet, sheet) if hidden else None, open_information=not hidden,
            )

        with luck.collecting(leaf, "hp-share") as ledger:
            record = play()
        plain = play()
        dump = [
            json.loads(json.dumps(r.to_json(objective="hp-share", search_limit=4)))
            for r in (record, plain)
        ]
        for game in dump:
            game.pop("searchSeconds")
        out.append((hidden, dump[0], dump[1], ledger.to_json()))
    return reg, leaf, out


def test_the_ledger_changes_no_game(ledgered) -> None:  # noqa: ANN001
    _reg, _leaf, games = ledgered
    for _hidden, with_ledger, without, _block in games:
        assert with_ledger == without
    # The control: two different seeds are two different games.
    assert games[0][1] != games[1][1]


def test_the_workers_chance_term_is_the_after_the_fact_one(ledgered) -> None:  # noqa: ANN001
    reg, leaf, games = ledgered
    resumed = sum(1 for g in games for s in g[3]["stages"] if s["st"] > 0)
    assert resumed, "no turn stopped mid-way in the fixture games"
    for _hidden, game, _plain, block in games:
        terms = aivat.game_terms(reg, game, leaf)
        assert {s["s"] for s in block["stages"]} <= {"matched", "paused"}
        assert len(block["stages"]) == len(terms.stages)
        for mine, theirs in zip(block["stages"], terms.stages, strict=True):
            assert (mine["d"], mine["st"]) == (theirs.decision, theirs.stage)
            assert mine["c"] == pytest.approx(theirs.c, abs=1e-12)
        assert block["C"] == pytest.approx(terms.total, abs=1e-12)
        assert block["turns"] == sum(1 for d in game["decisions"] if d["kind"] == "move")


def test_the_workers_action_term_is_the_matrix_priced_one(ledgered) -> None:  # noqa: ANN001
    reg, leaf, games = ledgered
    corrected = 0
    for hidden, game, _plain, block in games:
        if hidden:
            # The hidden bench's matrices are its completions': read, not recomputable.
            assert {a["s"] for a in block["actions"]} <= {"pure", "corrected", "uncovered"}
            continue
        after = aivat.action_terms(reg, game, leaf, budget=Budget.matrix(), kind="matrix")
        assert [a["s"] for a in block["actions"]] == [a.status for a in after]
        for mine, theirs in zip(block["actions"], after, strict=True):
            assert mine["c"] == pytest.approx(theirs.c, abs=1e-12)
        corrected += sum(1 for a in after if a.status == "corrected")
    assert corrected, "no mixed decision in the open fixture games"
