"""IKA-81: both seats from one pool, and the selection solved where each game starts.

M-C has no own side, so a game is a pair of pool teams, and the selection of that pair is
solved with the leaf at the start of the game (the user's decision of 9/24 on IKA-77) and
kept per pair. What these tests hold:

- **the pair distribution is the one the issue names**: uniform over the ``n(n+1)/2``
  unordered pairs with repetition -- 2,145 for 65, of which 65 mirrors -- and a fair coin
  for the seats, so over the enumeration every team is at seat 0 exactly as often as at
  seat 1;
- **a true mirror is exactly 50%**: its selection game has value 0.5 and both seats draw
  from the same mixture, so the two seats are exchangeable (not only equal in value);
- **a store solved for another leaf stops the run**, and the pair filters leave the pairs
  they name.

The games themselves are the Rust worker's since IKA-458 stage 3.7: `test_rust_worker.py`
holds that a seed is a set of games, that the store changes no game, the record's selection
fields and the mirror's one mixture.

The pool here is three variants of our M-B roster written in the pool file's shape, so no
fetched data is needed. The leaf that solves the selection is an antisymmetric stub.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import draw_pair, load_pool, unordered_pairs
from pokeuraou.poolplay import SolvedSelections
from pokeuraou.regulation import load_regulation, repo_root
from pokeuraou.selection_book import BenchPrior
from pokeuraou.standin import RowStub
from pokeuraou.teams import TeamError

from ._harness import load_tool
from ._served import hp_share_leaf


def _hp_leads(rows):  # noqa: ANN001, ANN202
    strength = (rows.hp * np.where(rows.active, 2, 1)).sum(axis=-1)
    return 1.0 / (1.0 + np.exp(-(strength[:, 0] - strength[:, 1]) / 150.0))


#: Antisymmetric: HP summed per side, the active pair counted twice (leads matter). What a
#: selection solve reads: the port scores it on a stand-in inference server
#: (`standin.RowStub`, IKA-463). A test may also call it on positions. A game's nodes are
#: scored on the tests' hp-share server: `_served_stub`.
_stub = RowStub(load_regulation("gen9championsvgc2026regmb"), _hp_leads)


#: An arm's leaf in a played game: the tests' hp-share server (antisymmetric too).
_served_stub = hp_share_leaf()


def _roster_team() -> list[dict]:
    path = repo_root() / "configs" / "teams" / "rizabanadohido.json"
    return json.loads(path.read_text(encoding="utf-8"))["team"]


def _write_pool(path: Path, teams: list[list[dict]]) -> Path:
    data = {
        "id": "test-pool",
        "name": "test",
        "regulation": "gen9championsvgc2026regmb",
        "character": "a test pool",
        "validatedAgainst": {"formatId": "gen9championsvgc2026regmb"},
        "teams": [
            {"id": f"t{i}", "name": f"team {i}", "team": team} for i, team in enumerate(teams)
        ],
    }
    path.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    return path


def _variants() -> list[list[dict]]:
    base = _roster_team()
    rotated = copy.deepcopy(base[2:] + base[:2])
    respread = copy.deepcopy(base)
    for member in respread:
        member["sp"] = {"hp": 32, "spe": 32, "atk": 2}
    return [base, rotated, respread]


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture(scope="module")
def mirror_pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("mirror") / "p.json", [_roster_team()])
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


# ------------------------------------------------------------------------- the pair


def test_sixty_five_teams_make_2145_pairs_of_which_65_mirrors() -> None:
    pairs = unordered_pairs(65)
    assert len(pairs) == 2145 == 65 * 66 // 2
    assert len(set(pairs)) == 2145
    assert sum(a == b for a, b in pairs) == 65


def test_every_team_sits_at_each_seat_equally_often_over_the_enumeration() -> None:
    """Pair x coin, enumerated: seat 0 and seat 1 counts are equal per team, exactly."""
    n = 7
    seat0 = [0] * n
    seat1 = [0] * n
    for a, b in unordered_pairs(n):
        for swap in (False, True):
            x, y = (b, a) if swap else (a, b)
            seat0[x] += 1
            seat1[y] += 1
    assert seat0 == seat1


def test_the_pair_draw_takes_two_numbers_whatever_the_pair() -> None:
    """A mirror draws its coin too, so what follows the pair does not depend on it."""
    pairs = unordered_pairs(3)
    for seed in range(40):
        rng = np.random.default_rng(seed)
        draw_pair(rng, pairs)
        after = rng.random()
        twin = np.random.default_rng(seed)
        twin.integers(len(pairs))
        twin.random()
        assert after == twin.random()


def test_the_pair_draw_reaches_mirrors_at_their_share() -> None:
    pairs = unordered_pairs(65)
    rng = np.random.default_rng(81)
    draws = [draw_pair(rng, pairs) for _ in range(20_000)]
    mirrors = sum(a == b for _, a, b in draws) / len(draws)
    assert abs(mirrors - 65 / 2145) < 0.004  # 1/65 = 0.0154 would be ordered pairs


def test_a_pool_team_that_fails_the_roster_checks_stops_the_load(tmp_path) -> None:  # noqa: ANN001
    teams = _variants()
    teams[1][2]["nature"] = None
    path = _write_pool(tmp_path / "bad.json", teams)
    with pytest.raises(TeamError, match=r"team 1 \(t1\).*nature"):
        load_pool(path)


# ------------------------------------------------------------- the solved selection


def test_a_true_mirror_is_exactly_even_and_its_seats_exchangeable(mirror_pool) -> None:  # noqa: ANN001
    solver = SolvedSelections(mirror_pool.reg, mirror_pool.teams, _stub)
    entry = solver.entry(0, 0)
    assert entry.value == pytest.approx(0.5, abs=1e-9)
    assert np.array_equal(entry.their_strategies[0], entry.our_strategy)
    species = [s.species for s in mirror_pool.teams[0].sets]
    assert (
        BenchPrior.of(entry, 0, species).probabilities
        == BenchPrior.of(entry, 1, species).probabilities
    )
    # And the stub is not flat, so the equilibrium is a real one and not an LP tie.
    assert float(np.max(entry.our_ev_loss)) > 0.01


def test_the_other_seating_is_the_same_solve_transposed(pool) -> None:  # noqa: ANN001
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    ab = solver.entry(0, 2)
    ba = solver.entry(2, 0)
    assert solver.solves == 1 and solver.reused == 1
    assert np.array_equal(ba.our_strategy, ab.their_strategies[0])
    assert np.array_equal(ba.their_strategies[0], ab.our_strategy)
    assert ba.value == pytest.approx(1.0 - ab.value, abs=1e-12)
    assert [s.species for s in ba.class_sets[0]] == [s.species for s in pool.teams[0].sets]
    # The LP's value is unique, so solving the other chair directly agrees on it.
    direct = SolvedSelections(pool.reg, [pool.teams[2], pool.teams[0]], _stub).entry(0, 1)
    assert direct.value == pytest.approx(ba.value, abs=1e-7)


def test_a_store_solved_for_another_leaf_stops_the_run(pool, tmp_path) -> None:  # noqa: ANN001
    store = tmp_path / "store"
    SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="leaf A").entry(0, 1)
    other = SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="leaf B")
    with pytest.raises(ValueError, match="another pool or another leaf"):
        other.entry(1, 0)


# ------------------------------------------------------------- a subset of the pairs


def test_the_pair_filters_leave_the_pairs_they_name(pool) -> None:  # noqa: ANN001
    """IKA-444: `--pairs-no-mirror` / `--pairs-with` (tools/selfplay.py's `pool_pairs`, which
    the Rust worker's config carries); neither means every pair."""
    tool = load_tool("selfplay")

    def error(message: str) -> None:
        raise SystemExit(message)

    assert tool.pool_pairs(pool, None, False, error) is None
    assert tool.pool_pairs(pool, None, True, error) == [p for p in pool.pairs if p[0] != p[1]]
    species = pool.teams[0].sets[0].species
    assert tool.pool_pairs(pool, [species], False, error) == list(pool.pairs), (
        "every team of this pool holds it"
    )
    with pytest.raises(SystemExit):
        tool.pool_pairs(pool, ["notaspecies"], False, error)


