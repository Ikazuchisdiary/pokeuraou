"""IKA-458 stage 3.5 (`prepGame`): a game's preparation made by the port, against
`poolplay.generate_pool`'s own (`draw_pair`, `BookEntry.draw`, `BenchPrior.of`,
`position_from_sets`).

What these hold, on a pool of three variants of our M-B roster (the leaf that solves the
selection is `test_poolplay`'s antisymmetric stub):

- **the same preparation**: the pair and its seats, both sides' four, the selection's policies and
  mixtures, the value, the bench priors, the start position (leads' switch-ins applied) and the
  generator's state after all of it (numpy's `bit_generator.state`, 32-bit buffer included) equal
  the Python road's, over three settings of the exploration share;
- **the positive controls**: the port prepared the games (`portlp.COUNTS["prepared"]`), a pair
  with no file was solved here and asked again, and a played game used the game the port defined;
- **the control that can fail**: another seed is other pairs and other fours;
- a store of another tag stops the run, and a game played from a prepared start is the game the
  Python road plays, byte for byte.

No server and no torch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokeuraou import portlp, prepgame, rustnode
from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import HP_SHARE
from pokeuraou.pool import draw_pair, load_pool
from pokeuraou.poolplay import SolvedSelections, generate_pool
from pokeuraou.selection_book import BenchPrior
from pokeuraou.selfplay import position_from_sets

from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture(scope="module")
def store(pool, tmp_path_factory):  # noqa: ANN001, ANN201
    """Every pair of the pool solved and written."""
    path = tmp_path_factory.mktemp("store")
    solver = SolvedSelections(pool.reg, pool.teams, _stub, store=path, tag="t")
    for lo, hi in pool.pairs:
        solver.canonical(lo, hi)
    return path


@pytest.fixture
def on():  # noqa: ANN201
    was = (portlp.PREP[0], portlp.HIDDEN[0], portlp.ADVANCE[0], portlp.DECIDE[0], portlp.MENUS[0],
           portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0],
           portlp.GEN[0])
    rustnode.hold_positions(True)
    portlp.set_prep(True)
    yield
    (portlp.PREP[0], portlp.HIDDEN[0], portlp.ADVANCE[0], portlp.DECIDE[0], portlp.MENUS[0],
     portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0],
     portlp.GEN[0]) = was


def _solver(pool, store):  # noqa: ANN001, ANN202
    return SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="t")


def _python(pool, solver, seed, index, eps, temp):  # noqa: ANN001, ANN202
    rng = np.random.default_rng([seed, index])
    k, a, b = draw_pair(rng, pool.pairs)
    entry = solver.entry(a, b)
    drawn = entry.draw(rng, epsilon=eps, temperature=temp)
    species = ([s.species for s in pool.teams[a].sets], [s.species for s in pool.teams[b].sets])
    priors = (
        BenchPrior.of(entry, 0, species[0], epsilon=eps, temperature=temp),
        BenchPrior.of(entry, 1, species[1], epsilon=eps, temperature=temp),
    )
    own = [pool.teams[a].sets[i] for i in drawn.our_pick]
    foe = [pool.teams[b].sets[i] for i in drawn.foe_pick]
    start = position_from_sets(pool.reg, own, foe, rng=rng)
    return k, a, b, drawn, priors, start, rng


@pytest.mark.parametrize(("eps", "temp"), [(0.25, 0.5), (0.0, 1.0), (0.5, float("inf"))])
def test_the_preparation_is_pythons(pool, store, on, eps, temp) -> None:  # noqa: ANN001
    solver = _solver(pool, store)
    before = portlp.COUNTS["prepared"]
    seen_pairs = set()
    inexact = 0
    for index in range(36):
        want = _python(pool, solver, 45801, index, eps, temp)
        k, a, b, drawn, priors, start, rng = want
        got = prepgame.prepare(
            pool.reg, pool, solver, seed=45801, index=index, pairs=None,
            epsilon=eps, temperature=temp,
        )
        assert got is not None
        assert (got.pair, got.teams, got.mirror) == (k, (a, b), a == b)
        assert got.picks == (tuple(drawn.our_pick), tuple(drawn.foe_pick))
        assert got.value == drawn.value
        assert got.own_policy == [float(x) for x in drawn.our_equilibrium]
        assert got.foe_policy == [float(x) for x in drawn.foe_equilibrium]
        mixtures = (got.own_mixture == [float(x) for x in drawn.our_mixture],
                    got.foe_mixture == [float(x) for x in drawn.foe_mixture])
        inexact += not all(mixtures)
        for mixture, expected in ((got.own_mixture, drawn.our_mixture),
                                  (got.foe_mixture, drawn.foe_mixture)):
            assert np.allclose(mixture, expected, rtol=0, atol=1e-15)
        # The priors are built only if asked, and are the Python road's.
        assert got.priors is not None
        for side in (0, 1):
            assert got.priors[side].selections == priors[side].selections
            assert got.priors[side].species == priors[side].species
            assert np.allclose(got.priors[side].probabilities, priors[side].probabilities,
                               rtol=0, atol=1e-15)
        assert got.position.to_json() == start.to_json()
        state = rng.bit_generator.state
        assert got.rng_state["state"] == state["state"]
        assert got.rng_state["has_uint32"] == state["has_uint32"]
        assert got.rng_state["uinteger"] == state["uinteger"]
        seen_pairs.add((a, b))
    # Positive controls: the port made them, over every ordered pair of the three teams.
    assert portlp.COUNTS["prepared"] - before == 36
    assert len(seen_pairs) >= 6, seen_pairs
    # With an exploration share the mixtures go through `exp`, which this machine's numpy and the
    # port's libm differ on in the last bit for about 1 value in 200 (measured over 200,000
    # arguments: 977 differ, by one ulp): most 90-vectors then differ somewhere, within 1e-15.
    # Without one nothing calls `exp` and every mixture is to the bit.
    assert inexact == 0 or eps > 0.0, inexact


def test_the_mixtures_are_bit_exact_without_exploration(pool, store, on) -> None:  # noqa: ANN001
    """With no exploration share nothing calls `exp`, so the sums and divisions alone are held to the bit."""
    solver = _solver(pool, store)
    for index in range(20):
        _, _, _, drawn, _, _, _ = _python(pool, solver, 7, index, 0.0, 1.0)
        got = prepgame.prepare(
            pool.reg, pool, solver, seed=7, index=index, pairs=None,
            epsilon=0.0, temperature=1.0,
        )
        assert got.own_mixture == [float(x) for x in drawn.our_mixture]
        assert got.foe_mixture == [float(x) for x in drawn.foe_mixture]
        assert got.game is not None and got.priors is not None


def test_another_seed_is_other_games(pool, store, on) -> None:  # noqa: ANN001
    solver = _solver(pool, store)

    def run(seed):  # noqa: ANN001, ANN202
        return [
            (g.teams, g.picks)
            for g in (
                prepgame.prepare(pool.reg, pool, solver, seed=seed, index=i, pairs=None,
                                 epsilon=0.25, temperature=0.5)
                for i in range(12)
            )
        ]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_a_pair_with_no_file_is_solved_here_and_asked_again(pool, tmp_path, on) -> None:  # noqa: ANN001
    solver = SolvedSelections(pool.reg, pool.teams, _stub, store=tmp_path, tag="t")
    seeded = _solver(pool, tmp_path)
    got = prepgame.prepare(
        pool.reg, pool, solver, seed=3, index=0, pairs=None,
        epsilon=0.25, temperature=0.5,
    )
    assert got is not None and solver.solves == 1
    lo, hi = min(got.teams), max(got.teams)
    assert (lo, hi) in solver.solved_pairs and (tmp_path / f"{lo:03d}-{hi:03d}.json").exists()
    # The same game again finds the file: nothing is solved, and the pair counts as reused.
    again = prepgame.prepare(
        pool.reg, pool, solver, seed=3, index=0, pairs=None,
        epsilon=0.25, temperature=0.5,
    )
    assert solver.solves == 1 and again.picks == got.picks and solver.reused >= 1
    # A second process's solver reads what the first wrote.
    other = prepgame.prepare(
        pool.reg, pool, seeded, seed=3, index=0, pairs=None,
        epsilon=0.25, temperature=0.5,
    )
    assert seeded.solves == 0 and other.picks == got.picks


def test_a_pair_another_worker_solved_is_loaded_once(pool, store, tmp_path, on) -> None:  # noqa: ANN001
    """The counters `generate_pool` reports: a file read for the first time is `loaded`, after that
    `reused`; a store directory no process has read is a new key to the port."""
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(store, copy)
    solver = _solver(pool, copy)
    metted = set()
    for index in range(12):
        got = prepgame.prepare(
            pool.reg, pool, solver, seed=5, index=index, pairs=None,
            epsilon=0.25, temperature=0.5,
        )
        metted.add((min(got.teams), max(got.teams)))
    assert solver.solves == 0
    assert solver.loaded == len(metted) and solver.loaded + solver.reused == 12


def test_a_store_of_another_tag_stops_the_run(pool, store, on) -> None:  # noqa: ANN001
    other = SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="another leaf")
    with pytest.raises(ValueError, match="another leaf"):
        prepgame.prepare(
            pool.reg, pool, other, seed=3, index=0, pairs=None,
            epsilon=0.25, temperature=0.5,
        )


def test_a_subset_of_pairs_is_drawn_from_and_keeps_the_pools_index(pool, store, on) -> None:  # noqa: ANN001
    solver = _solver(pool, store)
    subset = (pool.pairs[1], pool.pairs[4])
    for index in range(10):
        rng = np.random.default_rng([9, index])
        _k, a, b = draw_pair(rng, subset)
        want = pool.pairs.index((min(a, b), max(a, b)))
        got = prepgame.prepare(
            pool.reg, pool, solver, seed=9, index=index, pairs=subset,
            epsilon=0.25, temperature=0.5,
        )
        assert (got.pair, got.teams) == (want, (a, b))


def _records(path):  # noqa: ANN001, ANN202
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for key in ("searchSeconds",):
            record.pop(key, None)
        out.append(record)
    return out


def test_a_game_from_a_prepared_start_is_the_python_roads_game(pool, store, tmp_path, on) -> None:  # noqa: ANN001
    """The real thing: games at width 2 played out, with the port's preparation and without."""
    runs = []
    for name, prep in (("py", False), ("port", True)):
        portlp.PREP[0] = prep
        solver = _solver(pool, store)
        out = tmp_path / f"{name}.jsonl"
        before = (portlp.COUNTS["prepared"], portlp.COUNTS["hidden"])
        generate_pool(
            pool.reg, pool, games=0, seed=81, out=out, solver=solver,
            objective=HP_SHARE, search_limit=2, max_turns=40, indices=list(range(6)),
        )
        runs.append((_records(out), portlp.COUNTS["prepared"] - before[0],
                     portlp.COUNTS["hidden"] - before[1]))
    (python, p_prepared, _), (port, prepared, hidden) = runs
    assert p_prepared == 0 and prepared == 6 and hidden > 0
    assert len(python) >= 4, "too few games finished; the comparison would be vacuous"
    # The record's four selection mixtures go through the port's `exp` (see above); they are held to
    # 1e-15 and everything else to the byte, the decisions the game made included.
    mixtures = ("ownSelectionMixture", "foeSelectionMixture")
    for made, expected in zip(port, python, strict=True):
        for key in mixtures:
            assert np.allclose(made.pop(key), expected.pop(key), rtol=0, atol=1e-15)
    assert port == python
