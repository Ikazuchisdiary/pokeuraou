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
- **the memo changes no game**: the same games with the memo on and off, with the memo
  shown to have fired;
- **a seed is a set of games**: the same seed and indices replay the same records; a
  different seed does not (the positive control);
- IKA-128: the pool path hides the bench without a flag and gives both seats a belief
  from the solve.

The pool here is three variants of our M-B roster written in the pool file's shape, so no
fetched data is needed. The leaf that solves the selection is an antisymmetric stub; the
games themselves are played with hp-share at width 2 to stay cheap.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import poolplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import HP_SHARE
from pokeuraou.pool import draw_pair, load_pool, unordered_pairs
from pokeuraou.poolplay import SolvedSelections, generate_pool
from pokeuraou.regulation import repo_root
from pokeuraou.selection_book import BenchPrior
from pokeuraou.selfplay import GameRecord
from pokeuraou.teams import TeamError

from ._harness import load_tool


def _stub(positions):  # noqa: ANN001, ANN202
    """Antisymmetric: HP summed per side, the active pair counted twice (leads matter)."""
    out = np.empty(len(positions), dtype=np.float64)
    for i, pos in enumerate(positions):
        strength = [
            sum(m.hp * (2 if m.active_index is not None else 1) for m in side.pokemon)
            for side in pos.sides
        ]
        out[i] = 1.0 / (1.0 + np.exp(-(strength[0] - strength[1]) / 150.0))
    return out


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


def test_a_solved_selection_needs_a_leaf(pool, tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="needs a leaf"):
        generate_pool(pool.reg, pool, games=1, hide_bench=True, out=tmp_path / "g.jsonl")


# ------------------------------------------------------------------------ the games


def _fake_play(calls: list[dict]):  # noqa: ANN202
    """A `play_game` that records its inputs and the generator's state, and plays nothing.

    A game is a function of these inputs and of the generator, so two runs that hand it
    the same ones play the same game; the one real replay is
    `test_a_seed_and_its_indices_are_the_same_games_when_played`.
    """

    def fake(reg, rng, own, foe, label, **kwargs):  # noqa: ANN001, ANN003, ANN202
        priors = kwargs["bench_prior"]
        calls.append({
            "own": [(s.species, tuple(sorted(s.sp.items()))) for s in own],
            "foe": [(s.species, tuple(sorted(s.sp.items()))) for s in foe],
            "label": label,
            "selection": kwargs["selection"],
            "sheets": kwargs["sheets"] is not None,
            "open": kwargs["open_information"],
            "priors": None if priors is None else [
                None if p is None else (p.species, p.probabilities) for p in priors
            ],
            "state": rng.bit_generator.state["state"],
        })
        record = GameRecord(own_team=[], foe_team=[], foe_archetype=label)
        record.outcome = float(rng.random() < 0.5)
        record.turns = 1
        return record

    return fake


def _games(pool, out: Path, *, seed: int, indices, monkeypatch, memo: bool = True,  # noqa: ANN001
           hide_bench: bool = True, solver: SolvedSelections | None = None):  # noqa: ANN202
    calls: list[dict] = []
    monkeypatch.setattr(poolplay, "play_game", _fake_play(calls))
    solver = solver or SolvedSelections(pool.reg, pool.teams, _stub, memo=memo)
    stats = generate_pool(
        pool.reg, pool, games=0, hide_bench=hide_bench, seed=seed, out=out,
        solver=solver, objective=HP_SHARE, indices=list(indices),
    )
    records = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line
    ]
    return stats, solver, records, calls


@pytest.fixture(scope="module")
def shared(pool):  # noqa: ANN001, ANN201
    """One solver for the tests that do not ask about the memo: a solve is seconds."""
    return SolvedSelections(pool.reg, pool.teams, _stub)


def test_a_seed_and_its_indices_are_the_same_games_when_played(pool, tmp_path) -> None:  # noqa: ANN001
    """The real thing, twice: two games at width 2, played out, byte for byte."""
    runs = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.jsonl"
        solver = SolvedSelections(pool.reg, pool.teams, _stub)
        generate_pool(
            pool.reg, pool, games=0, hide_bench=True, seed=81, out=out, solver=solver,
            objective=HP_SHARE, search_limit=2, max_turns=40, indices=[0, 1],
        )
        records = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln]
        for record in records:
            record.pop("searchSeconds")
        runs.append(records)
    assert runs[0], "no game finished; the comparison would be vacuous"
    assert runs[0] == runs[1]


def test_a_seed_and_its_indices_are_the_same_inputs(pool, shared, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    _, _, first, calls = _games(
        pool, tmp_path / "a.jsonl", seed=81, indices=range(6), monkeypatch=monkeypatch,
        solver=shared,
    )
    _, _, again, calls_again = _games(
        pool, tmp_path / "b.jsonl", seed=81, indices=range(6), monkeypatch=monkeypatch,
        solver=shared,
    )
    assert len(first) == 6
    assert first == again and calls == calls_again
    # Positive control: another seed is other games.
    _, _, other, calls_other = _games(
        pool, tmp_path / "c.jsonl", seed=82, indices=range(6), monkeypatch=monkeypatch,
        solver=shared,
    )
    assert calls_other != calls
    # And an index is its own game, whichever order the queue deals them in.
    _, _, _, backwards = _games(
        pool, tmp_path / "d.jsonl", seed=81, indices=reversed(range(6)),
        monkeypatch=monkeypatch, solver=shared,
    )
    assert backwards == calls[::-1]


def test_the_memo_changes_no_game(pool, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    kept_stats, kept, with_memo, calls = _games(
        pool, tmp_path / "m.jsonl", seed=5, indices=range(10), monkeypatch=monkeypatch
    )
    _, fresh, without, calls_fresh = _games(
        pool, tmp_path / "n.jsonl", seed=5, indices=range(10), monkeypatch=monkeypatch,
        memo=False,
    )
    assert kept.reused > 0, "no pair came twice, so the memo was never read"
    assert kept.solves + kept.reused == kept_stats["games"] == 10
    assert fresh.reused == 0 and fresh.solves == 10
    assert with_memo == without
    assert calls == calls_fresh


def test_the_store_shares_the_solves_between_workers_and_changes_no_game(
    pool, tmp_path, monkeypatch  # noqa: ANN001
) -> None:
    store = tmp_path / "store"
    first = SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="t")
    _, _, _, calls_first = _games(
        pool, tmp_path / "w0.jsonl", seed=11, indices=range(6), monkeypatch=monkeypatch,
        solver=first,
    )
    assert first.solves > 0 and first.loaded == 0
    # A second worker with its own process memory: everything it meets is on disk.
    second = SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="t")
    _, _, with_store, calls_second = _games(
        pool, tmp_path / "w1.jsonl", seed=11, indices=range(6), monkeypatch=monkeypatch,
        solver=second,
    )
    assert second.solves == 0 and second.loaded == first.solves
    _, _, without, calls_fresh = _games(
        pool, tmp_path / "w2.jsonl", seed=11, indices=range(6), monkeypatch=monkeypatch,
        memo=False,
    )
    assert calls_second == calls_fresh == calls_first
    assert with_store == without


def test_a_store_solved_for_another_leaf_stops_the_run(pool, tmp_path) -> None:  # noqa: ANN001
    store = tmp_path / "store"
    SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="leaf A").entry(0, 1)
    other = SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="leaf B")
    with pytest.raises(ValueError, match="another pool or another leaf"):
        other.entry(1, 0)


def test_the_record_says_solved_and_carries_the_pair(pool, shared, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    _, _, records, calls = _games(
        pool, tmp_path / "r.jsonl", seed=7, indices=range(8), monkeypatch=monkeypatch,
        solver=shared,
    )
    assert len(records) == 8
    for record, call in zip(records, calls, strict=True):
        assert record["selectionSource"] == "solved"
        assert record["pool"]["benchPrior"] == ["solved", "solved"]
        assert record["pool"]["id"] == "test-pool"
        a, b = record["pool"]["teams"]
        assert record["pool"]["mirror"] == (a == b)
        assert len(record["ownSelectionPolicy"]) == 90
        assert sum(record["foeSelectionMixture"]) == pytest.approx(1.0)
        assert call["label"] == ("mirror" if a == b else "pool")
        # The six each seat played are its pool team's, spreads included.
        six = {t.id: t for t in pool.teams}
        assert list(call["selection"][0]) == [s.species for s in six[a].sets]
        assert list(call["selection"][1]) == [s.species for s in six[b].sets]


def test_both_seats_get_a_belief_from_the_solve(pool, shared, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """IKA-128: hidden, with both sides' bench priors from the pair's solve."""
    _, _, _, calls = _games(
        pool, tmp_path / "s.jsonl", seed=9, indices=range(4), monkeypatch=monkeypatch,
        solver=shared,
    )
    for call in calls:
        assert call["sheets"] and call["open"] is False
        assert call["priors"] is not None and all(p is not None for p in call["priors"])
        own_six, foe_six = call["selection"][0], call["selection"][1]
        assert list(call["priors"][0][0]) == own_six
        assert list(call["priors"][1][0]) == foe_six


def test_the_uniform_reference_names_itself(pool, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    calls: list[dict] = []
    monkeypatch.setattr(poolplay, "play_game", _fake_play(calls))
    out = tmp_path / "u.jsonl"
    generate_pool(
        pool.reg, pool, games=3, hide_bench=True, seed=1, out=out, selection="uniform",
    )
    records = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln]
    assert [r["selectionSource"] for r in records] == ["uniform"] * 3
    assert all(r["pool"]["benchPrior"] == ["uniform", "uniform"] for r in records)
    assert all(c["priors"] is None for c in calls)


def test_mirror_games_draw_both_seats_from_one_mixture(mirror_pool, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    stats, _, records, _ = _games(
        mirror_pool, tmp_path / "mm.jsonl", seed=3, indices=range(4),
        monkeypatch=monkeypatch,
    )
    assert stats["mirror_games"] == stats["games"] == 4
    assert len(records) == 4
    for record in records:
        assert record["selectionValue"] == pytest.approx(0.5, abs=1e-9)
        assert record["ownSelectionMixture"] == record["foeSelectionMixture"]
        assert record["ownSelectionPolicy"] == record["foeSelectionPolicy"]


# ---------------------------------------------------------------------------- the CLI


def test_the_pool_cli_hides_the_bench_without_a_flag(pool, monkeypatch) -> None:  # noqa: ANN001
    tool = load_tool("selfplay")
    import pokeuraou.pool as pool_module

    called: dict = {}

    def fake_generate(reg, pool_arg, **kwargs):  # noqa: ANN001, ANN003, ANN202
        called.update(kwargs)
        return {"games": 0, "finished": 0, "discarded_unfinished": 0, "wins": 0,
                "decisions": 0, "turns": 0, "mirror_games": 0, "mirror_wins": 0,
                "mirror_draws": 0, "path": "x"}

    monkeypatch.setattr(pool_module, "load_pool", lambda name: pool)
    monkeypatch.setattr(poolplay, "generate_pool", fake_generate)
    monkeypatch.setattr(tool, "build_leaf", lambda args, reg: (_stub, "value:stub"))
    monkeypatch.setattr("sys.argv", ["selfplay.py", "--pool", "p", "--games", "1"])
    tool.main()
    assert called["hide_bench"] is True
    assert called["selection"] == "solved"

    called.clear()
    monkeypatch.setattr(
        "sys.argv", ["selfplay.py", "--pool", "p", "--games", "1", "--open-bench"]
    )
    tool.main()
    assert called["hide_bench"] is False, "the open reference is still there by name"


def test_the_pool_cli_refuses_the_roster_path_flags(monkeypatch) -> None:  # noqa: ANN001
    tool = load_tool("selfplay")
    monkeypatch.setattr(
        "sys.argv", ["selfplay.py", "--pool", "p", "--roster", "rizabanadohido"]
    )
    with pytest.raises(SystemExit):
        tool.main()
