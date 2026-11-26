"""IKA-458 stage 3.4 (`hiddenStep`): what each side has shown and the positions its unseen slots
could hold, made by the port, against `hidden.seen_identities`, `hidden.seen_slots`,
`selfplay._bench_weights` and `hidden.completions`.

What these hold, on real positions of games played with random legal actions:

- **the same step**: the identities and slots seen, the species and slots each completion fills,
  its weight (to the bit), its completed position (written out by the port, equal to the Python
  one's JSON) and the `pk` the node request reads, over positions with nothing, some and all of
  a bench hidden;
- **the positive controls**: the steps taken (`portlp.COUNTS["hidden"]`), the completions made,
  and a position the port wrote out once a Python read asked for it;
- **the control that can fail**: another prior weighs the completions otherwise;
- a sheet that cannot explain the board is an `error`, and a game the port was not given is a
  `fault` that leaves the step to Python.

No server and no torch.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pokeuraou import hiddenstep, port, portlp, rustnode
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions, seen_identities, seen_slots, shown_species
from pokeuraou.qhead import legal_pool
from pokeuraou.selection_book import BenchPrior
from pokeuraou.selfplay import GameRecord, _bench_weights, _sample_index, position_from_sets
from pokeuraou.teams import load_roster


@pytest.fixture(scope="module")
def world():  # noqa: ANN201
    """(reg, sheets, priors, [(position, leads, carried seen)]) of positions met playing."""
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    reg = loaded.reg
    sets = list(loaded.sets)
    sheets = (sets[:6], sets[:6][::-1])
    rng = np.random.default_rng(3)
    selections = [s for s in itertools.permutations(range(6), 4)]
    priors = []
    for side in (0, 1):
        raw = rng.random(len(selections)) ** 4
        raw[rng.random(len(selections)) < 0.5] = 0.0
        priors.append(BenchPrior(
            selections=tuple(selections),
            probabilities=tuple(float(p) for p in raw / raw.sum()),
            species=tuple(s.species for s in sheets[side]),
        ))
    positions = []
    rustnode.hold_positions(True)
    for _game in range(16):
        pick = [list(rng.permutation(6)[:4]) for _ in range(2)]
        own = [sheets[0][i] for i in pick[0]]
        foe = [sheets[1][i] for i in pick[1]]
        pos = position_from_sets(reg, own, foe, rng=rng)
        leads = [
            frozenset(shown_species(pos, i, frozenset(
                m.slot for m in pos.sides[i].pokemon if m.active_index is not None)))
            for i in (0, 1)
        ]
        seen = [frozenset(), frozenset()]
        for _turn in range(8):
            if pos.ended:
                break
            seen = [seen_identities(pos, i, seen[i]) for i in (0, 1)]
            positions.append((pos, leads, list(seen)))
            owed = port.replacements_needed(reg, pos)
            if any(owed[0]) or any(owed[1]):
                break
            pools = [legal_pool(reg, pos, s) for s in (0, 1)]
            chosen = [pools[s][int(rng.integers(len(pools[s])))] for s in (0, 1)]
            weights = port.weights(reg, pos, chosen, Budget.exact())
            counts = np.array(weights.branches + weights.suspended, dtype=np.float64)
            index = _sample_index(rng, counts)
            if index >= len(weights.branches):
                break
            pos = port.branch(reg, pos, chosen, Budget.exact(), index)
    return reg, sheets, priors, positions


@pytest.fixture
def on():  # noqa: ANN201
    was = (portlp.HIDDEN[0], portlp.ADVANCE[0], portlp.DECIDE[0], portlp.MENUS[0],
           portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0],
           portlp.FOLD[0], portlp.GEN[0])
    rustnode.hold_positions(True)
    portlp.set_hidden(True)
    yield
    (portlp.HIDDEN[0], portlp.ADVANCE[0], portlp.DECIDE[0], portlp.MENUS[0], portlp.REPLACE[0],
     portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0]) = was


def _python_step(reg, sheets, priors, pos, leads, seen):  # noqa: ANN001, ANN202
    seen_now = [seen_identities(pos, i, seen[i]) for i in (0, 1)]
    shown = [seen_slots(pos, i, seen_now[i]) for i in (0, 1)]
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
    spreads = {
        side: completions(
            reg, pos, side, sheets[side], seen=shown[side],
            weights=_bench_weights(priors, side, pos, shown[side], record, leads[side]),
        )
        for side in (0, 1)
    }
    return seen_now, shown, spreads, record.unmodelled


def test_the_step_is_pythons(world, on) -> None:  # noqa: ANN001
    reg, sheets, priors, positions = world
    game = hiddenstep.Game.of(sheets, priors)
    hidden_sides = exact_sides = 0
    before = portlp.COUNTS["hidden"]
    for pos, leads, seen in positions:
        want_seen, want_shown, want, want_notes = _python_step(
            reg, sheets, priors, pos, leads, seen
        )
        got = hiddenstep.step(reg, pos, game, seen, leads)
        assert got is not None and got.error is None
        assert got.seen == want_seen and got.shown == want_shown
        assert got.recorded == [sorted(s) for s in want_seen]
        assert got.notes == want_notes
        for side in (0, 1):
            assert len(got.spreads[side]) == len(want[side])
            for made, expected in zip(got.spreads[side], want[side], strict=True):
                assert (made.species, made.slots, made.exact) == (
                    expected.species, expected.slots, expected.exact
                )
                assert made.weight == expected.weight
                if expected.exact:
                    exact_sides += 1
                    assert made.position is pos
                    continue
                hidden_sides += 1
                assert made.position.to_json() == expected.position.to_json()
                assert made.pk == tuple(
                    (hiddenstep_id(m.species), hiddenstep_id(m.base_species), m.slot)
                    for m in expected.position.sides[side].pokemon
                )
    assert portlp.COUNTS["hidden"] - before == len(positions) > 30
    # Positive controls: completions were made, and the exact case was met too.
    assert hidden_sides > 30 and exact_sides > 5, (hidden_sides, exact_sides)


def hiddenstep_id(name: str) -> str:
    from pokeuraou.regulation import to_id

    return to_id(name)


def test_a_completion_is_written_out_only_when_read(world, on) -> None:  # noqa: ANN001
    reg, sheets, priors, positions = world
    game = hiddenstep.Game.of(sheets, priors)
    pos, leads, seen = positions[0]
    got = hiddenstep.step(reg, pos, game, seen, leads)
    held = got.spreads[1][0].position
    assert held.__class__ is rustnode.HeldPosition
    before = dict(rustnode.BARE)
    rustnode._position(held)  # noqa: SLF001 - naming it in a request reads nothing
    assert rustnode.BARE["materialized"] == before["materialized"]
    held.sides  # noqa: B018 - a read of it asks the port once
    assert rustnode.BARE["materialized"] == before["materialized"] + 1


def test_another_prior_weighs_otherwise(world, on) -> None:  # noqa: ANN001
    reg, sheets, priors, positions = world
    other = tuple(
        BenchPrior(p.selections, tuple(reversed(p.probabilities)), p.species) for p in priors
    )
    differs = 0
    for pos, leads, seen in positions[:12]:
        a = hiddenstep.step(reg, pos, hiddenstep.Game.of(sheets, priors), seen, leads)
        b = hiddenstep.step(reg, pos, hiddenstep.Game.of(sheets, other), seen, leads)
        for side in (0, 1):
            wa = [i.weight for i in a.spreads[side]]
            wb = [i.weight for i in b.spreads[side]]
            differs += wa != wb
    assert differs > 0


def test_a_game_the_port_lacks_is_a_fault(world, on) -> None:  # noqa: ANN001
    reg, sheets, priors, positions = world
    game = hiddenstep.Game.of(sheets, priors)
    pos, leads, seen = positions[0]
    faults = portlp.COUNTS["faults"]
    # A game whose definition never reaches the port: the step is its fault, left to Python.
    lacking = hiddenstep.Game(10**9, b"")
    assert hiddenstep.step(reg, pos, lacking, seen, leads) is None
    assert portlp.COUNTS["faults"] == faults + 1
    # The same position with its game defined is the port's.
    got = hiddenstep.step(reg, pos, game, seen, leads)
    assert got is not None and portlp.COUNTS["faults"] == faults + 1


def test_a_sheet_that_cannot_explain_the_board_is_an_error(world, on) -> None:  # noqa: ANN001
    reg, sheets, priors, positions = world
    short = (sheets[0][:3], sheets[1])
    game = hiddenstep.Game.of(short, (None, None))
    hit = 0
    for pos, leads, seen in positions:
        try:
            _python_step(reg, short, (None, None), pos, leads, seen)
            want = None
        except ValueError as problem:
            want = str(problem)
        got = hiddenstep.step(reg, pos, game, seen, leads)
        assert got.error == want
        hit += want is not None
    assert hit > 0, "no position had a sheet too short: the comparison is vacuous"
