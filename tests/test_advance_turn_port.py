"""IKA-458 stage 3.1 (`advanceTurn`): a turn's weights, the draw among them and the drawn branch
in one request to the port, against the three requests it replaces (`weights`, numpy's `choice`
in Python, `branch`, and the next position's `needed`).

What these hold, on real turns of a roster's games (both sides' random legal actions):

- **the same draw**: the index, the position it lands on, the replacements that position owes,
  the weights and the notes are the Python road's, and the generator ends in the same state as
  after numpy's `choice` -- over turns that land on a branch and turns that stop at a pause;
- **the positive controls**: the turns the port advanced (`portlp.COUNTS["advanced"]`) and the
  pauses among them, which only an index crosses;
- **the control that can fail**: another seed draws other outcomes;
- a fault, or a generator that is not PCG64, leaves the turn to Python with the generator
  untouched.

No server and no torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import port, portlp, rustnode
from pokeuraou.budget import Budget
from pokeuraou.payoff import HP_SHARE
from pokeuraou.qhead import legal_pool
from pokeuraou.selfplay import GameRecord, _advance_turn, _sample_index, position_from_sets

from .test_port_menus import load_roster, register_mega_stones


@pytest.fixture(scope="module")
def turns():  # noqa: ANN201
    """(reg, position, actions) of turns met playing, a pause among them kept for its own."""
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    reg = loaded.reg
    rng = np.random.default_rng(1)
    out = []
    for _game in range(14):
        pos = position_from_sets(reg, list(loaded.sets[:4]), list(loaded.sets[2:6]))
        for _turn in range(6):
            if pos.ended:
                break
            owed = port.replacements_needed(reg, pos)
            if any(owed[0]) or any(owed[1]):
                break
            pools = [legal_pool(reg, pos, s) for s in (0, 1)]
            chosen = [pools[0][int(rng.integers(len(pools[0])))],
                      pools[1][int(rng.integers(len(pools[1])))]]
            out.append((reg, pos, chosen))
            weights = port.weights(reg, pos, chosen, Budget.exact())
            counts = np.array(weights.branches + weights.suspended, dtype=np.float64)
            index = _sample_index(rng, counts)
            if index >= len(weights.branches):
                break
            pos = port.branch(reg, pos, chosen, Budget.exact(), index)
    return out


@pytest.fixture
def flags_off():  # noqa: ANN201
    was = (portlp.ADVANCE[0], portlp.DECIDE[0], portlp.MENUS[0], portlp.REPLACE[0],
           portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0],
           rustnode._SPAN_BLOCKS[0])  # noqa: SLF001
    yield
    (portlp.ADVANCE[0], portlp.DECIDE[0], portlp.MENUS[0], portlp.REPLACE[0], portlp.EXACT[0],
     portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0]) = was[:9]
    rustnode._SPAN_BLOCKS[0] = was[9]  # noqa: SLF001


def _python_road(reg, pos, chosen, rng):  # noqa: ANN001, ANN202
    weights = port.weights(reg, pos, chosen, Budget.exact())
    counts = np.array(weights.branches + weights.suspended, dtype=np.float64)
    index = _sample_index(rng, counts)
    landed = (
        port.branch(reg, pos, chosen, Budget.exact(), index)
        if index < len(weights.branches) else None
    )
    return weights, index, landed


def test_the_draw_is_numpys(turns, flags_off) -> None:  # noqa: ANN001
    paused = branched = 0
    for k, (reg, pos, chosen) in enumerate(turns):
        mine, theirs = np.random.default_rng(100 + k), np.random.default_rng(100 + k)
        weights, index, landed = _python_road(reg, pos, chosen, mine)
        got = port.advance(reg, pos, chosen, Budget.exact(), theirs)
        assert got.index == index
        assert got.branches == weights.branches and got.suspended == weights.suspended
        assert got.exact == weights.exact and got.unmodelled == weights.unmodelled
        assert theirs.bit_generator.state == mine.bit_generator.state
        # And the generator goes on the same: the next draws are the same.
        assert theirs.random() == mine.random()
        if landed is None:
            paused += 1
            assert got.position is None
        else:
            branched += 1
            assert got.position.to_json() == landed.to_json()
            assert port.replacements_needed(reg, got.position) == port.replacements_needed(
                reg, landed
            )
    assert paused >= 1 and branched >= 5, (paused, branched)


def test_another_seed_is_another_draw(turns, flags_off) -> None:  # noqa: ANN001
    """The comparison can fail: seeds that differ draw other outcomes somewhere."""
    indices = {0: [], 1: []}
    for k, (reg, pos, chosen) in enumerate(turns):
        for which, seed in enumerate((11, 12)):
            indices[which].append(
                port.advance(reg, pos, chosen, Budget.exact(), np.random.default_rng(seed + k)).index
            )
    assert indices[0] != indices[1]


def _advance_both(reg, pos, chosen, seed):  # noqa: ANN001, ANN202
    results = []
    for advance in (False, True):
        portlp.ADVANCE[0] = False
        if advance:
            portlp.set_advance(True)
        rng = np.random.default_rng(seed)
        record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
        before = portlp.COUNTS["advanced"]
        landed = _advance_turn(reg, rng, pos, chosen, record, (None, None), HP_SHARE)
        results.append((landed, rng.bit_generator.state, record.unmodelled,
                        portlp.COUNTS["advanced"] - before))
    return results


def test_a_turn_of_a_game_is_the_same(turns, flags_off) -> None:  # noqa: ANN001
    """`_advance_turn` itself, pauses carried on in Python in both roads."""
    ran = 0
    for k, (reg, pos, chosen) in enumerate(turns):
        (py, py_state, py_notes, c0), (ported, port_state, port_notes, c1) = _advance_both(
            reg, pos, chosen, 500 + k
        )
        assert (c0, c1) == (0, 1), "the port did not take the turn: the comparison is vacuous"
        ran += 1
        assert (py is None) == (ported is None)
        if py is not None:
            assert ported.to_json() == py.to_json()
        assert port_state == py_state
        assert port_notes == py_notes
    assert ran == len(turns) and ran > 20


def test_a_fault_leaves_the_turn_to_python(turns, flags_off, monkeypatch) -> None:  # noqa: ANN001
    reg, pos, chosen = turns[0]
    want = _advance_both(reg, pos, chosen, 9)[0]
    portlp.set_advance(True)
    faults, advanced = portlp.COUNTS["faults"], portlp.COUNTS["advanced"]
    rng = np.random.default_rng(9)
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
    # Only `advanceTurn` is made to fault; the rest of the turn is asked as it always is.
    real = rustnode.RustNode._exchange  # noqa: SLF001

    def exchange(self, request):  # noqa: ANN001, ANN202
        if request.get("kind") == "advanceTurn":
            return {"fault": "injected"}
        return real(self, request)

    monkeypatch.setattr(rustnode.RustNode, "_exchange", exchange)
    landed = _advance_turn(reg, rng, pos, chosen, record, (None, None), HP_SHARE)
    assert portlp.COUNTS["faults"] == faults + 1 and portlp.COUNTS["advanced"] == advanced
    assert rng.bit_generator.state == want[1]
    assert (landed is None) == (want[0] is None)
    if landed is not None:
        assert landed.to_json() == want[0].to_json()


def test_an_unknown_kind_is_answered_at_once(turns) -> None:  # noqa: ANN001
    """A request the binary does not know fails the caller instead of waiting for a body."""
    reg = turns[0][0]
    node = rustnode.require_node(reg)
    with pytest.raises(RuntimeError, match="unknown request kind"):
        node._exchange({"kind": "aRequestFromTheFuture"})  # noqa: SLF001


def test_an_unknown_kind_stops_without_restarting_the_port(turns) -> None:  # noqa: ANN001
    """A binary older than the request is the same binary after a restart: `port.ask` stops at
    once with "rebuild it", where it used to restart until the restarts ran out (stage 3.5's
    old-binary test took 180 s to give up)."""
    from pokeuraou import port

    reg = turns[0][0]
    restarts = rustnode._RESTARTS  # noqa: SLF001
    with pytest.raises(rustnode.PortUnavailable, match="rebuild it"):
        port.ask(reg, lambda node: node._exchange({"kind": "aRequestFromTheFuture"}))  # noqa: SLF001
    assert restarts == rustnode._RESTARTS  # noqa: SLF001
    # The port is still the one that answered: the next request is answered as usual.
    with pytest.raises(rustnode.PortUnavailable):
        rustnode.require_node(reg)._exchange({"kind": "aRequestFromTheFuture"})  # noqa: SLF001


def test_a_generator_that_is_not_pcg64_is_python_s(turns, flags_off) -> None:  # noqa: ANN001
    reg, pos, chosen = turns[0]
    portlp.set_advance(True)
    rng = np.random.Generator(np.random.MT19937(5))
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
    advanced = portlp.COUNTS["advanced"]
    _advance_turn(reg, rng, pos, chosen, record, (None, None), HP_SHARE)
    assert portlp.COUNTS["advanced"] == advanced
