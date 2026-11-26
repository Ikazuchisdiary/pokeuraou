"""IKA-458 stage 3.2: a self-switch (a mid-turn replacement) chosen and played on in the port,
inside the turn's `advanceTurn`, against the nodes Python plays (`_do_self_switch_node`,
`_advance`).

What these hold, on turn 1 of a roster after our U-turn (the pause that test_hidden_selfswitch
reads), on the inference server's numpy stand-in of `test_portserved`, which keeps each request's
rows and the bytes of every array:

- **the same node**: the decisions the chain records (the options, the policy, the value, the
  choices, what was shown), the position the turn ends in, the notes and the generator's state
  are the Python road's, with the opponent's bench hidden (the shared plans and the ones resolved
  per completion) and with nothing hidden;
- **the same requests**: the server is sent the same rows, in the same requests, in the same
  order (the bytes of every array, so the completion's Pokemon swapped into a shared leaf is the
  one Python patches into the encoded rows);
- the positive control is the nodes the port played (`portlp.COUNTS["selfswitched"]`); the
  control that the comparison can fail is another server, which chooses otherwise somewhere;
- a fault, a leaf that is not the server's and a game the port was not given leave the pause to
  Python with the generator untouched.

No torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import hiddenstep, portlp, rustnode
from pokeuraou.encode import Encoder
from pokeuraou.inference import RemoteValue, serve
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import MoveSlot
from pokeuraou.selfplay import GameRecord, _advance_turn, _HiddenBench

from .test_hidden import UTURN_TURN, _action, _sheet, _uturn_start, setup  # noqa: F401
from .test_portserved import Stub


class Inverted(Stub):
    """Another server: it likes what the other dislikes."""

    def __call__(self, arrays, rows):  # noqa: ANN001, ANN204
        return 1.0 - super().__call__(arrays, rows)


@pytest.fixture(scope="module")
def servers():  # noqa: ANN201
    stub, other = Stub(), Inverted()
    shut, address = serve({"stub": stub, "other": other})
    yield stub, other, address
    shut.shutdown()


@pytest.fixture
def flags():  # noqa: ANN201
    was = (portlp.SELFSWITCH[0], portlp.PREP[0], portlp.HIDDEN[0], portlp.ADVANCE[0],
           portlp.DECIDE[0], rustnode._HOLD[0])  # noqa: SLF001
    portlp.set_selfswitch(False)
    portlp.set_advance(True)
    portlp.set_hidden(True)
    rustnode.hold_positions(True)
    yield
    (portlp.SELFSWITCH[0], portlp.PREP[0], portlp.HIDDEN[0], portlp.ADVANCE[0],
     portlp.DECIDE[0]) = was[:5]
    rustnode.hold_positions(was[5])


def _turn(reg, sheet, *, both):  # noqa: ANN001, ANN202
    """Turn 1 after our U-turn; with `both`, side 1 U-turns too."""
    start = _uturn_start(reg, sheet[:4], sheet[:4])
    turn = dict(UTURN_TURN)
    if both:
        start.sides[1].pokemon[0].moves[0] = MoveSlot(id="uturn", pp=20, maxpp=20)
        turn[1] = "move 1 1, move 4"
    return start, [_action(reg, start, side, turn[side]) for side in (0, 1)]


def _hidden(reg, start, sheet, *, defined):  # noqa: ANN001, ANN202
    """A hidden bench whose game the port holds (`defined`), the way a move node leaves it."""
    game = hiddenstep.Game.of((sheet[:4], sheet[:4]), (None, None))
    if defined:
        got = hiddenstep.step(reg, start, game, [frozenset(), frozenset()], [None, None])
        assert got is not None and got.error is None
    return _HiddenBench(
        sheets=(sheet[:4], sheet[:4]), seen=[frozenset(), frozenset()], bench_prior=None,
        leads=[None, None], game=game.number,
    )


def _play(reg, start, chosen, seed, hidden, leaves, *, ported):  # noqa: ANN001, ANN202
    portlp.set_selfswitch(ported)
    rng = np.random.default_rng(seed)
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
    landed = _advance_turn(reg, rng, start.copy(), chosen, record, leaves, HP_SHARE, hidden=hidden)
    return landed, record, rng.bit_generator.state


def _compare(setup, servers, *, both, hidden, seeds=range(24)):  # noqa: ANN001, F811
    reg, roster = setup
    sheet = _sheet(roster)
    start, chosen = _turn(reg, sheet, both=both)
    stub, _other, address = servers
    nodes = 0
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        bench = _hidden(reg, start, sheet, defined=True) if hidden else None
        for seed in seeds:
            del stub.log[:]
            before = portlp.COUNTS["selfswitched"]
            py_landed, py_record, py_state = _play(
                reg, start, chosen, seed, bench, (leaf, leaf), ported=False
            )
            py_log = list(stub.log)
            assert portlp.COUNTS["selfswitched"] == before
            del stub.log[:]
            landed, record, state = _play(
                reg, start, chosen, seed, bench, (leaf, leaf), ported=True
            )
            took = portlp.COUNTS["selfswitched"] - before
            nodes += took
            assert state == py_state
            assert (landed is None) == (py_landed is None)
            if landed is not None:
                assert landed.to_json() == py_landed.to_json()
            assert record.decisions == py_record.decisions
            assert sorted(set(record.unmodelled)) == sorted(set(py_record.unmodelled))
            assert list(stub.log) == py_log, "the server was sent other requests"
            assert took == sum(1 for d in record.decisions if d.kind == "selfswitch")
    return nodes


def test_a_hidden_bench_self_switch_is_the_ports(setup, servers, flags) -> None:  # noqa: ANN001, F811
    nodes = _compare(setup, servers, both=False, hidden=True)
    assert nodes >= 3, "the port played too few nodes: the comparison is vacuous"


def test_a_self_switch_by_both_sides_is_the_ports(setup, servers, flags) -> None:  # noqa: ANN001, F811
    nodes = _compare(setup, servers, both=True, hidden=True)
    assert nodes >= 3, "the port played too few nodes: the comparison is vacuous"


def test_a_self_switch_with_nothing_hidden_is_the_ports(setup, servers, flags) -> None:  # noqa: ANN001, F811
    nodes = _compare(setup, servers, both=False, hidden=False)
    assert nodes >= 3, "the port played too few nodes: the comparison is vacuous"


def test_another_server_chooses_otherwise(setup, servers, flags) -> None:  # noqa: ANN001, F811
    """The comparison can fail: a server that likes the opposite picks another replacement
    somewhere, so the decisions recorded differ."""
    reg, roster = setup
    sheet = _sheet(roster)
    start, chosen = _turn(reg, sheet, both=False)
    _stub, _other, address = servers
    bench = _hidden(reg, start, sheet, defined=True)
    chosen_by = {}
    for name in ("stub", "other"):
        with RemoteValue(address, name, Encoder(reg), buffer_bytes=64 << 20) as leaf:
            picks = []
            for seed in range(24):
                _landed, record, _state = _play(
                    reg, start, chosen, seed, bench, (leaf, leaf), ported=True
                )
                picks.append([(d.own_chosen, d.foe_chosen) for d in record.decisions])
            chosen_by[name] = picks
    assert chosen_by["stub"] != chosen_by["other"]


def test_a_game_the_port_does_not_hold_is_left_to_python(setup, servers, flags) -> None:  # noqa: ANN001, F811
    reg, roster = setup
    sheet = _sheet(roster)
    start, chosen = _turn(reg, sheet, both=False)
    stub, _other, address = servers
    bench = _hidden(reg, start, sheet, defined=False)
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        for seed in range(24):
            faults, nodes = portlp.COUNTS["faults"], portlp.COUNTS["selfswitched"]
            py_landed, py_record, py_state = _play(
                reg, start, chosen, seed, bench, (leaf, leaf), ported=False
            )
            landed, record, state = _play(reg, start, chosen, seed, bench, (leaf, leaf), ported=True)
            assert state == py_state and record.decisions == py_record.decisions
            assert (landed is None) == (py_landed is None)
            assert portlp.COUNTS["selfswitched"] == nodes
            if record.decisions:
                assert portlp.COUNTS["faults"] > faults
                return
    raise AssertionError("no draw reached a pause")


def test_a_fault_leaves_the_pause_to_python(setup, servers, flags, monkeypatch) -> None:  # noqa: ANN001, F811
    reg, roster = setup
    sheet = _sheet(roster)
    start, chosen = _turn(reg, sheet, both=False)
    _stub, _other, address = servers
    bench = _hidden(reg, start, sheet, defined=True)
    real = rustnode.RustNode.advance_turn

    def faulty(self, pos, actions, budget, rng, self_switch=None):  # noqa: ANN001, ANN202
        if self_switch is not None:
            return "injected"
        return real(self, pos, actions, budget, rng, None)

    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        want = [
            _play(reg, start, chosen, seed, bench, (leaf, leaf), ported=False) for seed in range(24)
        ]
        monkeypatch.setattr(rustnode.RustNode, "advance_turn", faulty)
        nodes = portlp.COUNTS["selfswitched"]
        faults = portlp.COUNTS["faults"]
        got = [_play(reg, start, chosen, seed, bench, (leaf, leaf), ported=True) for seed in range(24)]
    assert portlp.COUNTS["selfswitched"] == nodes and portlp.COUNTS["faults"] > faults
    for (landed, record, state), (py_landed, py_record, py_state) in zip(got, want, strict=True):
        assert state == py_state and record.decisions == py_record.decisions
        assert (landed is None) == (py_landed is None)
    assert any(record.decisions for _landed, record, _state in want)
