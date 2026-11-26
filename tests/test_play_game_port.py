"""IKA-458 stage 3.6b: a game's turns played in one request to the port (`playGame`), against the
game Python plays one request a step (`play_game` with `portlp.PLAY` off).

What these hold, on a hidden-bench game of the `rizabanadohido` roster and one `inference.serve`
that is both the leaf's server (`test_portserved`'s numpy stand-in, keeping every request's rows) and
the Q's (`test_port_menus`' stand-in):

- **the same game**: the record (every decision, the notes, the end), the generator's state and
  the requests the server was sent are the road before it, for several seeds;
- the positive control is the moves the port played (`portlp.COUNTS["played"]`); the control that the
  comparison can fail is another Q, which plays another game;
- a fault, and a game the port does not hold, leave the game to Python from its first step with the
  generator untouched.

No torch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokeuraou import hiddenstep, portlp, rustnode
from pokeuraou.selfplay import play_game

from .test_move_node_port import FILL, OtherQ, _serve, flags_off  # noqa: F401 - fixture
from .test_port_menus import QStub, roster  # noqa: F401 - fixture


@pytest.fixture
def flags(flags_off):  # noqa: ANN001, ANN201, F811
    was = (portlp.PLAY[0], portlp.SELFSWITCH[0], portlp.PREP[0], portlp.HIDDEN[0], portlp.ADVANCE[0],
           rustnode._HOLD[0])  # noqa: SLF001
    portlp.set_play(False)
    portlp.set_hidden(True)
    portlp.set_node(True)
    portlp.set_replace(True)
    portlp.set_decide(True)
    portlp.set_exact(True)
    rustnode.hold_positions(True)
    yield
    (portlp.PLAY[0], portlp.SELFSWITCH[0], portlp.PREP[0], portlp.HIDDEN[0], portlp.ADVANCE[0]) = was[:5]
    rustnode.hold_positions(was[5])


def _play(roster, leaf, seed, *, play, number=None):  # noqa: ANN001, ANN202, F811
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    portlp.set_play(play)
    game = hiddenstep.Game.of((sheet, sheet), (None, None))
    # The port holds the game once it has been asked for a step of it.
    from pokeuraou import port

    request = _step_request(reg, sheet, game)
    port.ask(reg, lambda node: node.hidden_step(request, game))
    if number is not None:
        game = hiddenstep.Game(number, b"")
    rng = np.random.default_rng(seed)
    record = play_game(
        reg, rng, sheet[:4], sheet[2:6], "test", evaluate=leaf, rank_by_leaf=True, rank_fill=FILL,
        search_limit=(4, 4), max_turns=8, sheets=(sheet, sheet), hidden_game=game,
    )
    payload = record.to_json(objective="value:stub", search_limit=(4, 4))
    for key in ("engine", "searchSeconds"):
        payload.pop(key, None)
    return json.loads(json.dumps(payload)), rng.bit_generator.state


def _step_request(reg, sheet, game):  # noqa: ANN001, ANN202
    from pokeuraou.selfplay import position_from_sets

    pos = position_from_sets(reg, sheet[:4], sheet[2:6], rng=np.random.default_rng(0))
    return {"kind": "hiddenStep", "game": game.number, "position": rustnode._position(pos),  # noqa: SLF001
            "seen": [[], []], "leads": [None, None]}


def test_a_game_is_the_ports(roster, flags) -> None:  # noqa: ANN001, F811
    q_stub = QStub("")
    leaf_stub, leaf, close = _serve(roster, q_stub)
    try:
        played = 0
        outcomes = set()
        for seed in range(4):
            del q_stub.log[:], leaf_stub.log[:]
            want, want_state = _play(roster, leaf, seed, play=False)
            want_q, want_leaf = list(q_stub.log), list(leaf_stub.log)
            del q_stub.log[:], leaf_stub.log[:]
            before = portlp.COUNTS["played"]
            got, got_state = _play(roster, leaf, seed, play=True)
            played += portlp.COUNTS["played"] - before
            assert portlp.COUNTS["played"] > before, "the port played no move: vacuous"
            assert got == want
            assert got_state == want_state
            assert list(q_stub.log) == want_q, "the Q was sent other requests"
            assert list(leaf_stub.log) == want_leaf, "the leaf was sent other rows"
            outcomes.add(json.dumps([d["ownChosen"] for d in got["decisions"]]))
        assert played > 8
        assert len(outcomes) > 1, "every seed played the same game: the comparison is weak"
    finally:
        close()


def test_another_q_plays_another_game(roster, flags) -> None:  # noqa: ANN001, F811
    """The comparison can fail: a Q whose rows come in the other order picks other menus."""
    first = QStub("")
    leaf_stub, leaf, close = _serve(roster, first)
    try:
        want, _ = _play(roster, leaf, 0, play=True)
    finally:
        close()
    other = OtherQ("")
    leaf_stub, leaf, close = _serve(roster, other)
    try:
        got, _ = _play(roster, leaf, 0, play=True)
    finally:
        close()
    assert got != want


def test_a_game_the_port_does_not_hold_is_left_to_python(roster, flags) -> None:  # noqa: ANN001, F811
    q_stub = QStub("")
    leaf_stub, leaf, close = _serve(roster, q_stub)
    try:
        want, want_state = _play(roster, leaf, 1, play=False)
        faults, played = portlp.COUNTS["faults"], portlp.COUNTS.get("played", 0)
        got, got_state = _play(roster, leaf, 1, play=True, number=10_000_000)
        assert portlp.COUNTS["faults"] > faults and portlp.COUNTS.get("played", 0) == played
        assert got_state == want_state
    finally:
        close()


def test_a_fault_leaves_the_game_to_python(roster, flags, monkeypatch) -> None:  # noqa: ANN001, F811
    q_stub = QStub("")
    leaf_stub, leaf, close = _serve(roster, q_stub)
    try:
        want, want_state = _play(roster, leaf, 2, play=False)
        monkeypatch.setattr(
            rustnode.RustNode, "play_game", lambda self, request, game: {"fault": "injected"}  # noqa: ARG005
        )
        faults, played = portlp.COUNTS["faults"], portlp.COUNTS.get("played", 0)
        got, got_state = _play(roster, leaf, 2, play=True)
        assert portlp.COUNTS["faults"] == faults + 1 and portlp.COUNTS.get("played", 0) == played
        assert got == want and got_state == want_state
    finally:
        close()
