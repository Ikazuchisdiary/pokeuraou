"""IKA-458 stage 3.6a: a game's replacement phase -- the options, the node, both draws and the
phase -- chosen and played in one request to the port (`replacementPhase`), against the node and
the phase Python plays (`_do_replacement_node` with `resolve_replacements`).

What these hold, on the recorded mid-game replacement of `test_hidden_search` and the inference
server's numpy stand-in of `test_portserved`, which keeps each request's rows and the bytes of every
array:

- **the same phase**: the decision it records (the options, both strategies, the values, the
  choices), the position the phase ends in, the notes and the generator's state are the Python
  road's, for several seeds (so several draws);
- **the same requests**: the server is sent the same rows, in the same order;
- the positive control is the phases the port played (`portlp.COUNTS["phased"]`); the control
  that the comparison can fail is another server;
- a fault, and a game the port does not hold, leave the node to Python with the generator
  untouched.

No torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import hiddenstep, portlp, rustnode
from pokeuraou.encode import Encoder
from pokeuraou.inference import RemoteValue, serve
from pokeuraou.selfplay import GameRecord, _do_replacement_node

from .test_hidden_search import _fixture_sheets, replacement  # noqa: F401 - fixture
from .test_portlp import openblas_core
from .test_portserved import Stub

EXACT = openblas_core() == "SkylakeX"


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
    was = (portlp.PLAY[0], portlp.SELFSWITCH[0], portlp.PREP[0], portlp.HIDDEN[0], portlp.ADVANCE[0],
           portlp.REPLACE[0], rustnode._HOLD[0])  # noqa: SLF001
    portlp.set_play(False)
    portlp.set_hidden(True)
    portlp.set_node(True)
    portlp.set_replace(True)
    rustnode.hold_positions(True)
    yield
    (portlp.PLAY[0], portlp.SELFSWITCH[0], portlp.PREP[0], portlp.HIDDEN[0], portlp.ADVANCE[0],
     portlp.REPLACE[0]) = was[:6]
    rustnode.hold_positions(was[6])


def _run(reg, position, owed, sheets, leaves, seed, *, play, defined=True):  # noqa: ANN001, ANN202
    portlp.set_play(play)
    game = hiddenstep.Game.of(sheets, (None, None))
    stepped = hiddenstep.step(reg, position, game, [frozenset(), frozenset()], [None, None])
    assert stepped is not None and stepped.error is None
    number = game.number if defined else game.number + 10_000
    rng = np.random.default_rng(seed)
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
    after = _do_replacement_node(
        reg, rng, position.copy(), owed, record, leaves,
        sheets=sheets, shown=stepped.shown, leads=[None, None], recorded_shown=stepped.recorded,
        stepped=stepped, hidden_number=number,
    )
    assert record.decisions, "the node recorded nothing"
    return record, after, rng.bit_generator.state


def _same(a, b) -> bool:  # noqa: ANN001
    compare = np.array_equal if EXACT else (lambda x, y: np.allclose(x, y, rtol=0, atol=1e-12))
    return (
        a.own_actions == b.own_actions and a.foe_actions == b.foe_actions
        and compare(a.own_policy, b.own_policy) and compare(a.foe_policy, b.foe_policy)
        and (a.search_value == b.search_value if EXACT else abs(a.search_value - b.search_value) < 1e-12)
        and a.own_chosen == b.own_chosen and a.foe_chosen == b.foe_chosen
        and a.shown == b.shown and a.kind == b.kind and a.turn == b.turn and a.position == b.position
    )


def test_a_replacement_phase_is_the_ports(replacement, servers, flags) -> None:  # noqa: ANN001, F811
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    stub, other, address = servers
    phased = 0
    drawn = set()
    for name, server in (("stub", stub), ("other", other)):
        with RemoteValue(address, name, Encoder(reg), buffer_bytes=64 << 20) as leaf:
            for seed in range(12):
                del server.log[:]
                py_record, py_after, py_state = _run(
                    reg, position, owed, sheets, (leaf, leaf), seed, play=False
                )
                py_log = list(server.log)
                del server.log[:]
                before = portlp.COUNTS["phased"]
                record, after, state = _run(reg, position, owed, sheets, (leaf, leaf), seed, play=True)
                phased += portlp.COUNTS["phased"] - before
                assert portlp.COUNTS["phased"] - before == 1, "the port did not play the phase"
                assert state == py_state
                assert _same(record.decisions[0], py_record.decisions[0])
                assert after.to_json() == py_after.to_json()
                assert sorted(set(record.unmodelled)) == sorted(set(py_record.unmodelled))
                assert list(server.log) == py_log, "the server was sent other requests"
                drawn.add((record.decisions[0].own_chosen, record.decisions[0].foe_chosen))
    assert phased == 24
    assert len(drawn) > 1, "every seed drew the same pair: the comparison is weak"


def test_another_server_chooses_otherwise(replacement, servers, flags) -> None:  # noqa: ANN001, F811
    """The comparison can fail: a server that likes the opposite gives other strategies."""
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    _stub, _other, address = servers
    got = {}
    for name in ("stub", "other"):
        with RemoteValue(address, name, Encoder(reg), buffer_bytes=64 << 20) as leaf:
            record, _after, _state = _run(reg, position, owed, sheets, (leaf, leaf), 0, play=True)
            got[name] = record.decisions[0]
    assert not _same(got["stub"], got["other"])


def test_a_game_the_port_does_not_hold_is_left_to_python(replacement, servers, flags) -> None:  # noqa: ANN001, F811
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    _stub, _other, address = servers
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        want = _run(reg, position, owed, sheets, (leaf, leaf), 1, play=False)
        faults, phased = portlp.COUNTS["faults"], portlp.COUNTS["phased"]
        got = _run(reg, position, owed, sheets, (leaf, leaf), 1, play=True, defined=False)
    assert portlp.COUNTS["faults"] == faults + 1 and portlp.COUNTS["phased"] == phased
    assert got[2] == want[2] and _same(got[0].decisions[0], want[0].decisions[0])
    assert got[1].to_json() == want[1].to_json()


def test_a_fault_leaves_the_node_to_python(replacement, servers, flags, monkeypatch) -> None:  # noqa: ANN001, F811
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    _stub, _other, address = servers
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        want = _run(reg, position, owed, sheets, (leaf, leaf), 2, play=False)
        monkeypatch.setattr(
            rustnode.RustNode, "replacement_phase", lambda self, request: {"fault": "injected"}  # noqa: ARG005
        )
        faults, phased = portlp.COUNTS["faults"], portlp.COUNTS["phased"]
        got = _run(reg, position, owed, sheets, (leaf, leaf), 2, play=True)
    assert portlp.COUNTS["faults"] == faults + 1 and portlp.COUNTS["phased"] == phased
    assert got[2] == want[2] and _same(got[0].decisions[0], want[0].decisions[0])
    assert got[1].to_json() == want[1].to_json()
