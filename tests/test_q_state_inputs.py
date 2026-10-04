"""A Q's trunk can be a leaf that reads the state columns (IKA-427).

`QConfig.state_inputs` builds the trunk as `ValueConfig(state_inputs=True)`, so the
weights of a leaf trained with `train_value.py --state-inputs` (value-mc4st) load into it
and the Q reads revision 3's state columns as that leaf does. Off is the default and what
a file saved before the key means: every Q up to q-mc4 loads and answers as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="Q needs the optional learn group")

from pokeuraou import qhead  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.value import ValueConfig, build, grow_state_inputs, save_model  # noqa: E402
from tests.test_encode_state_port import FORMAT, _base, _with_state  # noqa: E402

ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    encoder = Encoder(load_regulation(FORMAT))
    base = _base()
    positions = [Position.from_json(p) for p in (base, _with_state(base, durations=True))]
    encoded = encoder.encode_positions(positions)
    batch = {name: torch.from_numpy(getattr(encoded, name)) for name in ARRAYS}
    torch.manual_seed(427)
    plain = build(encoder, ValueConfig()).eval()
    leaf = grow_state_inputs(plain, encoder).eval()
    # Nonzero weights on the new inputs, so a trunk that ignored them would answer otherwise.
    gen = torch.Generator().manual_seed(4270)
    with torch.no_grad():
        for weight in (leaf.mon_mlp[0].weight, leaf.side_mlp[0].weight):
            weight.copy_(torch.randn(weight.shape, generator=gen) * 0.3)
    return encoder, batch, plain, leaf


def test_the_default_is_off_and_an_old_file_reads_as_off() -> None:
    assert qhead.QConfig().state_inputs is False
    old = {"act_dim": 128, "pair_dim": 128, "rank": 32, "dropout": 0.1, "properties": False, "attend": True}
    assert qhead.QConfig(**old).state_inputs is False


def test_the_trunk_reads_the_widths_of_its_leaf(setup) -> None:  # noqa: ANN001
    encoder, _batch, _plain, _leaf = setup
    assert qhead.build_net(encoder, qhead.QConfig()).trunk.in_widths == encoder.base_widths
    state = qhead.build_net(encoder, qhead.QConfig(state_inputs=True)).trunk.in_widths
    assert state == encoder.state_widths


def test_a_state_leaf_loads_only_into_a_state_trunk(setup, tmp_path: Path) -> None:  # noqa: ANN001
    encoder, batch, _plain, leaf = setup
    path = tmp_path / "leaf.pt"
    save_model(path, leaf, leaf.state_dict(), encoder.vocab, leaf.config, meta={})
    net = qhead.build_net(encoder, qhead.QConfig(state_inputs=True)).eval()
    qhead.load_trunk(net, path, encoder)
    with torch.no_grad():
        features, _mask = leaf.mon_inputs(batch)
        want = leaf.mon_mlp(features)
        got = net.trunk.mon_mlp(net.trunk.mon_inputs(batch)[0])
    assert got.numpy().tobytes() == want.numpy().tobytes()
    # The state moves the trunk's rows: the two positions differ only in state columns.
    assert float((want[1] - want[0]).abs().max()) > 1e-4
    # The control: the default trunk has no weights for the state columns.
    with pytest.raises(RuntimeError):
        qhead.load_trunk(qhead.build_net(encoder, qhead.QConfig()), path, encoder)


def test_a_state_q_round_trips_through_its_file(setup, tmp_path: Path) -> None:  # noqa: ANN001
    encoder, batch, _plain, _leaf = setup
    torch.manual_seed(4271)
    net = qhead.build_net(encoder, qhead.QConfig(state_inputs=True)).eval()
    path = tmp_path / "q.pt"
    torch.save(
        {
            "state": net.state_dict(),
            "config": qhead.config_dict(net.config),
            "vocab_fingerprint": encoder.vocab.fingerprint(),
            "regulation": FORMAT,
            "move_table": None,
        },
        path,
    )
    loaded = qhead.load_q(path)
    assert loaded.config.state_inputs is True
    with torch.no_grad():
        a, _ = net.mons(batch)
        b, _ = loaded.mons(batch)
    assert a.numpy().tobytes() == b.numpy().tobytes()
