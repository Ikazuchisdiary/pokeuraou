"""A Q's trunk can be a leaf that reads the bind columns (IKA-430).

`QConfig.bind_inputs` builds the trunk as `ValueConfig(state_inputs=True, bind_inputs=True)`,
so the weights of a leaf trained with `train_value.py --state-inputs --bind-inputs`
(value-mc4bindaux) load into it and the Q reads revision 4's bind columns as that leaf does.
Off is the default and what a file saved before the key means: q-mc4st loads as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="Q needs the optional learn group")

from pokeuraou import qhead  # noqa: E402
from pokeuraou.encode import BIND_MON_FEATURES, Encoder  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.value import ValueConfig, build, grow_bind_inputs, grow_state_inputs, save_model  # noqa: E402
from tests.test_encode_bind_port import FORMAT, _one_hp  # noqa: E402
from tests.test_encode_state_port import _base, _with_state  # noqa: E402

ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    encoder = Encoder(load_regulation(FORMAT))
    base = _base()
    raw = [base, _one_hp(base, 1), _with_state(_one_hp(base, 0), durations=True)]
    encoded = encoder.encode_positions([Position.from_json(p) for p in raw])
    assert encoded.mon[..., -len(BIND_MON_FEATURES) :].any()
    batch = {name: torch.from_numpy(getattr(encoded, name)) for name in ARRAYS}
    torch.manual_seed(430)
    leaf = grow_bind_inputs(grow_state_inputs(build(encoder, ValueConfig()), encoder), encoder).eval()
    # Nonzero weights on every input, the bind columns included, so a trunk that ignored
    # the bind columns would answer otherwise.
    gen = torch.Generator().manual_seed(4300)
    with torch.no_grad():
        for weight in (leaf.mon_mlp[0].weight, leaf.side_mlp[0].weight):
            weight.copy_(torch.randn(weight.shape, generator=gen) * 0.3)
    return encoder, batch, leaf


def test_the_default_is_off_and_an_old_file_reads_as_off() -> None:
    assert qhead.QConfig().bind_inputs is False
    q_mc4st = {
        "act_dim": 128, "pair_dim": 128, "rank": 32, "dropout": 0.1,
        "properties": True, "attend": True, "state_inputs": True,
    }
    assert qhead.QConfig(**q_mc4st).bind_inputs is False


def test_the_trunk_reads_the_widths_of_its_leaf(setup) -> None:  # noqa: ANN001
    encoder, _batch, _leaf = setup
    state = qhead.build_net(encoder, qhead.QConfig(state_inputs=True)).trunk.in_widths
    assert state == encoder.state_widths
    bind = qhead.build_net(encoder, qhead.QConfig(state_inputs=True, bind_inputs=True))
    assert bind.trunk.in_widths == encoder.widths != encoder.state_widths


def test_a_bind_leaf_loads_only_into_a_bind_trunk(setup, tmp_path: Path) -> None:  # noqa: ANN001
    encoder, batch, leaf = setup
    path = tmp_path / "leaf.pt"
    save_model(path, leaf, leaf.state_dict(), encoder.vocab, leaf.config, meta={})
    net = qhead.build_net(encoder, qhead.QConfig(state_inputs=True, bind_inputs=True)).eval()
    qhead.load_trunk(net, path, encoder)
    with torch.no_grad():
        want = leaf.mon_mlp(leaf.mon_inputs(batch)[0])
        got = net.trunk.mon_mlp(net.trunk.mon_inputs(batch)[0])
        # The bind columns move the rows: zero them and the answer changes.
        blind = dict(batch)
        blind["mon"] = batch["mon"].clone()
        blind["mon"][..., -len(BIND_MON_FEATURES) :] = 0
        moved = net.trunk.mon_mlp(net.trunk.mon_inputs(blind)[0])
    assert got.numpy().tobytes() == want.numpy().tobytes()
    assert float((moved - got).abs().max()) > 1e-4
    # The control: a state-only trunk has no weights for the bind columns.
    with pytest.raises(RuntimeError):
        qhead.load_trunk(qhead.build_net(encoder, qhead.QConfig(state_inputs=True)), path, encoder)


def test_a_bind_q_round_trips_through_its_file(setup, tmp_path: Path) -> None:  # noqa: ANN001
    encoder, batch, _leaf = setup
    torch.manual_seed(4301)
    net = qhead.build_net(encoder, qhead.QConfig(state_inputs=True, bind_inputs=True)).eval()
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
    assert loaded.config.bind_inputs is True
    with torch.no_grad():
        a, _ = net.mons(batch)
        b, _ = loaded.mons(batch)
    assert a.numpy().tobytes() == b.numpy().tobytes()
