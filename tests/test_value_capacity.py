"""IKA-439: capacity knobs of the value net.

- `tools/train_value.py --width-scale` (IKA-398) composes with state inputs, bind inputs and the
  auxiliary targets of the production recipe: the net builds, trains, and keeps its antisymmetry;
- `ValueConfig.head_layers` (default 2): the default builds the old net bit for bit and saves a
  config without the field; 3 adds one hidden layer that changes the answer, keeps the
  antisymmetry, and round-trips through save and load.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="value function needs the optional learn group")

from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.value import (  # noqa: E402
    ValueConfig,
    aux_target_arrays,
    build,
    final_material_targets,
    load_model,
    save_model,
    train,
)
from tests.test_value_aux_targets import CPU, _dataset, _same  # noqa: E402

MC = "gen9championsvgc2026regmc"


@pytest.fixture(scope="module")
def mc():  # noqa: ANN201
    encoder = Encoder(load_regulation(MC))
    return encoder, _dataset(encoder)


def _batch(encoder, dataset):  # noqa: ANN001, ANN202
    rows = slice(0, 6)
    e = dataset.encoded
    return {
        "species": torch.from_numpy(e.species[rows]),
        "ability": torch.from_numpy(e.ability[rows]),
        "item": torch.from_numpy(e.item[rows]),
        "moves": torch.from_numpy(e.moves[rows]),
        "mon": torch.from_numpy(e.mon[rows]),
        "side": torch.from_numpy(e.side[rows]),
        "field": torch.from_numpy(e.field[rows]),
        "mask": torch.from_numpy(e.mask[rows]),
    }


def _flip(batch):  # noqa: ANN001, ANN202
    return {k: (v.flip(1) if k != "field" else v) for k, v in batch.items()}


def test_default_head_layers_is_the_old_net(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    assert ValueConfig().head_layers == 2
    torch.manual_seed(1)
    a = build(encoder, ValueConfig())
    torch.manual_seed(1)
    b = build(encoder, ValueConfig(head_layers=2))
    assert _same(a.state_dict(), b.state_dict())
    assert len(a.head) == 9
    batch = _batch(encoder, dataset)
    a.eval()
    b.eval()
    assert torch.equal(a(batch), b(batch))


def test_three_head_layers_adds_one_and_moves_the_answer(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    torch.manual_seed(1)
    two = build(encoder, ValueConfig())
    torch.manual_seed(1)
    three = build(encoder, ValueConfig(head_layers=3))
    assert len(three.head) == 13 and three.head[-1].out_features == 1
    shared = {k: v for k, v in two.state_dict().items() if not k.startswith("head.")}
    assert _same(shared, {k: three.state_dict()[k] for k in shared})
    two.eval()
    three.eval()
    batch = _batch(encoder, dataset)
    # positive control: the new layer changes the answer ...
    assert not torch.allclose(two(batch), three(batch), atol=1e-6)
    # ... and the net is still antisymmetric in the two sides
    assert torch.allclose(three(batch), -three(_flip(batch)), atol=1e-5)
    with pytest.raises(ValueError, match="at least 2"):
        build(encoder, ValueConfig(head_layers=1))


def test_head_layers_round_trip_and_default_config_is_unchanged(mc, tmp_path) -> None:  # noqa: ANN001
    encoder, dataset = mc
    batch = _batch(encoder, dataset)
    for layers in (2, 3):
        config = ValueConfig(head_layers=layers)
        net = build(encoder, config)
        path = tmp_path / f"h{layers}.pt"
        save_model(path, net, net.state_dict(), encoder.vocab, config, meta={})
        blob = torch.load(path, weights_only=False)
        assert ("head_layers" in blob["config"]) == (layers != 2)
        back, _ = load_model(path, encoder)
        assert back.config.head_layers == layers
        net.eval()
        assert torch.equal(net(batch), back(batch))


@pytest.mark.parametrize("scale", [1.5, 2.0])
@pytest.mark.parametrize("layers", [2, 3])
def test_scaled_width_with_the_production_recipe(mc, scale, layers) -> None:  # noqa: ANN001
    encoder, dataset = mc
    base = ValueConfig()
    sizes = {
        name: int(round(getattr(base, name) * scale))
        for name in ("species_dim", "ability_dim", "item_dim", "move_dim", "mon_dim",
                     "side_dim", "head_dim")
    }
    config = ValueConfig(
        state_inputs=True, bind_inputs=True, aux_weight=5.0, aux_targets="search:1,ahead2:5",
        epochs=2, batch_size=8, keep="last", seed=3, split_seed=0, head_layers=layers, **sizes,
    )
    torch.manual_seed(3)
    net = build(encoder, config)
    assert net.head[0].out_features == sizes["head_dim"]
    assert net.aux.in_features == sizes["head_dim"]
    targets = aux_target_arrays(dataset, ["search", "ahead2"], encoder)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    _h, weights = train(
        net, dataset, config, device=CPU, holdout=0.25,
        aux_target=final_material_targets(dataset), aux_targets=targets,
    )
    assert any(not torch.equal(before[k], weights[k]) for k in ("mon_mlp.0.weight", "head.0.weight"))
    net.load_state_dict(weights)
    net.eval()
    batch = _batch(encoder, dataset)
    assert torch.allclose(net(batch), -net(_flip(batch)), atol=1e-5)
