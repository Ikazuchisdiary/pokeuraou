"""`widen_net` grows the hidden layers and answers exactly as before (IKA-405).

The reason it is not just "add units with zero outgoing weights": every hidden layer is
followed by a LayerNorm, and over a wider layer its mean and variance change. The controls
here show that the naive form does not match (the comparison can fail) and that a nonzero
added outgoing weight moves the answer (the added units are really wired in).
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="value function needs the optional learn group")

from pokeuraou.encode import ENCODING_REVISION, Encoder  # noqa: E402
from pokeuraou.regulation import load_regulation, repo_root  # noqa: E402
from pokeuraou.selfplay import position_from_sets  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402
from pokeuraou.value import (  # noqa: E402
    Dataset,
    ValueConfig,
    build,
    load_dataset,
    load_model,
    predict,
    save_dataset,
    save_model,
    widen_net,
)

MC = "gen9championsvgc2026regmc"
CPU = torch.device("cpu")


def _dataset(encoder: Encoder, games: int = 8) -> Dataset:
    reg = encoder.reg
    sets = load_roster("rizabanadohido").sets
    n = reg.meta.picked_team_size
    rng = np.random.default_rng(405)
    positions, game, outcome = [], [], []
    for g in range(games):
        order = rng.permutation(len(sets))
        own = [sets[i] for i in order[:n]]
        foe = [sets[i] for i in order[len(sets) - n :]]
        result = float(rng.integers(2))
        for _ in range(3):
            positions.append(position_from_sets(reg, own, foe))
            game.append(g)
            outcome.append(result)
    rows = len(positions)
    return Dataset(
        encoded=encoder.encode_positions(positions),
        outcome=np.array(outcome, np.float32),
        game=np.array(game, np.int32),
        turn=np.ones(rows, np.int16),
        search_value=np.full(rows, 0.5, np.float32),
        hp_share=np.full(rows, 0.5, np.float32),
        kind=np.zeros(rows, np.int8),
        foe=np.zeros(rows, np.int32),
        foe_names=("roster",),
    )


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    encoder = Encoder(load_regulation(MC))
    data = _dataset(encoder)
    torch.manual_seed(405)
    net = build(encoder, ValueConfig())
    with torch.no_grad():  # LayerNorm affine is not the identity in a trained net
        for module in net.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.mul_(1.0 + 0.3 * torch.randn_like(module.weight))
                module.bias.add_(0.1 * torch.randn_like(module.bias))
    net.eval()
    return encoder, data, net


def _logits(net, data) -> np.ndarray:  # noqa: ANN001
    return predict(net, data, np.arange(len(data)), device=CPU)


def test_the_default_config_is_the_plain_layer_norm() -> None:
    assert ValueConfig().width_groups == 1
    encoder = Encoder(load_regulation(MC))
    net = build(encoder, ValueConfig())
    assert isinstance(net.head[1], torch.nn.LayerNorm)


@pytest.mark.parametrize("factor", [2, 3])
def test_a_widened_net_answers_as_the_original(setup, factor) -> None:  # noqa: ANN001
    encoder, data, net = setup
    base = _logits(net, data)
    wide = widen_net(net, encoder, factor, seed=1)
    assert wide.config.head_dim == factor * 256 and wide.config.width_groups == factor
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in net.parameters())
    assert float(np.abs(base).max()) > 0.05  # the logits are not all ~0
    assert float(np.abs(_logits(wide, data) - base).max()) < 1e-5


def test_a_nonzero_added_outgoing_weight_moves_the_answer(setup) -> None:  # noqa: ANN001
    # Positive control: the added units are wired in, so the equality above is not vacuous.
    encoder, data, net = setup
    base = _logits(net, data)
    wide = widen_net(net, encoder, 2, seed=1)
    with torch.no_grad():
        wide.head[8].weight[:, 256:] = 0.5
    assert float(np.abs(_logits(wide, data) - base).max()) > 1e-2


def test_the_naive_form_with_one_layer_norm_does_not_match(setup) -> None:  # noqa: ANN001
    # The shape the LayerNorm problem names: same zero outgoing weights, but the norm taken
    # over the whole wider layer. It must differ, or the grouped norm was never needed.
    encoder, data, net = setup
    base = _logits(net, data)
    wide = widen_net(net, encoder, 2, seed=1)
    with torch.no_grad():
        for seq in (wide.mon_mlp, wide.side_mlp, wide.head):
            for i, module in enumerate(seq):
                if hasattr(module, "groups"):
                    plain = torch.nn.LayerNorm(module.weight.numel())
                    plain.weight.copy_(module.weight)
                    plain.bias.copy_(module.bias)
                    seq[i] = plain
    assert float(np.abs(_logits(wide, data) - base).max()) > 1e-3


def _train_value_tool():  # noqa: ANN202
    path = repo_root() / "tools" / "train_value.py"
    spec = importlib.util.spec_from_file_location("train_value_tool_widen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_value_widen_at_zero_epochs_saves_an_equal_wider_model(  # noqa: ANN001
    setup, tmp_path, monkeypatch
) -> None:
    encoder, data, net = setup
    model = tmp_path / "m.pt"
    save_model(model, net, net.state_dict(), encoder.vocab, ValueConfig(), meta={},
               widths=encoder.widths)
    npz = tmp_path / "d.npz"
    save_dataset(npz, data, meta={"format_id": MC, "encoding_revision": ENCODING_REVISION})
    out = tmp_path / "wide.pt"
    argv = ["train_value.py", "--data", str(npz), "--init-from", str(model), "--epochs", "0",
            "--widen", "2", "--out", str(out), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    _train_value_tool().main()
    wide, meta = load_model(out, encoder)
    assert wide.config.width_groups == 2 and wide.config.mon_dim == 320
    assert meta["init_from"]["widened"] == 2
    loaded = load_dataset(npz)
    assert float(np.abs(_logits(wide, loaded) - _logits(net, loaded)).max()) < 1e-5


def test_widen_refuses_the_attention_layer(setup) -> None:  # noqa: ANN001
    from dataclasses import replace

    encoder, _data, _net = setup
    attn = build(encoder, replace(ValueConfig(), attention=True))
    with pytest.raises(ValueError, match="attention"):
        widen_net(attn, encoder, 2)
