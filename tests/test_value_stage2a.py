"""IKA-425 stage 2a: the training options that are off by default.

- encoder dropout (`ValueConfig.encoder_dropout`), the auxiliary material head
  (`aux_weight`, `final_material_targets`) and per-game weights (`game_weights`, `train`'s
  `row_weight`), each with a control that it moves training and one that off is the old run;
- `tools/train_value.py --dropout / --weight-decay / --encoder-dropout / --aux-weight /
  --game-weights`: a warm start takes them, a model saved with the head drops it.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="value function needs the optional learn group")

from pokeuraou.encode import ENCODING_REVISION, Encoder, side_feature_names  # noqa: E402
from pokeuraou.regulation import load_regulation, repo_root  # noqa: E402
from pokeuraou.selfplay import position_from_sets  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402
from pokeuraou.value import (  # noqa: E402
    Dataset,
    ValueConfig,
    build,
    final_material_targets,
    game_weights,
    load_model,
    save_dataset,
    save_model,
    train,
)

MC = "gen9championsvgc2026regmc"
CPU = torch.device("cpu")
ALIVE, HP = (side_feature_names().index(n) for n in ("alive_fraction", "team_hp_fraction"))


def _dataset(encoder: Encoder, games: int = 12) -> Dataset:
    reg = encoder.reg
    sets = load_roster("rizabanadohido").sets
    n = reg.meta.picked_team_size
    rng = np.random.default_rng(425)
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
    encoded = encoder.encode_positions(positions)
    # Every game ends 0.5 alive and 0.25 HP ahead for side 0 at its last row, 0.1 + g/100
    # less alive for side 1 -- so a target that is read off the wrong row or side shows.
    for g in range(games):
        last = 3 * g + 2
        encoded.side[last, 0, ALIVE] = 1.0
        encoded.side[last, 1, ALIVE] = 0.5 - g / 100
        encoded.side[last, 0, HP] = 0.75
        encoded.side[last, 1, HP] = 0.5
        encoded.side[last - 1, 0, ALIVE] = 0.0  # earlier rows hold other values
    return Dataset(
        encoded=encoded,
        outcome=np.array(outcome, np.float32),
        game=np.array(game, np.int32),
        turn=np.tile(np.arange(1, 4), games).astype(np.int16),
        search_value=np.full(rows, 0.5, np.float32),
        hp_share=np.full(rows, 0.5, np.float32),
        kind=np.zeros(rows, np.int8),
        foe=np.zeros(rows, np.int32),
        foe_names=("roster",),
    )


@pytest.fixture(scope="module")
def mc():  # noqa: ANN201
    encoder = Encoder(load_regulation(MC))
    return encoder, _dataset(encoder)


def _state(net) -> dict:  # noqa: ANN001
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _fit(encoder, dataset, config, **kwargs):  # noqa: ANN001, ANN202
    torch.manual_seed(config.seed)
    net = build(encoder, config)
    _h, weights = train(net, dataset, config, device=CPU, holdout=0.25, **kwargs)
    return net, weights


def _config(**kw) -> ValueConfig:  # noqa: ANN003
    return ValueConfig(epochs=2, batch_size=8, keep="last", seed=3, split_seed=0, **kw)


# -- off is the old run -----------------------------------------------------------------


def test_the_defaults_build_the_same_net_and_train_the_same_run(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    config = _config()
    assert (config.encoder_dropout, config.aux_weight) == (0.0, 0.0)
    net, weights = _fit(encoder, dataset, config)
    assert not hasattr(net, "aux") and not any(k.startswith("aux") for k in weights)
    again, weights2 = _fit(encoder, dataset, config)
    assert _same(weights, weights2)  # the run is reproducible, so the next checks mean something
    # Weights of one everywhere are the plain mean up to float rounding of the reduction.
    _n, ones = _fit(encoder, dataset, config, row_weight=np.ones(len(dataset), np.float32))
    assert all(torch.allclose(weights[k], ones[k], atol=1e-5) for k in weights)


# -- weights ----------------------------------------------------------------------------


def test_game_weights_read_the_ranges() -> None:
    game = np.array([0, 0, 4, 5, 9, 12], np.int32)
    assert game_weights(game, "0:0.5,5:1,9:2").tolist() == [0.5, 0.5, 0.5, 1.0, 2.0, 2.0]
    for bad in ("1:1", "0:1,5:1,3:1", "0:-1"):
        with pytest.raises(ValueError):
            game_weights(game, bad)


def test_weights_change_the_run_and_a_zero_weight_removes_the_rows(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    config = _config()
    _n, plain = _fit(encoder, dataset, config)
    # Positive control: the weights reach the loss.
    weights = game_weights(dataset.game, "0:4,6:0.25")
    _n, weighted = _fit(encoder, dataset, config, row_weight=weights)
    assert not _same(plain, weighted)
    # A batch whose rows all weigh zero gives a zero loss, not a NaN.
    _n, zero = _fit(encoder, dataset, config, row_weight=np.zeros(len(dataset), np.float32))
    assert all(torch.isfinite(v).all() for v in zero.values())


# -- encoder dropout --------------------------------------------------------------------


def test_encoder_dropout_acts_in_training_only(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    batch = dataset.tensors(np.arange(12), CPU)
    torch.manual_seed(1)
    plain = build(encoder, ValueConfig())
    torch.manual_seed(1)
    dropped = build(encoder, ValueConfig(encoder_dropout=0.3))
    assert _same(_state(plain), _state(dropped))  # no parameters of its own
    plain.eval()
    dropped.eval()
    with torch.no_grad():
        assert torch.equal(plain(batch), dropped(batch))
        dropped.train()
        torch.manual_seed(2)
        first = dropped(batch)
        torch.manual_seed(3)
        second = dropped(batch)
    assert not torch.equal(first, second)  # a different mask each time
    # The head's dropout is on in train mode too, so compare with it switched off.
    assert not torch.equal(first, plain.eval()(batch))


# -- the auxiliary head -----------------------------------------------------------------


def test_final_material_targets_come_from_each_games_last_row(mc) -> None:  # noqa: ANN001
    _encoder, dataset = mc
    target = final_material_targets(dataset)
    assert target.shape == (len(dataset), 2) and target.dtype == np.float32
    for g in range(12):
        rows = target[3 * g : 3 * g + 3]
        assert np.allclose(rows, [[0.5 + g / 100, 0.25]] * 3, atol=1e-6)  # all rows, last's value
    # The comparison can fail: out-of-order game ids are refused rather than read.
    shuffled = Dataset(**{**{f: getattr(dataset, f) for f in Dataset.__dataclass_fields__},
                          "game": dataset.game[::-1].copy()})
    with pytest.raises(ValueError):
        final_material_targets(shuffled)


def test_the_aux_head_is_antisymmetric_and_the_logit_is_the_old_one(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    torch.manual_seed(5)
    net = build(encoder, ValueConfig(aux_weight=0.5)).eval()
    batch = dataset.tensors(np.arange(24), CPU)
    flipped = {
        k: (v.flip(1) if v.dim() > 1 and v.shape[1] == 2 else v) for k, v in batch.items()
    }
    with torch.no_grad():
        logit, aux = net.forward_aux(batch)
        _l2, aux_flipped = net.forward_aux(flipped)
        assert torch.allclose(logit, net(batch), atol=1e-6)
    assert aux.shape == (24, 2)
    assert torch.allclose(aux + aux_flipped, torch.zeros_like(aux), atol=1e-6)
    assert float(aux.abs().max()) > 0  # not trivially zero


def test_the_aux_loss_reaches_the_trunk_and_needs_its_target(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    target = final_material_targets(dataset)
    _n, plain = _fit(encoder, dataset, _config())
    net, with_aux = _fit(encoder, dataset, _config(aux_weight=5.0), aux_target=target)
    trunk = [k for k in plain if k.startswith("mon_mlp.0")]
    assert trunk and not _same({k: plain[k] for k in trunk}, {k: with_aux[k] for k in trunk})
    assert hasattr(net, "aux") and any(k.startswith("aux.") for k in with_aux)
    with pytest.raises(ValueError, match="go together"):
        _fit(encoder, dataset, _config(aux_weight=1.0))
    with pytest.raises(ValueError, match="go together"):
        _fit(encoder, dataset, _config(), aux_target=target)


# -- the tool ----------------------------------------------------------------------------


def _tool():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "train_value_tool_2a", repo_root() / "tools" / "train_value.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_tool_takes_the_flags_and_saves_a_plain_model(mc, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    encoder, dataset = mc
    data = tmp_path / "mc.npz"
    save_dataset(data, dataset, meta={"format_id": MC, "encoding_revision": ENCODING_REVISION})
    torch.manual_seed(7)
    net = build(encoder, ValueConfig()).eval()
    parent = tmp_path / "parent.pt"
    save_model(parent, net, net.state_dict(), encoder.vocab, ValueConfig(), meta={"m": 1},
               widths=encoder.widths)

    def run(out, *flags) -> None:  # noqa: ANN001, ANN002
        monkeypatch.setattr(
            sys, "argv",
            ["train_value.py", "--data", str(data), "--init-from", str(parent), "--epochs", "1",
             "--batch-size", "8", "--keep", "last", "--lr", "5e-4", "--out", str(out),
             "--device", "cpu", *flags],
        )
        _tool().main()

    base, tuned = tmp_path / "base.pt", tmp_path / "tuned.pt"
    run(base)
    run(tuned, "--dropout", "0.2", "--weight-decay", "0.05", "--encoder-dropout", "0.1",
        "--aux-weight", "1.0", "--game-weights", "0:2,6:0.5")
    plain_net, _ = load_model(base, encoder)
    tuned_net, _ = load_model(tuned, encoder)
    assert plain_net.config.dropout == 0.4 and plain_net.config.encoder_dropout == 0.0
    assert (tuned_net.config.dropout, tuned_net.config.encoder_dropout) == (0.2, 0.1)
    blob = torch.load(tuned, weights_only=False)
    assert blob["config"]["weight_decay"] == 0.05 and blob["config"]["aux_weight"] == 0.0
    assert not any(k.startswith("aux.") for k in blob["weights"])  # the head is dropped
    assert not hasattr(tuned_net, "aux")
    assert not _same(_state(plain_net), _state(tuned_net))
