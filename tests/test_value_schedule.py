"""What `train` hands back, and where a warm start begins (IKA-86, IKA-194).

- ``epochs=0`` returns the weights it was given, bit for bit: the null control of a warm
  start, and the only way `tools/train_value.py --init-from` can save a model unchanged;
- ``keep="last"`` runs the whole schedule -- no early stop -- and returns the final weights,
  or an average of them (EMA / SWA), because the recipe up to value-gen11L stopped near the
  top of OneCycle and never kept the decay;
- `tools/train_value.py --init-from` reads the model through `load_model`, so an M-B model
  starts an M-C run with zero rows, and a 0-epoch run saves a model that scores M-B
  positions exactly as the original does. One epoch moves them (positive control).
"""

from __future__ import annotations

import importlib.util
import json
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
    train,
)

MB = "gen9championsvgc2026regmb"
MC = "gen9championsvgc2026regmc"
CPU = torch.device("cpu")


def _dataset(encoder: Encoder, games: int = 12) -> Dataset:
    reg = encoder.reg
    sets = load_roster("rizabanadohido").sets
    n = reg.meta.picked_team_size
    rng = np.random.default_rng(86)
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
def mb():  # noqa: ANN201
    encoder = Encoder(load_regulation(MB))
    return encoder, _dataset(encoder)


def _state(net) -> dict:  # noqa: ANN001
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_zero_epochs_returns_the_weights_it_was_given(mb) -> None:  # noqa: ANN001
    encoder, dataset = mb
    torch.manual_seed(1)
    net = build(encoder, ValueConfig())
    before = _state(net)
    history, weights = train(net, dataset, ValueConfig(epochs=0), device=CPU)
    assert history == [] and _same(weights, before)


def test_keep_last_runs_the_whole_schedule_and_returns_its_end(mb) -> None:  # noqa: ANN001
    encoder, dataset = mb
    config = ValueConfig(epochs=4, batch_size=8, patience=1, keep="last", swa_from=0.5)
    net = build(encoder, config)
    snaps: dict = {}
    history, weights = train(net, dataset, config, device=CPU, holdout=0.25, snapshots=snaps)
    # patience 1 would have stopped a "best" run at the first epoch that did not improve.
    assert [r.epoch for r in history] == [1, 2, 3, 4]
    assert set(snaps) == {"last", "ema", "swa"}
    assert _same(weights, snaps["last"]) and _same(weights, _state(net))
    # The averages are different weights from the end point (the net moved in the tail).
    assert not _same(snaps["swa"], snaps["last"])
    assert not _same(snaps["ema"], snaps["last"])


def test_an_average_is_what_is_returned_when_asked_for(mb) -> None:  # noqa: ANN001
    encoder, dataset = mb
    for average in ("ema", "swa"):
        config = ValueConfig(epochs=3, batch_size=8, keep="last", average=average, seed=2)
        snaps: dict = {}
        _h, weights = train(
            build(encoder, config), dataset, config, device=CPU, holdout=0.25, snapshots=snaps
        )
        assert _same(weights, snaps[average])
    # SWA over the last epoch alone is the last epoch's weights: the mean of one.
    config = ValueConfig(epochs=3, batch_size=8, keep="last", average="swa", swa_from=1.0)
    snaps = {}
    _h, weights = train(
        build(encoder, config), dataset, config, device=CPU, holdout=0.25, snapshots=snaps
    )
    assert _same(weights, snaps["last"])


def test_an_average_needs_the_whole_schedule(mb) -> None:  # noqa: ANN001
    encoder, dataset = mb
    with pytest.raises(ValueError, match="keep='last'"):
        train(build(encoder, ValueConfig()), dataset, ValueConfig(average="ema"), device=CPU)


def test_the_shipped_recipe_is_unchanged_by_default() -> None:
    # Every model up to value-gen11L: a stored config without the new keys reads back as it.
    config = ValueConfig()
    assert (config.keep, config.average, config.pct_start) == ("best", "none", 0.2)


# -- tools/train_value.py --init-from ---------------------------------------------------


def _train_value_tool():  # noqa: ANN202
    path = repo_root() / "tools" / "train_value.py"
    spec = importlib.util.spec_from_file_location("train_value_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch, *argv: str) -> None:  # noqa: ANN001
    monkeypatch.setattr(sys, "argv", ["train_value.py", *argv, "--device", "cpu"])
    _train_value_tool().main()


@pytest.fixture()
def warm(mb, tmp_path):  # noqa: ANN001, ANN201
    """A seeded M-B model on disk, and the same positions encoded for M-C on disk."""
    mb_encoder, mb_data = mb
    torch.manual_seed(194)
    net = build(mb_encoder, ValueConfig()).eval()
    model = tmp_path / "mb.pt"
    save_model(model, net, net.state_dict(), mb_encoder.vocab, ValueConfig(), meta={"m": 1},
               widths=mb_encoder.widths)
    mc_encoder = Encoder(load_regulation(MC))
    mc_data = _dataset(mc_encoder)
    data = tmp_path / "mc.npz"
    save_dataset(data, mc_data, meta={"format_id": MC, "encoding_revision": ENCODING_REVISION})
    base = predict(net, mb_data, np.arange(len(mb_data)), device=CPU)
    return model, data, mc_encoder, base


def test_init_from_at_zero_epochs_saves_the_model_it_read(warm, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    model, data, mc_encoder, base = warm
    out = tmp_path / "null.pt"
    _run(monkeypatch, "--data", str(data), "--init-from", str(model), "--epochs", "0",
         "--out", str(out))
    net, meta = load_model(out, mc_encoder)
    mc_data = load_dataset(data)
    after = predict(net, mc_data, np.arange(len(mc_data)), device=CPU)
    assert after.tobytes() == base.tobytes()
    blob = torch.load(out, weights_only=False)
    assert blob["format_id"] == MC
    assert blob["vocab_fingerprint"] == mc_encoder.vocab.fingerprint()
    record = meta["init_from"]
    assert record["path"] == str(model) and record["format_id"] == MB
    assert record["vocab_extended_from"] == MB
    assert meta["epochs_run"] == 0
    json.dumps(meta)  # the record stays plain data


def test_init_from_one_epoch_moves_the_scores(warm, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # Positive control for the one above: the comparison can fail.
    model, data, mc_encoder, base = warm
    out = tmp_path / "one.pt"
    _run(monkeypatch, "--data", str(data), "--init-from", str(model), "--epochs", "1",
         "--batch-size", "8", "--out", str(out))
    net, _meta = load_model(out, mc_encoder)
    mc_data = load_dataset(data)
    assert predict(net, mc_data, np.arange(len(mc_data)), device=CPU).tobytes() != base.tobytes()
