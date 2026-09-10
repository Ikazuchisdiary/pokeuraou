"""The value function's guarantees, as opposed to its accuracy.

Accuracy is measured by ``tools/train_value.py`` on held-out games and is a property of
the data. What belongs in a test is the set of things that must hold for *any* weights,
because each of them is a way for a trained model to be confidently meaningless:

- ``V(x) + V(mirror x) = 1`` exactly, at initialisation and after training, because the
  game is zero-sum and the architecture is supposed to make that free rather than learned;
- a train/validation split that never puts two decisions from the same game on both sides,
  since they share one label and the validation number would measure memorisation;
- weights that refuse to load against a vocabulary they were not trained on, because the
  same integer indexes a different Pokemon after a regulation rotation.

Skipped when torch is absent: it lives in the optional ``learn`` group so that the solver
and the differential tests stay installable without a CUDA wheel.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="value function needs the optional learn group")

from pokeuraou.encode import Encoder, build_vocabulary  # noqa: E402
from pokeuraou.priors import build_cooccurrence, find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.selfplay import position_from_sets  # noqa: E402
from pokeuraou.teams import load_roster, sample_metagame_team  # noqa: E402
from pokeuraou.value import (  # noqa: E402
    Dataset,
    ValueConfig,
    auc,
    build,
    load_model,
    predict,
    save_model,
    train,
)


@pytest.fixture(scope="module")
def bundle():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    roster = load_roster("rizabanadohido")
    cached = find_cached_chaos(reg.meta.format_id)
    if cached is None:
        pytest.skip("no cached usage stats")
    prior = load_chaos(cached, reg)
    cooc = build_cooccurrence(prior)
    rng = np.random.default_rng(11)

    positions = []
    games = []
    outcomes = []
    for game in range(24):
        foe = sample_metagame_team(rng, reg, prior, cooc)
        pos = position_from_sets(
            reg, roster.sets[: reg.meta.picked_team_size], foe[: reg.meta.picked_team_size]
        )
        result = float(rng.integers(2))
        for _ in range(5):
            positions.append(pos.to_json())
            games.append(game)
            outcomes.append(result)

    encoder = Encoder(reg)
    encoded = encoder.encode(positions)
    dataset = Dataset(
        encoded=encoded,
        outcome=np.array(outcomes, dtype=np.float32),
        game=np.array(games, dtype=np.int32),
        turn=np.ones(len(positions), dtype=np.int16),
        proxy=np.full(len(positions), 0.5, dtype=np.float32),
        kind=np.zeros(len(positions), dtype=np.int8),
        foe=np.zeros(len(positions), dtype=np.int32),
        foe_names=("metagame",),
    )
    return encoder, dataset


def _flip(batch):  # noqa: ANN001, ANN202
    return {
        k: (v.flip(1) if v.dim() > 1 and v.shape[1] == 2 else v) for k, v in batch.items()
    }


def test_the_value_is_antisymmetric_at_initialisation(bundle) -> None:  # noqa: ANN001
    encoder, dataset = bundle
    device = torch.device("cpu")
    net = build(encoder, ValueConfig()).to(device)
    net.eval()
    batch = dataset.tensors(np.arange(len(dataset)), device)
    with torch.no_grad():
        forward = torch.sigmoid(net(batch))
        mirrored = torch.sigmoid(net(_flip(batch)))
    # Zero-sum: the two sides' win probabilities have to sum to one, and the architecture
    # is meant to make that hold for arbitrary weights rather than after fitting.
    assert float((forward + mirrored - 1.0).abs().max()) < 1e-5


def test_it_stays_antisymmetric_after_training(bundle) -> None:  # noqa: ANN001
    encoder, dataset = bundle
    device = torch.device("cpu")
    config = ValueConfig(epochs=2, batch_size=32, patience=2)
    net = build(encoder, config).to(device)
    _history, best = train(net, dataset, config, device=device, holdout=0.25)
    net.load_state_dict(best)
    net.eval()
    batch = dataset.tensors(np.arange(len(dataset)), device)
    with torch.no_grad():
        forward = torch.sigmoid(net(batch))
        mirrored = torch.sigmoid(net(_flip(batch)))
    assert float((forward + mirrored - 1.0).abs().max()) < 1e-5


def test_a_balanced_position_is_exactly_even(bundle) -> None:  # noqa: ANN001
    """A mirror match with identical spreads has to come out at 0.500.

    It is the one position whose answer is known without any data, and antisymmetry forces
    it: if both sides are the same, V = 1 - V.
    """
    encoder, _dataset = bundle
    reg = encoder.reg
    roster = load_roster("rizabanadohido")
    four = roster.sets[: reg.meta.picked_team_size]
    position = position_from_sets(reg, four, four).to_json()

    net = build(encoder, ValueConfig())
    net.eval()
    batch = Dataset(
        encoded=encoder.encode([position]),
        outcome=np.zeros(1, dtype=np.float32),
        game=np.zeros(1, dtype=np.int32),
        turn=np.ones(1, dtype=np.int16),
        proxy=np.zeros(1, dtype=np.float32),
        kind=np.zeros(1, dtype=np.int8),
    ).tensors(np.arange(1), torch.device("cpu"))
    with torch.no_grad():
        probability = float(torch.sigmoid(net(batch))[0])
    assert probability == pytest.approx(0.5, abs=1e-6)


def test_the_split_never_shares_a_game(bundle) -> None:  # noqa: ANN001
    _encoder, dataset = bundle
    train_idx, val_idx = dataset.split_by_game(0.25, seed=5)
    assert len(train_idx) + len(val_idx) == len(dataset)
    assert not set(dataset.game[train_idx].tolist()) & set(dataset.game[val_idx].tolist())
    assert len(val_idx) > 0


def test_saving_and_loading_reproduces_the_predictions(bundle, tmp_path) -> None:  # noqa: ANN001
    encoder, dataset = bundle
    device = torch.device("cpu")
    config = ValueConfig(epochs=1, batch_size=32, patience=1)
    net = build(encoder, config).to(device)
    index = np.arange(len(dataset))
    before = predict(net, dataset, index, device=device)

    path = tmp_path / "value.pt"
    save_model(path, net, net.state_dict(), encoder.vocab, config, meta={"note": "test"})
    loaded, meta = load_model(path, encoder)
    after = predict(loaded, dataset, index, device=device)
    assert meta["note"] == "test"
    assert np.allclose(before, after, atol=1e-6)


def test_loading_refuses_another_regulation(bundle, tmp_path) -> None:  # noqa: ANN001
    encoder, _dataset = bundle
    config = ValueConfig()
    net = build(encoder, config)
    path = tmp_path / "value.pt"
    save_model(path, net, net.state_dict(), encoder.vocab, config, meta={})

    other = Encoder(load_regulation("gen9championsvgc2026regmc"))
    with pytest.raises(ValueError, match="trained on"):
        load_model(path, other)


def test_loading_refuses_a_changed_vocabulary(bundle, tmp_path) -> None:  # noqa: ANN001
    encoder, _dataset = bundle
    config = ValueConfig()
    net = build(encoder, config)
    path = tmp_path / "value.pt"
    save_model(path, net, net.state_dict(), encoder.vocab, config, meta={})

    # Same regulation id, different mapping -- what a re-dump with a new species would do.
    vocab = build_vocabulary(encoder.reg)
    shifted = type(vocab)(
        format_id=vocab.format_id,
        species={k: v + 1 for k, v in vocab.species.items()},
        abilities=vocab.abilities,
        items=vocab.items,
        moves=vocab.moves,
        types=vocab.types,
    )
    tampered = Encoder(encoder.reg)
    object.__setattr__(tampered, "vocab", shifted)
    with pytest.raises(ValueError, match="fingerprint"):
        load_model(path, tampered)


def test_auc_is_the_rank_statistic() -> None:
    # Perfect separation, perfect inversion, and a constant score.
    assert auc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([0, 0, 1, 1])) == 0.0
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 0, 1, 1])) == 0.5
