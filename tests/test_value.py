"""The value function's guarantees, as opposed to its accuracy.

Accuracy is measured by ``tools/train_value.py`` on held-out games and is a property of
the data. What belongs in a test is the set of things that must hold for *any* weights,
because each of them is a way for a trained model to be confidently meaningless:

- ``V(x) + V(mirror x) = 1`` exactly, at initialisation and after training, because the
  game is zero-sum and the architecture is supposed to make that free rather than learned;
- a train/validation split that never puts two decisions from the same game on both sides,
  since they share one label and the validation number would measure memorisation;
- a split that follows `split_seed` and not `seed`, because `seed` also moves
  initialisation and batch order, and two runs separated by it were being marked against
  different games as well as fitted differently -- a difference in what the number is
  about, which no number of repeats averages away;
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
    split_for,
    train,
)

from .test_concat_datasets import shard  # noqa: E402


@pytest.fixture(scope="module")
def encoder():  # noqa: ANN201
    """The vocabulary alone, which needs the regulation dump and nothing from `data/`.

    Apart from `bundle`, which samples real opponents and so needs usage stats on disk.
    The things that are true of any weights -- the split, what a saved model records --
    are true without a single real game, and a test that skipped for want of a chaos
    file would be reporting on the fixture rather than on the code.
    """
    return Encoder(load_regulation("gen9championsvgc2026regmb"))


@pytest.fixture(scope="module")
def bundle(encoder):  # noqa: ANN001, ANN201
    reg = encoder.reg
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

    encoded = encoder.encode(positions)
    dataset = Dataset(
        encoded=encoded,
        outcome=np.array(outcomes, dtype=np.float32),
        game=np.array(games, dtype=np.int32),
        turn=np.ones(len(positions), dtype=np.int16),
        search_value=np.full(len(positions), 0.5, dtype=np.float32),
        hp_share=np.full(len(positions), 0.5, dtype=np.float32),
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
        search_value=np.zeros(1, dtype=np.float32),
        hp_share=np.zeros(1, dtype=np.float32),
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


def pool() -> Dataset:
    """Twenty-four games of five decisions each, with nothing real in them.

    Which games a seed holds out is arithmetic over the `game` column and does not look at
    a single position, so these tests take the synthetic shard rather than `bundle`: a
    split test that skipped for want of a usage-stats file would be reporting on the
    fixture.
    """
    return shard(120, games=24, foe_names=("worlds",), outcome=1.0)


def held_out(dataset: Dataset, holdout: float, config: ValueConfig) -> frozenset[int]:
    _train_idx, val_idx = split_for(dataset, holdout, config)
    return frozenset(dataset.game[val_idx].tolist())


def test_the_same_split_seed_holds_out_the_same_games() -> None:
    """Two runs that differ only in how they were fitted have to be marked against the
    same games.

    `seed` moves initialisation and batch order *and* the split, so before the split had
    its own seed, a pair of runs compared to decide something were also being asked a
    different question each: part of the gap between their validation numbers was which
    games happened to be easy. That part does not average away with more repeats -- it is
    not noise around one quantity, it is two quantities.
    """
    dataset = pool()
    fitted_one_way = held_out(dataset, 0.25, ValueConfig(seed=1, split_seed=3))
    fitted_another_way = held_out(dataset, 0.25, ValueConfig(seed=2, split_seed=3))
    assert fitted_one_way == fitted_another_way


def test_a_different_split_seed_holds_out_different_games() -> None:
    """The other half of the same claim: the seed has to reach the splitter.

    A flag that is parsed, recorded and never read would pass the test above perfectly --
    every split identical is exactly what "it changes nothing" looks like -- so the two
    are only worth anything together.
    """
    dataset = pool()
    by_seed = {
        seed: held_out(dataset, 0.25, ValueConfig(split_seed=seed)) for seed in range(4)
    }
    assert len(set(by_seed.values())) == 4


def test_no_split_seed_follows_the_fit_seed() -> None:
    """`None` is what a config stored before the field existed reads back as.

    Those runs took their split from `seed`, so resolving `None` to 0 instead would file a
    model under a description of games it was never marked against. The training tool
    defaults its flag to 0 rather than to `None`, which is a statement about new runs; this
    is about old records, and they disagree only for a run that passed `--seed`.
    """
    dataset = pool()
    followed = held_out(dataset, 0.25, ValueConfig(seed=4))
    assert followed == frozenset(dataset.game[dataset.split_by_game(0.25, 4)[1]].tolist())


def test_every_split_seed_still_splits_by_game() -> None:
    """Whichever games are held out, no game may be on both sides: they share one label."""
    dataset = pool()
    for seed in range(4):
        train_idx, val_idx = split_for(dataset, 0.25, ValueConfig(split_seed=seed))
        assert len(train_idx) + len(val_idx) == len(dataset)
        assert len(val_idx) > 0
        assert not set(dataset.game[train_idx].tolist()) & set(
            dataset.game[val_idx].tolist()
        )


def test_a_saved_model_records_both_seeds(encoder, tmp_path) -> None:  # noqa: ANN001
    """A model's record has to name the games its validation number was measured on.

    Two models reporting different AUCs are not comparable unless the record says whether
    they were marked against the same games, and one number standing for both the fit and
    the split cannot say it.
    """
    config = ValueConfig(epochs=1, batch_size=32, patience=1, seed=1, split_seed=3)
    net = build(encoder, config)
    path = tmp_path / "value.pt"
    save_model(
        path,
        net,
        net.state_dict(),
        encoder.vocab,
        config,
        meta={"seed": config.seed, "split_seed": config.split_seed},
    )
    loaded, meta = load_model(path, encoder)
    assert (meta["seed"], meta["split_seed"]) == (1, 3)
    assert loaded.config.split_seed == 3


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
