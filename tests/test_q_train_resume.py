"""`q_train.py` cut into calls gives the same Q bit for bit as one call (IKA-348).

A 30-epoch Q takes about an hour; the machine's rule is a call of 30 minutes at most, so the
run is resumed from a checkpoint written after every epoch. The claim is exact: the weights
after 1 + 1 + 1 epochs in three calls are the weights after 3 epochs in one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import qhead
from pokeuraou.encode import Encoder

ROOT = Path(__file__).resolve().parents[1]
ENCODED = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def _q_train():
    spec = importlib.util.spec_from_file_location("q_train_under_test", ROOT / "tools" / "q_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shards(reg, where: Path, views: int = 6) -> Path:
    """One shard of `views` real positions, each its own game, with random matrices."""
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import all_selections, load_roster

    roster = load_roster("rizabanadohido")
    picks = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
    encoder = Encoder(reg)
    rng = np.random.default_rng(348)
    parts: dict[str, list[np.ndarray]] = {f"enc_{n}": [] for n in ENCODED}
    n0s, n1s, cells, acts0, acts1 = [], [], [], [], []
    for k in range(views):
        pos = position_from_sets(
            reg,
            [roster.sets[i] for i in picks[k % len(picks)]],
            [roster.sets[i] for i in picks[(k + 3) % len(picks)]],
        )
        enc = encoder.encode_positions([pos])
        for n in ENCODED:
            parts[f"enc_{n}"].append(np.asarray(getattr(enc, n)))
        a0 = qhead.encode_actions(encoder.vocab, pos, 0, qhead.legal_pool(reg, pos, 0))[:12]
        a1 = qhead.encode_actions(encoder.vocab, pos, 1, qhead.legal_pool(reg, pos, 1))[:10]
        acts0.append(a0)
        acts1.append(a1)
        n0s.append(len(a0))
        n1s.append(len(a1))
        cells.append(rng.uniform(0.05, 0.95, size=len(a0) * len(a1)).astype(np.float32))
    where.mkdir(parents=True)
    np.savez(
        where / "shard-000000.npz",
        k=np.arange(views), side=np.zeros(views, dtype=np.int8), turn=np.ones(views, dtype=np.int16),
        game=np.arange(views), n0=np.array(n0s), n1=np.array(n1s),
        cells=np.concatenate(cells), cell_start=np.cumsum([0, *[c.size for c in cells[:-1]]]),
        acts0=np.concatenate(acts0), acts1=np.concatenate(acts1),
        act0_start=np.cumsum([0, *n0s[:-1]]), act1_start=np.cumsum([0, *n1s[:-1]]),
        **{name: np.concatenate(v) for name, v in parts.items()},
    )
    return where


def _argv(shards: Path, out: Path, *more: str) -> list[str]:
    return ["--shards", str(shards), "--out", str(out), "--epochs", "3", "--batch", "2",
            "--holdout", "0.34", "--eval-limit", "2", "--device", "cpu", "--seed", "5", *more]


def test_a_run_cut_into_calls_ends_with_the_weights_of_one_call(reg, tmp_path):
    torch = pytest.importorskip("torch")
    q_train = _q_train()
    shards = _shards(reg, tmp_path / "shards")
    whole = tmp_path / "whole.pt"
    q_train.main(_argv(shards, whole))

    pieces = tmp_path / "pieces.pt"
    ckpt = tmp_path / "q.ckpt"
    for call in range(3):
        q_train.main(_argv(shards, pieces, "--checkpoint", str(ckpt), "--stop-after", "1"))
        assert pieces.exists() == (call == 2)  # the model only once the last epoch is done

    a = torch.load(whole, map_location="cpu", weights_only=False)
    b = torch.load(pieces, map_location="cpu", weights_only=False)
    assert b["calls"] == [0, 1, 2] and a["calls"] == [0]
    assert [h["epoch"] for h in b["history"]] == [1, 2, 3]
    assert a["state"].keys() == b["state"].keys()
    moved = 0
    for name, tensor in a["state"].items():
        assert torch.equal(tensor, b["state"][name]), name
        moved += 0 if tensor.dtype == torch.bool else 1
    assert moved > 10
    assert [h["train_bce"] for h in a["history"]] == [h["train_bce"] for h in b["history"]]


def test_a_checkpoint_of_another_run_is_refused(reg, tmp_path):
    pytest.importorskip("torch")
    q_train = _q_train()
    shards = _shards(reg, tmp_path / "shards")
    ckpt = tmp_path / "q.ckpt"
    q_train.main(_argv(shards, tmp_path / "q.pt", "--checkpoint", str(ckpt), "--stop-after", "1"))
    with pytest.raises(SystemExit, match="lr"):
        q_train.main(_argv(shards, tmp_path / "q.pt", "--checkpoint", str(ckpt), "--lr", "0.002"))
