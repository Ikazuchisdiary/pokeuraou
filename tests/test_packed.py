"""The packed dataset (`pokeuraou.packed`, `load_dataset(packed=True)`) is the same numbers
in less memory: every batch it makes is bit-identical to the plain dataset's."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="the dataset needs the optional learn group")

import torch  # noqa: E402

from pokeuraou import packed  # noqa: E402
from pokeuraou.encode import Encoded  # noqa: E402
from pokeuraou.packed import PackedFloat  # noqa: E402
from pokeuraou.value import Dataset, concat_datasets, load_dataset, save_dataset  # noqa: E402


def _dataset(n: int, seed: int, *, late_half: bool = False) -> Dataset:
    rng = np.random.default_rng(seed)
    mon = (rng.random((n, 2, 4, 9)) < 0.3).astype(np.float32)  # columns of 0s and 1s
    mon[..., 2] = rng.random((n, 2, 4)).astype(np.float32)  # a real-valued column
    mon[..., 3] = np.float32(-0.0)  # equal to 0.0, not the same bits: must stay float32
    if late_half:
        # binary in every early row, not in the last chunk: the demotion path
        mon[-5:, 0, 0, 5] = np.float32(0.5)
    mon[0, 0, 0, 6] = np.nan  # NaN != NaN, so this only passes if bits are compared
    encoded = Encoded(
        species=rng.integers(0, 300, (n, 2, 4)).astype(np.int64),
        ability=rng.integers(0, 70_000, (n, 2, 4)).astype(np.int64),  # does not fit int16
        item=rng.integers(0, 100, (n, 2, 4)).astype(np.int64),
        moves=rng.integers(0, 500, (n, 2, 4, 4)).astype(np.int64),
        mon=mon,
        mask=(rng.random((n, 2, 4)) < 0.8).astype(np.float32),
        side=(rng.random((n, 2, 3)) < 0.5).astype(np.float32),
        field=rng.random((n, 5)).astype(np.float32),
        unknown_volatiles={"ghost": n},
    )
    return Dataset(
        encoded=encoded,
        outcome=(rng.random(n) < 0.5).astype(np.float32),
        game=np.arange(n, dtype=np.int32) // 3,
        turn=np.arange(n, dtype=np.int16),
        search_value=rng.random(n).astype(np.float32),
        hp_share=rng.random(n).astype(np.float32),
        kind=np.zeros(n, np.int8),
        foe=np.zeros(n, np.int32),
        foe_names=("a",),
    )


def _same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].dtype == b[key].dtype, key
        assert a[key].numpy().tobytes() == b[key].numpy().tobytes(), key


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packed, "CHUNK_ROWS", 7)


def test_packed_load_makes_the_same_batches(tmp_path, small_chunks) -> None:  # noqa: ANN001
    plain = _dataset(40, 1, late_half=True)
    path = tmp_path / "d.npz"
    save_dataset(path, plain, {"format_id": "x"})
    back = load_dataset(path)
    assert isinstance(back.encoded.mon, PackedFloat)
    # Positive controls: the packing ran (uint8 columns exist, ids narrowed, the id that
    # does not fit 16 bits widened), and the demotion path ran (column 5 had a late 0.5).
    assert back.encoded.mon.bin.dtype == np.uint8 and back.encoded.mon.bin.shape[-1] > 0
    assert 5 in back.encoded.mon.wide_cols and 3 in back.encoded.mon.wide_cols
    assert back.encoded.species.dtype == np.int16
    assert back.encoded.ability.dtype == np.int32
    assert back.encoded.mon.nbytes < plain.encoded.mon.nbytes
    index = np.array([39, 0, 5, 5, 17, 38])
    cpu = torch.device("cpu")
    _same(plain.tensors(index, cpu), back.tensors(index, cpu))
    _same(plain.tensors(np.arange(40), cpu), back.tensors(np.arange(40), cpu))
    assert np.asarray(back.encoded.mon).tobytes() == plain.encoded.mon.tobytes()
    assert back.encoded.unknown_volatiles == {"ghost": 40}


def test_the_comparison_can_fail(tmp_path, small_chunks) -> None:  # noqa: ANN001
    """Control: a packed copy with one value changed is caught by `_same`."""
    plain = _dataset(20, 2)
    path = tmp_path / "d.npz"
    save_dataset(path, plain, {"format_id": "x"})
    back = load_dataset(path)
    back.encoded.mon.wide[3, 0, 0, 0] += np.float32(1e-7)
    cpu = torch.device("cpu")
    with pytest.raises(AssertionError):
        _same(plain.tensors(np.arange(20), cpu), back.tensors(np.arange(20), cpu))


def test_unpacked_load_is_the_plain_read(tmp_path) -> None:  # noqa: ANN001
    plain = _dataset(12, 3)
    path = tmp_path / "d.npz"
    save_dataset(path, plain, {"format_id": "x"})
    back = load_dataset(path, packed=False)
    assert isinstance(back.encoded.mon, np.ndarray)
    assert back.encoded.species.dtype == np.int64
    assert back.encoded.mon.tobytes() == plain.encoded.mon.tobytes()


def test_concat_of_packed_shards_and_a_streamed_save(tmp_path, small_chunks) -> None:  # noqa: ANN001
    a, b = _dataset(23, 4), _dataset(17, 5, late_half=True)
    paths = []
    for name, d in (("a", a), ("b", b)):
        paths.append(tmp_path / f"{name}.npz")
        save_dataset(paths[-1], d, {"format_id": "x"})
    joined = concat_datasets([load_dataset(p) for p in paths])
    assert isinstance(joined.encoded.mon, PackedFloat)
    want = concat_datasets([a, b])
    out = tmp_path / "joined.npz"
    save_dataset(out, joined, {"format_id": "x"})  # streamed: same layout, same dtypes
    with np.load(out) as data:
        for key in ("species", "ability", "item", "moves"):
            assert data[key].dtype == np.int64 and np.array_equal(
                data[key], getattr(want.encoded, key)
            )
        for key in ("mon", "mask", "side", "field"):
            assert data[key].tobytes() == getattr(want.encoded, key).tobytes(), key
        assert np.array_equal(data["game"], want.game)
    back = load_dataset(out, packed=False)
    assert back.encoded.mon.tobytes() == want.encoded.mon.tobytes()
