"""Lossless small in-memory forms for the encoded training arrays (value-train-memory).

An encoded dataset of 7.8 million decisions is 29 GB as the encoder writes it: float32
per-Pokemon features and int64 ids. Nearly all of that is redundant -- most feature
columns are exactly 0.0 or 1.0 and every id fits in 16 bits -- so the arrays are kept
narrow and widened to the old type only for the rows of a batch. Nothing here rounds: a
column is narrowed only when every value in it survives the round trip bit for bit, and
the batch a net sees is the same array, bit for bit, as from the unpacked dataset.

Reading is streamed: an ``.npz`` member is inflated a chunk at a time and packed as it
arrives, so the dense array (22 GB for the features) never exists.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Rows inflated at a time. 64k rows of (2, 4, 91) float32 features is 190 MB.
CHUNK_ROWS = 1 << 16

_ONE_BITS = np.float32(1.0).view(np.uint32)


class NpzMember:
    """One array of an ``.npz`` read as chunks of rows, never whole."""

    def __init__(self, path: str | Path, key: str) -> None:
        self.path = Path(path)
        self.key = key
        with zipfile.ZipFile(self.path) as zf, zf.open(key + ".npy") as fp:
            version = np.lib.format.read_magic(fp)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
            else:
                raise ValueError(f"{path}[{key}]: npy version {version}")
        if fortran or dtype.hasobject:
            raise ValueError(f"{path}[{key}]: fortran order or object dtype")
        self.shape: tuple[int, ...] = tuple(shape)
        self.dtype = dtype

    def chunks(self, rows: int | None = None) -> Iterator[np.ndarray]:
        rows = rows or CHUNK_ROWS
        row_shape = self.shape[1:]
        row_bytes = int(np.prod(row_shape, dtype=np.int64)) * self.dtype.itemsize
        total = self.shape[0]
        with zipfile.ZipFile(self.path) as zf, zf.open(self.key + ".npy") as fp:
            version = np.lib.format.read_magic(fp)
            if version == (1, 0):
                np.lib.format.read_array_header_1_0(fp)
            else:
                np.lib.format.read_array_header_2_0(fp)
            done = 0
            while done < total:
                n = min(rows, total - done)
                out = np.empty((n, *row_shape), dtype=self.dtype)
                view = memoryview(out.reshape(-1).view(np.uint8))
                want = n * row_bytes
                got = 0
                while got < want:
                    read = fp.readinto(view[got:want])
                    if not read:
                        raise EOFError(f"{self.path}[{self.key}] ends at row {done}")
                    got += read
                yield out
                done += n


def narrow_ints(chunks: Iterator[np.ndarray], total: int, row_shape: tuple[int, ...]) -> np.ndarray:
    """An integer array from its chunks, in the narrowest of int8/int16/int32/int64 that
    holds every value. Widens (copying once) if a later chunk does not fit."""
    order = (np.int8, np.int16, np.int32, np.int64)
    level = 0
    out = np.empty((total, *row_shape), dtype=order[level])
    at = 0
    for chunk in chunks:
        lo, hi = int(chunk.min()), int(chunk.max())
        while not (np.iinfo(order[level]).min <= lo and hi <= np.iinfo(order[level]).max):
            level += 1
            out = out.astype(order[level])
        out[at : at + len(chunk)] = chunk
        at += len(chunk)
    return out


class PackedFloat:
    """A float32 array kept as uint8 for the columns (last axis) that are exactly 0 or 1.

    ``self[index]`` with an index array (or slice, or int) on axis 0 answers the dense
    float32 rows. Any other kind of key materialises the whole array; nothing in training
    does that. ``np.asarray(packed)`` does, on purpose, so a caller that does not know the
    type still gets the right numbers.
    """

    dtype = np.dtype(np.float32)

    def __init__(
        self,
        shape: tuple[int, ...],
        ones: np.ndarray,
        bin_part: np.ndarray,
        wide_part: np.ndarray,
    ) -> None:
        self.shape = tuple(shape)
        #: column numbers held as uint8, and the ones held as float32, both ascending.
        self.bin_cols = ones
        width = shape[-1]
        self.wide_cols = np.setdiff1d(np.arange(width), ones)
        self.bin = bin_part
        self.wide = wide_part

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def nbytes(self) -> int:
        return int(self.bin.nbytes + self.wide.nbytes)

    def _dense(self, bin_rows: np.ndarray, wide_rows: np.ndarray) -> np.ndarray:
        out = np.empty((*bin_rows.shape[:-1], self.shape[-1]), dtype=np.float32)
        out[..., self.bin_cols] = bin_rows
        out[..., self.wide_cols] = wide_rows
        return out

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, (int, np.integer)):
            return self._dense(self.bin[key], self.wide[key])
        if isinstance(key, (slice, np.ndarray, list)):
            return self._dense(self.bin[key], self.wide[key])
        return np.asarray(self)[key]

    def chunks(self, rows: int | None = None) -> Iterator[np.ndarray]:
        rows = rows or CHUNK_ROWS
        for start in range(0, len(self), rows):
            yield self[start : start + rows]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        out = np.empty(self.shape, dtype=np.float32)
        at = 0
        for chunk in self.chunks():
            out[at : at + len(chunk)] = chunk
            at += len(chunk)
        return out if dtype is None else out.astype(dtype)

    # -- building ---------------------------------------------------------------------

    @classmethod
    def from_chunks(cls, chunks: Iterator[np.ndarray], shape: tuple[int, ...]) -> PackedFloat:
        """Packs a float32 array given as chunks of rows, never holding the dense array.

        A column is a uint8 column while every value seen so far has the bits of 0.0 or
        1.0. The first value that is not demotes the column to float32, and its rows so
        far (0s and 1s, exactly) are widened from what was stored.
        """
        total, width = shape[0], shape[-1]
        middle = shape[1:-1]
        ones: np.ndarray | None = None
        bin_part = wide_part = None
        at = 0
        for chunk in chunks:
            if chunk.dtype != np.float32:
                raise TypeError(f"PackedFloat packs float32, got {chunk.dtype}")
            chunk = np.ascontiguousarray(chunk)
            flat = chunk.reshape(-1, width)
            bits = flat.view(np.uint32)
            binary = ((bits == 0) | (bits == _ONE_BITS)).all(axis=0)
            if ones is None:
                ones = np.flatnonzero(binary)
                wide = np.setdiff1d(np.arange(width), ones)
                bin_part = np.empty((total, *middle, len(ones)), dtype=np.uint8)
                wide_part = np.empty((total, *middle, len(wide)), dtype=np.float32)
            else:
                bad = ~binary[ones]
                if bad.any():
                    ones, bin_part, wide_part = _demote(width, ones, bin_part, wide_part, bad, at)
            wide = np.setdiff1d(np.arange(width), ones)
            n = len(chunk)
            bin_part[at : at + n] = chunk[..., ones].astype(np.uint8)
            wide_part[at : at + n] = chunk[..., wide]
            at += n
        if ones is None or at != total:
            raise ValueError(f"expected {total} rows, got {at}")
        return cls(shape, ones, bin_part, wide_part)

    @classmethod
    def from_array(cls, array: np.ndarray) -> PackedFloat:
        array = np.asarray(array, dtype=np.float32)
        return cls.from_chunks(
            (array[i : i + CHUNK_ROWS] for i in range(0, max(len(array), 1), CHUNK_ROWS)),
            array.shape,
        )

    @classmethod
    def concat(cls, parts: Sequence[PackedFloat | np.ndarray]) -> PackedFloat:
        """Joins shards, dense or packed, without building the dense join."""
        shape = (sum(len(p) for p in parts), *parts[0].shape[1:])

        def rows() -> Iterator[np.ndarray]:
            for part in parts:
                if isinstance(part, PackedFloat):
                    yield from part.chunks()
                else:
                    for i in range(0, len(part), CHUNK_ROWS):
                        yield np.ascontiguousarray(part[i : i + CHUNK_ROWS], dtype=np.float32)

        return cls.from_chunks(rows(), shape)


def _demote(
    width: int,
    ones: np.ndarray,
    bin_part: np.ndarray,
    wide_part: np.ndarray,
    bad: np.ndarray,
    at: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moves the uint8 columns flagged in `bad` (a mask over `ones`) to float32. The rows
    stored so far (`at` of them) held 0s and 1s exactly, so they widen exactly."""
    moved = ones[bad]
    keep = ones[~bad]
    old_wide = np.setdiff1d(np.arange(width), ones)
    new_wide = np.setdiff1d(np.arange(width), keep)
    grown = np.empty((*wide_part.shape[:-1], len(new_wide)), dtype=np.float32)
    grown[..., np.searchsorted(new_wide, old_wide)] = wide_part
    if at:
        grown[:at, ..., np.searchsorted(new_wide, moved)] = bin_part[:at][..., bad]
    return keep, np.ascontiguousarray(bin_part[..., ~bad]), grown


class AsType:
    """An array to be written as another dtype, converted a chunk at a time."""

    def __init__(self, array: np.ndarray, dtype: Any) -> None:
        self.array = array
        self.dtype = np.dtype(dtype)
        self.shape = array.shape

    def chunks(self, rows: int | None = None) -> Iterator[np.ndarray]:
        rows = rows or CHUNK_ROWS
        for start in range(0, len(self.array), rows):
            yield np.ascontiguousarray(self.array[start : start + rows], dtype=self.dtype)


def save_npz_streamed(path: str | Path, entries: dict[str, Any]) -> None:
    """``np.savez_compressed`` for entries that may be :class:`PackedFloat` or
    :class:`AsType`: the same npz layout, written a chunk at a time so a dense copy of a
    big array is never built. ``np.load`` reads the result like any other npz."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for key, value in entries.items():
            with zf.open(key + ".npy", "w", force_zip64=True) as fid:
                if isinstance(value, (PackedFloat, AsType)):
                    np.lib.format.write_array_header_1_0(
                        fid,
                        {
                            "descr": np.lib.format.dtype_to_descr(np.dtype(value.dtype)),
                            "fortran_order": False,
                            "shape": tuple(value.shape),
                        },
                    )
                    for chunk in value.chunks():
                        fid.write(np.ascontiguousarray(chunk).tobytes())
                else:
                    np.lib.format.write_array(fid, np.asanyarray(value), allow_pickle=False)
