"""Lossless small in-memory forms for the encoded training arrays (value-train-memory).

An encoded dataset of 7.8 million decisions is 29 GB as the encoder writes it: float32
per-Pokemon features and int64 ids. Nearly all of that is redundant -- most feature
columns are exactly 0.0 or 1.0, most of the others take a few dozen values, and every id
fits in 16 bits -- so the arrays are kept narrow and widened to the old type only for the
rows of a batch. Nothing here rounds: a column is narrowed only by a form that gives back
every value bit for bit, and the batch a net sees is the same array, bit for bit, as from
the unpacked dataset.

The float columns (last axis) of a :class:`PackedFloat` are held in one of four forms
(IKA-426):

* a column of only 0.0 and 1.0: one bit, eight columns to a byte;
* a column of at most 256 distinct values: a uint8 code into a per-column table;
* at most 65,536 distinct values: a uint16 code;
* anything else: float32 as it was.

The tables are keyed on the float32 bits, so -0.0 and 0.0, or two NaNs, stay apart. On
the gen-4 pool (12.4 million decisions) the per-Pokemon features go from 16.2 GB (one
byte per 0/1 column, float32 for the rest) to about 3.4 GB.

Reading is streamed: an ``.npz`` member is inflated a chunk at a time and packed as it
arrives, so the dense array (36 GB for the gen-4 features) never exists.
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

#: The forms a float column can be held in, narrowest first.
BIN, U8, U16, F32 = 0, 1, 2, 3
_CAP = {U8: 256, U16: 65536}
_DTYPE = {BIN: np.uint8, U8: np.uint8, U16: np.uint16, F32: np.float32}


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


class _Dictionary:
    """One column's distinct values as float32 bits, numbered in the order first seen."""

    def __init__(self, keys: np.ndarray) -> None:
        self.keys = np.asarray(keys, dtype=np.uint32)
        self._index()

    def _index(self) -> None:
        self.order = np.argsort(self.keys, kind="stable")
        self.sorted = self.keys[self.order]

    def __len__(self) -> int:
        return len(self.keys)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """The codes of `bits`, adding the values not seen before (in ascending order)."""
        if len(self.sorted):
            at = np.minimum(np.searchsorted(self.sorted, bits), len(self.sorted) - 1)
            hit = self.sorted[at] == bits
            if hit.all():
                return self.order[at]
            new = np.unique(bits[~hit])
        else:
            new = np.unique(bits)
        self.keys = np.concatenate([self.keys, new])
        self._index()
        return self.order[np.searchsorted(self.sorted, bits)]

    def table(self, size: int) -> np.ndarray:
        out = np.zeros(size, dtype=np.uint32)
        out[: len(self.keys)] = self.keys
        return out.view(np.float32)


class PackedFloat:
    """A float32 array whose columns (last axis) are held in the narrowest exact form.

    ``self[index]`` with an index array (or slice, or int) on axis 0 answers the dense
    float32 rows. Any other kind of key materialises the whole array; nothing in training
    does that. ``np.asarray(packed)`` does, on purpose, so a caller that does not know the
    type still gets the right numbers.
    """

    dtype = np.dtype(np.float32)

    def __init__(
        self,
        shape: tuple[int, ...],
        cols: dict[int, np.ndarray],
        parts: dict[int, np.ndarray],
        tables: dict[int, np.ndarray],
    ) -> None:
        self.shape = tuple(shape)
        #: column numbers held in each form, in the order of the last axis of its part.
        self.cols = cols
        #: BIN: bits packed little-endian along the last axis; U8/U16: codes; F32: values.
        self.parts = parts
        #: U8/U16: (columns, 256 or 65536) float32, row j the table of the part's column j.
        self.tables = tables
        #: row j of a table for the part's column j, broadcast against a batch of codes.
        self._rows = {form: np.arange(len(cols[form])) for form in (U8, U16)}

    # The names the earlier two-form layout had: columns of 0s and 1s, and the rest.
    @property
    def bin_cols(self) -> np.ndarray:
        return self.cols[BIN]

    @property
    def wide_cols(self) -> np.ndarray:
        return np.sort(np.concatenate([self.cols[U8], self.cols[U16], self.cols[F32]]))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def nbytes(self) -> int:
        return int(sum(p.nbytes for p in self.parts.values()) + sum(
            t.nbytes for t in self.tables.values()
        ))

    def _dense(self, key: Any) -> np.ndarray:
        bits = self.parts[BIN][key]
        out = np.empty((*bits.shape[:-1], self.shape[-1]), dtype=np.float32)
        if len(self.cols[BIN]):
            out[..., self.cols[BIN]] = np.unpackbits(
                bits, axis=-1, count=len(self.cols[BIN]), bitorder="little"
            )
        for form in (U8, U16):
            if len(self.cols[form]):
                out[..., self.cols[form]] = self.tables[form][self._rows[form], self.parts[form][key]]
        if len(self.cols[F32]):
            out[..., self.cols[F32]] = self.parts[F32][key]
        return out

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, (int, np.integer, slice, np.ndarray, list)):
            return self._dense(key)
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

        The first chunk puts each column in the narrowest form that holds it. A later
        chunk that does not fit moves the column to a wider form, and the rows stored so
        far are rewritten exactly into it (`_Builder.move`).
        """
        builder = _Builder(shape)
        for chunk in chunks:
            builder.add(chunk)
        return builder.finish()

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


def _width(form: int, n: int) -> int:
    """Last-axis length of a part holding `n` columns."""
    return (n + 7) // 8 if form == BIN else n


class _Builder:
    """The state of a :class:`PackedFloat` while its chunks arrive."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        self.total, self.width = shape[0], shape[-1]
        self.middle = tuple(shape[1:-1])
        self.cols: dict[int, list[int]] | None = None
        self.parts: dict[int, np.ndarray] = {}
        self.dicts: dict[int, _Dictionary] = {}
        self.at = 0

    def _empty(self, form: int, n: int) -> np.ndarray:
        return np.empty((self.total, *self.middle, _width(form, n)), dtype=_DTYPE[form])

    def _start(self, flat: np.ndarray, binary: np.ndarray) -> None:
        self.cols = {BIN: [], U8: [], U16: [], F32: []}
        for c in range(self.width):
            if binary[c]:
                self.cols[BIN].append(c)
                continue
            seen = np.unique(flat[:, c])
            form = U8 if len(seen) <= _CAP[U8] else U16 if len(seen) <= _CAP[U16] else F32
            self.cols[form].append(c)
            if form != F32:
                self.dicts[c] = _Dictionary(seen)
        self.parts = {form: self._empty(form, len(c)) for form, c in self.cols.items()}

    def _column(self, form: int, j: int, start: int, stop: int) -> np.ndarray:
        """Stored rows [start, stop) of the part's j-th column, in that part's dtype."""
        part = self.parts[form][start:stop]
        if form == BIN:
            return ((part[..., j // 8] >> (j % 8)) & 1).astype(np.uint8)
        return part[..., j]

    def move(self, c: int, to: int) -> None:
        """Moves column `c` to the wider form `to`, rewriting the stored rows exactly."""
        assert self.cols is not None
        old = next(f for f, cs in self.cols.items() if c in cs)
        j = self.cols[old].index(c)
        if old == BIN:
            # 0.0 then 1.0: the stored bit is the code.
            self.dicts[c] = _Dictionary(np.array([0, _ONE_BITS], dtype=np.uint32))
        keep = [i for i in range(len(self.cols[old])) if i != j]
        old_part = self._empty(old, len(keep))
        new_part = self._empty(to, len(self.cols[to]) + 1)
        for start in range(0, self.at, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, self.at)
            value = self._column(old, j, start, stop)
            if to == F32:
                value = self.dicts[c].table(len(self.dicts[c]))[value]
            if old == BIN:
                kept = np.unpackbits(
                    self.parts[BIN][start:stop], axis=-1, count=len(self.cols[BIN]),
                    bitorder="little",
                )[..., keep]
                old_part[start:stop] = np.packbits(kept, axis=-1, bitorder="little")
            else:
                old_part[start:stop] = self.parts[old][start:stop][..., keep]
            new_part[start:stop, ..., :-1] = self.parts[to][start:stop]
            new_part[start:stop, ..., -1] = value
        self.parts[old], self.parts[to] = old_part, new_part
        self.cols[old].remove(c)
        self.cols[to].append(c)
        if to == F32:
            del self.dicts[c]

    def add(self, chunk: np.ndarray) -> None:
        if chunk.dtype != np.float32:
            raise TypeError(f"PackedFloat packs float32, got {chunk.dtype}")
        chunk = np.ascontiguousarray(chunk)
        n = len(chunk)
        flat = chunk.reshape(-1, self.width).view(np.uint32)
        binary = ((flat == 0) | (flat == _ONE_BITS)).all(axis=0)
        if self.cols is None:
            self._start(flat, binary)
        assert self.cols is not None
        for c in [c for c in self.cols[BIN] if not binary[c]]:
            self.move(c, U8)
        codes: dict[int, np.ndarray] = {}
        for c in self.cols[U8] + self.cols[U16]:
            code = self.dicts[c].encode(flat[:, c])
            size = len(self.dicts[c])
            if size > _CAP[U16]:
                self.move(c, F32)  # written below with the other float32 columns
                continue
            if size > _CAP[U8] and c in self.cols[U8]:
                self.move(c, U16)
            codes[c] = code
        rows = slice(self.at, self.at + n)
        lead = (n, *self.middle)
        if self.cols[BIN]:
            self.parts[BIN][rows] = np.packbits(
                (flat[:, self.cols[BIN]] != 0).reshape(*lead, -1), axis=-1, bitorder="little"
            )
        for form in (U8, U16):
            if self.cols[form]:
                block = np.empty((*lead, len(self.cols[form])), dtype=_DTYPE[form])
                for j, c in enumerate(self.cols[form]):
                    block[..., j] = codes[c].reshape(lead)
                self.parts[form][rows] = block
        if self.cols[F32]:
            self.parts[F32][rows] = chunk[..., self.cols[F32]]
        self.at += n

    def finish(self) -> PackedFloat:
        if self.cols is None or self.at != self.total:
            raise ValueError(f"expected {self.total} rows, got {self.at}")
        tables = {
            form: (
                np.stack([self.dicts[c].table(_CAP[form]) for c in self.cols[form]])
                if self.cols[form]
                else np.zeros((0, _CAP[form]), dtype=np.float32)
            )
            for form in (U8, U16)
        }
        cols = {form: np.array(cs, dtype=np.int64) for form, cs in self.cols.items()}
        return PackedFloat(self.shape, cols, self.parts, tables)


#: Bumped whenever the files below change meaning; a cache of another version is rebuilt.
CACHE_VERSION = 1


def cache_dir(source: str | Path) -> Path:
    """Where the packed arrays of `source` (an encoded ``.npz``) are kept on disk.

    Keyed on the CRC-32 and length of every member, read from the zip's central
    directory without inflating anything, so a re-encoded file (a new
    `ENCODING_REVISION`, more games) gets a cache of its own and never reads another's,
    and a copy of the same file finds the same cache.
    """
    import hashlib

    source = Path(source)
    with zipfile.ZipFile(source) as zf:
        members = sorted((i.filename, i.CRC, i.file_size) for i in zf.infolist())
    key = hashlib.sha256(repr(members).encode()).hexdigest()[:20]
    return source.with_name(source.name + ".packed") / f"{key}-v{CACHE_VERSION}"


def save_cache(directory: Path, arrays: dict[str, np.ndarray | PackedFloat]) -> None:
    """Writes the packed arrays as plain ``.npy`` files, to be mapped by `load_cache`.

    Written into a private temporary directory and renamed into place, so a reader sees
    a whole cache or none; when two runs build the same cache at once the second rename
    fails and its copy is dropped.
    """
    import json
    import os
    import shutil

    tmp = directory.with_name(f"{directory.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    index: dict[str, Any] = {}
    for key, value in arrays.items():
        if isinstance(value, PackedFloat):
            index[key] = {
                "shape": list(value.shape),
                "cols": {str(form): value.cols[form].tolist() for form in value.cols},
            }
            for form, part in value.parts.items():
                np.save(tmp / f"{key}.{form}.npy", part, allow_pickle=False)
            for form, table in value.tables.items():
                np.save(tmp / f"{key}.table{form}.npy", table, allow_pickle=False)
        else:
            index[key] = None
            np.save(tmp / f"{key}.npy", np.ascontiguousarray(value), allow_pickle=False)
    (tmp / "index.json").write_bytes(json.dumps(index).encode())
    try:
        tmp.rename(directory)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not (directory / "index.json").exists():
            raise


def load_cache(directory: Path) -> dict[str, np.ndarray | PackedFloat] | None:
    """The arrays `save_cache` wrote, memory-mapped read-only, or None if there is no
    whole cache. A mapped file is backed by the file, not by the page file: it adds
    nothing to a process's commit, and runs that map the same cache share one copy of
    its pages."""
    import json

    path = directory / "index.json"
    if not path.exists():
        return None
    index = json.loads(path.read_bytes())
    out: dict[str, np.ndarray | PackedFloat] = {}
    for key, info in index.items():
        if info is None:
            out[key] = _map(directory / f"{key}.npy")
            continue
        cols = {int(f): np.array(c, dtype=np.int64) for f, c in info["cols"].items()}
        parts = {form: _map(directory / f"{key}.{form}.npy") for form in cols}
        tables = {form: np.load(directory / f"{key}.table{form}.npy") for form in (U8, U16)}
        out[key] = PackedFloat(tuple(info["shape"]), cols, parts, tables)
    return out


def _map(path: Path) -> np.ndarray:
    """``np.load(mmap_mode="r")``, except for an empty array, which cannot be mapped."""
    with path.open("rb") as fp:
        version = np.lib.format.read_magic(fp)
        read = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape = read(fp)[0]
    if int(np.prod(shape, dtype=np.int64)) == 0:
        return np.load(path)
    return np.load(path, mmap_mode="r")


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
