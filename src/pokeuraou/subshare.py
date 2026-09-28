"""IKA-378: a read's child sub-games, read once and shared by every process that reads it.

A depth-2 cell of the ladder's reading (`ladder`, `search._refine_cells`) is its turn, and each
kept branch's child position a sub-game: the two sides' menus there, the matrix of their
turns scored by the leaf, and its value. Other cells reach the same children: in a
1-process reading at 16 s (L5, count clock), 36% of the sub-games of 8 open positions and 43%
of 8 hidden ones were a child with the same two menus and knock-out fork as one filled
before in the same read (24% and 35% of their cells), a third of those in the same call; on
every core most of the rest are the same stage's cells on other worker processes (IKA-375).
A deep cell's children (depth 3 and up, `ladder._children_at_once`) are met again as often:
34-50% of them in the 1-process reads that reached depth 3.

So a sub-game filled and solved is kept here under its key -- the port's digest of the
child position (`rustnode.DIGESTS`), both menus and the fork -- with its value, its cell
count (what the count clock charges, as if it were filled again) and its notes, and a
sub-game with a key already kept is not filled again; a deep cell's child likewise
(`kid_key`), with the work its read counted (`pack_work`). Kept per read (`Table.begin`):
a read's leaf and Q are fixed, and nothing of one read is asked by the next.

`Table` is a lock-free table in shared memory for the ladder's worker processes (created
by the reader, attached by each worker by name): fixed rows of eight 64-bit words, the last
a check word over the others, so a row read while another process writes it -- or two
processes writing one row at once -- fails its check and reads as absent; a row is found
within `PROBE` slots of its key or not at all, and a full neighbourhood is overwritten. The
notes (sets of strings) are kept once each in a second region, by their hash. `Local` is
the same for a read with no worker processes, in a dict.
"""

from __future__ import annotations

import hashlib
import os
import zlib
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

#: Rows of the shared table (64 bytes each): 2^20 is 64 MB. ``POKEURAOU_LADDER_SHARE_SLOTS``.
SLOTS = int(os.environ.get("POKEURAOU_LADDER_SHARE_SLOTS", str(1 << 20)))
#: Slots looked at from a key's own: a key not there is absent (and a put overwrites).
PROBE = 8
#: Note sets kept, and the bytes of one (a longer set is not shared: its sub-game is not).
NOTE_SLOTS = 4096
NOTE_BYTES = 2040
_WORDS = 8
_NOTE_ROW = 24 + NOTE_BYTES
_MASK = (1 << 64) - 1
_GEN_MASK = (1 << 24) - 1
_FIELD = (1 << 21) - 1

#: What a table was asked this process's life (the positive control): sub-games taken from
#: it (``hits``) and from an earlier sub-game of the same call (``same``), put (``puts``),
#: looked for and not there (``misses``), rows whose check failed (``torn``); a deep cell's
#: children taken and put (``kidHits``, ``kidPuts``).
COUNTS: dict[str, int] = {"hits": 0, "same": 0, "puts": 0, "misses": 0, "torn": 0,
                          "kidHits": 0, "kidPuts": 0}

#: What a row holds: the value (None: none), the cells charged, the notes, and the packed
#: work of a deep cell's child (`pack_work`; 0 for a sub-game).
Row = tuple[float | None, int, frozenset[str], int]


def _u64(data: bytes) -> tuple[int, int]:
    return int.from_bytes(data[:8], "little"), int.from_bytes(data[8:16], "little")


def _menus(row: Iterable[Any], col: Iterable[Any]) -> str:  # noqa: ANN401
    return "/".join(a.to_choice() for a in row) + "#" + "/".join(a.to_choice() for a in col)


def menu_key(digest: Sequence[int], row: Iterable[Any], col: Iterable[Any],  # noqa: ANN401
             knockouts: bool) -> tuple[int, int]:
    """A sub-game's key: its child's digest, both menus (their choice strings, which name
    an action the same in every process) and the knock-out fork."""
    text = f"{digest[0]:x}.{digest[1]:x}|{int(knockouts)}|" + _menus(row, col)
    return _u64(hashlib.blake2b(text.encode(), digest_size=16).digest())


def kid_key(digest: Sequence[int], row: Iterable[Any], col: Iterable[Any], knockouts: bool,  # noqa: ANN401
            stage: str) -> tuple[int, int]:
    """A deep cell's child's key: its digest, its menus, the fork its matrix is filled on,
    and the stage (its label) the child is read by."""
    text = f"kid|{stage}|{digest[0]:x}.{digest[1]:x}|{int(knockouts)}|" + _menus(row, col)
    return _u64(hashlib.blake2b(text.encode(), digest_size=16).digest())


def completion_digest(digest: Sequence[int], completion: int) -> tuple[int, int]:
    """A digest for a branch read off another completion's turn (`search._cell_turns`): the
    reference branch's digest and the completion it was made for -- the same made position
    for the same pair, within a read (the completions are the read's)."""
    text = f"{digest[0]:x}.{digest[1]:x}@{completion}"
    return _u64(hashlib.blake2b(text.encode(), digest_size=16).digest())


def pack_work(work: dict[str, int]) -> int:
    """A deep cell's child's counted turns, sub-games and Q passes in one word (its cells go
    as a sub-game's do); 0 when one does not fit (then it is not kept)."""
    turns, subgames, qs = work.get("turns", 0), work.get("subgames", 0), work.get("qs", 0)
    if max(turns, subgames, qs) > _FIELD:
        return 0
    return (turns << 42) | (subgames << 21) | qs | (1 << 63)


def unpack_work(word: int) -> dict[str, int]:
    return {"turns": (word >> 42) & _FIELD, "subgames": (word >> 21) & _FIELD, "qs": word & _FIELD}


def _notes_hash(notes: frozenset[str]) -> tuple[int, bytes]:
    if not notes:
        return 0, b""
    data = "\n".join(sorted(notes)).encode()
    h = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little") or 1
    return h, data


def _check(words: Sequence[int]) -> int:
    x = 0x243F6A8885A308D3
    for w in words[:_WORDS - 1]:
        x = ((x ^ int(w)) * 0x9E3779B97F4A7C15) & _MASK
        x ^= x >> 29
    return x | 1


class Local:
    """The table of a read with no worker processes: a dict."""

    def __init__(self) -> None:
        self._rows: dict[tuple[int, int], Row] = {}

    def begin(self, generation: int = 0) -> None:
        del generation
        self._rows.clear()

    def get(self, key: tuple[int, int], *, kid: bool = False) -> Row | None:
        got = self._rows.get(key)
        COUNTS[("kidHits" if kid else "hits") if got is not None else "misses"] += 1
        return got

    def put(self, key: tuple[int, int], value: float | None, cells: int, notes: Iterable[str],
            extra: int = 0, *, kid: bool = False) -> None:
        COUNTS["kidPuts" if kid else "puts"] += 1
        self._rows[key] = (value, cells, frozenset(notes), extra)


class Table:
    """The shared table (the module's docstring). ``name`` None creates it (the reader);
    a worker attaches by the creator's `name`."""

    def __init__(self, name: str | None = None, slots: int = SLOTS) -> None:
        from multiprocessing import shared_memory

        size = slots * _WORDS * 8 + NOTE_SLOTS * _NOTE_ROW
        if name is None:
            # New shared memory reads as zeros (an empty row: its check word is 0).
            self._shm = shared_memory.SharedMemory(create=True, size=size)
            self.owner = True
        else:
            self._shm = shared_memory.SharedMemory(name=name)
            self.owner = False
            if os.name != "nt":
                # The creator unlinks it; a worker's tracker would unlink it again at exit.
                from multiprocessing import resource_tracker

                resource_tracker.unregister(self._shm._name, "shared_memory")  # noqa: SLF001
        self.name = self._shm.name
        self.slots = slots
        self.rows = np.ndarray((slots, _WORDS), dtype=np.uint64, buffer=self._shm.buf)
        self.notes = np.ndarray((NOTE_SLOTS, _NOTE_ROW), dtype=np.uint8, buffer=self._shm.buf,
                                offset=slots * _WORDS * 8)
        self.generation = 0
        #: Note sets seen by this process, by hash.
        self._known: dict[int, frozenset[str]] = {0: frozenset()}

    def begin(self, generation: int) -> None:
        """A new read: rows of other reads are absent (and overwritten)."""
        self.generation = generation & _GEN_MASK

    def _key(self, key: tuple[int, int]) -> tuple[int, int]:
        # The read's generation in the key: a row of another read never matches.
        return key[0] ^ (self.generation * 0x9E3779B97F4A7C15 & _MASK), key[1]

    def get(self, key: tuple[int, int], *, kid: bool = False) -> Row | None:
        """The row kept under ``key`` in this read, or None. ``kid``: a deep cell's child's
        (counted apart)."""
        k1, k2 = self._key(key)
        base = k1 % self.slots
        for step in range(PROBE):
            row = self.rows[(base + step) % self.slots].tolist()
            if row[0] != k1 or row[1] != k2:
                continue
            if _check(row) != row[_WORDS - 1]:
                COUNTS["torn"] += 1
                break
            notes = self._note(int(row[4]))
            if notes is None:
                break
            COUNTS["kidHits" if kid else "hits"] += 1
            value = float(np.uint64(row[2]).view(np.float64))
            return (None if value != value else value), int(row[3]) >> 24, notes, int(row[5])  # noqa: PLR0124 - NaN
        COUNTS["misses"] += 1
        return None

    def put(self, key: tuple[int, int], value: float | None, cells: int, notes: Iterable[str],
            extra: int = 0, *, kid: bool = False) -> None:
        notes = frozenset(notes)
        nh, data = _notes_hash(notes)
        if len(data) > NOTE_BYTES:
            return
        if nh and not self._put_note(nh, data):
            return
        self._known.setdefault(nh, notes)
        k1, k2 = self._key(key)
        base = k1 % self.slots
        at = base
        for step in range(PROBE):
            slot = (base + step) % self.slots
            row = self.rows[slot].tolist()
            if (row[0] == k1 and row[1] == k2) or (row[3] & _GEN_MASK) != self.generation \
                    or row[_WORDS - 1] == 0:
                at = slot
                break
        v = int(np.float64(np.nan if value is None else value).view(np.uint64))
        meta = (min(cells, (1 << 40) - 1) << 24) | self.generation
        words = [k1, k2, v, meta, nh, extra, 0]
        words.append(_check(words))
        self.rows[at] = np.array(words, dtype=np.uint64)
        COUNTS["kidPuts" if kid else "puts"] += 1

    def _note(self, nh: int) -> frozenset[str] | None:
        got = self._known.get(nh)
        if got is not None:
            return got
        base = nh % NOTE_SLOTS
        for step in range(4):
            row = self.notes[(base + step) % NOTE_SLOTS]
            head = row[:24].view(np.uint64)
            if int(head[0]) != nh:
                continue
            n = int(head[1])
            data = row[24:24 + n].tobytes() if n <= NOTE_BYTES else b""
            if n > NOTE_BYTES or zlib.crc32(data) != int(head[2]):
                return None
            notes = frozenset(data.decode().split("\n"))
            self._known[nh] = notes
            return notes
        return None

    def _put_note(self, nh: int, data: bytes) -> bool:
        base = nh % NOTE_SLOTS
        for step in range(4):
            row = self.notes[(base + step) % NOTE_SLOTS]
            head = row[:24].view(np.uint64)
            if int(head[0]) == nh and self._note(nh) is not None:
                return True
            if int(head[0]) in (0, nh) or step == 3:
                head[0] = 0
                row[24:24 + len(data)] = np.frombuffer(data, dtype=np.uint8)
                head[1] = len(data)
                head[2] = zlib.crc32(data)
                head[0] = nh
                return True
        return False

    def close(self) -> None:
        self.rows = self.notes = None  # type: ignore[assignment]
        self._shm.close()
        if self.owner:
            self._shm.unlink()


__all__ = ["COUNTS", "Local", "Table", "completion_digest", "kid_key", "menu_key", "pack_work",
           "unpack_work"]
