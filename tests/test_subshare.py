"""IKA-378: the ladder's table of child sub-games (`subshare`).

What these hold: a row put by one handle is read by another handle of the same shared
memory (as a worker process reads the reader's) with its value, cell count and notes; a
sub-game with no value (an unsolvable matrix) reads as None; a row of another read (another
generation) is absent; a row whose words do not agree (a read while another process
writes) reads as absent, never as another value; and the keys tell menus and forks apart.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pokeuraou import subshare


def _acts(*names: str) -> list:
    return [SimpleNamespace(to_choice=lambda n=n: n) for n in names]


def test_a_row_is_read_by_another_handle_with_its_notes() -> None:
    table = subshare.Table(slots=1024)
    other = subshare.Table(table.name, slots=1024)
    try:
        for t in (table, other):
            t.begin(7)
        key = subshare.menu_key((1, 2), _acts("move 1", "move 2"), _acts("move 3"), False)
        assert other.get(key) is None
        table.put(key, 0.25, 12, {"a note", "another"})
        assert other.get(key) == (0.25, 12, frozenset({"a note", "another"}), 0)
        none = subshare.menu_key((1, 2), _acts("move 1"), _acts("move 3"), False)
        table.put(none, None, 3, ())
        assert other.get(none) == (None, 3, frozenset(), 0)
        # Another read's rows are absent.
        other.begin(8)
        assert other.get(key) is None
        # A row whose words disagree is absent, not another value.
        other.begin(7)
        k1 = key[0] ^ (7 * 0x9E3779B97F4A7C15 & ((1 << 64) - 1))
        slot = next(s for s in range(1024) if int(table.rows[s][0]) == k1)
        table.rows[slot, 2] = np.uint64(int(np.float64(0.5).view(np.uint64)))
        before = subshare.COUNTS["torn"]
        assert other.get(key) is None
        assert subshare.COUNTS["torn"] == before + 1
    finally:
        other.close()
        table.close()


def test_a_sub_game_being_filled_is_busy_elsewhere(monkeypatch) -> None:  # noqa: ANN001
    """IKA-380: a claim marks a key absent from the table as being filled by its claimer:
    another handle's claim answers `BUSY` and its get reads it as absent, until the put
    writes the value over the mark; the claimer's own later claim finds the value. Another
    read's mark is absent. Without marks (``MARKS`` off) a claim is a get."""
    table = subshare.Table(slots=1024)
    other = subshare.Table(table.name, slots=1024)
    try:
        for t in (table, other):
            t.begin(3)
        key = subshare.menu_key((5, 6), _acts("move 1"), _acts("move 2"), True)
        assert table.claim(key) is None
        before = dict(subshare.COUNTS)
        assert other.claim(key) is subshare.BUSY
        assert other.get(key) is None
        assert subshare.COUNTS["busy"] == before["busy"] + 1
        other.begin(4)
        assert other.claim(key) is None  # another read: its own mark now
        other.begin(3)
        table.put(key, 0.75, 9, {"n"})
        assert other.claim(key) == (0.75, 9, frozenset({"n"}), 0)
        assert table.claim(key) == (0.75, 9, frozenset({"n"}), 0)
        monkeypatch.setattr(subshare, "MARKS", False)
        fresh = subshare.menu_key((7, 8), _acts("move 1"), _acts("move 2"), True)
        assert table.claim(fresh) is None and other.claim(fresh) is None
        local = subshare.Local()
        assert local.claim(fresh) is None
        local.put(fresh, 0.5, 1, ())
        assert local.claim(fresh) == (0.5, 1, frozenset(), 0)
    finally:
        other.close()
        table.close()


def test_the_key_tells_menus_and_forks_apart() -> None:
    a = subshare.menu_key((1, 2), _acts("move 1", "move 2"), _acts("move 3"), False)
    assert a == subshare.menu_key((1, 2), _acts("move 1", "move 2"), _acts("move 3"), False)
    assert a != subshare.menu_key((1, 2), _acts("move 1", "move 2"), _acts("move 3"), True)
    assert a != subshare.menu_key((1, 2), _acts("move 1"), _acts("move 2", "move 3"), False)
    assert a != subshare.menu_key((1, 3), _acts("move 1", "move 2"), _acts("move 3"), False)
    made = subshare.completion_digest((1, 2), 0)
    assert made != subshare.completion_digest((1, 2), 1) and made != (1, 2)


def test_a_childs_work_is_kept_whole() -> None:
    work = {"turns": 37, "subgames": 120, "qs": 5}
    packed = subshare.pack_work(work)
    assert packed and subshare.unpack_work(packed) == work
    assert subshare.pack_work({"turns": 1 << 22, "subgames": 1, "qs": 0}) == 0
    kid = subshare.kid_key((1, 2), _acts("move 1"), _acts("move 3"), False, "d3r2ban4/r2ban4")
    assert kid != subshare.kid_key((1, 2), _acts("move 1"), _acts("move 3"), False, "d3r2ban4/r3ban4")
    assert kid != subshare.menu_key((1, 2), _acts("move 1"), _acts("move 3"), False)


def test_a_local_table_starts_empty_each_read() -> None:
    table = subshare.Local()
    table.begin()
    table.put((1, 2), 0.5, 4, {"n"})
    assert table.get((1, 2)) == (0.5, 4, frozenset({"n"}), 0)
    table.begin()
    assert table.get((1, 2)) is None
