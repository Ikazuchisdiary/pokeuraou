"""An arm's search read through another port executable (IKA-413).

`rustnode.binary_scope(path)` makes every road on the thread ask the port at `path`;
`play_game(rust_binary=...)` puts an arm's move-node search (its menus and solves) inside it,
and nothing else. None is no scope at all. (The board's arm on another executable is the Rust
board worker's since IKA-458 stage 3.7: `test_rust_board_worker.py`.)

The executable here is a COPY of the process's own, so these tests hold in CI without an older
build: they show the scope asks the copy's process and hands back, and that the default is
untouched. That an older build gives another answer is the oracle's:
`scratchpad/ika413/binary_scope_check.py` runs the Showdown cases through it (IKA-413's record).
"""

from __future__ import annotations

import shutil

import pytest

from pokeuraou import rustnode
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool

from .test_poolplay import _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture()
def copy_of_the_port(tmp_path):  # noqa: ANN001, ANN201
    path = rustnode.binary_path()
    if not path.exists():
        pytest.fail(f"no Rust binary at {path}; `cargo build --release`")
    copy = tmp_path / path.name
    shutil.copy(path, copy)
    yield copy
    rustnode.reset()


def test_no_scope_is_the_modules_own_port(pool) -> None:  # noqa: ANN001
    own = rustnode.node_for(pool.reg)
    assert own is not None
    with rustnode.binary_scope(None):
        assert rustnode.node_for(pool.reg) is own
        assert rustnode.require_node(pool.reg) is own


def test_a_scope_asks_the_other_executable_and_hands_back(pool, copy_of_the_port) -> None:  # noqa: ANN001
    own = rustnode.node_for(pool.reg)
    with rustnode.binary_scope(copy_of_the_port):
        inside = rustnode.node_for(pool.reg)
        assert inside is not own and inside.binary == copy_of_the_port
        assert rustnode.require_node(pool.reg) is inside
        # The same executable is one process, not one per ask.
        with rustnode.binary_scope(copy_of_the_port):
            assert rustnode.node_for(pool.reg) is inside
        with rustnode.binary_scope(None):
            assert rustnode.node_for(pool.reg) is inside
        assert rustnode.node_for(pool.reg) is inside
    assert rustnode.node_for(pool.reg) is own


def test_a_scope_holds_positions_in_its_own_process(pool, copy_of_the_port) -> None:  # noqa: ANN001
    """IKA-460: with positions held by number, the scope's process is told what a request names
    (`_known` is its own set), and it forgets with the others when a decision ends."""
    rustnode.hold_positions()
    try:
        own = rustnode.node_for(pool.reg)
        with rustnode.binary_scope(copy_of_the_port):
            inside = rustnode.node_for(pool.reg)
            assert inside is not own and inside.binary == copy_of_the_port
            inside._known.add(2)  # noqa: SLF001
            rustnode.hold_positions()  # a new decision: every process forgets
            assert not inside._known and inside._forget_owed  # noqa: SLF001
    finally:
        rustnode.hold_positions(False)
