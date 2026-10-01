"""An arm's search read through another port executable (IKA-413).

`rustnode.binary_scope(path)` makes every road on the thread ask the port at `path`;
`PoolArm.rust_binary` / `play_game(rust_binary=...)` put an arm's move-node search (its menus
and solves) inside it, and nothing else: the game itself, the replacements and the selection
stay with the process's own port. None is no scope at all.

The executable here is a COPY of the process's own, so these tests hold in CI without an older
build: they show the scope reaches the arm (the copy's process is asked, the process's own is
asked by the other arm and the game), that a copy answers the same bytes (the record is equal
but for the executable it names), and that the default is untouched. That an older build gives
another answer is the oracle's: `scratchpad/ika413/binary_scope_check.py` runs the Showdown
cases through it (IKA-413's record).
"""

from __future__ import annotations

import shutil
from collections import Counter

import pytest

from pokeuraou import rustnode
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game

from .test_board_belief_once import _other
from .test_poolplay import _stub, _variants, _write_pool


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


def test_a_scope_does_not_hold_positions(pool, copy_of_the_port) -> None:  # noqa: ANN001
    rustnode.hold_positions()
    try:
        with pytest.raises(RuntimeError, match="hold"), rustnode.binary_scope(copy_of_the_port):
            pass
    finally:
        rustnode.hold_positions(False)


def _arm(pool, name, leaf, binary=None):  # noqa: ANN001, ANN202
    return PoolArm(
        name=name, evaluate=leaf, solver=SolvedSelections(pool.reg, pool.teams, leaf),
        limit=4, rank_by_leaf=True, rank_fill="refs2", rust_binary=binary,
    )


def _asked(monkeypatch):  # noqa: ANN001, ANN202
    """The exchanges each executable's process answered, by path."""
    seen: Counter = Counter()
    real = rustnode.RustNode._with_deadline

    def counting(self, call, what):  # noqa: ANN001, ANN202
        seen[str(self.binary)] += 1
        return real(self, call, what)

    monkeypatch.setattr(rustnode.RustNode, "_with_deadline", counting)
    return seen


def _play(pool, arms, which):  # noqa: ANN001, ANN202
    record, sides = pool_match_game(
        pool.reg, pool, arms, seed=413, game_index=0, which=which, hide_bench=True, max_turns=4,
    )
    payload = record.to_json(objective="value:arm", search_limit=(4, 4))
    payload.pop("searchSeconds", None)
    return record, sides, payload


@pytest.mark.parametrize("which", [0, 1])
def test_an_arms_search_reads_its_port_and_the_game_plays_the_same(  # noqa: ANN201
    pool, copy_of_the_port, monkeypatch, which  # noqa: ANN001
):
    # Two leaves, so each arm builds its own menu and solve either way.
    own = rustnode.binary_path()
    seen = _asked(monkeypatch)
    _, _, plain = _play(pool, (_arm(pool, "a", _stub), _arm(pool, "b", _other)), which)
    assert seen[str(own)] > 0 and seen[str(copy_of_the_port)] == 0
    assert "rustBinary" not in plain

    seen.clear()
    record, sides, scoped = _play(
        pool, (_arm(pool, "a", _stub, copy_of_the_port), _arm(pool, "b", _other)), which
    )
    # The tested arm sits at side `which`.
    assert sides["binaries"][which] == str(copy_of_the_port)
    assert sides["binaries"][1 - which] is None
    assert record.rust_binary[which]["path"] == str(copy_of_the_port)
    assert record.rust_binary[1 - which] is None
    # The positive control: the copy's process was asked (the arm's search), and so was the
    # process's own (the other arm's search and the game's own turns).
    assert seen[str(copy_of_the_port)] > 0, seen
    assert seen[str(own)] > 0, seen
    # A copy answers the same bytes, so the game is the same game: equal but for what it names.
    named = scoped.pop("rustBinary")
    assert named[which]["path"] == str(copy_of_the_port) and named[1 - which] is None
    assert scoped == plain


def test_two_arms_of_one_leaf_with_two_ports_build_two_menus(  # noqa: ANN201
    pool, copy_of_the_port, monkeypatch  # noqa: ANN001
):
    """Equal leaves share one construction; two ports are two answers, so they must not."""
    from pokeuraou import selfplay

    built = []
    real = selfplay._menus

    def counting(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(selfplay, "_menus", counting)
    _play(pool, (_arm(pool, "a", _stub), _arm(pool, "b", _stub)), 0)
    shared = len(built)
    built.clear()
    _play(pool, (_arm(pool, "a", _stub, copy_of_the_port), _arm(pool, "b", _stub)), 0)
    assert shared > 0 and len(built) == 2 * shared
