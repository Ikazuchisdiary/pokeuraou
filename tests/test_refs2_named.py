"""IKA-341: refs2 is played only where it is named; the library resolves an unnamed fill as
M-C generation does.

The user's decision of 9/28: the library's unnamed fill of a leaf-ranked menu is q-nocover,
which needs the process's Q and stops without one (IKA-338's rule, now in `_menus` and
`PoolArm`); a person's game with no Q it can read falls back to refs2 with a note; the refs
fills stay, played when named. What these tests hold:

- **unnamed is q-nocover**: a leaf-ranked `PoolArm` with no fill resolves it at construction;
- **a person's menus**: a Q of another vocabulary than the leaf's falls back to refs2 with a
  note when the fill was not named, and stops when it was.

Nothing here needs torch, CUDA or data/: the Q is a stand-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pokeuraou import qrank
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm
from pokeuraou.search import DEFAULT_RANK_FILL, SHIPPED_RANK_FILL

from ._harness import load_tool
from .test_poolplay import _served_stub, _variants, _write_pool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture()
def no_q(monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.setattr(qrank, "_INSTALLED", {})


def test_the_constants() -> None:
    assert DEFAULT_RANK_FILL == "refs2"
    assert SHIPPED_RANK_FILL == "q-nocover"


# ------------------------------------------------------------------ the library


def test_a_pool_arm_resolves_its_fill_at_construction() -> None:
    leafy = PoolArm(name="a", evaluate=_served_stub, solver=None, limit=2, rank_by_leaf=True)
    damage = PoolArm(name="b", evaluate=None, solver=None, limit=2, rank_by_leaf=False)
    named = PoolArm(name="c", evaluate=_served_stub, solver=None, limit=2, rank_by_leaf=True,
                    rank_fill="refs2")
    assert (leafy.rank_fill, damage.rank_fill, named.rank_fill) == ("q-nocover", "refs2", "refs2")


# ------------------------------------------------------------------ the fill is held


# ------------------------------------------------------------------ a person's menus


class _Encoder:
    pass


class _VocabQ:
    """A local Q whose file reads another vocabulary than the leaf's."""

    def __init__(self, path, encoder, device="cpu") -> None:  # noqa: ANN001
        raise qrank.QVocabularyError(f"{path} reads another vocabulary than this encoder's")


class _GoodQ:
    def __init__(self, path, encoder, device="cpu") -> None:  # noqa: ANN001
        self.path = Path(path)

    def describe(self) -> list[str]:
        return [self.path.name]


@pytest.fixture()
def q_file(tmp_path):  # noqa: ANN001, ANN201
    path = tmp_path / "q-mc0.pt"
    path.write_bytes(b"")
    return path


def test_a_persons_menus_fall_back_on_a_q_of_another_vocabulary(
    monkeypatch, q_file, no_q  # noqa: ANN001
) -> None:
    tool = load_tool("play_human")
    said: list[str] = []
    monkeypatch.setattr(qrank, "LocalQ", _VocabQ)
    fill, files = tool.install_menus(None, q_file, _Encoder(), None, said.append)
    assert (fill, files) == ("refs2", [])
    assert len(said) == 1 and "another vocabulary" in said[0] and "refs2" in said[0]
    assert qrank._INSTALLED == {}  # noqa: SLF001
    # Named, the fill is played or the tool stops: no fallback.
    with pytest.raises(qrank.QVocabularyError):
        tool.install_menus("q-nocover", q_file, _Encoder(), None, said.append)
    # It is both errors the two Q classes raised before, so their callers still catch it.
    assert issubclass(qrank.QVocabularyError, ValueError)
    assert issubclass(qrank.QVocabularyError, RuntimeError)


def test_a_persons_menus_otherwise_as_before(monkeypatch, q_file, tmp_path, no_q) -> None:  # noqa: ANN001
    tool = load_tool("play_human")
    said: list[str] = []
    monkeypatch.setattr(qrank, "LocalQ", _GoodQ)
    assert tool.install_menus(None, q_file, _Encoder(), None, said.append) == (
        "q-nocover", ["q-mc0.pt"])
    assert said == [] and qrank.installed().path == q_file
    # No file, or no leaf's encoder: refs2 with a note.
    assert tool.install_menus(None, tmp_path / "none.pt", _Encoder(), None,
                              said.append)[0] == "refs2"
    assert tool.install_menus(None, q_file, None, None, said.append)[0] == "refs2"
    assert len(said) == 2 and all("note: menus ranked by refs2" in s for s in said)
    # Named refs2 asks for nothing.
    assert tool.install_menus("refs2", tmp_path / "none.pt", None, None,
                              said.append) == ("refs2", [])
