"""IKA-341: refs2 is played only where it is named; the library resolves an unnamed fill as
M-C generation does.

The user's decision of 9/28: the library's unnamed fill of a leaf-ranked menu is q-nocover,
which needs the process's Q and stops without one (IKA-338's rule, now in `play_game`,
`_menus`, `generate_pool` and `PoolArm`); M-B's roster path names refs2
(`search.ROSTER_RANK_FILL`, M-B has no Q); a person's game with no Q it can read falls
back to refs2 with a note; the refs fills stay, played when named. What these tests hold:

- **unnamed is q-nocover or a stop**: a leaf-ranked `play_game` / `generate_pool` /
  `PoolArm` with no fill ranks by the installed Q (the record's ``rankFill``), and with no Q
  it stops at the first menu -- never refs2;
- **a side that reads no fill keeps its bytes**: damage-ranked, the record carries no
  ``rankFill``, as before;
- **M-B names refs2**: `selfplay.generate` passes `ROSTER_RANK_FILL`, and `agent_drift` holds
  a roster tool that does not to account;
- **the rank-score file has cells to record**: under a q fill it stops, where it wrote
  empty lines (IKA-278's file, since IKA-338's default);
- **a person's menus**: a Q of another vocabulary than the leaf's falls back to refs2 with a
  note when the fill was not named, and stops when it was.

Nothing here needs torch, CUDA or data/: the Q is a stand-in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import qrank, selfplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.payoff import HP_SHARE
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, generate_pool
from pokeuraou.search import DEFAULT_RANK_FILL, ROSTER_RANK_FILL, SHIPPED_RANK_FILL

from ._harness import load_tool
from .test_poolplay import _stub, _variants, _write_pool
from .test_q_rank import _StubQ

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


@pytest.fixture()
def stub_q(pool, monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.setattr(qrank, "_INSTALLED", {})
    model = _StubQ(Encoder(pool.reg))
    qrank.install(model)
    return model


def _game(pool, **kwargs):  # noqa: ANN001, ANN003, ANN202
    team = list(pool.teams[0].sets)
    return selfplay.play_game(
        pool.reg, np.random.default_rng(3), team[:4], team[:4], "t",
        search_limit=3, max_turns=2, evaluate=_stub, open_information=True, **kwargs,
    )


def test_the_constants() -> None:
    assert ROSTER_RANK_FILL == DEFAULT_RANK_FILL == "refs2"
    assert SHIPPED_RANK_FILL == "q-nocover"


# ------------------------------------------------------------------ the library


def test_an_unnamed_leaf_ranked_game_stops_without_a_q(pool, no_q) -> None:  # noqa: ANN001
    with pytest.raises(RuntimeError, match="needs a Q"):
        _game(pool, rank_by_leaf=True)


def test_an_unnamed_leaf_ranked_game_plays_q_nocover_by_the_q(pool, stub_q) -> None:  # noqa: ANN001
    before = dict(qrank.QRANKED.get("q-nocover", {"rankings": 0}))
    record = _game(pool, rank_by_leaf=True)
    assert record.rank_fill == ["q-nocover", "q-nocover"]
    payload = record.to_json(objective="hp-share", search_limit=3)
    assert payload["rankFill"] == ["q-nocover", "q-nocover"]
    assert qrank.QRANKED["q-nocover"]["rankings"] > before.get("rankings", 0)
    assert stub_q.calls > 0


def test_refs2_named_needs_no_q_and_a_damage_side_keeps_its_bytes(pool, no_q) -> None:  # noqa: ANN001
    named = _game(pool, rank_by_leaf=True, rank_fill="refs2")
    assert named.rank_fill == ["refs2", "refs2"]
    assert "rankFill" not in named.to_json(objective="hp-share", search_limit=3)
    # Damage-ranked, unnamed: the fill is never read, no Q is asked for, and the record is
    # the one it was (no rankFill).
    damage = _game(pool, rank_by_leaf=False)
    assert damage.rank_fill == ["refs2", "refs2"]
    assert "rankFill" not in damage.to_json(objective="hp-share", search_limit=3)
    # Per side: one leaf-ranked side named, the other damage-ranked and unnamed.
    mixed = _game(pool, rank_by_leaf=(True, False), rank_fill=("refs1", None))
    assert mixed.rank_fill == ["refs1", "refs2"]


def test_a_pool_arm_resolves_its_fill_at_construction() -> None:
    leafy = PoolArm(name="a", evaluate=_stub, solver=None, limit=2, rank_by_leaf=True)
    damage = PoolArm(name="b", evaluate=None, solver=None, limit=2, rank_by_leaf=False)
    named = PoolArm(name="c", evaluate=_stub, solver=None, limit=2, rank_by_leaf=True,
                    rank_fill="refs2")
    assert (leafy.rank_fill, damage.rank_fill, named.rank_fill) == ("q-nocover", "refs2", "refs2")


def _generate(pool, out: Path, **kwargs):  # noqa: ANN001, ANN003, ANN202
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    generate_pool(
        pool.reg, pool, games=0, hide_bench=True, seed=278, out=out, solver=solver,
        evaluate=_stub, objective=HP_SHARE, search_limit=3, max_turns=8, rank_by_leaf=True,
        indices=[0, 1, 2], **kwargs,
    )
    return [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]


def test_generate_pool_resolves_as_generation_does(pool, tmp_path, no_q) -> None:  # noqa: ANN001
    with pytest.raises(RuntimeError, match="needs a Q"):
        _generate(pool, tmp_path / "none.jsonl")
    qrank.install(_StubQ(Encoder(pool.reg)))
    games = _generate(pool, tmp_path / "q.jsonl")
    assert games and all(g["rankFill"] == ["q-nocover", "q-nocover"] for g in games)


def test_the_rank_score_file_stops_under_a_q_fill(pool, tmp_path, stub_q) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="refs leaf ranking"):
        _generate(pool, tmp_path / "g.jsonl", rank_scores_out=tmp_path / "rank.jsonl.gz")
    with pytest.raises(ValueError, match="refs leaf ranking"):
        _generate(pool, tmp_path / "g.jsonl", rank_fill="q", rank_scores_out=tmp_path / "r.jsonl.gz")
    assert not (tmp_path / "g.jsonl").exists()
    # A refs fill records, as before.
    _generate(pool, tmp_path / "refs.jsonl", rank_fill="refs2",
              rank_scores_out=tmp_path / "rank-refs.jsonl.gz")
    assert (tmp_path / "rank-refs.jsonl.gz").stat().st_size > 0


def test_the_generation_worker_refuses_to_record_rank_scores_under_a_q_fill(
    tmp_path, capsys  # noqa: ANN001
) -> None:
    tool = load_tool("selfplay")
    args = type("A", (), {"record_rank_scores": True, "rank_leaf": True, "rank_fill": "q-nocover"})()
    with pytest.raises(SystemExit):
        tool.rank_scores_path(args, _Parser(), tmp_path / "g.jsonl")
    assert "name --rank-fill refs2" in capsys.readouterr().err
    args.rank_fill = "refs2"
    assert tool.rank_scores_path(args, _Parser(), tmp_path / "g.jsonl") is not None


class _Parser:
    """`ArgumentParser.error`'s behaviour: the message on stderr, exit 2."""

    def error(self, message: str) -> None:
        print(message, file=sys.stderr)
        raise SystemExit(2)


# ------------------------------------------------------------------ M-B's roster path


def test_m_b_generation_names_the_roster_fill(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    from pokeuraou.teams import load_roster

    roster = load_roster("rizabanadohido")
    seen: dict = {}

    class Stop(Exception):
        pass

    def fake_play(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(selfplay, "play_game", fake_play)
    with pytest.raises(Stop):
        selfplay.generate(roster.reg, None, roster, [], games=1, mirror_share=1.0,
                          hide_bench=True, rank_by_leaf=True, out=tmp_path / "g.jsonl")
    assert seen["rank_fill"] == ROSTER_RANK_FILL


def test_agent_drift_holds_a_roster_tool_that_leaves_the_fill_off(tmp_path) -> None:  # noqa: ANN001
    drift = load_tool("agent_drift")
    assert "rank_fill" in drift.shipping_args()
    # Every roster tool that ranks by the leaf names it (generation_match is the board).
    for name in ("generation_match.py", "bench_generation.py", "book_check.py",
                 "selection_check.py", "resume_generate.py", "worker_growth.py",
                 "forced_handoff.py"):
        passed = set().union(*(kw for _line, kw in drift.calls(ROOT / "tools" / name)))
        assert "rank_fill" in passed, name
    # Positive control: a call that leaves it off is missing it.
    tool = tmp_path / "roster_tool.py"
    tool.write_bytes(
        b"play_game(reg, rng, a, b, 'x', evaluate=e, rank_by_leaf=True, sheets=s,\n"
        b"          search_limit=12, bench_prior=p)\n"
    )
    passed = set().union(*(kw for _line, kw in drift.calls(tool)))
    assert drift.shipping_args() - passed == {"rank_fill"}


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
    fill, files = tool.install_menus(None, q_file, _Encoder(), _stub, None, None, said.append)
    assert (fill, files) == ("refs2", [])
    assert len(said) == 1 and "another vocabulary" in said[0] and "refs2" in said[0]
    assert qrank._INSTALLED == {}  # noqa: SLF001
    # Named, the fill is played or the tool stops: no fallback.
    with pytest.raises(qrank.QVocabularyError):
        tool.install_menus("q-nocover", q_file, _Encoder(), _stub, None, None, said.append)
    # It is both errors the two Q classes raised before, so their callers still catch it.
    assert issubclass(qrank.QVocabularyError, ValueError)
    assert issubclass(qrank.QVocabularyError, RuntimeError)


def test_a_persons_menus_otherwise_as_before(monkeypatch, q_file, tmp_path, no_q) -> None:  # noqa: ANN001
    tool = load_tool("play_human")
    said: list[str] = []
    monkeypatch.setattr(qrank, "LocalQ", _GoodQ)
    assert tool.install_menus(None, q_file, _Encoder(), _stub, None, None, said.append) == (
        "q-nocover", ["q-mc0.pt"])
    assert said == [] and qrank.installed().path == q_file
    # No file, or no leaf's encoder: refs2 with a note.
    assert tool.install_menus(None, tmp_path / "none.pt", _Encoder(), _stub, None, None,
                              said.append)[0] == "refs2"
    assert tool.install_menus(None, q_file, None, None, None, None, said.append)[0] == "refs2"
    assert len(said) == 2 and all("note: menus ranked by refs2" in s for s in said)
    # Named refs2 asks for nothing.
    assert tool.install_menus("refs2", tmp_path / "none.pt", None, None, None, None,
                              said.append) == ("refs2", [])
