"""IKA-458 stage 3.7: the board's worker is `pokeuraou-damage pool-match-worker`.

The board's Python loop (`tools/pool_match.py`'s, over `poolplay.pool_match_game`) was removed,
so these tests hold the Rust board worker by itself, on the tests' own inference server (two
leaves, hp-share and a Q of stand-ins, no GPU):

- **the records pair up**: for a board of two leaves and a board against hp-share, both seats of
  every game, the provenance naming the tested arm's side, each arm's leaf, width and selection
  in whichever side it sits, the same pair in both seats; every move node is solved per side
  (the positive control that the side's own node was played);
- **the null**: identical arms play the same game in both seats, and another seed other games;
- **the records cannot drift from their templates unseen**: every key of the generation worker's
  and the board worker's record templates is either a key the Rust worker fills per game (its
  `PER_GAME`, read from the source) or one this test names as the same for every game. A key
  `GameRecord.to_json` gains is neither, and this test fails until someone says which it is (a
  per-game key copied from the template would be the same in every record); an arm with
  generation's leaf and Q gets generation's node, budget and turn cap;
- **a flag the Rust board worker does not play stops the config** (`rustworker.board_unsupported`).

Until stage 3.7 this file compared the Rust board's records with the Python board's key for key
(records/IKA-458.md sections 17.3 and 18).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import rustworker
from pokeuraou.encode import Encoder
from pokeuraou.regulation import repo_root

from ._served import hp_share_model, model_of, serve
from .test_q_rank import _ServedStub
from .test_rust_worker import _leaf_model, pool  # noqa: F401 - the module's pool fixture

ROOT = repo_root()
GAMES = 3  # per seat

#: The template keys that are the same in every game of a run (the rest are `PER_GAME`).
STATIC_GENERATION = {
    "searchObjective", "searchLimit", "targetIsRealOutcome", "engine", "selectionSource",
    "information", "ranking", "rankView", "rankFill", "qModel", "qModelSha256",
}
STATIC_BOARD = STATIC_GENERATION - {"qModel", "qModelSha256"} | {
    "ownSelectionPolicy", "foeSelectionPolicy", "ownSelectionMixture", "foeSelectionMixture",
    "selectionValue", "provenance", "seatIndex", "qModel", "qModelSha256", "rustBinary",
}


def _per_game(source: str) -> set[str]:
    text = (ROOT / "rust" / "src" / source).read_bytes().decode("utf-8")
    block = re.search(r"const PER_GAME: &\[&str\] = &\[(.*?)\];", text, re.S)
    assert block, f"{source} has no PER_GAME"
    return set(re.findall(r'"(\w+)"', block.group(1)))


@pytest.fixture(scope="module")
def served(pool):  # noqa: ANN001, ANN201, F811
    encoder = Encoder(pool.reg)
    base = _leaf_model(encoder)

    def other(arrays: dict, rows: int) -> np.ndarray:
        """Another leaf: the same order of positions, other values."""
        return 0.25 + 0.5 * base(arrays, rows)

    server, address = serve(
        {"value": base, "baseline": other, "hp-share": model_of(hp_share_model(encoder))},
        arms={"value": ["stub.pt"], "baseline": ["other.pt"], "hp-share": ["hp-share"]},
        q_models={"q": _ServedStub(encoder.vocab.fingerprint())},
    )
    yield address
    server.shutdown()
    server.server_close()


def _flags(pool_path: Path, address: str, tmp: Path, kind: str) -> list[str]:
    flags = [
        "--pool", str(pool_path), "--inference", address, "--inference-arm", "value", "--q-arm", "q",
        "--hide-bench", "--limit", "4", "--baseline-limit", "3", "--rank-leaf", "--baseline-rank-leaf",
        "--seed", "4581", "--games", str(GAMES), "--max-turns", "40",
        "--selection-store", str(tmp / "store-value"),
    ]
    if kind == "hp":
        return [*flags, "--baseline-hp-share"]
    if kind == "same":
        flags[flags.index("--baseline-limit") + 1] = "4"
        return [*flags, "--baseline-inference-arm", "value"]
    return [*flags, "--baseline-inference-arm", "baseline",
            "--baseline-selection-store", str(tmp / "store-baseline")]


def _records(path: Path) -> dict[tuple[int, int], dict]:
    rows = [json.loads(line) for line in path.read_bytes().decode("utf-8").splitlines() if line]
    return {(r["gameIndex"], r["seatIndex"]): r for r in rows}


def _rust_board(flags: list[str], tmp: Path, address: str) -> tuple[dict, dict, str]:
    config = tmp / "worker-config.json"
    made = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "tools" / "pool_match.py"), *flags,
         "--games-out", str(tmp / "rust.jsonl"), "--rust-config", str(config)],
        capture_output=True, text=True, timeout=300, check=False, cwd=str(ROOT),
    )
    assert made.returncode == 0, made.stderr
    out = tmp / "rust.jsonl"
    done = subprocess.run(  # noqa: S603
        rustworker.board_command(config, address, out, tmp / "summary.jsonl",
                                 first_index=0, indices=2 * GAMES),
        capture_output=True, text=True, timeout=900, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "handoffs 0" in done.stderr, done.stderr
    return _records(out), json.loads(config.read_bytes()), done.stderr


@pytest.mark.parametrize("kind", ["leaves", "hp"])
def test_the_board_s_records_pair_up_with_each_side_s_own_node(pool, served, tmp_path, kind) -> None:  # noqa: ANN001, F811
    flags = _flags(Path(pool.path), served, tmp_path, kind)
    rust, config, stderr = _rust_board(flags, tmp_path, served)
    assert sorted(rust) == [(g, s) for g in range(GAMES) for s in (0, 1)]
    # The positive control: each side's own node (two leaves, or a leaf against hp-share).
    assert re.search(r"move nodes (\d+) \(\1 with each side's own node\)", stderr), stderr
    other = "hp-share" if kind == "hp" else "other"
    if kind == "hp":
        assert config["arms"][1]["store"] is None
    for (game, which), record in rust.items():
        source = record["provenance"]
        assert source["kind"] == "pool-match"
        assert source["seat"] == f"stub@w4 = side {which}"
        assert source["leaves"][which] == "stub" and source["leaves"][1 - which] == other
        assert source["books"][which] == "solved"
        assert source["books"][1 - which] == ("uniform" if kind == "hp" else "solved")
        assert source["information"] == ["hidden-bench", "hidden-bench"]
        assert record["searchLimit"][which] == 4 and record["searchLimit"][1 - which] == 3
        # The two seats of a game are the same pair, the arms swapped.
        assert record["pool"]["teams"] == rust[(game, 1 - which)]["pool"]["teams"]


def test_identical_arms_play_the_same_game_in_both_seats(pool, served, tmp_path) -> None:  # noqa: ANN001, F811
    """The null: one leaf, one Q and one width on both arms. A game's two seats are then the same
    game (a record is side 0's), so a pair scores exactly 0.5 -- and the control, another seed,
    is other games."""
    flags = _flags(Path(pool.path), served, tmp_path, "same")
    rust, _, _ = _rust_board(flags, tmp_path, served)
    drop = ("searchSeconds", "seatIndex", "provenance")
    for game in range(GAMES):
        a, b = ({k: v for k, v in rust[(game, s)].items() if k not in drop} for s in (0, 1))
        assert a == b
    other = _flags(Path(pool.path), served, tmp_path / "o", "same")
    other[other.index("--seed") + 1] = "4582"
    moved, _, _ = _rust_board(other, tmp_path / "o", served)
    assert any(moved[(g, 0)]["decisions"] != rust[(g, 0)]["decisions"] for g in range(GAMES))


def test_the_records_keys_are_per_game_or_named_static(pool, served, tmp_path) -> None:  # noqa: ANN001, F811
    from pokeuraou import qrank
    from pokeuraou.inference import RemoteValue

    encoder = Encoder(pool.reg)
    leaf = RemoteValue(served, "value", encoder)
    q = qrank.RemoteQ(served, "q", encoder)
    qrank.install(q)
    try:
        args = Namespace(
            inference=served, rank_leaf=True, rank_fill="q-nocover", seed=1, max_turns=40,
            explore_epsilon=0.25, explore_temperature=0.5, limit=4,
        )
        generation = rustworker.build_config(
            args, pool, leaf, "value:stub", qrank.record_fields({"": q}), None, tmp_path / "s",
        )
    finally:
        qrank._INSTALLED.pop("", None)  # noqa: SLF001
        leaf.close()
        q.close()
    keys = [k for k, _ in generation["record"]]
    per_game = _per_game("worker.rs")
    assert per_game <= set(keys)
    assert set(keys) - per_game == STATIC_GENERATION, (
        "a record key is neither filled per game by worker.rs nor named static here"
    )
    flags = _flags(Path(pool.path), served, tmp_path / "b", "leaves")
    _, board, _ = _rust_board(flags, tmp_path / "b", served)
    # The board plays generation's agent: an arm of the same leaf and Q gets generation's node (the
    # servers, blocks and widths are the worker's to fill), its budgets and its turn cap.
    same = {k: v for k, v in generation["node"].items() if k not in ("server", "qServer", "widths")}
    assert board["arms"][0]["node"] == same
    assert board["exactBudget"] == generation["exactBudget"]
    assert board["maxTurns"] == generation["maxTurns"]
    per_game = _per_game("boardworker.rs")
    for template in board["records"]:
        keys = {k for k, _ in template}
        assert per_game <= keys
        assert keys - per_game <= STATIC_BOARD, keys - per_game - STATIC_BOARD


def test_a_board_flag_the_rust_worker_does_not_play_is_named() -> None:
    base = dict(
        inference="h:1", rank_fill="q-nocover", baseline_rank_fill="q-nocover", rank_leaf=True,
        baseline_rank_leaf=True,
    )
    assert rustworker.board_unsupported(Namespace(**base)) == []
    assert rustworker.board_unsupported(Namespace(**{**base, "baseline_rank_fill": "refs2"})) == [
        "--baseline-rank-fill refs2 (only a q-nocover fill)"
    ]
    assert rustworker.board_unsupported(Namespace(**{**base, "inference": None})) == [
        "a leaf not on an inference server (--inference)"
    ]
    assert rustworker.board_unsupported(Namespace(**{**base, "rank_leaf": False})) == [
        "the damage ranking (no --rank-leaf)"
    ]
