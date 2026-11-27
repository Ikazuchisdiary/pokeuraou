"""IKA-458 stage 3.7: the board's worker is `pokeuraou-damage pool-match-worker`.

What these tests hold, on the tests' own inference server (two leaves, hp-share and a Q of
stand-ins, no GPU):

- **the Rust board worker writes the games the Python board worker wrote**: the same seed and
  indices played by `tools/pool_match.py` and by the Rust worker, for a board of two leaves and a
  board against hp-share, give the same records key for key and in the same key order, apart from
  the clocks; the policies and values to 1e-9 (the Rust worker sums the selection's mixtures in
  another order than numpy, which moves the bench belief's last bits);
- **the records cannot drift from their templates unseen**: every key of the generation worker's
  and the board worker's record templates is either a key the Rust worker fills per game (its
  `PER_GAME`, read from the source) or one this test names as the same for every game. A key
  `GameRecord.to_json` gains is neither, and this test fails until someone says which it is (a
  per-game key copied from the template would be the same in every record);
- **a flag the Rust board worker does not play stops the config** (`rustworker.board_unsupported`).
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
    return [*flags, "--baseline-inference-arm", "baseline",
            "--baseline-selection-store", str(tmp / "store-baseline")]


def _records(path: Path) -> dict[tuple[int, int], dict]:
    rows = [json.loads(line) for line in path.read_bytes().decode("utf-8").splitlines() if line]
    return {(r["gameIndex"], r["seatIndex"]): r for r in rows}


def _python_board(flags: list[str], tmp: Path) -> dict:
    out = tmp / "python.jsonl"
    done = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "tools" / "pool_match.py"), *flags, "--games-out", str(out)],
        capture_output=True, text=True, timeout=900, check=False, cwd=str(ROOT),
    )
    assert done.returncode == 0, done.stderr
    return _records(out)


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


def _close(got: object, want: object, path: str = "") -> None:
    """Equal, the floats to 1e-9."""
    if isinstance(want, float) or isinstance(got, float):
        assert abs(float(got) - float(want)) <= 1e-9, path
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), path
        for k, (a, b) in enumerate(zip(got, want, strict=True)):
            _close(a, b, f"{path}[{k}]")
    elif isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), path
        for k in want:
            _close(got[k], want[k], f"{path}.{k}")
    else:
        assert got == want, path


@pytest.mark.parametrize("kind", ["leaves", "hp"])
def test_the_rust_board_worker_writes_the_python_board_s_games(pool, served, tmp_path, kind) -> None:  # noqa: ANN001, F811
    pool_path = Path(pool.path)
    python = _python_board(_flags(pool_path, served, tmp_path / "p", kind), tmp_path / "p")
    flags = _flags(pool_path, served, tmp_path / "r", kind)
    rust, config, stderr = _rust_board(flags, tmp_path / "r", served)
    assert len(python) == 2 * GAMES, "no game finished; the comparison would be vacuous"
    assert sorted(rust) == sorted(python)
    # The positive control: each side's own node (two leaves, or a leaf against hp-share).
    assert re.search(r"move nodes (\d+) \(\1 with each side's own node\)", stderr), stderr
    if kind == "hp":
        assert config["arms"][1]["store"] is None
    for key, want in python.items():
        got = rust[key]
        assert list(got) == list(want)
        drop = ("searchSeconds", "engine")
        _close({k: v for k, v in got.items() if k not in drop},
               {k: v for k, v in want.items() if k not in drop})


def test_the_records_keys_are_per_game_or_named_static(pool, served, tmp_path) -> None:  # noqa: ANN001, F811
    from pokeuraou import qrank
    from pokeuraou.inference import RemoteValue

    encoder = Encoder(pool.reg)
    leaf = RemoteValue(served, "value", encoder)
    q = qrank.RemoteQ(served, "q", encoder)
    qrank.install(q)
    try:
        args = Namespace(
            inference=served, rank_leaf=True, rank_fill="q-nocover", bench_drop="none",
            deepen="none", depth=1, solve_restricted=False, seed=1, max_turns=40,
            explore_epsilon=0.25, explore_temperature=0.5, limit=4,
            uniform_selection=False, no_selection_memo=False, record_rank_scores=False,
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
        baseline_rank_leaf=True, line1_back=None, line_only_with=None, ally_only_with=None,
        drive_baton=False, drive_screens=None,
    )
    for prefix in ("", "baseline_"):
        base.update({
            f"{prefix}bench_drop": "none", f"{prefix}deepen": "none", f"{prefix}depth": 1,
            f"{prefix}solve_restricted": False, f"{prefix}eq_select": "lp",
            f"{prefix}knockouts": False, f"{prefix}line1": False, f"{prefix}hold_baton": False,
            f"{prefix}bench_oracle": False, f"{prefix}net_scores_ends": False,
            f"{prefix}ally_targets": "off",
        })
    assert rustworker.board_unsupported(Namespace(**base)) == []
    assert rustworker.board_unsupported(Namespace(**{**base, "baseline_deepen": "m8"})) == [
        "--baseline-deepen m8"
    ]
    # Flags with no baseline form are named too, each as its own flag.
    for name, value, said in (
        ("line1_back", ["a", "b"], "--line1-back ['a', 'b']"),
        ("line_only_with", ["x"], "--line-only-with ['x']"),
        ("ally_only_with", ["x"], "--ally-only-with ['x']"),
        ("drive_baton", True, "--drive-baton"),
        ("drive_screens", "x", "--drive-screens x"),
    ):
        assert rustworker.board_unsupported(Namespace(**{**base, name: value})) == [said]
    assert rustworker.board_unsupported(Namespace(**{**base, "inference": None}))[0].startswith(
        "leaves loaded in the worker"
    )
    assert rustworker.board_unsupported(Namespace(**{**base, "rank_leaf": False})) == [
        "the damage ranking (no --rank-leaf)"
    ]
