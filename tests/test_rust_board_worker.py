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
- **a flag the Rust board worker does not play stops the config** (`rustworker.board_unsupported`);
- **each arm plays its own settings, read off the games**: its Q (`--q-arm-named`: the tested
  side's turn-1 menu is the one of a board with one Q, the other side's moves), its store (each
  side's pick is in the support of its arm's solve), its executable (`--rust-binary`: the worker
  sends it the arm's nodes), its width (the widest menu of each side) and its leaf (the side of
  the inverted leaf loses every game). Each was shown to fail with the two arms' setting swapped
  in the config (records/IKA-458.md section 19).

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
    # `tools/profile_stages.py` reads this worker's last line from its log; the control is a line
    # that is not the worker's.
    from ._harness import load_tool

    profile = load_tool("profile_stages")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "worker0.log").write_bytes(stderr.encode("utf-8"))
    found = profile.rust_workers(tmp_path)
    assert found["workers"] == 1 and found["moves"] > 0 and found["moves"] == found["split"], found
    (tmp_path / "logs" / "worker0.log").write_bytes(stderr.replace("pool-match worker", "w").encode())
    assert profile.rust_workers(tmp_path) == {}
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


# ------------------------------------------- each arm's own Q, store, executable, leaf and width


class _Counted:
    """A server leaf that counts the rows it scored."""

    def __init__(self, model) -> None:  # noqa: ANN001
        self.model = model
        self.rows = 0

    def __call__(self, arrays: dict, rows: int) -> np.ndarray:
        self.rows += rows
        return self.model(arrays, rows)


class _AltQ(_ServedStub):
    """Another made-up Q: ranks the same actions in another order than `_ServedStub`."""

    def __init__(self, fingerprint: str) -> None:
        super().__init__(fingerprint)
        self.files = ["alt-q.pt"]

    def __call__(self, arrays):  # noqa: ANN001, ANN204
        self.calls += 1
        a0 = arrays["acts0"].reshape(len(arrays["acts0"]), -1).astype(np.float64).sum(1)
        a1 = arrays["acts1"].reshape(len(arrays["acts1"]), -1).astype(np.float64).sum(1)
        return 0.5 + 0.4 * np.cos(0.91 * a0[:, None] + 0.29 * a1[None, :])


@pytest.fixture(scope="module")
def split(pool):  # noqa: ANN001, ANN201, F811
    """Two leaves, each counted: the tested arm's, and the other arm's that reads every position
    the other way round (one minus it), so the side that plays it plays to lose and its selection
    equilibrium differs. And two Qs (`q`, `alt`)."""
    encoder = Encoder(pool.reg)
    base = _leaf_model(encoder)

    def other(arrays: dict, rows: int) -> np.ndarray:
        return 1.0 - base(arrays, rows)

    leaves = {"value": _Counted(base), "baseline": _Counted(other)}
    qs = {"q": _ServedStub(encoder.vocab.fingerprint()), "alt": _AltQ(encoder.vocab.fingerprint())}
    server, address = serve(leaves, arms={"value": ["stub.pt"], "baseline": ["other.pt"]},
                            q_models=qs)
    yield address, leaves, qs
    server.shutdown()
    server.server_close()


def _first_moves(records: dict) -> dict:
    """Per (game, seat): the first move decision's menus, (side 0's, side 1's)."""
    out = {}
    for key, record in records.items():
        first = next(d for d in record["decisions"] if d["kind"] == "move")
        out[key] = (first["ownActions"], first["foeActions"])
    return out


def test_each_arm_ranks_by_its_own_q(pool, split, tmp_path) -> None:  # noqa: ANN001, F811
    """The other arm ranks by `alt` (`--baseline-rank-fill q-nocover.alt`), the tested arm by `q`.
    Against the same board with `q` on both arms, the tested side's turn-1 menu is the same and the
    other side's moves. Swapped Qs would move the tested side's menu instead (the control)."""
    address, _, qs = split
    flags = _flags(Path(pool.path), address, tmp_path / "s", "leaves")
    both_q, _, _ = _rust_board(flags, tmp_path / "s", address)
    before = qs["alt"].calls
    flags = _flags(Path(pool.path), address, tmp_path / "a", "leaves") + [
        "--baseline-rank-fill", "q-nocover.alt", "--q-arm-named", "alt", "alt"]
    with_alt, config, _ = _rust_board(flags, tmp_path / "a", address)
    assert qs["alt"].calls > before
    same, other = _first_moves(both_q), _first_moves(with_alt)
    moved = 0
    for (game, which), menus in other.items():
        assert menus[which] == same[(game, which)][which], (game, which)
        moved += menus[1 - which] != same[(game, which)][1 - which]
    assert moved > 0, "the other arm's Q never moved its menu: the comparison could not fail"
    assert [config["qServers"][arm["q"]]["model"] for arm in config["arms"]] == ["q", "alt"]


def test_each_arm_draws_and_believes_from_its_own_store(pool, split, tmp_path) -> None:  # noqa: ANN001, F811
    """Each arm's store holds its own leaf's solves (the tag), and each side's pick is in the
    support of its arm's solve (the board plays the equilibrium). Read against the other arm's
    store, some pick is outside its support (the control)."""
    from pokeuraou.poolplay import SolvedSelections

    address, _, _ = split
    flags = _flags(Path(pool.path), address, tmp_path, "leaves")
    records, _, _ = _rust_board(flags, tmp_path, address)
    stores = {}
    for arm, name in (("value", "stub"), ("baseline", "other")):
        tag = f"{pool.sha256}|value:{name}"
        files = sorted((tmp_path / f"store-{arm}").glob("*.json"))
        assert files and all(json.loads(f.read_bytes())["tag"] == tag for f in files), arm
        stores[arm] = SolvedSelections(pool.reg, pool.teams, None, store=tmp_path / f"store-{arm}",
                                       tag=tag)
    ids = [t.id for t in pool.teams]
    mismatched = 0
    for (_, which), record in records.items():
        a, b = (ids.index(t) for t in record["pool"]["teams"])
        side_arms = ("value", "baseline") if which == 0 else ("baseline", "value")
        mine = stores[side_arms[0]].entry(a, b)
        theirs = stores[side_arms[1]].entry(a, b)
        own = mine.selections.index(tuple(record["ownPick"]))
        foe = theirs.selections.index(tuple(record["foePick"]))
        # The board plays the equilibrium (epsilon 0): each pick is in its own arm's support.
        assert mine.our_strategy[own] > 0, (which, "side 0")
        assert theirs.their_strategies[0][foe] > 0, (which, "side 1")
        mismatched += theirs.our_strategy[own] == 0 or mine.their_strategies[0][foe] == 0
    assert mismatched > 0, "every pick is in both arms' supports: the comparison could not fail"


def test_an_arm_on_another_executable_asks_it(pool, split, tmp_path) -> None:  # noqa: ANN001, F811
    """`--rust-binary` (a copy of this build): the worker starts it and sends the tested arm's
    move nodes to it; the record names it on the tested side only. Without the flag nothing is
    sent there (the control)."""
    import shutil

    from pokeuraou.rustnode import binary_path

    address, _, _ = split
    copy = tmp_path / "arm-exe" / binary_path().name
    copy.parent.mkdir()
    shutil.copy2(binary_path(), copy)
    flags = _flags(Path(pool.path), address, tmp_path / "x", "leaves") + ["--rust-binary", str(copy)]
    records, config, stderr = _rust_board(flags, tmp_path / "x", address)
    asked = re.search(r"requests to other executables (\d+)", stderr)
    assert asked and int(asked.group(1)) > 0, stderr
    for (_, which), record in records.items():
        assert record["rustBinary"][which] is not None and record["rustBinary"][1 - which] is None
    assert config["arms"][0]["binary"] == str(copy.resolve()) and config["arms"][1]["binary"] is None
    _, _, plain = _rust_board(_flags(Path(pool.path), address, tmp_path / "p", "leaves"),
                              tmp_path / "p", address)
    assert re.search(r"requests to other executables 0\b", plain), plain


def test_each_side_plays_its_arm_s_leaf_and_width(pool, split, tmp_path) -> None:  # noqa: ANN001, F811
    """Read off the games and the server, not the config: the tested side's menus reach width 4
    and never more, the other side's 3; both leaves were asked, and the side of the inverted leaf
    lost every game. A board of one leaf asks the second leaf nothing (the control of the count).
    Arms whose widths or leaves were swapped fail (records/IKA-458.md section 19)."""
    address, leaves, _ = split
    before = leaves["baseline"].rows
    records, _, _ = _rust_board(_flags(Path(pool.path), address, tmp_path / "w", "leaves"),
                                tmp_path / "w", address)
    assert leaves["baseline"].rows > before and leaves["value"].rows > 0
    widest = {0: 0, 1: 0}  # by arm: 0 tested, 1 other
    for (_, which), record in records.items():
        for d in record["decisions"]:
            if d["kind"] != "move":
                continue
            for side, menu in ((0, d["ownActions"]), (1, d["foeActions"])):
                arm = 0 if side == which else 1
                widest[arm] = max(widest[arm], len(menu))
    assert widest == {0: 4, 1: 3}
    # The leaf: the side of the inverted leaf plays to lose, so the tested arm wins its games.
    won = sum((r["outcome"] > 0.5) == (which == 0) for (_, which), r in records.items())
    assert won == len(records), f"the tested arm won {won} of {len(records)}"
    before = leaves["baseline"].rows
    _rust_board(_flags(Path(pool.path), address, tmp_path / "o", "same"), tmp_path / "o", address)
    assert leaves["baseline"].rows == before
