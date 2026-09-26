"""The port's `encode-games` writes the bytes `tools/encode_dataset.py`'s own loop wrote (IKA-347).

`encode_dataset.py` hands its per-game loop to the port and keeps the Python loop as
`--engine python`. Both roads are run here on the same small directory of real games and
the shards they cache are compared member by member, meta included -- not within a
tolerance: the arrays are the value function's input.

The directory is built to carry what the Python loop had rules for: an unfinished game
(no outcome), a blank line, a partial last line, a game from another provenance, a
missing meta key beside a present-but-null one, a decision without `foeSearchValue`, and
two files so the join renumbers games and opponents. `--kinds`, `--limit` and `--jobs`
are compared the same way.

The positive control is a build with `--features ika347-control` (the turn feature one slot
along), pointed at through `POKEURAOU_RUST_NODE_BIN`: every comparison here fails on it.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import play_game
from pokeuraou.teams import load_roster
from tests._harness import load_tool

# `tools/encode_dataset.py` imports `pokeuraou.value`, which imports torch at module scope.
pytest.importorskip("torch", reason="the dataset needs the optional learn group")
encode_dataset = load_tool("encode_dataset")


def _game(seed: int, max_turns: int, hidden: bool) -> dict:
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    record = play_game(
        reg, np.random.default_rng(seed), sheet[:4], [sheet[i] for i in (1, 0, 4, 5)], "test",
        search_limit=2, max_turns=max_turns,
        sheets=(sheet, sheet) if hidden else None,
        open_information=not hidden,
    )
    return json.loads(json.dumps(record.to_json(objective="hp-share", search_limit=2)))


@pytest.fixture(scope="module")
def games() -> list[dict]:
    """A finished hidden-bench game, one stopped by the cap, and a second finished one."""
    return [_game(5, 40, True), _game(8, 3, False), _game(11, 6, True)]


def _line(record: dict) -> bytes:
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def _write_games(directory: Path, games: list[dict]) -> None:
    """Two files of the games and their variants, with the lines the loop skips."""
    directory.mkdir()
    a, b, c = games
    unfinished = copy.deepcopy(a)
    unfinished["outcome"] = None
    assert b["outcome"] is None, "the capped game is the one left unfinished"
    matched = copy.deepcopy(a)
    matched["provenance"] = {"kind": "generation-match", "arms": ["x", "y"]}
    matched["foeArchetype"] = "mirror"
    bare = copy.deepcopy(c)
    for key in ("information", "searchLimit", "engine", "foeArchetype"):
        bare.pop(key, None)
    bare["selectionSource"] = None
    for decision in bare["decisions"][::2]:
        decision.pop("foeSearchValue", None)
    (directory / "games-worker0.jsonl").write_bytes(
        _line(a) + b"\n" + _line(unfinished) + _line(b) + _line(matched) + _line(bare)
    )
    (directory / "games-worker1.jsonl").write_bytes(
        _line(c) + _line(matched) + _line(a)[:5000]  # a run still writing
    )


def _args(**overrides: object) -> argparse.Namespace:
    args = argparse.Namespace(
        kinds=None, limit=0, force=True, regulation=None, chunk=4096, engine="rust", jobs=1
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _members(path: Path) -> dict[str, bytes]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].tobytes() + str(data[name].dtype).encode() for name in data.files}


def _differing(a: dict[str, bytes], b: dict[str, bytes]) -> list[str]:
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def test_the_port_writes_the_python_loops_shard(tmp_path: Path, games: list[dict]) -> None:
    directory = tmp_path / "games"
    _write_games(directory, games)
    shard = encode_dataset.shard_path(directory)
    written = {}
    for engine in ("python", "rust"):
        dataset, meta = encode_dataset.encode_dir(directory, _args(engine=engine))
        written[engine] = _members(shard)
        shard.rename(tmp_path / f"{engine}.npz")
    assert _differing(written["python"], written["rust"]) == []
    # The directory exercised what it was built for.
    assert meta["games"] == 5
    assert meta["provenances"] == {"selfplay": 3, "generation-match": 2}
    assert len(dataset.foe_names) == 3  # "pool", "mirror" and the missing key's "?"
    assert int(np.isnan(dataset.foe_search_value).sum()) > 0
    assert meta["selections"].get("None") == 1


@pytest.mark.parametrize(
    "overrides",
    [{"kinds": ["generation-match"]}, {"limit": 2}, {"limit": 4}, {"jobs": 2}],
    ids=["kinds", "limit-2", "limit-4-across-files", "jobs-2"],
)
def test_the_flags_filter_as_the_python_loop_did(
    tmp_path: Path, games: list[dict], overrides: dict
) -> None:
    directory = tmp_path / "games"
    _write_games(directory, games)
    got = {}
    for engine in ("python", "rust"):
        dataset, meta = encode_dataset.encode_dir(
            directory, _args(engine=engine, force=True, **overrides)
        )
        out = tmp_path / f"{engine}.npz"
        encode_dataset.save_dataset(out, dataset, meta={k: v for k, v in meta.items() if k != "source_dir"})
        got[engine] = _members(out)
        cached = encode_dataset.shard_path(directory)
        if cached.exists():
            cached.unlink()
    assert _differing(got["python"], got["rust"]) == []


def test_a_partial_line_mid_file_stops_the_port(tmp_path: Path, games: list[dict]) -> None:
    """Python skipped any line it could not parse; the port skips only a file's last one."""
    directory = tmp_path / "games"
    directory.mkdir()
    (directory / "games.jsonl").write_bytes(_line(games[0])[:5000] + b"\n" + _line(games[2]))
    with pytest.raises(SystemExit, match="encode-games failed"):
        encode_dataset.encode_dir(directory, _args())
