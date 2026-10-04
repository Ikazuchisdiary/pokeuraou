"""`encode_dataset.py --dir A B --out X` joins the shards from the files (IKA-430).

Until IKA-430 the join loaded every shard and wrote `save_dataset(concat_datasets(...))`;
for gen-0..4 that held 13 GB (IKA-427 §2.4). `write_joined` writes the same npz while
holding only the per-decision columns: the big arrays go from each shard's file to the
output a chunk of rows at a time. Here three shards (two of real games, one without
`foe_search_value`) are joined both ways and compared member by member, meta included,
with chunks of a few rows so every member crosses chunk and shard boundaries.

Positive control: shift the game offset in `join_small` (e.g. drop the `+ 1`) and the
`game` member differs; read the shards in reverse in `write_joined` and the big members
differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tests._harness import load_tool

pytest.importorskip("torch", reason="the dataset needs the optional learn group")
encode_dataset = load_tool("encode_dataset")
from tests.test_encode_games_port import _args, _game, _line, _members, _write_games  # noqa: E402


@pytest.fixture(scope="module")
def played() -> list[dict]:
    """test_encode_games_port's three games."""
    return [_game(5, 40, True), _game(8, 3, False), _game(11, 6, True)]


def _shards(tmp_path: Path, played: list[dict]) -> list[Path]:
    a, b, c = played
    first = tmp_path / "gen-a"
    _write_games(first, played)
    second = tmp_path / "gen-b"
    second.mkdir()
    relabel = json.loads(json.dumps(c))
    relabel["foeArchetype"] = "other"
    (second / "games-worker0.jsonl").write_bytes(_line(c) + _line(relabel) + _line(a))
    third = tmp_path / "gen-c"
    third.mkdir()
    (third / "games-worker0.jsonl").write_bytes(_line(a) + _line(c))
    dirs = [first, second, third]
    for d in dirs:
        encode_dataset.encode_dir(d, _args())
    # A shard from before `foe_search_value` (IKA-127): the join fills it with NaN.
    old = encode_dataset.shard_path(third)
    with np.load(old, allow_pickle=False) as data:
        members = {k: data[k] for k in data.files if k != "foe_search_value"}
    np.savez_compressed(old, **members)
    return dirs


def test_the_streamed_join_writes_the_in_memory_join(
    tmp_path: Path, played: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    from pokeuraou import packed
    from pokeuraou.value import concat_datasets, load_dataset, save_dataset

    dirs = _shards(tmp_path, played)
    shards = [encode_dataset.shard_path(d) for d in dirs]
    args = argparse.Namespace(dir=dirs)
    metas = [json.loads(str(np.load(s, allow_pickle=False)["meta_json"])) for s in shards]
    meta = encode_dataset._joined_meta(args, metas)

    memory = tmp_path / "memory.npz"
    save_dataset(memory, concat_datasets([load_dataset(s, cache=False) for s in shards]), meta=meta)

    monkeypatch.setattr(packed, "CHUNK_ROWS", 7)
    streamed = tmp_path / "streamed.npz"
    joined = encode_dataset.join_small(shards)
    encode_dataset.write_joined(streamed, joined, meta)

    want, got = _members(memory), _members(streamed)
    assert sorted(want) == sorted(got)
    assert [k for k in want if want[k] != got[k]] == []
    with np.load(streamed, allow_pickle=False) as data:
        assert list(data.files) == list(np.load(memory, allow_pickle=False).files)
        n = [len(np.load(s)["outcome"]) for s in shards]
        assert len(data["outcome"]) == sum(n)
        # the join exercised what it is for: games renumbered, foes remapped, NaN filled
        assert int(data["game"].max()) + 1 == sum(m["games"] for m in metas)
        assert "other" in json.loads(str(data["meta_json"]))["foe_names"]
        assert np.isnan(data["foe_search_value"][-n[2]:]).all()


def test_main_joins_from_the_shards(
    tmp_path: Path, played: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command line takes the streamed road and writes what `save_dataset` would."""
    from pokeuraou.value import concat_datasets, load_dataset, save_dataset

    dirs = _shards(tmp_path, played)
    out = tmp_path / "joined.npz"
    monkeypatch.setattr(
        "sys.argv", ["encode_dataset.py", "--dir", *map(str, dirs), "--out", str(out)]
    )

    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("the join loaded a shard")

    monkeypatch.setattr(encode_dataset, "load_dataset", refuse)
    encode_dataset.main()
    shards = [encode_dataset.shard_path(d) for d in dirs]
    metas = [json.loads(str(np.load(s, allow_pickle=False)["meta_json"])) for s in shards]
    memory = tmp_path / "memory.npz"
    save_dataset(
        memory,
        concat_datasets([load_dataset(s, cache=False) for s in shards]),
        meta=encode_dataset._joined_meta(argparse.Namespace(dir=dirs), metas),
    )
    want, got = _members(memory), _members(out)
    assert [k for k in want.keys() | got.keys() if want.get(k) != got.get(k)] == []
