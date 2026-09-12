"""Joining per-generation shards must not merge two generations' game numbers.

The split that keeps validation honest is by game, and every shard numbers its games from
zero. If the join left those numbers alone, generation 6's game 12 and generation 7's game
12 would be one game to `split_by_game`, and 26 decisions carrying two different outcomes
would land together on one side of the split. That is the same failure -- near-duplicate
labels straddling the split -- that splitting by game exists to prevent, reintroduced by
the thing that made encoding cheap.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.encode import Encoded
from pokeuraou.value import Dataset, concat_datasets


def shard(n: int, *, games: int, foe_names: tuple[str, ...], outcome: float) -> Dataset:
    """A dataset with the right shapes and nothing real in it."""
    encoded = Encoded(
        species=np.zeros((n, 2, 4), np.int64),
        ability=np.zeros((n, 2, 4), np.int64),
        item=np.zeros((n, 2, 4), np.int64),
        moves=np.zeros((n, 2, 4, 4), np.int64),
        mon=np.zeros((n, 2, 4, 3), np.float32),
        mask=np.ones((n, 2, 4), np.float32),
        side=np.zeros((n, 2, 2), np.float32),
        field=np.zeros((n, 2), np.float32),
        unknown_volatiles={"ghost": n},
    )
    return Dataset(
        encoded=encoded,
        outcome=np.full(n, outcome, np.float32),
        game=np.arange(n, dtype=np.int32) % games,
        turn=np.arange(n, dtype=np.int16),
        search_value=np.full(n, outcome, np.float32),
        hp_share=np.full(n, outcome, np.float32),
        kind=np.zeros(n, np.int8),
        foe=np.arange(n, dtype=np.int32) % len(foe_names),
        foe_names=foe_names,
    )


def test_game_numbers_do_not_collide() -> None:
    a = shard(6, games=3, foe_names=("worlds",), outcome=1.0)
    b = shard(4, games=2, foe_names=("worlds",), outcome=0.0)
    joined = concat_datasets([a, b])

    assert len(joined) == 10
    from_a = set(joined.game[:6].tolist())
    from_b = set(joined.game[6:].tolist())
    assert from_a == {0, 1, 2}
    assert not (from_a & from_b), "a game number must mean one game across the whole pool"
    # Every decision of one game carries one outcome, which is the property the split
    # relies on; a collision would break it and nothing else here would notice.
    for game in np.unique(joined.game):
        assert len(set(joined.outcome[joined.game == game].tolist())) == 1


def test_opponent_labels_are_remapped_not_reused() -> None:
    a = shard(4, games=2, foe_names=("worlds", "regional"), outcome=1.0)
    b = shard(4, games=2, foe_names=("mirror", "worlds"), outcome=0.0)
    joined = concat_datasets([a, b])

    assert joined.foe_names == ("worlds", "regional", "mirror")
    labels = [joined.foe_names[i] for i in joined.foe]
    assert labels[:4] == ["worlds", "regional", "worlds", "regional"]
    # Shard b's index 1 was "worlds" locally and must not stay 1, which is "regional".
    assert labels[4:] == ["mirror", "worlds", "mirror", "worlds"]


def test_unknown_volatiles_are_summed() -> None:
    joined = concat_datasets(
        [
            shard(3, games=1, foe_names=("worlds",), outcome=1.0),
            shard(5, games=1, foe_names=("worlds",), outcome=0.0),
        ]
    )
    assert joined.encoded.unknown_volatiles == {"ghost": 8}


def test_a_shard_without_hp_share_is_refused() -> None:
    a = shard(4, games=2, foe_names=("worlds",), outcome=1.0)
    b = shard(4, games=2, foe_names=("worlds",), outcome=0.0)
    b.hp_share = np.zeros(0, np.float32)
    with pytest.raises(ValueError, match="hp_share"):
        concat_datasets([a, b])


def test_one_shard_is_returned_unchanged() -> None:
    a = shard(4, games=2, foe_names=("worlds",), outcome=1.0)
    assert concat_datasets([a]) is a
