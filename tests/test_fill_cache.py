"""IKA-471: `resolve::fill_cache` changes no game.

Within one fill, the port answers an action executed from a state an earlier cell already
executed it from with that cell's outcomes. The answer must be the one `execute` would have
given, so the Rust worker's games are held to the games it plays with the cache off
(`POKEURAOU_FILL_CACHE=off`), record for record. The worker's `fill cache:` line shows that the
cache answered something in the run that has it (the positive control that the path ran) and
nothing in the run that does not.

The pool is the two fixture teams, whose first carries two Sucker Punch users: the negative
control, the build with the feature `ika471-cache-control`, leaves Sucker Punch's read out of the
cache's key, and the comparison has to fail against it (records/IKA-471.md section 5.3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from .test_rust_worker import _config, _loaded, _play, _records, served  # noqa: F401

FIXTURES = Path(__file__).parent / "fixtures"

#: Games a run: enough that the negative control's wrong answers reach a record.
GAMES = 6


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    """`test_poolplay._write_pool`'s pool of the two fixture teams, in M-C (their regulation)."""
    names = ("team_a.json", "team_b.json")
    teams = [json.loads((FIXTURES / name).read_text(encoding="utf-8")) for name in names]
    data = {
        "id": "test-pool",
        "name": "test",
        "regulation": "gen9championsvgc2026regmc",
        "character": "a test pool",
        "validatedAgainst": {"formatId": "gen9championsvgc2026regmc"},
        "teams": [{"id": f"t{i}", "name": f"team {i}", "team": team} for i, team in enumerate(teams)],
    }
    path = tmp_path_factory.mktemp("pool") / "p.json"
    path.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    return _loaded(path)


def _cache_line(log: str) -> tuple[int, int, int]:
    found = re.search(r"fill cache: (\d+) lookups, (\d+) hits, (\d+) kept", log)
    assert found, log
    return int(found.group(1)), int(found.group(2)), int(found.group(3))


def test_the_fill_cache_changes_no_game(pool, served, tmp_path, monkeypatch) -> None:  # noqa: ANN001, F811
    on = tmp_path / "on"
    log_on = _play(_config(pool, served, on, on / "store"), served, on / "g.jsonl", 0, GAMES)
    lookups, hits, kept = _cache_line(log_on)
    assert hits > 0 and kept > 0 and lookups >= hits + kept, log_on

    monkeypatch.setenv("POKEURAOU_FILL_CACHE", "off")
    off = tmp_path / "off"
    log_off = _play(_config(pool, served, off, off / "store"), served, off / "g.jsonl", 0, GAMES)
    assert _cache_line(log_off) == (0, 0, 0), log_off

    played_on, played_off = _records(on / "g.jsonl"), _records(off / "g.jsonl")
    assert sorted(played_on) == sorted(played_off)
    assert played_on == played_off
