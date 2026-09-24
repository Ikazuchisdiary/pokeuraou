"""`tools/diff_turn.py` draws the same teams whatever PYTHONHASHSEED is (IKA-218).

`load_chaos` builds the ability table of a species seen only as its mega forme by iterating
a set, so the table's order followed the hash seed, and `weighted_choice` drew another
ability from the same generator state: the same `--seed` played other turns (IKA-207 saw
2,726 and 2,727 compared turns). `diff_turn.hash_free_prior` puts the tables in the dex's
order. Two interpreters with different hash seeds are the only way to see it, so the check
runs in subprocesses; the raw tables are the positive control (they do differ).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import json, sys
sys.path.insert(0, {tools!r})
import diff_turn
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.regulation import load_regulation
reg = load_regulation(diff_turn.FORMAT_ID)
path = find_cached_chaos(diff_turn.FORMAT_ID)
if path is None:
    print("null"); raise SystemExit
raw = {{s: list(p.abilities) for s, p in load_chaos(path, reg).species.items()}}
fixed = diff_turn.hash_free_prior(reg, load_chaos(path, reg))
print(json.dumps({{"raw": raw, "fixed": {{s: list(p.abilities) for s, p in fixed.species.items()}}}}))
"""


def _tables(hash_seed: str) -> dict | None:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = str(ROOT / "src")
    done = subprocess.run(
        [sys.executable, "-c", _PROBE.format(tools=str(ROOT / "tools"))],
        env=env, capture_output=True, text=True, check=True, cwd=ROOT,
    )
    return json.loads(done.stdout)


def test_the_ability_tables_do_not_follow_the_hash_seed() -> None:
    zero, one = _tables("0"), _tables("1")
    if zero is None:
        pytest.skip("no cached usage stats; run tools/fetch_priors.py")
    assert zero["raw"] != one["raw"], "positive control: the raw tables should follow the seed"
    assert zero["fixed"] == one["fixed"]
    # Only the order moved: the same abilities, the same species.
    assert {s: sorted(a) for s, a in zero["fixed"].items()} == {
        s: sorted(a) for s, a in zero["raw"].items()
    }
