"""A pairing replayed on one seed's draws is one look, not two (IKA-44).

Games are seeded from `[seed, index]`, so two matches launched with one seed deal the same
teams and the same fours in the same order. When the two also field the same pair of
agents, the second run is a second look at the first run's draws, and a fit that counts
both as independent reads twice the games and shrinks the interval by about root two for
information it does not have.

The corpus had three such pairs on 2026-09-23, all with every draw shared: one where the
rerun is the same games move for move (`gen11L-vs-gen10-ownroster-uniform` and its crashed
twin), and two where the rerun is a later build on the same draws, correlated but not
identical (`hidden-gen11h-vs-gen10` and `-fixed`, `h12-vs-o12-hidden` and its crashed
twin). The rule `ratings.py` now applies -- count the pairing's draws once, from the
newest run -- is exact for the first and is what a rerun that exists because the first
run was wrong asks for in the other two.

The first test is the one the tool failed before the change: its interval must be the
interval of one run. The others pin what the rule must NOT do, because a rule that drops
runs is one bad key away from dropping independent evidence: a shared seed with other
draws, or with another pairing, repeats nothing and is counted.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from ._harness import load_tool

ratings = load_tool("ratings")

GAMES = 100  # game indices per run; each is played in both seats, so 200 games a run


def _provenance(side0: str, side1: str) -> dict:
    return {
        "kind": "generation-match",
        "seat": f"{side0} = side 0",
        "leaves": [side0, side1],
        "limits": [24, 24],
        "depths": [1, 1],
        "rankings": ["damage", "damage"],
        "solvers": ["full", "full"],
        "books": ["uniform", "uniform"],
        "information": ["open", "open"],
    }


def write_run(
    root: Path,
    name: str,
    *,
    seed: int,
    arms: tuple[str, str] = ("x", "y"),
    program: int = 0,
    field: str = "foe",
    mtime: int = 1_700_000_000,
    done: str | None = "ok",
    workers: int = 2,
) -> Path:
    """One queued match in the layout `tools/match_queue.py` writes.

    Index `i` is game `i // 2` in seat `i % 2`, dealt round robin to `workers` files. Both
    seats of a game share its draw (`own{g}` against `{field}{g}`), which is what the real
    runs do: the draw comes off the stream seeded by the game index, before either seat
    plays. `program` stands for the build: the same program on the same draws plays the
    same games, a different one plays different games on them.
    """
    directory = root / name
    directory.mkdir(parents=True)
    rng = np.random.default_rng(program)
    tested_wins = rng.random(GAMES) < 0.6
    files = [(directory / f"games-worker{k}.jsonl").open("w", encoding="utf-8") for k in range(workers)]
    tally = [[[0, 0], [0, 0]] for _ in range(workers)]  # per worker, per seat: played, wins
    for index in range(2 * GAMES):
        game, seat = index // 2, index % 2
        side0, side1 = (arms[0], arms[1]) if seat == 0 else (arms[1], arms[0])
        won = bool(tested_wins[game])
        outcome = float(won) if seat == 0 else float(not won)
        record = {
            "ownTeam": [{"species": f"own{game}"}],
            "foeTeam": [{"species": f"{field}{game}"}],
            "outcome": outcome,
            "engine": {"sources": f"program{program}"},
            "provenance": _provenance(side0, side1),
        }
        worker = index % workers
        files[worker].write(json.dumps(record) + "\n")
        tally[worker][seat][0] += 1
        tally[worker][seat][1] += int(won)
    for handle in files:
        handle.close()
    for worker in range(workers):
        with (directory / f"worker{worker}.jsonl").open("w", encoding="utf-8") as handle:
            for seat in (0, 1):
                played, wins = tally[worker][seat]
                handle.write(
                    json.dumps(
                        {
                            "seat": f"{arms[0]} = side {seat}",
                            "seed": seed,
                            "model": arms[0],
                            "baseline": arms[1],
                            "limit": 24,
                            "played": played,
                            "gen2_wins": wins,
                        }
                    )
                    + "\n"
                )
        os.utime(directory / f"games-worker{worker}.jsonl", (mtime, mtime))
    if done is not None:
        (directory / "DONE").write_text(done + "\n", encoding="utf-8")
    return directory


def table(root: Path, *extra: str) -> tuple[dict[str, tuple[float, float, int]], str]:
    """Run the tool's own entry point and read its table back: name -> (Elo, +-, games)."""
    argv = sys.argv
    sys.argv = ["ratings.py", "--matches", str(root), "--anchor", "y/w24", *extra]
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            ratings.main()
    finally:
        sys.argv = argv
    text = out.getvalue()
    rows: dict[str, tuple[float, float, int]] = {}
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.split()[:2] == ["agent", "Elo"])
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) < 6 or "/" not in parts[0]:
            break
        rows[parts[0]] = (float(parts[1]), float(parts[2]), int(parts[3]))
    return rows, text


def test_a_pairing_replayed_on_one_seed_is_one_run(tmp_path: Path) -> None:
    """The defect as it was: the second look at one seed's draws halved the variance.

    Two runs, one seed, one pairing, one program -- the rerun is the first run's games
    move for move, which is what `gen11L-vs-gen10-ownroster-uniform` and its crashed twin
    are on disk. Counted once, the table must be exactly the table of one run: same
    rating, same interval, same games. Counted twice, the rating is the same and the
    interval is about 1/root 2 of it, which is the claim this issue made.
    """
    alone = tmp_path / "alone"
    write_run(alone, "first", seed=31337)
    both = tmp_path / "both"
    write_run(both, "first", seed=31337, mtime=1_700_000_000)
    write_run(both, "first-rerun", seed=31337, mtime=1_700_003_600)

    one, _ = table(alone)
    two, _ = table(both)
    assert two["x/w24"][1] == one["x/w24"][1], (
        f"+-{two['x/w24'][1]} Elo from two looks at one seed's draws against +-{one['x/w24'][1]} "
        "from one: the replay was counted as independent evidence"
    )
    assert two["x/w24"][2] == one["x/w24"][2] == 2 * GAMES
    assert two["x/w24"][0] == one["x/w24"][0]


def test_the_newest_run_is_the_one_counted(tmp_path: Path) -> None:
    """A rerun on the same draws by a later build: the later build's games are the fit's.

    `hidden-gen11h-vs-gen10-fixed` exists because the first run predates two search
    fixes. Averaging the two would put a search nobody uses into the agents' ratings, so
    the rule keeps one run and it has to be the newer.
    """
    root = tmp_path / "both"
    write_run(root, "early", seed=7, program=1, mtime=1_700_000_000)
    write_run(root, "late", seed=7, program=2, mtime=1_700_009_000)
    only_late = tmp_path / "late"
    write_run(only_late, "late", seed=7, program=2)

    got, text = table(root)
    want, _ = table(only_late)
    assert got["x/w24"] == want["x/w24"]
    assert "early" in text and "not counted" in text


def test_a_failed_run_is_not_the_one_kept(tmp_path: Path) -> None:
    """A run whose DONE says FAILED loses to one that finished, however new it is."""
    root = tmp_path / "both"
    write_run(root, "finished", seed=7, program=1, mtime=1_700_000_000)
    write_run(root, "crashed", seed=7, program=2, mtime=1_700_009_000, done="FAILED (exit 1)")
    only = tmp_path / "only"
    write_run(only, "finished", seed=7, program=1)

    got, _ = table(root)
    want, _ = table(only)
    assert got["x/w24"] == want["x/w24"]


def test_one_seed_with_other_draws_is_two_runs(tmp_path: Path) -> None:
    """Same seed, same pairing, different draws: independent, and both are counted.

    `anchor-gen11L-vs-hpshare` shares seed 20260919 with two other matches and not one of
    its 1,696 draws, because its selection is drawn uniformly and theirs from a book --
    the stream is the same, what is drawn from it is not. A rule keyed on the seed alone
    would throw away independent games; this is the negative control for that.
    """
    root = tmp_path / "root"
    write_run(root, "one", seed=77, program=1, mtime=1_700_000_000)
    write_run(root, "other-field", seed=77, program=2, field="other", mtime=1_700_009_000)
    got, text = table(root)
    assert got["x/w24"][2] == 4 * GAMES
    assert "only 0% of the smaller one's draws are the other's -- both counted" in text


def test_one_seed_across_pairings_is_all_counted(tmp_path: Path) -> None:
    """Seed 77 is shared by eleven runs of eleven different pairings; none repeats another.

    The draws are common to x-y and x-z, but no pairing sees them twice, so nothing is a
    replay and every game is counted.
    """
    root = tmp_path / "root"
    write_run(root, "x-vs-y", seed=77, arms=("x", "y"))
    write_run(root, "x-vs-z", seed=77, arms=("x", "z"))
    got, _ = table(root)
    assert got["x/w24"][2] == 4 * GAMES
    assert got["z/w24"][2] == 2 * GAMES


def test_independent_convention_is_the_old_count(tmp_path: Path) -> None:
    """`--shared-seed independent` is the tool before IKA-44, kept so old numbers can be
    reproduced -- and named in the output, because the choice moves the interval."""
    root = tmp_path / "both"
    write_run(root, "first", seed=31337, mtime=1_700_000_000)
    write_run(root, "first-rerun", seed=31337, mtime=1_700_003_600)
    newest, said_newest = table(root)
    independent, said_independent = table(root, "--shared-seed", "independent")
    assert independent["x/w24"][2] == 2 * newest["x/w24"][2]
    assert independent["x/w24"][1] < newest["x/w24"][1]
    assert 'convention "newest"' in said_newest
    assert 'convention "independent"' in said_independent


def test_a_worker_that_wrote_no_rows_takes_its_directory_seed(tmp_path: Path) -> None:
    """A worker that died leaves games and no per-seat rows -- which is how a crashed run
    looks, and a crashed run is the likeliest replay there is. Its games must still be
    recognised as the run's.

    The crashed run is the older one, so it is the one not counted, and the orphaned
    worker's games are exactly what would stay behind if they were not recognised: the
    first version of this test had the crashed run newer, kept it whole either way, and
    passed with the directory seed switched off.
    """
    root = tmp_path / "root"
    crashed = write_run(root, "crashed", seed=5, mtime=1_700_000_000, done="FAILED (exit 1)")
    (crashed / "worker1.jsonl").unlink()
    write_run(root, "rerun", seed=5, mtime=1_700_009_000)
    got, _ = table(root)
    assert got["x/w24"][2] == 2 * GAMES


def test_a_cache_from_before_the_draws_is_read_again(tmp_path: Path) -> None:
    """The cache written before this change has no draws in it. Trusting it would find no
    replay anywhere and quietly restore the old count -- the fix present and unreached."""
    root = tmp_path / "both"
    write_run(root, "first", seed=31337, mtime=1_700_000_000)
    write_run(root, "first-rerun", seed=31337, mtime=1_700_003_600)
    table(root)
    cache_path = root / ".ratings-cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    for entry in cache.values():
        entry.pop("draws", None)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    got, _ = table(root)
    assert got["x/w24"][2] == 2 * GAMES
