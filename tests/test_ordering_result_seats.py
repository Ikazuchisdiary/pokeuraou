"""`tools/ordering_result.py` must read the tested arm's result from the seat it sat in.

Since IKA-17 `selection_check` plays every matchup in both seats and writes one row per
seat, with `outcome` always SIDE 0's and `seat` saying which side our four held. The
reader used to take `outcome` as our result whatever the seat (IKA-134), so on a two-seat
run half the rows were the opponent's result: the printed rates were pulled toward 50%
and a difference whose seat-1 half had the opposite sign came out with that sign. No
error was raised. These are synthetic rows, because no two-seat ordering run exists on
disk yet; the rows are written the way `selection_check` writes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ._harness import load_tool

ordering_result = load_tool("ordering_result")

HEAVY = "charizard+garchomp / toxapex+incineroar"
SECOND = "charizard+sylveon / venusaur+garchomp"


def _row(arm: str, game: int, seat: int, we_won: bool) -> dict:
    # `outcome` is side 0's: in seat 1 our win is side 0's loss.
    side0_won = we_won if seat == 0 else not we_won
    return {"arm": arm, "game": game, "seat": seat, "outcome": 1.0 if side0_won else 0.0}


def _write(path: Path, rows: list[dict], *, header: bool = True) -> Path:
    lines = [json.dumps({"header": {"claimed": 0.5, "seats": 2}})] if header else []
    lines += [json.dumps(r) for r in rows]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def _two_seat_run(tmp_path: Path) -> Path:
    """The heavy four wins 3 of 4 matchups, the second wins 1 of 4, in BOTH seats."""
    rows = []
    for game in range(4):
        for seat in (0, 1):
            rows.append(_row(HEAVY, game, seat, we_won=game != 3))
            rows.append(_row(SECOND, game, seat, we_won=game == 0))
    return _write(tmp_path / "place1.part0.jsonl", rows)


def test_two_seat_rates_are_the_tested_arms(tmp_path: Path) -> None:
    by_arm = ordering_result.read_place([_two_seat_run(tmp_path)])
    diffs, hp, sp, n = ordering_result.paired(by_arm, HEAVY, SECOND)
    assert n == 8
    assert hp == 0.75
    assert sp == 0.25
    assert sum(diffs) / len(diffs) == 0.5
    # Paired per matchup: both seats of one game are one pair, not two.
    assert len(diffs) == 4


def test_the_rate_does_not_depend_on_the_seat(tmp_path: Path) -> None:
    """Seat 0 alone, seat 1 alone and both together must give the same answer."""
    rows = [json.loads(x) for x in _two_seat_run(tmp_path).read_text().splitlines()[1:]]
    answers = []
    for keep in ({0}, {1}, {0, 1}):
        d = tmp_path / f"seats{''.join(map(str, sorted(keep)))}"
        d.mkdir()
        path = _write(d / "place1.part0.jsonl", [r for r in rows if r["seat"] in keep])
        _diffs, hp, sp, _n = ordering_result.paired(
            ordering_result.read_place([path]), HEAVY, SECOND
        )
        answers.append((hp, sp))
    assert answers == [(0.75, 0.25)] * 3


def test_seatless_rows_read_as_side_0(tmp_path: Path) -> None:
    """Everything on disk before IKA-17 is one seat, our four at side 0, with no `seat`."""
    rows = [
        {"arm": HEAVY, "game": 0, "outcome": 1.0},
        {"arm": SECOND, "game": 0, "outcome": 0.0},
    ]
    path = _write(tmp_path / "place1.part0.jsonl", rows, header=False)
    by_arm = ordering_result.read_place([path])
    assert by_arm[HEAVY] == {(0, 0): 1.0}
    assert by_arm[SECOND] == {(0, 0): 0.0}


def test_seatless_and_seated_rows_do_not_mix(tmp_path: Path) -> None:
    """Pooling a one-seat run with a two-seat one puts the seat back into half of it."""
    rows = [
        {"arm": HEAVY, "game": 0, "outcome": 1.0},
        _row(HEAVY, 1, 1, we_won=True),
    ]
    path = _write(tmp_path / "place1.part0.jsonl", rows)
    with pytest.raises(SystemExit, match="seat"):
        ordering_result.read_place([path])


def test_a_duplicate_seat_game_is_refused(tmp_path: Path) -> None:
    """Two rows for one (arm, game, seat) are an overlapping shard split, not two games."""
    a = _write(tmp_path / "place1.part0.jsonl", [_row(HEAVY, 0, 1, we_won=True)])
    b = _write(tmp_path / "place1.part1.jsonl", [_row(HEAVY, 0, 1, we_won=False)])
    with pytest.raises(SystemExit, match="two"):
        ordering_result.read_place([a, b])


def test_the_reader_uses_the_shared_flip() -> None:
    """The flip lives once, in `tools/seats.py`; a third copy is how a sign gets lost."""
    body = (Path(__file__).resolve().parents[1] / "tools" / "ordering_result.py").read_text(
        encoding="utf-8"
    )
    assert "from seats import" in body
    assert "our_win(" in body
