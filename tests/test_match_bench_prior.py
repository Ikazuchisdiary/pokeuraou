"""A match's bench belief belongs to the arm, in both seats (IKA-122).

Generation has handed `play_game` a `bench_prior` since IKA-5: the belief over the hidden
bench is the opponent's selection equilibrium, not a uniform draw over every pair the
sheet allows. `generation_match` never did, so every `--hide-bench` match measured an
agent generation does not ship -- and `agent_drift` called it the same agent, because
`bench_prior` was not on its list.

What the fix has to get right is whose belief it is. `bench_prior[s]` prices side `s`'s
bench, so it is side `1 - s`'s search that reads it -- and it must come from THAT arm's
book, not from the book side `s` actually drew from (which is the opponent's private
strategy). An arm without a book, or told to keep the uniform belief, holds the uniform
one; and all of it has to follow the arm when the seats swap.

Played through `main` with the loaders and `play_game` replaced, so what is under test is
the tool's own plumbing: point masses on four different selections, so a belief taken from
the wrong book, the wrong side or the wrong seat cannot land on the right answer.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou.provenance import agent_name
from pokeuraou.selection_book import BookEntry
from pokeuraou.standings import TeamMember, TournamentTeam
from pokeuraou.teams import all_selections, load_roster
from tests._harness import load_tool

SELECTIONS = tuple(all_selections(6, 4))
# Tested arm's book: our side plays 3, their side 4. The other arm's: 7 and 9.
TESTED_OURS, TESTED_THEIRS, OTHER_OURS, OTHER_THEIRS = 3, 4, 7, 9


def _point_mass(index: int) -> np.ndarray:
    out = np.zeros(len(SELECTIONS))
    out[index] = 1.0
    return out


def _entry(ours: int, theirs: int, sets, model: str) -> BookEntry:  # noqa: ANN001
    return BookEntry(
        key="k",
        player="Opponent",
        place=1,
        selections=SELECTIONS,
        our_strategy=_point_mass(ours),
        our_ev_loss=np.zeros(len(SELECTIONS)),
        class_weights=np.ones(1),
        class_sets=(tuple(sets),),
        their_strategies=(_point_mass(theirs),),
        their_ev_loss=(np.zeros(len(SELECTIONS)),),
        value=0.5,
        model=model,
    )


class _Book:
    def __init__(self, entry: BookEntry, model: str) -> None:
        self.entry = entry
        self.model = model

    def get(self, _team) -> BookEntry:  # noqa: ANN001
        return self.entry

    def __len__(self) -> int:
        return 1


class _Leaf:
    def __init__(self, _address, arm: str, _encoder) -> None:  # noqa: ANN001
        self.arm = arm

    def describe(self) -> list[str]:
        return [f"{self.arm}.pt"]


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    return load_roster("rizabanadohido")


def _run(monkeypatch, tmp_path, roster, *extra: str, hide: bool = True):  # noqa: ANN001, ANN202
    """Plays one game a seat through `main`; returns the `play_game` calls and the rows."""
    gm = load_tool("generation_match")
    import pokeuraou.inference as inference

    books = {
        "tested.jsonl.gz": _Book(
            _entry(TESTED_OURS, TESTED_THEIRS, roster.sets, "a.pt"), "a.pt"
        ),
        "other.jsonl.gz": _Book(
            _entry(OTHER_OURS, OTHER_THEIRS, roster.sets, "b.pt"), "b.pt"
        ),
    }
    team = TournamentTeam(
        player="Opponent", place=1, country="jp", wins=1, losses=0,
        made_cut=False, phase_two=False,
        members=tuple(
            TeamMember(
                species=s.species, ability=s.ability, item=s.item,
                nature=s.nature, moves=tuple(s.moves),
            )
            for s in roster.sets
        ),
    )
    monkeypatch.setattr(gm, "find_cached_chaos", lambda *_a: None)
    monkeypatch.setattr(gm, "load_chaos", lambda *_a: None)
    monkeypatch.setattr(gm, "find_cached_standings", lambda *_a: None)
    monkeypatch.setattr(
        gm, "load_standings", lambda *_a: SimpleNamespace(pool=lambda _n: [team])
    )
    monkeypatch.setattr(
        gm, "SelectionBook", SimpleNamespace(read=lambda path: books[path.name])
    )
    monkeypatch.setattr(inference, "RemoteValue", _Leaf)

    calls: list[dict] = []

    def fake_play_game(*_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(kwargs)
        return SimpleNamespace(
            outcome=1.0,
            selection_source="",
            decisions=[],
            search_seconds=(0.0, 0.0),
            to_json=lambda **_k: {},
        )

    monkeypatch.setattr(gm, "play_game", fake_play_game)
    games = tmp_path / "games.jsonl"
    argv = [
        "generation_match.py",
        "--inference", "127.0.0.1:1",
        "--inference-arm", "a",
        "--baseline-inference-arm", "b",
        "--selection-book", "tested.jsonl.gz",
        "--games", "1",
        "--games-out", str(games),
        *(["--hide-bench"] if hide else []),
        *extra,
    ]
    if "--baseline-uniform-selection" not in extra:
        argv += ["--baseline-selection-book", "other.jsonl.gz"]
    monkeypatch.setattr(sys, "argv", argv)
    gm.main()
    rows = [json.loads(line) for line in games.read_text(encoding="utf-8").splitlines()]
    return calls, rows


def _mass(prior) -> int | None:  # noqa: ANN001
    """The one selection a point-mass prior names, or None for the uniform belief."""
    if prior is None:
        return None
    return int(np.argmax(prior.probabilities))


def _beliefs(call: dict, tested_side: int) -> tuple[int | None, int | None]:
    """(what the tested arm believes about its opponent, what the other arm believes).

    `bench_prior[s]` prices side `s`'s bench, which side `1 - s` searches over -- so the
    tested arm at side `t` reads `bench_prior[1 - t]`.
    """
    prior = call.get("bench_prior") or (None, None)
    return _mass(prior[1 - tested_side]), _mass(prior[tested_side])


def test_each_arm_prices_the_bench_from_its_own_book_in_both_seats(
    monkeypatch, tmp_path, roster  # noqa: ANN001
) -> None:
    calls, rows = _run(monkeypatch, tmp_path, roster)
    assert len(calls) == 2, "one game a seat"
    # Seat 0: the tested arm is side 0 and believes its book's column strategy for side 1;
    # the other arm (side 1) believes ITS book's row strategy for side 0.
    assert _beliefs(calls[0], 0) == (TESTED_THEIRS, OTHER_OURS), (
        "seat 0: each side's belief must be its own book's model of the opponent"
    )
    # Seat 1, swapped: the tested arm is side 1 now, so what it believes about side 0 is
    # its book's row strategy -- and still ITS book, not the one side 0 drew from.
    assert _beliefs(calls[1], 1) == (TESTED_OURS, OTHER_THEIRS), (
        "seat 1: the belief has to follow the arm, not the seat"
    )
    # Never the book the opponent actually drew from: that is its private strategy.
    for call in calls:
        prior = call["bench_prior"]
        assert prior[0] is not None and prior[1] is not None
    # And the provenance says which belief each side held, so the rating can tell.
    for row, tested_side in zip(rows, (0, 1), strict=True):
        source = row["provenance"]
        assert source["beliefs"] == ["book", "book"]
        assert agent_name(source, tested_side).endswith("/hidden-bench/belief:book")


def test_an_arm_told_to_keep_the_uniform_belief_keeps_it_in_both_seats(
    monkeypatch, tmp_path, roster  # noqa: ANN001
) -> None:
    calls, rows = _run(monkeypatch, tmp_path, roster, "--uniform-bench-belief")
    assert _beliefs(calls[0], 0) == (None, OTHER_OURS)
    assert _beliefs(calls[1], 1) == (None, OTHER_THEIRS)
    # The tested arm's name says uniform in both seats, the other arm's says book.
    assert agent_name(rows[0]["provenance"], 0).endswith("/hidden-bench")
    assert agent_name(rows[1]["provenance"], 1).endswith("/hidden-bench")
    assert agent_name(rows[0]["provenance"], 1).endswith("/belief:book")
    assert agent_name(rows[1]["provenance"], 0).endswith("/belief:book")


def test_an_arm_without_a_book_holds_the_uniform_belief_in_both_seats(
    monkeypatch, tmp_path, roster  # noqa: ANN001
) -> None:
    calls, _rows = _run(monkeypatch, tmp_path, roster, "--baseline-uniform-selection")
    # The other arm has no selection cache, so it has nothing to condition on; the tested
    # arm still reads its own. The single book draws the opponent's four for both seats,
    # which is exactly the book the other arm must NOT be reading its belief from.
    assert _beliefs(calls[0], 0) == (TESTED_THEIRS, None)
    assert _beliefs(calls[1], 1) == (TESTED_OURS, None)


def test_the_open_game_carries_no_bench_belief(monkeypatch, tmp_path, roster) -> None:  # noqa: ANN001
    """No hidden bench, nothing to believe about it -- and the name stays the open one."""
    calls, rows = _run(monkeypatch, tmp_path, roster, hide=False)
    assert all(call.get("bench_prior") is None for call in calls)
    assert all(call["sheets"] is None for call in calls)
    assert "belief" not in agent_name(rows[0]["provenance"], 0)


def test_agent_drift_counts_the_bench_prior_as_part_of_the_agent() -> None:
    """The check said `generation_match` was generation's agent while it was not.

    Generation passes `bench_prior`, so it has to be among the arguments a tool is held
    to; and with it there, `--check` has to pass on the fixed tree -- which it did not
    with `generation_match` omitting it (NEW: generation_match.py).
    """
    drift = load_tool("agent_drift")
    assert "bench_prior" in drift.shipping_args()
    found = drift.calls(drift.ROOT / "tools" / "generation_match.py")
    assert found and all("bench_prior" in kwargs for _, kwargs in found)
    assert drift.main(["--check"]) == 0


def test_old_hidden_records_read_as_the_uniform_belief() -> None:
    """Every hidden-bench match before IKA-122 played the uniform belief, so it is named so."""
    source = {
        "leaves": ["a", "b"],
        "limits": [24, 24],
        "books": ["x", "x"],
        "information": ["hidden-bench", "hidden-bench"],
    }
    assert agent_name(source, 0) == "a/w24/book:x/hidden-bench"
    # An open game has no bench to believe anything about, whatever the field says.
    source["information"] = ["open", "open"]
    source["beliefs"] = ["book", "book"]
    assert agent_name(source, 0) == "a/w24/book:x"


def test_the_rank_view_belongs_to_the_arm_in_both_seats(
    monkeypatch, tmp_path, roster  # noqa: ANN001
) -> None:
    """IKA-143: the other arm told to rank from the first completion does so in both seats,
    and only it is named for it."""
    calls, rows = _run(monkeypatch, tmp_path, roster, "--baseline-rank-first-completion")
    assert calls[0]["rank_view"] == ("heaviest", "first")
    assert calls[1]["rank_view"] == ("first", "heaviest")
    assert rows[0]["provenance"]["rankViews"] == ["heaviest", "first"]
    assert rows[1]["provenance"]["rankViews"] == ["first", "heaviest"]
    for row, tested_side in zip(rows, (0, 1), strict=True):
        assert "rankview" not in agent_name(row["provenance"], tested_side)
        assert agent_name(row["provenance"], 1 - tested_side).endswith("/rankview:first")


def test_an_ordinary_match_ranks_from_the_heaviest_and_records_nothing_new(
    monkeypatch, tmp_path, roster  # noqa: ANN001
) -> None:
    calls, rows = _run(monkeypatch, tmp_path, roster)
    assert all(call["rank_view"] == ("heaviest", "heaviest") for call in calls)
    assert all("rankViews" not in row["provenance"] for row in rows)
