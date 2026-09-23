"""The key a rating is fitted on.

`agent_name` decides which recorded games are the same agent. Get it wrong in one
direction and the fit pools things that are five points apart; get it wrong in the other
and it splits one agent into two that have never played, leaving a component of the graph
with no anchor in it and an interval that says +-200 rather than "undefined".

Both mistakes have happened: `value-all.pt` against `value-all`, and a uniform arm named
after the book its OPPONENT drew from. Neither was visible in any single match -- they
only showed up as disconnected groups in the fit -- so they get a test.
"""

from __future__ import annotations

from pokeuraou.provenance import agent_name, provenance


def source(**over: object) -> dict[str, object]:
    base = provenance(
        "generation-match",
        seat="a = side 0",
        leaves=("value-a", "value-b"),
        limits=(24, 24),
        information=("open", "open"),
    )
    base.update(over)
    return base


def test_model_and_width() -> None:
    at = source()
    assert agent_name(at, 0) == "value-a/w24"
    assert agent_name(at, 1) == "value-b/w24"


def test_defaults_are_left_out_of_the_name() -> None:
    """Depth 1, damage ordering, the full solver, uniform selection and an open bench are
    what every record written before those fields existed actually used, so a name that
    spelled them would not match the older rows they belong with."""
    assert agent_name(source(), 0) == "value-a/w24"
    assert agent_name({"leaves": ["value-a", "value-b"], "limits": [24, 24]}, 0) == (
        "value-a/w24"
    )


def test_non_default_settings_are_in_the_name() -> None:
    at = source(
        depths=[2, 1],
        rankings=["leaf", "damage"],
        solvers=["sparse", "full"],
        information=["hidden-bench", "open"],
    )
    assert agent_name(at, 0) == "value-a/w24/d2/leaf/sparse/hidden-bench"
    assert agent_name(at, 1) == "value-b/w24"


def test_file_name_and_stem_are_one_agent() -> None:
    """Older matches recorded `value-all.pt` and newer ones `value-all`. The split meant a
    match run specifically to join two halves of the graph joined nothing."""
    assert agent_name(source(leaves=["value-a.pt", "value-b"]), 0) == "value-a/w24"


def test_own_book_is_in_the_name() -> None:
    """Drawing the four from a solved equilibrium is worth more than any two models on
    this scale, so it is a different agent."""
    at = source(books=["riza-value-gen9", "uniform"])
    assert agent_name(at, 0) == "value-a/w24/book:riza-value-gen9"
    assert agent_name(at, 1) == "value-b/w24"


def test_the_opponents_book_is_not_in_the_name() -> None:
    """An agent is its model together with ITS OWN book.

    `generation_match` wrote `uniform-against-<their book>` for a uniform arm whose
    opponent used one. That is not a different way of playing -- the arm draws uniformly
    either way -- and Bradley-Terry already charges the opponent's difficulty to the
    opponent. Keeping it apart put every book match in a component with no anchor, so an
    arm could never be compared to hp-share however many games it played.
    """
    faced_a_book = source(books=["uniform-against-riza-value-all", "riza-value-all"])
    plain = source(books=["uniform", "uniform"])
    assert agent_name(faced_a_book, 0) == agent_name(plain, 0) == "value-a/w24"
    # The arm that actually used the book keeps it.
    assert agent_name(faced_a_book, 1) == "value-b/w24/book:riza-value-all"


def test_an_undone_fix_is_another_agent_and_only_then_recorded() -> None:
    """IKA-141: a leaf scored with a fix undone is named for it; an ordinary run is not
    touched -- no `encodings` key at all, so its records stay byte for byte as they were."""
    assert "encodings" not in source()
    at = provenance(
        "generation-match",
        seat="a = side 0",
        leaves=("value-a", "value-a"),
        limits=(12, 12),
        information=("open", "open"),
        encodings=("new", "old-can-mega+old-patch"),
    )
    assert at["encodings"] == ["new", "old-can-mega+old-patch"]
    assert agent_name(at, 0) == "value-a/w12"
    assert agent_name(at, 1) == "value-a/w12/enc:old-can-mega+old-patch"


def test_the_old_rank_view_is_another_agent_and_only_then_recorded() -> None:
    """IKA-143: an arm ranking its menu from the first completion instead of the heaviest
    is named for it; an ordinary run writes no `rankViews` key."""
    assert "rankViews" not in source()
    at = provenance(
        "generation-match",
        seat="a = side 0",
        leaves=("value-a", "value-a"),
        limits=(12, 12),
        information=("open", "open"),
        rank_views=("first", "heaviest"),
    )
    assert at["rankViews"] == ["first", "heaviest"]
    assert agent_name(at, 0) == "value-a/w12/rankview:first"
    assert agent_name(at, 1) == "value-a/w12"
