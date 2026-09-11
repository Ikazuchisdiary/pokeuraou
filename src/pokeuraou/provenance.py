"""Recording games that were played for some other reason, without lying about them.

The head-to-head tools -- generation match, width match, cycle match, book check -- play
thousands of real games and keep only the win rate. Those games carry the same label
self-play does, the actual outcome, so throwing them away is throwing away training data
that has already been paid for: one afternoon's experiments came to about four thousand
games, seven tenths of a generation.

They are not interchangeable with self-play, and the difference has to travel with them.
A value function is always conditional on the policy that produced the trajectory, and in
these games the two sides are *deliberately* mismatched -- a different leaf, a different
candidate width, a fixed mirror selection, an equilibrium selection instead of a uniform
one. Self-play says "the value under equal play at this strength"; a width match says
"the value when side 0 searched 24 wide and side 1 searched 16". Mixing them without
saying so blurs the conditioning in a way nothing downstream could detect.

So every recorded game carries a `provenance` block naming the tool, the leaf and the
search width *per side*, and the seat label. A dataset builder can then include, exclude
or reweight them deliberately, and the effect of doing so is measurable the only way
anything here is measured: train with and without, and play the two against each other.

Whether it helps is not assumed. This module only makes the question askable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Recorded games whose `provenance.kind` is this came from ordinary self-play, where both
#: sides are the same agent. Absent provenance means the same thing -- every game written
#: before this existed was self-play.
SELF_PLAY = "selfplay"


def provenance(
    kind: str,
    *,
    seat: str,
    leaves: tuple[str, str],
    limits: tuple[int, int],
    selection: str = "uniform",
    note: str = "",
) -> dict[str, Any]:
    """What produced this game, per side, in the order the sides appear in the record.

    ``leaves`` and ``limits`` are per side rather than per agent on purpose: a reader of
    the dataset has a position with side 0 and side 1 in it, and needs to know which of
    them was searched how. Translating "the new generation won this seat" into "side 1
    held it" is exactly the step that gets done wrong.
    """
    return {
        "kind": kind,
        "seat": seat,
        "leaves": list(leaves),
        "limits": list(limits),
        "selection": selection,
        **({"note": note} if note else {}),
    }


def write_game(
    handle: Any,  # noqa: ANN401 - any text file object
    record: Any,  # noqa: ANN401 - GameRecord, kept loose to avoid a circular import
    *,
    objective: str,
    search_limit: int | tuple[int, int],
    source: dict[str, Any],
) -> None:
    """One JSONL line, in the same shape self-play writes, plus its provenance."""
    payload = record.to_json(objective=objective, search_limit=search_limit)
    payload["provenance"] = source
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def open_games(path: Path | None) -> Any:  # noqa: ANN401
    """The output file, or a sink that drops everything when no path was given."""
    if path is None:
        class _Sink:
            def write(self, _text: str) -> None: ...
            def flush(self) -> None: ...
            def close(self) -> None: ...
            def __enter__(self) -> Any: return self  # noqa: ANN401
            def __exit__(self, *_exc: object) -> None: ...

        return _Sink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


__all__ = ["SELF_PLAY", "open_games", "provenance", "write_game"]
