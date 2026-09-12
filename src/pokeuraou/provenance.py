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

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .regulation import repo_root

#: Recorded games whose `provenance.kind` is this came from ordinary self-play, where both
#: sides are the same agent. Absent provenance means the same thing -- every game written
#: before this existed was self-play.
SELF_PLAY = "selfplay"


#: Computed once per process: it reads a few dozen files and never changes while running.
_ENGINE: dict[str, Any] | None = None


def engine_fingerprint() -> dict[str, Any]:
    """Which engine produced this game, so a later fix can be dated against the data.

    Generations 2, 3 and 4 were all generated with Hyper Beam costing nothing -- no
    recharge turn, 569 of them in one worker's 500 games -- and the records say nothing
    about it. There is no field to filter on, so the only way to know which pool predates
    the fix is to remember. At a few tens of thousands of games that is merely unpleasant;
    at the millions this is heading for, a defect found late means either discarding
    everything or trusting a memory.

    Two identifiers, because they fail differently. The commit is what a person wants --
    it names the fixes that were in -- and it is a lie the moment the tree is dirty, which
    is most of the time during a day's work. The source hash cannot lie: two games with
    the same hash came from the same code, whatever anyone remembers.

    The hash covers every Python source and every Rust source, which is deliberately more
    than "the rules". A docstring edit changes it and that is a false alarm; the opposite
    error is what cost three generations. Over-broad is recoverable by looking, and
    under-broad is not recoverable at all.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    root = repo_root()
    digest = hashlib.blake2b(digest_size=8)
    for pattern in ("src/pokeuraou/*.py", "rust/src/*.rs"):
        for path in sorted(root.glob(pattern)):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    commit, dirty = "", False
    try:
        commit = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=root, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(  # noqa: S603
                ["git", "status", "--porcelain"],  # noqa: S607
                cwd=root, capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    _ENGINE = {"sources": digest.hexdigest(), "commit": commit, "dirty": dirty}
    return _ENGINE


def provenance(
    kind: str,
    *,
    seat: str,
    leaves: tuple[str, str],
    limits: tuple[int, int],
    depths: tuple[int, int] = (1, 1),
    rankings: tuple[str, str] = ("damage", "damage"),
    selection: str = "uniform",
    note: str = "",
) -> dict[str, Any]:
    """What produced this game, per side, in the order the sides appear in the record.

    Everything that distinguishes the two agents is per side and structured, because a
    rating fitted across matches has to know *which agent* won a game, and an agent here is
    not a model -- it is a model together with the search that ran it. Width 48 beat width
    24 with the same model by +5.5, so a record that says only which model played says
    almost nothing.

    They are per side rather than per agent for the same reason they always were: a reader
    has a position with side 0 and side 1 in it. Translating "the new generation won this
    seat" into "side 1 held it" is exactly the step that gets done wrong.

    `depths` and `rankings` were once carried in the free-text `seat` label and in `note`
    respectively, which was three places for one kind of fact. A match whose two agents
    differed only in the candidate ranking recorded identical `leaves` and identical
    `limits`, and the only thing separating them was a substring of a human-readable
    string -- fine to read, impossible to fit a rating from.
    """
    return {
        "kind": kind,
        "seat": seat,
        "leaves": list(leaves),
        "limits": list(limits),
        "depths": list(depths),
        "rankings": list(rankings),
        "selection": selection,
        **({"note": note} if note else {}),
    }


def agent_name(source: dict[str, Any], side: int) -> str:
    """A stable name for the agent that played one side of a recorded game.

    The name is the whole configuration, because that is what was played: the leaf, the
    candidate width, the search depth and the ranking. Two records naming the same agent
    must mean the same agent, or a rating pools games that were not comparable.

    Records written before a field existed are read at its default, which is what those
    runs actually used.
    """
    leaf = (source.get("leaves") or ["?", "?"])[side]
    limit = (source.get("limits") or [0, 0])[side]
    depth = (source.get("depths") or [1, 1])[side]
    ranking = (source.get("rankings") or ["damage", "damage"])[side]
    name = f"{leaf}/w{limit}"
    if depth != 1:
        name += f"/d{depth}"
    if ranking != "damage":
        name += f"/{ranking}"
    return name


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


__all__ = [
    "SELF_PLAY",
    "agent_name",
    "engine_fingerprint",
    "open_games",
    "provenance",
    "write_game",
]
