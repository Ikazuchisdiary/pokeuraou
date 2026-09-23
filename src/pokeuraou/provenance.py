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

    The compiled node is hashed separately, because the Rust *source* is not what played
    the game -- the binary is, and the two come apart every time a Rust change is pulled
    and not yet rebuilt. In that window the source hash would name a fix the running code
    does not contain, which is the exact failure this function exists to prevent, dressed
    as a solution to it. A run with no binary records `null` rather than a hash, because
    "there was no Rust node" is a real and different fact from "the node was this one".
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
    _ENGINE = {
        "sources": digest.hexdigest(),
        "node": _node_digest(),
        "commit": commit,
        "dirty": dirty,
    }
    return _ENGINE


def _node_digest() -> str | None:
    """The compiled Rust node as it is on disk, or None when there is no binary."""
    from .rustnode import binary_path

    try:
        data = binary_path().read_bytes()
    except OSError:
        return None
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def provenance(
    kind: str,
    *,
    seat: str,
    leaves: tuple[str, str],
    limits: tuple[int, int],
    depths: tuple[int, int] = (1, 1),
    rankings: tuple[str, str] = ("damage", "damage"),
    solvers: tuple[str, str] = ("full", "full"),
    books: tuple[str, str] = ("uniform", "uniform"),
    information: tuple[str, str] = ("open", "open"),
    beliefs: tuple[str, str] = ("uniform", "uniform"),
    encodings: tuple[str, str] = ("new", "new"),
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

    `beliefs` is what each side's search believed the opponent's hidden bench holds:
    ``"book"`` weighted by that side's own selection cache, ``"uniform"`` every pair the
    sheet allows at one weight. It means something only under a hidden bench, and every
    hidden match recorded before the field existed (IKA-122) played the uniform one.
    """
    return {
        "kind": kind,
        "seat": seat,
        "leaves": list(leaves),
        "limits": list(limits),
        "depths": list(depths),
        "rankings": list(rankings),
        "solvers": list(solvers),
        "books": list(books),
        "information": list(information),
        "beliefs": list(beliefs),
        # Derived, never passed. It was a free-text summary with a default of "uniform",
        # and callers stopped passing it when `books` arrived per side -- so every match
        # played from a solved selection recorded `"selection": "uniform"` beside a
        # `books` field naming the two it actually drew from. Nothing reads it, which is
        # the only reason it cost nothing; a field that contradicts the game in the same
        # file is what the comment beside `books` exists to prevent.
        "selection": (
            "uniform"
            if set(books) == {"uniform"}
            else " vs ".join(books)
        ),
        # Which encoding rules each side's leaf was scored under (`EncodingRules.label`).
        # Written only when a side ran an undone fix, so every record from an ordinary run
        # stays byte for byte what it was (IKA-141).
        **({"encodings": list(encodings)} if set(encodings) != {"new"} else {}),
        **({"note": note} if note else {}),
    }


def agent_name(source: dict[str, Any], side: int) -> str:
    """A stable name for the agent that played one side of a recorded game.

    The name is the whole configuration, because that is what was played: the leaf, the
    candidate width, the search depth, the ranking and how the node was solved. Two
    records naming the same agent must mean the same agent, or a rating pools games that
    were not comparable.

    The solver is in the name even though both settings solve the same game to the same
    value. They land on different vertices of a degenerate optimum, so they play different
    mixtures, and whether that costs anything is the question -- a name that could not
    tell them apart would answer it by assumption.

    Records written before a field existed are read at its default, which is what those
    runs actually used.
    """
    leaf = (source.get("leaves") or ["?", "?"])[side]
    # `value-all.pt` and `value-all` are the same leaf. Older matches recorded the file
    # name and newer ones its stem, and the difference split one model into two agents
    # that had never played each other -- so a match run specifically to join two halves
    # of the rating graph joined nothing, and the table went on reporting both halves as
    # unrelated. Normalised here rather than at each call site, because this function is
    # what a rating is keyed on.
    if leaf.endswith(".pt"):
        leaf = leaf[: -len(".pt")]
    limit = (source.get("limits") or [0, 0])[side]
    depth = (source.get("depths") or [1, 1])[side]
    ranking = (source.get("rankings") or ["damage", "damage"])[side]
    solver = (source.get("solvers") or ["full", "full"])[side]
    book = (source.get("books") or ["uniform", "uniform"])[side]
    # An agent is its model together with ITS OWN book. What the opponent draws from is
    # the opponent's identity, not this side's.
    #
    # `generation_match` used to write `uniform-against-<their book>` here, reasoning that
    # an arm drawing uniformly against a book-drawing opponent faces a harder field than
    # one drawing against another uniform arm, so pooling the two would charge that
    # difficulty to the uniform arm as weakness. Bradley-Terry already removes it: the
    # opponent's difficulty *is* its rating, and the fit subtracts it. The split protected
    # against a bias the model does not have, and the price was total: an arm in a book
    # match can only ever be named after the book it faced, so every match played to
    # connect the book corpus to the anchored one created a new node instead of an edge.
    # Thirteen thousand games sat in a component with no zero in it and no way to get one.
    #
    # Same class of defect as the `.pt` normalisation above, and the same repair: fix it
    # where the rating is keyed, so the games already recorded are joined without replay.
    if book.startswith("uniform-against-"):
        book = "uniform"
    name = f"{leaf}/w{limit}"
    if depth != 1:
        name += f"/d{depth}"
    if ranking != "damage":
        name += f"/{ranking}"
    if solver != "full":
        name += f"/{solver}"
    # The selection rule, because drawing the four of six from a solved equilibrium is
    # worth +22.1 points against drawing them uniformly -- a bigger difference than any
    # two models on this scale. An agent that selects from a book is not the same agent.
    if book != "uniform":
        name += f"/book:{book}"
    # What the search was allowed to see. An agent solving the opponent's true four is
    # not a weaker or stronger version of one that solves over the six it could be -- it
    # is playing a different game, and the advice it produces moved in 90% of openings.
    # Every record written before this field existed was omniscient, which is what the
    # default says.
    information = (source.get("information") or ["open", "open"])[side]
    if information != "open":
        name += f"/{information}"
        # What it believed that bench holds. The uniform belief and the one weighted by
        # the agent's own book are two agents -- IKA-5 measured +12.9% srch-act between
        # them at turn 1 -- and until IKA-122 every hidden match played the uniform one
        # under the name generation's agent would get. Absent means uniform, because
        # that is what those records played; the name they already carry is kept.
        belief = (source.get("beliefs") or ["uniform", "uniform"])[side]
        if belief != "uniform":
            name += f"/belief:{belief}"
    # A leaf scored with a fix undone is another agent, for as long as such matches exist
    # (IKA-141). A record without the field ran its own tree's rules and keeps its old key
    # -- which for anything before 9/23 means revision 1, unnamed.
    encoding = (source.get("encodings") or ["new", "new"])[side]
    if encoding != "new":
        name += f"/enc:{encoding}"
    return name


def write_game(
    handle: Any,  # noqa: ANN401 - any text file object
    record: Any,  # noqa: ANN401 - GameRecord, kept loose to avoid a circular import
    *,
    objective: str,
    search_limit: int | tuple[int, int],
    source: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """One JSONL line, in the same shape self-play writes, plus its provenance.

    `extra` carries facts about the RUN rather than about the agents -- the game's index
    above all. A head-to-head plays index `i` in both seats from one seed, so the two are
    the same matchup with the arms swapped and the difference is paired; without the index
    written down, nothing downstream can find the pair, and every interval this project has
    reported on a queued match treated 2N paired games as 2N independent ones. Measured on
    `gen10-vs-gen9`: 61% of the 848 pairs scored exactly 0.5, and the honest interval is
    +-2.10 against the +-2.38 reported.
    """
    payload = record.to_json(objective=objective, search_limit=search_limit)
    payload["provenance"] = source
    if extra:
        payload.update(extra)
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
