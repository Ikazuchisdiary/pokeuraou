"""IKA-458 stage 3.4: what each side has shown and the positions its unseen slots could hold,
asked of the port in one request (`hiddenStep`, `rust/src/hiddenbench.rs`) -- off by default.

`play_game` carried each side's seen identities (`hidden.seen_identities`), turned them into
this position's seen slots, priced each way of filling the unseen slots with the selection prior
(`selfplay._bench_weights`) and built each completed position (`hidden.completions`) once a
step. With `portlp.HIDDEN` on the port does all of it. A completed position is kept in the port
under a number (`rustnode.HeldPosition`) and Python names it in later requests; it is written
out only if Python reads it (a fallback road does).

Where this road does not go -- a binary that predates it, positions not held, a side that
believes the dex's base species (`dex_base_belief`), a fault the port answers -- `step` returns
None and `play_game` builds them as it did.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import port, portlp, rustnode, timing
from .hidden import Completion

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from .regulation import Regulation

_GAME_NUMBERS = itertools.count(1)
#: The port forgets its games past this many (`hiddenbench.rs` `GAMES_MAX`); this end keeps the
#: same account so a game it thinks defined is one the port holds.
GAMES_MAX = 16


@dataclass(slots=True)
class Game:
    """A game's sheets and selection priors, numbered for the port (`hiddenGame`)."""

    number: int
    line: bytes

    @classmethod
    def of(cls, sheets: Sequence[Sequence[Any]], bench_prior: Sequence[Any]) -> Game:  # noqa: ANN401
        number = next(_GAME_NUMBERS)
        body = {
            "kind": "hiddenGame",
            "id": number,
            "sheets": [
                [
                    {
                        "species": s.species, "ability": s.ability, "item": s.item,
                        "nature": s.nature, "sp": dict(s.sp), "moves": list(s.moves),
                    }
                    for s in sheet
                ]
                for sheet in sheets
            ],
            "priors": [
                None if p is None else {
                    "selections": [list(s) for s in p.selections],
                    "probabilities": portlp._b64(np.asarray(p.probabilities, dtype=np.float64)),  # noqa: SLF001
                    "species": list(p.species),
                }
                for p in bench_prior
            ],
        }
        return cls(number, (json.dumps(body, ensure_ascii=False) + "\n").encode("utf-8"))


@dataclass(slots=True)
class Step:
    """The identities and slots a side has shown now, and each side's completions (None when
    a side's sheet cannot explain the board: `error` says so, as `completions`' ValueError did)."""

    seen: list[frozenset[str]]
    shown: list[frozenset[int]]
    recorded: list[list[str]]
    spreads: dict[int, list[Completion]] | None
    error: str | None
    notes: list[str]


def usable() -> bool:
    return portlp.HIDDEN[0] and rustnode._HOLD[0]  # noqa: SLF001


def step(  # noqa: PLR0913 - the step's pieces
    reg: Regulation,
    pos: Any,  # noqa: ANN401 - a Position
    game: Game,
    seen: Sequence[frozenset[str]],
    leads: Sequence[frozenset[str] | None],
) -> Step | None:
    """The step at `pos`, or None where the port does not do it (the caller then does)."""
    if not usable():
        return None
    request: dict[str, Any] = {
        "kind": "hiddenStep",
        "game": game.number,
        "position": rustnode._position(pos),  # noqa: SLF001
        "seen": [sorted(s) for s in seen],
        "leads": [None if lead is None else sorted(lead) for lead in leads],
    }

    def call(node: Any) -> dict[str, Any]:  # noqa: ANN401
        answer = node.hidden_step(request, game)
        if "fault" in answer:
            return answer
        # The positions the port kept: held here by number, defined to it already.
        for side in (0, 1):
            for item in (answer["sides"][side] or {}).get("items", ()):
                if item["held"] is not None:
                    key = int(item["held"])
                    held = rustnode.HeldPosition(key, False)
                    held.node = node
                    held.stored = rustnode._Stored(key, held)  # noqa: SLF001
                    node._known.add(key)  # noqa: SLF001
                    item["position"] = held
        return answer

    answer = port.ask(reg, call)
    portlp.COUNTS["crossings"] += 1
    if "fault" in answer:
        portlp.faulted("hiddenStep", answer["fault"])
        return None
    portlp.COUNTS["hidden"] += 1
    seen_now = [frozenset(ids) for ids in answer["seen"]]
    shown = [frozenset(slots) for slots in answer["shown"]]
    recorded = [list(ids) for ids in answer["seen"]]
    spreads: dict[int, list[Completion]] = {}
    error = None
    for side in (0, 1):
        got = answer["sides"][side]
        if got is None:
            continue
        if "error" in got:
            error = str(got["error"])
            spreads = {}
            break
        items = []
        for item in got["items"]:
            if item["exact"]:
                items.append(Completion(pos, (), (), 1.0, exact=True))
            else:
                items.append(Completion(
                    item["position"], tuple(item["species"]), tuple(item["slots"]),
                    float(item["weight"]), pk=tuple(tuple(p) for p in item["pk"]),
                ))
        spreads[side] = items
    timing.count("hidden.completions", sum(len(v) for v in spreads.values()))
    return Step(seen_now, shown, recorded, None if error else spreads, error, list(answer["notes"]))
