"""IKA-458 stage 3.5: a game's preparation asked of the port in one request (`prepGame`,
`rust/src/prep.rs`).

`poolplay.generate_pool` drew the pair and its seats from `default_rng([seed, index])`, read the
pair's selection solve, drew both sides' four from it, built the two bench priors and the turn-1
position with the leads' switch-ins applied. The port does all of it from the pool's number, the
seed and the game's index. The pool crosses once (`poolDefine`); the solves are read there from
the store directory, and a pair with no file is solved here (it needs the leaf) and asked again.
The game's sheets and priors are defined in the port under a number for `hiddenStep`
(`hiddenstep.Game`), so they never cross.

Where this road does not go -- positions not held, no store, no queue index, a generator that is
not numpy's PCG64, a selection game with more than one class of the opponent -- `prepare` returns
None and `generate_pool` does the preparation itself. A fault the port answers is
`portlp.PortFault`.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import hiddenstep, port, portlp, timing
from .position import Position

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator, Sequence

    from .pool import Pool
    from .poolplay import SolvedSelections
    from .regulation import Regulation

_POOL_NUMBERS = itertools.count(1)
#: The pools defined to the port, by the pool's sha256 (a process asks for the pool it plays).
_DEFINED: dict[str, PoolDefinition] = {}


@dataclass(slots=True)
class PoolDefinition:
    """A pool's teams and pairs, numbered for the port (`poolDefine`)."""

    number: int
    line: bytes


def define(pool: Pool) -> PoolDefinition:
    found = _DEFINED.get(pool.sha256)
    if found is not None:
        return found
    number = next(_POOL_NUMBERS)
    body = {
        "kind": "poolDefine",
        "id": number,
        "teams": [
            {
                "id": team.id,
                "sets": [
                    {
                        "species": s.species, "ability": s.ability, "item": s.item,
                        "nature": s.nature, "sp": dict(s.sp), "moves": list(s.moves),
                    }
                    for s in team.sets
                ],
            }
            for team in pool.teams
        ],
        "pairs": [[a, b] for a, b in pool.pairs],
    }
    found = PoolDefinition(number, (json.dumps(body, ensure_ascii=False) + "\n").encode("utf-8"))
    _DEFINED[pool.sha256] = found
    return found


class LazyPriors:
    """The two bench priors of a prepared game, built only if a road that reads them asks.

    The port holds the game's priors (`hiddenStep` reads them there); Python needs them only for
    the road that builds the completions itself, which is the fallback of a fault.
    """

    def __init__(self, build: Any) -> None:  # noqa: ANN401 - a callable returning the pair
        self._build = build
        self._priors: tuple[Any, Any] | None = None

    def _get(self) -> tuple[Any, Any]:
        if self._priors is None:
            self._priors = self._build()
        return self._priors

    def __getitem__(self, side: int) -> Any:  # noqa: ANN401
        return self._get()[side]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._get())

    def __len__(self) -> int:
        return 2


@dataclass(slots=True)
class Prepared:
    """What the port made of a game's start."""

    pair: int
    teams: tuple[int, int]
    mirror: bool
    picks: tuple[tuple[int, ...], tuple[int, ...]]
    position: Position
    own_policy: list[float]
    foe_policy: list[float]
    own_mixture: list[float]
    foe_mixture: list[float]
    value: float
    game: hiddenstep.Game | None
    priors: LazyPriors | None
    rng_state: dict[str, Any] = field(repr=False, default_factory=dict)

    def apply(self, rng: np.random.Generator) -> None:
        """Leaves `rng` where the preparation left numpy's own generator."""
        rng.bit_generator.state = self.rng_state


def usable() -> bool:
    return hiddenstep.usable()


def prepare(  # noqa: PLR0913 - the request's pieces
    reg: Regulation,
    pool: Pool,
    solver: SolvedSelections,
    *,
    seed: int,
    index: int,
    pairs: Sequence[tuple[int, int]] | None,
    epsilon: float,
    temperature: float,
) -> Prepared | None:
    """The game `index` of the run, prepared by the port, or None where the caller does it."""
    if not usable() or solver.store is None or not solver.memo:
        return None
    if seed < 0 or index < 0:
        return None
    defined = define(pool)
    number = next(hiddenstep._GAME_NUMBERS)  # noqa: SLF001
    request: dict[str, Any] = {
        "kind": "prepGame",
        "pool": defined.number,
        "seed": int(seed),
        "index": int(index),
        "pairs": None if pairs is None else [[a, b] for a, b in pairs],
        "store": str(solver.store),
        "tag": solver.tag,
        "epsilon": portlp._b64(np.array([epsilon], dtype=np.float64)),  # noqa: SLF001
        "temperature": portlp._b64(np.array([temperature], dtype=np.float64)),  # noqa: SLF001
        "hideBench": True,
        "game": number,
    }

    def call(node: Any) -> dict[str, Any]:  # noqa: ANN401
        return node.prep_game(request, defined)

    answer = port.ask(reg, call)
    portlp.COUNTS["crossings"] += 1
    if "unsolved" in answer:
        # The pair has no file yet: solved here, with the leaf, which writes it for the port.
        lo, hi = (int(v) for v in answer["unsolved"])
        solver.canonical(lo, hi)
        answer = port.ask(reg, call)
        portlp.COUNTS["crossings"] += 1
        if "unsolved" in answer:
            raise RuntimeError(f"the pair {lo}-{hi} was solved and written but the port found no file")
    if "stop" in answer:
        raise ValueError(str(answer["stop"]))
    if "fault" in answer:
        raise portlp.PortFault("prepGame", str(answer["fault"]))
    portlp.COUNTS["prepared"] += 1
    a, b = (int(t) for t in answer["teams"])
    key = (min(a, b), max(a, b))
    # What `SolvedSelections.canonical` counted: a solve made here is reused after, a file is loaded once.
    if answer["read"] and key not in solver.solved_pairs:
        solver.loaded += 1
    else:
        solver.reused += 1
    own_mixture = np.frombuffer(_b64d(answer["ownMixture"]), dtype="<f8")
    foe_mixture = np.frombuffer(_b64d(answer["foeMixture"]), dtype="<f8")
    state = answer["rng"]
    picks = tuple(tuple(int(i) for i in p) for p in answer["picks"])
    from .selection_book import BenchPrior

    def build_priors() -> tuple[Any, Any]:
        entry = solver.entry(a, b)
        species = ([s.species for s in pool.teams[a].sets], [s.species for s in pool.teams[b].sets])
        return (
            BenchPrior.of(entry, 0, species[0], epsilon=epsilon, temperature=temperature),
            BenchPrior.of(entry, 1, species[1], epsilon=epsilon, temperature=temperature),
        )

    with timing.stage("prep.read"):
        position = Position.from_json(answer["position"])
    return Prepared(
        pair=int(answer["pair"]),
        teams=(a, b),
        mirror=bool(answer["mirror"]),
        picks=(picks[0], picks[1]),  # type: ignore[arg-type]
        position=position,
        own_policy=_floats(answer["ownPolicy"]),
        foe_policy=_floats(answer["foePolicy"]),
        own_mixture=own_mixture.tolist(),
        foe_mixture=foe_mixture.tolist(),
        value=float(answer["value"]),
        game=hiddenstep.Game(number, b""),
        priors=LazyPriors(build_priors),
        rng_state={
            "bit_generator": "PCG64",
            "state": {"state": int(state["state"], 16), "inc": int(state["inc"], 16)},
            "has_uint32": int(bool(state["hasUint32"])),
            "uinteger": int(state["uinteger"]),
        },
    )


def _b64d(text: str) -> bytes:
    import base64

    return base64.b64decode(text)


def _floats(text: str) -> list[float]:
    return np.frombuffer(_b64d(text), dtype="<f8").tolist()
