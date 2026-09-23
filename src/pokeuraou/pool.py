"""A pool of teams that both seats are drawn from, and how a pair of them is drawn (IKA-81).

M-B generation fixed our six at seat 0 (`--roster`) and drew the opponent from a field.
M-C has no own side (IKA-77, decided 9/23): both seats come from the same pool, so the
value function learns "field against field" rather than "one roster against the field".

**The pool.** `data/pool/<id>.json`, the file `tools/fetch_pastes.py` writes: 65 teams from
the Match Up Web sheet with their SP (IKA-78, and IKA-138 supplied the one missing nature,
so 65 of 65 are kept). Each team passes the checks a roster file does
(`teams.roster_from_data`). The file is fetched and not committed, so the pool records
its own sha256 for a game to cite.

**The pair.** A game draws one *unordered* pair uniformly from the ``n(n+1)/2`` pairs with
repetition -- 2,145 for 65 teams, 65 of them mirrors -- and then which team sits at seat 0
by a fair coin. So:

- the mirror is not a special case any more (`--mirror-share` is an M-B device): it is
  65/2,145 = 3.03% of the games, the share the issue names;
- each team is at seat 0 exactly as often as at seat 1, over the enumeration and not only
  on average, which is what makes the seat term R measurable straight off the pool;
- the draw takes the same two numbers from the generator whatever the pair is (the coin is
  drawn for a mirror too and changes nothing there), so what a game does after the pair
  never depends on which pair it was.

Uniform over the pairs, not over ordered pairs: two independent uniform draws would make
the mirror 1/65 = 1.54%. The issue says 65/2,145, and a pair is the unit the selection
solve is shared over (one solve answers both seats).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .regulation import Regulation, load_regulation, repo_root
from .teams import Roster, TeamError, roster_from_data


def pool_dir() -> Path:
    return repo_root() / "data" / "pool"


@dataclass(frozen=True)
class Pool:
    """The teams both seats are drawn from, in the file's order."""

    id: str
    reg: Regulation
    teams: tuple[Roster, ...]
    path: str
    sha256: str
    #: The file's own statement of what it is ("prep list, not field"), kept so a run
    #: prints it rather than letting the pool pass for a metagame.
    character: str = ""

    def __len__(self) -> int:
        return len(self.teams)

    @cached_property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return unordered_pairs(len(self.teams))

    def summary(self) -> str:
        n = len(self.teams)
        count = len(self.pairs)
        return (
            f"pool {self.id}: {n} teams, {count} pairs of which {n} mirrors "
            f"({n / count * 100:.2f}%), sha256 {self.sha256[:12]}"
        )


def unordered_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """Every pair ``(a, b)`` with ``a <= b``, mirrors included, in a fixed order."""
    return tuple((a, b) for a in range(n) for b in range(a, n))


def draw_pair(
    rng: np.random.Generator, pairs: Sequence[tuple[int, int]]
) -> tuple[int, int, int]:
    """``(pair index, seat-0 team, seat-1 team)``: a uniform pair, then a fair coin for seats.

    Always two draws from ``rng``, mirror or not -- see the module docstring.
    """
    k = int(rng.integers(len(pairs)))
    swap = bool(rng.random() < 0.5)
    a, b = pairs[k]
    return (k, b, a) if swap else (k, a, b)


def find_pool(name: str | Path) -> Path:
    path = Path(name)
    if path.exists():
        return path
    candidate = pool_dir() / f"{name}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"no pool at {name} or {candidate}; fetch it with tools/fetch_pastes.py"
    )


def load_pool(name: str | Path, reg: Regulation | None = None) -> Pool:
    """Reads a pool file and checks every team as a roster.

    A team that fails stops the load with its id: a pool with a silently dropped team is a
    different pool from the one its sha256 names.
    """
    path = find_pool(name)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    format_id = str(data["regulation"])
    if reg is None:
        reg = load_regulation(format_id)
    elif reg.meta.format_id != format_id:
        raise ValueError(
            f"the pool is {format_id} but the regulation given is {reg.meta.format_id}"
        )
    checked = (data.get("validatedAgainst") or {}).get("formatId")
    if checked is not None and checked != format_id:
        raise ValueError(f"the pool says {format_id} but was validated against {checked}")
    teams: list[Roster] = []
    for index, team in enumerate(data["teams"]):
        default_id = str(team.get("id", f"{data.get('id', path.stem)}[{index}]"))
        try:
            teams.append(
                roster_from_data(team, default_id=default_id, where=default_id, reg=reg)
            )
        except (TeamError, KeyError, ValueError) as exc:
            raise TeamError(f"pool {path.name}, team {index} ({default_id}): {exc}") from exc
    if not teams:
        raise TeamError(f"pool {path.name} has no teams")
    ids = [t.id for t in teams]
    if len(set(ids)) != len(ids):
        raise TeamError(f"pool {path.name}: team ids repeat")
    return Pool(
        id=str(data.get("id", path.stem)),
        reg=reg,
        teams=tuple(teams),
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        character=str(data.get("character", "")),
    )


__all__ = [
    "Pool",
    "draw_pair",
    "find_pool",
    "load_pool",
    "pool_dir",
    "unordered_pairs",
]
