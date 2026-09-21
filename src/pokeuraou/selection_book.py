"""Cached selection equilibria, one per opponent team sheet.

Self-play used to pick four of six uniformly on both sides. That is a fine way to cover
the state space and a bad way to model an opponent: the win rate across our own fifteen
sets of four ranged 43.4%-87.3%, so a uniform draw spends most of its games in selections
neither player would ever make. This module lets generation draw from the *solved* 6->4
equilibrium instead -- for both sides -- without paying the 12 seconds a solve costs per
game, by solving once per opponent team and keying the answer on their team sheet.

Solving per game would cost 7,000 x 12s ~ 23 hours. There are 394 distinct teams in the
field, so solving per team costs 394 x 12s ~ 80 minutes, once.

Four ways a cache like this leaks the opponent's private choice back into ours -- the
後出しジャンケン the whole design has to avoid -- and what stops each:

1. **The key is their six, never their four.** :func:`key_for_team` reads the team sheet,
   which is public before selection. Keying on the four they brought would make every
   cached row a best response to a decision we are not allowed to have seen.

2. **The two draws are independent.** :meth:`BookEntry.draw` takes our uniform *first*,
   before it has even sampled which spread class the opponent is, and our distribution is
   a function of the entry alone. The equilibrium row strategy already answers their whole
   mixture; best-responding to a sampled column would be 後出し with extra steps.

3. **Our strategy is never indexed by spread class.** ``solve_bayesian`` returns one row
   strategy for us and one column strategy *per class*, because they know their investment
   and we do not. ``our_strategy`` is therefore a single vector, and the only per-class
   arrays here are the opponent's. Indexing ours by ``class_index`` would hand us their
   private information; the shape of the data makes that impossible to do by accident.

4. **The bench still leaks in battle, and this does not fix it.** We select without seeing
   their four, but the search then reads their full bench from turn one, so the cells of
   the selection game come from a value function that assumes in-battle omniscience. That
   is a known, separate gap (the belief layer over their hidden two), and the numbers here
   inherit it. Stated rather than quietly ignored.

And one reporting rule: generation draws from a *softened* mixture, not from the
equilibrium, so the win rate it observes is not the equilibrium value. Two different
quantities; :class:`BookEntry` keeps both the equilibrium and the mixture actually drawn
from so a reader can never confuse them.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .priors import SampledSet
from .regulation import repo_root
from .standings import TournamentTeam

#: Exploration defaults. `epsilon` is the share of games drawn from the softened
#: distribution instead of the equilibrium; `temperature` shapes that share towards
#: selections that are nearly optimal. Both are guesses until they are measured head to
#: head (a model trained on this book against one trained on uniform selection), which is
#: the same shape of experiment the candidate-width match used.
DEFAULT_EPSILON = 0.25
#: In win-probability units: a selection that gives up this much gets weight e^-1 relative
#: to the best.
#:
#: 0.5, and it was 0.05. The smaller number was chosen for precision -- the second-best
#: selection in the first solve was 0.0026 away, inside the value function's own error, so
#: calling one of them "the" selection was false precision -- and it starves the tail.
#: Expected games per selection over all 394 solved teams, weighted by the team draw, in a
#: 24,000-game generation against `value-gen11L`'s book:
#:
#:      T       min    5th   median     max    selections under 30 games
#:      0.05    3.9    6.0     40.7    6445    34 of 90
#:      0.5    53.4   54.3     66.5    6223     0 of 90
#:
#: The most-played selection gives up 3.5% of its games and the rarest goes from 3.9 to
#: 53.4, so the equilibrium is still what is mostly generated. A selection with no games
#: is a failure this project has already had.
#:
#: `tools/generate_parallel.sh` has passed 0.5 with that argument for some time. It never
#: reached anything: generation moved to `tools/generate_queue.py`, which passes no
#: exploration flags at all, so generations 10, 11h and 11L were every one of them made at
#: 0.05 while a comment in an unused driver explained why they were not. Fixed here, where
#: all three consumers read it -- self-play, its CLI, and `book_check`'s `gen` arm, which
#: exists to imitate generation and would otherwise imitate the wrong thing.
DEFAULT_TEMPERATURE = 0.5


def selection_dir() -> Path:
    return repo_root() / "data" / "selection"


def _member_key(species: str, ability: str | None, item: str | None,
                nature: str, moves: Sequence[str]) -> str:
    return "|".join(
        (species, ability or "?", item or "-", nature, ",".join(sorted(moves)))
    )


def key_for_team(team: TournamentTeam) -> str:
    """The cache key: the opponent's six exactly as their team sheet shows them.

    Public information, available to both players before selection -- see constraint 1 in
    the module docstring. An ability the sheet did not show (the mega forme case) is keyed
    as unknown rather than filled in, because filling it in would key on something we
    cannot see and would split one team into several entries.

    Species order is kept, not sorted: two entries with the same six in a different party
    order are the same sheet, but the selection indices in a cached strategy refer to
    *this* order, so an entry may only be reused for a team that lists them the same way.
    """
    return "/".join(
        _member_key(
            m.species,
            None if m.ability_is_post_mega else m.ability,
            m.item,
            m.nature,
            m.moves,
        )
        for m in team.members
    )


def set_to_json(entry: SampledSet) -> dict[str, Any]:
    return {
        "species": entry.species,
        "ability": entry.ability,
        "item": entry.item,
        "nature": entry.nature,
        # Every stat, zeros included. The generated games' own set dump drops zeros
        # (absent means 0 everywhere it is read), but a cache that hands back a set which
        # is merely equivalent to the one solved for is a cache that cannot be diffed.
        "sp": dict(entry.sp),
        "moves": list(entry.moves),
    }


def set_from_json(data: dict[str, Any]) -> SampledSet:
    return SampledSet(
        species=data["species"],
        ability=data["ability"],
        item=data.get("item"),
        nature=data["nature"],
        sp=dict(data.get("sp", {})),
        moves=list(data["moves"]),
    )


def explore_mixture(
    equilibrium: np.ndarray,
    ev_loss: np.ndarray,
    *,
    epsilon: float = DEFAULT_EPSILON,
    temperature: float = DEFAULT_TEMPERATURE,
) -> np.ndarray:
    """``(1 - eps) * equilibrium + eps * softmax(-loss / T)``.

    The equilibrium stays the base, so what generation plays is still mostly what the
    solver recommends. The exploration share is shaped by EV loss rather than spread
    uniformly because the next solve needs values for all 90 selections and the ones worth
    accurate cells are the ones a player might actually make: the worst selection in the
    first solve gave up 45 points, and games spent there teach the value function to
    separate hopeless from hopeless.

    ``temperature=inf`` makes the exploration share uniform, which is the old behaviour of
    generation restricted to ``epsilon`` of the games; ``epsilon=0`` is the pure
    equilibrium, which collapses coverage to one selection pair per opponent and is
    offered only so the collapse can be measured.
    """
    eq = np.asarray(equilibrium, dtype=np.float64).reshape(-1)
    loss = np.asarray(ev_loss, dtype=np.float64).reshape(-1)
    if eq.shape != loss.shape:
        raise ValueError(f"{eq.shape} strategy against {loss.shape} losses")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be a probability, got {epsilon}")
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be > 0, got {temperature}; a zero temperature is the "
            "argmin, not the equilibrium -- use epsilon=0 for the pure equilibrium"
        )
    total = eq.sum()
    base = eq / total if total > 0 else np.full(eq.shape, 1.0 / eq.size)
    if epsilon == 0.0:
        return base
    if math.isinf(temperature):
        shaped = np.full(loss.shape, 1.0 / loss.size)
    else:
        scores = -np.maximum(loss, 0.0) / temperature
        scores -= scores.max()
        shaped = np.exp(scores)
        shaped /= shaped.sum()
    mixed = (1.0 - epsilon) * base + epsilon * shaped
    return mixed / mixed.sum()


def perplexity(distribution: np.ndarray) -> float:
    """``exp(entropy)``: how many selections a distribution effectively plays.

    Reported next to every solve because it is the coverage number. A book with
    perplexity 1 generates one selection pair per opponent, and the next generation's
    solve would then have 89 columns of cells the value function never saw.
    """
    p = np.asarray(distribution, dtype=np.float64).reshape(-1)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    # Normalised first, because this is also called on distributions *summed* over teams
    # -- "how wide is our play across the whole field" -- and entropy on a vector summing
    # to 394 comes out as zero, which reads as total collapse instead of as a mistake.
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


@dataclass(frozen=True, slots=True)
class SelectionDraw:
    """One selection, drawn for both sides from a cached equilibrium."""

    #: The opponent's six *with* the spreads of the sampled class. Generation must use
    #: these rather than drawing spreads again: the column strategy it drew their four
    #: from is the strategy of a player holding exactly this investment.
    foe_six: tuple[SampledSet, ...]
    our_pick: tuple[int, ...]
    foe_pick: tuple[int, ...]
    #: The equilibrium mixtures -- the policy targets, and what "the solver recommends".
    our_equilibrium: np.ndarray
    foe_equilibrium: np.ndarray
    #: The distributions actually sampled from. Different from the above by `epsilon`,
    #: which is why a generated win rate is not the equilibrium value.
    our_mixture: np.ndarray
    foe_mixture: np.ndarray
    value: float
    class_index: int


@dataclass(slots=True)
class BookEntry:
    """The solved selection game against one opponent sheet, ready to draw from."""

    key: str
    player: str
    place: int
    #: Ordered selections as party indices, shared by both sides (both bring 4 of 6).
    selections: tuple[tuple[int, ...], ...]
    #: (90,) our equilibrium mixture. One vector, not one per class -- constraint 3.
    our_strategy: np.ndarray
    our_ev_loss: np.ndarray
    #: (K,) class probabilities, and per class: their six with spreads, their column
    #: strategy, and their EV losses.
    class_weights: np.ndarray
    class_sets: tuple[tuple[SampledSet, ...], ...]
    their_strategies: tuple[np.ndarray, ...]
    their_ev_loss: tuple[np.ndarray, ...]
    value: float
    duality_gap: float = 0.0
    antisymmetry_error: float = 0.0
    seconds: float = 0.0
    model: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def our_mixture(
        self, *, epsilon: float = DEFAULT_EPSILON, temperature: float = DEFAULT_TEMPERATURE
    ) -> np.ndarray:
        # No `class_index` parameter, by design. Our distribution cannot depend on their
        # spread class because we do not know it -- see constraint 3.
        return explore_mixture(
            self.our_strategy, self.our_ev_loss, epsilon=epsilon, temperature=temperature
        )

    def their_mixture(
        self,
        class_index: int,
        *,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> np.ndarray:
        return explore_mixture(
            self.their_strategies[class_index],
            self.their_ev_loss[class_index],
            epsilon=epsilon,
            temperature=temperature,
        )

    def draw(
        self,
        rng: np.random.Generator,
        *,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> SelectionDraw:
        """Draws both sides' selections. Ours first, and independently of theirs."""
        # Constraint 2, enforced by the order of these three lines: our uniform is taken
        # before the class is sampled, so nothing downstream of the opponent's private
        # type -- not even the position in the random stream -- can reach our draw.
        ours = self.our_mixture(epsilon=epsilon, temperature=temperature)
        our_index = int(np.searchsorted(np.cumsum(ours), rng.random(), side="right"))
        our_index = min(our_index, len(self.selections) - 1)

        weights = np.asarray(self.class_weights, dtype=np.float64)
        weights = weights / weights.sum()
        class_index = int(
            np.searchsorted(np.cumsum(weights), rng.random(), side="right")
        )
        class_index = min(class_index, len(self.class_sets) - 1)

        theirs = self.their_mixture(
            class_index, epsilon=epsilon, temperature=temperature
        )
        their_index = int(
            np.searchsorted(np.cumsum(theirs), rng.random(), side="right")
        )
        their_index = min(their_index, len(self.selections) - 1)

        return SelectionDraw(
            foe_six=self.class_sets[class_index],
            our_pick=self.selections[our_index],
            foe_pick=self.selections[their_index],
            our_equilibrium=np.asarray(self.our_strategy, dtype=np.float64),
            foe_equilibrium=np.asarray(
                self.their_strategies[class_index], dtype=np.float64
            ),
            our_mixture=ours,
            foe_mixture=theirs,
            value=self.value,
            class_index=class_index,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "player": self.player,
            "place": self.place,
            "selections": [list(s) for s in self.selections],
            "ourStrategy": [float(x) for x in self.our_strategy],
            "ourEvLoss": [float(x) for x in self.our_ev_loss],
            "classWeights": [float(x) for x in self.class_weights],
            "classSets": [[set_to_json(s) for s in sets] for sets in self.class_sets],
            "theirStrategies": [
                [float(x) for x in y] for y in self.their_strategies
            ],
            "theirEvLoss": [[float(x) for x in y] for y in self.their_ev_loss],
            "value": self.value,
            "dualityGap": self.duality_gap,
            "antisymmetryError": self.antisymmetry_error,
            "seconds": self.seconds,
            "model": self.model,
            "notes": list(self.notes),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BookEntry:
        return cls(
            key=data["key"],
            player=data.get("player", ""),
            place=int(data.get("place", 0)),
            selections=tuple(tuple(int(i) for i in s) for s in data["selections"]),
            our_strategy=np.array(data["ourStrategy"], dtype=np.float64),
            our_ev_loss=np.array(data["ourEvLoss"], dtype=np.float64),
            class_weights=np.array(data["classWeights"], dtype=np.float64),
            class_sets=tuple(
                tuple(set_from_json(s) for s in sets) for sets in data["classSets"]
            ),
            their_strategies=tuple(
                np.array(y, dtype=np.float64) for y in data["theirStrategies"]
            ),
            their_ev_loss=tuple(
                np.array(y, dtype=np.float64) for y in data["theirEvLoss"]
            ),
            value=float(data["value"]),
            duality_gap=float(data.get("dualityGap", 0.0)),
            antisymmetry_error=float(data.get("antisymmetryError", 0.0)),
            seconds=float(data.get("seconds", 0.0)),
            model=data.get("model", ""),
            notes=tuple(data.get("notes", ())),
        )


@dataclass(slots=True)
class SelectionBook:
    """Solved selections keyed by team sheet, plus what they were solved with."""

    entries: dict[str, BookEntry] = field(default_factory=dict)
    #: The roster the row strategies belong to. A book solved for one of our teams says
    #: nothing about another, and loading it against the wrong roster would silently
    #: reinterpret 90 indices -- so the roster is stored and checked.
    roster: str = ""
    model: str = ""
    format_id: str = ""
    #: What the games that trained this book's leaf could see, as
    #: ``{"open": n, "hidden-bench": m}``. The book prints a win probability, and that is a
    #: prediction about whoever played the games the leaf learned from -- measured on place
    #: 109, a leaf whose pool is 83% omniscient claims 62.8% where open play returns 59.2%
    #: and the hidden-bench play the product actually ships returns 50.0%. Empty means the
    #: book predates this being recorded, not that the pool was open.
    information: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, team: TournamentTeam) -> BookEntry | None:
        return self.entries.get(key_for_team(team))

    def add(self, entry: BookEntry) -> None:
        self.entries[entry.key] = entry

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "header": {
                            "roster": self.roster,
                            "model": self.model,
                            "format": self.format_id,
                            "information": self.information,
                            "entries": len(self.entries),
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for entry in self.entries.values():
                handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")

    @classmethod
    def read(cls, path: Path) -> SelectionBook:
        book = cls()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                header = data.get("header")
                if header is not None:
                    book.roster = header.get("roster", "")
                    book.model = header.get("model", "")
                    book.format_id = header.get("format", "")
                    book.information = dict(header.get("information") or {})
                    continue
                book.add(BookEntry.from_json(data))
        return book

    def covered(self, teams: Iterable[TournamentTeam]) -> tuple[int, int]:
        """(how many of these teams the book answers, how many there are)."""
        total = 0
        hit = 0
        for team in teams:
            total += 1
            hit += int(key_for_team(team) in self.entries)
        return hit, total

    def require_roster(self, roster: str) -> None:
        """Refuses a book solved for another of our teams.

        The 90 numbers in a row strategy are indices into *a party order*, so reading a
        book against a different roster would not fail -- it would quietly recommend the
        wrong four. Checked wherever a book is loaded rather than trusted to the filename.
        """
        if self.roster and roster and self.roster != roster:
            raise ValueError(
                f"this selection book was solved for roster {self.roster!r}, not "
                f"{roster!r}; its 90 selections index that team's party order"
            )


#: The selection rules a head-to-head run can put on either seat, as "ours/theirs".
#: "book" is the equilibrium, "gen" is the equilibrium plus the exploration share that
#: generation actually plays, and "uniform" is the draw generation used before any of this.
ARMS = ("uniform/uniform", "book/uniform", "book/book", "gen/gen")


def _pick(rng: np.random.Generator, weights: np.ndarray) -> int:
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64))
    cumulative /= cumulative[-1]
    return min(
        int(np.searchsorted(cumulative, rng.random(), side="right")), len(cumulative) - 1
    )


def draw_across(
    ours: BookEntry | None,
    theirs: BookEntry | None,
    rng: np.random.Generator,
    *,
    epsilon: float = 0.0,
    temperature: float = 1.0,
) -> SelectionDraw:
    """One draw where the two sides consult DIFFERENT books.

    :meth:`BookEntry.draw` takes both sides' selections from one entry, which is right
    when both sides are the same agent and wrong the moment they are not: a book is the
    equilibrium of the selection game as one model's value function defines it, so an
    agent drawing from another model's book selects by an evaluation it does not use.

    `tools/generation_match.py` accepted `--baseline-selection-book` and then consulted a
    single book per game -- whichever arm sat at side 0 -- so the flag only ever swapped
    which book governed BOTH sides, in half the games. Two matches run to compare own
    books against a shared one returned the identical 380/848 in the seat they shared,
    which is what finally showed it; the provenance said each arm used its own the whole
    time.

    The order of the three draws is the invariant, and it is spelled out again here for
    the reason `draw_arm` gives: two sides drawn by different code paths is exactly where
    the tempting shortcut reappears. Ours is taken first and BEFORE the class, so nothing
    downstream of the opponent's private type -- not even the position in the stream --
    can reach our draw.

    The class is neither arm's to choose. It is the opponent's investment, and it comes
    from the spread prior rather than from any model: every entry of every book solved
    for this roster carries identical `class_sets`, `class_weights` and `selections`,
    which is checked here rather than trusted, because a class drawn from the wrong book
    would silently pair one side's column strategy with the other side's spreads.

    Either side may be ``None``, meaning that arm has no book and draws its four
    uniformly. That is what "uniform selection" is -- a property of the ARM, which travels
    with it into whichever seat it sits in. `generation_match`'s
    `--baseline-uniform-selection` promised "the other arm draws its four uniformly" and
    delivered it only where that arm sat at side 0, because the uniform draw replaced our
    pick and side 0 is always our roster; in the other seat the same arm took its four
    from the book's column strategy. The flag was correct in half the games.
    """
    field = ours if ours is not None else theirs
    if field is None:
        raise ValueError("at least one side must have a book to name the opponent")
    if ours is not None and theirs is not None and ours.class_sets != theirs.class_sets:
        raise ValueError(
            "the two books disagree about the opponent's spread classes, so there is no "
            "single opponent for the two sides to be playing"
        )
    count = len(field.selections)
    uniform = np.full(count, 1.0 / count)

    our_mixture = (
        uniform
        if ours is None
        else ours.our_mixture(epsilon=epsilon, temperature=temperature)
    )
    our_index = _pick(rng, our_mixture)
    class_index = _pick(rng, np.asarray(field.class_weights, dtype=np.float64))
    theirs_mixture = (
        uniform
        if theirs is None
        else theirs.their_mixture(
            class_index, epsilon=epsilon, temperature=temperature
        )
    )
    their_index = _pick(rng, theirs_mixture)
    return SelectionDraw(
        foe_six=field.class_sets[class_index],
        our_pick=field.selections[our_index],
        foe_pick=field.selections[their_index],
        our_equilibrium=(
            uniform if ours is None
            else np.asarray(ours.our_strategy, dtype=np.float64)
        ),
        foe_equilibrium=(
            uniform if theirs is None
            else np.asarray(theirs.their_strategies[class_index], dtype=np.float64)
        ),
        our_mixture=our_mixture,
        foe_mixture=theirs_mixture,
        # Whose value? The side that has a book, and ours when both do -- it is the
        # reason two books give different numbers at all, each cell value being that
        # model's estimate of the same position.
        value=field.value,
        class_index=class_index,
    )


def draw_arm(
    arm: str,
    entry: BookEntry,
    class_index: int,
    rng: np.random.Generator,
    *,
    epsilon: float = DEFAULT_EPSILON,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[int, int]:
    """(our selection index, theirs) under one named rule, ours drawn first.

    Lives here rather than in the measurement tool because it carries the same invariant
    :meth:`BookEntry.draw` does, and an invariant worth stating is worth testing: whatever
    the arm, our index is a function of the entry and the stream, never of
    ``class_index``. That argument exists only to look up the opponent's column strategy,
    which is theirs to know -- they hold the investment, we do not.

    A harness that mixes rules is exactly where the leak would reappear, because there the
    two sides are drawn by different code paths and the tempting shortcut is to pass the
    class into both.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    ours_rule, theirs_rule = arm.split("/")
    count = len(entry.selections)

    if ours_rule == "book":
        ours = _pick(rng, entry.our_strategy)
    elif ours_rule == "gen":
        ours = _pick(
            rng, entry.our_mixture(epsilon=epsilon, temperature=temperature)
        )
    else:
        ours = int(rng.integers(count))

    if theirs_rule == "book":
        theirs = _pick(rng, entry.their_strategies[class_index])
    elif theirs_rule == "gen":
        theirs = _pick(
            rng,
            entry.their_mixture(
                class_index, epsilon=epsilon, temperature=temperature
            ),
        )
    else:
        theirs = int(rng.integers(count))
    return ours, theirs


def start_book(
    path: Path,
    *,
    roster: str,
    model: str,
    format_id: str,
    information: dict[str, int] | None = None,
) -> None:
    """Writes the header if ``path`` is new, so entries can be appended one at a time.

    An 80-minute solve that only lands on disk at the end is an 80-minute solve that a
    Ctrl-C throws away. gzip members concatenate, so appending is a valid file at every
    point and :meth:`SelectionBook.read` needs no special case.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "header": {
                        "roster": roster,
                        "model": model,
                        "format": format_id,
                        "information": information or {},
                    }
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def append_entry(path: Path, entry: BookEntry) -> None:
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")


def find_cached_book(name: str) -> Path | None:
    path = selection_dir() / f"{name}.jsonl.gz"
    return path if path.exists() else None


@dataclass(frozen=True)
class BenchPrior:
    """One side's selection distribution, carried into a game to price its hidden bench.

    A plain 3-tuple would do and is the shape that gets mis-ordered: `selections` indexes
    into `species`, and swapping two of these three arguments produces weights that are
    wrong rather than an error.
    """

    #: Party-index selections, shared by both sides -- both bring 4 of 6.
    selections: tuple[tuple[int, ...], ...]
    #: A distribution over `selections` for THIS side.
    probabilities: tuple[float, ...]
    #: This side's six, in the order `selections` indexes.
    species: tuple[str, ...]

    @classmethod
    def of(
        cls,
        entry: BookEntry,
        side_index: int,
        species: Sequence[str],
        *,
        epsilon: float = DEFAULT_EPSILON,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> BenchPrior:
        """The distribution that side is actually drawn from, not the equilibrium.

        `BookEntry.draw` samples from `our_mixture` and `their_mixture`, which soften the
        equilibrium by `epsilon` and `temperature`. A belief built on the raw equilibrium
        would be wrong about the very opponent generating the game -- confidently so,
        since these supports run to two or three selections of ninety and G28 measured
        the cells they are chosen on at +-11 points. **A caller must pass the same
        epsilon and temperature it passes to `draw`**; two settings of the same knob is
        how a belief and its world come apart.

        Theirs is one mixture per spread class and we do not know the class, so they are
        averaged by `class_weights` -- the same marginalisation the selection solve makes
        when it prices a cell, and never an index into their private information.
        """
        if side_index == 0:
            probabilities = entry.our_mixture(
                epsilon=epsilon, temperature=temperature
            )
        else:
            weights = np.asarray(entry.class_weights, dtype=np.float64)
            columns = np.asarray(
                [
                    entry.their_mixture(k, epsilon=epsilon, temperature=temperature)
                    for k in range(len(entry.their_strategies))
                ]
            )
            probabilities = weights @ columns
        return cls(
            selections=tuple(tuple(s) for s in entry.selections),
            probabilities=tuple(float(p) for p in probabilities),
            species=tuple(species),
        )

    def weights(self, seen: Collection[str]) -> dict[tuple[str, ...], float]:
        return bench_weights(self.selections, self.probabilities, self.species, seen)


def bench_weights(
    selections: Sequence[Sequence[int]],
    probabilities: Sequence[float],
    species: Sequence[str],
    seen: Collection[str],
) -> dict[tuple[str, ...], float]:
    """What a side's unseen members are, given the ones already on the board.

    `hidden.completions` enumerates the ways to fill the unseen slots and takes `weights`
    to price them; its docstring has always said those should be "the opponent's selection
    equilibrium marginalised onto their back two", and until now the only caller that
    passed anything was a test. Uniform is not a neutral stand-in. The opponent brings the
    back two they judge best, and a uniform belief counts their weak pairs at the same
    weight, so the opponent inside the belief is weaker than the one across the table --
    which is the shape of the search's 13-point optimism at turn 1 under a hidden bench,
    measured against 1 point in the open game (G27/G30).

    `probabilities` is a distribution over `selections`, which are party indices into
    `species`. A selection that does not contain everything already seen is one they
    cannot have brought, so it contributes nothing; what is left is renormalised onto the
    species pairs the unseen slots could hold, which is the key `completions` wants.

    An empty result means no selection in the distribution explains the board. The caller
    gets uniform, which is what `completions` does with missing weights, and should say so
    rather than let the disagreement pass unrecorded.
    """
    from .regulation import to_id

    sheet = {to_id(name) for name in species}
    # `seen` arrives holding BOTH a revealed Pokemon's own id and its base form, because
    # the candidate filter needs both to keep a Mega from also being offered as its base.
    # Only one of the pair can be a sheet member, so intersecting picks it out -- and
    # requiring the whole set instead is what made this explain nothing in 591 of 600
    # games: Floette-Eternal is on the sheet, its base `floette` is not, and every
    # selection was rejected for lacking a Pokemon the opponent never had.
    want = {name for name in (to_id(name) for name in seen) if name in sheet}
    out: dict[tuple[str, ...], float] = {}
    for selection, probability in zip(selections, probabilities, strict=True):
        if probability <= 0.0:
            continue
        brought = [to_id(species[i]) for i in selection]
        if not want.issubset(brought):
            continue
        key = tuple(sorted(name for name in brought if name not in want))
        out[key] = out.get(key, 0.0) + float(probability)
    total = sum(out.values())
    return {key: value / total for key, value in out.items()} if total > 0 else {}


def entry_bench_weights(
    entry: BookEntry, side_index: int, species: Sequence[str], seen: Collection[str]
) -> dict[tuple[str, ...], float]:
    """`bench_weights` for one side of a solved entry, in one call.

    Delegates rather than repeating `BenchPrior.of`: which side reads one vector and
    which reads a class-weighted average is the part worth having in a single place.
    """
    return BenchPrior.of(entry, side_index, species).weights(seen)


__all__ = [
    "ARMS",
    "DEFAULT_EPSILON",
    "DEFAULT_TEMPERATURE",
    "BookEntry",
    "SelectionBook",
    "SelectionDraw",
    "append_entry",
    "BenchPrior",
    "bench_weights",
    "draw_across",
    "draw_arm",
    "entry_bench_weights",
    "explore_mixture",
    "find_cached_book",
    "key_for_team",
    "perplexity",
    "selection_dir",
    "set_from_json",
    "set_to_json",
    "start_book",
]
