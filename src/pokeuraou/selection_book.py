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

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .priors import SampledSet

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

    def weights(
        self, seen: Collection[str], leads: Collection[str] | None = None
    ) -> dict[tuple[str, ...], float]:
        return bench_weights(
            self.selections, self.probabilities, self.species, seen, leads=leads
        )


def bench_weights(
    selections: Sequence[Sequence[int]],
    probabilities: Sequence[float],
    species: Sequence[str],
    seen: Collection[str],
    *,
    leads: Collection[str] | None = None,
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

    `leads` is the pair that led turn 1, spelled like `seen` (a Pokemon's own id and its
    base form both welcome; the sheet member is picked out). Given, only selections that
    *led* that pair count. A selection is ordered -- its first two lead, which is why
    there are ninety and not fifteen -- and containing the leads is not the same as
    leading them: at turn 1 the seen species are exactly the two leads, so without this a
    selection that planned one of them for the back, and led something that is not on
    the field, kept its back pair in the belief. Measured on the gen11L book that was a
    total variation of about 0.19 under the generation mixture (IKA-118). The lead pair is
    public from turn 1 and never changes, so it holds for the whole game. Omitted (a game
    resumed mid-way has no turn 1 to read it from), only `seen` conditions.

    An empty result means no selection in the distribution explains the board. The caller
    gets uniform, which is what `completions` does with missing weights, and should say so
    rather than let the disagreement pass unrecorded.
    """
    from .regulation import to_id

    sheet = {to_id(name) for name in species}
    led: frozenset[str] | None = None
    if leads is not None:
        led = frozenset(name for name in (to_id(name) for name in leads) if name in sheet)
        if len(led) != 2:
            raise ValueError(
                f"leads {sorted(leads)!r} name {len(led)} sheet members, not the 2 that "
                f"led; the sheet is {sorted(sheet)!r}"
            )
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
        if led is not None and frozenset(brought[:2]) != led:
            continue
        key = tuple(sorted(name for name in brought if name not in want))
        out[key] = out.get(key, 0.0) + float(probability)
    total = sum(out.values())
    return {key: value / total for key, value in out.items()} if total > 0 else {}


def entry_bench_weights(
    entry: BookEntry,
    side_index: int,
    species: Sequence[str],
    seen: Collection[str],
    leads: Collection[str] | None = None,
) -> dict[tuple[str, ...], float]:
    """`bench_weights` for one side of a solved entry, in one call.

    Delegates rather than repeating `BenchPrior.of`: which side reads one vector and
    which reads a class-weighted average is the part worth having in a single place.
    """
    return BenchPrior.of(entry, side_index, species).weights(seen, leads)


__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_TEMPERATURE",
    "BenchPrior",
    "BookEntry",
    "SelectionDraw",
    "bench_weights",
    "entry_bench_weights",
    "explore_mixture",
    "perplexity",
    "set_from_json",
    "set_to_json",
]
