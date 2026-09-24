"""How much of a turn's randomness to enumerate: the `Budget` every road hands the port.

Moved out of resolve.py (IKA-209) so that the production roads, which no longer resolve a
turn in Python, do not import the resolver for it. `resolve.Budget` is this class.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

#: The default for `Budget.merge_duplicates`, read once at import.
#:
#: An environment variable rather than only a field because the comparison that matters is
#: a whole generation run against itself -- every worker process, both budgets, and the
#: Rust port, which is handed the field over the bridge. A flag on one tool could not
#: reach any of that.
#:
#:     POKEURAOU_MERGE_BRANCHES=0 uv run python tools/bench_generation.py --games 6
MERGE_BRANCHES_DEFAULT = os.environ.get("POKEURAOU_MERGE_BRANCHES", "1") not in (
    "0",
    "false",
    "no",
)


@dataclass(frozen=True, slots=True)
class Budget:
    """How much of the randomness to enumerate.

    ``damage_rolls`` is how many of the 16 rolls to keep. Fewer means a stratified subset
    with correct weights -- each representative stands for the contiguous block nearest it
    -- not a random sample, so the result stays reproducible and the loss is a known
    quantisation of the damage distribution rather than noise.
    """

    damage_rolls: int = 16
    enumerate_crit: bool = True
    enumerate_accuracy: bool = True
    enumerate_status_checks: bool = True
    enumerate_secondary: bool = True
    enumerate_speed_ties: bool = True
    #: Reproduce the oracle's pinned randomness policy rather than the real distribution.
    #:
    #: Only the differential test wants this. That policy answers `sample(values)` with
    #: `values[0]` and `random(a, b)` with `a`, so a Champions sleep reads as 2 turns and a
    #: bind as 5 -- and the resolver has to agree for the comparison to be an equality test.
    #: Everywhere else those are the *shortest* outcome of a real distribution, and
    #: assuming them is a bias in favour of whoever is being locked down.
    pinned_policy: bool = False
    #: Hard cap on live branches, enforced *while* each generation is built so the peak
    #: is bounded too. When it binds, the least likely branches are dropped, the rest are
    #: renormalised, and the reduction is recorded -- ``TurnResult.exact`` goes false.
    #:
    #: The default is deliberately modest: the branch tree is the product over every
    #: damaging hit in the turn, so an unbounded exact enumeration of a four-hit turn is
    #: tens of thousands of copied positions.
    max_branches: int = 512
    #: Fold branches that reached the same state into one, summing their probabilities.
    #:
    #: Lossless, so this is on: the branches it merges are indistinguishable in the
    #: position, the queue behind them and every piece of within-turn bookkeeping, and
    #: whatever the rest of the turn would do to one it would do to the other. The 16
    #: damage rolls of a guaranteed knock-out are the clearest case -- the roll decided
    #: nothing -- and integer HP produces the same collapse below the threshold.
    #:
    #: The knob exists because "lossless" is a claim that has to be checkable: with it
    #: off the resolver produces the unmerged tree, and `tools/branch_dedup.py` holds the
    #: two against each other outcome by outcome. `POKEURAOU_MERGE_BRANCHES=0` turns it
    #: off for a whole run -- see :data:`MERGE_BRANCHES_DEFAULT`.
    merge_duplicates: bool = field(default_factory=lambda: MERGE_BRANCHES_DEFAULT)

    @staticmethod
    def exact() -> Budget:
        """Every source of chance enumerated.

        The branch tree is the *product* over every damaging hit in the turn, so with 16
        rolls and four hits this is tens of thousands of states. Measured at roughly five
        seconds a turn; use it to check one interesting line, not to fill a matrix.
        """
        return Budget()

    @staticmethod
    def matrix(roll: int = 8) -> Budget:
        """The median damage roll, and Speed ties enumerated because they must be.

        Measured at about 1.5 ms a turn, so a 24x24 matrix is under a second. The damage
        *distribution* does not have to come from branching the turn -- the calculator
        returns all sixteen rolls at once -- so this is the setting for filling a matrix
        and a richer budget is for the handful of cells worth looking at closely.

        Speed ties are the one exception to collapsing, and it is a correctness matter
        rather than an accuracy one. The LP is handed a single matrix and reads a maximin
        off it for one player and a minimax for the other, which is only coherent if the
        matrix is zero-sum: in a mirror, ``M[i][j] + M[j][i]`` must be 1. Resolving a tie
        one way breaks that -- measured at up to 0.042 per cell, mean 0.011, biased +0.005
        toward side 0, because the side the canonical order puts first plans as though it
        wins every tie while the other plans as though it loses every one. Compounded over
        a game it showed up as a 59.1% side-0 win rate in a mirror, where symmetry forces
        50%. Every *other* collapse here (crit, accuracy, secondary, status) leaves the
        identity exact, so ties are the only one that has to be paid for.

        The cost is 1.06x on real matchups, where different spreads make exact ties rare;
        it is 3.4x in a mirror, where every pair is tied. `max_branches` has to allow for
        the tie branches or they would be truncated back out again.

        Status checks and secondaries are enumerated too, and both are bargains. Measured
        on 12 mid-game positions from recorded games with each knob turned on alone
        (`tools/budget_effect.py`), against the equilibrium this budget produces:

            knob             value shift   policy TV    cost   worst value shift
            accuracy              0.0001       0.006   1.08x              0.0010
            crit                  0.0005       0.003   2.68x              0.0026
            status checks         0.0040       0.083   0.97x              0.0485
            secondary             0.0083       0.087   1.20x              0.0957

        "policy TV" is the share of the equilibrium's mass that moves, which is the share
        of the time the two strategies would actually choose differently.

        Collapsing status checks had the search believe a paralysed Pokemon always acts, a
        confused one never hits itself and a repeated Protect always fails, for no
        measurable saving. Collapsing secondaries had Heat Wave never burn and Iron Head
        never flinch -- and flinch is not chip damage, it is the target losing its action,
        with Iron Head on 36% of the tournament field and Rock Slide on 62%.

        Accuracy is enumerated as well. Its measured policy effect is the smallest of the
        four, but 1.08x is close enough to free that faithfulness wins the tie: a move that
        misses one time in four is a move the search should know can miss. Hurricane is 70%,
        Sleep Powder 75%, and Heat Wave and Rock Slide -- the two most common moves in the
        field -- are 90% each, thrown twice a turn for a dozen turns.

        Crit is the one that stays collapsed: 2.7x for a thirtieth of the policy effect, and
        unlike a miss a crit changes one damage number rather than whether the move happened
        at all.
        """
        # `pinned_policy` is cleared explicitly. It is built from `deterministic`, which
        # exists to reproduce the oracle's pinned randomness, and inheriting that flag would
        # quietly put the *search* in "answer every roll the way the differential test's
        # policy does" mode -- a Champions sleep read as its one-in-three short outcome
        # rather than its modal one.
        return replace(
            Budget.deterministic(roll),
            enumerate_speed_ties=True,
            enumerate_status_checks=True,
            enumerate_secondary=True,
            enumerate_accuracy=True,
            max_branches=16,
            pinned_policy=False,
        )

    @staticmethod
    def fast() -> Budget:
        """Enough resolution to rank actions, cheap enough for a whole matrix."""
        # Two rolls, not four: the branch tree is the product over every damaging hit in
        # the turn, so four rolls across four hits is 256 states and each one is a copied
        # position. Two rolls keeps the high/low spread that decides knock-outs while
        # keeping a whole matrix affordable.
        return Budget(
            damage_rolls=2,
            enumerate_crit=False,
            enumerate_status_checks=False,
            enumerate_secondary=False,
            max_branches=64,
        )

    @staticmethod
    def deterministic(roll: int = 0) -> Budget:
        """One branch: a fixed damage roll, always hits, never crits.

        This is the setting the differential test uses, because it is the one Showdown's
        pinned randomness policy reproduces exactly.
        """
        return Budget(
            damage_rolls=1,
            enumerate_crit=False,
            enumerate_accuracy=False,
            enumerate_status_checks=False,
            enumerate_secondary=False,
            enumerate_speed_ties=False,
            pinned_policy=True,
            max_branches=1,
        ).with_fixed_roll(roll)

    def with_fixed_roll(self, roll: int) -> Budget:
        return replace(self, damage_rolls=-1 - max(0, min(15, roll)))

    def per_action_branches(self) -> int:
        """Roughly how many outcomes one action can produce under this budget.

        An estimate, not a bound: multi-hit counts and a spread move's second target
        multiply on top. The incremental prune catches what this misses; the point here is
        to avoid generating branches that are certain to be discarded.
        """
        rolls = 1 if self.fixed_roll is not None else max(1, min(16, self.damage_rolls))
        factor = rolls
        if self.enumerate_crit:
            factor *= 2
        if self.enumerate_accuracy:
            factor *= 2
        if self.enumerate_status_checks:
            factor *= 2
        return factor

    def narrowed(self, room: int) -> Budget:
        """This budget, reduced until one action produces at most ``room`` outcomes.

        Gives up, in order: damage-roll resolution, then crit, then the status checks,
        then accuracy. Accuracy is last because a miss changes the resulting position more
        than any of the others.
        """
        if room >= self.per_action_branches() or self.fixed_roll is not None:
            return self
        narrowed = self
        # Damage rolls first: halve until they fit or only one is left.
        while narrowed.damage_rolls > 1 and narrowed.per_action_branches() > room:
            narrowed = replace(narrowed, damage_rolls=max(1, narrowed.damage_rolls // 2))
        for field_name in ("enumerate_crit", "enumerate_status_checks", "enumerate_secondary"):
            if narrowed.per_action_branches() <= room:
                break
            if getattr(narrowed, field_name):
                narrowed = replace(narrowed, **{field_name: False})
        if narrowed.per_action_branches() > room and narrowed.enumerate_accuracy:
            narrowed = replace(narrowed, enumerate_accuracy=False)
        return narrowed

    @property
    def fixed_roll(self) -> int | None:
        """The single roll index, when this budget pins one."""
        return None if self.damage_rolls >= 0 else -self.damage_rolls - 1


__all__ = ["MERGE_BRANCHES_DEFAULT", "Budget"]
