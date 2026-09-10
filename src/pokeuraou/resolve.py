"""Single-turn resolver.

Takes a fully-known position and one chosen action per side, and returns the possible
successor positions with their exact probabilities. The caller integrates over them; the
resolver never averages anything itself, so nothing it returns is a point estimate
standing in for a distribution.

Three design commitments shape everything here.

**The position is fully known.** The belief layer hands the resolver one concrete SP
spread per particle, so from inside there is no hidden information. That keeps the engine
scalar and lets it be compared against Showdown -- also fully informed -- turn by turn and
state field by state field.

**Randomness is enumerated, not sampled.** A turn's chance comes from a small number of
independent draws: 16 damage rolls per hit, a crit, an accuracy check, a secondary, a
Speed tie, a Quick Claw. Enumerating them gives exact probabilities and reproducible
output; sampling would put Monte Carlo noise into a displayed win probability, which is
the kind of number that cannot be justified. When the branch count would exceed the
budget, damage rolls are *stratified* -- a fixed subset carrying the weight of the block
it represents -- and :class:`TurnResult` records what was reduced.

**Showdown's mid-turn re-sort is honoured.** From generation 8 on, Speed is recomputed and
the remaining queue re-sorted after every action, so Tailwind going up or a Speed drop
landing reorders what has not happened yet. Each branch re-sorts against its own state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .actions import (
    MoveAction,
    PassAction,
    SideAction,
    SwitchAction,
    switch_actions_after_faint,
)
from .battler import Battler, FieldState
from .damage import calculate, crit_probability
from .effects import (
    RESIST_BERRIES,
    SURVIVE_AT_ONE_ABILITIES,
    SURVIVE_AT_ONE_ITEMS,
    SURVIVE_CHANCE_ITEMS,
    WEATHER_ABILITIES,
    item_is_removable,
)
from .moveinfo import MoveContext
from .position import Effect, Pokemon, Position
from .regulation import BOOST_IDS, Move, Regulation
from .speed import QueuedAction, build_queue, effective_speed, order_groups
from .view import battler, field_state, move_hits_multiple

#: Protect-family volatiles, and what each blocks. Endure is deliberately absent: it is a
#: stalling move that shares the counter, but its volatile caps damage rather than
#: blocking the hit, so it is handled in `deal_damage` instead.
PROTECT_VOLATILES: dict[str, str] = {
    "protect": "all",
    "detect": "all",
    "banefulbunker": "all",
    "burningbulwark": "all",
    "spikyshield": "all",
    "kingsshield": "damaging",
    "obstruct": "damaging",
    "silktrap": "damaging",
    "maxguard": "all",
}

#: Moves that put a Protect-family volatile on their user.
PROTECT_MOVES = frozenset(PROTECT_VOLATILES)

#: Moves that raise the Protect counter without ever consulting it. Showdown gives Wide
#: Guard and Quick Guard `source.addVolatile('stall')` in `onHitSide` but no
#: `stallingMove` flag, so they always succeed themselves and still make the next Protect
#: a 1-in-3. Which moves *consult* the counter is not written out here -- it comes from
#: the dumped `stallingMove` flag, so a regulation that changes the family needs no code
#: change.
STALL_BUMPING_MOVES = frozenset({"wideguard", "quickguard"})

#: Abilities that block priority moves aimed at the holder's side.
PRIORITY_BLOCKING_ABILITIES = frozenset({"armortail", "queenlymajesty", "dazzling"})

#: Volatiles that redirect single-target moves to their holder.
REDIRECTION_VOLATILES = ("followme", "ragepowder", "spotlight")

#: Abilities that draw a move of their type in doubles.
REDIRECTION_ABILITIES: dict[str, str] = {"lightningrod": "Electric", "stormdrain": "Water"}

#: Pseudo-weathers that switch off when their move is used again. Everything else fails
#: when it is already up.
TOGGLING_PSEUDO_WEATHER = frozenset({"trickroom", "magicroom", "wonderroom"})

#: Abilities that stop the *opposing* side from eating berries.
BERRY_BLOCKING_ABILITIES = frozenset({"unnerve", "asonechillingneigh", "asonegrimneigh"})

#: Abilities that ignore Intimidate.
INTIMIDATE_PROOF_ABILITIES = frozenset(
    {"innerfocus", "oblivious", "owntempo", "scrappy", "guarddog", "clearbody",
     "whitesmoke", "fullmetalbody", "hypercutter"}
)

#: Full paralysis. The champions mod overrides the base game's 1/4:
#:     par: { inherit: true, onBeforeMove(pokemon) { if (this.randomChance(1, 8)) ... } }
#: Verified by reading the roll the simulator asked for -- `randomChance(1, 8)` on every
#: turn a paralysed Pokemon tries to move. This was 0.25, i.e. twice the real rate, and the
#: search multiplied it through every branch.
FULL_PARALYSIS_CHANCE = 1.0 / 8.0

CONFUSION_SELF_HIT_CHANCE = 1.0 / 3.0

#: Thawing. The champions mod rewrites `frz.onBeforeMove` to `randomChance(1, 4)` with a
#: hard three-turn cap, where the base game uses 1/5 and no cap. Taken from the mod's source
#: rather than from an observed roll: the policy has to force secondaries on before a freeze
#: can be applied at all, and no thaw roll was recorded in that state, so this one is not
#: independently confirmed the way paralysis is.
THAW_CHANCE = 0.25

#: Champions sleep is `sample([2, 3, 3])` in the mod: the counter starts at 2 one time in
#: three and at 3 the rest. With the decrement in `_can_act` that is one turn of sleep or
#: two -- "the first action is always asleep, the second wakes one time in three, the third
#: always acts". The base game rolls `random(2, 5)`, so 1, 2 or 3 turns equally.
#:
#: This is the modal outcome, which is what an unbranched sleep uses. Pinning 2 assumed the
#: lucky case on every sleep in the search.
SLEEP_COUNTER_MODAL = 3

#: Freeze, as the mod writes it: `startTime = 3`, decremented on each attempt to move, and
#: a forced thaw at zero on top of the 1/4 roll. Without the cap a freeze had only the
#: geometric tail of the roll to end it, so it could hold for the rest of the battle.
FREEZE_COUNTER = 3

#: The oracle's policy answers `sample(values)` with `values[0]`, so this is what the
#: differential test's Showdown produces.
SLEEP_COUNTER_PINNED = 2

#: Residual amounts, as (numerator, denominator) of max HP.
BURN_DAMAGE = (1, 16)
POISON_DAMAGE = (1, 8)
SANDSTORM_DAMAGE = (1, 16)
LEECH_SEED_DRAIN = (1, 8)
LEFTOVERS_HEAL = (1, 16)
PARTIAL_TRAP_DAMAGE = (1, 8)
SALT_CURE_DAMAGE = (1, 8)
SALT_CURE_DAMAGE_WEAK = (1, 4)

#: Abilities that trade HP with the weather, as ability -> weather -> (numerator,
#: denominator) of max HP. A negative numerator is damage. All of them are `onWeather`
#: handlers in Showdown, so they run with the weather's own residual.
WEATHER_ABILITY_HP: dict[str, dict[str, tuple[int, int]]] = {
    "solarpower": {"sunnyday": (-1, 8), "desolateland": (-1, 8)},
    "dryskin": {
        "sunnyday": (-1, 8),
        "desolateland": (-1, 8),
        "raindance": (1, 8),
        "primordialsea": (1, 8),
    },
    "raindish": {"raindance": (1, 16), "primordialsea": (1, 16)},
    "icebody": {"hail": (1, 16), "snowscape": (1, 16)},
}

SANDSTORM_IMMUNE_TYPES = frozenset({"Rock", "Ground", "Steel"})
SANDSTORM_IMMUNE_ABILITIES = frozenset(
    {"sandveil", "sandrush", "sandforce", "overcoat", "magicguard"}
)

#: Status moves whose whole effect is captured by the declarative fields plus the special
#: cases handled below, so their Showdown handlers add nothing the resolver misses.
STATUS_MOVES_FULLY_MODELLED = frozenset(
    PROTECT_MOVES
    | {
        "followme", "ragepowder", "spotlight", "helpinghand", "wideguard", "quickguard",
        "tailwind", "trickroom", "lightscreen", "reflect", "auroraveil", "sunnyday",
        "raindance", "sandstorm", "snowscape", "electricterrain", "grassyterrain",
        "mistyterrain", "psychicterrain", "spikes", "toxicspikes", "stealthrock",
        "stickyweb", "swordsdance", "nastyplot", "calmmind", "irondefense", "amnesia",
        "agility", "bulkup", "howl", "growl", "leer", "tailwhip", "screech", "charm",
        "faketears", "metalsound", "willowisp", "thunderwave", "toxic", "hypnosis",
        "spore", "sleeppowder", "confuseray", "leechseed", "taunt", "encore",
        "partingshot", "trick", "switcheroo", "roar", "whirlwind", "lifedew",
        "moonlight", "synthesis", "morningsun", "recover", "softboiled", "slackoff",
        "milkdrink", "roost", "yawn", "knockoff", "thief", "covet", "perishsong",
    }
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


def stratified_rolls(budget: Budget) -> list[tuple[int, float]]:
    """(roll index, probability) covering all 16 rolls with the budget's resolution.

    With 16 this is the exact distribution. With fewer, each representative carries the
    weight of the block it stands for, so the distribution's mean is preserved and the
    only loss is resolution near a knock-out threshold. A pinned roll returns just that
    roll with weight 1, which is what the differential test needs.
    """
    fixed = budget.fixed_roll
    if fixed is not None:
        return [(fixed, 1.0)]
    count = max(1, min(16, budget.damage_rolls))
    edges = np.linspace(0, 16, count + 1).round().astype(int)
    out: list[tuple[int, float]] = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        if hi <= lo:
            continue
        out.append(((lo + hi - 1) // 2, (hi - lo) / 16.0))
    return out


@dataclass
class Branch:
    """One fully determined outcome of the turn."""

    probability: float
    position: Position
    #: Readable trace of what happened, in order.
    events: list[str] = field(default_factory=list)


@dataclass
class SuspendedTurn:
    """A turn Showdown stopped halfway through to ask for a replacement.

    A self-switching move sets `switchFlag`, and the check at the end of ``runAction`` --
    which runs after every action -- turns that into a fresh switch request. So the turn is
    genuinely paused: the replacement enters before the rest of the queue runs, the residual
    phase included.

    The pause is also the honest information state. Whoever owes the replacement chooses it
    without seeing how the remaining actions turn out, so the choice has to be made here,
    once, rather than separately inside each of the outcomes that follow it.

    Resume with :func:`resume_turn`. ``position`` is the state at the pause, which is what
    the chooser can see.
    """

    probability: float
    position: Position
    #: Readable trace of what happened up to the pause.
    events: list[str] = field(default_factory=list)
    #: Continuation state. Private because resuming has to go through `resume_turn`, which
    #: copies it -- one suspension is resumed once per candidate replacement.
    _turn: _Turn | None = None
    _remaining: tuple[QueuedAction, ...] = ()


@dataclass
class TurnResult:
    branches: list[Branch]
    #: True when every source of chance was enumerated in full.
    exact: bool
    #: What was reduced, and how often, when the budget bound.
    reductions: dict[str, int] = field(default_factory=dict)
    #: Effects encountered that the resolver does not model. Reported, never ignored.
    unmodelled: tuple[str, ...] = ()
    #: Outcomes that stopped at a mid-turn replacement request. These are *not* finished
    #: turns: `branches` and `suspended` together carry the turn's probability, and a
    #: caller that ignores this field silently drops that mass.
    suspended: tuple[SuspendedTurn, ...] = ()

    @property
    def total_probability(self) -> float:
        """The whole turn's mass, suspended outcomes included."""
        return sum(b.probability for b in self.branches) + sum(
            s.probability for s in self.suspended
        )

    def expected(self, value: Callable[[Position], float]) -> float:
        total = self.total_probability
        if total <= 0:
            return 0.0
        self._require_resumed("expected")
        return sum(b.probability * value(b.position) for b in self.branches) / total

    def _require_resumed(self, what: str) -> None:
        """Refuses to summarise a turn that has not finished.

        Averaging over `branches` alone while a suspension holds part of the mass gives a
        number that is short by that fraction and looks perfectly ordinary. Whoever owes the
        mid-turn replacement has to choose it first -- see `resume_turn`.
        """
        if self.suspended:
            raise ValueError(
                f"{what}() on a turn with {len(self.suspended)} suspended outcome(s): a "
                "self-switching move paused the turn for a replacement choice. Resolve "
                "them with resume_turn() first."
            )

    def collapse(self, key: Callable[[Position], object]) -> dict[object, float]:
        """Groups branches by a projection of the position, summing probabilities.

        "Which Pokemon faint" is such a projection, and has a handful of values even when
        there are hundreds of branches -- which is what makes the output readable.
        """
        self._require_resumed("collapse")
        out: dict[object, float] = {}
        for b in self.branches:
            k = key(b.position)
            out[k] = out.get(k, 0.0) + b.probability
        total = sum(out.values()) or 1.0
        return dict(sorted(((k, v / total) for k, v in out.items()), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Working state
# ---------------------------------------------------------------------------


class _Turn:
    """Mutable working state for one branch of one turn."""

    __slots__ = ("reg", "pos", "budget", "attacks", "events", "unmodelled",
                 "hurt_this_turn", "move_failed", "move_damage_total", "move_connected",
                 "acted", "actions_remaining", "self_switch_pending",
                 "pending_secondaries", "current_actor")

    def __init__(
        self,
        reg: Regulation,
        pos: Position,
        budget: Budget,
        attacks: dict[tuple[int, int], bool],
    ) -> None:
        self.reg = reg
        self.pos = pos
        self.budget = budget
        #: Which slots are about to use a damaging move, for Sucker Punch.
        self.attacks = attacks
        self.events: list[str] = []
        self.unmodelled: set[str] = set()
        self.hurt_this_turn: set[tuple[int, int]] = set()
        self.move_failed: set[tuple[int, int]] = set()
        #: Damage the move currently resolving has dealt, summed over its targets. Recoil
        #: and Life Orb are computed from the total once, not per target.
        self.move_damage_total = 0
        #: Whether the move currently resolving connected with anything. A move that was
        #: blocked, missed or had no effect does not apply its own `self` effect.
        self.move_connected = False
        #: Slots that have already taken their action this turn. Sucker Punch fails against
        #: a target that has already moved, which is the difference between a read that
        #: worked and one that did not.
        self.acted: set[tuple[int, int]] = set()
        #: How many actions are still queued behind the one resolving. Protect fails when
        #: this is zero -- Showdown gates it on `queue.willAct()`.
        self.actions_remaining = 0
        #: Set when a self-switching move has left a slot owing a replacement, which
        #: suspends the turn. A flag rather than a scan of the position because it is
        #: consulted after every action in the resolver's innermost loop.
        self.self_switch_pending = False
        #: Sub-100% secondaries the hit would apply, as (chance, secondary, target).
        #: Recorded rather than applied because applying one is a branch and this state is
        #: singular; the hit loop owns the fan-out.
        self.pending_secondaries: list[tuple[float, dict, tuple[int, int]]] = []
        #: The slot whose move is resolving. Disable's duration depends on whether its
        #: subject is the Pokemon that triggered it, which is how Cursed Body gets four
        #: turns rather than five.
        self.current_actor: tuple[int, int] | None = None

    def clone(self) -> _Turn:
        fresh = _Turn(self.reg, self.pos.copy(), self.budget, self.attacks)
        fresh.events = list(self.events)
        fresh.unmodelled = set(self.unmodelled)
        fresh.hurt_this_turn = set(self.hurt_this_turn)
        fresh.move_failed = set(self.move_failed)
        fresh.move_damage_total = self.move_damage_total
        fresh.move_connected = self.move_connected
        fresh.acted = set(self.acted)
        fresh.actions_remaining = self.actions_remaining
        fresh.self_switch_pending = self.self_switch_pending
        fresh.pending_secondaries = list(self.pending_secondaries)
        fresh.current_actor = self.current_actor
        return fresh

    # -- lookups ------------------------------------------------------------

    def mon_at(self, side: int, slot: int) -> Pokemon | None:
        party = self.pos.sides[side].active[slot]
        return None if party is None else self.pos.sides[side].pokemon[party]

    def battler_at(self, side: int, slot: int) -> Battler | None:
        mon = self.mon_at(side, slot)
        if mon is None or mon.fainted:
            return None
        return battler(self.reg, mon)

    def types_of(self, mon: Pokemon) -> tuple[str, ...]:
        return mon.types or self.reg.species[mon.species].types

    def field(self) -> FieldState:
        return field_state(self.pos, self.reg)

    def log(self, message: str) -> None:
        self.events.append(message)

    @staticmethod
    def name(side: int, slot: int) -> str:
        return f"p{side + 1}{'ab'[slot]}"

    # -- HP -----------------------------------------------------------------

    def fraction_of_max(self, side: int, slot: int, ratio: tuple[int, int]) -> int:
        mon = self.mon_at(side, slot)
        return 0 if mon is None else max(1, mon.maxhp * ratio[0] // ratio[1])

    def deal_damage(
        self, side: int, slot: int, amount: int, *, reason: str, from_move: bool = False
    ) -> int:
        """Applies damage, honouring survival effects, and returns what was dealt.

        Endure, Focus Sash and Sturdy all test `effect.effectType === 'Move'` in Showdown,
        so `from_move` decides whether they apply at all: none of them saves from
        sandstorm, recoil or Life Orb.
        """
        mon = self.mon_at(side, slot)
        if mon is None or mon.fainted or amount <= 0:
            return 0
        dealt = min(amount, mon.hp)
        if from_move and dealt >= mon.hp:
            # Endure runs at onDamagePriority -10 and the items at -40, so Endure caps the
            # damage first and the Sash, seeing a survivable hit, is never consumed.
            if mon.has_volatile("endure"):
                dealt = mon.hp - 1
            elif mon.hp >= mon.maxhp and (
                mon.item in SURVIVE_AT_ONE_ITEMS
                or mon.ability in SURVIVE_AT_ONE_ABILITIES
            ):
                dealt = mon.hp - 1
                if mon.item in SURVIVE_AT_ONE_ITEMS:
                    self.consume_item(side, slot, reason=mon.item or "")
            elif mon.item in SURVIVE_CHANCE_ITEMS:
                # A 1-in-10 survival from any HP. The branch is not plumbed through damage
                # application, so say so rather than quietly resolving it either way.
                self.unmodelled.add(f"survival chance not branched:{mon.item}")
        mon.hp -= dealt
        self.hurt_this_turn.add((side, slot))
        self.log(f"{self.name(side, slot)} -{dealt} ({reason})")
        if mon.hp <= 0:
            self.faint(side, slot)
        else:
            self.check_berry(side, slot)
        return dealt

    def heal(self, side: int, slot: int, amount: int, *, reason: str) -> int:
        mon = self.mon_at(side, slot)
        if mon is None or mon.fainted or amount <= 0:
            return 0
        healed = min(amount, mon.maxhp - mon.hp)
        mon.hp += healed
        if healed:
            self.log(f"{self.name(side, slot)} +{healed} ({reason})")
        return healed

    def faint(self, side: int, slot: int) -> None:
        mon = self.mon_at(side, slot)
        if mon is None or mon.fainted:
            return
        mon.hp = 0
        mon.fainted = True
        mon.boosts = {}
        mon.volatiles = []
        # Showdown records a fainted Pokemon's status as 'fnt'.
        mon.status = "fnt"
        mon.status_counter = None
        self.log(f"{self.name(side, slot)} fainted")

    # -- items, status, boosts ---------------------------------------------

    def consume_item(self, side: int, slot: int, *, reason: str) -> None:
        mon = self.mon_at(side, slot)
        if mon is None or mon.item is None:
            return
        self.log(f"{self.name(side, slot)} lost {mon.item} ({reason})")
        # Unburden's doubled Speed keys off a volatile added when the item is lost, not
        # off simply holding nothing.
        if mon.ability == "unburden" and not mon.has_volatile("unburden"):
            mon.volatiles.append(Effect(id="unburden"))
        mon.item = None

    def berries_blocked(self, side: int) -> bool:
        """Whether an opposing Unnerve stops this side from eating berries."""
        foe_side = 1 - side
        for slot in range(len(self.pos.sides[foe_side].active)):
            foe = self.mon_at(foe_side, slot)
            if foe is not None and not foe.fainted and foe.ability in BERRY_BLOCKING_ABILITIES:
                return True
        return False

    def check_berry(self, side: int, slot: int) -> None:
        mon = self.mon_at(side, slot)
        if mon is None or mon.item is None or mon.fainted:
            return
        if self.berries_blocked(side):
            return
        if mon.item in ("sitrusberry", "oranberry") and mon.hp * 2 <= mon.maxhp:
            amount = max(1, mon.maxhp // 4) if mon.item == "sitrusberry" else 10
            self.consume_item(side, slot, reason="pinch berry")
            self.heal(side, slot, amount, reason="berry")

    def apply_boosts(
        self, side: int, slot: int, boosts: dict[str, int], *, reason: str, from_foe: bool = True
    ) -> bool:
        """Applies a boost table, and says whether any stat actually moved.

        The return value is Showdown's `success` from `Battle#boost`, which Parting Shot
        reads to decide whether it switches out at all.
        """
        mon = self.mon_at(side, slot)
        if mon is None or mon.fainted:
            return False
        if mon.ability == "contrary":
            # `onChangeBoost` inverts every stat change aimed at the holder, whatever the
            # source -- including the drops from its own Close Combat, which is the point
            # of Mega Staraptor. The inversion happens *before* anything reads the sign,
            # so Defiant does not see a foe's drop that became a raise.
            boosts = {stat: -delta for stat, delta in boosts.items()}
        changed = False
        for stat, delta in boosts.items():
            if stat not in BOOST_IDS:
                self.unmodelled.add(f"boost:{stat}")
                continue
            # Clear Body and friends only block a drop inflicted by another Pokemon; a
            # move's own drawback still applies to its user.
            if (
                delta < 0
                and from_foe
                and mon.ability in ("clearbody", "whitesmoke", "fullmetalbody")
            ):
                self.log(f"{self.name(side, slot)} {mon.ability} blocked the drop")
                continue
            before = mon.boosts.get(stat, 0)
            after = max(-6, min(6, before + delta))
            if after == before:
                continue
            if after:
                mon.boosts[stat] = after
            else:
                mon.boosts.pop(stat, None)
            self.log(f"{self.name(side, slot)} {stat} {delta:+d} -> {after} ({reason})")
            changed = True
            # `runEvent('AfterEachBoost', ...)` sits *inside* Showdown's per-stat loop and
            # only fires when the stat actually moved, so Defiant answers each drop
            # separately: Parting Shot's two drops are +4, not +2. It also fires between
            # them, so the raise is part of the state the next drop is applied to.
            if delta < 0 and from_foe:
                self.on_stat_lowered_by_foe(side, slot)
        return changed

    def on_stat_lowered_by_foe(self, side: int, slot: int) -> None:
        """Defiant and Competitive react only to a drop the opponent caused."""
        mon = self.mon_at(side, slot)
        if mon is None:
            return
        if mon.ability == "defiant":
            self.apply_boosts(side, slot, {"atk": 2}, reason="defiant", from_foe=False)
        elif mon.ability == "competitive":
            self.apply_boosts(side, slot, {"spa": 2}, reason="competitive", from_foe=False)

    def apply_status(self, side: int, slot: int, status: str, *, reason: str) -> bool:
        mon = self.mon_at(side, slot)
        if mon is None or mon.fainted or mon.status is not None:
            return False
        types = self.types_of(mon)
        immune_by_type = {
            "psn": ("Poison", "Steel"),
            "tox": ("Poison", "Steel"),
            "brn": ("Fire",),
            "par": ("Electric",),
            "frz": ("Ice",),
        }.get(status, ())
        if any(t in types for t in immune_by_type):
            return False
        if mon.ability in ("immunity", "limber", "waterveil", "insomnia", "vitalspirit",
                           "comatose", "purifyingsalt", "thermalexchange"):
            return False
        if self.pos.field.terrain == "mistyterrain" and _grounded(self, mon):
            return False
        if status == "slp" and self.pos.field.terrain == "electricterrain" and _grounded(self, mon):
            return False
        mon.status = status
        if status == "tox":
            mon.status_counter = 0
        elif status == "frz":
            mon.status_counter = FREEZE_COUNTER
        elif status == "slp":
            # The duration is part of the state, so it has to be a number here rather than
            # a distribution -- branching it would mean branching the position from inside
            # a status application, which this function cannot do. The modal outcome is
            # used and reported; under the pinned policy the oracle's `sample` returns the
            # first element, and matching that is what keeps the differential an equality
            # test.
            if self.budget.pinned_policy:
                mon.status_counter = SLEEP_COUNTER_PINNED
            else:
                mon.status_counter = SLEEP_COUNTER_MODAL
                self.unmodelled.add(
                    f"sleep duration ({SLEEP_COUNTER_MODAL - 1} turns, the modal outcome "
                    f"of Champions' 1-or-2; not branched)"
                )
        self.log(f"{self.name(side, slot)} -> {status} ({reason})")
        if mon.item == "lumberry":
            self.consume_item(side, slot, reason="lumberry")
            mon.status = None
            mon.status_counter = None
        return True

    def add_volatile(self, side: int, slot: int, vid: str, *, duration: int | None = None) -> None:
        mon = self.mon_at(side, slot)
        if mon is None or mon.fainted or mon.has_volatile(vid):
            return
        mon.volatiles.append(Effect(id=vid, duration=duration))

    def add_side_condition(self, side: int, cid: str, *, duration: int | None = None) -> None:
        existing = self.pos.sides[side].side_condition(cid)
        if existing is not None:
            if cid in ("spikes", "toxicspikes"):
                cap = 3 if cid == "spikes" else 2
                existing.layers = min((existing.layers or 1) + 1, cap)
            return
        self.pos.sides[side].side_conditions.append(
            Effect(id=cid, duration=duration, layers=1)
        )
        self.log(f"p{side + 1} side +{cid}")


def _grounded(turn: _Turn, mon: Pokemon) -> bool:
    if mon.has_volatile("smackdown") or mon.has_volatile("ingrain") or mon.item == "ironball":
        return True
    if mon.has_volatile("magnetrise") or mon.has_volatile("telekinesis"):
        return False
    if mon.ability == "levitate" or mon.item == "airballoon":
        return False
    return "Flying" not in turn.types_of(mon)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def pending_attacks(
    reg: Regulation, side_actions: list[SideAction]
) -> dict[tuple[int, int], bool]:
    """Which active slots are about to use a damaging move.

    Sucker Punch needs this, and it is knowable precisely because the resolver evaluates a
    *pair* of chosen actions -- which is what makes Sucker Punch a read belonging in the
    matrix rather than a coin flip.
    """
    out: dict[tuple[int, int], bool] = {}
    for side_index, action in enumerate(side_actions):
        for slot_action in action.slots:
            if isinstance(slot_action, MoveAction):
                out[(side_index, slot_action.slot)] = (
                    reg.moves[slot_action.move_id].category != "Status"
                )
            elif isinstance(slot_action, SwitchAction | PassAction):
                out[(side_index, slot_action.slot)] = False
    return out


def resolve_turn(
    reg: Regulation,
    pos: Position,
    side_actions: list[SideAction],
    *,
    budget: Budget | None = None,
) -> TurnResult:
    """Every outcome of one turn, with exact probabilities.

    The position must be fully known: the belief layer supplies a concrete spread per
    particle, so a hidden SP here is a programming error rather than something to guess at.
    """
    budget = budget or Budget.exact()
    for side in pos.sides:
        for mon in side.pokemon:
            if mon.sp is None and mon.stats_override is None:
                raise ValueError(
                    f"{mon.species} has no SP spread and no stats: the resolver needs a "
                    "fully-known position, so the belief layer must fill it in first"
                )

    attacks = pending_attacks(reg, side_actions)
    fs = field_state(pos, reg)
    battlers = [
        [
            battler(reg, m) if m is not None and not m.fainted else None
            for m in side.active_pokemon()
        ]
        for side in pos.sides
    ]
    queues = build_queue(reg, pos, side_actions, battlers, fs)

    branches: list[Branch] = []
    suspended: list[SuspendedTurn] = []
    unmodelled: set[str] = set()
    reductions: dict[str, int] = {}
    exact = True

    for queue in queues:
        if not queue:
            continue
        weight = queue[0].branch_probability
        groups = order_groups(queue, trick_room=pos.field.trick_room)
        # The position is fully known, so exactly one order group applies.
        group = groups[0]
        for order, tie_weight in _tie_permutations(group.order, group.ties, budget):
            sequence = [queue[i] for i in order]
            sub = _resolve_sequence(reg, pos, sequence, budget, attacks)
            for branch in sub.branches:
                branch.probability *= weight * tie_weight
                branches.append(branch)
            for pause in sub.suspended:
                pause.probability *= weight * tie_weight
                suspended.append(pause)
            unmodelled |= set(sub.unmodelled)
            exact = exact and sub.exact
            for key, value in sub.reductions.items():
                reductions[key] = reductions.get(key, 0) + value

    return TurnResult(
        branches=branches,
        exact=exact,
        reductions=reductions,
        unmodelled=tuple(sorted(unmodelled)),
        suspended=tuple(suspended),
    )


def _tie_permutations(
    order: tuple[int, ...], ties: tuple[tuple[int, ...], ...], budget: Budget
) -> list[tuple[tuple[int, ...], float]]:
    """Orders produced by resolving Speed ties, with their probabilities.

    Showdown shuffles a tied group, so each permutation is equally likely. Pairwise ties
    are what occur in doubles and are expanded exactly; a larger tied group keeps the
    canonical order and is reported, rather than being silently collapsed.
    """
    if not ties or not budget.enumerate_speed_ties:
        return [(order, 1.0)]
    result: list[tuple[tuple[int, ...], float]] = [(order, 1.0)]
    for group in ties:
        if len(group) != 2:
            continue
        a, b = group
        expanded: list[tuple[tuple[int, ...], float]] = []
        for seq, weight in result:
            expanded.append((seq, weight / 2))
            swapped = list(seq)
            i, j = swapped.index(a), swapped.index(b)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            expanded.append((tuple(swapped), weight / 2))
        result = expanded
    return result


@dataclass
class _Live:
    """A branch still being resolved: its weight, state and remaining queue."""

    weight: float
    turn: _Turn
    remaining: list[QueuedAction]


def _resolve_sequence(
    reg: Regulation,
    pos: Position,
    sequence: list[QueuedAction],
    budget: Budget,
    attacks: dict[tuple[int, int], bool],
) -> TurnResult:
    """Resolves one ordered action sequence, enumerating each action's randomness."""
    return _run_queue(
        reg, [_Live(1.0, _Turn(reg, pos.copy(), budget, attacks), list(sequence))], budget
    )


def _run_queue(reg: Regulation, start: list[_Live], budget: Budget) -> TurnResult:
    """Runs branches until their queues are empty or a replacement request interrupts them.

    Entered both at the top of a turn and again from `resume_turn`, so a turn interrupted
    twice -- two U-turns on the same side, say -- goes through exactly one code path.
    """
    live = list(start)
    finished: list[_Live] = []
    paused: list[_Live] = []
    reductions: dict[str, int] = {}
    exact = True

    dropped = 0

    def prune(items: list[_Live]) -> list[_Live]:
        """Keeps the most likely branches, and says how many were let go."""
        nonlocal dropped
        if len(items) <= budget.max_branches:
            return items
        items.sort(key=lambda item: -item.weight)
        kept = items[: budget.max_branches]
        dropped += len(items) - len(kept)
        return kept

    while live:
        # Divide the remaining branch budget among the live branches, and narrow the
        # per-action resolution to fit. Without this the generation is built at full
        # resolution and then thrown away, which costs the time anyway.
        pending = [item for item in live if item.remaining]
        room = max(1, budget.max_branches // max(1, len(pending)))
        step_budget = budget.narrowed(room)
        if step_budget != budget:
            reductions["resolution narrowed to fit the branch budget"] = (
                reductions.get("resolution narrowed to fit the branch budget", 0) + 1
            )
            exact = False

        nxt: list[_Live] = []
        for item in live:
            if not item.remaining:
                finished.append(item)
                continue
            # Each branch re-sorts against its own state: a Speed change earlier in the
            # turn only reorders that branch.
            ordered = _resort(reg, item.turn, item.remaining)
            action, rest = ordered[0], ordered[1:]
            # `queue.willAct()` counts the actions still queued behind this one, which is
            # what Protect is gated on.
            item.turn.actions_remaining = len(rest)
            for weight, turn, note in _execute(reg, item.turn, action, step_budget):
                if note:
                    reductions[note] = reductions.get(note, 0) + 1
                    exact = False
                # Showdown ends the battle as soon as a side is wiped and abandons the
                # rest of the queue. Carrying on would apply moves that never happened --
                # including a spread move hitting the winner's own partner.
                wiped = _side_wiped(turn)
                remaining_actions = [] if wiped else list(rest)
                child = _Live(item.weight * weight, turn, remaining_actions)
                # A self-switching move ends `runAction` with `switchFlag` set, and
                # Showdown answers that with a fresh switch request -- so the turn stops
                # here, before the rest of the queue and before the residual phase. The
                # branch is handed back for the replacement choice instead of being run on
                # with the wrong Pokemon still standing in the slot.
                if not wiped and turn.self_switch_pending:
                    paused.append(child)
                else:
                    nxt.append(child)
            # Prune as the generation is built, not after. Building it in full first would
            # peak at the cap times the branching factor -- a quarter of a million copied
            # positions for an exact budget, which is enough to exhaust memory.
            if len(nxt) > 2 * budget.max_branches:
                nxt = prune(nxt)

        nxt = prune(nxt)
        finished = prune(finished)
        paused = prune(paused)
        if dropped:
            exact = False
        live = nxt

    if dropped:
        reductions["branches dropped"] = reductions.get("branches dropped", 0) + dropped

    # Renormalise once, at the end: dropping low-probability branches leaves the rest
    # summing to less than one, and rescaling after every generation would compound the
    # rounding.
    total_weight = sum(item.weight for item in finished) + sum(
        item.weight for item in paused
    )
    if total_weight > 0 and abs(total_weight - 1.0) > 1e-12:
        for item in (*finished, *paused):
            item.weight /= total_weight

    out_branches: list[Branch] = []
    unmodelled: set[str] = set()
    for item in finished:
        _residuals(reg, item.turn)
        item.turn.pos.turn += 1
        out_branches.append(
            Branch(probability=item.weight, position=item.turn.pos, events=list(item.turn.events))
        )
        unmodelled |= item.turn.unmodelled

    # A paused branch gets no residuals and no turn increment: the residual phase is behind
    # the interrupt, so it belongs to whatever `resume_turn` produces.
    out_suspended: list[SuspendedTurn] = []
    for item in paused:
        out_suspended.append(
            SuspendedTurn(
                probability=item.weight,
                position=item.turn.pos,
                events=list(item.turn.events),
                _turn=item.turn,
                _remaining=tuple(item.remaining),
            )
        )
        unmodelled |= item.turn.unmodelled

    return TurnResult(
        branches=out_branches,
        exact=exact,
        reductions=reductions,
        unmodelled=tuple(sorted(unmodelled)),
        suspended=tuple(out_suspended),
    )


def _side_wiped(turn: _Turn) -> bool:
    """Whether either side has no conscious Pokemon left."""
    return any(all(mon.fainted for mon in side.pokemon) for side in turn.pos.sides)


def _resort(reg: Regulation, turn: _Turn, remaining: list[QueuedAction]) -> list[QueuedAction]:
    """Re-sorts the remaining queue against the current state.

    Reproduces the generation 8+ behaviour in ``Battle#runAction``: Speed is recomputed and
    the queue re-sorted before each move, so a Speed change earlier in the turn reorders
    what is left.
    """
    if len(remaining) < 2:
        return list(remaining)
    fs = turn.field()
    refreshed: list[QueuedAction] = []
    for action in remaining:
        mon = turn.battler_at(action.side, action.slot)
        if mon is None:
            refreshed.append(action)
            continue
        conditions = frozenset(c.id for c in turn.pos.sides[action.side].side_conditions)
        refreshed.append(replace(action, speed=effective_speed(reg, mon, fs, conditions)))
    groups = order_groups(refreshed, trick_room=turn.pos.field.trick_room)
    if not groups:
        return list(remaining)
    return [remaining[i] for i in groups[0].order]


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

#: An outcome of one action: (probability, resulting state, note about any reduction).
Outcome = tuple[float, _Turn, str]


def _execute(
    reg: Regulation, turn: _Turn, action: QueuedAction, budget: Budget
) -> list[Outcome]:
    if action.kind == "switch":
        _do_switch(reg, turn, action)
        return [(1.0, turn, "")]
    if action.kind == "mega":
        _do_mega(reg, turn, action)
        return [(1.0, turn, "")]
    outcomes = _do_move(reg, turn, action, budget)
    for _weight, state, _note in outcomes:
        state.acted.add((action.side, action.slot))
    return outcomes


def _do_switch(
    reg: Regulation, turn: _Turn, action: QueuedAction, *, run_switch_in: bool = True
) -> None:
    """Brings a benched Pokemon in, mirroring Showdown's party-slot swap.

    ``BattleActions#switchIn`` swaps the two Pokemon's positions in ``side.pokemon``
    whenever there is an outgoing Pokemon at all -- the swap is inside ``if (oldActive)``,
    with no test for whether it fainted -- so an active Pokemon always sits at its own
    active index and ``side.active`` stays ``[0, 1]``. The party index is what ``switch N``
    choice strings count, so this has to match exactly or every later choice refers to the
    wrong Pokemon.

    The fainted case only differs in what happens to the Pokemon leaving: the conscious
    one keeps its status and loses its volatiles, while ``if (oldActive.fainted)
    oldActive.status = ''`` clears the fainted marker as it goes to the bench.
    """
    side = turn.pos.sides[action.side]
    if action.switch_to is None:
        return
    # Resolve by identity, not by index: the first switch of a double switch swaps party
    # positions, so the index recorded when the action was queued may now point at a
    # different Pokemon. Species Clause makes the species id unique within a team.
    incoming = _find_switch_target(side, action)
    if incoming is None or incoming.fainted or incoming.is_active:
        return
    leaving = turn.mon_at(action.side, action.slot)

    if leaving is not None:
        if not leaving.fainted and leaving.ability == "regenerator":
            turn.heal(
                action.side, action.slot, max(1, leaving.maxhp // 3), reason="regenerator"
            )
        leaving.active_index = None
        leaving.boosts = {}
        # Volatiles do not survive a switch out. Unburden's marker goes with them; the
        # ability re-arms only when an item is lost again.
        leaving.volatiles = []
        leaving.last_move = None
        leaving.locked_move = None
        # Volatiles do not survive a switch, and Disable's flag lives on the move slot
        # rather than in the volatile, so it has to be cleared alongside them.
        for move_slot in leaving.moves:
            move_slot.disabled = False
        leaving.newly_switched = False
        if leaving.fainted:
            # `if (oldActive.fainted) oldActive.status = ''`: the fainted marker is
            # cleared on the way to the bench, so a replaced Pokemon reads as statusless.
            leaving.status = None
            leaving.status_counter = None
        # Swap party positions: the incoming Pokemon takes the active slot and the
        # outgoing one takes the index the incoming vacated. Showdown does this whether or
        # not the outgoing Pokemon fainted.
        vacated = incoming.slot
        side.pokemon[action.slot] = incoming
        side.pokemon[vacated] = leaving
        incoming.slot = action.slot
        leaving.slot = vacated
    side.active[action.slot] = action.slot

    incoming.active_index = action.slot
    incoming.newly_switched = True
    for move_slot in incoming.moves:
        move_slot.used = False
    # `tox.onSwitchIn` resets the stage, so a badly poisoned Pokemon coming back in starts
    # its counter over rather than resuming where it left off.
    if incoming.status == "tox":
        incoming.status_counter = 0
    turn.log(f"{turn.name(action.side, action.slot)} <- {incoming.species}")
    # A replacement phase places everyone before any of them sees a hazard or an
    # Intimidate, so the caller can ask for the effects to be run separately and ordered.
    if run_switch_in:
        _on_switch_in(reg, turn, action.side, action.slot)


def _find_switch_target(side, action: QueuedAction):  # noqa: ANN001, ANN202
    """The Pokemon a switch action refers to, by species identity then by index."""
    if action.switch_species:
        for mon in side.pokemon:
            if mon.species == action.switch_species or mon.base_species == action.switch_species:
                return mon
    if action.switch_to is not None and 0 <= action.switch_to < len(side.pokemon):
        return side.pokemon[action.switch_to]
    return None


def _check_white_herb(turn: _Turn) -> None:
    """Consumes White Herb on any holder that currently has a lowered stat.

    Showdown hangs this on `onAnySwitchIn` / `onAnyAfterMove` / `onAnyAfterMega` and on the
    residual, all of which call the same handler, so the faithful shape is one scan over
    every active slot rather than a check tied to whoever just acted.
    """
    for side in range(len(turn.pos.sides)):
        for slot in range(len(turn.pos.sides[side].active)):
            mon = turn.mon_at(side, slot)
            if mon is None or mon.fainted or mon.item != "whiteherb":
                continue
            if not any(value < 0 for value in mon.boosts.values()):
                continue
            mon.boosts = {stat: v for stat, v in mon.boosts.items() if v > 0}
            turn.consume_item(side, slot, reason="whiteherb")
            turn.log(f"{turn.name(side, slot)} stat drops undone (whiteherb)")


def _on_switch_in(reg: Regulation, turn: _Turn, side: int, slot: int) -> None:
    """Entry hazards, then the switch-in ability."""
    mon = turn.mon_at(side, slot)
    if mon is None:
        return
    types = turn.types_of(mon)
    grounded = _grounded(turn, mon)

    for condition in list(turn.pos.sides[side].side_conditions):
        if condition.id == "stealthrock":
            mult = reg.type_effectiveness("Rock", tuple(types))
            turn.deal_damage(
                side, slot, max(1, int(mon.maxhp * mult / 8)), reason="stealthrock"
            )
        elif condition.id == "spikes" and grounded:
            denominator = {1: 8, 2: 6, 3: 4}[min(condition.layers or 1, 3)]
            turn.deal_damage(side, slot, max(1, mon.maxhp // denominator), reason="spikes")
        elif condition.id == "toxicspikes" and grounded:
            if "Poison" in types:
                # A grounded Poison type absorbs the spikes and clears them for the whole
                # side, which is a bigger deal than the status it dodges.
                turn.pos.sides[side].side_conditions = [
                    c for c in turn.pos.sides[side].side_conditions if c.id != "toxicspikes"
                ]
                turn.log(f"{turn.name(side, slot)} absorbed toxicspikes")
            elif "Steel" not in types:
                status = "tox" if (condition.layers or 1) >= 2 else "psn"
                turn.apply_status(side, slot, status, reason="toxicspikes")
        elif condition.id == "stickyweb" and grounded:
            turn.apply_boosts(side, slot, {"spe": -1}, reason="stickyweb")

    if mon.fainted:
        return
    _switch_in_ability(turn, side, slot)
    # `onAnySwitchInPriority: -2`, so this runs after the switch-in abilities: an
    # Intimidate drop is undone before anything else reads the stat.
    _check_white_herb(turn)


def _switch_in_ability(turn: _Turn, side: int, slot: int) -> None:
    mon = turn.mon_at(side, slot)
    if mon is None or mon.fainted:
        return

    weather = WEATHER_ABILITIES.get(mon.ability)
    if weather is not None and turn.pos.field.weather != weather:
        turn.pos.field.weather = weather
        rock = {"sunnyday": "heatrock", "raindance": "damprock",
                "sandstorm": "smoothrock", "snowscape": "icyrock"}.get(weather)
        turn.pos.field.weather_duration = 8 if rock and mon.item == rock else 5
        turn.log(f"{turn.name(side, slot)} set {weather}")

    if mon.ability == "intimidate":
        for foe_slot in range(len(turn.pos.sides[1 - side].active)):
            foe = turn.mon_at(1 - side, foe_slot)
            if foe is None or foe.fainted or foe.ability in INTIMIDATE_PROOF_ABILITIES:
                continue
            turn.apply_boosts(1 - side, foe_slot, {"atk": -1}, reason="intimidate")

    if mon.ability == "hospitality":
        ally_slot = 1 - slot
        ally = turn.mon_at(side, ally_slot)
        if ally is not None and not ally.fainted:
            turn.heal(side, ally_slot, max(1, ally.maxhp // 4), reason="hospitality")


def _do_mega(reg: Regulation, turn: _Turn, action: QueuedAction) -> None:
    mon = turn.mon_at(action.side, action.slot)
    if mon is None or mon.fainted:
        return
    target = reg.mega_target(mon.species, mon.item)
    if target is None:
        return
    species = reg.species[target]
    maxhp_before = mon.maxhp
    mon.species = target
    mon.types = species.types
    mon.is_mega = True
    mon.ability = species.abilities[0].lower().replace(" ", "")
    # Stats are recomputed from the new base stats. HP keeps its absolute value, adjusted
    # by any change in maximum -- no mega changes the HP base stat, so this is a no-op in
    # practice and correct if one ever does.
    refreshed = battler(reg, mon)
    mon.maxhp = int(refreshed.maxhp[0])
    mon.hp = min(mon.maxhp, mon.hp + (mon.maxhp - maxhp_before))
    turn.pos.sides[action.side].mega_used = True
    turn.log(f"{turn.name(action.side, action.slot)} -> {target}")
    # The new ability's switch-in effect fires, which is how a mega Intimidate lands.
    _switch_in_ability(turn, action.side, action.slot)
    # `onAnyAfterMega`: a drop the mega's own Intimidate just caused is undone at once.
    _check_white_herb(turn)


def _do_move(
    reg: Regulation, turn: _Turn, action: QueuedAction, budget: Budget
) -> list[Outcome]:
    assert action.move_id is not None
    move = reg.moves[action.move_id]
    outcomes: list[Outcome] = []
    checks = _can_act(turn, action, budget)

    for index, (act_probability, blocked) in enumerate(checks):
        if act_probability <= 0:
            continue
        state = turn if index == len(checks) - 1 else turn.clone()
        if blocked is not None:
            mon = state.mon_at(action.side, action.slot)
            if blocked == "flinch" and mon is not None:
                mon.volatiles = [v for v in mon.volatiles if v.id != "flinch"]
            if blocked == "confusion":
                state.deal_damage(
                    action.side, action.slot,
                    _confusion_damage(state, action.side, action.slot),
                    reason="confusion",
                )
            state.log(f"{action.label(reg)} did not happen ({blocked})")
            state.move_failed.add((action.side, action.slot))
            outcomes.append((act_probability, state, ""))
            continue
        _stance_change(reg, state, action, move)
        for weight, sub_state, note in _use_move(reg, state, action, move, budget):
            outcomes.append((act_probability * weight, sub_state, note))

    return outcomes or [(1.0, turn, "")]


def _stance_change(
    reg: Regulation, turn: _Turn, action: QueuedAction, move: Move
) -> None:
    """Aegislash takes the forme that matches the move it is about to use.

    Showdown hangs this on `onModifyMove`, which runs while the move is being set up, so
    the new forme's stats apply to that very move. A status move other than King's Shield
    leaves the forme alone.

    The two formes are 50/140 and 140/50 in attack and defence, so getting this wrong is
    not cosmetic: a Shield-forme Aegislash using Iron Head was priced at a third of the
    attack it really has.
    """
    mon = turn.mon_at(action.side, action.slot)
    if mon is None or mon.fainted or mon.ability != "stancechange":
        return
    if mon.transformed:
        return
    base = reg.species.get(mon.species)
    if base is None or base.base_species != "Aegislash":
        return
    if move.category == "Status" and move.id != "kingsshield":
        return
    target = "aegislash" if move.id == "kingsshield" else "aegislashblade"
    if mon.species == target or target not in reg.species:
        return

    species = reg.species[target]
    maxhp_before = mon.maxhp
    mon.species = target
    mon.types = species.types
    refreshed = battler(reg, mon)
    mon.maxhp = int(refreshed.maxhp[0])
    # No Aegislash forme changes the HP base stat, so this is a no-op today and correct if
    # a regulation ever changes one.
    mon.hp = min(mon.maxhp, mon.hp + (mon.maxhp - maxhp_before))
    turn.log(f"{turn.name(action.side, action.slot)} -> {target} (stancechange)")


def _can_act(turn: _Turn, action: QueuedAction, budget: Budget) -> list[tuple[float, str | None]]:
    """(probability, reason it could not act) for the pre-move checks."""
    mon = turn.mon_at(action.side, action.slot)
    if mon is None or mon.fainted:
        return [(1.0, "fainted")]
    if mon.has_volatile("flinch"):
        return [(1.0, "flinch")]
    if mon.status == "slp":
        # `slp.onBeforeMove` decrements the counter when the Pokemon tries to move and
        # cures it at zero, so waking up happens here rather than at end of turn. The
        # duration was rolled when sleep was applied and is part of the state, so this is
        # deterministic.
        mon.status_counter = (mon.status_counter or 0) - 1
        if mon.ability == "earlybird":
            mon.status_counter -= 1
        if mon.status_counter <= 0:
            mon.status = None
            mon.status_counter = None
            turn.log(f"{turn.name(action.side, action.slot)} woke up")
            return [(1.0, None)]
        return [(1.0, "slp")]
    if mon.status == "frz":
        # `time--; if (time <= 0 || randomChance(1, 4))` -- the counter is spent on the
        # attempt to move, and reaching zero thaws regardless of the roll.
        mon.status_counter = (mon.status_counter or FREEZE_COUNTER) - 1
        if mon.status_counter <= 0:
            mon.status = None
            mon.status_counter = None
            turn.log(f"{turn.name(action.side, action.slot)} thawed (counter)")
            return [(1.0, None)]
        if not budget.enumerate_status_checks:
            return [(1.0, "frz")]
        # The two outcomes differ in more than "did it act": one of them is no longer
        # frozen next turn, and `_can_act` returns weights rather than states, so it cannot
        # express that. The weights are right and the cured state is reported as missing.
        turn.unmodelled.add("thaw roll (1 in 4; the cured state is not branched)")
        return [(THAW_CHANCE, None), (1 - THAW_CHANCE, "frz")]

    # Priority-blocking abilities and Psychic Terrain stop the move before it starts --
    # but only a move aimed at the protected side. A self-targeting priority move such as
    # Follow Me is unaffected.
    move = turn.reg.moves[action.move_id] if action.move_id else None
    aimed_at_foes = move is not None and move.target not in (
        "self", "allySide", "allyTeam", "allies", "adjacentAlly", "adjacentAllyOrSelf", "all"
    )
    if action.priority > 0 and aimed_at_foes:
        foe_side = 1 - action.side
        for slot in range(len(turn.pos.sides[foe_side].active)):
            foe = turn.mon_at(foe_side, slot)
            if foe is not None and not foe.fainted and foe.ability in PRIORITY_BLOCKING_ABILITIES:
                return [(1.0, f"ability: {foe.ability}")]
        if turn.pos.field.terrain == "psychicterrain":
            for slot in range(len(turn.pos.sides[foe_side].active)):
                foe = turn.mon_at(foe_side, slot)
                if foe is not None and not foe.fainted and _grounded(turn, foe):
                    return [(1.0, "psychicterrain")]

    outcomes: list[tuple[float, str | None]] = [(1.0, None)]
    if mon.status == "par" and budget.enumerate_status_checks:
        outcomes = [(1 - FULL_PARALYSIS_CHANCE, None), (FULL_PARALYSIS_CHANCE, "par")]
    if mon.has_volatile("confusion") and budget.enumerate_status_checks:
        expanded: list[tuple[float, str | None]] = []
        for weight, reason in outcomes:
            if reason is not None:
                expanded.append((weight, reason))
                continue
            expanded.append((weight * (1 - CONFUSION_SELF_HIT_CHANCE), None))
            expanded.append((weight * CONFUSION_SELF_HIT_CHANCE, "confusion"))
        outcomes = expanded
    return outcomes


def _confusion_damage(turn: _Turn, side: int, slot: int) -> int:
    """A confusion self-hit: 40 base power, physical, typeless, no STAB or effectiveness."""
    mon = turn.battler_at(side, slot)
    if mon is None:
        return 0
    attack = mon.stat("atk")
    defence = np.maximum(mon.stat("def"), 1)
    base = np.trunc(
        np.trunc(np.trunc(np.trunc(2 * mon.level / 5 + 2) * 40 * attack) / defence) / 50
    ).astype(np.int64) + 2
    return int(base[0])


def _use_move(
    reg: Regulation, turn: _Turn, action: QueuedAction, move: Move, budget: Budget
) -> list[Outcome]:
    """The move itself: PP, the Sucker Punch condition, targets, then effects."""
    assert action.move_id is not None
    _spend_pp(turn, action)
    turn.current_actor = (action.side, action.slot)
    mon = turn.mon_at(action.side, action.slot)
    if mon is not None:
        mon.last_move = action.move_id
        if mon.item == "choicescarf":
            turn.add_volatile(action.side, action.slot, "choicelock")
            locked = mon.volatile("choicelock")
            if locked is not None:
                locked.move = action.move_id

    # Stance Change: Aegislash takes its Blade forme to attack and its Shield forme back
    # with King's Shield. The forme decides its stats, so leaving it alone puts every
    # later damage number on the wrong Pokemon.
    if mon is not None and mon.ability == "stancechange":
        wanted = (
            "aegislash"
            if action.move_id == "kingsshield"
            else "aegislashblade"
            if move.category != "Status"
            else None
        )
        if wanted is not None and mon.species != wanted:
            _change_forme(turn, action.side, action.slot, wanted)

    if action.move_id == "lastresort" and mon is not None:
        # Fails until every *other* move the Pokemon knows has been used, and needs at
        # least two moves to begin with.
        others = [m for m in mon.moves if m.id != "lastresort"]
        if len(mon.moves) < 2 or not all(m.used for m in others):
            turn.log(f"{action.label(reg)} failed (other moves not all used)")
            turn.move_failed.add((action.side, action.slot))
            return [(1.0, turn, "")]

    if action.move_id == "suckerpunch":
        # The target must still be *waiting* to attack. One that already moved this turn --
        # commonly a faster Aqua Jet in the same priority bracket -- leaves nothing to
        # counter, and Showdown's `willMove` returns nothing.
        candidates = _resolve_targets(reg, turn, action, move) or []
        pending = [
            slot
            for slot in candidates
            if slot[0] != action.side
            and turn.attacks.get(slot, False)
            and slot not in turn.acted
        ]
        if not pending:
            turn.log(f"{action.label(reg)} failed (nothing left to counter)")
            turn.move_failed.add((action.side, action.slot))
            return [(1.0, turn, "")]

    skip_weather = TWO_TURN_MOVES.get(action.move_id)
    if skip_weather is not None:
        if mon_charged(turn, action):
            # The charge already happened last turn: drop the marker and attack. Showdown's
            # `onTryMove` starts with `if (attacker.removeVolatile(move.id)) return;`, so
            # no boost this time.
            if mon is not None:
                mon.volatiles = [v for v in mon.volatiles if v.id != "twoturnmove"]
        else:
            # The charge-turn boost applies whether or not the charge is skipped: Showdown
            # boosts first and only then checks the weather. Electro Shot in rain both
            # raises Special Attack and attacks in the same turn, so leaving the boost out
            # understates its damage by a whole stage.
            boosts = CHARGE_TURN_BOOSTS.get(action.move_id)
            if boosts:
                turn.apply_boosts(
                    action.side, action.slot, dict(boosts), reason=action.move_id,
                    from_foe=False,
                )
            if turn.pos.field.weather not in skip_weather:
                turn.add_volatile(action.side, action.slot, "twoturnmove")
                turn.log(f"{action.label(reg)} is charging")
                return [(1.0, turn, "")]

    targets = _resolve_targets(reg, turn, action, move)
    no_target_needed = move.target in ("self", "allySide", "allyTeam", "all", "foeSide")
    if not targets and not no_target_needed:
        turn.log(f"{action.label(reg)} had no target")
        turn.move_failed.add((action.side, action.slot))
        return [(1.0, turn, "")]

    if move.category == "Status":
        return _do_status_move(reg, turn, action, move, targets, budget)

    spread = move_hits_multiple(reg, action.move_id, len(targets))
    turn.move_damage_total = 0
    turn.move_connected = False
    branches: list[Outcome] = [(1.0, turn, "")]
    for target in targets:
        expanded: list[Outcome] = []
        for weight, state, note in branches:
            for w2, s2, n2 in _hit_target(reg, state, action, move, target, spread, budget):
                expanded.append((weight * w2, s2, note or n2))
        branches = expanded
    # Once-per-move effects run after every target: recoil and drain are computed from the
    # total damage dealt, and a move's own `self` effect fires once however many Pokemon
    # it hit.
    for _weight, state, _note in branches:
        _after_move(state, action, move)
    return branches


def mon_charged(turn: _Turn, action: QueuedAction) -> bool:
    """Whether the user already spent a turn charging this move."""
    mon = turn.mon_at(action.side, action.slot)
    return mon is not None and mon.has_volatile("twoturnmove")


def _change_forme(turn: _Turn, side: int, slot: int, species_id: str) -> None:
    """Switches a Pokemon to another forme, recomputing its stats.

    HP keeps its absolute value adjusted by any change in maximum, which is how Showdown's
    `formeChange` plus `updateMaxHp` behaves.
    """
    mon = turn.mon_at(side, slot)
    species = turn.reg.species.get(species_id)
    if mon is None or species is None:
        return
    maxhp_before = mon.maxhp
    mon.species = species_id
    mon.types = species.types
    refreshed = battler(turn.reg, mon)
    mon.maxhp = int(refreshed.maxhp[0])
    mon.hp = min(mon.maxhp, max(1, mon.hp + (mon.maxhp - maxhp_before)))
    turn.log(f"{turn.name(side, slot)} -> {species_id}")


def _spend_pp(turn: _Turn, action: QueuedAction) -> None:
    mon = turn.mon_at(action.side, action.slot)
    if mon is None or action.move_id is None:
        return
    slot = mon.move_slot(action.move_id)
    if slot is not None:
        if slot.pp > 0:
            slot.pp -= 1
        slot.used = True


def _resolve_targets(
    reg: Regulation, turn: _Turn, action: QueuedAction, move: Move
) -> list[tuple[int, int]]:
    """Which (side, slot) the move actually hits, after redirection."""
    me = (action.side, action.slot)
    foe_side = 1 - action.side
    slots = range(len(turn.pos.sides[foe_side].active))

    def live(side: int, slot: int) -> bool:
        mon = turn.mon_at(side, slot)
        return mon is not None and not mon.fainted

    kind = move.target
    if kind in ("self", "allySide", "allyTeam", "all", "foeSide"):
        return [me]
    if kind == "allAdjacentFoes":
        return [(foe_side, s) for s in slots if live(foe_side, s)]
    if kind == "allAdjacent":
        out = [(foe_side, s) for s in slots if live(foe_side, s)]
        ally = 1 - action.slot
        if live(action.side, ally):
            out.append((action.side, ally))
        return out
    if kind == "allies":
        return [
            (action.side, s)
            for s in range(len(turn.pos.sides[action.side].active))
            if live(action.side, s)
        ]
    if kind in ("adjacentAlly", "adjacentAllyOrSelf"):
        if action.target is None:
            return [me]
        slot = -action.target - 1
        return [(action.side, slot)] if live(action.side, slot) else []
    if kind == "randomNormal":
        candidates = [(foe_side, s) for s in slots if live(foe_side, s)]
        return candidates[:1]

    if action.target is None:
        chosen = next(((foe_side, s) for s in slots if live(foe_side, s)), None)
        if chosen is None:
            return []
    elif action.target > 0:
        chosen = (foe_side, action.target - 1)
    else:
        chosen = (action.side, -action.target - 1)

    # "If a targeted foe faints, the move is retargeted" (Battle#getTarget). A move whose
    # target has already fallen this turn hits the other one rather than failing.
    if not live(*chosen) and chosen[0] != action.side:
        replacement = next(((foe_side, s) for s in slots if live(foe_side, s)), None)
        if replacement is not None:
            turn.log(f"{action.label(reg)} retargeted to {turn.name(*replacement)}")
            chosen = replacement

    redirected = _redirection_target(turn, action, move, chosen)
    if redirected is not None and redirected != chosen:
        turn.log(f"{action.label(reg)} redirected to {turn.name(*redirected)}")
        chosen = redirected
    return [chosen] if live(*chosen) else []


def _redirection_target(
    turn: _Turn, action: QueuedAction, move: Move, chosen: tuple[int, int]
) -> tuple[int, int] | None:
    """Follow Me / Rage Powder, then the type-drawing abilities."""
    if chosen[0] == action.side:
        return None
    foe_side = chosen[0]
    user = turn.mon_at(action.side, action.slot)
    # Rage Powder is a powder effect, so it does not pull in a move used by a Grass type,
    # a Safety Goggles holder or an Overcoat Pokemon. The exemption is on the move's user,
    # not on the redirector, and Follow Me has no such exemption.
    powder_immune = user is not None and (
        "Grass" in turn.types_of(user)
        or user.item == "safetygoggles"
        or user.ability == "overcoat"
    )
    for slot in range(len(turn.pos.sides[foe_side].active)):
        mon = turn.mon_at(foe_side, slot)
        if mon is None or mon.fainted:
            continue
        drawing = [v for v in REDIRECTION_VOLATILES if mon.has_volatile(v)]
        if not drawing:
            continue
        if drawing == ["ragepowder"] and powder_immune:
            continue
        return (foe_side, slot)
    for slot in range(len(turn.pos.sides[foe_side].active)):
        mon = turn.mon_at(foe_side, slot)
        if mon is None or mon.fainted or (foe_side, slot) == chosen:
            continue
        if REDIRECTION_ABILITIES.get(mon.ability) == move.type:
            return (foe_side, slot)
    return None


def _do_status_move(
    reg: Regulation,
    turn: _Turn,
    action: QueuedAction,
    move: Move,
    targets: list[tuple[int, int]],
    budget: Budget,
) -> list[Outcome]:
    """A status move, which can miss (Hypnosis 60, Will-O-Wisp 85, Thunder Wave 90).

    Protect also blocks status moves that carry the `protect` flag, and the accuracy check
    is per target, so a spread status move against two Pokemon branches twice.
    """
    attacker = turn.battler_at(action.side, action.slot)
    if attacker is None:
        return [(1.0, turn, "")]

    reachable: list[tuple[int, int]] = []
    for target in targets:
        if target != (action.side, action.slot):
            blocked = _blocked_by_protect(turn, action, move, target)
            if blocked is not None:
                turn.log(f"{action.label(reg)} blocked by {blocked}")
                continue
        if _prankster_immune(reg, turn, action, move, target):
            turn.log(f"{turn.name(*target)} immune (prankster vs Dark)")
            continue
        reachable.append(target)

    if not reachable and targets:
        turn.move_failed.add((action.side, action.slot))
        return [(1.0, turn, "")]

    accuracy = 1.0
    for target in reachable:
        defender = turn.battler_at(*target)
        if defender is not None and target != (action.side, action.slot):
            accuracy = min(accuracy, _accuracy(turn, move, attacker, defender))

    if move.raw.get("stallingMove"):
        return _do_protect(reg, turn, action, move, budget)

    # Wide Guard and Quick Guard carry Protect's `onTry() { return !!queue.willAct(); }`
    # gate: a guard that resolves after everything else has moved has nothing to guard.
    if move.id in STALL_BUMPING_MOVES and turn.actions_remaining <= 0:
        turn.log(f"{action.label(reg)} failed (nothing left to act)")
        turn.move_failed.add((action.side, action.slot))
        return [(1.0, turn, "")]

    if not budget.enumerate_accuracy or accuracy >= 1.0 or accuracy <= 0.0:
        if accuracy <= 0.0:
            turn.log(f"{action.label(reg)} missed")
            turn.move_failed.add((action.side, action.slot))
            return [(1.0, turn, "")]
        _apply_status_move(reg, turn, action, move, reachable)
        return [(1.0, turn, "")]

    hit_state = turn.clone()
    _apply_status_move(reg, hit_state, action, move, reachable)
    miss_state = turn
    miss_state.log(f"{action.label(reg)} missed")
    miss_state.move_failed.add((action.side, action.slot))
    return [(accuracy, hit_state, ""), (1 - accuracy, miss_state, "")]


def stall_success_chance(counter: int) -> float:
    """``randomChance(1, counter)``: certain on the first use, a third on the second."""
    return 1.0 / max(1, counter)


def _bump_stall(turn: _Turn, side: int, slot: int) -> None:
    """Raises the Protect counter, as Showdown's `addVolatile('stall')` does.

    The volatile lasts two turns, so a turn spent on anything else lets it lapse and the
    next Protect is certain again. Wide Guard and Quick Guard come through here without
    ever having consulted the counter.
    """
    mon = turn.mon_at(side, slot)
    if mon is None:
        return
    existing = mon.volatile("stall")
    if existing is None:
        mon.volatiles.append(Effect(id="stall", duration=2, counter=3))
    else:
        existing.counter = min((existing.counter or 1) * 3, 729)
        existing.duration = 2


def _do_protect(
    reg: Regulation, turn: _Turn, action: QueuedAction, move: Move, budget: Budget
) -> list[Outcome]:
    """A Protect-family move, which fails more often the more it is repeated.

    The `stall` volatile carries the counter and lasts two turns, so a turn spent on
    anything else resets it.
    """
    me = (action.side, action.slot)
    mon = turn.mon_at(*me)
    if mon is None:
        return [(1.0, turn, "")]

    # Showdown gates Protect on `!!this.queue.willAct()`: a Protect that resolves last in
    # the turn fails outright. In doubles that is the common case for a slow side, so
    # letting it succeed is not a rounding error -- it is the wrong answer to the question
    # the player asked.
    if turn.actions_remaining <= 0:
        turn.log(f"{action.label(reg)} failed (nothing left to act)")
        turn.move_failed.add(me)
        return [(1.0, turn, "")]

    stall = mon.volatile("stall")
    counter = (stall.counter or 1) if stall is not None else 1
    chance = stall_success_chance(counter)

    def succeed(state: _Turn) -> None:
        # One turn, stated here rather than relying on membership of `PROTECT_VOLATILES`:
        # Endure is a stalling move and is *not* in that table, so it used to be added with
        # no duration, miss the unconditional removal, and last the rest of the battle --
        # the Pokemon surviving every lethal hit at 1 HP forever.
        state.add_volatile(*me, move.id, duration=1)
        _bump_stall(state, *me)
        state.log(f"{action.label(reg)} protected (1 in {counter})")

    def fail(state: _Turn) -> None:
        state.log(f"{action.label(reg)} failed (1 in {counter})")
        state.move_failed.add(me)
        target = state.mon_at(*me)
        if target is not None:
            target.volatiles = [v for v in target.volatiles if v.id != "stall"]

    if chance >= 1.0:
        succeed(turn)
        return [(1.0, turn, "")]
    if not budget.enumerate_status_checks:
        # Showdown's pinned policy answers `randomChance(1, counter)` with "no" for any
        # counter above 1, so the deterministic reading is failure.
        fail(turn)
        return [(1.0, turn, "")]

    hit_state = turn.clone()
    succeed(hit_state)
    fail(turn)
    return [(chance, hit_state, ""), (1 - chance, turn, "")]


def _prankster_immune(
    reg: Regulation,
    turn: _Turn,
    action: QueuedAction,
    move: Move,
    target: tuple[int, int],
) -> bool:
    """Showdown's natural Prankster immunity: a foe Dark type ignores the move entirely.

        gen >= 7 && move.pranksterBoosted && pokemon.hasAbility('prankster') &&
            !targets[i].isAlly(pokemon) && !this.dex.getImmunity('prankster', target)

    `pranksterBoosted` is set by the ability's own `onModifyPriority`, so it means exactly
    "a Status move used by a Prankster holder" -- which is what is checked here. Allies are
    exempt, so Prankster Tailwind and screens are unaffected; it is the moves aimed across
    the field that fail.
    """
    if move.category != "Status" or target[0] == action.side:
        return False
    attacker = turn.mon_at(action.side, action.slot)
    if attacker is None or attacker.ability != "prankster":
        return False
    defender = turn.mon_at(*target)
    if defender is None or defender.fainted:
        return False
    return reg.immune_to_effect("prankster", tuple(turn.types_of(defender)))


def _apply_status_move(
    reg: Regulation,
    turn: _Turn,
    action: QueuedAction,
    move: Move,
    targets: list[tuple[int, int]],
) -> None:
    raw = move.raw
    me = (action.side, action.slot)
    #: Set by a move whose own handler deletes `selfSwitch` when it achieved nothing.
    suppress_self_switch = False

    if raw.get("sideCondition"):
        side_condition = str(raw["sideCondition"])
        turn.add_side_condition(
            action.side,
            side_condition,
            duration=_effect_duration(turn, move, side_condition, action.side, action.slot),
        )
        if _duration_is_random(move, side_condition):
            turn.unmodelled.add(
                f"{side_condition} duration (Showdown rolls it; pinned to the low end)"
            )
    if raw.get("weather"):
        turn.pos.field.weather = str(raw["weather"]).lower().replace(" ", "")
        turn.pos.field.weather_duration = 5
        turn.log(f"weather -> {turn.pos.field.weather}")
    if raw.get("terrain"):
        turn.pos.field.terrain = str(raw["terrain"]).lower().replace(" ", "")
        turn.pos.field.terrain_duration = 5
        turn.log(f"terrain -> {turn.pos.field.terrain}")
    if raw.get("pseudoWeather"):
        pid = str(raw["pseudoWeather"]).lower().replace(" ", "")
        already = turn.pos.field.has_pseudo_weather(pid)
        if already and pid in TOGGLING_PSEUDO_WEATHER:
            # Only the room moves switch themselves off when used again
            # (`onFieldRestart`); Gravity and the rest just fail.
            turn.pos.field.pseudo_weather = [
                p for p in turn.pos.field.pseudo_weather if p.id != pid
            ]
            turn.log(f"{pid} ended")
        elif not already:
            turn.pos.field.pseudo_weather.append(Effect(id=pid, duration=5))
            turn.log(f"{pid} started")
        else:
            turn.log(f"{pid} failed (already active)")

    self_effect = raw.get("self") or {}
    if self_effect.get("boosts"):
        turn.apply_boosts(*me, dict(self_effect["boosts"]), reason=move.id, from_foe=False)
    if self_effect.get("volatileStatus"):
        turn.add_volatile(*me, str(self_effect["volatileStatus"]))
    if self_effect.get("sideCondition"):
        turn.add_side_condition(action.side, str(self_effect["sideCondition"]))

    for target in targets:
        own_side = target[0] == action.side
        if raw.get("boosts"):
            turn.apply_boosts(
                *target, dict(raw["boosts"]), reason=move.id, from_foe=not own_side
            )
        if raw.get("status"):
            turn.apply_status(*target, str(raw["status"]), reason=move.id)
        if raw.get("volatileStatus"):
            volatile_id = str(raw["volatileStatus"])
            if volatile_id == "disable":
                # Disable has to record *which* move, and fails outright when the target
                # has not moved, so the generic path cannot express it.
                if not _apply_disable(turn, *target, move=move):
                    turn.log(f"{action.label(reg)} failed (nothing to disable)")
                    turn.move_failed.add((action.side, action.slot))
            else:
                turn.add_volatile(
                    *target,
                    volatile_id,
                    duration=_effect_duration(
                        turn, move, volatile_id, action.side, action.slot
                    ),
                )
        if raw.get("heal"):
            mon = turn.mon_at(*target)
            if mon is not None:
                # Generation 5 and later round this rather than flooring it, so a quarter
                # of 227 heals 57 and not 56.
                turn.heal(*target, _round_fraction(mon.maxhp, raw["heal"]), reason=move.id)

    if move.id in WEATHER_RECOVERY_MOVES:
        ratio = _weather_recovery_fraction(turn.pos.field.weather)
        mon = turn.mon_at(*me)
        if mon is not None:
            turn.heal(*me, _round_fraction(mon.maxhp, ratio), reason=move.id)

    if move.id in ("trick", "switcheroo"):
        for target in targets:
            _swap_items(turn, (action.side, action.slot), target)

    if move.id == "partingshot":
        # Showdown applies the drops in an onHit handler, so they are not in the dumped
        # declarative fields:
        #     const success = this.boost({atk: -1, spa: -1}, target, source);
        #     if (!success && !target.hasAbility('mirrorarmor')) delete move.selfSwitch;
        # So a target that cannot be lowered any further, or that is behind Clear Body,
        # leaves the user standing. Mirror Armor is the exception: it bounces the drops back
        # and the user still leaves.
        landed = False
        for target in targets:
            if turn.apply_boosts(*target, {"atk": -1, "spa": -1}, reason="partingshot"):
                landed = True
            mon = turn.mon_at(*target)
            if mon is not None and mon.ability == "mirrorarmor":
                landed = True
        if not landed:
            suppress_self_switch = True
            turn.log(f"{action.label(reg)} did nothing, so nobody switched")

    if move.id in ("followme", "ragepowder", "spotlight"):
        turn.add_volatile(*me, move.id)
    if move.id == "helpinghand":
        for target in targets:
            turn.add_volatile(*target, "helpinghand")
    if move.id in STALL_BUMPING_MOVES:
        turn.add_side_condition(action.side, move.id, duration=1)
        _bump_stall(turn, action.side, action.slot)

    if move.id == "perishsong":
        # onHitField: everything active on both sides, the user's own side included.
        # Duration 4 counts down at the end of this same turn, so the faint lands three
        # turns later.
        for side in range(2):
            for slot in range(len(turn.pos.sides[side].active)):
                mon = turn.mon_at(side, slot)
                if mon is None or mon.fainted or mon.has_volatile("perishsong"):
                    continue
                if mon.ability == "soundproof" and (side, slot) != me:
                    turn.log(f"{turn.name(side, slot)} immune (soundproof)")
                    continue
                mon.volatiles.append(Effect(id="perishsong", duration=4))
                turn.log(f"{turn.name(side, slot)} perish3")

    if move.raw.get("selfSwitch") and not suppress_self_switch:
        _mark_self_switch(turn, action)
    if move.raw.get("forceSwitch"):
        for target in targets:
            _mark_force_switch(turn, target)

    if move.has_custom_code and move.id not in STATUS_MOVES_FULLY_MODELLED:
        turn.unmodelled.add(f"status move: {move.id}")


def _duration(move: Move, effect_id: str | None = None) -> int | None:
    """How long the effect this move applies lasts, from the regulation dump.

    Keyed by the effect actually being added, because one move can name a volatile, a side
    condition and a slot condition and they need not share a duration. Passing ``None``
    falls back to the move's single declared effect when there is exactly one, which keeps
    the older call sites working.

    This used to be a hand-written list, and the list was the root cause of a whole class
    of bug: the dump carried no `condition` at all, so the function's first branch never
    fired and anything missing from the list became a *permanent* effect. Four were --
    including a bind the target could never escape and an Endure that survived every lethal
    hit for the rest of the battle. The list was also simply wrong about Tailwind, which
    Showdown gives 4 turns and the list gave 5.

    A ``durationCallback`` in the dump means the fixed number is not the whole answer
    (`partiallytrapped` declares 5 and returns 5 or 6; Tailwind declares 4 and returns 6
    under Persistent). The fixed value is the pinned reading, which is what the
    differential test's randomness policy produces, and the caller reports the
    approximation.
    """
    durations = move.raw.get("durations")
    if not isinstance(durations, dict) or not durations:
        return None
    if effect_id is not None:
        entry = durations.get(effect_id)
    elif len(durations) == 1:
        entry = next(iter(durations.values()))
    else:
        entry = None
    if isinstance(entry, dict) and isinstance(entry.get("duration"), int):
        return int(entry["duration"])
    return None


def _duration_is_random(move: Move, effect_id: str) -> bool:
    """Whether Showdown really *rolls* this duration.

    `durationCallback` alone does not mean that. Thirteen of the fourteen effects that have
    one use it to apply an item or ability extension, and only `partiallytrapped` calls
    `this.random`. The dumper now distinguishes the two by calling the callback, so this
    reads `rolled` rather than the mere presence of a function -- which is what had Tailwind
    and every screen reported as "Showdown rolls it".
    """
    entry = _duration_entry(move, effect_id)
    return bool(entry and entry.get("rolled"))


def _duration_entry(move: Move | None, effect_id: str) -> dict | None:
    if move is None:
        return None
    durations = move.raw.get("durations")
    if not isinstance(durations, dict):
        return None
    entry = durations.get(effect_id)
    return entry if isinstance(entry, dict) else None


def _effect_duration(
    turn: _Turn, move: Move, effect_id: str, side: int, slot: int
) -> int | None:
    """How long this effect lasts for *this* user, extensions included.

    `durationCallback(target, source)` reads the source's item and ability, and the dump
    carries what it returns for each -- so Light Clay's screens, Grip Claw's binds,
    Persistent's rooms and Terrain Extender's terrains all follow from the position rather
    than from a number written down here.
    """
    entry = _duration_entry(move, effect_id)
    if entry is None:
        return _duration(move, effect_id)
    mon = turn.mon_at(side, slot)
    if mon is not None:
        by_item = entry.get("byItem")
        if isinstance(by_item, dict) and mon.item and mon.item in by_item:
            return int(by_item[mon.item])
        by_ability = entry.get("byAbility")
        if isinstance(by_ability, dict) and mon.ability in by_ability:
            return int(by_ability[mon.ability])
    base = entry.get("base")
    if isinstance(base, int):
        return base
    declared = entry.get("duration")
    return int(declared) if isinstance(declared, int) else None


def _blocked_by_protect(
    turn: _Turn, action: QueuedAction, move: Move, target: tuple[int, int]
) -> str | None:
    """Whether a Protect-family effect or a guard blocks this hit."""
    if "protect" not in move.flags or bool(move.raw.get("breaksProtect")):
        return None
    # A same-side target is still protected: Earthquake hits its own partner, and a
    # partner behind Protect is not hit.
    side = turn.pos.sides[target[0]]
    from_foe = target[0] != action.side
    if (
        from_foe
        and move.target in ("allAdjacentFoes", "allAdjacent")
        and side.has_side_condition("wideguard")
    ):
        return "wideguard"
    if from_foe and action.priority > 0 and side.has_side_condition("quickguard"):
        return "quickguard"
    mon = turn.mon_at(*target)
    if mon is None:
        return None
    for vid, blocks in PROTECT_VOLATILES.items():
        if not mon.has_volatile(vid):
            continue
        if blocks == "damaging" and move.category == "Status":
            continue
        return vid
    return None


#: Showdown's hit-count distribution for a [2, 5] multi-hit move in generation 5 and
#: later: `sample([2, 2, 3, 3, 4, 5])`.
MULTIHIT_2_5 = ((2, 1 / 3), (3, 1 / 3), (4, 1 / 6), (5, 1 / 6))


def multihit_counts(move: Move, budget: Budget) -> list[tuple[int, float]]:
    """(hit count, probability) for a move, or [(1, 1.0)] when it hits once.

    Under a pinned budget the count is the minimum, which is what Showdown's
    ``multihit='min'`` policy produces -- ``sample`` returns the first element and
    ``random(a, b)`` returns ``a``.
    """
    multihit = move.raw.get("multihit")
    if not multihit:
        return [(1, 1.0)]
    if isinstance(multihit, int):
        return [(multihit, 1.0)]
    low, high = int(multihit[0]), int(multihit[-1])
    if not budget.enumerate_secondary:
        return [(low, 1.0)]
    if (low, high) == (2, 5):
        return list(MULTIHIT_2_5)
    span = high - low + 1
    return [(low + i, 1.0 / span) for i in range(span)]


def _hit_target(
    reg: Regulation,
    turn: _Turn,
    action: QueuedAction,
    move: Move,
    target: tuple[int, int],
    spread: bool,
    budget: Budget,
) -> list[Outcome]:
    """One damaging hit on one target, enumerating accuracy, crit and damage roll."""
    assert action.move_id is not None
    blocked = _blocked_by_protect(turn, action, move, target)
    if blocked is not None:
        turn.log(f"{action.label(reg)} blocked by {blocked}")
        _protect_punish(turn, action, move, blocked)
        return [(1.0, turn, "")]

    attacker = turn.battler_at(action.side, action.slot)
    defender = turn.battler_at(*target)
    if attacker is None or defender is None:
        return [(1.0, turn, "")]

    busted = _bust_disguise(turn, move, target)
    if busted:
        # The hit is absorbed entirely, and the forme change means the next one is not.
        turn.log(f"{action.label(reg)} absorbed by {busted}")
        return [(1.0, turn, "")]

    accuracy = _accuracy(turn, move, attacker, defender)
    crit_p = crit_probability(reg, attacker, defender, action.move_id)

    # With accuracy not enumerated the move hits: that is what the budget means, and it
    # is what Showdown's pinned `accuracy: 'hit'` policy does. Deciding by whether the
    # accuracy exceeds 50% would make Hurricane miss in sun.
    accuracy_branches = (
        [(accuracy, True), (1 - accuracy, False)]
        if budget.enumerate_accuracy and 0.0 < accuracy < 1.0
        else [(1.0, accuracy > 0.0)]
    )
    crit_branches = (
        [(crit_p, True), (1 - crit_p, False)]
        if budget.enumerate_crit and 0.0 < crit_p < 1.0
        else [(1.0, crit_p >= 1.0)]
    )
    rolls = stratified_rolls(budget)
    note = "" if (budget.fixed_roll is not None or budget.damage_rolls >= 16) else (
        "damage rolls stratified"
    )

    attacker_mon = turn.mon_at(action.side, action.slot)
    move_ctx = MoveContext(
        weather=turn.pos.field.weather,
        terrain=turn.pos.field.terrain,
        side_total_fainted=sum(1 for m in turn.pos.sides[action.side].pokemon if m.fainted),
        times_attacked=attacker_mon.times_attacked if attacker_mon else 0,
        target_hurt_this_turn=target in turn.hurt_this_turn,
        damaged_by_target=False,
        # Last turn's result, which is what Stomping Tantrum and Temper Flare read. The
        # within-turn failure set is a different thing and is never set for a move that
        # has not resolved yet.
        previous_move_failed=bool(attacker_mon and attacker_mon.move_last_turn_failed),
    )

    outcomes: list[Outcome] = []
    for acc_weight, hit in accuracy_branches:
        if acc_weight <= 0:
            continue
        if not hit:
            state = turn.clone()
            state.log(f"{action.label(reg)} missed")
            state.move_failed.add((action.side, action.slot))
            outcomes.append((acc_weight, state, note))
            continue
        for crit_weight, crit in crit_branches:
            if crit_weight <= 0:
                continue
            result = calculate(
                reg, attacker, defender, action.move_id, turn.field(),
                defender_side=target[0], spread=spread, crit=crit, move_ctx=move_ctx,
            )
            turn.unmodelled |= set(result.unmodelled)
            if result.immune:
                state = turn.clone()
                state.log(f"{action.label(reg)} had no effect")
                state.move_failed.add((action.side, action.slot))
                _absorb(state, move, target)
                outcomes.append((acc_weight * crit_weight, state, note))
                continue
            # The raw roll, not effective_damage: `deal_damage` owns the cap at the
            # target's HP *and* the Focus Sash / Sturdy consumption that goes with it.
            # Capping here as well would leave the item on the field.
            for roll, roll_weight in rolls:
                for hits, hit_weight in multihit_counts(move, budget):
                    state = turn.clone()
                    total = 0
                    for hit_index in range(hits):
                        mon = state.mon_at(*target)
                        if mon is None or mon.fainted:
                            break
                        if hit_index == 0:
                            amount = int(result.rolls[0, roll])
                        else:
                            # Between the hits of a multi-hit move the defender's state has
                            # moved on -- Stamina has raised its Defence, a berry has
                            # fired, its HP is lower for a fraction-of-HP move -- so the
                            # damage has to be computed again rather than reused.
                            live_attacker = state.battler_at(action.side, action.slot)
                            live_defender = state.battler_at(*target)
                            if live_attacker is None or live_defender is None:
                                break
                            again = calculate(
                                reg, live_attacker, live_defender, action.move_id,
                                state.field(), defender_side=target[0], spread=spread,
                                crit=crit,
                                move_ctx=replace(move_ctx, hit_index=hit_index + 1),
                            )
                            if again.immune:
                                break
                            amount = int(again.rolls[0, roll])
                        dealt_now = state.deal_damage(
                            *target, amount, reason=action.move_id, from_move=True
                        )
                        total += dealt_now
                        _after_hit(
                            state, action, move, target, dealt_now, budget,
                            type_mod=result.type_mod,
                        )
                    if hits > 1:
                        state.log(f"{action.label(reg)} hit {hit_index + 1}x for {total}")
                    weight = acc_weight * crit_weight * roll_weight * hit_weight
                    for extra, expanded in _spread_secondaries(state, action, hits > 1):
                        outcomes.append((weight * extra, expanded, note))
    return outcomes or [(1.0, turn, "")]


#: How many sub-100% secondaries one hit will branch before the rest are collapsed. Each
#: doubles the states for that hit, and the turn's tree is the product over every hit, so
#: four targets each carrying two secondaries would be 256 states from this alone. Two is
#: enough for every real move: a spread move's two targets each carry one.
MAX_BRANCHED_SECONDARIES = 2


def _spread_secondaries(
    state: _Turn, action: QueuedAction, multihit: bool
) -> list[tuple[float, _Turn]]:
    """Fans one resolved hit out over the secondaries it would roll.

    Returns (weight, state) pairs summing to one. With nothing pending that is the state
    itself, which is the overwhelmingly common case and costs one list allocation.

    Each secondary is independent, so the fan-out is a cross product. It is capped: past
    the cap the remaining secondaries are collapsed to "did not happen" and reported, which
    is the old behaviour applied to the tail rather than to everything.
    """
    pending = state.pending_secondaries
    if not pending:
        return [(1.0, state)]
    state.pending_secondaries = []

    branched = pending[:MAX_BRANCHED_SECONDARIES]
    for chance, secondary, _target in pending[MAX_BRANCHED_SECONDARIES:]:
        state.unmodelled.add(
            f"secondary {int(chance * 100)}%: beyond the {MAX_BRANCHED_SECONDARIES} "
            "branched on one hit (not branched)"
        )
        del secondary

    out: list[tuple[float, _Turn]] = [(1.0, state)]
    for chance, secondary, target in branched:
        expanded: list[tuple[float, _Turn]] = []
        for weight, current in out:
            fired = current.clone()
            fired.pending_secondaries = []
            _apply_secondary(fired, action, secondary, target)
            if multihit:
                fired.unmodelled.add(
                    "secondary on a multi-hit move (applied after the last hit)"
                )
            expanded.append((weight * chance, fired))
            expanded.append((weight * (1 - chance), current))
        out = expanded
    return out


#: Abilities that heal a quarter of maximum HP from the type they absorb.
ABSORB_HEAL_ABILITIES: dict[str, str] = {
    "waterabsorb": "Water",
    "dryskin": "Water",
    "voltabsorb": "Electric",
    "eartheater": "Ground",
}

#: Abilities that take a stat boost from the type they draw instead of healing.
ABSORB_BOOST_ABILITIES: dict[str, tuple[str, dict[str, int]]] = {
    "lightningrod": ("Electric", {"spa": 1}),
    "stormdrain": ("Water", {"spa": 1}),
    "motordrive": ("Electric", {"spe": 1}),
    "sapsipper": ("Grass", {"atk": 1}),
    "windrider": ("Flying", {"atk": 1}),
    "wellbakedbody": ("Fire", {"def": 2}),
    "steamengine": ("Fire", {"spe": 6}),
}


def _absorb(turn: _Turn, move: Move, target: tuple[int, int]) -> None:
    """What an immune defender gains from the hit it just shrugged off.

    Treating these purely as immunities loses half the mechanic: Dry Skin heals off a Water
    move and Lightning Rod gains Special Attack from an Electric one, and either can decide
    the next turn.
    """
    mon = turn.mon_at(*target)
    if mon is None or mon.fainted:
        return
    if ABSORB_HEAL_ABILITIES.get(mon.ability) == move.type:
        turn.heal(*target, max(1, mon.maxhp // 4), reason=mon.ability)
        return
    entry = ABSORB_BOOST_ABILITIES.get(mon.ability)
    if entry is not None and entry[0] == move.type:
        turn.apply_boosts(*target, dict(entry[1]), reason=mon.ability, from_foe=False)
        return
    if mon.ability == "flashfire" and move.type == "Fire":
        turn.add_volatile(*target, "flashfire")


#: Forme-guard abilities: the species that has the guard intact, the species it becomes,
#: and whether it only guards physical hits.
FORME_GUARDS: dict[str, tuple[str, str, bool]] = {
    "disguise": ("mimikyu", "mimikyubusted", False),
    "iceface": ("eiscue", "eiscuenoice", True),
}


def _bust_disguise(turn: _Turn, move: Move, target: tuple[int, int]) -> str | None:
    """Absorbs one hit with Disguise or Ice Face, and busts the forme.

    Absorbing without busting would swallow every hit for the rest of the battle. From
    generation 8 Disguise also costs its holder an eighth of its maximum HP.
    """
    mon = turn.mon_at(*target)
    if mon is None or mon.fainted or move.category == "Status":
        return None
    entry = FORME_GUARDS.get(mon.ability)
    if entry is None:
        return None
    intact_species, busted_species, physical_only = entry
    if mon.species != intact_species:
        return None
    if physical_only and move.category != "Physical":
        return None
    mon.species = busted_species
    species = turn.reg.species.get(busted_species)
    if species is not None:
        mon.types = species.types
    turn.log(f"{turn.name(*target)} -> {busted_species}")
    if mon.ability == "disguise":
        turn.deal_damage(*target, max(1, mon.maxhp // 8), reason="disguise")
    return mon.ability


#: Stat changes a charge turn brings with it. Electro Shot and Meteor Beam raise Special
#: Attack while winding up, which is most of the reason to use them.
CHARGE_TURN_BOOSTS: dict[str, dict[str, int]] = {
    "electroshot": {"spa": 1},
    "meteorbeam": {"spa": 1},
    "geomancy": {"spa": 2, "spd": 2, "spe": 2},
}

#: Moves that spend a turn charging before they land. Solar Beam and Electro Shot skip the
#: charge in the right weather.
TWO_TURN_MOVES: dict[str, tuple[str, ...]] = {
    "solarbeam": ("sunnyday", "desolateland"),
    "solarblade": ("sunnyday", "desolateland"),
    "electroshot": ("raindance", "primordialsea"),
    "fly": (),
    "dig": (),
    "dive": (),
    "bounce": (),
    "phantomforce": (),
    "shadowforce": (),
    "skyattack": (),
    "meteorbeam": (),
    "geomancy": (),
}


def _accuracy(turn: _Turn, move: Move, attacker: Battler, defender: Battler) -> float:
    """Hit chance in [0, 1]."""
    if move.accuracy is None or bool(move.raw.get("alwaysHit")):
        return 1.0
    if attacker.ability == "noguard" or defender.ability == "noguard":
        return 1.0
    accuracy = float(move.accuracy)
    weather = turn.pos.field.weather
    if move.id == "blizzard" and weather in ("hail", "snowscape", "snow"):
        return 1.0
    if move.id in ("thunder", "hurricane"):
        if weather in ("raindance", "primordialsea"):
            return 1.0
        if weather in ("sunnyday", "desolateland"):
            accuracy = 50.0
    if attacker.ability == "compoundeyes":
        accuracy *= 1.3
    if attacker.ability == "hustle" and move.category == "Physical":
        accuracy *= 0.8
    if attacker.item == "widelens":
        accuracy *= 1.1
    if defender.item == "brightpowder":
        accuracy *= 0.9
    if not bool(move.raw.get("ignoreEvasion")):
        stages = max(
            -6, min(6, attacker.boosts.get("accuracy", 0) - defender.boosts.get("evasion", 0))
        )
        ratio = (3 + stages) / 3 if stages >= 0 else 3 / (3 - stages)
        accuracy *= ratio
    return max(0.0, min(1.0, accuracy / 100.0))


def _protect_punish(turn: _Turn, action: QueuedAction, move: Move, blocked: str) -> None:
    """Spiky Shield and friends punish the blocked attacker."""
    if "contact" not in move.flags:
        return
    me = (action.side, action.slot)
    attacker = turn.mon_at(*me)
    if attacker is None:
        return
    if blocked == "spikyshield":
        turn.deal_damage(*me, max(1, attacker.maxhp // 8), reason="spikyshield")
    elif blocked == "banefulbunker":
        turn.apply_status(*me, "psn", reason="banefulbunker")
    elif blocked == "burningbulwark":
        turn.apply_status(*me, "brn", reason="burningbulwark")
    elif blocked == "kingsshield":
        turn.apply_boosts(*me, {"atk": -1}, reason="kingsshield")
    elif blocked == "obstruct":
        turn.apply_boosts(*me, {"def": -2}, reason="obstruct")
    elif blocked == "silktrap":
        turn.apply_boosts(*me, {"spe": -1}, reason="silktrap")


def _after_hit(
    turn: _Turn,
    action: QueuedAction,
    move: Move,
    target: tuple[int, int],
    dealt: int,
    budget: Budget,
    type_mod: int = 0,
) -> None:
    """Drain, recoil, item reactions, contact effects and secondaries."""
    raw = move.raw
    me = (action.side, action.slot)
    attacker = turn.mon_at(*me)
    defender = turn.mon_at(*target)

    # Showdown rounds these rather than truncating:
    #   clampIntRange(Math.round(damageDealt * recoil[0] / recoil[1]), 1)
    turn.move_damage_total += dealt
    turn.move_connected = True

    # A damaging move can carry a volatile or a status outright, not only as a chance-based
    # secondary: Infestation's trap, Salt Cure, Nuzzle's paralysis.
    if defender is not None and not defender.fainted:
        if raw.get("volatileStatus"):
            vid = str(raw["volatileStatus"])
            turn.add_volatile(
                *target,
                vid,
                duration=_effect_duration(turn, move, vid, action.side, action.slot),
            )
            if _duration_is_random(move, vid):
                turn.unmodelled.add(
                    f"{vid} duration (Showdown rolls it; pinned to the low end)"
                )
            # A trap ends when whoever applied it leaves, so record who that was.
            applied = defender.volatile(vid)
            if applied is not None and vid == "partiallytrapped":
                applied.source_slot = f"{me[0]}{me[1]}"
        if raw.get("status"):
            turn.apply_status(*target, str(raw["status"]), reason=move.id)

    # Spicy Spray (Mega Scovillain) is an `onDamagingHit` handler too, but it is *not*
    # contact-gated and it does not roll:
    #     onDamagingHit(damage, target, source, move) { source.trySetStatus('brn', target); }
    # So any damaging hit burns the attacker, Moonblast included. It was declared modelled
    # because it changes no damage number, which is true of the calculator and was wrong of
    # the resolver -- the ability was silently doing nothing. Unlike Static and Flame Body,
    # which are 1-in-3 and are reported rather than branched, this one is certain and can
    # simply be applied.
    if (
        defender is not None
        and attacker is not None
        and dealt > 0
        and defender.ability == "spicyspray"
    ):
        turn.apply_status(*me, "brn", reason="spicyspray")

    # Throat Chop adds its own condition from a 100%-chance `secondary.onHit`, so there is
    # nothing declarative in the dump to drive it. Two turns, which the dump now carries
    # because the duration collector reads a condition named after the move itself.
    if move.id == "throatchop" and defender is not None and not defender.fainted and dealt > 0:
        turn.add_volatile(*target, "throatchop", duration=_duration(move, "throatchop"))
        turn.log(f"{turn.name(*target)} cannot use sound moves (throatchop)")

    # The contact effects are `onDamagingHit` handlers, which Showdown runs from `damage()`
    # -- before the faint is processed. So Rough Skin still hurts the attacker when the
    # Pokemon holding it is knocked out by that very hit, and gating on survival loses the
    # chip damage that often decides the next turn.
    # Cursed Body: `onDamagingHit` with `randomChance(3, 10)`, not gated on contact and
    # not gated on the target surviving. 43 of the 394 tournament teams carry it, which
    # makes it the most common ability the resolver did not model.
    if (
        defender is not None
        and defender.ability == "cursedbody"
        and attacker is not None
        and dealt > 0
        and not attacker.has_volatile("disable")
    ):
        if budget.enumerate_secondary:
            turn.pending_secondaries.append((0.3, {"disable": True}, me))
        elif not budget.pinned_policy:
            # Same reasoning as a secondary: under the pinned policy `randomChance(3, 10)`
            # is answered with no, so not applying it is exact rather than approximate.
            turn.unmodelled.add("cursedbody (30% disable, not branched)")

    if defender is not None and "contact" in move.flags and dealt > 0:
        if defender.ability in ("roughskin", "ironbarbs") and attacker is not None:
            turn.deal_damage(*me, max(1, attacker.maxhp // 8), reason=defender.ability)
        if defender.item == "rockyhelmet" and attacker is not None:
            turn.deal_damage(*me, max(1, attacker.maxhp // 6), reason="rockyhelmet")
        if defender.ability in ("static", "flamebody", "effectspore", "poisonpoint", "cutecharm"):
            turn.unmodelled.add(f"contact ability: {defender.ability}")

    # A resist berry is eaten only by a hit it actually weakened. Occa Berry's handler is
    # `if (move.type === 'Fire' && typeMod > 0) { if (target.eatItem()) ... }`, so a Fire
    # move that is *not* super effective leaves the berry alone. Chilan Berry is the one
    # exception: it halves Normal regardless of effectiveness.
    if defender is not None and not defender.fainted and not turn.berries_blocked(target[0]):
        berry_type = RESIST_BERRIES.get(defender.item or "")
        weakened = berry_type == move.type and (berry_type == "Normal" or type_mod > 0)
        if weakened:
            turn.consume_item(*target, reason=defender.item)

    # Knock Off removes what it hit; Thief and Covet take it when the attacker has nothing.
    if defender is not None and not defender.fainted and defender.item:
        # A stone refuses to leave the species it belongs to, and only that species:
        # `onTakeItem` returns `!item.megaStone?.[source.baseSpecies.baseSpecies]`.
        removable = item_is_removable(turn.reg, defender.species, defender.item)
        if move.id == "knockoff" and removable:
            turn.consume_item(*target, reason="knockoff")
        elif (
            move.id in ("thief", "covet")
            and removable
            and attacker is not None
            and attacker.item is None
        ):
            stolen = defender.item
            turn.consume_item(*target, reason=move.id)
            attacker.item = stolen
            turn.log(f"{turn.name(*me)} stole {stolen}")

    if attacker is not None and attacker.ability == "poisontouch" and "contact" in move.flags:
        turn.unmodelled.add("ability: poisontouch (30% poison not branched)")

    for secondary in raw.get("secondaries") or []:
        if attacker is not None and attacker.ability == "sheerforce":
            continue  # Sheer Force trades secondaries for power.
        chance = float(secondary.get("chance", 100)) / 100.0
        if chance >= 1.0:
            _apply_secondary(turn, action, secondary, target)
            continue
        if not budget.enumerate_secondary:
            # Collapsed to "it did not happen", which is what the cheap budgets buy. Said
            # out loud, because under such a budget a Rock Slide never flinches -- except
            # under the pinned policy, where "it did not happen" is not an approximation at
            # all: the oracle answers every secondary roll with no, so the collapse is
            # exact. Declaring it there would flag turns that are right, which lowers the
            # differential test's silent rate without improving anything.
            if not budget.pinned_policy:
                turn.unmodelled.add(
                    f"secondary {int(chance * 100)}%: {move.id} (not branched)"
                )
            continue
        # A branch, and this function holds one state. The hit loop fans it out.
        turn.pending_secondaries.append((chance, dict(secondary), target))

    if raw.get("forceSwitch"):
        _mark_force_switch(turn, target)

    _on_being_hit(turn, move, target, action.side)


def _after_move(turn: _Turn, action: QueuedAction, move: Move) -> None:
    """Effects that fire once per move use, not once per target.

    Showdown computes recoil from ``damageDealt`` -- the total across every target -- and
    Life Orb's recoil and a move's own ``self`` effect are single ``onAfterMoveSecondary``
    handlers. Running them per target multiplies them by the number of Pokemon hit.
    """
    raw = move.raw
    me = (action.side, action.slot)
    attacker = turn.mon_at(*me)
    total = turn.move_damage_total

    if raw.get("drain") and total > 0:
        turn.heal(*me, _round_fraction(total, raw["drain"]), reason="drain")
    if raw.get("recoil") and total > 0 and attacker is not None and attacker.ability != "rockhead":
        turn.deal_damage(*me, _round_fraction(total, raw["recoil"]), reason="recoil")
    if attacker is not None and attacker.item == "lifeorb" and total > 0:
        turn.deal_damage(*me, max(1, attacker.maxhp // 10), reason="lifeorb")
    # Shell Bell heals an eighth of the move's *total* damage, once, and truncates -- so a
    # move dealing under eight damage heals nothing.
    if attacker is not None and attacker.item == "shellbell" and total >= 8:
        turn.heal(*me, total // 8, reason="shellbell")

    self_effect = raw.get("self") or {}
    if attacker is not None and not attacker.fainted and turn.move_connected:
        if self_effect.get("boosts"):
            turn.apply_boosts(*me, dict(self_effect["boosts"]), reason=move.id, from_foe=False)
        if self_effect.get("volatileStatus"):
            turn.add_volatile(*me, str(self_effect["volatileStatus"]))
        # `selfBoost` is a separate field from `self`, applied once after the move
        # succeeds: Clanging Scales, Clangorous Soul and Scale Shot use it.
        self_boost = raw.get("selfBoost") or {}
        if self_boost.get("boosts"):
            turn.apply_boosts(*me, dict(self_boost["boosts"]), reason=move.id, from_foe=False)

    # `else if (move.selfSwitch && source.hp && !source.volatiles['commanded'])`, reached
    # only when the move `didAnything`: a U-turn into a Ghost type, or one that missed,
    # leaves its user in place.
    if raw.get("selfSwitch") and turn.move_connected:
        _mark_self_switch(turn, action)

    # `onAnyAfterMove`: every holder on the field is checked, not just the attacker.
    _check_white_herb(turn)

    turn.move_damage_total = 0
    turn.move_connected = False


#: Abilities that react to their holder taking a damaging hit, as
#: (stat changes, the move types that trigger it -- empty means any type).
ON_HIT_ABILITIES: dict[str, tuple[dict[str, int], tuple[str, ...]]] = {
    "stamina": ({"def": 1}, ()),
    "weakarmor": ({"def": -1, "spe": 2}, ()),
    "justified": ({"atk": 1}, ("Dark",)),
    "rattled": ({"spe": 1}, ("Bug", "Dark", "Ghost")),
    "steamengine": ({"spe": 6}, ("Fire", "Water")),
    "watercompaction": ({"def": 2}, ("Water",)),
}

#: On-hit abilities whose effect depends on how much HP was lost or on a chance, which the
#: resolver does not model and therefore reports.
ON_HIT_ABILITIES_UNMODELLED = frozenset({"angerpoint", "berserk", "angershell", "cursedbody"})


def _on_being_hit(
    turn: _Turn, move: Move, target: tuple[int, int], attacker_side: int
) -> None:
    """Abilities that trigger on the defender taking a hit."""
    defender = turn.mon_at(*target)
    if defender is None or defender.fainted:
        return
    if (
        defender.ability == "toxicdebris"
        and move.category == "Physical"
        and target[0] != attacker_side
    ):
        existing = turn.pos.sides[attacker_side].side_condition("toxicspikes")
        if existing is None or (existing.layers or 1) < 2:
            turn.add_side_condition(attacker_side, "toxicspikes")

    entry = ON_HIT_ABILITIES.get(defender.ability)
    if entry is not None:
        boosts, types = entry
        if not types or move.type in types:
            turn.apply_boosts(
                *target, dict(boosts), reason=defender.ability, from_foe=False
            )
    if defender.ability in ON_HIT_ABILITIES_UNMODELLED:
        turn.unmodelled.add(f"on-hit ability: {defender.ability}")


#: The recovery moves whose amount depends on the weather.
WEATHER_RECOVERY_MOVES = frozenset({"moonlight", "synthesis", "morningsun"})


def _weather_recovery_fraction(weather: str | None) -> tuple[int, int]:
    if weather in ("sunnyday", "desolateland"):
        return (2, 3)
    if weather is None:
        return (1, 2)
    return (1, 4)


def _swap_items(turn: _Turn, a: tuple[int, int], b: tuple[int, int]) -> None:
    """Trick and Switcheroo exchange held items.

    A mega stone cannot be moved, and neither can an item its holder is bound to, so the
    swap is refused in those cases rather than producing an impossible state.
    """
    first, second = turn.mon_at(*a), turn.mon_at(*b)
    if first is None or second is None:
        return
    for mon in (first, second):
        if mon.item and turn.reg.items.get(mon.item, None) is not None:
            entry = turn.reg.items[mon.item]
            if entry.mega_stone:
                turn.log("item swap refused (mega stone)")
                return
    first.item, second.item = second.item, first.item
    turn.log(f"{turn.name(*a)} and {turn.name(*b)} swapped items")
    for side_slot, mon in ((a, first), (b, second)):
        if mon.ability == "unburden" and mon.item is None and not mon.has_volatile("unburden"):
            mon.volatiles.append(Effect(id="unburden"))
        turn.check_berry(*side_slot)


def _mark_self_switch(turn: _Turn, action: QueuedAction) -> None:
    """Records that the user of a self-switching move has to be replaced.

    Which Pokemon comes in is the player's choice, so the resolver does not pick one. The
    mark is what suspends the turn: `_run_queue` sees it and hands the branch back with its
    remaining queue, and `resume_turn` continues once the choice is made.

    A Pokemon with an empty bench is not marked at all -- Showdown's `switchFlag` has
    nothing to answer it with, and the move simply leaves it in place.
    """
    mon = turn.mon_at(action.side, action.slot)
    if mon is None or mon.fainted:
        return
    bench = [p for p in turn.pos.sides[action.side].pokemon if not p.fainted and not p.is_active]
    if not bench:
        return
    turn.add_volatile(action.side, action.slot, "pendingselfswitch")
    turn.self_switch_pending = True
    turn.log(f"{turn.name(action.side, action.slot)} must switch out")


def _mark_force_switch(turn: _Turn, target: tuple[int, int]) -> None:
    """Records that a forced switch (Roar, Whirlwind, Dragon Tail) is pending.

    Showdown drags in a random replacement immediately. That is a genuine coin flip over
    the bench, so it is recorded rather than resolved to one arbitrary Pokemon.
    """
    mon = turn.mon_at(*target)
    if mon is None or mon.fainted:
        return
    bench = [p for p in turn.pos.sides[target[0]].pokemon if not p.fainted and not p.is_active]
    if not bench:
        return
    turn.add_volatile(*target, "pendingforceswitch")
    turn.log(f"{turn.name(*target)} is forced out")
    turn.unmodelled.add("forceSwitch (replacement is drawn at random)")


def _round_fraction(amount: int, ratio: list[int] | tuple[int, int]) -> int:
    """``clampIntRange(round(amount * num / den), 1)``, as Showdown does for recoil."""
    numerator, denominator = int(ratio[0]), int(ratio[1])
    # Python rounds halves to even; JavaScript's Math.round rounds halves up.
    scaled = amount * numerator
    value = (scaled * 2 + denominator) // (2 * denominator)
    return max(1, int(value))


def _apply_disable(turn: _Turn, side: int, slot: int, move: Move | None = None) -> bool:
    """Disables the target's last move, as Showdown's `disable` condition does.

    Fails when the target has not moved yet, or when the move it last used has no PP left
    -- both are `return false` in `onStart`, which means no volatile at all rather than a
    volatile that disables nothing.

    The duration is decremented immediately when the target still owes an action this turn
    or when it is the Pokemon whose own move triggered this, which covers Cursed Body and
    the ordinary Disable; five turns is left for disabling something that has already acted.
    """
    mon = turn.mon_at(side, slot)
    if mon is None or mon.fainted or mon.last_move is None:
        return False
    if mon.has_volatile("disable"):
        return False
    slot_for_move = next((m for m in mon.moves if m.id == mon.last_move), None)
    if slot_for_move is None or slot_for_move.pp <= 0:
        return False

    duration = _duration(move, "disable") if move is not None else 5
    if duration is None:
        duration = 5
    if (side, slot) not in turn.acted or (side, slot) == turn.current_actor:
        duration -= 1

    mon.volatiles.append(Effect(id="disable", duration=duration, move=mon.last_move))
    slot_for_move.disabled = True
    turn.log(f"{turn.name(side, slot)} cannot use {mon.last_move} (disable)")
    return True


def _apply_secondary(
    turn: _Turn, action: QueuedAction, secondary: dict, target: tuple[int, int]
) -> None:
    if secondary.get("disable"):
        # Cursed Body, pushed through the secondary fan-out because it is the same shape:
        # a chance whose consequence outlives the turn.
        _apply_disable(turn, *target)
        return
    if secondary.get("status"):
        turn.apply_status(*target, str(secondary["status"]), reason="secondary")
    if secondary.get("volatileStatus"):
        turn.add_volatile(*target, str(secondary["volatileStatus"]))
    if secondary.get("boosts"):
        turn.apply_boosts(*target, dict(secondary["boosts"]), reason="secondary")
    self_boosts = (secondary.get("self") or {}).get("boosts")
    if self_boosts:
        turn.apply_boosts(
            action.side, action.slot, dict(self_boosts), reason="secondary", from_foe=False
        )


# ---------------------------------------------------------------------------
# End of turn
# ---------------------------------------------------------------------------


def _trapper_gone(turn: _Turn, trap: Effect) -> bool:
    """Whether the Pokemon that applied a trap has left the field.

    A trap whose source is gone ends without dealing damage, so keeping it going costs the
    trapped Pokemon an eighth of its HP a turn that it should not lose.
    """
    if not trap.source_slot or len(trap.source_slot) < 2:
        return False
    try:
        side, slot = int(trap.source_slot[0]), int(trap.source_slot[1])
    except ValueError:
        return False
    source = turn.mon_at(side, slot)
    return source is None or source.fainted


#: Volatiles that record a replacement the resolver deliberately did not choose.
PENDING_REPLACEMENT_VOLATILES = ("pendingselfswitch", "pendingforceswitch")


def self_switches_needed(pos: Position) -> tuple[tuple[bool, ...], ...]:
    """Per side, per active slot, whether a self-switching move is waiting on a choice.

    Narrower than `replacements_needed` on purpose. A faint is answered after the turn and
    a forced switch is a random drag, but a self-switch interrupts the turn *now*, so only
    these slots may be filled by `resume_turn`.
    """
    out: list[tuple[bool, ...]] = []
    for side in pos.sides:
        bench = sum(1 for mon in side.pokemon if not mon.fainted and not mon.is_active)
        flags: list[bool] = []
        for party_index in side.active:
            mon = side.pokemon[party_index] if party_index is not None else None
            flags.append(
                bench > 0
                and mon is not None
                and not mon.fainted
                and mon.has_volatile("pendingselfswitch")
            )
        out.append(tuple(flags))
    return tuple(out)


def resume_turn(
    reg: Regulation, paused: SuspendedTurn, choices: list[SideAction]
) -> TurnResult:
    """Finishes a turn that stopped at a mid-turn replacement request.

    ``choices`` is one :class:`SideAction` per side, shaped like the post-turn replacement
    phase: a switch for each slot that owes one, a pass everywhere else. The replacement
    enters and then the rest of the queue runs, the residual phase included -- which is the
    whole point, since Showdown puts the interrupt in front of both.

    Returns a :class:`TurnResult` because resuming can suspend again: two U-turns on the
    same side produce two separate requests, exactly as Showdown does.

    The suspension is not consumed. One pause is resumed once per candidate replacement
    when the caller is choosing between them, so the state is copied here.
    """
    if paused._turn is None:
        raise ValueError("this SuspendedTurn carries no continuation state")
    turn = paused._turn.clone()
    owed = self_switches_needed(turn.pos)
    unmodelled: set[str] = set()

    placed: list[tuple[int, int, np.ndarray]] = []
    for side_index, side_action in enumerate(choices):
        for slot_action in side_action.slots:
            if isinstance(slot_action, PassAction):
                if owed[side_index][slot_action.slot]:
                    unmodelled.add(
                        f"self-switch replacement owed at "
                        f"p{side_index + 1}[{slot_action.slot}] but none was chosen"
                    )
                continue
            if not isinstance(slot_action, SwitchAction):
                unmodelled.add(f"a replacement only takes switches, got {slot_action!r}")
                continue
            slot = slot_action.slot
            if not owed[side_index][slot]:
                # Only the interrupted slot may move. Filling any other one here would be
                # a free switch that the turn never offered.
                unmodelled.add(
                    f"p{side_index + 1}[{slot}] does not owe a self-switch replacement"
                )
                continue
            outgoing = turn.mon_at(side_index, slot)
            if outgoing is not None:
                outgoing.volatiles = [
                    v for v in outgoing.volatiles if v.id != "pendingselfswitch"
                ]
            queued = QueuedAction(
                side=side_index,
                slot=slot,
                kind="switch",
                order=103,
                priority=0,
                fractional=0.0,
                speed=np.zeros(1, dtype=np.int64),
                switch_to=slot_action.party_index,
                switch_species=slot_action.species,
            )
            _do_switch(reg, turn, queued, run_switch_in=False)
            incoming = turn.battler_at(side_index, slot)
            speed = (
                effective_speed(
                    reg,
                    incoming,
                    turn.field(),
                    frozenset(c.id for c in turn.pos.sides[side_index].side_conditions),
                )
                if incoming is not None
                else np.zeros(1, dtype=np.int64)
            )
            placed.append((side_index, slot, speed))

    # `runSwitch` is order 101 sorted on speed, fastest first, so a fast replacement takes
    # the hazards and fires its ability before a slow one. Same rule as the post-turn phase.
    placed.sort(key=lambda entry: (-int(entry[2][0]), entry[0], entry[1]))
    for side_index, slot, _speed in placed:
        _on_switch_in(reg, turn, side_index, slot)

    turn.self_switch_pending = any(any(f) for f in self_switches_needed(turn.pos))
    result = _run_queue(reg, [_Live(1.0, turn, list(paused._remaining))], turn.budget)
    if unmodelled:
        result.unmodelled = tuple(sorted(set(result.unmodelled) | unmodelled))
    for branch in result.branches:
        branch.probability *= paused.probability
    for pause in result.suspended:
        pause.probability *= paused.probability
    return result


@dataclass
class ReplacementResult:
    """The position after a replacement phase."""

    position: Position
    events: list[str] = field(default_factory=list)
    unmodelled: tuple[str, ...] = ()


def replacements_needed(pos: Position) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Per side, per active slot, whether the player owes a replacement.

    Three causes, and all of them leave the slot needing an answer before the next turn:
    the Pokemon fainted, it used a self-switching move, or it was forced out. A side with
    nothing left on the bench owes nothing -- the slot simply stays empty.
    """
    out: list[tuple[bool, ...]] = []
    for side in pos.sides:
        bench = sum(1 for mon in side.pokemon if not mon.fainted and not mon.is_active)
        flags: list[bool] = []
        for slot, party_index in enumerate(side.active):
            del slot
            mon = side.pokemon[party_index] if party_index is not None else None
            if mon is None:
                flags.append(bench > 0)
                continue
            pending = any(mon.has_volatile(v) for v in PENDING_REPLACEMENT_VOLATILES)
            flags.append(bool(bench) and (mon.fainted or pending))
        out.append(tuple(flags))
    return out[0], out[1]


def resolve_replacements(
    reg: Regulation, pos: Position, choices: list[SideAction]
) -> ReplacementResult:
    """Applies both sides' replacement choices.

    ``choices`` is one :class:`SideAction` per side, as produced by
    ``actions.switch_actions_after_faint``; a slot that owes nothing carries a
    ``PassAction``. A slot that owes a replacement and is given a pass is left alone and
    reported, because silently choosing for the player is the thing this module exists not
    to do.
    """
    state = _Turn(reg, pos.copy(), Budget.deterministic(0), {})
    needed = replacements_needed(state.pos)
    unmodelled: set[str] = set()

    placed: list[tuple[int, int, np.ndarray]] = []
    for side_index, side_action in enumerate(choices):
        for slot_action in side_action.slots:
            if isinstance(slot_action, PassAction):
                if needed[side_index][slot_action.slot]:
                    unmodelled.add(
                        f"replacement owed at p{side_index + 1}[{slot_action.slot}] "
                        "but none was chosen"
                    )
                continue
            if not isinstance(slot_action, SwitchAction):
                unmodelled.add(
                    f"a replacement phase only takes switches, got {slot_action!r}"
                )
                continue
            slot = slot_action.slot
            outgoing = state.mon_at(side_index, slot)
            for volatile in PENDING_REPLACEMENT_VOLATILES:
                if outgoing is not None and outgoing.has_volatile(volatile):
                    outgoing.volatiles = [
                        v for v in outgoing.volatiles if v.id != volatile
                    ]
            queued = QueuedAction(
                side=side_index,
                slot=slot,
                kind="switch",
                order=103,
                priority=0,
                fractional=0.0,
                speed=np.zeros(1, dtype=np.int64),
                switch_to=slot_action.party_index,
                switch_species=slot_action.species,
            )
            _do_switch(reg, state, queued, run_switch_in=False)
            incoming = state.battler_at(side_index, slot)
            speed = (
                effective_speed(
                    reg,
                    incoming,
                    state.field(),
                    frozenset(c.id for c in state.pos.sides[side_index].side_conditions),
                )
                if incoming is not None
                else np.zeros(1, dtype=np.int64)
            )
            placed.append((side_index, slot, speed))

    # `runSwitch` carries order 101 and is sorted on speed, fastest first, so a fast
    # replacement eats the hazards and fires its ability before a slow one.
    placed.sort(key=lambda entry: (-int(entry[2][0]), entry[0], entry[1]))
    for side_index, slot, _speed in placed:
        _on_switch_in(reg, state, side_index, slot)

    for side_index, side in enumerate(state.pos.sides):
        if all(mon.fainted for mon in side.pokemon):
            state.pos.ended = True
            state.pos.winner = state.pos.sides[1 - side_index].id

    return ReplacementResult(
        position=state.pos,
        events=list(state.events),
        unmodelled=tuple(sorted(unmodelled | state.unmodelled)),
    )


def resume_alternatives(
    reg: Regulation, paused: SuspendedTurn
) -> tuple[int | None, list[tuple[SideAction, TurnResult]]]:
    """Every replacement the interrupted side could send in, and the turn each produces.

    Showdown checks `switchFlag` after each action, so at most one side is ever asked at a
    time and the choice belongs to one player. If both sides somehow owe one at once the
    choices interact and it is no longer a single-player decision, so that is reported
    rather than quietly treated as one.
    """
    owed = self_switches_needed(paused.position)
    sides = [i for i, flags in enumerate(owed) if any(flags)]
    if not sides:
        return None, []
    chooser = sides[0]
    options = switch_actions_after_faint(
        reg, paused.position, chooser, list(owed[chooser])
    )
    other = 1 - chooser
    passes = SideAction(
        slots=tuple(
            PassAction(slot=i) for i in range(len(paused.position.sides[other].active))
        )
    )
    out: list[tuple[SideAction, TurnResult]] = []
    for option in options:
        choices = [option, passes] if chooser == 0 else [passes, option]
        resumed = resume_turn(reg, paused, choices)
        if len(sides) > 1:
            resumed.unmodelled = tuple(
                sorted({*resumed.unmodelled, "simultaneous mid-turn replacements"})
            )
        out.append((option, resumed))
    return chooser, out


@dataclass
class LeafRef:
    """One evaluated position."""

    index: int


@dataclass
class Average:
    """What chance decides: the weighted mean of its parts, normalised by their weight."""

    parts: list[tuple[float, Fold]] = field(default_factory=list)


@dataclass
class BestOf:
    """What a player decides: the option that side likes most.

    Side 0 is the maximiser the payoff matrix is written for, so side 1 choosing means
    taking the value it likes least.
    """

    chooser: int
    options: list[Fold] = field(default_factory=list)


Fold = LeafRef | Average | BestOf


def fold_value(node: Fold, values: Sequence[float]) -> float:
    """Collapses a fold tree against one value per leaf position."""
    if isinstance(node, LeafRef):
        return float(values[node.index])
    if isinstance(node, BestOf):
        scored = [fold_value(option, values) for option in node.options]
        if not scored:
            return 0.0
        return max(scored) if node.chooser == 0 else min(scored)
    total = sum(weight for weight, _ in node.parts)
    if total <= 0:
        return 0.0
    return sum(weight * fold_value(part, values) for weight, part in node.parts) / total


@dataclass
class TurnLeaves:
    """A turn's leaf positions and the fold that turns their values into the turn's value."""

    positions: list[Position]
    root: Fold
    unmodelled: tuple[str, ...] = ()

    def value(self, values: Sequence[float]) -> float:
        return fold_value(self.root, values)

    def shifted(self, offset: int) -> Fold:
        """The fold re-indexed for a caller that appended these positions to a larger batch.

        The search evaluates every leaf in a node in one forward pass, so each cell's
        positions land at an offset in a shared list.
        """
        return _shift(self.root, offset)


def turn_leaves(reg: Regulation, result: TurnResult, *, depth: int = 0) -> TurnLeaves:
    """Flattens a turn -- suspensions included -- into leaf positions plus a fold.

    Recurses because resuming can be interrupted again: two U-turns on the same side are two
    separate requests, exactly as Showdown issues them.
    """
    positions: list[Position] = []
    parts: list[tuple[float, Fold]] = []
    unmodelled: set[str] = set(result.unmodelled)

    def add_leaf(position: Position) -> LeafRef:
        positions.append(position)
        return LeafRef(index=len(positions) - 1)

    for branch in result.branches:
        parts.append((branch.probability, add_leaf(branch.position)))

    for pause in result.suspended:
        # Four Pokemon a side means a turn cannot interrupt itself indefinitely. The guard
        # is against a bug turning into unbounded recursion, and it reports rather than
        # hides the position it stopped at.
        if depth >= 4:
            unmodelled.add("more than four mid-turn replacements in one turn")
            parts.append((pause.probability, add_leaf(pause.position)))
            continue
        chooser, alternatives = resume_alternatives(reg, pause)
        if chooser is None or not alternatives:
            unmodelled.add("a suspended turn offered no replacement")
            parts.append((pause.probability, add_leaf(pause.position)))
            continue
        options: list[Fold] = []
        for _option, resumed in alternatives:
            sub = turn_leaves(reg, resumed, depth=depth + 1)
            offset = len(positions)
            positions.extend(sub.positions)
            options.append(_shift(sub.root, offset))
            unmodelled |= set(sub.unmodelled)
        parts.append((pause.probability, BestOf(chooser=chooser, options=options)))

    return TurnLeaves(
        positions=positions, root=Average(parts=parts), unmodelled=tuple(sorted(unmodelled))
    )


def _shift(node: Fold, offset: int) -> Fold:
    """Re-indexes a sub-tree's leaves after its positions were appended to a larger list."""
    if isinstance(node, LeafRef):
        return LeafRef(index=node.index + offset)
    if isinstance(node, BestOf):
        return BestOf(
            chooser=node.chooser, options=[_shift(o, offset) for o in node.options]
        )
    return Average(parts=[(w, _shift(part, offset)) for w, part in node.parts])


def turn_expectation(
    reg: Regulation, result: TurnResult, value: Callable[[Position], float]
) -> tuple[float, tuple[str, ...]]:
    """The turn's value under ``value``, with every mid-turn replacement chosen.

    The convenience form of `turn_leaves` for callers that score one turn at a time.
    ``value`` is read from side 0's point of view, which is the convention the payoff
    matrices and the LP already use.
    """
    if not result.suspended:
        return result.expected(value), result.unmodelled
    plan = turn_leaves(reg, result)
    return plan.value([value(p) for p in plan.positions]), plan.unmodelled


def _slot_of(turn: _Turn, source_slot: str | None) -> tuple[int, int] | None:
    """Turns a Showdown slot label like ``p2a`` into our (side, slot) pair."""
    if not source_slot or len(source_slot) < 3:
        return None
    try:
        side = int(source_slot[1]) - 1
    except ValueError:
        return None
    slot = ord(source_slot[2]) - ord("a")
    if side not in (0, 1) or not 0 <= slot < len(turn.pos.sides[side].active):
        return None
    return side, slot


def _residuals(reg: Regulation, turn: _Turn) -> None:
    """End-of-turn effects, in Showdown's residual order."""
    field_ = turn.pos.field

    def actives() -> list[tuple[int, int]]:
        return [
            (side, slot)
            for side in range(2)
            for slot in range(len(turn.pos.sides[side].active))
        ]

    # Residual order 1: weather. Its duration is decremented *before* its handler runs and
    # the handler is skipped when it expires, so the last turn of a sandstorm deals no
    # damage at all.
    weather_expired = False
    if field_.weather is not None and field_.weather_duration is not None:
        field_.weather_duration -= 1
        if field_.weather_duration <= 0:
            turn.log(f"{field_.weather} ended")
            field_.weather = None
            field_.weather_duration = None
            weather_expired = True

    if field_.weather == "sandstorm" and not weather_expired:
        for side, slot in actives():
            mon = turn.mon_at(side, slot)
            if mon is None or mon.fainted:
                continue
            if set(turn.types_of(mon)) & SANDSTORM_IMMUNE_TYPES:
                continue
            if mon.ability in SANDSTORM_IMMUNE_ABILITIES or mon.item == "safetygoggles":
                continue
            turn.deal_damage(
                side, slot, turn.fraction_of_max(side, slot, SANDSTORM_DAMAGE), reason="sandstorm"
            )

    # Abilities that gain or lose HP to the weather. Showdown hangs these on `onWeather`,
    # which runs with the weather's own residual, so they go here beside the sandstorm.
    if field_.weather is not None and not weather_expired:
        for side, slot in actives():
            mon = turn.mon_at(side, slot)
            if mon is None or mon.fainted:
                continue
            effect = WEATHER_ABILITY_HP.get(mon.ability)
            if effect is None:
                continue
            ratio = effect.get(field_.weather)
            if ratio is None:
                continue
            numerator, denominator = ratio
            amount = turn.fraction_of_max(side, slot, (abs(numerator), denominator))
            if numerator < 0:
                turn.deal_damage(side, slot, amount, reason=mon.ability)
            else:
                turn.heal(side, slot, amount, reason=mon.ability)

    for side, slot in actives():
        mon = turn.mon_at(side, slot)
        if mon is not None and not mon.fainted and mon.item == "leftovers":
            turn.heal(
                side, slot, turn.fraction_of_max(side, slot, LEFTOVERS_HEAL), reason="leftovers"
            )

    for side, slot in actives():
        mon = turn.mon_at(side, slot)
        if mon is None or mon.fainted or mon.ability == "magicguard":
            continue
        seed = mon.volatile("leechseed")
        if seed is not None:
            # `getAtSlot(sourceSlot)`: the HP goes to whoever planted it, and nowhere at
            # all if that Pokemon has left or fainted. Healing the nearest foe instead is
            # right only by coincidence.
            planter = _slot_of(turn, seed.source_slot)
            drained = turn.deal_damage(
                side, slot, turn.fraction_of_max(side, slot, LEECH_SEED_DRAIN),
                reason="leechseed",
            )
            if planter is not None and drained:
                receiver = turn.mon_at(*planter)
                if receiver is not None and not receiver.fainted:
                    turn.heal(*planter, drained, reason="leechseed")

    for side, slot in actives():
        mon = turn.mon_at(side, slot)
        if mon is None or mon.fainted or mon.ability == "magicguard":
            continue
        if mon.status == "brn":
            turn.deal_damage(side, slot, turn.fraction_of_max(side, slot, BURN_DAMAGE), reason="brn")
        elif mon.status == "psn":
            turn.deal_damage(
                side, slot, turn.fraction_of_max(side, slot, POISON_DAMAGE), reason="psn"
            )
        elif mon.status == "tox":
            stage = min((mon.status_counter or 0) + 1, 15)
            mon.status_counter = stage
            # Showdown: clampIntRange(baseMaxhp / 16, 1) * stage -- the sixteenth is
            # truncated first and then multiplied, which is not the same as truncating the
            # product.
            per_stage = max(1, mon.maxhp // 16)
            turn.deal_damage(side, slot, per_stage * stage, reason="tox")
        trap = mon.volatile("partiallytrapped")
        if trap is not None:
            if _trapper_gone(turn, trap):
                mon.volatiles = [v for v in mon.volatiles if v.id != "partiallytrapped"]
                turn.log(f"{turn.name(side, slot)} freed (trapper left)")
            else:
                turn.deal_damage(
                    side, slot, turn.fraction_of_max(side, slot, PARTIAL_TRAP_DAMAGE),
                    reason="partiallytrapped",
                )
        if mon.has_volatile("saltcure"):
            weak = bool(set(turn.types_of(mon)) & {"Water", "Steel"})
            ratio = SALT_CURE_DAMAGE_WEAK if weak else SALT_CURE_DAMAGE
            turn.deal_damage(side, slot, turn.fraction_of_max(side, slot, ratio), reason="saltcure")

    # Residual order 24: Perish Song. The counter is decremented like any other duration,
    # but on expiry the effect *runs* -- `onEnd` faints the Pokemon -- where weather's
    # handler is skipped instead. Missing it leaves a Pokemon alive that the game killed,
    # and every later turn of that battle is then scored against the wrong position.
    for side, slot in actives():
        mon = turn.mon_at(side, slot)
        if mon is None or mon.fainted:
            continue
        perish = mon.volatile("perishsong")
        if perish is None or perish.duration is None:
            continue
        perish.duration -= 1
        turn.log(f"{turn.name(side, slot)} perish{max(0, perish.duration)}")
        if perish.duration <= 0:
            turn.faint(side, slot)

    # Residual order 28: Speed Boost. `if (pokemon.activeTurns)` is what stops it firing on
    # the turn its holder came in, and `newly_switched` is that flag -- cleared at the very
    # end of this function, so it is still readable here.
    for side, slot in actives():
        mon = turn.mon_at(side, slot)
        if mon is None or mon.fainted or mon.ability != "speedboost":
            continue
        if mon.newly_switched:
            continue
        turn.apply_boosts(side, slot, {"spe": 1}, reason="speedboost", from_foe=False)

    # Residual order 29: the last of White Herb's four chances to fire.
    _check_white_herb(turn)

    for side in range(2):
        kept: list[Effect] = []
        for condition in turn.pos.sides[side].side_conditions:
            if condition.duration is not None:
                condition.duration -= 1
                if condition.duration <= 0:
                    turn.log(f"p{side + 1} side -{condition.id}")
                    continue
            kept.append(condition)
        turn.pos.sides[side].side_conditions = kept

    if field_.terrain_duration is not None:
        field_.terrain_duration -= 1
        if field_.terrain_duration <= 0:
            turn.log(f"{field_.terrain} ended")
            field_.terrain = None
            field_.terrain_duration = None
    kept_pseudo: list[Effect] = []
    for pseudo in field_.pseudo_weather:
        if pseudo.duration is not None:
            pseudo.duration -= 1
            if pseudo.duration <= 0:
                turn.log(f"{pseudo.id} ended")
                continue
        kept_pseudo.append(pseudo)
    field_.pseudo_weather = kept_pseudo

    # The Protect-family volatiles last one turn. `stall` is deliberately not here: it
    # carries the repeat counter and expires on its own two-turn duration.
    single_turn = set(PROTECT_VOLATILES) | {
        "flinch", "helpinghand", "followme", "ragepowder", "spotlight", "glaiverush",
    }
    # Perish Song already counted down at its own residual order above; decrementing it
    # again here would kill a Pokemon a turn and a half early.
    handled_earlier = {"perishsong"}
    for side, slot in actives():
        mon = turn.mon_at(side, slot)
        if mon is None:
            continue
        kept_volatiles: list[Effect] = []
        for volatile in mon.volatiles:
            if volatile.id in single_turn:
                continue
            if volatile.id in handled_earlier:
                kept_volatiles.append(volatile)
                continue
            if volatile.duration is not None:
                volatile.duration -= 1
                if volatile.duration <= 0:
                    # Yawn's whole effect is on expiry: the target falls asleep at the end
                    # of the turn after it lands, not when it hits.
                    if volatile.id == "yawn":
                        turn.apply_status(side, slot, "slp", reason="yawn")
                    if volatile.id == "disable" and volatile.move:
                        # The flag lives on the move slot, so it has to be cleared here or
                        # the move stays unusable for the rest of the battle.
                        for move_slot in mon.moves:
                            if move_slot.id == volatile.move:
                                move_slot.disabled = False
                    continue
            kept_volatiles.append(volatile)
        mon.volatiles = kept_volatiles
        mon.newly_switched = False
        # Carry this turn's outcome forward for Stomping Tantrum and Temper Flare.
        mon.move_last_turn_failed = (side, slot) in turn.move_failed

    del reg
    for side_index, side in enumerate(turn.pos.sides):
        if all(m.fainted for m in side.pokemon):
            turn.pos.ended = True
            turn.pos.winner = turn.pos.sides[1 - side_index].id


__all__ = [
    "Branch",
    "Budget",
    "ReplacementResult",
    "TurnResult",
    "replacements_needed",
    "resolve_replacements",
    "pending_attacks",
    "resolve_turn",
    "stratified_rolls",
]
