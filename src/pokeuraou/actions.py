"""Legal action enumeration for one side of a doubles turn.

A side's action is the product of the two active slots' choices: a move (with a target,
and optionally a Mega Evolution declaration), a switch, or a pass. The product is large --
a few hundred per side, tens of thousands of matrix cells -- so this module only has to be
*correct*; the narrowing happens in ``candidates.py``.

Correctness here is not asserted, it is tested: ``tests/test_actions.py`` enumerates
positions and asks Showdown, via the oracle's ``probe_choices``, whether it accepts
exactly the strings we produce and rejects everything we exclude.

Target encoding matches Showdown's choice strings: foes are ``1`` and ``2`` (their slot
order), allies are ``-1`` and ``-2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from .position import Position, Side
from .regulation import (
    TARGETS_REQUIRING_ALLY,
    TARGETS_REQUIRING_FOE,
    TARGETS_WITHOUT_CHOICE,
    Regulation,
)

if TYPE_CHECKING:
    # Display only, and imported for typing alone: enumerating legal actions is on the
    # hot path and has no business loading the name tables.
    from .names import Localiser

STRUGGLE = "struggle"


@dataclass(frozen=True, slots=True)
class TargetNames:
    """Which Pokemon each target index points at, so output can name it.

    A choice string's target is relative to the side that is choosing -- positive is a foe
    slot, negative is one of its own -- so "foe1" means different Pokemon depending on who
    is being asked. Resolving it therefore needs the position *and* the acting side, which
    is more than an action knows about itself; :func:`target_names` builds this and
    :meth:`MoveAction.describe` takes it.

    A slot with nothing in it stays ``None`` and the caller falls back to the index, which
    is the honest answer: there is no Pokemon to name.
    """

    #: Species ids by 1-based foe slot, and by 1-based own slot.
    foes: tuple[str | None, ...]
    allies: tuple[str | None, ...]

    def species_for(self, target: int) -> str | None:
        table = self.foes if target > 0 else self.allies
        index = abs(target) - 1
        return table[index] if 0 <= index < len(table) else None


def target_names(pos: Position, side_index: int) -> TargetNames:
    """Resolves target indices for one side's choices."""

    def species_of(side: Side) -> tuple[str | None, ...]:
        return tuple(
            None if mon is None or mon.fainted else mon.species
            for mon in side.active_pokemon()
        )

    return TargetNames(
        foes=species_of(pos.sides[1 - side_index]),
        allies=species_of(pos.sides[side_index]),
    )

#: The two trapping volatiles that are conditions in Showdown's `data/conditions.ts`
#: rather than a move's own condition, so the dump does not carry their hooks: Mean Look's
#: `trapped` and the binding moves' `partiallytrapped`. Each `onTrapPokemon` calls
#: `pokemon.tryTrap()`. The traps a move's condition sets -- Octolock, Ingrain, No Retreat,
#: Fairy Lock -- are read from the dump (`Regulation.trapping_volatiles` and
#: `trapping_pseudo_weather`, IKA-169).
TRAPPING_CONDITIONS = frozenset({"partiallytrapped", "trapped"})

#: The volatiles Showdown's `LockMove` event reads (`getLockedMove`), besides the recharge
#: turn: a charging move's second turn and Outrage's rampage. Each stores its move in
#: `effectState.move`, which is `Effect.move` here.
LOCKING_VOLATILES = ("twoturnmove", "lockedmove")

#: The fake move a Pokemon spends its recharge turn on. There is no `recharge` move in
#: the dex -- Showdown builds the request entry by hand:
#:
#:     if (lockedMove === 'recharge') return [{ move: 'Recharge', id: 'recharge' }];
#:
#: and `mustrecharge.onBeforeMove` intercepts the action before anything is executed, so
#: nothing ever looks the id up. It is a special case in Showdown and stays one here;
#: putting it in the regulation dump would mean inventing data that the simulator does
#: not have.
RECHARGE = "recharge"

#: Abilities on an adjacent foe that trap (`onFoeTrapPokemon`), each with its own
#: condition on the Pokemon it traps. Only Mega Gengar's Shadow Tag has a holder in the
#: champions dex; the dump names the hook but not the condition, which is code.
TRAPPING_ABILITIES = frozenset({"shadowtag", "arenatrap", "magnetpull"})


@dataclass(frozen=True, slots=True)
class MoveAction:
    """A move choice for one active slot."""

    slot: int
    #: 1-based index into the Pokemon's move slots, matching the choice string.
    move_index: int
    move_id: str
    #: Showdown target index, or None for moves that take no target.
    target: int | None
    mega: bool = False

    def to_choice(self) -> str:
        parts = ["move", str(self.move_index)]
        if self.target is not None:
            parts.append(str(self.target))
        if self.mega:
            parts.append("mega")
        return " ".join(parts)

    def describe(
        self,
        reg: Regulation,
        loc: Localiser | None = None,
        targets: TargetNames | None = None,
    ) -> str:
        """The choice as a line of text.

        With ``targets`` the target is named rather than numbered: "ねっぷう → ガオガエン"
        instead of "ねっぷう → 相手1". Species Clause guarantees the name is unambiguous
        within a side, and an empty slot falls back to the index because there is nothing
        to name.
        """
        if loc is not None:
            out = loc.move(self.move_id)
        else:
            move = reg.moves.get(self.move_id)
            out = move.name if move else self.move_id
        if self.target is not None:
            species_id = targets.species_for(self.target) if targets else None
            if species_id is not None:
                who = (
                    loc.species(species_id)
                    if loc is not None
                    else (
                        reg.species[species_id].name
                        if species_id in reg.species
                        else species_id
                    )
                )
                # The side still matters: an ally-targeting Fake Out reads very differently
                # from a foe-targeting one, and the name alone does not say which.
                if self.target < 0:
                    who = f"味方{who}" if loc is not None else f"ally {who}"
                out += f" → {who}" if loc is not None else f" -> {who}"
            elif loc is not None:
                out += f" → {'相手' if self.target > 0 else '味方'}{abs(self.target)}"
            else:
                out += f" -> {'foe' if self.target > 0 else 'ally'}{abs(self.target)}"
        if self.mega:
            out += " + メガ" if loc is not None else " + Mega"
        return out


@dataclass(frozen=True, slots=True)
class SwitchAction:
    slot: int
    #: 1-based party index, matching the choice string.
    party_index: int
    species: str

    def to_choice(self) -> str:
        return f"switch {self.party_index}"

    def describe(
        self,
        reg: Regulation,
        loc: Localiser | None = None,
        targets: TargetNames | None = None,  # noqa: ARG002 - a switch has no target
    ) -> str:
        if loc is not None:
            return f"交代 → {loc.species(self.species)}"
        species = reg.species.get(self.species)
        return f"switch -> {species.name if species else self.species}"


@dataclass(frozen=True, slots=True)
class PassAction:
    slot: int

    def to_choice(self) -> str:
        return "pass"

    def describe(
        self,
        reg: Regulation,  # noqa: ARG002
        loc: Localiser | None = None,
        targets: TargetNames | None = None,  # noqa: ARG002
    ) -> str:
        return "行動なし" if loc is not None else "pass"


SlotAction = MoveAction | SwitchAction | PassAction


@dataclass(frozen=True, slots=True)
class SideAction:
    """One side's complete choice for a turn: one slot action per active slot."""

    slots: tuple[SlotAction, ...]

    def to_choice(self) -> str:
        return ", ".join(a.to_choice() for a in self.slots)

    def describe(
        self,
        reg: Regulation,
        loc: Localiser | None = None,
        targets: TargetNames | None = None,
    ) -> str:
        return " | ".join(a.describe(reg, loc, targets) for a in self.slots)

    @property
    def declares_mega(self) -> bool:
        return any(isinstance(a, MoveAction) and a.mega for a in self.slots)

    @property
    def switch_indices(self) -> tuple[int, ...]:
        return tuple(a.party_index for a in self.slots if isinstance(a, SwitchAction))


def _usable_move_slots(mon, reg: Regulation) -> list[tuple[int, str]]:  # noqa: ANN001
    """1-based move indices the Pokemon may select, after locks and disables.

    Struggle is returned as an empty list, which the caller turns into the Struggle
    choice; Showdown itself represents this as a single fake move slot.
    """
    locked = mon.locked_move
    if locked:
        for i, m in enumerate(mon.moves, start=1):
            if m.id == locked and m.usable:
                # Choice-locked into Gigaton Hammer right after it: Struggle (IKA-176).
                if _disabled_after_itself(reg.moves.get(m.id), mon):
                    return []
                return [(i, m.id)]
        # A lock naming a move that is gone leaves the normal set available.

    taunted = mon.has_volatile("taunt")
    tormented = mon.has_volatile("torment")
    # Throat Chop's `onDisableMove` disables every sound move on the target for two turns,
    # so they never reach the request in the first place.
    throat_chopped = mon.has_volatile("throatchop")
    # Disable and Cursed Body set `MoveSlot.disabled`, which `MoveSlot.usable` already
    # honours -- but nothing set it until the resolver learned to, so this is where the
    # legality half of that mechanic lands.
    disabled = mon.volatile("disable")
    disabled_move = disabled.move if disabled is not None else None
    # Encore leaves exactly one move on offer. Showdown expresses it as an action override
    # rather than a request filter, but the request it sends does mark every other move
    # disabled, so the legal set is the same one move.
    encored = mon.volatile("encore")
    encored_move = encored.move if encored is not None else None
    # A Choice item locks its holder into the move it used. `onDisableMove` drops the
    # volatile when the item is gone or the move is, so both are checked here rather than
    # trusting the volatile to have been cleaned up.
    choice = mon.volatile("choicelock")
    choice_move = None
    if (
        choice is not None
        and choice.move
        and mon.item is not None
        and mon.item in reg.choice_items
        and any(m.id == choice.move for m in mon.moves)
    ):
        choice_move = choice.move
    out: list[tuple[int, str]] = []
    for i, m in enumerate(mon.moves, start=1):
        if not m.usable:
            continue
        move = reg.moves.get(m.id)
        if taunted and move is not None and move.category == "Status":
            continue
        if throat_chopped and move is not None and "sound" in move.flags:
            continue
        if disabled_move is not None and m.id == disabled_move:
            continue
        if encored_move is not None and m.id != encored_move:
            continue
        if choice_move is not None and m.id != choice_move:
            continue
        if tormented and mon.last_move == m.id:
            continue
        if mon.active_move_actions and _disabled_once_moved(move):
            continue
        if _disabled_after_itself(move, mon):
            continue
        out.append((i, m.id))
    return out


def _disabled_after_itself(move, mon) -> bool:  # noqa: ANN001
    """Gigaton Hammer right after Gigaton Hammer (IKA-176).

    `endTurn` disables a `cantusetwice` move that was the last one used
    (vendor/pokemon-showdown/sim/battle.ts:1695)::

        if (activeMove.flags['cantusetwice'] && pokemon.lastMove?.id === moveSlot.id) {
            pokemon.disableMove(pokemon.lastMove.id);
        }

    `lastMove` is set once the move starts (`moveUsed`, after `BeforeMove`), so a turn
    lost to flinch or sleep keeps it disabled, and switching out clears it. The dump's
    flag decides, so a dex without it offers the move.
    """
    return move is not None and "cantusetwice" in move.flags and mon.last_move == move.id


#: The moves whose champions `onDisableMove` reads the move counter
#: (vendor/pokemon-showdown/data/mods/champions/moves.ts, fakeout and firstimpression):
#:
#:     onDisableMove(pokemon) {
#:         if (pokemon.activeMoveActions) pokemon.disableMove('fakeout');
#:     },
#:
#: `endTurn` runs every `DisableMove` handler before each request (sim/battle.ts:1691), so
#: once the Pokemon has made one move action since coming in, the request marks the move
#: disabled and Showdown refuses the choice ("Fake Out is disabled"). Mat Block has the
#: same `onTry` but no such hook, and is not in the champions dex.
DISABLED_ONCE_MOVED = frozenset({"fakeout", "firstimpression"})


def _disabled_once_moved(move) -> bool:  # noqa: ANN001
    """Whether the dex disables this move after its user's first move action (IKA-166).

    The dump's `customHooks` decides, so a dex without the champions hook -- the base
    game, where Fake Out stays selectable and fails in `onTry` -- keeps offering it.
    """
    return (
        move is not None
        and move.id in DISABLED_ONCE_MOVED
        and "onDisableMove" in move.custom_hooks
    )


def is_struggling(mon, reg: Regulation) -> bool:  # noqa: ANN001
    """True when nothing on the moveset can be selected, so Showdown offers Struggle.

    Not "every move is out of PP": a Choice item can leave exactly one move selectable and
    that one can be the empty one, which is Struggle with three full move slots beside it.
    The condition is the legal set being empty, and it is named here so a reader of a game
    log and the search enumerating that turn cannot disagree about what was on the menu.
    """
    return not _usable_move_slots(mon, reg)


def _targets_for(
    reg: Regulation,
    move_id: str,
    slot: int,
    foe: Side,
    active_per_side: int,
    user_types: tuple[str, ...] = (),
) -> list[int | None]:
    """Legal target indices for one move from one slot.

    Showdown rejects a choice that supplies a target for a move that takes none, and
    rejects one that omits a target for a single-target move, so the two cases are
    distinct rather than optional.

    The target Showdown validates against is the request's, not the dex entry's
    (`Pokemon.getMoves`), and the two differ for Curse: a user that is not Ghost now is
    asked for no target (`if (!this.hasType('Ghost')) target = 'self'`), so "move N 1"
    is refused (IKA-168). Before Showdown 2345119 the dump said so as `nonGhostTarget`.
    """
    move = reg.moves.get(move_id)
    if move is None:
        # Unknown move: let it through with no target rather than inventing one.
        return [None]
    target = move.target
    if move_id == "curse" and "Ghost" not in user_types:
        target = "self"

    if target in TARGETS_WITHOUT_CHOICE:
        return [None]

    if target in TARGETS_REQUIRING_FOE:
        options: list[int | None] = []
        for i in range(active_per_side):
            mon = foe.pokemon[foe.active[i]] if foe.active[i] is not None else None
            if mon is not None and not mon.fainted:
                options.append(i + 1)
        if target == "any":
            # `any` also reaches the ally slot (but not self).
            for i in range(active_per_side):
                if i != slot:
                    options.append(-(i + 1))
        return options

    if target in TARGETS_REQUIRING_ALLY:
        options = []
        for i in range(active_per_side):
            if target == "adjacentAlly" and i == slot:
                continue
            options.append(-(i + 1))
        return options

    return [None]


def slot_actions(
    reg: Regulation,
    pos: Position,
    side_index: int,
    slot: int,
    *,
    allow_mega: bool = True,
    allow_switch: bool = True,
) -> list[SlotAction]:
    """Every legal choice for one active slot, ignoring cross-slot constraints.

    Cross-slot constraints (only one Mega per turn, two slots cannot switch to the same
    Pokemon) are applied in :func:`side_actions`.
    """
    side = pos.sides[side_index]
    foe = pos.sides[1 - side_index]
    active_per_side = len(side.active)

    party_slot = side.active[slot] if slot < len(side.active) else None
    if party_slot is None:
        return [PassAction(slot=slot)]
    mon = side.pokemon[party_slot]
    if mon.fainted:
        return [PassAction(slot=slot)]

    # Hyper Beam's recharge turn replaces the whole request. Showdown offers one fake move
    # and sets `trapped: true`, so there is exactly one legal action and switching is not
    # among them:
    #
    #     {"moves": [{"move": "Recharge", "id": "recharge"}], "trapped": true}
    #
    # That has to be honoured here rather than inside the resolver, because the action set
    # *is* what the equilibrium is computed over: offering the real four moves would put
    # probability on choices Showdown rejects, and offering a switch would let the search
    # escape a downside the move is priced on. Before this, Hyper Beam was a 150-power
    # move with no cost at all, and 569 of them were played in one worker's 500 games.
    if mon.has_volatile("mustrecharge"):
        return [MoveAction(slot=slot, move_index=1, move_id=RECHARGE, target=None)]

    # The same shape for a charging move's second turn and Outrage's rampage (IKA-169):
    # `getMoveRequestData` sets `trapped = true` for any `getLockedMove()`, after the
    # TrapPokemon event, so neither Shed Shell nor a Ghost type escapes it, and offers the
    # one move with no Mega (`if (!lockedMove) { if (this.canMegaEvo) ... }`). Showdown
    # then fires at the stored target whatever the choice says, and refuses a choice that
    # names one: the request's entry has no `target` (IKA-176). The charge stores it as
    # `twoturnmove`'s `targetLoc`, which the queue reads (`speed.build_queue`), so the
    # menu is the one choice with no target. A marker without it -- a record from before
    # IKA-176 -- keeps one choice per target, as before.
    locked = locked_move(reg, mon)
    if locked is not None:
        index = next((i for i, m in enumerate(mon.moves, start=1) if m.id == locked), None)
        if index is not None:
            charging = mon.volatile("twoturnmove")
            if charging is not None and charging.move == locked and charge_target(mon) is not None:
                return [MoveAction(slot=slot, move_index=index, move_id=locked, target=None)]
            return [
                MoveAction(slot=slot, move_index=index, move_id=locked, target=target)
                for target in _targets_for(reg, locked, slot, foe, active_per_side)
            ]

    out: list[SlotAction] = []

    usable = _usable_move_slots(mon, reg)
    mega_target = None
    if allow_mega and not side.mega_used and not mon.is_mega:
        mega_target = reg.mega_target(mon.species, mon.item)

    if usable:
        user_types = mon.types or _species_types(reg, mon.species)
        for move_index, move_id in usable:
            for target in _targets_for(reg, move_id, slot, foe, active_per_side, user_types):
                out.append(
                    MoveAction(slot=slot, move_index=move_index, move_id=move_id, target=target)
                )
                if mega_target is not None:
                    out.append(
                        MoveAction(
                            slot=slot,
                            move_index=move_index,
                            move_id=move_id,
                            target=target,
                            mega=True,
                        )
                    )
    else:
        # No usable move: Showdown offers Struggle in move slot 1.
        for target in _targets_for(reg, STRUGGLE, slot, foe, active_per_side):
            out.append(MoveAction(slot=slot, move_index=1, move_id=STRUGGLE, target=target))

    if allow_switch and not _is_trapped(reg, pos, side_index, mon):
        for candidate in side.pokemon:
            if candidate.fainted or candidate.is_active:
                continue
            out.append(
                SwitchAction(slot=slot, party_index=candidate.slot + 1, species=candidate.species)
            )

    return out


def locked_move(reg: Regulation, mon) -> str | None:  # noqa: ANN001
    """The move Showdown's `getLockedMove` names for this Pokemon, recharge aside.

    A position straight from Showdown, and one our resolver built since IKA-169, carries
    the move on the volatile. A recorded position from before carries a bare
    `twoturnmove` that the resolver never removed when the Pokemon chose another move, so
    there the lock is read off the last move, and only if that move charges (the dump's
    `charge` flag): a charge turn's last move is the charging move, and a leaked marker
    sits beside some other one.
    """
    for vid in LOCKING_VOLATILES:
        effect = mon.volatile(vid)
        if effect is None:
            continue
        if effect.move:
            return effect.move
        if vid == "twoturnmove" and mon.last_move:
            last = reg.moves.get(mon.last_move)
            if last is not None and "charge" in last.flags:
                return last.id
    return None


def charge_target(mon) -> int | None:  # noqa: ANN001
    """The Showdown target loc a charging move stored on its first turn (IKA-176).

    Showdown keeps it on the move's own volatile (`volatiles[move].targetLoc`); the
    position carries it on `twoturnmove` instead, as `extra.targetLoc`, since that is the
    volatile both resolvers model. 0 and a missing value are "none".
    """
    charging = mon.volatile("twoturnmove")
    if charging is None or not charging.move:
        return None
    loc = charging.extra.get("targetLoc") if charging.extra else None
    return loc if isinstance(loc, int) and not isinstance(loc, bool) and loc != 0 else None


def _is_trapped(reg: Regulation, pos: Position, side_index: int, mon) -> bool:  # noqa: ANN001
    """Whether a Pokemon may not switch out.

    ``mon.trapped`` is Showdown's own verdict when the position came from the oracle, and
    is authoritative because it already accounts for Ghost types, Shed Shell and ability
    suppression. Everything after it covers hand-written positions that omit the flag, and
    the positions our resolver builds -- the search's children, and every position of a
    generated game -- which carry the trap's source and a flag nobody computes, so it has
    to know each source and each escape: Run Away (IKA-136), Ghost types and Shed Shell
    (IKA-163), and since IKA-169 the move locks, Ingrain, No Retreat, Fairy Lock and the
    foe's Shadow Tag.
    """
    if mon.trapped:
        return True
    if locked_move(reg, mon) is not None:
        # `getMoveRequestData` traps after every escape has run (IKA-169).
        return True
    if _escapes_traps(reg, mon):
        return False
    if any(v.id in TRAPPING_CONDITIONS or v.id in reg.trapping_volatiles for v in mon.volatiles):
        return True
    if any(e.id in reg.trapping_pseudo_weather for e in pos.field.pseudo_weather):
        return True
    return _trapped_by_foe_ability(reg, pos, side_index, mon)


def _trapped_by_foe_ability(reg: Regulation, pos: Position, side_index: int, mon) -> bool:  # noqa: ANN001
    """Showdown's `onFoeTrapPokemon` for each ability in `TRAPPING_ABILITIES`
    (`data/abilities.ts`), from every active foe adjacent to this Pokemon::

        shadowtag:  if (!pokemon.hasAbility('shadowtag') && pokemon.isAdjacent(holder))
        arenatrap:  if (!pokemon.isAdjacent(holder)) return; if (pokemon.isGrounded())
        magnetpull: if (pokemon.hasType('Steel') && pokemon.isAdjacent(holder))

    each then `pokemon.tryTrap(true)`, so the escapes in `_escapes_traps` apply and the
    caller has already run them. `isAdjacent` is false when either side has fainted.

    Showdown computes this once, in `endTurn`, and our menus are built from the position
    at the start of a turn, which is the same moment: a Gengar that Mega Evolves traps from
    the next turn, and one that has fainted traps nobody.
    """
    own = pos.sides[side_index]
    foe = pos.sides[1 - side_index]
    if mon.active_index is None:
        return False
    types: tuple[str, ...] | None = None
    for position, party in enumerate(foe.active):
        if party is None:
            continue
        holder = foe.pokemon[party]
        if holder.fainted or holder.ability not in TRAPPING_ABILITIES:
            continue
        if not _adjacent_foes(len(own.active), mon.active_index, position):
            continue
        if holder.ability == "shadowtag":
            if mon.ability != "shadowtag":
                return True
            continue
        if types is None:
            types = mon.types or _species_types(reg, mon.species)
        if holder.ability == "magnetpull" and "Steel" in types:
            return True
        if holder.ability == "arenatrap" and _grounded(pos, mon, types):
            return True
    return False


def _adjacent_foes(active_per_side: int, mine: int, theirs: int) -> bool:
    """`Pokemon.isAdjacent` across the field (`sim/pokemon.ts`)::

        if (this.battle.activePerHalf <= 2) return this !== pokemon2;
        return Math.abs(this.position + pokemon2.position + 1 - this.side.active.length) <= 1;
    """
    if active_per_side <= 2:
        return True
    return abs(mine + theirs + 1 - active_per_side) <= 1


def _grounded(pos: Position, mon, types: tuple[str, ...]) -> bool:  # noqa: ANN001
    """Showdown's `isGrounded` for Arena Trap, which has no holder in the champions dex.

    Gravity, Ingrain, Smack Down and Iron Ball ground; a Flying type, Levitate, Magnet
    Rise, Telekinesis and Air Balloon do not. The resolver's own `_grounded` is the same
    list without Gravity, which it does not model.
    """
    if any(e.id == "gravity" for e in pos.field.pseudo_weather):
        return True
    if mon.has_volatile("ingrain") or mon.has_volatile("smackdown") or mon.item == "ironball":
        return True
    if "Flying" in types or mon.ability == "levitate":
        return False
    if mon.has_volatile("magnetrise") or mon.has_volatile("telekinesis"):
        return False
    return mon.item != "airballoon"


def _escapes_traps(reg: Regulation, mon) -> bool:  # noqa: ANN001
    """Whether Showdown frees this Pokemon from every trap that goes through `tryTrap`.

    That is every volatile in `TRAPPING_CONDITIONS` and `Regulation.trapping_volatiles`,
    Fairy Lock, and the foe's trapping abilities (IKA-169) -- all but the move locks,
    which `getMoveRequestData` sets afterwards. All three escapes follow the dump rather
    than a list written here:

    - **Ghost types** (IKA-163). `tryTrap` begins
      `if (!this.runStatusImmunity('trapped')) return false;`, which is the type chart's
      `ghost: { damageTaken: { trapped: 3 } }` -- the dump's `effectImmunities.trapped`.
      Mean Look and Octolock never land on a Ghost, but Infestation and the other binding
      moves do: `partiallytrapped` has no immunity of its own, only its trap does.
    - **Shed Shell** (IKA-163). `onTrapPokemonPriority: -10` sets `pokemon.trapped = false`
      after every trapper has run.
    - **Run Away** (IKA-136). The champions mod (Showdown d849b2200) gives it the same
      hook as Shed Shell; in a dex without the hook Run Away does nothing in battle,
      which is what it did here before the bump.

    Embargo, Magic Room and Klutz would silence Shed Shell, and Gastro Acid or Neutralizing
    Gas Run Away and a foe's Shadow Tag; the resolver models none of them, so neither does
    this.
    """
    types = mon.types or _species_types(reg, mon.species)
    if reg.immune_to_effect("trapped", types):
        return True
    if mon.item == "shedshell" and _frees_holder(reg.items.get("shedshell")):
        return True
    return mon.ability == "runaway" and _frees_holder(reg.abilities.get("runaway"))


def _species_types(reg: Regulation, species: str) -> tuple[str, ...]:
    entry = reg.species.get(species)
    return () if entry is None else entry.types


def _frees_holder(entry) -> bool:  # noqa: ANN001
    """Whether the dump gives this item or ability its own `onTrapPokemon` hook."""
    return entry is not None and "onTrapPokemon" in entry.raw.get("customHooks", ())


def side_actions(
    reg: Regulation,
    pos: Position,
    side_index: int,
    *,
    allow_mega: bool = True,
    allow_switch: bool = True,
) -> list[SideAction]:
    """Every legal side action: the product of the slots, minus illegal combinations.

    Excluded combinations, each verified against Showdown:
    - both slots declaring Mega Evolution ("You can only mega-evolve once per battle")
    - both slots switching to the same party member ("can only switch in once")
    """
    side = pos.sides[side_index]
    per_slot = [
        slot_actions(
            reg, pos, side_index, slot, allow_mega=allow_mega, allow_switch=allow_switch
        )
        for slot in range(len(side.active))
    ]

    out: list[SideAction] = []
    for combo in product(*per_slot):
        megas = sum(1 for a in combo if isinstance(a, MoveAction) and a.mega)
        if megas > 1:
            continue
        switches = [a.party_index for a in combo if isinstance(a, SwitchAction)]
        if len(switches) != len(set(switches)):
            continue
        out.append(SideAction(slots=tuple(combo)))
    return out


def switch_actions_after_faint(
    reg: Regulation,  # noqa: ARG001
    pos: Position,
    side_index: int,
    must_switch: list[bool],
) -> list[SideAction]:
    """Replacement choices after a faint.

    ``must_switch`` mirrors Showdown's ``forceSwitch``: one flag per active slot.

    A slot may pass when there is nobody left to fill it, and which slot the last Pokemon
    takes is the player's choice -- so two slots owing a replacement with one Pokemon on the
    bench gives two options, not none. The rule is "fill as many as you can": the number of
    switches is exactly ``min(bench, owed)``, which with enough bench excludes every pass and
    leaves the ordinary case as it was.

    Returning nothing here was a spin rather than a wrong answer: the caller passed both
    slots, the position came back unchanged, and the replacement phase was entered again on
    it.
    """
    side = pos.sides[side_index]
    bench = [p for p in side.pokemon if not p.fainted and not p.is_active]
    owed = sum(1 for needed in must_switch if needed)
    fillable = min(len(bench), owed)

    per_slot: list[list[SlotAction]] = []
    for slot, needed in enumerate(must_switch):
        if not needed:
            per_slot.append([PassAction(slot=slot)])
            continue
        options: list[SlotAction] = [
            SwitchAction(slot=slot, party_index=p.slot + 1, species=p.species) for p in bench
        ]
        options.append(PassAction(slot=slot))
        per_slot.append(options)

    out: list[SideAction] = []
    for combo in product(*per_slot):
        switches = [a.party_index for a in combo if isinstance(a, SwitchAction)]
        if len(switches) != len(set(switches)):
            continue
        if len(switches) != fillable:
            continue
        out.append(SideAction(slots=tuple(combo)))
    return out
