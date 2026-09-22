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

#: Volatiles that prevent switching outright.
TRAPPING_VOLATILES = frozenset({"partiallytrapped", "octolock", "trapped"})

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

#: Abilities on an adjacent foe that trap; each also has an escape condition, so this is
#: only consulted when the position does not already carry Showdown's `trapped` flag.
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
        out.append((i, m.id))
    return out


def is_struggling(mon, reg: Regulation) -> bool:  # noqa: ANN001
    """True when nothing on the moveset can be selected, so Showdown offers Struggle.

    Not "every move is out of PP": a Choice item can leave exactly one move selectable and
    that one can be the empty one, which is Struggle with three full move slots beside it.
    The condition is the legal set being empty, and it is named here so a reader of a game
    log and the search enumerating that turn cannot disagree about what was on the menu.
    """
    return not _usable_move_slots(mon, reg)


def _targets_for(
    reg: Regulation, move_id: str, slot: int, foe: Side, active_per_side: int
) -> list[int | None]:
    """Legal target indices for one move from one slot.

    Showdown rejects a choice that supplies a target for a move that takes none, and
    rejects one that omits a target for a single-target move, so the two cases are
    distinct rather than optional.
    """
    move = reg.moves.get(move_id)
    if move is None:
        # Unknown move: let it through with no target rather than inventing one.
        return [None]
    target = move.target

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

    out: list[SlotAction] = []

    usable = _usable_move_slots(mon, reg)
    mega_target = None
    if allow_mega and not side.mega_used and not mon.is_mega:
        mega_target = reg.mega_target(mon.species, mon.item)

    if usable:
        for move_index, move_id in usable:
            for target in _targets_for(reg, move_id, slot, foe, active_per_side):
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

    if allow_switch and not _is_trapped(mon):
        for candidate in side.pokemon:
            if candidate.fainted or candidate.is_active:
                continue
            out.append(
                SwitchAction(slot=slot, party_index=candidate.slot + 1, species=candidate.species)
            )

    return out


def _is_trapped(mon) -> bool:  # noqa: ANN001
    """Whether a Pokemon may not switch out.

    ``mon.trapped`` is Showdown's own verdict when the position came from the oracle, and
    is authoritative because it already accounts for Ghost types, Shed Shell and ability
    suppression. The volatile check covers hand-written positions that omit the flag.
    """
    if mon.trapped:
        return True
    return any(v.id in TRAPPING_VOLATILES for v in mon.volatiles)


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
