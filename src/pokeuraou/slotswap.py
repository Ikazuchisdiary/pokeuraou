"""Exchanging the left and right Pokemon of one side (IKA-412).

A doubles side has two field positions, left (slot 0) and right (slot 1). Which of the two
active Pokemon stands where is, in the rules, almost never something that changes a turn's
result; the exception the user named is Imposter (a Ditto transforms into whoever stands in
front of it), and the port check of IKA-412 lists any others it finds. This module builds the
swapped position (for that check) and the swapped *encoded* position (for training the
value net on both arrangements), and nothing else.

What a swap of one side's positions moves, in the canonical position (`position.py`):

* `Side.active` -- the party index standing in each position: reversed.
* `Side.slot_conditions` -- per-position conditions (Wish, Future Sight's slot, ...): reversed.
* `Pokemon.active_index` -- the position a Pokemon stands in, derived from `active`.
* `twoturnmove.extra.targetLoc` -- where a charging move will fire, a position number seen from
  its holder (found by the IKA-412 check: Phantom Force aimed at the foe's left position).
* `Effect.source_slot` -- "who applied this", by position, in either of the two encodings
  in use (`"01"` side digit and position digit, Showdown's `"p1a"`): the position part of
  every effect whose source stands on the swapped side is exchanged.

What it does not move: the party order (`Side.pokemon`, `Pokemon.slot` -- a party index,
not a position), `mega_capable_slots` (party indices), the bench, and the field.

A SideAction's `slots` tuple is indexed by position, and a move's `target` is a position
number (foes `1` and `2`, allies `-1` and `-2`), so a swap of the acting side reverses the
tuple and exchanges the ally targets, and a swap of the opposing side exchanges the foe
targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import MoveAction, PassAction, SideAction, SlotAction, SwitchAction
from .encode import Encoded
from .position import Position

#: Abilities that read the position in front of their holder: Imposter (Transform on switch-in).
POSITION_READING_ABILITIES = frozenset({"imposter"})


def _swap_position_of_source(source_slot: str | None, sides: tuple[bool, bool]) -> str | None:
    """Exchanges the position part of a `source_slot` naming a swapped side."""
    if not source_slot:
        return source_slot
    if len(source_slot) >= 2 and source_slot[0] in "01" and source_slot[1] in "01":
        if sides[int(source_slot[0])]:
            return source_slot[0] + ("1" if source_slot[1] == "0" else "0") + source_slot[2:]
        return source_slot
    if len(source_slot) >= 3 and source_slot[0] == "p" and source_slot[1] in "12":
        if sides[int(source_slot[1]) - 1] and source_slot[2] in "ab":
            return source_slot[:2] + ("b" if source_slot[2] == "a" else "a") + source_slot[3:]
        return source_slot
    return source_slot


def swap_positions(pos: Position, sides: tuple[bool, bool] = (True, True)) -> Position:
    """A copy with the left and right Pokemon of the chosen sides exchanged."""
    out = pos.copy()
    for index, side in enumerate(out.sides):
        if not sides[index]:
            continue
        # The party list is in field order: the Pokemon in position i is `pokemon[i]` (a
        # switch exchanges two list entries and renumbers `Pokemon.slot`), so exchanging
        # the positions exchanges list entries 0 and 1 and `active` follows them.
        def moved(index: int | None) -> int | None:
            return index if index is None or index > 1 else 1 - index

        side.pokemon[0], side.pokemon[1] = side.pokemon[1], side.pokemon[0]
        for number, mon in enumerate(side.pokemon):
            mon.slot = number
        side.active = [moved(side.active[1]), moved(side.active[0])]
        side.mega_capable_slots = [moved(s) for s in side.mega_capable_slots]
        conditions = list(side.slot_conditions)
        while len(conditions) < 2:
            conditions.append([])
        conditions[0], conditions[1] = conditions[1], conditions[0]
        side.slot_conditions = conditions
        for mon in side.pokemon:
            if mon.active_index is not None:
                mon.active_index = 1 - mon.active_index
    for effect in out.effects():
        effect.source_slot = _swap_position_of_source(effect.source_slot, sides)
    # A charging move fires at the position it was aimed at: `twoturnmove.extra.targetLoc`
    # is Showdown's target location (+1/+2 a foe position, -1/-2 an ally position), seen from
    # the Pokemon that holds the volatile.
    for index, side in enumerate(out.sides):
        for mon in side.pokemon:
            for volatile in mon.volatiles:
                loc = volatile.extra.get("targetLoc") if volatile.extra else None
                if not isinstance(loc, int) or loc == 0:
                    continue
                if (loc > 0 and sides[1 - index]) or (loc < 0 and sides[index]):
                    volatile.extra["targetLoc"] = (3 - loc) if loc > 0 else (-3 - loc)
    return out


def _swap_slot_action(
    action: SlotAction, slot: int, own: bool, foe: bool
) -> SlotAction:
    if isinstance(action, MoveAction):
        target = action.target
        if target is not None:
            if target > 0 and foe:
                target = 3 - target
            elif target < 0 and own:
                target = -3 - target
        return MoveAction(
            slot=slot,
            move_index=action.move_index,
            move_id=action.move_id,
            target=target,
            mega=action.mega,
        )
    if isinstance(action, SwitchAction):
        return SwitchAction(slot=slot, party_index=action.party_index, species=action.species)
    return PassAction(slot=slot)


def swap_action(action: SideAction, own: bool, foe: bool) -> SideAction:
    """The same choice when the acting side (`own`) and/or the opposing side (`foe`) swapped."""
    slots = action.slots
    # (new position, the action that moves there)
    order = [(0, slots[1]), (1, slots[0])] if own and len(slots) == 2 else list(enumerate(slots))
    return SideAction(
        slots=tuple(_swap_slot_action(a, k, own, foe) for k, a in order)
    )


def holds_position_reader(pos: Position) -> bool:
    """Whether any Pokemon of either side has (or may bring) an ability that reads its position.

    Over the whole party, not only the field: a bench Ditto steps into whichever position
    the swap left open. `ability` is the ability as it is now; `base_ability` is the one a
    transformed Pokemon gets back.
    """
    for side in pos.sides:
        for mon in side.pokemon:
            if mon.ability in POSITION_READING_ABILITIES:
                return True
            if mon.base_ability in POSITION_READING_ABILITIES:
                return True
    return False


# ---------------------------------------------------------------------------
# Encoded arrays
# ---------------------------------------------------------------------------


def _feature_columns(mon_names: tuple[str, ...], side_names: tuple[str, ...]):  # noqa: ANN202
    mon_a, mon_b = mon_names.index("active_slot_0"), mon_names.index("active_slot_1")
    slot_a = [i for i, n in enumerate(side_names) if n.startswith("slot0_")]
    slot_b = [i for i, n in enumerate(side_names) if n.startswith("slot1_")]
    assert len(slot_a) == len(slot_b) > 0
    assert [side_names[i][6:] for i in slot_a] == [side_names[i][6:] for i in slot_b]
    return mon_a, mon_b, slot_a, slot_b


def _swap_rows(array: np.ndarray, rows_b: np.ndarray, rows_s: np.ndarray) -> np.ndarray:
    """A copy of a (B, 2, M, ...) array with Pokemon rows 0 and 1 exchanged at (b, s)."""
    out = array.copy()
    first = array[rows_b, rows_s, 0]
    out[rows_b, rows_s, 0] = array[rows_b, rows_s, 1]
    out[rows_b, rows_s, 1] = first
    return out


def swap_encoded(
    enc: Encoded,
    flip: np.ndarray,
    mon_names: tuple[str, ...],
    side_names: tuple[str, ...],
) -> Encoded:
    """`enc` with the two positions of one side exchanged where `flip[b, s]` is true.

    `flip` is (B, 2) bool. What `swap_positions` does to the position, on its arrays: rows
    0 and 1 of the side's Pokemon exchange (a row is a list entry, and the list is in field
    order), each Pokemon's `active_slot_0/1` columns exchange, and the side vector's
    `slot0_* / slot1_*` columns exchange. The bench rows, the field and the labels do not move.
    Returns a copy; `encode_positions(swap_positions(p))` equals this on `encode_positions(p)`.
    """
    mon_a, mon_b, slot_a, slot_b = _feature_columns(mon_names, side_names)
    flip = np.asarray(flip, dtype=bool)
    rows_b, rows_s = np.nonzero(flip)
    species, ability, item, moves = enc.species, enc.ability, enc.item, enc.moves
    mon, mask, side = enc.mon, enc.mask, enc.side
    if rows_b.size:
        species = _swap_rows(species, rows_b, rows_s)
        ability = _swap_rows(ability, rows_b, rows_s)
        item = _swap_rows(item, rows_b, rows_s)
        moves = _swap_rows(moves, rows_b, rows_s)
        mon = _swap_rows(mon, rows_b, rows_s)
        mask = _swap_rows(mask, rows_b, rows_s)
        picked = mon[rows_b, rows_s]  # (n, M, F), a copy
        first = picked[..., mon_a].copy()
        picked[..., mon_a] = picked[..., mon_b]
        picked[..., mon_b] = first
        mon[rows_b, rows_s] = picked
        side = side.copy()
        sp = side[rows_b, rows_s]  # (n, S), a copy
        first = sp[:, slot_a].copy()
        sp[:, slot_a] = sp[:, slot_b]
        sp[:, slot_b] = first
        side[rows_b, rows_s] = sp
    return Encoded(
        species=species,
        ability=ability,
        item=item,
        moves=moves,
        mon=mon,
        mask=mask,
        side=side,
        field=enc.field,
        unknown_volatiles=dict(enc.unknown_volatiles),
        decided=enc.decided,
    )


def position_reader_rows(ability: np.ndarray, mask: np.ndarray, imposter_id: int) -> np.ndarray:
    """(B,) bool: positions where some Pokemon on either side has the ability that reads its
    position (Imposter), so the swap is not known to leave the value alone. `imposter_id` is the
    ability's index in the vocabulary (0 when the regulation has none -> nothing is excluded).
    `ability` and `mask` are `Encoded`'s, (B, 2, M)."""
    if imposter_id <= 0:
        return np.zeros(len(ability), dtype=bool)
    return ((ability == imposter_id) & (mask > 0)).any(axis=(1, 2))


# ---------------------------------------------------------------------------
# Training: a random arrangement per row, per epoch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapSlots:
    """What the trainer needs to exchange positions inside a batch of tensors.

    `exclude` is a (N,) bool over the dataset's rows: a row marked there is never exchanged
    (the positions `position_reader_rows` names). None excludes nothing.
    """

    mon_a: int
    mon_b: int
    side_a: tuple[int, ...]
    side_b: tuple[int, ...]
    exclude: np.ndarray | None = None

    @staticmethod
    def of(
        mon_names: tuple[str, ...],
        side_names: tuple[str, ...],
        exclude: np.ndarray | None = None,
    ) -> SwapSlots:
        mon_a, mon_b, slot_a, slot_b = _feature_columns(mon_names, side_names)
        return SwapSlots(mon_a, mon_b, tuple(slot_a), tuple(slot_b), exclude)

    def draw(self, rng: np.random.Generator, index: np.ndarray) -> np.ndarray:
        """(B, 2) bool: each side of each row exchanged with probability 1/2, apart from
        `exclude`d rows. Four arrangements, equally likely."""
        flip = rng.random((len(index), 2)) < 0.5
        if self.exclude is not None:
            flip &= ~self.exclude[index][:, None]
        return flip


def swap_batch(batch: dict, flip, swap: SwapSlots) -> dict:  # noqa: ANN001
    """`swap_encoded` on a dict of tensors (what `Dataset.tensors` returns).

    `flip` is a (B, 2) bool numpy array. Torch is imported here so that `slotswap` stays
    importable without it (the position-level check does not need it).
    """
    import torch  # noqa: PLC0415

    device = batch["mon"].device
    flip_t = torch.from_numpy(np.asarray(flip, dtype=bool)).to(device)

    def rows(x):  # noqa: ANN001, ANN202
        both = x.clone()
        both[:, :, 0] = x[:, :, 1]
        both[:, :, 1] = x[:, :, 0]
        shape = flip_t.shape + (1,) * (x.dim() - 2)
        return torch.where(flip_t.view(shape), both, x)

    out = dict(batch)
    for key in ("species", "ability", "item", "moves", "mon", "mask"):
        out[key] = rows(batch[key])
    mon = out["mon"].clone()
    a, b = swap.mon_a, swap.mon_b
    column_a, column_b = mon[..., a].clone(), mon[..., b].clone()
    flag = flip_t.unsqueeze(-1)
    mon[..., a] = torch.where(flag, column_b, column_a)
    mon[..., b] = torch.where(flag, column_a, column_b)
    out["mon"] = mon
    side = batch["side"].clone()
    sa, sb = list(swap.side_a), list(swap.side_b)
    block_a, block_b = side[..., sa].clone(), side[..., sb].clone()
    side[..., sa] = torch.where(flag, block_b, block_a)
    side[..., sb] = torch.where(flag, block_a, block_b)
    out["side"] = side
    return out
