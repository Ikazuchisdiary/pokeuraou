"""The port's compact action codes read back as `SideAction`s (IKA-389).

The port writes an action of a side as a list of slot codes (`legal::Act::to_json`):
``["m", slot, move index, move id, target, mega]``, ``["s", slot, party index, species]`` or
``["p", slot]``. `_action` reads one back, kept by its code, so the same code is the same
object. The menus the port builds (`rootmenus`), the port's deepening's root and its grown
menus (`portdeepen`) and the legal pools the port lists read their actions here.

IKA-389 built a depth-2 read's children's menus by the Q in the port here too (`q_menus`,
behind ``POKEURAOU_LADDER_PORT_MENUS``) beside the Python road (`child_menus`); IKA-464 removed
both with the depth-2 read (`search._refine_cells`). The port's deepening builds its
children's menus in its own read (`rust/src/qmenus.rs`).
"""

from __future__ import annotations

from .actions import MoveAction, PassAction, SideAction, SlotAction, SwitchAction

_SLOTS: dict[tuple, SlotAction] = {}
_ACTIONS: dict[tuple, SideAction] = {}


def _slot(code: list) -> SlotAction:
    """A slot action from the port's compact form (`legal::Act::to_json`)."""
    key = tuple(code)
    found = _SLOTS.get(key)
    if found is None:
        kind = code[0]
        if kind == "m":
            found = MoveAction(slot=int(code[1]), move_index=int(code[2]), move_id=str(code[3]),
                               target=None if code[4] is None else int(code[4]), mega=bool(code[5]))
        elif kind == "s":
            found = SwitchAction(slot=int(code[1]), party_index=int(code[2]), species=str(code[3]))
        elif kind == "p":
            found = PassAction(slot=int(code[1]))
        else:
            raise ValueError(f"the port wrote an action of kind {kind!r}")
        _SLOTS[key] = found
    return found


def _action(codes: list) -> SideAction:
    key = tuple(tuple(code) for code in codes)
    found = _ACTIONS.get(key)
    if found is None:
        found = _ACTIONS[key] = SideAction(slots=tuple(_slot(code) for code in codes))
    return found


__all__: list[str] = []
