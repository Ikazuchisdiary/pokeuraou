"""IKA-389: a depth-2 read's children's menus by the Q, built in the port -- off by default.

`child_menus` builds each child's two menus in Python: both sides' legal pools
(`qhead.legal_pool`: `side_actions` less `drop_dead_actions`), the Q's arrays
(`qrank._pool_arrays`: the position's encoding, `qhead.encode_actions`, and a `qfeatures`
crossing to the port a child), a `q_batch` request to the inference server per
`Q_MENU_CHUNK` children, the game on each matrix, and each side's `width` best by `narrow`
(no cover) -- which also asked the port for both pools' damage scores, only to write them
into a detail line nobody reads. With the switch on, one crossing does all of it
(`q_menus`, `qMenus`, `rust/src/qmenus.rs`): the port lists the pools
(`rust/src/legal.rs`), lays the same arrays into the Q's own block (`RemoteQ._block`, idle
while the port answers) and sends the server the same requests, solves the games (`lp.rs`,
scipy's HiGHS), scores the candidates with plain row and column products (IKA-460; they
were numpy's BLAS) and hands back the menus alone.

    POKEURAOU_LADDER_PORT_MENUS=1     # on (default off; implies `portserved` and `portlp`)

The switch's name is from the read that set it, the ladder (removed in IKA-464); what reads
these menus now is the depth-2 refine of a list of cells (`search._refine_cells` with
``child_q``). The port's deepening (`portdeepen`) builds its children's menus with the same
`qmenus.rs` inside its own read and does not come here; this module lends it `_action`.

Only where the Q is an inference server's (`qrank.RemoteQ`), its encoder `encode.Encoder`;
anything else takes the Python road. The menus are the Python road's except that the order
among tied candidates can differ (the last bits of the scores; records/IKA-389.md, IKA-460).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from . import portserved, rustnode
from .actions import MoveAction, PassAction, SideAction, SlotAction, SwitchAction

ENV = "POKEURAOU_LADDER_PORT_MENUS"
#: Whether a depth-2 read's children's Q menus are built in the port. A list so a test can
#: turn it on in one process.
ON = [os.environ.get(ENV, "0").strip() == "1"]
#: What went: crossings, positions asked about, of them the ones with a game (the positive
#: control), the Q's requests the port sent, and games the LP could not solve.
COUNTS = {"crossings": 0, "positions": 0, "asked": 0, "requests": 0, "unsolved": 0,
          # IKA-389 B3b: depth-2 turns whose branches stayed in the port, and those written
          # out whole because another completion's cells are read off them.
          "bareTurns": 0, "wholeTurns": 0}


BARE_ENV = "POKEURAOU_LADDER_PORT_BARE"
#: B3b: whether a depth-2 call's turns leave their branches in the port. Off by default; on
#: turns this road (and so `portserved` and `portlp`) on.
BARE_ON = [os.environ.get(BARE_ENV, "0").strip() == "1"]


def set_bare(on: bool) -> None:
    BARE_ON[0] = bool(on)
    if on:
        set_on(True)


def bare_turns() -> bool:
    """Whether a depth-2 call's turns leave their branches in the port (B3b): with the flag
    on and positions held (`rustnode.hold_positions`). The branches are then named by number
    -- to the port's menus and fills -- and a branch read otherwise is written out then
    (`rustnode.HeldPosition`)."""
    return BARE_ON[0] and ON[0] and rustnode._HOLD[0]  # noqa: SLF001


def set_on(on: bool) -> None:
    """Turn the road on or off in this process; on turns `portserved` (and `portlp`) on."""
    ON[0] = bool(on)
    if on:
        portserved.set_on(True)


set_on(ON[0] or BARE_ON[0])

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


def q_menus(
    reg: Any, positions: Sequence[Any], width: int, chunk: int  # noqa: ANN401 - a Regulation
) -> list[tuple[list[SideAction], list[SideAction]]] | None:
    """`child_menus(reg, positions, width)` in one crossing, or None where this road does
    not go (then `child_menus` builds them in Python)."""
    if not ON[0] or not positions:
        return None
    from . import actions

    if actions.ALLY_TARGETS[0] != "off":
        # IKA-181: the port's `legal.rs` lists no ally target of a `normal` move.
        return None
    from . import port, qrank
    from .encode import Encoder
    from .inference import request_failed

    model = qrank._INSTALLED.get("")  # noqa: SLF001 - absent: the old road stops as it would
    if type(model) is not qrank.RemoteQ or type(model.encoder) is not Encoder:
        return None
    encoder = model.encoder
    extra = {
        "width": int(width),
        "chunk": int(chunk),
        "properties": bool(model.properties),
        "megaFromSlots": bool(encoder.rules.mega_from_slots),
        "server": {
            "address": model.address,
            "model": model.model,
            "shm": model._block.name,  # noqa: SLF001 - the Q's own block, idle while the port asks
            "bytes": int(model.buffer_bytes),
        },
    }
    answer = port.ask(reg, lambda node: node.q_menus(list(positions), extra))
    served = answer.get("served") or {}
    requests = int(served.get("requests", 0))
    waited = float(served.get("waitUs", 0.0)) / 1e6
    # The server's share of the port's answer is the Q's clock, as it was on the old road.
    rustnode.PORT_WAITED[0] -= waited
    model.waited += waited
    model.calls += requests
    model.trips += requests
    COUNTS["crossings"] += 1
    COUNTS["requests"] += requests
    if "tooBig" in answer:
        raise RuntimeError(str(answer["tooBig"]))
    if "serverError" in answer:
        raise request_failed({"error": answer["serverError"], "oom": answer.get("oom"),
                              "capGb": answer.get("capGb")}, "Q request failed")
    if "invalid" in answer:
        raise ValueError(str(answer["invalid"]))
    menus = answer["menus"]
    if len(menus) != len(positions):
        raise RuntimeError(f"`qMenus` answered {len(menus)} of {len(positions)} positions")
    asked = int(answer.get("asked", 0))
    # `Encoder.encode_positions`' own count, one position a request as `_pool_arrays` encodes.
    if asked:
        encoder.note("python can_mega=" + ("slots" if encoder.rules.mega_from_slots else "holder"),
                     asked)
    COUNTS["positions"] += len(positions)
    COUNTS["asked"] += asked
    COUNTS["unsolved"] += int(answer.get("unsolved", 0))
    return [([_action(a) for a in rows], [_action(a) for a in cols]) for rows, cols in menus]


#: Positions a `child_menus` forward pass takes at most (IKA-362).
Q_MENU_CHUNK = 16


def child_menus(
    reg: Any, positions: Sequence[Any], width: int  # noqa: ANN401 - a Regulation
) -> list[tuple[list[SideAction], list[SideAction]]]:
    """Both sides' menus in each of `positions`: the `width` best of each side's whole pool
    (`qhead.legal_pool`, what `narrow` ranks) by the process's Q solved over both pools --
    the root's ``q-nocover`` ranking (`qrank.q_ranking`), without the cover. The Q is
    asked for every position in one forward pass (`batched`), `Q_MENU_CHUNK` positions at
    most. What `search._refine_cells` reads for ``child_q``; it was `deepen._q_menus` until
    IKA-464 removed Python's deepening.
    """
    import numpy as np

    from . import portlp, qhead, qrank
    from .equilibrium import EquilibriumError, solve
    from .narrow import narrow

    # IKA-389: all of it in the port in one crossing (`q_menus`, off by default): the same
    # menus, to the bit.
    ported = q_menus(reg, positions, width, Q_MENU_CHUNK)
    if ported is not None:
        return ported
    asks = [
        (pos, (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1)))
        for pos in positions
    ]
    wanted = [n for n, (_pos, pools) in enumerate(asks) if pools[0] and pools[1]]
    # In chunks of `Q_MENU_CHUNK` positions: a depth-2 read's children in one pass asked the
    # card for more than a process's cap holds (IKA-362).
    got: list = []
    for start in range(0, len(wanted), Q_MENU_CHUNK):
        got.extend(qrank.installed().batched(
            reg, [asks[n] for n in wanted[start:start + Q_MENU_CHUNK]]
        ))
    matrices = dict(zip(wanted, got, strict=True)) if wanted else {}
    # IKA-381: every game of the call solved in the port in one crossing (`portlp`, off by
    # default): the same `Equilibrium`s, each error where `solve` would have raised it.
    solved = (dict(zip(wanted, portlp.solve_many(
        reg, [np.asarray(matrices[n], dtype=np.float64) for n in wanted]), strict=True))
        if portlp.ON[0] and wanted else None)
    out: list[tuple[list[SideAction], list[SideAction]]] = []
    for n, (pos, pools) in enumerate(asks):
        if n not in matrices:
            out.append(([], []))
            continue
        q = np.asarray(matrices[n], dtype=np.float64)
        try:
            if solved is None:
                e = solve(q)
            else:
                e = solved[n]
                if isinstance(e, Exception):
                    raise e
            scores = (np.asarray(e.row_ev), -np.asarray(e.col_ev))
        except (EquilibriumError, ValueError):
            # The mean against every reply, as `qrank.q_ranking` falls back to.
            scores = (q.mean(axis=1), -q.mean(axis=0))
        menu: list[list[SideAction]] = []
        for side in (0, 1):
            by = {a.to_choice(): float(s) for a, s in zip(pools[side], scores[side], strict=True)}

            def rank(pool: list[SideAction], _scored: object = None, by: dict = by) -> list[float]:
                return [by[a.to_choice()] for a in pool]

            menu.append(
                narrow(reg, pos, side, limit=width, candidates=pools[side], rank=rank,
                       cover=False).actions
            )
        out.append((menu[0], menu[1]))
    return out


def counts() -> dict[str, int]:
    return dict(COUNTS)


#: What a ladder read reports of this road (`tally`), by name.
TALLY = ("portMenus", "portBare", "portWholeTurns", "portWrittenOut")


def tally() -> tuple[int, int, int, int]:
    """The process's totals a read reports the change of: children whose menus the port built,
    branches left in the port, turns written out whole for a share, and kept branches written
    out after all (the positive controls of B3a and B3b)."""
    return (COUNTS["asked"], rustnode.BARE["branches"], COUNTS["wholeTurns"],
            rustnode.BARE["materialized"])


__all__ = ["COUNTS", "ENV", "ON", "Q_MENU_CHUNK", "child_menus", "counts", "q_menus", "set_on"]
