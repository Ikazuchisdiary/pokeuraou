"""IKA-389: a read's children's menus by the Q, built in the port -- off by default.

`deepen._q_menus` built each child's two menus in the worker's Python: both sides' legal
pools (`qhead.legal_pool`: `side_actions` less `drop_dead_actions`), the Q's arrays
(`qrank._pool_arrays`: the position's encoding, `qhead.encode_actions`, and a `qfeatures`
crossing to the port a child), a `q_batch` request to the inference server per
`deepen.Q_MENU_CHUNK` children, the game on each matrix, and each side's `width` best by
`narrow` (no cover) -- which also asked the port for both pools' damage scores, only to
write them into a detail line nobody reads. Here one crossing does all of it (`qMenus`,
`rust/src/qmenus.rs`): the port lists the pools (`rust/src/legal.rs`), lays the same arrays
into the Q's own block (`RemoteQ._block`, idle while the port answers) and sends the server
the same requests, solves the games (`lp.rs`, scipy's HiGHS), scores the candidates with
numpy's own BLAS (`blas`: at an equilibrium the support's values tie up to their last bits,
so the order among them is those bits) and hands back the menus alone.

    POKEURAOU_LADDER_PORT_MENUS=1     # on (default off; implies `portserved` and `portlp`)

Only where the Q is an inference server's (`qrank.RemoteQ`), its encoder `encode.Encoder`,
and numpy's BLAS is an OpenBLAS the port can load; anything else takes the old road. The
menus are `_q_menus`' to the bit (records/IKA-389.md), so a read is the same read.
"""

from __future__ import annotations

import ctypes
import functools
import glob
import os
from collections.abc import Sequence
from typing import Any

from . import portserved, rustnode
from .actions import MoveAction, PassAction, SideAction, SlotAction, SwitchAction

ENV = "POKEURAOU_LADDER_PORT_MENUS"
#: Whether a read's children's Q menus are built in the port. A list so a test (and a
#: ladder worker, told by its reader) can turn it on in one process.
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

#: The gemv and thread-count symbols of numpy's OpenBLAS, by its build's prefix.
_SYMBOLS = (
    ("scipy_cblas_dgemv64_", "scipy_openblas_set_num_threads64_"),
    ("cblas_dgemv64_", "openblas_set_num_threads64_"),
)


@functools.cache
def blas() -> dict[str, str] | None:
    """numpy's own OpenBLAS -- the library file numpy loaded and its 64-bit-integer
    `cblas_dgemv` -- for the port to score with (`rust/src/blas.rs`), or None where there is
    none it can load, and the old road is taken."""
    import numpy as np

    root = os.path.dirname(np.__file__)
    found = sorted(
        path
        for pattern in ("../numpy.libs/*openblas*", ".libs/*openblas*", "../numpy.libs/*.so*")
        for path in glob.glob(os.path.join(root, pattern))
        if "openblas" in os.path.basename(path).lower()
    )
    for path in found:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for gemv, threads in _SYMBOLS:
            if hasattr(lib, gemv):
                return {"path": os.path.realpath(path), "gemv": gemv, "threads": threads}
    return None


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
    """`deepen._q_menus(reg, positions, width)` in one crossing, or None where this road
    does not go (then the caller builds them itself)."""
    if not ON[0] or not positions:
        return None
    from . import port, qrank
    from .encode import Encoder
    from .inference import request_failed

    model = qrank._INSTALLED.get("")  # noqa: SLF001 - absent: the old road stops as it would
    if type(model) is not qrank.RemoteQ or type(model.encoder) is not Encoder:
        return None
    library = blas()
    if library is None:
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
        "blas": library,
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


__all__ = ["COUNTS", "ENV", "ON", "blas", "counts", "q_menus", "set_on"]
