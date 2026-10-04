"""IKA-429's bind columns (縛り): Python places what the port computes, and they read the position.

Encoding revision 4 appends four columns to each Pokemon row and seven to each side: who
moves first and knocks out whom (`rust/src/bind.rs`). Only the port has a damage
calculator, so `Encoder.encode_positions` asks it (the `bind` request) and writes the
answer into its own arrays. Held here:

* Python's arrays are the bytes of the port's own `encode` subcommand, all eight of them,
  on positions whose bind columns are not zero (asserted, so agreement is not two zeros).
* the columns move with the position: a foe brought down to 1 HP is knocked out by every
  attacker that reaches it, a bench row reads 0, and swapping the sides swaps the columns.
* `bind=False` leaves them 0 and does not ask the port (`beliefnode._patched`).

The positive control is a build with `--features ika429-control` (the encoder writes the
side's bind columns one slot along; the `bind` request does not), pointed at through
`POKEURAOU_RUST_NODE_BIN`: the first test fails on it.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.encode import BIND_MON_FEATURES, BIND_SIDE_FEATURES, Encoder
from pokeuraou.position import Position
from pokeuraou.regulation import load_regulation
from tests.test_encode_state_port import FORMAT, _base, _port, _with_state


def _one_hp(base: dict, side: int) -> dict:
    """`base` with the first active Pokemon of `side` at 1 HP."""
    pos = copy.deepcopy(base)
    one = pos["sides"][side]
    mon = one["pokemon"][one["active"][0]]
    mon["hp"] = 1
    return pos


def _trick_room(pos: dict) -> dict:
    out = copy.deepcopy(pos)
    out["field"].setdefault("pseudoWeather", []).append({"id": "trickroom", "duration": 3})
    return out


@pytest.fixture(scope="module")
def positions() -> list[Position]:
    base = _base()
    raw = [
        base,
        _one_hp(base, 1),
        _one_hp(base, 0),
        _trick_room(_one_hp(base, 1)),
        _with_state(_one_hp(base, 1), durations=True),
    ]
    return [Position.from_json(p) for p in raw]


def _encoded(positions: list[Position]):  # noqa: ANN202
    return Encoder(load_regulation(FORMAT)).encode_positions(positions)


def test_python_writes_the_bind_columns_the_port_writes(
    positions: list[Position], tmp_path: Path
) -> None:
    encoder = Encoder(load_regulation(FORMAT))
    want = encoder.encode_positions(positions)
    k_mon, k_side = len(BIND_MON_FEATURES), len(BIND_SIDE_FEATURES)
    assert want.mon[..., -k_mon:].any() and want.side[..., -k_side:].any()
    header, body = _port(positions, tmp_path)
    widths = (header["monWidth"], header["sideWidth"], header["fieldWidth"])
    assert widths == (encoder.widths["mon"], encoder.widths["side"], encoder.widths["field"])
    n, m = header["positions"], header["monsPerSide"]
    layout = [
        ("species", np.int32, (n, 2, m)),
        ("ability", np.int32, (n, 2, m)),
        ("item", np.int32, (n, 2, m)),
        ("moves", np.int32, (n, 2, m, 4)),
        ("mon", np.float32, (n, 2, m, widths[0])),
        ("mask", np.float32, (n, 2, m)),
        ("side", np.float32, (n, 2, widths[1])),
        ("field", np.float32, (n, widths[2])),
    ]
    offset, differing = 0, []
    for name, dtype, shape in layout:
        count = int(np.prod(shape))
        got = np.frombuffer(body, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += count * np.dtype(dtype).itemsize
        if got.astype(getattr(want, name).dtype).tobytes() != getattr(want, name).tobytes():
            differing.append(name)
    assert differing == []


def test_the_bind_columns_read_the_position(positions: list[Position]) -> None:
    encoder = Encoder(load_regulation(FORMAT))
    enc = encoder.encode_positions(positions)
    mon_k = {n: k for k, n in enumerate(encoder.mon_names)}
    side_k = {n: k for k, n in enumerate(encoder.side_names)}
    base, foe_low, own_low, trick_room = 0, 1, 2, 3
    # Bench rows read 0 in every bind column.
    active = enc.mon[..., mon_k["is_active"]] > 0
    assert not enc.mon[~active][:, -len(BIND_MON_FEATURES) :].any()
    # The foe at 1 HP: every attacker that reaches it knocks it out with all 16 rolls, so
    # its `ko_in` is at least the base position's and not below 1/2 (one attacker of two).
    theirs = positions[foe_low].sides[1]
    p = theirs.active[0]
    assert enc.mon[foe_low, 1, p, mon_k["ko_in"]] >= 0.5
    assert enc.mon[foe_low, 1, p, mon_k["ko_in"]] >= enc.mon[base, 1, p, mon_k["ko_in"]]
    # ... and the side that faces it has a bound foe whenever one of its attackers moves first.
    if enc.mon[foe_low, 1, p, mon_k["bind_in"]] > 0:
        assert enc.side[foe_low, 0, side_k["foes_bound"]] > 0
    # The same Pokemon, the other way round: our own Pokemon at 1 HP.
    ours = positions[own_low].sides[0]
    q = ours.active[0]
    assert enc.mon[own_low, 0, q, mon_k["ko_in"]] >= 0.5
    # Trick Room reverses who moves first, so the binds on the 1-HP foe change unless every
    # attacker reaching it does so with a priority move or at a Speed tie.
    assert enc.mon[trick_room, 1, p, mon_k["ko_in"]] == enc.mon[foe_low, 1, p, mon_k["ko_in"]]
    assert (
        enc.mon[trick_room, 1, p, mon_k["bind_in"]] != enc.mon[foe_low, 1, p, mon_k["bind_in"]]
    ), "Trick Room moved no bind on the 1-HP foe; choose a position where it should"


def test_swapping_the_sides_swaps_the_bind_columns(positions: list[Position]) -> None:
    encoder = Encoder(load_regulation(FORMAT))
    enc = encoder.encode_positions(positions)
    swapped = []
    for pos in positions:
        raw = pos.to_json()
        raw["sides"] = raw["sides"][::-1]
        swapped.append(Position.from_json(raw))
    other = encoder.encode_positions(swapped)
    assert other.mon.tobytes() == enc.mirror().mon.copy().tobytes()
    assert other.side.tobytes() == enc.mirror().side.copy().tobytes()


def test_without_bind_the_columns_are_zero_and_the_rest_is_the_same(
    positions: list[Position],
) -> None:
    encoder = Encoder(load_regulation(FORMAT))
    full = encoder.encode_positions(positions)
    bare = encoder.encode_positions(positions, bind=False)
    k_mon, k_side = len(BIND_MON_FEATURES), len(BIND_SIDE_FEATURES)
    assert not bare.mon[..., -k_mon:].any() and not bare.side[..., -k_side:].any()
    assert bare.mon[..., :-k_mon].tobytes() == full.mon[..., :-k_mon].tobytes()
    assert bare.side[..., :-k_side].tobytes() == full.side[..., :-k_side].tobytes()
