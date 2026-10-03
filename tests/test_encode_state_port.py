"""IKA-425's state columns: the port writes the bytes `encode.py` writes, and both read the state.

Encoding revision 3 appends to each block what the canonical position carried and revision
2 did not read: turns left of Trick Room, Tailwind and the three screens, the sleep and
toxic counters, Perish Song's count, and the locked and the last move. The positions here
carry every one of them, on both sides, on the field and on the bench, beside a position
with none and one whose effects have no duration. The port's `encode` subcommand and
`Encoder.encode_positions` are compared array by array, bit for bit, and the new columns
are asserted to hold the values the position says -- so a comparison of two zero columns
cannot pass for agreement.

The positive controls are a port built before the change (the widths differ and the run
stops) and a build with `--features ika425-control` (the side's state columns one slot
along), each pointed at through `POKEURAOU_RUST_NODE_BIN`: every comparison here fails on
both.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.encode import (
    STATE_MON_FEATURES,
    STATE_PSEUDO_WEATHERS,
    STATE_SIDE_CONDITIONS,
    Encoder,
)
from pokeuraou.position import Position
from pokeuraou.regulation import load_regulation, regulation_dir
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

FORMAT = "gen9championsvgc2026regmb"


def _base() -> dict:
    reg = load_regulation(FORMAT)
    sets = list(load_roster("rizabanadohido").sets)
    m = reg.meta.picked_team_size
    return position_from_sets(reg, sets[:m], sets[2 : 2 + m]).to_json()


def _with_state(base: dict, durations: bool) -> dict:
    """Every state the columns read, on both sides; with or without durations."""
    pos = copy.deepcopy(base)
    pos["turn"] = 4

    def effect(eid: str, duration: int, **extra: object) -> dict:
        out: dict = {"id": eid, **extra}
        if durations:
            out["duration"] = duration
        return out

    pos["field"].setdefault("pseudoWeather", []).append(effect("trickroom", 3))
    ours, theirs = pos["sides"]
    ours.setdefault("sideConditions", []).extend([effect("tailwind", 2), effect("reflect", 5)])
    theirs.setdefault("sideConditions", []).extend(
        [effect("lightscreen", 7), effect("auroraveil", 1)]
    )
    a, b, bench = ours["pokemon"][0], ours["pokemon"][1], theirs["pokemon"][2]
    a["status"], a["statusCounter"] = "slp", 2
    theirs["pokemon"][1]["status"], theirs["pokemon"][1]["statusCounter"] = "tox", 3
    b.setdefault("volatiles", []).extend(
        [effect("perishsong", 2), {"id": "choicelock", "move": b["moves"][0]["id"]}]
    )
    theirs["pokemon"][0]["lastMove"] = theirs["pokemon"][0]["moves"][1]["id"]
    bench.setdefault("volatiles", []).append({"id": "encore", "move": bench["moves"][2]["id"]})
    # A burned Pokemon's counter is not a sleep or toxic counter.
    ours["pokemon"][2]["status"], ours["pokemon"][2]["statusCounter"] = "brn", 5
    return pos


@pytest.fixture(scope="module")
def positions() -> list[Position]:
    base = _base()
    return [
        Position.from_json(p)
        for p in (base, _with_state(base, durations=True), _with_state(base, durations=False))
    ]


def _port(positions: list[Position], tmp_path: Path) -> tuple[dict, bytes]:
    binary = rustnode.binary_path()
    if not binary.exists():
        pytest.fail(f"no Rust binary at {binary}; `cargo build --release`")
    fixture = tmp_path / "turns.json"
    fixture.write_bytes(
        json.dumps({"format_id": FORMAT, "positions": [p.to_json() for p in positions]}).encode()
    )
    out = tmp_path / "encoded.bin"
    done = subprocess.run(  # noqa: S603
        [str(binary), "encode", str(regulation_dir() / f"{FORMAT}.json"), str(fixture), str(out)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert done.returncode == 0, done.stderr
    raw = out.read_bytes()
    newline = raw.index(b"\n")
    return json.loads(raw[:newline]), raw[newline + 1 :]


def test_the_port_writes_the_state_columns_python_writes(
    positions: list[Position], tmp_path: Path
) -> None:
    encoder = Encoder(load_regulation(FORMAT))
    want = encoder.encode_positions(positions)
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


def test_the_state_columns_read_the_position(positions: list[Position]) -> None:
    """The columns compared above hold the state, so their agreement is not two zeros."""
    reg = load_regulation(FORMAT)
    encoder = Encoder(reg)
    enc = encoder.encode_positions(positions)
    mon = {n: k for k, n in enumerate(encoder.mon_names)}
    side = {n: k for k, n in enumerate(encoder.side_names)}
    field = {n: k for k, n in enumerate(encoder.field_names)}
    base, full, bare = 0, 1, 2
    # Nothing set: every state column is zero.
    for name in STATE_MON_FEATURES:
        assert not enc.mon[base, ..., mon[name]].any()
    with_state = positions[full]
    ours, theirs = with_state.sides
    expected_mon = {
        (0, 0, "sleep_turns"): 2 / 3,
        (1, 1, "toxic_stage"): 3 / 8,
        (0, 1, "perish_turns"): 2 / 4,
        (0, 1, "locked_move_id"): encoder.vocab.moves[ours.pokemon[1].moves[0].id],
        (1, 0, "last_move_id"): encoder.vocab.moves[theirs.pokemon[0].moves[1].id],
        (1, 2, "locked_move_id"): encoder.vocab.moves[theirs.pokemon[2].moves[2].id],
        (0, 2, "sleep_turns"): 0.0,  # burned, counter 5
        (0, 2, "toxic_stage"): 0.0,
    }
    for (s, p, name), value in expected_mon.items():
        assert enc.mon[full, s, p, mon[name]] == np.float32(value), (s, p, name)
    expected_side = {
        (0, "tailwind"): 2 / 8, (0, "reflect"): 5 / 8,
        (1, "lightscreen"): 7 / 8, (1, "auroraveil"): 1 / 8,
        (1, "tailwind"): 0.0, (0, "auroraveil"): 0.0,
    }
    for (s, cid), value in expected_side.items():
        assert enc.side[full, s, side[f"side_{cid}_turns"]] == np.float32(value), (s, cid)
    assert enc.field[full, field["pseudo_trickroom_turns"]] == np.float32(3 / 8)
    # No durations: the presence columns are set and the turns columns read zero.
    for cid in STATE_SIDE_CONDITIONS:
        assert not enc.side[bare, :, side[f"side_{cid}_turns"]].any()
    for pid in STATE_PSEUDO_WEATHERS:
        assert enc.field[bare, field[f"pseudo_{pid}"]] == 1.0
        assert enc.field[bare, field[f"pseudo_{pid}_turns"]] == 0.0
    # The state columns are the trailing ones of each revision-3 block (IKA-429's bind
    # columns come after them).
    narrow, state = encoder.base_widths, encoder.state_widths
    assert encoder.mon_names[narrow["mon"] : state["mon"]] == STATE_MON_FEATURES
    assert state["side"] - narrow["side"] == len(STATE_SIDE_CONDITIONS)
    assert state["field"] - narrow["field"] == len(STATE_PSEUDO_WEATHERS)
