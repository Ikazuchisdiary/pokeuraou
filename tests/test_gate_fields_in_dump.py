"""The port's gate on move fields reads the regulation dump, so the dump must write them (IKA-246).

`UNHANDLED_MOVE_FIELDS` (rust/src/resolve.rs) refuses a move that carries one of its fields,
and `reg.rs` looks for the field in the move's entry of `configs/regulations/*.json`. That
file is written by `DECLARATIVE_MOVE_KEYS` (packages/sim-bridge/src/regulation.ts): a field
the dump does not write is a gate that never fires. `multiaccuracy` was one until IKA-235;
`mindBlownRecoil` (Steel Beam), `sleepUsable` (Sleep Talk, Snore), `stealsBoosts` (Spectral
Thief) and `struggleRecoil` (Struggle) were the rest until IKA-246, and those moves went
through with the field unread.

Two fields of the gate are functions, not values -- `damageCallback` and `onHitField` -- and
the dump names those in a move's `customHooks` instead; `HOOK_FIELDS` holds them apart.

`struggleRecoil` left the gate when the dump began to carry it: the port models Struggle's
recoil by name (`level_struggle::struggle_recoil`, IKA-239), and left in, the gate would have
refused every Struggle.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from pokeuraou.actions import MoveAction, SideAction, side_actions
from pokeuraou.oracle import ORACLE_JS, TeamSet
from pokeuraou.position import MoveSlot, Position
from pokeuraou.regulation import Regulation

from ._port import Budget, PortRefused, resolve_turn
from .test_actions import _synthetic_position

ROOT = Path(__file__).resolve().parents[1]
REGULATIONS = ["gen9championsvgc2026regmb", "gen9championsvgc2026regmc"]

#: Gate fields that are functions: the dump lists them in `customHooks`, not as a key.
HOOK_FIELDS = frozenset({"damageCallback", "onHitField"})

#: Gate field -> the moves carrying it that M-C's dump holds. The port models none of them.
GATED = {
    "steelbeam": "mindBlownRecoil",
    "sleeptalk": "sleepUsable",
    "snore": "sleepUsable",
}


def _gate_fields() -> list[str]:
    source = (ROOT / "rust" / "src" / "resolve.rs").read_text(encoding="utf-8")
    body = source[source.index("pub(crate) const UNHANDLED_MOVE_FIELDS") :]
    body = body[body.index("= [") : body.index("];")]
    return re.findall(r'"([A-Za-z]+)"', body)


def _declarative_keys() -> set[str]:
    source = (ROOT / "packages" / "sim-bridge" / "src" / "regulation.ts").read_text(encoding="utf-8")
    body = source[source.index("const DECLARATIVE_MOVE_KEYS = [") :]
    body = body[: body.index("] as const;")]
    code = "\n".join(line.split("//")[0] for line in body.splitlines())
    return set(re.findall(r"'([A-Za-z]+)'", code))


def _dump(regulation: str) -> dict[str, dict]:
    doc = json.loads((ROOT / "configs" / "regulations" / f"{regulation}.json").read_text(encoding="utf-8"))
    return {m["id"]: m for m in doc["moves"]}


#: Every move of the format's dex with a non-empty value for one of the fields; a function
#: is reported as "<function>". Read from the same `pokemon-showdown` the dump is built from.
DEX_SCRIPT = """
const {Dex} = require('pokemon-showdown');
const fields = JSON.parse(process.argv[1]);
const dex = Dex.forFormat(Dex.formats.get(process.argv[2]));
const out = {};
for (const f of fields) {
  out[f] = {};
  for (const m of dex.moves.all()) {
    const v = m[f];
    if (v === undefined || v === null || v === false) continue;
    out[f][m.id] = typeof v === 'function' ? '<function>' : v;
  }
}
console.log(JSON.stringify(out));
"""


@cache
def _dex_carriers(regulation: str, fields: tuple[str, ...]) -> dict[str, dict[str, object]]:
    node = shutil.which("node")
    if node is None or not ORACLE_JS.exists():
        pytest.skip("needs node and the built sim-bridge, as the oracle tests do")
    done = subprocess.run(
        [node, "-e", DEX_SCRIPT, json.dumps(list(fields)), regulation],
        cwd=ROOT / "packages" / "sim-bridge",
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


def test_the_gate_lists_what_it_did() -> None:
    """The parse is not vacuous: the gate is read, and the hook fields are in it."""
    fields = _gate_fields()
    assert "multiaccuracy" in fields and "mindBlownRecoil" in fields, fields
    assert set(fields) >= HOOK_FIELDS, fields
    assert "struggleRecoil" not in fields, "Struggle's recoil is the port's (IKA-239)"


def test_every_gate_field_is_a_key_the_dump_writes() -> None:
    declared = _declarative_keys()
    assert "multiaccuracy" in declared, declared
    missing = [f for f in _gate_fields() if f not in declared and f not in HOOK_FIELDS]
    assert missing == [], f"the dump never writes {missing}: the gate cannot fire for them"
    assert not HOOK_FIELDS & declared, "a function in DECLARATIVE_MOVE_KEYS is dropped by JSON"


@pytest.mark.oracle
@pytest.mark.parametrize("regulation", REGULATIONS)
def test_every_move_the_dex_gives_a_gate_field_carries_it_in_the_dump(regulation: str) -> None:
    """Against the dex itself, so a field the dump drops shows up whichever move carries it."""
    fields = tuple(_gate_fields())
    carriers = _dex_carriers(regulation, fields)
    dump = _dump(regulation)
    wrong: list[str] = []
    checked = 0
    for field in fields:
        for move_id, value in carriers[field].items():
            entry = dump.get(move_id)
            if entry is None:  # not in this format (isNonstandard and the like)
                continue
            checked += 1
            if value == "<function>":
                if field not in entry.get("customHooks", []):
                    wrong.append(f"{move_id}.{field} (hook)")
            elif entry.get(field) != value:
                wrong.append(f"{move_id}.{field}={value!r}, dump {entry.get(field)!r}")
    assert wrong == [], wrong
    # 7 damageCallback, 2 multiaccuracy, 6 selfdestruct, 1 mindBlownRecoil, 1 smartTarget,
    # 2 sleepUsable, 4 onHitField, 3 willCrit: a count that fell would mean the dex was not read.
    assert checked >= 26, checked


@pytest.mark.parametrize("regulation", REGULATIONS)
def test_struggle_recoil_is_in_the_dump(regulation: str) -> None:
    """Out of the gate, but still written: the field is the dex's, and the dump copies it."""
    assert _dump(regulation)["struggle"].get("struggleRecoil") is True


# ---------------------------------------------------------------------------
# The port, asked


def _with_move(reg: Regulation, team: list[TeamSet], move_id: str) -> tuple[Position, SideAction]:
    """Our first active Pokemon's first move replaced by ``move_id``, chosen, beside the
    partner's first choice. Legality is not the question: what the port does with it is."""
    pos = _synthetic_position(reg, team)
    side = pos.sides[0]
    mon = side.pokemon[side.active[0]]
    pp = reg.moves[move_id].pp
    mon.moves[0] = MoveSlot(id=move_id, pp=pp, maxpp=pp)
    target = None if reg.moves[move_id].target == "self" else 1
    partner = side_actions(reg, pos, 0)[0].slots[1]
    ours = SideAction(slots=(MoveAction(slot=0, move_index=1, move_id=move_id, target=target), partner))
    return pos, ours


@pytest.mark.parametrize("move_id", sorted(GATED))
def test_a_gated_move_is_refused_or_noted(reg: Regulation, team_a: list[TeamSet], move_id: str) -> None:
    """Refused for its field, or a note naming the field -- never an answer with it unread.

    Before IKA-246 each of these resolved with a note naming the *move* (`damaging move:
    steelbeam`, `status move: sleeptalk`), which comes from its custom hooks (`onMoveFail`,
    `onTry`), not from the field: a port that modelled Steel Beam's `onMoveFail` would have
    dropped the note and still left the recoil on a hit unread. Whether the dump carries the
    field is the tests above; this one asks the port with whatever dump it is given."""
    field = GATED[move_id]
    pos, ours = _with_move(reg, team_a, move_id)
    theirs = side_actions(reg, pos, 1)[0]
    try:
        result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(0))
    except PortRefused as refused:
        assert f"move field {field}: {move_id}" in str(refused), refused
        return
    notes = [n for n in result.unmodelled if field in n]
    assert notes, f"{move_id} resolved with {field} unread and unnoted: {result.unmodelled}"


def test_struggle_is_not_refused(reg: Regulation, team_a: list[TeamSet]) -> None:
    """The control: the dump now carries `struggleRecoil`, and Struggle still resolves --
    with the field left in the gate, the port refused it ("move field struggleRecoil")."""
    pos = _synthetic_position(reg, team_a)
    side = pos.sides[0]
    user = side.pokemon[side.active[0]]
    for slot in user.moves:
        slot.pp = 0
    choice = next(
        c for c in side_actions(reg, pos, 0)
        if isinstance(c.slots[0], MoveAction) and c.slots[0].move_id == "struggle"
    )
    theirs = side_actions(reg, pos, 1)[0]
    result = resolve_turn(reg, pos, [choice, theirs], budget=Budget.deterministic(0))
    assert result.branches
    assert not [n for n in result.unmodelled if "struggle" in n.lower()], result.unmodelled
