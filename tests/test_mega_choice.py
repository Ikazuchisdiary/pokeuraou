"""IKA-345: the person's input lists each move once and Mega Evolution is a toggle on the slot
(`web/choice.js`); the move and the toggle are turned back into one of the prompt's legal
actions before they are sent.

- every legal action is reached: its moves without " mega", with the toggle on the slot that
  Mega Evolves (or off), resolve to that very action -- on a real position from the port's
  legal actions, where one lead can Mega Evolve and the other cannot;
- the toggles: only a slot that can Mega Evolve has one; a pair that is not legal (the toggle
  on a switch) resolves to nothing, so it cannot be sent (and the page lets one slot at a time
  hold the toggle);
- two slots that can both Mega Evolve (a hand-made prompt) resolve either way, never both.

Runs `choice.js` under node, as the page runs it (the suite's CI job has node).
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.regulation import repo_root
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

CHOICE = repo_root() / "src" / "pokeuraou" / "web" / "choice.js"

HARNESS = r"""
const C = require(process.argv[1]);
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = cases.map((slots) => {
  const megas = C.megaSlots(slots);
  const reached = slots.map((a) => {
    const chosen = a.map((x) => C.base(x[0]));
    const mega = a.findIndex((x) => C.isMega(x[0]));
    return C.resolve(slots, chosen, mega);
  });
  const lines = reached.map((i) => C.line(slots, i));
  const first = slots.length ? slots[0].map((x) => C.base(x[0])) : [];
  const opts = first.map((_, k) => C.options(slots, k, first).map((o) => o[0]));
  // Illegal pairs: the toggle on a slot whose choice is not a move.
  const bad = [];
  for (const a of slots) {
    const chosen = a.map((x) => C.base(x[0]));
    a.forEach((x, k) => { if (!x[0].startsWith("move")) bad.push(C.resolve(slots, chosen, k)); });
  }
  return { megas, reached, lines, opts, bad };
});
process.stdout.write(JSON.stringify(out));
"""


def _run(cases: list) -> list:  # noqa: ANN401
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on the path")
    got = subprocess.run(
        [node, "-e", HARNESS, str(CHOICE)], input=json.dumps(cases), capture_output=True,
        text=True, encoding="utf-8", check=True,
    )
    return json.loads(got.stdout)


def _prompt(reg, pos, side: int) -> list:  # noqa: ANN001
    legal = side_actions(reg, pos, side)
    return [[[s.to_choice(), s.describe(reg), None] for s in a.slots] for a in legal]


def test_moves_and_the_toggle_reach_every_legal_action() -> None:
    roster = load_roster("rizabanadohido")
    register_mega_stones(roster.reg)
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    slots = _prompt(reg, pos, 0)
    megas_expected = sorted({k for a in slots for k, x in enumerate(a) if x[0].endswith(" mega")})
    # Positive control: one lead can Mega Evolve, the other cannot.
    assert megas_expected == [0], megas_expected
    (got,) = _run([slots])
    assert got["megas"] == megas_expected
    assert got["reached"] == list(range(len(slots))), "a legal action the input cannot reach"
    assert got["lines"] == [", ".join(x[0] for x in a) for a in slots]
    # Each move once: no option carries " mega", and there are fewer than the choices.
    assert all(not c.endswith(" mega") for k in got["opts"] for c in k)
    assert len(got["opts"][0]) < len({a[0][0] for a in slots})
    assert all(i == -1 for i in got["bad"]), "an illegal pair resolved to an action"


def test_two_slots_that_can_both_mega_evolve_take_one_at_a_time() -> None:
    moves = ["move 1 1", "move 2 2"]
    slots = []
    for m0 in [*moves, "switch 3"]:
        for m1 in [*moves, "switch 3"]:
            if m0 == m1 == "switch 3":
                continue
            slots.append([[m0, m0, None], [m1, m1, None]])
            if m0.startswith("move"):
                slots.append([[m0 + " mega", m0, None], [m1, m1, None]])
            if m1.startswith("move"):
                slots.append([[m0, m0, None], [m1 + " mega", m1, None]])
    (got,) = _run([slots])
    assert got["megas"] == [0, 1]
    assert got["reached"] == list(range(len(slots)))
    assert all(i == -1 for i in got["bad"])
    assert got["opts"][0] == [*moves, "switch 3"]
    assert all(not c.endswith(" mega") for k in got["opts"] for c in k)


def test_no_mega_no_toggle() -> None:
    slots = [[["move 1 1", "a", None], ["move 2", "b", None]],
             [["move 2 2", "c", None], ["switch 3", "d", None]]]
    (got,) = _run([slots])
    assert got["megas"] == []
    assert got["reached"] == [0, 1]
