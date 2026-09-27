"""IKA-349: the port's notes as the analysis page shows them (`notes_ja`).

- every fixed note the port writes (`report("...")` with no argument in rust/src) has its
  Japanese; a new one there fails here until it is said;
- the notes with ids name the ability, item or move by Showdown's Japanese name (the
  localiser, generated from vendor/data/text), and the attacker's and the defender's note on
  one ability become one line;
- an unknown note keeps its words, with any id in it named.
"""

from __future__ import annotations

import re

import pytest

from pokeuraou.names import localiser
from pokeuraou.notes_ja import FIXED, notes_ja
from pokeuraou.regulation import repo_root
from pokeuraou.teams import load_roster


@pytest.fixture(scope="module")
def loc():  # noqa: ANN201
    roster = load_roster("rizabanadohido")
    return localiser(roster.reg, "ja")


def test_every_fixed_port_note_is_said() -> None:
    fixed = set()
    for path in (repo_root() / "rust" / "src").glob("*.rs"):
        fixed.update(re.findall(r'report\("([^"]+)"\)', path.read_text(encoding="utf-8")))
    assert fixed, "no fixed note found; the scan would be vacuous"
    assert fixed <= set(FIXED), sorted(fixed - set(FIXED))


def test_ids_are_named_and_one_ability_is_one_line(loc) -> None:  # noqa: ANN001
    got = notes_ja(loc, [
        "attacker.ability:intimidate", "defender.ability:intimidate", "damaging move: knockoff",
        "depth-2 kept the 3 likeliest branches of a refined cell", "something odd with sitrusberry",
    ])
    assert len(got) == 4
    assert loc.ability("intimidate") in got[0] and "攻撃側・防御側" in got[0]
    assert loc.move("knockoff") in got[1]
    assert got[2].startswith("選択的延長のセル")
    assert got[3].startswith("注記: something odd with") and loc.item("sitrusberry") in got[3]
    assert not any(re.search(r"attacker|defender|damaging move|depth-2", g) for g in got)
