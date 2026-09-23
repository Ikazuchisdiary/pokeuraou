"""The audit of what the port's gates pass, and proof that it can fail.

`tools/port_gate_audit.py` exists because one missing line in `item_handled` cost eight
days of refused positions for an effect the port had implemented all along. A check
written after that fact is worth nothing until it has been shown catching it, so the first
test here puts the port back in the state it was in -- `check_white_herb` present,
`"whiteherb"` struck from the gate -- and requires the audit to name it.

The rest guard the two ways such a check goes quiet: reporting nothing because it read the
gate wrong, and reporting nothing because the gate grew a clause it does not understand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ._harness import load_tool

ROOT = Path(__file__).resolve().parents[1]
REGULATION = "gen9championsvgc2026regmc"

audit_tool = load_tool("port_gate_audit")


@pytest.fixture
def sources() -> dict[str, str]:
    return audit_tool.port_sources()


def ungated(sources: dict[str, str]) -> set[str]:
    found, _, _ = audit_tool.audit(REGULATION, sources=sources)
    return {finding.identifier for finding in found}


def test_the_gate_as_it_stands_does_not_accuse_white_herb(sources):
    """The null control. Without it, the next test passes for any check that shouts."""
    assert '| "whiteherb"' in sources["resolve.rs"], "the gate no longer lists it"
    assert "whiteherb" not in ungated(sources)


def test_striking_white_herb_from_the_gate_is_caught(sources):
    """IKA-57's acceptance condition: delete the line, and this points at it."""
    sources["resolve.rs"] = sources["resolve.rs"].replace('\n            | "whiteherb"', "", 1)
    assert '| "whiteherb"' not in sources["resolve.rs"]
    assert "whiteherb" in ungated(sources)


def test_an_effect_the_gate_never_knew_is_caught_the_same_way(sources):
    """Not only the id that was once missing.

    Striking a line that exists tests the diff; an effect the gate has never listed is
    the shape the next one will have. `focussash` is implemented in `effects.rs` beside
    `focusband`, which was a finding until 2026-09-23 for exactly that reason -- absent
    from the gate while its neighbour was listed (IKA-70).
    """
    sources["resolve.rs"] = sources["resolve.rs"].replace('"focussash" | ', "", 1)
    assert "focussash" in ungated(sources)


def test_the_gate_reaching_an_effect_by_a_helper_is_not_a_finding(sources):
    """`item_handled` passes anything `type_boost_item` knows without naming it.

    Reading the gate as its `matches!` arms alone would report eighteen type-boosting
    items and eighteen resist berries as refused, and a report with thirty-six false
    entries is one that gets switched off.
    """
    found = ungated(sources)
    assert "charcoal" not in found
    assert "babiriberry" not in found
    # And the suffix clause, which is how every mega stone passes.
    assert "charizardite" not in found


def test_a_gate_clause_this_audit_cannot_read_stops_it(sources):
    """The failure that would otherwise be silent.

    If the gate grows a third escape hatch, every id it passes through that hatch starts
    looking refused. The audit has no way to be right about such a clause, so it stops
    rather than reporting findings that are already handled.
    """
    sources["resolve.rs"] = sources["resolve.rs"].replace(
        "    ) || crate::inert::item_is_inert(item)",
        "    ) || crate::inert::item_is_inert(item)\n        || reg.extra_items.contains(item)",
        1,
    )
    with pytest.raises(SystemExit, match="clause this check cannot read"):
        audit_tool.audit(REGULATION, sources=sources)


def test_a_helper_whose_ids_this_cannot_read_stops_it_too(sources):
    """The same silence by the other door.

    A gate clause can be recognised and still be misread, if the helper it delegates to
    stops being a `match id` table. Answering "it passes nothing" would make every id
    behind it look refused, so this stops as well.
    """
    sources["effects.rs"] = sources["effects.rs"].replace(
        '        "blackbelt" => Some("Fighting"),', "", 1
    )
    sources["effects.rs"] = re.sub(
        r'"[a-z]+" => Some\("[A-Za-z]+"\),', "", sources["effects.rs"]
    )
    with pytest.raises(SystemExit, match="read no ids out"):
        audit_tool.audit(REGULATION, sources=sources)


def test_the_recorded_sets_are_the_ones_the_port_produces():
    """What CI runs, as a test too.

    Both directions: a new finding fails, and a recorded one that has been fixed fails
    until it is struck off, or the lists stop describing anything.
    """
    assert audit_tool.main(["--check", "--regulation", REGULATION]) == 0


def unreferenced(sources: dict[str, str], regulation: str = REGULATION) -> set[str]:
    _, found, _ = audit_tool.audit(regulation, sources=sources)
    return {finding.identifier for finding in found}


WEATHER_RECOVERY = {"moonlight", "synthesis", "morningsun"}


@pytest.mark.parametrize("regulation", audit_tool.REGULATIONS)
def test_a_custom_code_move_the_port_never_names_is_caught(sources, regulation):
    """IKA-187. Moonlight, Synthesis and Morning Sun are fully modelled in Python, which
    heals them by name: the dump has no `heal` field, the amount is in `onHit`. The port
    was the state this puts back -- the gate passes them and nothing else names them --
    and the audit set every fully-modelled arm aside, so it healed nothing and nothing
    said so. `hasCustomCode` is what tells the two kinds of arm apart."""
    line = 'const WEATHER_RECOVERY_MOVES: [&str; 3] = ["moonlight", "synthesis", "morningsun"];'
    assert line in sources["moves.rs"]
    assert not WEATHER_RECOVERY & unreferenced(sources, regulation)
    sources["moves.rs"] = sources["moves.rs"].replace(line, "const WEATHER_RECOVERY_MOVES: [&str; 0] = [];")
    assert WEATHER_RECOVERY.issubset(unreferenced(sources, regulation))


def test_a_move_that_is_its_fields_is_still_set_aside(sources):
    """The null control for the test above: Recover is fully modelled, has no custom code
    (`heal: [1, 2]` is the whole move) and the port never names it outside the gate. It
    has to stay set aside, or the audit would demand a name for every field-only move."""
    gates = [audit_tool.parse_gate(g, sources["resolve.rs"], sources) for g in audit_tool.GATES]
    assert '"recover"' not in audit_tool.behaviour_text(sources, [g.inventory for g in gates])
    assert "recover" not in unreferenced(sources)


def test_check_runs_both_regulations(capsys):
    """IKA-187: the recorded games are M-B and `--check` asked only M-C."""
    assert audit_tool.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "== gen9championsvgc2026regmb" in out and "== gen9championsvgc2026regmc" in out
