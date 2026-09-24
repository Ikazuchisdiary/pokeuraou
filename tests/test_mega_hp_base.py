"""A mega keeps its maximum HP, and every mega on the books keeps its HP base (IKA-60).

Python's `_do_mega` rebuilds the battler after the forme change and writes its `maxhp`
back, but `view.battler` hands a known spread's maximum straight back from the position:
the maximum never moves. The port's `do_mega` used to write the HP stat recomputed from
the mega's base stats instead. The two agree only while every mega shares its base
form's HP base -- which is true of all 76 M-B and 82 M-C pairings, and was true by
accident. `change_forme` had the same shape and did diverge (IKA-53: a Kingambit given
Stance Change was a 167-HP Aegislash in the port and a 207-HP one in Python).

So two things are held here. The data: a mega that changes its HP base fails the first
test rather than going on quietly. And the port: on a regulation where Charizard-Mega-Y
is given another HP base, the port's mega must still come out as Python's does, branch by
branch -- which the build before IKA-60 does not.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from pokeuraou import rustnode, view
from pokeuraou.actions import MoveAction, side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.regulation import Regulation, regulation_dir
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster
from pokeuraou.view import battler

from ._port import Budget

FORMATS = ("gen9championsvgc2026regmb", "gen9championsvgc2026regmc")


def _dump(format_id: str) -> dict:
    return json.loads((regulation_dir() / f"{format_id}.json").read_text(encoding="utf-8"))


def hp_base_mismatches(reg: Regulation) -> list[tuple[str, str, str, int, int]]:
    """Every (base, stone, mega, base HP, mega HP) whose two HP bases differ."""
    wrong = []
    for (base, stone), mega in sorted(reg.mega_by_species.items()):
        before = reg.species[base].base_stats[0]
        after = reg.species[mega].base_stats[0]
        if before != after:
            wrong.append((base, stone, mega, before, after))
    return wrong


@pytest.mark.parametrize("format_id", FORMATS)
def test_every_mega_keeps_its_base_forms_hp_base(format_id: str) -> None:
    """If this fails, `_do_mega` and `do_mega` still agree -- on a maximum that stays.

    Whether it should stay is then a question for Showdown, not for these two.
    """
    reg = Regulation(_dump(format_id))
    assert len(reg.mega_by_species) > 50, "the premise: the regulation has its megas"
    for (base, _stone), mega in reg.mega_by_species.items():
        assert base in reg.species and mega in reg.species, (base, mega)
    assert hp_base_mismatches(reg) == []


def _with_mega_hp(format_id: str, mega: str, hp: int) -> dict:
    data = copy.deepcopy(_dump(format_id))
    entry = next(s for s in data["species"] if s["id"] == mega)
    entry["baseStats"]["hp"] = hp
    return data


def test_the_check_sees_a_mega_that_changes_its_hp_base() -> None:
    """The positive control for the test above: one changed number and it fails."""
    reg = Regulation(_with_mega_hp("gen9championsvgc2026regmb", "charizardmegay", 108))
    assert hp_base_mismatches(reg) == [
        ("charizard", "charizarditey", "charizardmegay", 78, 108)
    ]


@pytest.fixture()
def synthetic_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A regulation where Charizard-Mega-Y has an HP base of 108, on both sides of the bridge.

    The port reads its regulation from `configs/regulations` under the repository root, so
    the root is pointed at a directory holding the changed file -- after the binary's own
    path has been pinned, since that is found from the root too.
    """
    binary = rustnode.binary_path()
    if not binary.exists():
        pytest.fail(f"no Rust binary at {binary}; `cargo build --release`")
    format_id = "gen9championsvgc2026regmb"
    data = _with_mega_hp(format_id, "charizardmegay", 108)
    target = tmp_path / "configs" / "regulations" / f"{format_id}.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(json.dumps(data).encode("utf-8"))
    monkeypatch.setenv(rustnode.ENV_BINARY, str(binary))
    monkeypatch.setattr(rustnode, "repo_root", lambda: tmp_path)
    # `view._cached_stats` keys on the format id, which this regulation shares with the
    # real one: a test that ran earlier in the same process leaves the real 169 there, and
    # the control below would read it. A private memo for this regulation, and none of its
    # stats leak out to the tests after.
    monkeypatch.setattr(view, "_STATS_MEMO", {})
    reg = Regulation(data, source=target)
    register_mega_stones(reg)
    node = rustnode.RustNode(reg)
    yield reg, node
    node.close()
    os.environ.pop(rustnode.ENV_ENABLE, None)


def test_a_mega_that_changes_its_hp_base_keeps_its_maximum_over_there_too(
    synthetic_node, monkeypatch: pytest.MonkeyPatch  # noqa: ANN001
) -> None:
    from pokeuraou import selfplay

    reg, node = synthetic_node
    # The leads' switch-ins by the same port (IKA-210), not Python's.
    monkeypatch.setattr(
        selfplay, "apply_lead_abilities", lambda _reg, p, rng=None: node.apply_lead_abilities(p, rng=rng)
    )
    sheet = {entry.species: entry for entry in load_roster("rizabanadohido").sets}
    own = [sheet[n] for n in ("charizard", "sylveon", "venusaur", "garchomp")]
    foe = [sheet[n] for n in ("incineroar", "toxapex", "garchomp", "venusaur")]
    pos = position_from_sets(reg, own, foe)
    charizard = pos.sides[0].active_pokemon()[0]
    assert charizard is not None and charizard.species == "charizard"
    maxhp = charizard.maxhp

    megas = [
        action
        for action in side_actions(reg, pos, 0)
        if isinstance(action.slots[0], MoveAction) and action.slots[0].mega
    ]
    theirs = next(
        action
        for action in side_actions(reg, pos, 1)
        if all(
            isinstance(one, MoveAction) and one.move_id in ("protect", "banefulbunker")
            for one in action.slots
        )
    )
    assert megas, "the premise: Charizard can mega"
    budget = Budget.matrix()
    evolved = 0
    for ours in megas[:6]:
        there = node.resolve(pos, [ours, theirs], budget)
        assert there is not None, "the port refused the turn"
        for index in range(len(there.branches)):
            chosen = node.resolve(pos, [ours, theirs], budget, select=index)
            assert chosen is not None and chosen.position is not None
            after = chosen.position.sides[0].pokemon[0]
            if after.species == "charizardmegay":
                evolved += 1
                assert after.maxhp == maxhp, "the position's maximum stays"
                # Not vacuous: the HP stat this regulation recomputes is another number,
                # and that is what the pre-IKA-60 port wrote.
                assert int(battler(reg, after).stats[0][0]) != maxhp
    assert evolved, "no branch had Charizard mega evolve, so nothing was tested"
