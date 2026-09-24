"""A generated position's move slots start with Showdown's PP, not the dex's (IKA-244).

Showdown's `Pokemon` constructor (sim/pokemon.ts 354-374, vendor a5df827) gives every slot

    const ppUps = move.noPPBoosts || move.id === 'trumpcard' ? 0 : 3;
    const basePP = this.battle.calculatePP(move, ppUps);   // pp = maxpp = basePP

and the champions mod replaces `calculatePP` (data/mods/champions/scripts.ts 41-43) with

    return move.noPPBoosts ? move.pp : (move.pp / 5 + 1) * 4;

so Protect's dex 5 starts a battle at 8 and Fake Out's 10 at 12. Generation built its
turn-1 positions (`selfplay._make_pokemon`, which `position_from_sets` and
`hidden.substitute` call, and `setup._moves` for scenario files) from the dex's `pp`, so
382 of the 515 moves in either regulation started short and Protect ran out after 5 uses.
The dump now carries `startPP`, asked of a Battle for the format.

Every move of both regulations is compared -- which covers every move of every pool team
-- by putting them on Pokemon in oracle battles: the position at team preview (the
constructor's slots) and the turn-1 request of the two leads. The committed M-B rosters go
through `position_from_sets` as whole teams. Moves whose dex PP is 20 or which ignore PP
Ups (`noPPBoosts`) start where the dex says, and are the controls inside the same sweep.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pokeuraou.oracle import Oracle, TeamSet
from pokeuraou.priors import SampledSet
from pokeuraou.regulation import Regulation, load_regulation
from pokeuraou.selfplay import _make_pokemon, position_from_sets
from pokeuraou.setup import _moves
from pokeuraou.teams import load_roster, teams_dir

pytestmark = pytest.mark.oracle

FORMATS = ("gen9championsvgc2026regmc", "gen9championsvgc2026regmb")
#: A carrier for the moves: Showdown does not validate an oracle team, so any species can
#: hold any move, and the slots only read the move.
CARRIER = "kangaskhan"
PER_SIDE = 6
PER_MON = 4


def _chunks(ids: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


def _showdown_slots(oracle: Oracle, fmt: str, move_ids: list[str]) -> dict[str, tuple[int, int]]:
    """Move id -> (pp, maxpp) as Showdown starts a battle, read twice per battle."""
    out: dict[str, tuple[int, int]] = {}
    for chunk in _chunks(move_ids, 2 * PER_SIDE * PER_MON):
        mons = [
            TeamSet(
                species="Kangaskhan",
                ability="Scrappy",
                nature="Hardy",
                moves=chunk[i * PER_MON:(i + 1) * PER_MON] or ["tackle"],
                name=f"k{i}",
            )
            for i in range(2 * PER_SIDE)
        ]
        handle = oracle.create(fmt, mons[:PER_SIDE], mons[PER_SIDE:])
        try:
            for side in handle.position["sides"]:
                for mon in side["pokemon"]:
                    for slot in mon["moves"]:
                        out[slot["id"]] = (slot["pp"], slot["maxpp"])
            handle.step(["team 1234", "team 1234"])
            seen = 0
            for request in handle.requests:
                for active in (request or {}).get("active", []):
                    for slot in active["moves"]:
                        assert (slot["pp"], slot["maxpp"]) == out[slot["id"]], slot
                        seen += 1
            assert seen, "the turn-1 request carried no move slots"
        finally:
            handle.close()
    return out


@pytest.fixture(scope="module", params=FORMATS)
def fmt_reg(request: pytest.FixtureRequest) -> tuple[str, Regulation]:
    return request.param, load_regulation(request.param)


def test_every_move_starts_at_showdowns_pp(
    oracle: Oracle, fmt_reg: tuple[str, Regulation]
) -> None:
    fmt, reg = fmt_reg
    ids = sorted(reg.moves)
    showdown = _showdown_slots(oracle, fmt, ids)
    assert not set(ids) - set(showdown)  # the filler of the last battle may be extra

    wrong = []
    for chunk in _chunks(ids, PER_MON):
        entry = SampledSet(
            species=CARRIER, ability="scrappy", item=None, nature="Hardy", sp={}, moves=chunk
        )
        generated = _make_pokemon(reg, 0, entry, 0).moves
        scenario = _moves(reg, {"moves": chunk})
        for slot, scene in zip(generated, scenario, strict=True):
            want = showdown[slot.id]
            if (slot.pp, slot.maxpp) != want or (scene.pp, scene.maxpp) != want:
                wrong.append((slot.id, (slot.pp, slot.maxpp), (scene.pp, scene.maxpp), want))
    assert not wrong, f"{len(wrong)} of {len(ids)} moves start off Showdown: {wrong[:8]}"

    # The controls inside the sweep: moves the formula leaves alone are there and agree.
    same = [m for m in ids if showdown[m][1] == reg.moves[m].pp]
    moved = [m for m in ids if showdown[m][1] != reg.moves[m].pp]
    assert "struggle" in same and "revivalblessing" in same  # noPPBoosts
    assert "protect" in moved and showdown["protect"] == (8, 8)
    assert len(moved) == 382


def test_rosters_start_at_showdowns_pp(oracle: Oracle) -> None:
    rosters = sorted(teams_dir().glob("*.json"))
    assert rosters
    for path in rosters:
        roster = load_roster(path)
        reg = roster.reg
        handle = oracle.create(reg.meta.format_id, roster.team_sets(), roster.team_sets())
        try:
            handle.step(["team 1234", "team 1234"])
            showdown = [
                [(s["id"], s["pp"], s["maxpp"]) for s in mon["moves"]]
                for mon in handle.position["sides"][0]["pokemon"]
            ]
        finally:
            handle.close()
        position = position_from_sets(reg, roster.sets[:4], roster.sets[:4])
        generated = [
            [(s.id, s.pp, s.maxpp) for s in mon.moves] for mon in position.sides[0].pokemon
        ]
        assert generated == showdown[: len(generated)], path.name
