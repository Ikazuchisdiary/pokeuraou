"""A Choice item's lock: kept through Struggle, dropped when it names nothing (IKA-179).

vendor/pokemon-showdown/data/conditions.ts, `choicelock` (no Champions override)::

    onStart(pokemon) {
        if (!this.activeMove) throw new Error("Battle.activeMove is null");
        if (!this.activeMove.id || this.activeMove.hasBounced
            || this.activeMove.sourceEffect === 'snatch') return false;
        this.effectState.move = this.activeMove.id;
    },
    onBeforeMove(pokemon, target, move) {
        if (!pokemon.getItem().isChoice) { pokemon.removeVolatile('choicelock'); return; }
        ... a move other than this.effectState.move and not 'struggle' fails ...
    },
    onDisableMove(pokemon) {
        if (!pokemon.getItem().isChoice || !pokemon.hasMove(this.effectState.move)) {
            pokemon.removeVolatile('choicelock');
            return;
        }
        ... every other move slot disabled ...
    },

and data/items.ts, `choicescarf` (the one Choice item in Champions)::

    onStart(pokemon) { pokemon.removeVolatile('choicelock'); },
    onModifyMove(move, pokemon) { pokemon.addVolatile('choicelock'); },

`addVolatile` on a volatile already there does not run `onStart` again, so the lock names
the move that started it: a Struggle while locked (the locked move out of PP, or Taunted)
leaves it where it was. A Struggle with no lock starts one on `struggle`, which is in no
move slot, so `onDisableMove` at the end of the turn drops it and the next move locks
afresh. The same end of turn drops the lock of a Pokemon whose Scarf was knocked off, and
Trick drops both (the Scarf's `onStart` on the receiver; the giver's lock names a Choice
item it no longer holds).

The port rewrote the lock on every move, so a Struggle while locked pointed it at
`struggle` and the menu handed the whole moveset back. Python kept the first move but never
dropped a lock: after a Struggle with no lock its holder was never locked again.
"""

from __future__ import annotations

import dataclasses

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position

from ._port import Budget, resolve_turn
from .conftest import FORMAT_ID

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str], item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP), item=item
    )


FILL = [
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"]),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
]
GRIMMSNARL = _mon("Grimmsnarl", "Prankster", ["taunt", "spiritbreak", "protect", "reflect"])
GARCHOMP = _mon(
    "Garchomp", "Rough Skin", ["dragonclaw", "earthquake", "rockslide", "protect"], "Choice Scarf"
)


@dataclasses.dataclass(frozen=True)
class Case:
    ours: list[TeamSet]
    theirs: list[TeamSet]
    steps: list[list[str]]
    #: Showdown's lock on p1a after each step.
    locks: list[str | None]
    #: Whether the port plays the case (it refuses Trick).
    ported: bool = True
    #: How many steps our own game follows. Taunt on a Pokemon that has moved lasts a turn
    #: longer (`duration++`), which the resolver does not do: not this issue.
    generated: int | None = None


CASES: dict[str, Case] = {
    # Locked into Tailwind (Prankster, ahead of the Taunt), then Taunted: Struggle, and the
    # lock stays on Tailwind.
    "struggle while locked": Case(
        [_mon("Whimsicott", "Prankster", ["tailwind", "moonblast", "encore", "protect"],
              "Choice Scarf"), *FILL],
        [GRIMMSNARL, *FILL],
        [["move 1, move 2", "move 1 1, move 2"], ["move 1, move 2", "move 3, move 2"],
         ["move 1, move 2", "move 3, move 2"]],
        ["tailwind", "tailwind", "tailwind"],
        generated=2,
    ),
    # Switched in and Taunted before its first move, with nothing but status moves:
    # Struggle with no lock, which locks nothing; free once the Taunt ends, and then Sunny
    # Day locks. (Switched in, because a status move chosen before a Taunt lands that turn
    # fails in Showdown and goes through in the resolver: not this issue.)
    "struggle with no lock": Case(
        [FILL[0], FILL[2],
         _mon("Whimsicott", "Infiltrator", ["tailwind", "encore", "protect", "sunnyday"],
              "Choice Scarf"), FILL[1]],
        [GRIMMSNARL, *FILL],
        [["switch 3, move 2", "move 1 1, move 2"], ["move 1, move 2", "move 3, move 2"],
         ["move 1, move 2", "move 3, move 2"], ["move 4, move 2", "move 3, move 2"],
         ["move 4, move 2", "move 3, move 2"]],
        [None, None, None, "sunnyday", "sunnyday"],
    ),
    "knocked off": Case(
        [GARCHOMP, *FILL],
        [_mon("Kingambit", "Defiant", ["knockoff", "protect", "suckerpunch", "ironhead"]), *FILL],
        [["move 1 2, move 2", "move 1 1, move 2"], ["move 3, move 2", "move 2, move 2"]],
        [None, None],
    ),
    # Gardevoir's Scarf for Garchomp's: both locks go, and Rock Slide locks afresh.
    "trick scarf for scarf": Case(
        [GARCHOMP, *FILL],
        [_mon("Gardevoir", "Trace", ["trick", "moonblast", "psychic", "protect"],
              "Choice Scarf"), *FILL],
        [["move 1 2, move 2", "move 1 1, move 2"], ["move 3, move 2", "move 2 1, move 2"]],
        [None, "rockslide"],
        ported=False,
    ),
    # The control: Encore on the locked move changes nothing.
    "encored while locked": Case(
        [GARCHOMP, *FILL],
        [_mon("Whimsicott", "Prankster", ["encore", "moonblast", "tailwind", "protect"]), *FILL],
        [["move 1 2, move 2", "move 4, move 2"], ["move 1 2, move 2", "move 1 1, move 2"],
         ["move 1 2, move 2", "move 4, move 2"]],
        ["dragonclaw", "dragonclaw", "dragonclaw"],
    ),
}
STEPS = [(name, i) for name, case in sorted(CASES.items()) for i in range(len(case.steps))]


def _unported(name: str, step: int = 0) -> tuple:
    """A case the port refuses (Trick, IKA-208) is held as a strict xfail, so the day the
    port answers it the mark has to come off (IKA-210). Its Trick is the first step; the
    step after it the port answers."""
    if CASES[name].ported or step > 0:
        return ()
    return (pytest.mark.xfail(strict=True, reason="the port refuses Trick (IKA-208)"),)


def _play(oracle: Oracle, case: Case) -> tuple[list[dict], list[set[str]]]:
    """Showdown's positions before the first step and after each, and p1a's menu then."""
    handle = oracle.create(FORMAT_ID, case.ours, case.theirs, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    positions, menus = [handle.position], [_showdown_menu(handle)]
    for step in case.steps:
        handle.step(step)
        assert handle.choice_errors == [], handle.choice_errors
        positions.append(handle.position)
        menus.append(_showdown_menu(handle))
    handle.close()
    return positions, menus


def _showdown_menu(handle) -> set[str]:  # noqa: ANN001
    request = handle.requests[0]
    return {m["id"] for m in request["active"][0]["moves"] if not m.get("disabled")}


def _holder(pos: Position):  # noqa: ANN202
    return pos.sides[0].pokemon[pos.sides[0].active[0]]


def _lock(pos: Position) -> str | None:
    held = _holder(pos).volatile("choicelock")
    return None if held is None else held.move


def _menu(reg, pos: Position) -> set[str]:  # noqa: ANN001
    return {
        s.move_id
        for a in side_actions(reg, pos, 0)
        for s in a.slots
        if isinstance(s, MoveAction) and s.slot == 0
    }


def _chosen(reg, pos: Position, step: list[str]):  # noqa: ANN001, ANN202
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


def _start(positions: list[dict], i: int) -> Position:
    start = Position.from_json(positions[i])
    for side in start.sides:
        for mon in side.pokemon:
            mon.trapped = False
            # `position.ts` folds the Choice lock into `lockedMove`, which our menu offers
            # ahead of a Taunt; a generated game never sets it, and the volatile says it.
            if not mon.has_volatile("twoturnmove"):
                mon.locked_move = None
            # Showdown's move slots carry the last request's `disabled` (a Taunt's, the
            # lock's); ours is only Disable's, so the rest is recomputed from the volatiles.
            if not mon.has_volatile("disable"):
                for move_slot in mon.moves:
                    move_slot.disabled = False
    return start


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown(oracle: Oracle, name: str) -> None:
    """The facts the rest is held to."""
    case = CASES[name]
    positions, menus = _play(oracle, case)
    assert [_lock(Position.from_json(p)) for p in positions[1:]] == case.locks
    for lock, menu in zip(case.locks, menus[1:], strict=True):
        if lock is not None and "struggle" not in menu:
            assert menu == {lock}


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("name", "step"), [pytest.param(n, i, marks=_unported(n, i)) for n, i in STEPS]
)
def test_the_port_agrees(
    reg,  # noqa: ANN001
    oracle: Oracle,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    step: int,
) -> None:
    """The port resolves each turn from Showdown's position to Showdown's lock, and every
    branch has Showdown's lock and the menu Showdown offers after it.

    The binary built from master before IKA-179 locks `struggle` in the first two cases and
    keeps a knocked-off lock.
    """
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    case = CASES[name]
    positions, menus = _play(oracle, case)
    start = _start(positions, step)
    # Showdown's stats ride along as an override, which the port refuses as a transformed
    # Pokemon; the spreads are known, so the stats are too.
    for side in start.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    chosen = _chosen(reg, start, case.steps[step])
    try:
        node = rustnode.node_for(reg)
        assert node is not None
        ported = node.resolve(start, chosen, Budget.matrix(), select=0)
    finally:
        rustnode.reset()
    assert ported is not None and ported.position is not None, "the port refused the turn"
    assert _lock(ported.position) == case.locks[step]
    assert _menu(reg, ported.position) == menus[step + 1]
    result = resolve_turn(reg, start, chosen, budget=Budget.matrix())
    assert not result.suspended
    assert {_lock(b.position) for b in result.branches} == {case.locks[step]}
    for branch in result.branches:
        assert _menu(reg, branch.position) == menus[step + 1]


# ---------------------------------------------------------------------------
# No oracle.


def test_a_recorded_lock_on_struggle_does_not_hold(reg) -> None:  # noqa: ANN001
    """Records made before IKA-179 can carry a lock on `struggle` (the port wrote one on
    every Struggle). It locks nothing, and the next move locks afresh."""
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, [GARCHOMP, *FILL])
    other = _synthetic_position(reg, [GRIMMSNARL, *FILL])
    pos.sides[1] = other.sides[0]
    pos.sides[1].id = pos.sides[1].name = "p2"
    holder = _holder(pos)
    holder.item = "choicescarf"
    holder.volatiles.append(Effect(id="choicelock", move="struggle"))
    assert len(_menu(reg, pos)) == 4
    result = resolve_turn(
        reg, pos, _chosen(reg, pos, ["move 1 2, move 2", "move 3, move 2"]), budget=Budget.matrix()
    )
    assert {_lock(b.position) for b in result.branches} == {"dragonclaw"}


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207).


@pytest.mark.oracle
@pytest.mark.parametrize("name", [pytest.param(n, marks=_unported(n)) for n in sorted(CASES)])
def test_the_games_the_port_generates(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    """`test_the_games_we_generate` with the port playing every turn, branch 0 followed."""
    from ._port_showdown import port_branch

    case = CASES[name]
    positions, menus = _play(oracle, case)
    pos = _start(positions, 0)
    for step, want in enumerate(case.locks[: case.generated]):
        pos = port_branch(port, pos, _chosen(reg, pos, case.steps[step]), Budget.matrix(), 0)
        assert (_lock(pos), _menu(reg, pos)) == (want, menus[step + 1]), step
