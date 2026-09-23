"""Outrage's rampage: two or three turns of the one move, then confusion (IKA-174).

vendor/pokemon-showdown/data/conditions.ts, `lockedmove` (no Champions override)::

    duration: 2,
    onResidual(target) {
        if (target.status === 'slp') {
            // don't lock, and bypass confusion for calming
            delete target.volatiles['lockedmove'];
        }
        this.effectState.trueDuration--;
    },
    onStart(target, source, effect) {
        this.effectState.trueDuration = this.random(2, 4);
        this.effectState.move = effect.id;
    },
    onRestart() {
        if (this.effectState.trueDuration >= 2) {
            this.effectState.duration = 2;
        }
    },
    onAfterMove(pokemon) {
        if (this.effectState.duration === 1) {
            pokemon.removeVolatile('lockedmove');
        }
    },
    onEnd(target) {
        if (this.effectState.trueDuration > 1) return;
        target.addVolatile('confusion');
    },
    onLockMove(pokemon) { ... return this.effectState.move; },

Outrage, Petal Dance, Raging Fury and Thrash put it on their user with `self:
{volatileStatus: 'lockedmove'}`, which `selfDrops` applies only when a target was hit: a
Protect on the first turn starts nothing, and on a later turn it skips the restart, so the
rampage ends then -- confused only if that was its last turn anyway. The locked turns spend
no PP (`if (!lockedMove) deductPP`). Own Tempo, Misty Terrain and the Persim and Lum
Berries keep the confusion off.

Our resolver added a bare `lockedmove` -- no move, no duration -- and never took it off, so
in a generated game the menu never locked, the rampage never ended and nobody was ever
confused. Each case below is played by Showdown under `multihit='min'` (the roll is 2) and
`'max'` (3), and our resolver and the port are held to it.
"""

from __future__ import annotations

import dataclasses

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, SwitchAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position
from pokeuraou.resolve import Budget, resolve_turn

from .conftest import FORMAT_ID

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str], item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP), item=item
    )


RAMPAGE = ["outrage", "protect", "earthquake", "swordsdance"]
GARCHOMP = _mon("Garchomp", "Rough Skin", RAMPAGE)
PARTNERS = [
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"]),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
]
#: Outrage's `randomNormal` target is `sample(foes)`, the pinned policy's first: Hippowdon.
#: Our resolver draws either foe, a half each (IKA-178).
FOES = [
    _mon("Hippowdon", "Sand Stream", ["slackoff", "protect", "earthquake", "yawn"]),
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"]),
    _mon("Kingambit", "Defiant", ["kowtowcleave", "protect", "suckerpunch", "ironhead"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]
RAMP = ["move 1, move 1", "move 1, move 1"]
#: Both foes Protect, so the Outrage is stopped whichever foe it draws (IKA-178).
GUARD = ["move 1, move 1", "move 2, move 2"]
#: Hippowdon's Yawn at Garchomp: asleep at the end of the next turn.
YAWN = ["move 1, move 1", "move 4 1, move 1"]


@dataclasses.dataclass(frozen=True)
class Case:
    steps: list[list[str]]
    #: Showdown's lock after the last step, as (move, duration, trueDuration) or None, and
    #: whether it is confused -- for the roll of 2 and the roll of 3.
    #: None: not played for that roll (a confused turn's own randomness would be in it).
    short: tuple[tuple[str, int, int] | None, bool] | None
    long: tuple[tuple[str, int, int] | None, bool] | None
    first: TeamSet | None = None  # Garchomp


#: A new rampage starts on turn 3 after a short one that left no confusion behind.
AGAIN = (("outrage", 1, 1), False)
CASES: dict[str, Case] = {
    "turn 1": Case([RAMP], (("outrage", 1, 1), False), (("outrage", 1, 2), False)),
    "turn 2": Case([RAMP, RAMP], (None, True), (("outrage", 1, 1), False)),
    "turn 3": Case([RAMP, RAMP, RAMP], None, (None, True)),
    # The positive control of the lock: a Protect on the first turn starts nothing.
    "protected on turn 1": Case([GUARD], (None, False), (None, False)),
    # No restart, so the rampage ends; only the short one was on its last turn.
    "protected on turn 2": Case([RAMP, GUARD], (None, True), (None, False)),
    "protected on turn 3": Case([RAMP, RAMP, GUARD], None, (None, True)),
    # Yawn puts it to sleep at the end of turn 2 (order 23, ahead of the lock's residual):
    # a lock still running is dropped without confusion.
    "asleep on turn 2": Case([YAWN, RAMP], (None, True), (None, False)),
    "own tempo": Case([RAMP, RAMP, RAMP], AGAIN, (None, False),
                      _mon("Garchomp", "Own Tempo", RAMPAGE)),
    "persim berry": Case([RAMP, RAMP, RAMP], AGAIN, (None, False),
                         _mon("Garchomp", "Rough Skin", RAMPAGE, "Persim Berry")),
    "lum berry": Case([RAMP, RAMP, RAMP], AGAIN, (None, False),
                      _mon("Garchomp", "Rough Skin", RAMPAGE, "Lum Berry")),
}
ROLLS = {"short": "min", "long": "max"}
PLAYED = [
    (name, roll)
    for name, case in sorted(CASES.items())
    for roll in sorted(ROLLS)
    if getattr(case, roll) is not None
]


def _play(oracle: Oracle, case: Case, roll: str) -> list[dict]:
    """Showdown's positions: before the first step and after each."""
    handle = oracle.create(
        FORMAT_ID, [case.first or GARCHOMP, *PARTNERS], FOES, policy=RandomnessPolicy(multihit=ROLLS[roll])
    )
    handle.step(["team 1234", "team 1234"])
    positions = [handle.position]
    for step in case.steps:
        handle.step(step)
        assert handle.choice_errors == [], handle.choice_errors
        positions.append(handle.position)
    handle.close()
    return positions


def _rampager(pos: Position):  # noqa: ANN202
    return pos.sides[0].pokemon[pos.sides[0].active[0]]


def _lock(pos: Position, *, with_roll: bool = True):  # noqa: ANN202
    mon = _rampager(pos)
    held = mon.volatile("lockedmove")
    lock = None
    if held is not None:
        lock = (held.move, held.duration, held.extra.get("trueDuration")) if with_roll else (
            held.move, held.duration
        )
    return lock, mon.has_volatile("confusion")


def _pp(pos: Position) -> int:
    return _rampager(pos).moves[0].pp


def _menu(reg, pos: Position):  # noqa: ANN001, ANN202
    pieces = [a.slots[0] for a in side_actions(reg, pos, 0)]
    return (
        {p.move_id for p in pieces if isinstance(p, MoveAction)},
        any(isinstance(p, SwitchAction) for p in pieces),
    )


def _chosen(reg, pos: Position, step: list[str]):  # noqa: ANN001, ANN202
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


def _expected(case: Case, roll: str):  # noqa: ANN202
    return case.short if roll == "short" else case.long


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "roll"), PLAYED)
def test_showdown(oracle: Oracle, name: str, roll: str) -> None:
    """The facts the rest is held to, and the PP: only the first turn spends one."""
    case = CASES[name]
    positions = _play(oracle, case, roll)
    after = Position.from_json(positions[-1])
    assert _lock(after) == _expected(case, roll)
    if name.startswith("turn"):
        assert _pp(after) == _rampager(Position.from_json(positions[0])).moves[0].pp - 1
    if name.endswith("berry"):
        assert _rampager(after).item is None, "the berry is eaten"


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "roll"), PLAYED)
def test_our_turn_from_showdowns_position(reg, oracle: Oracle, name: str, roll: str) -> None:  # noqa: ANN001
    """The last turn resolved by us from Showdown's position, which carries the roll."""
    case = CASES[name]
    positions = _play(oracle, case, roll)
    start = Position.from_json(positions[-2])
    after = Position.from_json(positions[-1])
    # Showdown's verdict rides into our children (`position.py` copies it), so it is
    # cleared as a searched child carries it (IKA-169); the menu reads the lock itself.
    for side in start.sides:
        for mon in side.pokemon:
            mon.trapped = False
    result = resolve_turn(reg, start, _chosen(reg, start, case.steps[-1]), budget=Budget.matrix())
    assert not result.suspended
    # The roll is on the position, so nothing branches; a rampage that starts this turn has
    # no roll yet (`_roll_rampage` makes it on the second turn).
    want = _expected(case, roll)
    started = _rampager(start).volatile("lockedmove") is None
    if started and want[0] is not None:
        want = ((*want[0][:2], None), want[1])
    locks = {_lock(b.position) for b in result.branches}
    assert locks == {want}, (locks, want)
    for branch in result.branches:
        assert _pp(branch.position) == _pp(after)
        assert _rampager(branch.position).item == _rampager(after).item
        moves, switches = _menu(reg, branch.position)
        if want[0] is not None:
            assert (moves, switches) == ({"outrage"}, False)
        else:
            assert len(moves) == 4 and switches


def _generated(reg, positions: list[dict], case: Case, budget: Budget):  # noqa: ANN001, ANN202
    """Every turn resolved by us from Showdown's first position: the generation form.

    Returns the leaves as {(lock, confused): probability}, following every branch.
    """
    frontier = [(1.0, Position.from_json(positions[0]))]
    for step in case.steps:
        nxt = []
        for weight, pos in frontier:
            result = resolve_turn(reg, pos, _chosen(reg, pos, step), budget=budget)
            assert not result.suspended
            nxt.extend((weight * b.probability, b.position) for b in result.branches)
        frontier = nxt
    out: dict = {}
    for weight, pos in frontier:
        key = _lock(pos, with_roll=False)
        out[key] = out.get(key, 0.0) + weight
    return out


MULTI_TURN = sorted(
    n for n, c in CASES.items() if len(c.steps) > 1 and c.short is not None and c.long is not None
)


@pytest.mark.oracle
@pytest.mark.parametrize("name", MULTI_TURN)
def test_the_games_we_generate(reg, oracle: Oracle, name: str) -> None:
    """Showdown rolls 2 or 3 at the start; we branch when it first matters, one half each.

    So the leaves of our tree are Showdown's two games, weighted a half each (or one game
    at weight 1 when both rolls end alike).
    """
    case = CASES[name]
    want: dict = {}
    for roll in ROLLS:
        positions = _play(oracle, case, roll)
        lock, confused = _expected(case, roll)
        key = (None if lock is None else lock[:2], confused)
        want[key] = want.get(key, 0.0) + 0.5
    ours = _generated(reg, positions, case, Budget.matrix())
    assert ours.keys() == want.keys(), (ours, want)
    for key, weight in want.items():
        assert ours[key] == pytest.approx(weight), (key, ours, want)


@pytest.mark.oracle
@pytest.mark.parametrize("name", MULTI_TURN)
def test_the_pinned_budget_rolls_two(reg, oracle: Oracle, name: str) -> None:
    """`random(2, 4)` is 2 under the oracle's pinned policy, and the pinned budget agrees."""
    case = CASES[name]
    positions = _play(oracle, case, "short")
    lock, confused = case.short
    want = {(None if lock is None else lock[:2], confused): 1.0}
    pinned = dataclasses.replace(Budget.deterministic(), pinned_policy=True)
    assert _generated(reg, positions, case, pinned) == pytest.approx(want)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("name", "roll"),
    [p for p in PLAYED if p[0] in ("turn 2", "turn 3", "protected on turn 2", "asleep on turn 2",
                                   "own tempo", "lum berry")],
)
def test_the_port_agrees(
    reg,  # noqa: ANN001
    oracle: Oracle,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    roll: str,
) -> None:
    """The port resolves the last turn from Showdown's position as Python does.

    The binary built from master before IKA-174 keeps a bare `lockedmove` for good.
    """
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    case = CASES[name]
    positions = _play(oracle, case, roll)
    start = Position.from_json(positions[-2])
    # Showdown's stats ride along as an override, which the port refuses as a
    # transformed Pokemon; the spreads are known, so the stats are too.
    for side in start.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    chosen = _chosen(reg, start, case.steps[-1])
    try:
        node = rustnode.node_for(reg)
        assert node is not None
        ported = node.resolve(start, chosen, Budget.matrix(), select=0)
    finally:
        rustnode.reset()
    assert ported is not None and ported.position is not None, "the port refused the turn"
    ours = resolve_turn(reg, start, chosen, budget=Budget.matrix())
    # One branch per foe the Outrage draws (IKA-178), each held to Showdown's lock.
    assert len(ours.branches) == len(ported.branches) >= 1
    want = _expected(case, roll)
    if _rampager(start).volatile("lockedmove") is None and want[0] is not None:
        want = ((*want[0][:2], None), want[1])
    try:
        node = rustnode.node_for(reg)
        assert node is not None
        for index, branch in enumerate(ours.branches):
            picked = node.resolve(start, chosen, Budget.matrix(), select=index)
            assert picked is not None and picked.position is not None
            assert _lock(picked.position) == _lock(branch.position) == want
            assert _pp(picked.position) == _pp(branch.position)
    finally:
        rustnode.reset()


@pytest.mark.oracle
def test_the_port_branches_the_roll(reg, oracle: Oracle, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """Turn 2 of a rampage our resolver started: both engines split it a half each."""
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    positions = _play(oracle, CASES["turn 2"], "short")
    first = Position.from_json(positions[0])
    for side in first.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    ours = resolve_turn(reg, first, _chosen(reg, first, RAMP), budget=Budget.matrix())
    # Hippowdon or Milotic took the first Outrage (IKA-178); either starts the rampage.
    assert len(ours.branches) == 2
    start = ours.branches[0].position
    chosen = _chosen(reg, start, RAMP)
    try:
        node = rustnode.node_for(reg)
        assert node is not None
        weights = node.resolve(start, chosen, Budget.matrix(), select=None).branches
        picked = [
            node.resolve(start, chosen, Budget.matrix(), select=i).position
            for i in range(len(weights))
        ]
    finally:
        rustnode.reset()
    python = resolve_turn(reg, start, chosen, budget=Budget.matrix())
    # The length's half and half, each split again by the foe drawn (IKA-178).
    want = [(b.probability, _lock(b.position)) for b in python.branches]
    got = [(w, _lock(p)) for w, p in zip(weights, picked, strict=True)]
    assert [lock for _, lock in got] == [lock for _, lock in want]
    assert [w for w, _ in got] == pytest.approx([w for w, _ in want])
    by_lock: dict = {}
    for weight, lock in want:
        by_lock[lock] = by_lock.get(lock, 0.0) + weight
    assert by_lock.keys() == {(None, True), (("outrage", 1, 1), False)}
    assert list(by_lock.values()) == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# No oracle: what the oracle cases cannot reach.


def _hand_built(reg):  # noqa: ANN001, ANN202
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, [GARCHOMP, *PARTNERS])
    other = _synthetic_position(reg, FOES)
    pos.sides[1] = other.sides[0]
    pos.sides[1].id = pos.sides[1].name = "p2"
    return pos


def _flinched_on_turn_two(reg, true_duration: int):  # noqa: ANN001, ANN202
    pos = _hand_built(reg)
    mon = _rampager(pos)
    mon.volatiles.append(
        Effect(id="lockedmove", duration=1, move="outrage", extra={"trueDuration": true_duration})
    )
    mon.volatiles.append(Effect(id="flinch", duration=1))
    result = resolve_turn(reg, pos, _chosen(reg, pos, RAMP), budget=Budget.matrix())
    return {_lock(b.position) for b in result.branches}


@pytest.mark.parametrize(("true_duration", "confused"), [(1, True), (2, False)])
def test_a_rampage_that_could_not_move_ends_at_the_residual(
    reg, true_duration: int, confused: bool  # noqa: ANN001
) -> None:
    """No `AfterMove` for a flinched Pokemon: the lock runs out at the end of the turn, and
    `onEnd` confuses it only if the rampage was on its last turn."""
    assert _flinched_on_turn_two(reg, true_duration) == {(None, confused)}


def test_a_recorded_bare_lock_is_replaced(reg) -> None:  # noqa: ANN001
    """Records made before IKA-174 carry a bare `lockedmove`: it locks nothing, and the next
    Outrage starts a real rampage rather than restarting the bare one."""
    pos = _hand_built(reg)
    _rampager(pos).volatiles.append(Effect(id="lockedmove"))
    moves, switches = _menu(reg, pos)
    assert len(moves) == 4 and switches
    result = resolve_turn(reg, pos, _chosen(reg, pos, RAMP), budget=Budget.matrix())
    assert {_lock(b.position, with_roll=False) for b in result.branches} == {(("outrage", 1), False)}


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207).


def _port_generated(reg, port, positions: list[dict], case: Case, budget: Budget):  # noqa: ANN001, ANN202
    """`_generated` with the port resolving every branch."""
    from ._port_showdown import given, port_branches

    frontier = [(1.0, given(Position.from_json(positions[0])))]
    for step in case.steps:
        nxt = []
        for weight, pos in frontier:
            nxt.extend((weight * w, p) for w, p in port_branches(port, pos, _chosen(reg, pos, step), budget))
        frontier = nxt
    out: dict = {}
    for weight, pos in frontier:
        key = _lock(pos, with_roll=False)
        out[key] = out.get(key, 0.0) + weight
    return out


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "roll"), PLAYED)
def test_the_ports_turn_from_showdowns_position(reg, oracle: Oracle, port, name: str, roll: str) -> None:  # noqa: ANN001
    """`test_our_turn_from_showdowns_position` with the port's branches, every case."""
    from ._port_showdown import port_branches

    case = CASES[name]
    positions = _play(oracle, case, roll)
    start = Position.from_json(positions[-2])
    after = Position.from_json(positions[-1])
    for side in start.sides:
        for mon in side.pokemon:
            mon.trapped = False
    chosen = _chosen(reg, start, case.steps[-1])
    branches = [p for _, p in port_branches(port, start, chosen, Budget.matrix())]
    want = _expected(case, roll)
    started = _rampager(start).volatile("lockedmove") is None
    if started and want[0] is not None:
        want = ((*want[0][:2], None), want[1])
    locks = {_lock(p) for p in branches}
    assert locks == {want}, (locks, want)
    for pos in branches:
        assert _pp(pos) == _pp(after)
        assert _rampager(pos).item == _rampager(after).item
        moves, switches = _menu(reg, pos)
        if want[0] is not None:
            assert (moves, switches) == ({"outrage"}, False)
        else:
            assert len(moves) == 4 and switches


@pytest.mark.oracle
@pytest.mark.parametrize("name", MULTI_TURN)
def test_the_games_the_port_generates(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    """`test_the_games_we_generate` with the port: Showdown's two games, a half each."""
    case = CASES[name]
    want: dict = {}
    for roll in ROLLS:
        positions = _play(oracle, case, roll)
        lock, confused = _expected(case, roll)
        key = (None if lock is None else lock[:2], confused)
        want[key] = want.get(key, 0.0) + 0.5
    ours = _port_generated(reg, port, positions, case, Budget.matrix())
    assert ours.keys() == want.keys(), (ours, want)
    for key, weight in want.items():
        assert ours[key] == pytest.approx(weight), (key, ours, want)


@pytest.mark.oracle
@pytest.mark.parametrize("name", MULTI_TURN)
def test_the_ports_pinned_budget_rolls_two(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    """`test_the_pinned_budget_rolls_two` with the port."""
    case = CASES[name]
    positions = _play(oracle, case, "short")
    lock, confused = case.short
    want = {(None if lock is None else lock[:2], confused): 1.0}
    pinned = dataclasses.replace(Budget.deterministic(), pinned_policy=True)
    assert _port_generated(reg, port, positions, case, pinned) == pytest.approx(want)
