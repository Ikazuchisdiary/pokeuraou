"""Confusion wears off after `random(2, 6)` tries, and the Persim and Lum Berries cure any
confusion (IKA-177).

vendor/pokemon-showdown/data/conditions.ts, `confusion` (no Champions override)::

    onStart(target, source, sourceEffect) {
        ...
        const min = sourceEffect?.id === 'axekick' ? 3 : 2;
        this.effectState.time = this.random(min, 6);
    },
    onBeforeMovePriority: 3,
    onBeforeMove(pokemon) {
        pokemon.volatiles['confusion'].time--;
        if (!pokemon.volatiles['confusion'].time) {
            pokemon.removeVolatile('confusion');
            return;
        }
        this.add('-activate', pokemon, 'confusion');
        if (!this.randomChance(33, 100)) {
            return;
        }
        ... this.actions.getConfusionDamage(pokemon, 40) ...
        return false;
    },

`random(2, 6)` is 2 to 5: confused on the first one to four tries, cured on the next. Priority
3 is below sleep and freeze (10) and flinch (8), so a Pokemon that cannot get that far spends
no try, and above paralysis (1): a paralysed confused Pokemon rolls the self-hit first.
`persimberry` and `lumberry` eat themselves in `onUpdate` whenever `volatiles['confusion']`.

Our resolver (both engines) kept confusion until the Pokemon left the field, rolled paralysis
before the self-hit at 1/3 rather than 33/100, and only the rampage's fatigue confusion met
the berries. The cases are played by Showdown with `multihit='min'` (the roll is its lowest)
and `'max'` (its highest), and our resolver and the port are held to them.
"""

from __future__ import annotations

import dataclasses

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position
from pokeuraou.resolve import CONFUSION_SELF_HIT_CHANCE, FULL_PARALYSIS_CHANCE, Budget, resolve_turn

from .conftest import FORMAT_ID

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}
FAST = {"hp": 20, "atk": 20, "def": 10, "spa": 0, "spd": 0, "spe": 32}
SLOW = {"hp": 32, "atk": 20, "def": 10, "spa": 0, "spd": 4, "spe": 0}


def _mon(species: str, ability: str, moves: list[str], item: str | None = None,
         sp: dict | None = None) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves,
                   sp=dict(sp or SP), item=item)


GENGAR = _mon("Gengar", "Cursed Body", ["confuseray", "thunderwave", "nastyplot", "protect"], sp=FAST)
MEDICHAM = _mon("Medicham", "Telepathy", ["axekick", "calmmind", "protect", "icepunch"], sp=FAST)
ALLIES = [
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"]),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
]
FOE_ALLIES = [
    _mon("Hippowdon", "Sand Stream", ["slackoff", "protect", "earthquake", "yawn"]),
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]


def _kingambit(item: str | None = None) -> TeamSet:
    return _mon("Kingambit", "Defiant", ["swordsdance", "protect", "ironhead", "kowtowcleave"],
                item=item, sp=SLOW)


GARCHOMP = _mon("Garchomp", "Rough Skin", ["swordsdance", "protect", "earthquake", "outrage"], sp=SLOW)

#: p1a confuses p2a (move 1 at target 1); afterwards p1a sets up and p2a keeps trying to
#: Swords Dance. Milotic recovers and Hippowdon slacks off at full HP: nothing random.
CONFUSE = ["move 1 1, move 1", "move 1, move 1"]
SET_UP = ["move 3, move 1", "move 1, move 1"]
#: Thunder Wave first, so p2a is paralysed when it is confused.
PARALYSE = ["move 2 1, move 1", "move 1, move 1"]
#: Medicham's Axe Kick at Garchomp, then Calm Mind.
KICK = ["move 1 1, move 1", "move 1, move 1"]
MEDITATE = ["move 2, move 1", "move 1, move 1"]


@dataclasses.dataclass(frozen=True)
class Case:
    p1: TeamSet
    p2: TeamSet
    steps: list[list[str]]
    #: After each step, p2a's confusion as Showdown's `time` (0 when it has none), for the
    #: lowest roll and the highest.
    short: list[int]
    long: list[int]
    #: The roll Showdown asks for when it confuses: `random(low, 6)`.
    low: int = 2
    secondary: bool = False


CASES: dict[str, Case] = {
    # random(2, 6) -> 2: one confused try, cured on the second; -> 5: four, then cured.
    "confuse ray": Case(GENGAR, _kingambit(), [CONFUSE] + [SET_UP] * 5,
                        [1, 0, 0, 0, 0, 0], [4, 3, 2, 1, 0, 0]),
    # Axe Kick's `min = 3`: two confused tries at least.
    "axe kick": Case(MEDICHAM, GARCHOMP, [KICK] + [MEDITATE] * 5,
                     [2, 1, 0, 0, 0, 0], [4, 3, 2, 1, 0, 0], low=3, secondary=True),
    # The berries eat any confusion at once, not only a rampage's.
    "persim berry": Case(GENGAR, _kingambit("Persim Berry"), [CONFUSE, SET_UP], [0, 0], [0, 0]),
    "lum berry": Case(GENGAR, _kingambit("Lum Berry"), [CONFUSE, SET_UP], [0, 0], [0, 0]),
}
ROLLS = {"short": "min", "long": "max"}
PLAYED = [(name, roll) for name in sorted(CASES) for roll in sorted(ROLLS)]


def _play(oracle: Oracle, case: Case, roll: str, steps: list[list[str]] | None = None):  # noqa: ANN202
    """Showdown's positions before the first step and after each, and each step's rolls."""
    policy = RandomnessPolicy(multihit=ROLLS[roll], secondary=case.secondary)
    handle = oracle.create(FORMAT_ID, [case.p1, *ALLIES], [case.p2, *FOE_ALLIES], policy=policy)
    handle.step(["team 1234", "team 1234"])
    positions = [handle.position]
    rolls = []
    for step in steps or case.steps:
        handle.step(step)
        assert handle.choice_errors == [], handle.choice_errors
        positions.append(handle.position)
        rolls.append(handle.rolls)
    handle.close()
    return positions, rolls


def _target(pos: Position):  # noqa: ANN202
    return pos.sides[1].pokemon[pos.sides[1].active[0]]


def _time(pos: Position) -> int | None:
    """p2a's confusion: Showdown's `time`, None when the position does not carry one, 0
    when it is not confused."""
    held = _target(pos).volatile("confusion")
    if held is None:
        return 0
    time = held.extra.get("time")
    return time if isinstance(time, int) else None


def _chosen(reg, pos: Position, step: list[str]):  # noqa: ANN001, ANN202
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


def _loaded(raw: dict) -> Position:
    """Showdown's position as our resolver and the port take it: Showdown's trapped verdict
    cleared as a searched child carries it (IKA-169), and its stats dropped for the spreads,
    which the port would refuse as a transformed Pokemon."""
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
            mon.stats_override = None
    return pos


def _want(case: Case, roll: str) -> list[int]:
    return case.short if roll == "short" else case.long


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "roll"), PLAYED)
def test_showdown(oracle: Oracle, name: str, roll: str) -> None:
    """The facts: the length, the roll it came from, the self-hit's odds, the berry."""
    case = CASES[name]
    positions, rolls = _play(oracle, case, roll)
    assert [_time(Position.from_json(p)) for p in positions[1:]] == _want(case, roll)
    asked = [(r["kind"], r.get("from"), r.get("to"), r.get("numerator"), r.get("denominator"))
             for r in rolls[0]]
    assert ("random", case.low, 6, None, None) in asked
    if name.endswith("berry"):
        assert _target(Position.from_json(positions[-1])).item is None, "the berry is eaten"
    else:
        assert ("chance", None, None, 33, 100) in asked


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "roll"), PLAYED)
def test_our_turn_from_showdowns_position(reg, oracle: Oracle, name: str, roll: str) -> None:  # noqa: ANN001
    """Every step resolved by us from Showdown's position before it, which carries `time`.

    A confusion that starts in the step has no `time` in ours (the roll waits for the try
    where the lengths first differ), so only its presence is compared then.
    """
    case = CASES[name]
    positions, _ = _play(oracle, case, roll)
    for index, step in enumerate(case.steps):
        start = _loaded(positions[index])
        after = Position.from_json(positions[index + 1])
        result = resolve_turn(reg, start, _chosen(reg, start, step), budget=Budget.matrix())
        assert not result.suspended
        want = _time(after)
        got = {_time(b.position) for b in result.branches}
        items = {_target(b.position).item for b in result.branches}
        if want and _time(start) == 0:
            # Axe Kick's miss and its 70% without confusion are branches of ours too.
            assert None in got, (index, got)
            assert _target(after).item in items
            continue
        assert got == {want}, (index, got, want)
        assert items == {_target(after).item}


def _tree(reg, start: Position, steps: list[list[str]], budget: Budget) -> list[dict]:  # noqa: ANN001
    """Every step resolved by us from `start`, every branch followed: after each step, the
    probability that p2a is still confused, keyed by whether it is."""
    frontier = [(1.0, start)]
    out = []
    for step in steps:
        nxt = []
        for weight, pos in frontier:
            result = resolve_turn(reg, pos, _chosen(reg, pos, step), budget=budget)
            assert not result.suspended
            nxt.extend((weight * b.probability, b.position) for b in result.branches)
        frontier = nxt
        confused = sum(w for w, p in frontier if _target(p).has_volatile("confusion"))
        out.append(confused / sum(w for w, _ in frontier))
    return out


@pytest.mark.oracle
def test_the_games_we_generate(reg, oracle: Oracle) -> None:
    """Showdown rolls `time` once, 2 to 5; we branch each try where it may end. After k
    tries the chance of still being confused is P(time > k) = (5 - k) / 4, and Showdown's
    two games are the ends of it."""
    case = CASES["confuse ray"]
    positions, _ = _play(oracle, case, "short")
    ours = _tree(reg, _loaded(positions[0]), case.steps, Budget.matrix())
    assert ours == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0, 0.0])


@pytest.mark.oracle
def test_axe_kicks_confusion_from_our_own_position(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """`random(3, 6)`: after Axe Kick's first try the chance of still being confused is 1,
    then 2/3, 1/3, 0. From Showdown's position after the kick with the roll taken off, as
    our resolver writes its own."""
    case = CASES["axe kick"]
    positions, _ = _play(oracle, case, "short")
    start = _loaded(positions[1])
    held = _target(start).volatile("confusion")
    assert held is not None
    held.extra = {"tries": 1, "min": 3}
    ours = _tree(reg, start, case.steps[1:], Budget.matrix())
    assert ours == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0, 0.0])


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["confuse ray", "axe kick"])
def test_the_pinned_budget_rolls_the_lowest(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """`random(a, 6)` is `a` under the oracle's pinned policy, and the pinned budget agrees.

    The pinned budget collapses Axe Kick's 30% to "did not happen", so that case starts
    after the kick, from Showdown's position with the roll taken off."""
    case = CASES[name]
    positions, _ = _play(oracle, case, "short")
    pinned = dataclasses.replace(Budget.deterministic(), pinned_policy=True)
    first = 0 if name == "confuse ray" else 1
    start = _loaded(positions[first])
    if first:
        held = _target(start).volatile("confusion")
        assert held is not None
        held.extra = {"tries": 1, "min": 3}
    ours = _tree(reg, start, case.steps[first:], pinned)
    assert ours == [1.0 if t else 0.0 for t in case.short[first:]]


@pytest.mark.oracle
def test_confusion_is_rolled_before_paralysis(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Priority 3 against paralysis's 1: Showdown asks the self-hit first, and under the
    policy it hits, so paralysis is never asked. Our weights follow the same order."""
    case = CASES["confuse ray"]
    positions, rolls = _play(oracle, case, "long", [PARALYSE, CONFUSE, SET_UP])
    asked = [(r["kind"], r.get("numerator"), r.get("denominator")) for r in rolls[2]]
    assert ("chance", 33, 100) in asked and ("chance", 1, 8) not in asked, asked
    start = _loaded(positions[2])
    assert _target(start).status == "par" and _time(start) == 4
    result = resolve_turn(reg, start, _chosen(reg, start, SET_UP), budget=Budget.matrix())
    before = _target(start)
    by_kind: dict[str, float] = {}
    for branch in result.branches:
        mon = _target(branch.position)
        kind = ("self-hit" if mon.hp < before.hp
                else "acted" if mon.boosts.get("atk", 0) > before.boosts.get("atk", 0)
                else "paralysed")
        by_kind[kind] = by_kind.get(kind, 0.0) + branch.probability
    assert CONFUSION_SELF_HIT_CHANCE == 0.33
    hit = CONFUSION_SELF_HIT_CHANCE
    assert by_kind == pytest.approx({
        "self-hit": hit,
        "paralysed": (1 - hit) * FULL_PARALYSIS_CHANCE,
        "acted": (1 - hit) * (1 - FULL_PARALYSIS_CHANCE),
    })


def _port(reg, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001, ANN202
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    return node


def _both(reg, node, start: Position, step: list[str]):  # noqa: ANN001, ANN202
    """Python's branches and the port's, as sorted (probability, time, item)."""
    chosen = _chosen(reg, start, step)
    python = resolve_turn(reg, start, chosen, budget=Budget.matrix())
    weights = node.resolve(start, chosen, Budget.matrix(), select=None)
    assert weights is not None, "the port refused the turn"
    picked = [
        node.resolve(start, chosen, Budget.matrix(), select=i).position
        for i in range(len(weights.branches))
    ]

    def key(pos: Position):  # noqa: ANN202
        held = _target(pos).volatile("confusion")
        return (_time(pos), None if held is None else held.extra.get("tries"),
                _target(pos).item, _target(pos).hp)

    want = sorted(((b.probability, key(b.position)) for b in python.branches), key=repr)
    got = sorted(((w, key(p)) for w, p in zip(weights.branches, picked, strict=True)), key=repr)
    return want, got


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "roll"), PLAYED)
def test_the_port_agrees(
    reg, oracle: Oracle, monkeypatch: pytest.MonkeyPatch, name: str, roll: str  # noqa: ANN001
) -> None:
    """The port resolves every step from Showdown's position as Python does."""
    case = CASES[name]
    positions, _ = _play(oracle, case, roll)
    node = _port(reg, monkeypatch)
    try:
        for index, step in enumerate(case.steps):
            want, got = _both(reg, node, _loaded(positions[index]), step)
            assert [k for _, k in got] == [k for _, k in want], index
            assert [w for w, _ in got] == pytest.approx([w for w, _ in want]), index
    finally:
        rustnode.reset()


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["confuse ray", "axe kick"])
def test_the_port_branches_the_roll(
    reg, oracle: Oracle, monkeypatch: pytest.MonkeyPatch, name: str  # noqa: ANN001
) -> None:
    """From positions our resolver made (no `time`), both engines branch each try alike."""
    case = CASES[name]
    positions, _ = _play(oracle, case, "short")
    node = _port(reg, monkeypatch)
    try:
        start = _loaded(positions[0])
        for step in case.steps[:5]:
            want, got = _both(reg, node, start, step)
            assert [k for _, k in got] == [k for _, k in want]
            assert [w for w, _ in got] == pytest.approx([w for w, _ in want])
            # Follow the heaviest branch that is still confused, if any.
            python = resolve_turn(reg, start, _chosen(reg, start, step), budget=Budget.matrix())
            confused = [b for b in python.branches if _target(b.position).has_volatile("confusion")]
            if not confused:
                break
            start = max(confused, key=lambda b: b.probability).position
    finally:
        rustnode.reset()


# ---------------------------------------------------------------------------
# No oracle.


def _hand_built(reg, extra: dict | None, item: str | None = None):  # noqa: ANN001, ANN202
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, [GENGAR, *ALLIES])
    other = _synthetic_position(reg, [_kingambit(item), *FOE_ALLIES])
    pos.sides[1] = other.sides[0]
    pos.sides[1].id = pos.sides[1].name = "p2"
    if extra is not None:
        _target(pos).volatiles.append(Effect(id="confusion", extra=dict(extra)))
    return pos


@pytest.mark.parametrize(
    ("extra", "cured"),
    [
        # Showdown's own: the try spends one, cured at zero.
        ({"time": 1}, 1.0), ({"time": 2}, 0.0),
        # Ours: the chance the length ends at this try, given it has not yet.
        ({"tries": 0}, 0.0), ({"tries": 1}, 1 / 4), ({"tries": 2}, 1 / 3),
        ({"tries": 3}, 1 / 2), ({"tries": 4}, 1.0),
        ({"tries": 1, "min": 3}, 0.0), ({"tries": 2, "min": 3}, 1 / 3),
        # A record made before IKA-177 carries none: read as fresh.
        ({}, 0.0),
    ],
)
def test_the_try_that_cures(reg, extra: dict, cured: float) -> None:  # noqa: ANN001
    pos = _hand_built(reg, extra)
    result = resolve_turn(reg, pos, _chosen(reg, pos, SET_UP), budget=Budget.matrix())
    got = sum(b.probability for b in result.branches
              if not _target(b.position).has_volatile("confusion"))
    assert got == pytest.approx(cured)


def test_a_pokemon_that_cannot_try_spends_nothing(reg) -> None:  # noqa: ANN001
    """Flinch (priority 8) answers before confusion (3): no try, no cure, no roll."""
    pos = _hand_built(reg, {"time": 1})
    _target(pos).volatiles.append(Effect(id="flinch", duration=1))
    result = resolve_turn(reg, pos, _chosen(reg, pos, SET_UP), budget=Budget.matrix())
    assert {_time(b.position) for b in result.branches} == {1}
    pos = _hand_built(reg, {"tries": 1})
    _target(pos).volatiles.append(Effect(id="flinch", duration=1))
    result = resolve_turn(reg, pos, _chosen(reg, pos, SET_UP), budget=Budget.matrix())
    assert len(result.branches) == 1
    held = _target(result.branches[0].position).volatile("confusion")
    assert held is not None and held.extra.get("tries") == 1


def test_unnerve_keeps_the_berry(reg) -> None:  # noqa: ANN001
    """The control of the berry: an opposing Unnerve forbids eating it."""
    pos = _hand_built(reg, None, item="persimberry")
    pos.sides[0].pokemon[pos.sides[0].active[0]].ability = "unnerve"
    result = resolve_turn(reg, pos, _chosen(reg, pos, CONFUSE), budget=Budget.matrix())
    for branch in result.branches:
        assert _target(branch.position).item == "persimberry"
        assert _target(branch.position).has_volatile("confusion")
