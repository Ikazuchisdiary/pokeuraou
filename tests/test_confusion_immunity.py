"""What refuses a confusion, and the self-hit's damage roll (IKA-189).

vendor/pokemon-showdown (a5df827; the champions mod overrides none of these)::

    // data/abilities.ts, owntempo (flags: {breakable: 1})
    onUpdate(pokemon) { if (pokemon.volatiles['confusion']) { ...; pokemon.removeVolatile('confusion'); } },
    onTryAddVolatile(status, pokemon) { if (status.id === 'confusion') return null; },

    // data/moves.ts, mistyterrain.condition
    onTryAddVolatile(status, target, source, effect) {
        if (!target.isGrounded() || target.isSemiInvulnerable()) return;
        if (status.id === 'confusion') { ...; return null; }
    },

    // data/moves.ts, safeguard.condition
    onTryAddVolatile(status, target, source, effect) {
        if (!effect || !source) return;
        if (effect.effectType === 'Move' && effect.infiltrates && !target.isAlly(source)) return;
        if ((status.id === 'confusion' || status.id === 'yawn') && target !== source) { ...; return null; }
    },

    // sim/battle-actions.ts
    getConfusionDamage(pokemon, basePower) {
        const attack = pokemon.calculateStat('atk', pokemon.boosts['atk']);
        const defense = pokemon.calculateStat('def', pokemon.boosts['def']);
        const baseDamage = tr(tr(tr(tr(2 * level / 5 + 2) * basePower * attack) / defense) / 50) + 2;
        let damage = tr(baseDamage, 16);
        damage = this.battle.randomizer(damage);
        return Math.max(1, damage);
    }

`addVolatile` runs `TryAddVolatile` for every confusion -- Confuse Ray, Swagger's after its
boost, Hurricane's secondary -- and fills in `source = this` when there is none, so a
rampage's fatigue is never Safeguard's (`target !== source`). A Mold Breaker move passes
Own Tempo's `onTryAddVolatile` (the ability is `breakable`): the confusion starts, a Persim
or Lum Berry eats it, and otherwise Own Tempo's `onUpdate` cures it at once -- so Own Tempo
always ends unconfused, and only Mold Breaker costs it the berry.

Our resolver (both engines) looked at Own Tempo and Misty Terrain only for the fatigue, at
Safeguard never, and dealt the self-hit at the highest roll whatever the budget's.
"""

from __future__ import annotations

import dataclasses

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position
from pokeuraou.resolve import CONFUSION_SELF_HIT_CHANCE, Budget, resolve_turn

from .conftest import FORMAT_ID

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}
FAST = {"hp": 20, "atk": 20, "def": 10, "spa": 0, "spd": 0, "spe": 32}
SLOW = {"hp": 32, "atk": 20, "def": 10, "spa": 0, "spd": 4, "spe": 0}


def _mon(species: str, ability: str, moves: list[str], item: str | None = None,
         sp: dict | None = None) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves,
                   sp=dict(sp or SP), item=item)


CONFUSERS = ["confuseray", "swagger", "hurricane", "protect"]
GENGAR = _mon("Gengar", "Cursed Body", CONFUSERS, sp=FAST)
ALLIES = [
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"]),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
]
#: p2b lays Misty Terrain (move 1) or Safeguard (move 2) on the first step.
FOE_ALLIES = [
    _mon("Sylveon", "Pixilate", ["mistyterrain", "safeguard", "protect", "moonblast"], sp=FAST),
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]
SETUP = ["swordsdance", "protect", "slackoff", "scald"]
KINGAMBIT = _mon("Kingambit", "Defiant", SETUP, sp=SLOW)
SLOWBRO = _mon("Slowbro", "Own Tempo", SETUP, sp=SLOW)
CHARIZARD = _mon("Charizard", "Blaze", SETUP, sp=SLOW)
MOVES = {"confuse ray": 1, "swagger": 2, "hurricane": 3}
LAY = {"none": 3, "misty": 1, "safeguard": 2}


def _steps(lay: str, move: str) -> list[list[str]]:
    """p2a Protects while p2b lays the guard; then p1a confuses p2a, which Swords Dances."""
    return [["move 4, move 1", f"move 2, move {LAY[lay]}"],
            [f"move {MOVES[move]} 1, move 1", "move 1, move 3"]]


@dataclasses.dataclass(frozen=True)
class Case:
    p1: TeamSet
    p2: TeamSet
    lay: str
    move: str
    #: Whether p2a is confused after Showdown's second step.
    confused: bool

    @property
    def steps(self) -> list[list[str]]:
        return _steps(self.lay, self.move)


def _cases() -> dict[str, Case]:
    out: dict[str, Case] = {}
    for move in MOVES:
        out[f"own tempo, {move}"] = Case(GENGAR, SLOWBRO, "none", move, False)
        out[f"misty terrain, {move}"] = Case(GENGAR, KINGAMBIT, "misty", move, False)
        # The control of the terrain: it guards only the grounded.
        out[f"misty terrain, airborne, {move}"] = Case(GENGAR, CHARIZARD, "misty", move, True)
        out[f"safeguard, {move}"] = Case(GENGAR, KINGAMBIT, "safeguard", move, False)
    # The control of Safeguard: an Infiltrator's move passes it.
    out["safeguard, infiltrator"] = Case(
        _mon("Gengar", "Infiltrator", CONFUSERS, sp=FAST), KINGAMBIT, "safeguard", "confuse ray", True
    )
    # Mold Breaker passes `onTryAddVolatile`, so the confusion starts and the Lum Berry eats
    # it; Own Tempo's `onUpdate` would have cured it anyway.
    out["own tempo, mold breaker"] = Case(
        _mon("Gengar", "Mold Breaker", CONFUSERS, sp=FAST),
        _mon("Slowbro", "Own Tempo", SETUP, item="Lum Berry", sp=SLOW), "none", "confuse ray", False,
    )
    # Its control: without Mold Breaker nothing starts, and the berry stays.
    out["own tempo, lum berry"] = Case(
        GENGAR, _mon("Slowbro", "Own Tempo", SETUP, item="Lum Berry", sp=SLOW), "none",
        "confuse ray", False,
    )
    # The control of it all: nothing guards.
    out["unguarded"] = Case(GENGAR, KINGAMBIT, "none", "confuse ray", True)
    return out


CASES = _cases()


def _play(oracle: Oracle, p1: TeamSet, p2: TeamSet, steps: list[list[str]],
          policy: RandomnessPolicy) -> list[dict]:
    """Showdown's positions before the first step and after each."""
    handle = oracle.create(FORMAT_ID, [p1, *ALLIES], [p2, *FOE_ALLIES], policy=policy)
    handle.step(["team 1234", "team 1234"])
    positions = [handle.position]
    for step in steps:
        handle.step(step)
        assert handle.choice_errors == [], handle.choice_errors
        positions.append(handle.position)
    handle.close()
    return positions


def _policy(case: Case) -> RandomnessPolicy:
    return RandomnessPolicy(secondary=case.move == "hurricane", multihit="max")


def _target(pos: Position):  # noqa: ANN202
    return pos.sides[1].pokemon[pos.sides[1].active[0]]


def _chosen(reg, pos: Position, step: list[str]):  # noqa: ANN001, ANN202
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


def _loaded(raw: dict) -> Position:
    """Showdown's position as our resolver and the port take it (as IKA-177's tests)."""
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
            mon.stats_override = None
    return pos


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown(oracle: Oracle, name: str) -> None:
    """The facts: who ends the step confused; Swagger's boost lands either way; only the
    Mold Breaker case loses its Lum Berry."""
    case = CASES[name]
    positions = _play(oracle, case.p1, case.p2, case.steps, _policy(case))
    after = _target(Position.from_json(positions[-1]))
    assert after.has_volatile("confusion") == case.confused
    boost = 2 + (2 if case.move == "swagger" else 0)
    if not case.confused:  # a confused Pokemon hits itself under the policy: no dance
        assert after.boosts.get("atk", 0) == boost
    if name.startswith("own tempo, ") and case.p2.item:
        assert after.item == (None if name == "own tempo, mold breaker" else "lumberry")


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_our_turn_from_showdowns_position(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """The confusing step resolved by us from Showdown's position before it. A refused
    confusion is in none of our branches; an allowed one is in some (a miss, Hurricane's
    70% and Swagger's 85% are branches of ours)."""
    case = CASES[name]
    positions = _play(oracle, case.p1, case.p2, case.steps, _policy(case))
    start = _loaded(positions[1])
    result = resolve_turn(reg, start, _chosen(reg, start, case.steps[1]), budget=Budget.matrix())
    assert not result.suspended
    confused = {_target(b.position).has_volatile("confusion") for b in result.branches}
    assert (True in confused) == case.confused, confused
    if case.move == "swagger":
        # The boost lands on every branch Swagger hits, guarded or not.
        assert any(_target(b.position).boosts.get("atk", 0) == 2 for b in result.branches)
    if name.startswith("own tempo, ") and case.p2.item:
        eaten = name == "own tempo, mold breaker"
        assert {_target(b.position).item for b in result.branches} == {None if eaten else "lumberry"}


# ---------------------------------------------------------------------------
# The self-hit's roll.

#: p1a confuses p2a; next step p1a Protects and p2a, confused, tries to Swords Dance -- the
#: policy's `randomChance(33, 100)` says it hits itself. `multihit='max'` makes `time` 5.
HURT = [["move 1 1, move 1", "move 1, move 3"], ["move 4, move 1", "move 1, move 3"]]
#: The same through Swagger, so the self-hit is at +2 Attack.
HURT_BOOSTED = [["move 2 1, move 1", "move 1, move 3"], ["move 4, move 1", "move 1, move 3"]]


def _hurt(oracle: Oracle, roll: int, steps: list[list[str]]) -> list[dict]:
    policy = RandomnessPolicy(damage_roll=roll, multihit="max")
    return _play(oracle, GENGAR, KINGAMBIT, steps, policy)


def _self_hits(reg, start: Position, step: list[str], budget: Budget) -> dict[int, float]:  # noqa: ANN001
    """HP after the self-hit -> our probability of it, over the branches where p2a hurt itself."""
    before = _target(start)
    result = resolve_turn(reg, start, _chosen(reg, start, step), budget=budget)
    out: dict[int, float] = {}
    for branch in result.branches:
        mon = _target(branch.position)
        if mon.hp < before.hp:
            out[mon.hp] = out.get(mon.hp, 0.0) + branch.probability
    return out


@pytest.mark.oracle
@pytest.mark.parametrize("roll", [0, 5, 8, 15])
@pytest.mark.parametrize("steps", ["plain", "boosted"])
def test_the_self_hit_takes_the_budgets_roll(reg, oracle: Oracle, roll: int, steps: str) -> None:  # noqa: ANN001
    """A budget pinned to roll r deals Showdown's self-hit at `damageRoll` r."""
    played = HURT if steps == "plain" else HURT_BOOSTED
    positions = _hurt(oracle, roll, played)
    start = _loaded(positions[1])
    want = _target(Position.from_json(positions[2])).hp
    assert want < _target(start).hp
    got = _self_hits(reg, start, played[1], Budget.matrix(roll))
    assert got == pytest.approx({want: CONFUSION_SELF_HIT_CHANCE})


@pytest.mark.oracle
def test_the_exact_budget_branches_every_roll(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """All sixteen rolls, each a sixteenth of the 33/100: Showdown's sixteen games. Each
    game's first step hurt it at its own roll too, so the damage is what is compared."""
    want: dict[int, float] = {}
    for roll in range(16):
        positions = _hurt(oracle, roll, HURT)
        dealt = _target(Position.from_json(positions[1])).hp - _target(
            Position.from_json(positions[2])
        ).hp
        want[dealt] = want.get(dealt, 0.0) + CONFUSION_SELF_HIT_CHANCE / 16
    assert len(want) > 1
    start = _loaded(_hurt(oracle, 0, HURT)[1])
    got = _self_hits(reg, start, HURT[1], Budget.exact())
    assert {_target(start).hp - hp: p for hp, p in got.items()} == pytest.approx(want)


# ---------------------------------------------------------------------------
# The port.


def _port(reg, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001, ANN202
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    return node


def _both(reg, node, start: Position, step: list[str], budget: Budget):  # noqa: ANN001, ANN202
    """Python's branches and the port's, as sorted (probability, confused, atk, item, hp)."""
    chosen = _chosen(reg, start, step)
    python = resolve_turn(reg, start, chosen, budget=budget)
    weights = node.resolve(start, chosen, budget, select=None)
    assert weights is not None, "the port refused the turn"
    picked = [node.resolve(start, chosen, budget, select=i).position
              for i in range(len(weights.branches))]

    def key(pos: Position):  # noqa: ANN202
        mon = _target(pos)
        return (mon.has_volatile("confusion"), mon.boosts.get("atk", 0), mon.item, mon.hp)

    # Sorted by the key, then the weight rounded: merged branches' sums differ in the last
    # bit between the engines, which is enough to reorder a sort on the weight's repr.
    def order(item):  # noqa: ANN001, ANN202
        return (repr(item[1]), round(item[0], 12))

    want = sorted(((b.probability, key(b.position)) for b in python.branches), key=order)
    got = sorted(((w, key(p)) for w, p in zip(weights.branches, picked, strict=True)), key=order)
    return want, got


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_agrees(reg, oracle: Oracle, monkeypatch: pytest.MonkeyPatch, name: str) -> None:  # noqa: ANN001
    case = CASES[name]
    positions = _play(oracle, case.p1, case.p2, case.steps, _policy(case))
    node = _port(reg, monkeypatch)
    try:
        want, got = _both(reg, node, _loaded(positions[1]), case.steps[1], Budget.matrix())
        assert [k for _, k in got] == [k for _, k in want]
        assert [w for w, _ in got] == pytest.approx([w for w, _ in want])
    finally:
        rustnode.reset()


@pytest.mark.oracle
@pytest.mark.parametrize("budget", ["matrix 15", "exact"])
def test_the_port_rolls_the_self_hit(
    reg, oracle: Oracle, monkeypatch: pytest.MonkeyPatch, budget: str  # noqa: ANN001
) -> None:
    start = _loaded(_hurt(oracle, 0, HURT_BOOSTED)[1])
    chosen = Budget.matrix(15) if budget == "matrix 15" else Budget.exact()
    node = _port(reg, monkeypatch)
    try:
        want, got = _both(reg, node, start, HURT_BOOSTED[1], chosen)
        assert [k for _, k in got] == [k for _, k in want]
        assert [w for w, _ in got] == pytest.approx([w for w, _ in want])
    finally:
        rustnode.reset()


# ---------------------------------------------------------------------------
# No oracle.


def _hand_built(reg, p1: TeamSet, p2: TeamSet) -> Position:  # noqa: ANN001
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, [p1, *ALLIES])
    other = _synthetic_position(reg, [p2, *FOE_ALLIES])
    pos.sides[1] = other.sides[0]
    pos.sides[1].id = pos.sides[1].name = "p2"
    return pos


def test_safeguard_does_not_stop_the_fatigue(reg) -> None:  # noqa: ANN001
    """A rampage's `onEnd` confusion has no source, so `addVolatile` makes it the Pokemon
    itself and Safeguard's `target !== source` lets it through."""
    rampager = _mon("Garchomp", "Rough Skin", ["outrage", "protect", "earthquake", "swordsdance"])
    pos = _hand_built(reg, GENGAR, rampager)
    pos.sides[1].side_conditions.append(Effect(id="safeguard", duration=3))
    _target(pos).volatiles.append(
        Effect(id="lockedmove", duration=1, move="outrage", extra={"trueDuration": 1})
    )
    step = ["move 4, move 1", "move 1, move 3"]
    result = resolve_turn(reg, pos, _chosen(reg, pos, step), budget=Budget.matrix())
    assert all(_target(b.position).has_volatile("confusion") for b in result.branches)
