"""Does a guaranteed KO still branch sixteen ways?

`stratified_rolls` is handed the budget and nothing else, so it cannot know that every
roll produces the same position. The question is whether anything downstream collapses
them. Isolated by turning off every other source of randomness and using a move with no
recoil and no secondary, so the ONLY thing that can branch is the damage roll.

Three targets, same move: one the roll cannot save, one it cannot kill, and one where it
decides. If the first two return one branch each, the collapse is already there; if they
return sixteen, it is the free reduction G26 puts first.

**They returned sixteen, and the merge that answers it is in** (`Budget.merge_duplicates`,
IKA-10), so this now prints one branch for the guaranteed knock-out. It is kept as the
measurement that opened the issue and as the check that the fold still happens:

    POKEURAOU_MERGE_BRANCHES=0 uv run python tools/ko_branch_count.py

prints the sixteen it used to.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "C:/Users/Ikazuchi/repos/pokeuraou/src")
sys.path.insert(0, "C:/Users/Ikazuchi/repos/pokeuraou/tools")

from human_baseline import CASES, position_of  # noqa: E402

from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn  # noqa: E402

case = next(c for c in CASES if c.name == "sash-ko")
reg = load_regulation("gen9championsvgc2026regmb")
register_mega_stones(reg)

# Only the damage roll may branch.
budget = Budget(
    enumerate_crit=False,
    enumerate_accuracy=False,
    enumerate_status_checks=False,
    enumerate_secondary=False,
    enumerate_speed_ties=False,
)

# Sylveon's Quick Attack: single target, no recoil, no secondary. Incineroar Protects so
# it contributes nothing. Their Charizard Protects and Venusaur uses Sleep Powder at us,
# so neither of their actions damages anything.
OURS = "move 2 2, move 4"
THEIRS = "move 4, move 3 1"

for label, hp, item in (
    ("確定KO（どのロールでも死ぬ）", 5, None),
    ("確定で生き残る（どのロールでも死なない）", 187, None),
    ("ロールが決める（タスキ無し・満タン）", 187, "focussash"),
):
    pos, _ = position_of(case)
    target = pos.sides[1].pokemon[1]
    assert target.species == "venusaur"
    target.hp = hp
    target.item = item
    legal = {s: {a.to_choice(): a for a in side_actions(reg, pos, s)} for s in (0, 1)}
    if OURS not in legal[0] or THEIRS not in legal[1]:
        print(f"  {label}: 手が非合法 {OURS!r} / {THEIRS!r}")
        continue
    res = resolve_turn(reg, pos, [legal[0][OURS], legal[1][THEIRS]], budget=budget)
    dead = sum(
        b.probability
        for b in res.branches
        if (m := b.position.sides[1].active_pokemon()[1]) is None or m.fainted
    )
    hps = sorted(
        {
            (m.hp if (m := b.position.sides[1].active_pokemon()[1]) is not None else -1)
            for b in res.branches
        }
    )
    print(
        f"  {label:34s}  分岐 {len(res.branches):3d}  "
        f"相異なる残りHP {len(hps):2d}  KO率 {dead:.1%}  exact={res.exact}"
    )

# And the same question one level up: does the payoff builder evaluate duplicate leaves?
print("\n  同一の後続局面が葉のリストに何回入るか（確定KOの列）")
pos, _ = position_of(case)
target = pos.sides[1].pokemon[1]
target.hp, target.item = 5, None
legal = {s: {a.to_choice(): a for a in side_actions(reg, pos, s)} for s in (0, 1)}
res = resolve_turn(reg, pos, [legal[0][OURS], legal[1][THEIRS]], budget=budget)
keys = [
    tuple(
        (m.species, m.hp, m.status, tuple(v.id for v in m.volatiles))
        for side in b.position.sides
        for m in side.pokemon
    )
    for b in res.branches
]
print(f"    枝 {len(keys)} 本 → 相異なる局面 {len(set(keys))} 個")
print(f"    16ロールを保ったまま {len(keys) - len(set(keys))} 本が重複")
