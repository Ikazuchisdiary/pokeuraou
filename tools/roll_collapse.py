"""How far sixteen damage rolls collapse -- losslessly, and by interval -- on real attacks.

Two different collapses, measured per hit on the attacks recorded games actually contain:

    distinct    how many different damage *numbers* the sixteen rolls produce. Integer HP
                does this for free, and it is exactly what the lossless merge (IKA-10)
                folds: equal damage is the same successor position
    intervals   how many groups the rolls fall into once they are separated only where
                something discrete changes -- the target faints, or its HP crosses a
                level its own item or ability reads (a pinch berry at 1/2, Blaze and
                friends at 1/3). That is the floor interval stratification (IKA-11) aims at

The two are far apart, which is the point: a hit whose sixteen rolls give eight different
damage numbers is eight branches after the lossless merge and one branch after
stratification, if none of those eight crosses anything that matters.

A spread move hitting two targets multiplies: 16x16 = 256 unmerged, distinct^2 merged,
intervals^2 stratified. That is the opening-turn case this was written to answer.

No turn is resolved here -- this is the damage calculator alone, which is why a large
sample is cheap. What it cannot tell you is what the *value* does; `tools/roll_headroom.py`
measures that.

    uv run python tools/roll_collapse.py --positions 80 --max-turn 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.damage import calculate, register_mega_stones  # noqa: E402
from pokeuraou.position import Pokemon, Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.view import battler, field_state, move_hits_multiple  # noqa: E402

#: Items whose holder reads a fraction of its max HP after damage.
HALF_BERRIES = frozenset(
    {"sitrusberry", "figyberry", "wikiberry", "magoberry", "aguavberry", "iapapaberry"}
)
#: Abilities that read 1/3.
PINCH_ABILITIES = frozenset({"blaze", "torrent", "overgrow", "swarm"})


def boundaries(mon: Pokemon) -> list[int]:
    """The HP levels crossing which changes something discrete for this Pokemon.

    Zero is always one of them: fainting is the boundary that matters most and every
    Pokemon has it. The others are read off the target's own item and ability rather than
    assumed, because a level nothing reads is not a boundary -- counting it would split an
    interval that nothing can tell apart and overstate what stratification has to keep.
    """
    out = [0]
    if (mon.item or "") in HALF_BERRIES:
        out.append(mon.maxhp // 2)
    if (mon.ability or "") in PINCH_ABILITIES:
        out.append(mon.maxhp // 3)
    return out


def load_positions(games_dir: Path, seed: int, cap: int) -> list[tuple[int, Position]]:
    out: list[tuple[int, Position]] = []
    for path in sorted(games_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for decision in record.get("decisions", ()):
                    if decision.get("kind") == "move":
                        out.append(
                            (
                                int(decision.get("turn", 0)),
                                Position.from_json(decision["position"]),
                            )
                        )
                if len(out) > cap:
                    break
        break
    random.Random(seed).shuffle(out)
    return out


def summarise(counter: Counter[int]) -> str:
    total = sum(counter.values())
    mean = sum(k * v for k, v in counter.items()) / total
    shares = " ".join(
        f"{k}:{v * 100 // total}%" for k, v in sorted(counter.items()) if v * 100 // total
    )
    return f"mean {mean:5.2f}   {shares}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=80)
    ap.add_argument("--max-turn", type=int, default=99, help="3 for the opening only")
    ap.add_argument("--min-turn", type=int, default=0)
    ap.add_argument("--games-dir", type=Path, default=Path("data/selfplay-gen11L"))
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    positions = load_positions(args.games_dir, args.seed, 3000)
    if not positions:
        raise SystemExit(f"no recorded positions found under {args.games_dir}")
    reg = load_regulation(positions[0][1].format)
    register_mega_stones(reg)

    distinct_counts: Counter[int] = Counter()
    interval_counts: Counter[int] = Counter()
    pairs = used = 0
    for turn, pos in positions:
        if turn > args.max_turn or turn < args.min_turn:
            continue
        if used >= args.positions:
            break
        used += 1
        fs = field_state(pos, reg)
        for side in (0, 1):
            foe = 1 - side
            live_foes = sum(
                1 for m in pos.sides[foe].active_pokemon() if m is not None and not m.fainted
            )
            for slot in (0, 1):
                mon = pos.sides[side].active_pokemon()[slot]
                if mon is None or mon.fainted:
                    continue
                attacker = battler(reg, mon)
                for move_slot in mon.moves:
                    move = reg.moves.get(move_slot.id)
                    if move is None or move.category == "Status":
                        continue
                    spread = move_hits_multiple(reg, move_slot.id, live_foes)
                    for foe_slot in (0, 1):
                        target = pos.sides[foe].active_pokemon()[foe_slot]
                        if target is None or target.fainted:
                            continue
                        try:
                            result = calculate(
                                reg,
                                attacker,
                                battler(reg, target),
                                move_slot.id,
                                fs,
                                defender_side=foe,
                                spread=spread,
                            )
                        except Exception:  # noqa: BLE001 - an unmodelled move is not the subject
                            continue
                        if result.immune:
                            continue
                        rolls = np.asarray(result.rolls[0], dtype=np.int64)
                        if rolls.max() <= 0:
                            continue
                        pairs += 1
                        distinct_counts[len(set(rolls.tolist()))] += 1
                        left = np.maximum(target.hp - rolls, 0)
                        levels = boundaries(target)
                        groups = {
                            tuple(int(hp <= level) for level in levels) for hp in left.tolist()
                        }
                        interval_counts[len(groups)] += 1

    if not pairs:
        raise SystemExit("no attacker-target pair produced damage")
    print(
        f"  {used} positions with turn in [{args.min_turn}, {args.max_turn}], "
        f"{pairs} attacker-target pairs\n"
    )
    print(f"  distinct damage values per hit   {summarise(distinct_counts)}")
    print(f"  intervals per hit                {summarise(interval_counts)}")
    distinct = sum(k * v for k, v in distinct_counts.items()) / pairs
    intervals = sum(k * v for k, v in interval_counts.items()) / pairs
    print(
        f"\n  one hit       16 -> {distinct:.1f} lossless -> {intervals:.2f} by interval\n"
        f"  two targets  256 -> {distinct ** 2:.0f} lossless -> {intervals ** 2:.2f} by interval\n"
        f"  four hits  65,536 -> {distinct ** 4:,.0f} lossless -> "
        f"{intervals ** 4:.2f} by interval"
    )
    print(
        "\n  the multiplications assume the hits are independent, which they are not once a\n"
        "  target has fainted -- read them as the shape of the tree, not as a count."
    )


if __name__ == "__main__":
    main()
