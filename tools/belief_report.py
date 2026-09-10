"""How much the belief layer actually buys, measured rather than asserted.

The design note said the particle set would be made affordable by "position-dependent
lossless equivalence classes". That claim needs a number, and the number turned out to be
small: in an ordinary doubles position most stats are read by something, so most particles
are genuinely distinguishable and the lossless collapse is worth tens of percent rather
than orders of magnitude.

What does the work instead is the revealed nature -- a large, exact cut taken straight
from the open team sheet -- and truncating to the heaviest joint classes, whose error is
measurable per position with ``--stability``. This tool measures all three over positions
sampled from the usage prior, so the README can quote them instead of hoping.

    uv run python tools/belief_report.py --positions 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.belief import build_belief, live_stats_for, reduce_for_position
from pokeuraou.cli import analyse
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.payoff import FAINTS, HP_SHARE
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team
from pokeuraou.regulation import load_regulation
from pokeuraou.setup import parse_scenario

FORMAT_ID = "gen9championsvgc2026regmc"


def scenario_from_samples(reg, prior, rng) -> dict:  # noqa: ANN001
    """A scenario JSON built from two teams the prior actually generates."""

    def entry(sampled, known: bool) -> dict:  # noqa: ANN001
        out = {
            "species": sampled.species,
            "ability": sampled.ability,
            "item": sampled.item,
            "nature": sampled.nature,
            "moves": sampled.moves,
        }
        if known:
            out["sp"] = {k: int(v) for k, v in sampled.sp.items() if v}
        return out

    sides = []
    for side_index in range(2):
        # `sample_team` builds the six brought to the tournament; the selection phase is
        # out of scope, so the analysis starts from the four that were picked.
        team = [entry(s, side_index == 0) for s in sample_team(rng, reg, prior)[:4]]
        team[0]["active"] = True
        team[1]["active"] = True
        sides.append({"id": f"p{side_index + 1}", "team": team})
    return {"regulation": FORMAT_ID, "turn": 1, "sides": sides, "field": {}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument(
        "--stability", type=int, default=0,
        help="この数の局面で、同値類 4/8/16 の均衡を解き比べる（既定 0 = やらない）",
    )
    ap.add_argument("--stability-limit", type=int, default=8)
    args = ap.parse_args()

    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    register_mega_stones(reg)
    rng = np.random.default_rng(args.seed)

    raw_particles: list[int] = []
    conditioned: list[int] = []
    classes: list[int] = []
    factors: list[float] = []
    top_mass: list[float] = []
    covered: list[float] = []
    live_counter: Counter[str] = Counter()
    live_sizes: Counter[int] = Counter()
    candidates: list[int] = []
    kept: list[int] = []
    coverage_failures = 0
    skipped = 0

    stability_rows: list[tuple[float, float, float, float]] = []
    started = time.perf_counter()
    for position_index in range(args.positions):
        try:
            scenario = parse_scenario(scenario_from_samples(reg, prior, rng))
        except Exception:  # noqa: BLE001 - an unsampleable team is not what is measured
            skipped += 1
            continue

        beliefs = {}
        ok = True
        for (side_index, party_index), nature in scenario.hidden.items():
            mon = scenario.position.sides[side_index].pokemon[party_index]
            try:
                beliefs[(side_index, party_index)] = build_belief(
                    reg, prior, mon.species, nature
                )
            except Exception:  # noqa: BLE001
                ok = False
                break
        if not ok or not beliefs:
            skipped += 1
            continue

        from pokeuraou.setup import with_spreads

        modal = {
            key: b.spreads[int(np.argmax(b.weights))] for key, b in beliefs.items()
        }
        base = with_spreads(scenario, modal)

        active = {}
        for side_index, side in enumerate(base.sides):
            for slot, party_index in enumerate(side.active):
                if (side_index, party_index) in beliefs:
                    active[(side_index, slot)] = beliefs[(side_index, party_index)]

        per_slot_masses = []
        for key, belief in sorted(active.items()):
            entry = prior.species[
                base.sides[key[0]].pokemon[base.sides[key[0]].active[key[1]]].species
            ]
            raw_particles.append(entry.n_particles)
            conditioned.append(belief.size)
            reduction = reduce_for_position(reg, base, key, active)
            classes.append(reduction.size)
            factors.append(reduction.factor)
            top_mass.append(float(reduction.belief.weights.max()))
            per_slot_masses.append(np.sort(reduction.belief.weights)[::-1])
            live, _reasons = live_stats_for(reg, base, key)
            live_counter.update(live)
            live_sizes[len(live)] += 1

        if len(per_slot_masses) == 2:
            joint = np.outer(per_slot_masses[0], per_slot_masses[1]).reshape(-1)
            joint.sort()
            covered.append(float(joint[::-1][: args.classes].sum()))

        for side_index in (0, 1):
            result = narrow(reg, base, side_index, limit=24)
            candidates.append(result.considered)
            kept.append(len(result.kept))
            if result.uncovered:
                coverage_failures += 1

        if position_index < args.stability:
            solved = [
                analyse(
                    scenario,
                    objective=HP_SHARE,
                    cross=FAINTS,
                    limit=args.stability_limit,
                    belief_classes=k,
                )
                for k in (4, 8, 16)
            ]
            reference = solved[-1].equilibrium.row_strategy
            stability_rows.append(
                (
                    solved[0].covered_mass,
                    solved[-1].covered_mass,
                    float(np.abs(solved[0].equilibrium.row_strategy - reference).max()),
                    abs(solved[0].equilibrium.value - solved[-1].equilibrium.value),
                )
            )

    seconds = time.perf_counter() - started
    if not classes:
        raise SystemExit("nothing measured")

    def line(label: str, values: list[float], fmt: str = ".0f") -> str:
        return (
            f"  {label:34} 中央 {statistics.median(values):{fmt}}"
            f"  最小 {min(values):{fmt}}  最大 {max(values):{fmt}}"
        )

    print(
        f"{FORMAT_ID}: {args.positions - skipped} 局面 / {len(classes)} 体（"
        f"{seconds:.1f} 秒、skip {skipped}）\n"
    )
    print("パーティクル数")
    print(line("使用率データの全配分", [float(v) for v in raw_particles]))
    print(line("性格で条件付けたあと", [float(v) for v in conditioned]))
    print(line("無損失同値類まで縮約したあと", [float(v) for v in classes]))
    print(line("縮約の倍率", factors, ".2f"))
    print()
    print("質量の集中")
    print(line("最頻クラスの質量", top_mass, ".3f"))
    if covered:
        print(line(f"上位{args.classes}同時クラスの質量", covered, ".3f"))
    print()
    print("この局面で効く能力値（体ごと）")
    total = sum(live_sizes.values())
    for stat, count in live_counter.most_common():
        print(f"  {stat:4} {count / total * 100:5.1f}% の体で live")
    for size in sorted(live_sizes):
        print(f"  live {size} 個: {live_sizes[size] / total * 100:5.1f}%")
    print()
    if stability_rows:
        print("切り詰めの誤差（同値類 4 と 16 の比較、片側 "
              f"{args.stability_limit} 手）")
        print(
            line(
                "覆った質量（4クラス）",
                [row[0] for row in stability_rows], ".3f",
            )
        )
        print(
            line(
                "覆った質量（16クラス）",
                [row[1] for row in stability_rows], ".3f",
            )
        )
        print(
            line(
                "採用頻度の最大変化",
                [row[2] for row in stability_rows], ".3f",
            )
        )
        print(line("均衡値の変化", [row[3] for row in stability_rows], ".4f"))
        print()
    print("候補の絞り込み")
    print(line("合法手（片側）", [float(v) for v in candidates]))
    print(line("行列に載せた手", [float(v) for v in kept]))
    print(f"  網羅できなかった側: {coverage_failures} / {len(kept)}")


if __name__ == "__main__":
    main()
