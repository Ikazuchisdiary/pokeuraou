"""The single-turn analysis tool.

Reads a scenario -- two open team sheets and the field -- and prints what a solver prints:
the equilibrium mixed strategy for both sides, how often each action is played, how much
EV each action gives up, and what is known about the opponent's spread. There is no "best
move and a score" anywhere in the output, because in a simultaneous-move game with hidden
information there generally is no best move: the answer is a distribution, and collapsing
it to one line would be a different, wrong answer.

Everything printed is either measured or derived from something measured, and the header
says which. The objective is named with its formula and with what it cannot see; the
belief section says how much probability mass the matrix actually covered; the narrowing
section says what it left out. A number with no defensible derivation is not printed at
all -- which is why the row marked "勝率" is absent until milestone 3 supplies a value
function that can honestly carry that label.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .actions import SideAction, TargetNames, target_names
from .belief import (
    Reduction,
    SpreadBelief,
    build_belief,
    faster_probability,
    reduce_for_position,
)
from .belief import battler_for as belief_battler
from .budget import Budget
from .damage import register_mega_stones
from .equilibrium import BayesianEquilibrium, solve_bayesian
from .hpdisplay import Band
from .names import Localiser, localiser
from .narrow import DEFAULT_LIMIT, Narrowed, narrow
from .observe import Observation, UpdateReport, parse_observations, update
from .payoff import OBJECTIVES, Objective
from .port import batched_payoffs
from .position import Position
from .priors import find_cached_chaos, load_chaos
from .setup import Scenario, load_scenario, with_spreads
from .view import battler

#: How many joint spread classes of the opposing pair the matrix is averaged over. The
#: classes are lossless, so this truncation is the one approximation in the pipeline and
#: the covered probability mass is printed with the result.
DEFAULT_BELIEF_CLASSES = 4


@dataclass(slots=True)
class JointClass:
    """One assignment of spreads to every hidden Pokemon, with its probability."""

    weight: float
    #: (side, party index) -> SP vector.
    assignment: dict[tuple[int, int], np.ndarray]
    #: (side, party index) -> the exact current HP this class assumes, chosen from the
    #: band the displayed percentage allows.
    hp: dict[tuple[int, int], int]
    label: str


@dataclass(slots=True)
class Analysis:
    scenario: Scenario
    objective: Objective
    cross: Objective
    budget_name: str
    row: Narrowed
    col: Narrowed
    #: The class-weighted average, kept for diagnostics only. It is *not* what is
    #: solved: see `matrices`.
    payoff: np.ndarray
    cross_payoff: np.ndarray
    #: One payoff matrix per belief class, which is what the Bayesian LP consumes.
    matrices: list[np.ndarray]
    cross_matrices: list[np.ndarray]
    equilibrium: BayesianEquilibrium
    cross_equilibrium: BayesianEquilibrium
    reductions: dict[tuple[int, int], Reduction]
    beliefs: dict[tuple[int, int], SpreadBelief]
    classes: list[JointClass]
    covered_mass: float
    bench_fixed: tuple[str, ...]
    unmodelled: tuple[str, ...] = ()
    cells: int = 0
    seconds: float = 0.0
    exact_cells: int = 0
    field_summary: dict[str, str] = field(default_factory=dict)
    #: The position with the modal spread filled in everywhere. Reporting needs a
    #: materialised position, because a hidden one has no stats to speak of.
    base: Position | None = None
    #: (side, active slot) -> the HP band the displayed percentage allows. Printed so the
    #: reader knows how wide the observation actually is.
    hp_bands: dict[tuple[int, int], Band] = field(default_factory=dict)
    #: The observations that were applied, and what each one bought.
    observations: tuple[Observation, ...] = ()
    update_report: UpdateReport | None = None


def _modal(belief: SpreadBelief) -> np.ndarray:
    return belief.spreads[int(np.argmax(belief.weights))]


def build_beliefs(scenario: Scenario) -> dict[tuple[int, int], SpreadBelief]:
    """A particle set for every Pokemon whose spread the scenario leaves hidden."""
    chaos = find_cached_chaos(scenario.reg.meta.format_id)
    if chaos is None:
        raise SystemExit(
            "no cached usage stats for this regulation; run tools/fetch_priors.py first. "
            "A belief has to come from data, and a flat prior over 32^6 spreads would be "
            "a worse lie than refusing."
        )
    prior = load_chaos(chaos, scenario.reg)
    if prior.format_id != scenario.reg.meta.format_id:
        # A newly launched regulation has no usage data of its own for about a month, so
        # the loader falls back to the previous one. That is a reasonable default and a
        # terrible thing to do silently: the belief would be labelled with one regulation
        # and built from another.
        print(
            f"注意: 使用率データは {prior.format_id} のものです"
            f"（局面は {scenario.reg.meta.format_id}）。"
            "新しいレギュレーションのデータが出るまでの代用です。",
            file=sys.stderr,
        )
    out: dict[tuple[int, int], SpreadBelief] = {}
    for (side_index, party_index), nature in scenario.hidden.items():
        mon = scenario.position.sides[side_index].pokemon[party_index]
        out[(side_index, party_index)] = build_belief(
            scenario.reg, prior, mon.species, nature
        )
    return out


def _heaviest_combinations(
    per_slot: list[list[tuple[float, tuple[int, int], np.ndarray, int, str]]],
    limit: int,
) -> list[tuple[tuple[float, tuple[int, int], np.ndarray, int, str], ...]]:
    """The heaviest `limit` combinations, without building the rest.

    This used to be `itertools.product` followed by a sort, and the product is the problem:
    two hidden Pokemon with about 1,150 spread-and-HP entries each is 1.36 million
    combinations built, sorted, and thrown away except for four. Measured at 4.6 seconds,
    which was 29% of a whole analysis.

    Each slot's entries are already sorted heaviest first and a combination's weight is the
    product of one entry per slot, so the heaviest combination is the head of every list
    and every other candidate is one step down one axis from a combination already taken.
    That makes it a k-way merge over a lattice: keep a frontier in a heap, and expand a
    point into its neighbours when it is popped.

    Ties are broken by position in `itertools.product` order -- the last axis varying
    fastest -- because that is what the stable sort over the materialised list did, and the
    answer must not change.
    """
    if not per_slot:
        return [()]
    if any(not entries for entries in per_slot):
        return []

    sizes = [len(entries) for entries in per_slot]
    strides = [1] * len(sizes)
    for axis in range(len(sizes) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * sizes[axis + 1]

    def weight_of(index: tuple[int, ...]) -> float:
        product = 1.0
        for axis, position in enumerate(index):
            product *= per_slot[axis][position][0]
        return product

    def ordinal(index: tuple[int, ...]) -> int:
        return sum(position * strides[axis] for axis, position in enumerate(index))

    start = (0,) * len(sizes)
    frontier: list[tuple[float, int, tuple[int, ...]]] = [
        (-weight_of(start), ordinal(start), start)
    ]
    seen = {start}
    out: list[tuple[tuple[float, tuple[int, int], np.ndarray, int, str], ...]] = []
    while frontier and len(out) < limit:
        _weight, _order, index = heapq.heappop(frontier)
        out.append(tuple(per_slot[axis][position] for axis, position in enumerate(index)))
        for axis, position in enumerate(index):
            if position + 1 >= sizes[axis]:
                continue
            neighbour = index[:axis] + (position + 1,) + index[axis + 1 :]
            if neighbour in seen:
                continue
            seen.add(neighbour)
            heapq.heappush(
                frontier, (-weight_of(neighbour), ordinal(neighbour), neighbour)
            )
    return out


def joint_classes(
    scenario: Scenario,
    reductions: dict[tuple[int, int], Reduction],
    beliefs: dict[tuple[int, int], SpreadBelief],
    *,
    limit: int,
) -> tuple[list[JointClass], float, tuple[str, ...]]:
    """The heaviest joint spread classes of the hidden *active* Pokemon.

    The two opposing Pokemon are treated as independent because the usage prior is built
    that way -- it gives a spread distribution per species with no joint structure -- and
    pretending otherwise would invent a correlation the data does not contain.

    Hidden Pokemon on the bench are pinned to their most likely spread. They can only
    enter this turn by being switched in, and the alternative would multiply the matrix by
    their class count for a contribution limited to one switch-in; the pinning is named in
    the report rather than absorbed.
    """
    bench: list[str] = []
    fixed: dict[tuple[int, int], np.ndarray] = {}
    for key, belief in beliefs.items():
        if key in {(s, scenario.position.sides[s].active[t]) for s, t in reductions}:
            continue
        fixed[key] = _modal(belief)
        mon = scenario.position.sides[key[0]].pokemon[key[1]]
        # The species id, not the English name: the display resolves it, so the same
        # analysis can be printed in either language.
        bench.append(mon.species)

    per_slot: list[list[tuple[float, tuple[int, int], np.ndarray, int, str]]] = []
    for (side_index, slot), reduction in sorted(reductions.items()):
        party_index = scenario.position.sides[side_index].active[slot]
        mon = scenario.position.sides[side_index].pokemon[party_index]
        name = scenario.reg.species[mon.species].name
        entries: list[tuple[float, tuple[int, int], np.ndarray, int, str]] = []
        for i in range(reduction.belief.size):
            spread = reduction.belief.spreads[i]
            maxhp = int(reduction.belief.stats[i, 0])
            allowed = scenario.band_for((side_index, party_index), maxhp)
            # Nothing distinguishes the values inside a band, so they are equally
            # weighted. That is an assumption, and the report names it.
            share = float(reduction.belief.weights[i]) / allowed.width
            sp_label = "/".join(str(int(v)) for v in spread)
            for hp in allowed.values:
                entries.append(
                    (
                        share,
                        (side_index, party_index),
                        spread,
                        int(hp),
                        f"{name} {sp_label} @{hp}",
                    )
                )
        entries.sort(key=lambda e: -e[0])
        per_slot.append(entries)

    kept: list[JointClass] = []
    for combination in _heaviest_combinations(per_slot, limit):
        weight = 1.0
        assignment = dict(fixed)
        hp_choice: dict[tuple[int, int], int] = {}
        labels = []
        for w, key, spread, hp, label in combination:
            weight *= w
            assignment[key] = spread
            hp_choice[key] = hp
            labels.append(label)
        kept.append(
            JointClass(
                weight=weight,
                assignment=assignment,
                hp=hp_choice,
                label=" + ".join(labels),
            )
        )

    covered = sum(c.weight for c in kept)
    # The total over *every* combination, without enumerating them: a combination's weight
    # is the product of one entry per slot, so the sum over the product is the product of
    # the sums. Exactly, and in two multiplications rather than 1.36 million.
    total = 1.0
    for entries in per_slot:
        total *= sum(entry[0] for entry in entries)
    total = total or 1.0
    for c in kept:
        c.weight /= covered or 1.0
    return kept, covered / total, tuple(sorted(set(bench)))


def _budget(name: str) -> Budget:
    if name == "matrix":
        return Budget.matrix()
    if name == "fast":
        return Budget.fast()
    if name == "exact":
        return Budget.exact()
    raise SystemExit(f"unknown budget {name!r}; use matrix, fast or exact")


def analyse(
    scenario: Scenario,
    *,
    objective: Objective,
    cross: Objective,
    limit: int = DEFAULT_LIMIT,
    belief_classes: int = DEFAULT_BELIEF_CLASSES,
    budget_name: str = "matrix",
) -> Analysis:
    reg = scenario.reg
    register_mega_stones(reg)
    beliefs = build_beliefs(scenario)

    # A concrete position is needed before anything can be computed, including the
    # reduction's own Speed thresholds, so start from the modal spread everywhere.
    modal = {key: _modal(belief) for key, belief in beliefs.items()}
    base = with_spreads(scenario, modal) if beliefs else scenario.position.copy()

    active_beliefs: dict[tuple[int, int], SpreadBelief] = {}
    for side_index, side in enumerate(base.sides):
        for slot, party_index in enumerate(side.active):
            if (side_index, party_index) in beliefs:
                active_beliefs[(side_index, slot)] = beliefs[(side_index, party_index)]

    # The observations come first: conditioning on evidence before reducing means the
    # reduction only has to group the particles that survived, and the equivalence classes
    # that end up in the matrix are the ones the evidence left standing.
    observations = tuple(parse_observations(reg, scenario.raw_observations))
    update_report: UpdateReport | None = None
    if observations:
        active_beliefs, update_report = update(
            reg, base, active_beliefs, list(observations)
        )
        for (side_index, slot), belief in active_beliefs.items():
            beliefs[(side_index, base.sides[side_index].active[slot])] = belief

    reductions = {
        key: reduce_for_position(reg, base, key, active_beliefs) for key in active_beliefs
    }
    reduced_active = {key: r.belief for key, r in reductions.items()}
    classes, covered, bench_fixed = joint_classes(
        scenario, reductions, beliefs, limit=belief_classes
    )

    # Belief-weighted Battlers for the damage score, so narrowing ranks against the whole
    # particle set rather than against one guess at the opponent's investment. Each
    # particle carries its own exact HP, taken from the band its max HP allows.
    score_battlers = {}
    score_weights = {}
    hp_bands: dict[tuple[int, int], Band] = {}
    for side_index, side in enumerate(base.sides):
        for slot, mon in enumerate(side.active_pokemon()):
            if mon is None or mon.fainted:
                continue
            key = (side_index, slot)
            belief = reduced_active.get(key)
            if belief is None:
                score_battlers[key] = battler(reg, mon)
            else:
                score_battlers[key] = belief_battler(reg, base, key, belief)
                score_weights[key] = belief.weights
                hp_bands[key] = scenario.band_for(
                    (side_index, mon.slot), int(np.max(belief.stats[:, 0]))
                )

    row = narrow(
        reg, base, 0, limit=limit, battlers=score_battlers, weights=score_weights
    )
    col = narrow(
        reg, base, 1, limit=limit, battlers=score_battlers, weights=score_weights
    )
    if not row.kept or not col.kept:
        raise SystemExit("no legal actions for one side; is the position a replacement turn?")

    budget = _budget(budget_name)
    positions = [
        (c.weight, with_spreads(scenario, c.assignment, c.hp)) for c in classes
    ] or [(1.0, base)]

    matrices: list[np.ndarray] = []
    cross_matrices: list[np.ndarray] = []
    unmodelled: set[str] = set()
    # A cell counts as exact only if it was exact for *every* spread class, because the
    # number the matrix carries is the average over them.
    all_exact = np.ones((len(row.kept), len(col.kept)), dtype=bool)
    started = time.perf_counter()
    for _weight, pos in positions:
        # One resolve, both objectives, every leaf of the node in one forward pass.
        # Measured on this path with a learned value function: 10.47 seconds against 5.63
        # for a 24x24 matrix over four spread classes. The output is the same to 1.2e-7,
        # which is float32 accumulating in a different order and four decimals short of
        # anything the tool prints.
        filled, notes, exact = batched_payoffs(
            reg,
            pos,
            row.actions,
            col.actions,
            [objective.batch, cross.batch],
            budget=budget,
        )
        matrices.append(filled[0])
        cross_matrices.append(filled[1])
        unmodelled.update(notes)
        all_exact &= exact
    seconds = time.perf_counter() - started
    exact_cells = int(all_exact.sum())

    weights = np.array([weight for weight, _pos in positions], dtype=np.float64)
    # The average is for the report's "the matrix ranges over" line only. Solving it
    # would answer a question nobody asked.
    payoff = sum(w * mat for w, mat in zip(weights, matrices, strict=True))
    cross_payoff = sum(w * mat for w, mat in zip(weights, cross_matrices, strict=True))

    return Analysis(
        scenario=scenario,
        objective=objective,
        cross=cross,
        budget_name=budget_name,
        row=row,
        col=col,
        payoff=np.asarray(payoff),
        cross_payoff=np.asarray(cross_payoff),
        matrices=matrices,
        cross_matrices=cross_matrices,
        equilibrium=solve_bayesian(matrices, weights),
        cross_equilibrium=solve_bayesian(cross_matrices, weights),
        reductions=reductions,
        beliefs=beliefs,
        classes=classes,
        covered_mass=covered,
        bench_fixed=bench_fixed,
        unmodelled=tuple(sorted(unmodelled)),
        cells=int(np.asarray(payoff).size),
        seconds=seconds,
        exact_cells=exact_cells,
        field_summary=_field_summary(base),
        base=base,
        hp_bands=hp_bands,
        observations=observations,
        update_report=update_report,
    )


def _field_summary(pos: Position) -> dict[str, str]:
    out: dict[str, str] = {}
    if pos.field.weather:
        out["天候"] = pos.field.weather
    if pos.field.terrain:
        out["フィールド"] = pos.field.terrain
    if pos.field.pseudo_weather:
        out["場"] = ", ".join(p.id for p in pos.field.pseudo_weather)
    for i, side in enumerate(pos.sides):
        if side.side_conditions:
            out[f"p{i + 1} 場"] = ", ".join(c.id for c in side.side_conditions)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _strategy_table(
    reg: object,
    kept: list[object],
    strategy: np.ndarray,
    ev: np.ndarray,
    ev_loss: np.ndarray,
    *,
    header: str,
    loc: Localiser | None = None,
    targets: TargetNames | None = None,
) -> str:
    lines = [header, f"  {'頻度':>7}  {'EV':>7}  {'EV損':>7}  行動"]
    order = np.argsort(-strategy)
    shown = 0
    for index in order:
        frequency = float(strategy[index])
        if frequency <= 1e-6 and shown >= 6:
            continue
        action: SideAction = kept[int(index)].action  # type: ignore[attr-defined]
        lines.append(
            f"  {frequency * 100:6.2f}%  {float(ev[index]):7.4f}  "
            f"{float(ev_loss[index]):7.4f}  {action.describe(reg, loc, targets)}"  # type: ignore[arg-type]
        )
        shown += 1
    return "\n".join(lines)


def _best_reply_sets(analysis: Analysis, tolerance: float = 1e-9) -> list[set[int]]:
    """Per belief class, the opponent's actions that give up nothing.

    A property of the game, unlike the particular mixed strategy the LP happened to
    return: where a class is indifferent, the solver picks a vertex arbitrarily, so
    comparing the returned strategies overstates how much the spread matters.
    """
    return [
        {int(j) for j in np.flatnonzero(loss <= tolerance)}
        for loss in analysis.equilibrium.col_ev_loss
    ]


def _opponent_table(
    reg: object,
    analysis: Analysis,
    loc: Localiser | None = None,
    targets: TargetNames | None = None,
) -> str:
    """What the opponent does, and whether their own spread changes it."""
    eq = analysis.equilibrium
    marginal = eq.col_marginal
    assert marginal is not None
    replies = _best_reply_sets(analysis)
    total = len(replies)

    lines = [
        "□ 相手の均衡戦略（配分クラスで平均した観測頻度）",
        f"  {'頻度':>7}  {'最適':>6}  行動",
    ]
    shown = 0
    for index in np.argsort(-marginal):
        frequency = float(marginal[index])
        count = sum(1 for reply in replies if int(index) in reply)
        if frequency <= 1e-6 and shown >= 6:
            continue
        action = analysis.col.kept[int(index)].action
        lines.append(
            f"  {frequency * 100:6.2f}%  {count:>3}/{total}  "
            f"{action.describe(reg, loc, targets)}"  # type: ignore[arg-type]
        )
        shown += 1

    union = set().union(*replies) if replies else set()
    common = set(replies[0]).intersection(*replies) if replies else set()
    lines.append(
        f"  「最適」は EV 損 0 になった配分クラスの数（全 {total} クラス）"
    )
    if union == common:
        lines.append(
            "  → どの配分でも最適反応は同じ。相手が自分の配分を知っていても指し手は変わりません。"
        )
    else:
        conditional = sorted(union - common)[:3]
        names = ", ".join(
            analysis.col.kept[j].action.describe(reg, loc, targets)  # type: ignore[arg-type]
            for j in conditional
        )
        lines.append(
            f"  → 配分によって最適反応が変わります（例: {names}）。"
            "相手の指し手そのものが配分の情報になります。"
        )
    return "\n".join(lines)


def render(analysis: Analysis, loc: Localiser | None = None) -> str:
    a = analysis
    reg = a.scenario.reg
    eq = a.equilibrium
    out: list[str] = []
    # A target index means different Pokemon depending on who is choosing, so each side
    # gets its own resolver. Built from the materialised position, which is the one with
    # the hidden spreads filled in and therefore the one whose slots are populated.
    reference = a.base or a.scenario.position
    ours_targets = target_names(reference, 0)
    theirs_targets = target_names(reference, 1)

    out.append(f"pokeuraou 1ターン検討  [{reg.meta.format_id}]  {a.scenario.source}")
    if a.field_summary:
        out.append("  場: " + " / ".join(f"{k}={v}" for k, v in a.field_summary.items()))
    out.append("")
    out.append("■ 評価軸（この数値が何なのか）")
    out.append(f"  {a.objective.name}: {a.objective.formula}")
    out.append(f"  見えていないもの: {a.objective.blind_to}")
    if a.objective.name.startswith("value:"):
        out.append(
            "  これは勝率です（学習した価値関数の単位）。ただし校正はその学習に使った"
            "探索の強さと相手プールに条件付きで、絶対的な勝率ではありません。"
        )
    else:
        out.append(
            "  これは勝率ではありません。勝率で見るには --value に学習した価値関数を渡してください。"
        )
    out.append("")

    if a.update_report is not None:
        out.append(a.update_report.render(reg))
        out.append(
            "  尤度はこの局面で評価しています。観測時と天候・ランクが違う場合は、"
            "その局面の JSON に観測を付けて別に流してください。"
        )
        out.append("")
        out.append("■ 相手の配分（観測で更新済みの事後分布）")
    else:
        out.append("■ 相手の配分（事前分布。観測が与えられていません）")
    reference = a.base or a.scenario.position
    for (side_index, slot), reduction in sorted(a.reductions.items()):
        party_index = reference.sides[side_index].active[slot]
        mon = reference.sides[side_index].pokemon[party_index]
        name = loc.species(mon.species) if loc else reg.species[mon.species].name
        nature = loc.nature(mon.nature) if loc else mon.nature
        out.append(f"  {name}（{nature}）")
        out.append("    " + reduction.render(reg, loc).replace("\n", "\n    "))
        observed = a.hp_bands.get((side_index, slot))
        if observed is not None:
            out.append(
                f"    HP 表示は {observed.percent}% "
                f"→ 実 HP は最大 {observed.width} 通り"
                f"（最大HP {observed.maxhp} なら {observed.low}..{observed.high}）"
            )
            if observed.width > 1:
                out.append("    帯の中は等確率としています（区別する情報がないため）")
        for our_slot in range(len(reference.sides[0].active)):
            try:
                faster, tie = faster_probability(
                    reg,
                    reference,
                    (side_index, slot),
                    reduction.belief,
                    (0, our_slot),
                )
            except Exception:  # noqa: BLE001 - a missing slot is not worth failing over
                continue
            ours = reference.sides[0].pokemon[reference.sides[0].active[our_slot]]
            ours_name = (
                loc.species(ours.species) if loc else reg.species[ours.species].name
            )
            out.append(
                f"    {ours_name} より速い: {faster * 100:.1f}%"
                f"（同速 {tie * 100:.1f}%）"
            )
        top = sorted(
            range(reduction.belief.size), key=lambda i: -reduction.belief.weights[i]
        )[:3]
        for i in top:
            spread = "/".join(str(int(v)) for v in reduction.belief.spreads[i])
            out.append(
                f"    {reduction.belief.weights[i] * 100:5.1f}%  SP {spread}"
                f"  {reduction.belief.stats[i].tolist()}"
            )
    if a.bench_fixed:
        bench = [
            loc.species(sid) if loc else reg.species[sid].name for sid in a.bench_fixed
        ]
        out.append(
            f"  控えは最頻配分で固定: {', '.join(bench)}"
            "（このターン出てくるのは交代のときだけ）"
        )
    out.append(
        f"  行列が覆った同値類の質量: {a.covered_mass * 100:.1f}%"
        f"（上位 {len(a.classes)} 組の同時クラス、2体は独立と仮定）"
    )
    out.append("")

    out.append("■ 候補の絞り込み")
    out.append("  自陣: " + a.row.render(reg, loc, ours_targets).replace("\n", "\n  "))
    out.append("  相手: " + a.col.render(reg, loc, theirs_targets).replace("\n", "\n  "))
    out.append("")

    out.append("■ 均衡（厳密解、HiGHS）")
    out.append(
        f"  均衡値 {eq.value:.4f}（{a.objective.name} 単位）"
        f"  双対ギャップ {eq.duality_gap:.2e}"
    )
    out.append(
        "  相手は自分の配分を知っているので、配分クラスごとに別の戦略を取れる"
        "ベイズ型ゼロ和ゲームとして解いています"
        "（行列を平均してから解くと相手を不利に見積もることになります）"
    )
    out.append(
        f"  行列 {a.payoff.shape[0]}x{a.payoff.shape[1]} = {a.cells} セル、"
        f"{len(a.classes) or 1} 同値類 × {a.budget_name} 予算で {a.seconds:.1f} 秒"
        f"（うち完全列挙 {a.exact_cells} セル）"
    )
    out.append("")
    out.append(
        _strategy_table(
            reg, a.row.kept, eq.row_strategy, eq.row_ev, eq.row_ev_loss,
            header="□ 自陣の均衡戦略（採用頻度）",
            loc=loc,
            targets=ours_targets,
        )
    )
    out.append("")
    if eq.col_marginal is not None:
        out.append(_opponent_table(reg, a, loc, theirs_targets))
    out.append("")

    out.append("■ 評価軸を変えても同じか（頑健性）")
    spread = float(a.cross_payoff.max() - a.cross_payoff.min())
    if spread < 1e-12:
        out.append(
            f"  {a.cross.name} はこの行列の全セルで同じ値 "
            f"({float(a.cross_payoff.reshape(-1)[0]):.4f}) になりました。"
        )
        out.append(
            "  1ターンでは誰も倒れないので、この評価軸はこの局面を区別できません。"
            "頑健性の判定材料にはなりません。"
        )
        return "\n".join(out + _unmodelled_lines(a))
    cross_eq = a.cross_equilibrium
    row_support = set(eq.row_support().tolist())
    cross_support = set(cross_eq.row_support().tolist())
    shared = row_support & cross_support
    out.append(
        f"  {a.cross.name} で解き直した均衡値 {cross_eq.value:.4f}、"
        f"支持集合の一致 {len(shared)}/{len(row_support | cross_support)}"
    )
    if row_support == cross_support:
        out.append("  → 採用する行動の集合は評価軸に依存しません。")
    else:
        only_main = row_support - cross_support
        only_cross = cross_support - row_support
        if only_main:
            names = ", ".join(
                a.row.kept[i].action.describe(reg, loc, ours_targets)
                for i in sorted(only_main)
            )
            out.append(f"  → {a.objective.name} だけが採用: {names}")
        if only_cross:
            names = ", ".join(
                a.row.kept[i].action.describe(reg, loc, ours_targets)
                for i in sorted(only_cross)
            )
            out.append(f"  → {a.cross.name} だけが採用: {names}")
        out.append(
            "  → 評価軸で結論が変わる局面です。どちらが正しいかは M3 の価値関数まで決まりません。"
        )

    return "\n".join(out + _unmodelled_lines(a))


def _unmodelled_lines(a: Analysis) -> list[str]:
    """Effects the resolver flagged while filling the matrix.

    These are the documented gaps, not silent errors: the resolver said so before
    producing an answer, so the reader knows which numbers carry a caveat.
    """
    if not a.unmodelled:
        return []
    out = ["", "■ 解決器が申告した未実装効果（この行列に混ざっている）"]
    out.extend(f"  - {item}" for item in a.unmodelled)
    return out


def render_stability(
    base: Analysis, doubled: Analysis, loc: Localiser | None = None
) -> str:
    ours_targets = target_names(base.base or base.scenario.position, 0)
    """Whether doubling the belief truncation would have changed the answer."""
    shift = np.abs(base.equilibrium.row_strategy - doubled.equilibrium.row_strategy)
    lines = [
        "■ 切り詰めの妥当性（同値類を倍にして解き直し）",
        f"  同値類 {len(base.classes)} → {len(doubled.classes)}、"
        f"覆った質量 {base.covered_mass * 100:.1f}% → "
        f"{doubled.covered_mass * 100:.1f}%",
        f"  均衡値 {base.equilibrium.value:.4f} → "
        f"{doubled.equilibrium.value:.4f}"
        f"（差 {abs(base.equilibrium.value - doubled.equilibrium.value):.4f}）",
    ]
    if shift.size:
        worst = int(np.argmax(shift))
        lines.append(
            f"  採用頻度の最大変化 {shift[worst] * 100:.1f}pt: "
            f"{base.row.kept[worst].action.describe(base.scenario.reg, loc, ours_targets)}"
        )
        if float(shift.max()) > 0.05:
            lines.append(
                "  → 頻度が動いています。"
                "--belief-classes を上げるか、"
                "この頻度は幅つきで読んでください。"
            )
        else:
            lines.append(
                "  → この局面では切り詰めの"
                "影響は誤差の範囲です。"
            )
    return "\n".join(lines)


def to_json(analysis: Analysis) -> dict[str, object]:
    """The same content as :func:`render`, for a caller that wants to plot it."""
    a = analysis
    reg = a.scenario.reg
    return {
        "regulation": reg.meta.format_id,
        "objective": {
            "name": a.objective.name,
            "formula": a.objective.formula,
            "blindTo": a.objective.blind_to,
            "isWinProbability": False,
        },
        "equilibrium": {
            "value": a.equilibrium.value,
            "dualityGap": a.equilibrium.duality_gap,
            "form": "bayesian: the opponent observes their own spread class",
            "classes": len(a.matrices),
        },
        "own": [
            {
                "action": c.action.describe(reg),
                "choice": c.action.to_choice(),
                "frequency": float(f),
                "ev": float(e),
                "evLoss": float(loss),
                "narrowScore": c.score,
            }
            for c, f, e, loss in zip(
                a.row.kept,
                a.equilibrium.row_strategy,
                a.equilibrium.row_ev,
                a.equilibrium.row_ev_loss,
                strict=True,
            )
        ],
        "opponent": [
            {
                "action": c.action.describe(reg),
                "frequency": float(f),
                "perClassFrequency": [float(y[index]) for y in a.equilibrium.col_strategies],
            }
            for index, (c, f) in enumerate(
                zip(
                    a.col.kept,
                    a.equilibrium.col_marginal
                    if a.equilibrium.col_marginal is not None
                    else np.zeros(len(a.col.kept)),
                    strict=True,
                )
            )
        ],
        "belief": {
            f"p{side + 1}a{slot}": {
                "particles": r.original_size,
                "classes": r.size,
                "liveStats": list(r.live_stats),
                "speedThresholds": list(r.speed_thresholds),
                "provenance": r.belief.provenance,
                "meanSp": r.belief.mean_sp(),
                "isPosterior": bool(a.observations),
                "hpDisplay": (
                    None
                    if (side, slot) not in a.hp_bands
                    else {
                        "percent": a.hp_bands[(side, slot)].percent,
                        "candidates": a.hp_bands[(side, slot)].width,
                        "uniformWithinBand": True,
                    }
                ),
            }
            for (side, slot), r in sorted(a.reductions.items())
        },
        "coverage": {
            "beliefMass": a.covered_mass,
            "candidatesOwn": {"considered": a.row.considered, "kept": len(a.row.kept)},
            "candidatesOpponent": {"considered": a.col.considered, "kept": len(a.col.kept)},
            "uncoveredOwn": list(a.row.uncovered),
            "benchPinned": list(a.bench_fixed),
        },
        "opponentBestReplies": [sorted(reply) for reply in _best_reply_sets(a)],
        "crossCheck": {
            "objective": a.cross.name,
            "value": a.cross_equilibrium.value,
            "sameSupport": set(a.equilibrium.row_support().tolist())
            == set(a.cross_equilibrium.row_support().tolist()),
        },
        "observations": [obs.describe() for obs in a.observations],
        "update": (
            None
            if a.update_report is None
            else {
                "applied": [
                    {
                        "observation": label,
                        "slot": list(key),
                        "candidatesBefore": before,
                        "candidatesAfter": after,
                        "bitsGained": bits,
                    }
                    for label, key, before, after, bits in a.update_report.entries
                ],
                "uninformative": list(a.update_report.uninformative),
                "skipped": list(a.update_report.skipped),
            }
        ),
        "unmodelled": list(a.unmodelled),
        "cost": {"cells": a.cells, "seconds": a.seconds, "budget": a.budget_name},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pokeuraou",
        description="Champions ダブルの1ターンを均衡混合戦略として解く",
    )
    ap.add_argument("scenario", help="局面 JSON（src/pokeuraou/setup.py が読み方の定義）")
    ap.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"片側の候補数上限（既定 {DEFAULT_LIMIT}）",
    )
    ap.add_argument(
        "--belief-classes", type=int, default=DEFAULT_BELIEF_CLASSES,
        help=f"相手2体の同時配分クラス数（既定 {DEFAULT_BELIEF_CLASSES}）",
    )
    ap.add_argument(
        "--value",
        type=Path,
        default=None,
        help="学習した価値関数（data/models/*.pt）を評価軸にする。セルが勝率になるので、"
        "均衡値も EV 損もそのまま勝率のポイントとして読める。--objective より優先。"
        "指定しない場合は hp-share のままで、それは勝率ではありません",
    )
    ap.add_argument(
        "--device", default=None, help="cuda か cpu。既定は cuda があればそれ"
    )
    ap.add_argument(
        "--objective", default="hp-share", choices=sorted(OBJECTIVES),
        help="評価軸（既定 hp-share）",
    )
    ap.add_argument(
        "--cross-check", default="faints", choices=sorted(OBJECTIVES),
        help="頑健性チェックに使う別の評価軸（既定 faints）",
    )
    ap.add_argument(
        "--budget", default="matrix", choices=("matrix", "fast", "exact"),
        help="1セルあたりの乱数列挙の細かさ（既定 matrix）",
    )
    ap.add_argument(
        "--stability", action="store_true",
        help="同値類を倍にして解き直し、切り詰めの影響を測る",
    )
    ap.add_argument("--json", action="store_true", help="機械可読な出力")
    ap.add_argument(
        "--port-threads", type=int, default=None,
        help="port の中で 1 ノードのセルを解くスレッド数（IKA-32）。答えはスレッド数によらず同じで、"
        "速さだけが変わる。既定は環境変数 POKEURAOU_PORT_THREADS、無ければ 1",
    )
    ap.add_argument(
        "--lang",
        default="ja",
        help="表示言語。ja（既定）または en。訳の無い項目は英語名にフォールバックし、"
        "tools/names_report.py が不足を数える。--json の出力は常に英語（機械向け）",
    )
    args = ap.parse_args(argv)
    if args.port_threads is not None:
        from . import rustnode

        rustnode.set_port_threads(args.port_threads)

    scenario = load_scenario(args.scenario)
    objective = OBJECTIVES[args.objective]
    if args.value is not None:
        # Imported here so a run without --value never loads torch: the resolver, the
        # differential tests and the whole M1 path stay installable without a CUDA wheel.
        import torch

        from .encode import Encoder
        from .value import BatchedValue, load_model

        if not args.value.exists():
            raise SystemExit(f"{args.value} が無い。tools/train_value.py で学習してください")
        register_mega_stones(scenario.reg)
        encoder = Encoder(scenario.reg)
        net, meta = load_model(args.value, encoder)
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # One position at a time: `analyse` folds each cell's branches through the
        # objective protocol, so batching would mean restructuring that loop. The forward
        # pass is a millisecond and the resolve around it is the cost -- measured, not
        # assumed: `matrix` budget spends 11.5s on a 24x24x4 matrix either way.
        objective = BatchedValue(net.to(device), encoder, device=device).objective(
            f"value:{args.value.stem}"
        )
        print(
            f"評価軸 = {args.value.name} on {device}"
            f"（{meta.get('games', '?')} ゲームで学習、検証 AUC "
            f"{meta.get('val_auc', float('nan')):.4f}）",
            file=sys.stderr,
        )
    analysis = analyse(
        scenario,
        objective=objective,
        cross=OBJECTIVES[args.cross_check],
        limit=args.limit,
        belief_classes=args.belief_classes,
        budget_name=args.budget,
    )
    doubled = None
    if args.stability:
        doubled = analyse(
            scenario,
            objective=objective,
            cross=OBJECTIVES[args.cross_check],
            limit=args.limit,
            belief_classes=args.belief_classes * 2,
            budget_name=args.budget,
        )

    loc = localiser(scenario.reg, args.lang)
    if args.json:
        # The JSON output stays in English ids and names: it is a machine interface, and a
        # consumer keyed on "Heat Wave" should not start receiving ねっぷう because a
        # display flag changed.
        payload = to_json(analysis)
        if doubled is not None:
            payload["stability"] = {
                "classes": [len(analysis.classes), len(doubled.classes)],
                "coveredMass": [analysis.covered_mass, doubled.covered_mass],
                "value": [analysis.equilibrium.value, doubled.equilibrium.value],
                "maxFrequencyShift": float(
                    np.abs(
                        analysis.equilibrium.row_strategy
                        - doubled.equilibrium.row_strategy
                    ).max()
                ),
            }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(render(analysis, loc))
        if doubled is not None:
            print()
            print(render_stability(analysis, doubled, loc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
