"""Solving the 6->4 selection, which is a game and not a preprocessing step.

Both team sheets are open and both players choose four simultaneously, so selection is a
simultaneous-move zero-sum game with 90 actions per side -- C(6,2) lead pairs x C(4,2)
bench pairs, ordered because the first two start on the field. Its cells are win
probabilities from the resulting turn-1 position, which is exactly what the learned value
function produces. Nothing else here is new machinery: the LP is the same one the turn
solver uses.

Two things make it worth doing rather than approximating.

**The stakes are large and measured.** Over 13,395 self-play games the win rate ranged from
43.4% to 87.3% across our fifteen sets of four, and the two members left behind moved it by
23 points. Selection is not a detail on top of the turn analysis; it is a bigger lever than
anything inside a turn.

**Uncertainty has the same shape as everywhere else.** The sheet reveals the opponent's
species, ability, item, nature and moves, and hides the SP spread -- so the opponent knows
which spread they brought and we do not. That is the Bayesian game
:func:`~pokeuraou.equilibrium.solve_bayesian` already solves: one matrix per spread class,
the opponent free to select differently per class, us with one strategy. Averaging the
matrices first would answer a different question and overstate our value.

And one honest caveat, printed with the result: the value function is calibrated on the
distribution it was trained on, which is *uniform* selection. Solving for the equilibrium
selection moves play off that distribution, so the numbers here are the best available and
not the final word -- the fix is another generation of self-play drawn from this
equilibrium, which is the loop this module exists to start.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .equilibrium import BayesianEquilibrium, solve_bayesian
from .position import Position
from .priors import SampledSet
from .regulation import Regulation
from .selfplay import position_from_sets
from .teams import all_selections


@dataclass(frozen=True, slots=True)
class SpreadClass:
    """One assignment of spreads to the opponent's six, with its probability."""

    weight: float
    sets: tuple[SampledSet, ...]
    label: str


@dataclass(slots=True)
class SelectionAnalysis:
    """The solved selection game, with everything needed to read it."""

    reg: Regulation
    our_six: tuple[SampledSet, ...]
    their_six: tuple[SampledSet, ...]
    #: Ordered selections, leads first, as indices into each side's six.
    ours: tuple[tuple[int, ...], ...]
    theirs: tuple[tuple[int, ...], ...]
    #: One (len(ours), len(theirs)) matrix per spread class; cells are win probabilities.
    matrices: tuple[np.ndarray, ...]
    classes: tuple[SpreadClass, ...]
    equilibrium: BayesianEquilibrium
    positions_evaluated: int = 0
    seconds: float = 0.0
    #: max |V(x) + V(mirror x) - 1| over a sample, which must be ~0 for the matrix to be a
    #: zero-sum game at all. Reported rather than trusted.
    antisymmetry_error: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def value(self) -> float:
        return float(self.equilibrium.value)

    def our_frequencies(self) -> list[tuple[tuple[int, ...], float, float]]:
        """(selection, adoption frequency, EV loss), most-played first."""
        eq = self.equilibrium
        order = np.argsort(-eq.row_strategy)
        return [
            (self.ours[i], float(eq.row_strategy[i]), float(eq.row_ev_loss[i]))
            for i in order
        ]

    def their_frequencies(self) -> list[tuple[tuple[int, ...], float]]:
        marginal = self.equilibrium.col_marginal
        if marginal is None:
            return []
        order = np.argsort(-marginal)
        return [(self.theirs[j], float(marginal[j])) for j in order]

    def label(self, side: int, selection: tuple[int, ...], namer: Any = None) -> str:  # noqa: ANN401
        """"lead1+lead2 / back1+back2" for one selection."""
        six = self.our_six if side == 0 else self.their_six

        def name(index: int) -> str:
            species_id = six[index].species
            if namer is not None:
                return namer(species_id)
            found = self.reg.species.get(species_id)
            return found.name if found else species_id

        leads = "+".join(name(i) for i in selection[:2])
        back = "+".join(name(i) for i in selection[2:])
        return f"{leads} / {back}"


def solve_selection(
    reg: Regulation,
    our_six: Sequence[SampledSet],
    classes: Sequence[SpreadClass],
    evaluate: Callable[[list[Position]], np.ndarray],
    *,
    pick: int | None = None,
) -> SelectionAnalysis:
    """Builds the 90x90 matrix per spread class and solves it.

    ``evaluate`` scores a batch of positions and returns side 0's win probability, which is
    the only thing this needs from the value function -- so a different one, or a
    hand-checked stub in a test, drops straight in.

    No turn is resolved. A cell is the value of the turn-1 position itself, because the
    value function already answers "who wins from here" and re-resolving a turn would be
    asking a strictly worse question at 90x90x256 the cost.
    """
    import time

    if not classes:
        raise ValueError("no spread classes to solve over")
    size = pick or reg.meta.picked_team_size
    selections = tuple(all_selections(reg.meta.team_size, size))
    started = time.perf_counter()

    matrices: list[np.ndarray] = []
    evaluated = 0
    for spread_class in classes:
        positions = [
            position_from_sets(
                reg,
                [our_six[i] for i in ours],
                [spread_class.sets[j] for j in theirs],
            )
            for ours in selections
            for theirs in selections
        ]
        values = evaluate(positions)
        evaluated += len(positions)
        matrices.append(values.reshape(len(selections), len(selections)))

    # The matrix is only a zero-sum game if the value function is antisymmetric. It is by
    # construction, so this is a check on the wiring rather than on the model: a mirrored
    # position must come back at exactly one minus the original.
    sample = [
        position_from_sets(
            reg,
            [our_six[i] for i in selections[0]],
            [classes[0].sets[j] for j in selections[k]],
        )
        for k in range(min(8, len(selections)))
    ]
    mirrored = []
    for pos in sample:
        flipped = pos.copy()
        flipped.sides = [flipped.sides[1], flipped.sides[0]]
        flipped.sides[0].id, flipped.sides[1].id = (
            flipped.sides[1].id,
            flipped.sides[0].id,
        )
        mirrored.append(flipped)
    forward = evaluate(sample)
    backward = evaluate(mirrored)
    evaluated += len(sample) + len(mirrored)
    antisymmetry = float(np.abs(forward + backward - 1.0).max()) if len(sample) else 0.0

    weights = np.array([c.weight for c in classes], dtype=np.float64)
    equilibrium = solve_bayesian(matrices, weights)

    return SelectionAnalysis(
        reg=reg,
        our_six=tuple(our_six),
        their_six=tuple(classes[0].sets),
        ours=selections,
        theirs=selections,
        matrices=tuple(matrices),
        classes=tuple(classes),
        equilibrium=equilibrium,
        positions_evaluated=evaluated,
        seconds=time.perf_counter() - started,
        antisymmetry_error=antisymmetry,
        notes=(
            "価値関数は一様選出のデータで校正されているので、均衡選出はその分布の外にある。"
            "この均衡から次世代の自己対戦を生成して再学習するのが正しい閉じ方。",
        ),
    )


def cyclic_share(matrix: np.ndarray) -> tuple[float, float, float]:
    """Splits a symmetric game's matrix into transitive and cyclic parts.

    Only meaningful when both sides hold the same actions -- the mirror -- because the
    decomposition is of the skew-symmetric part of ``A - 0.5``:

        A - 0.5 = T + C,  T[i][j] = t[i] - t[j],  t = row means

    ``T`` is a ladder: every action has a strength and the stronger one wins. ``C`` is the
    remainder, and rock-paper-scissors is pure ``C``. Returns ``(|T|, |C|, cyclic share)``.

    A pure equilibrium is impossible when the cyclic share dominates, so this is the number
    that says whether a pure recommendation is credible. Note the direction of error: noise
    in the cells adds to ``C``, so a small cyclic share is an upper bound rather than
    something noise could have hidden.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.shape[0] != a.shape[1]:
        raise ValueError("the decomposition needs a square, symmetric-game matrix")
    skew = a - 0.5
    strength = skew.mean(axis=1)
    transitive = strength[:, None] - strength[None, :]
    cyclic = skew - transitive
    t_norm = float(np.linalg.norm(transitive))
    c_norm = float(np.linalg.norm(cyclic))
    total = t_norm**2 + c_norm**2
    return t_norm, c_norm, (c_norm**2 / total if total > 0 else 0.0)


def render(analysis: SelectionAnalysis, namer: Any = None, top: int = 8) -> str:  # noqa: ANN401
    """The solved selection, in the shape a solver prints: frequencies, not a pick."""
    eq = analysis.equilibrium
    out: list[str] = []
    out.append(
        f"pokeuraou 選出検討  [{analysis.reg.meta.format_id}]  "
        f"{len(analysis.ours)}x{len(analysis.theirs)} × {len(analysis.classes)} 配分クラス"
    )
    out.append("")
    out.append("■ 均衡値")
    out.append(
        f"  自陣の勝率 {analysis.value * 100:.1f}%"
        f"（学習した価値関数の単位。これは勝率です）"
    )
    out.append(f"  双対ギャップ {eq.duality_gap:.2e}")
    out.append(
        f"  反対称性の検査 |V(x)+V(鏡像x)-1| = {analysis.antisymmetry_error:.2e}"
        "（ゼロサムゲームとして解ける前提）"
    )
    out.append(
        f"  {analysis.positions_evaluated:,} 局面を評価、{analysis.seconds:.1f} 秒"
    )
    out.append("")

    out.append("■ 自陣の選出（均衡混合戦略）")
    out.append(f"  {'頻度':>7}  {'EV損':>7}  先発 / 控え")
    # Always show the supported selections, then pad to `top` with the best of the rest --
    # a reader needs to see what was rejected and by how much, not only what was chosen.
    for shown, (selection, frequency, loss) in enumerate(analysis.our_frequencies()):
        if shown >= top:
            break
        out.append(
            f"  {frequency * 100:6.2f}%  {loss:7.4f}  "
            f"{analysis.label(0, selection, namer)}"
        )
    support = len(eq.row_support())
    out.append(
        f"  採用される選出は {support} 通り / {len(analysis.ours)} 通り"
        f"（EV損 0 が最適反応であることの意味）"
    )
    if support == 1:
        out.append(
            "  → 純戦略です。つまり「混ぜろ」ではなく「これが支配的」という主張。"
            "じゃんけん構造があるなら純戦略の均衡は存在しないので、下の巡回成分と併せて読む"
        )
    if len(analysis.ours) == len(analysis.theirs) and analysis.our_six == analysis.their_six:
        # Same six on both sides: the matrix is a symmetric game and decomposes.
        t_norm, c_norm, share = cyclic_share(analysis.matrices[0])
        out.append(
            f"  行列の形: 推移的 |T|={t_norm:.3f}, 巡回 |C|={c_norm:.3f} "
            f"→ 巡回の割合 {share * 100:.1f}%"
            "（じゃんけん=100%、はしご=0%）"
        )
        out.append(
            "  セルの推定誤差は巡回成分を*増やす*側に働くので、この値は上限。"
            "小さいことをノイズでは説明できない"
        )
    out.append("")

    theirs = analysis.their_frequencies()
    if theirs:
        out.append("■ 相手の選出（配分クラスで平均した観測頻度）")
        out.append(f"  {'頻度':>7}  先発 / 控え")
        for selection, frequency in theirs[:top]:
            if frequency <= 1e-6:
                break
            out.append(
                f"  {frequency * 100:6.2f}%  {analysis.label(1, selection, namer)}"
            )
        out.append("")

    worst = int(np.argmax(eq.row_ev_loss))
    out.append("■ 読み方")
    out.append(
        f"  最も損な選出は {analysis.label(0, analysis.ours[worst], namer)}"
        f"（EV損 {float(eq.row_ev_loss[worst]):.4f} = 勝率 "
        f"{float(eq.row_ev_loss[worst]) * 100:.1f} ポイント）"
    )
    for note in analysis.notes:
        out.append(f"  注: {note}")
    return "\n".join(out)


__all__ = [
    "SelectionAnalysis",
    "SpreadClass",
    "cyclic_share",
    "render",
    "solve_selection",
]
