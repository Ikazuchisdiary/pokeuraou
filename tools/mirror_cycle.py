"""Does the value function reproduce the mirror's known three-way cycle?

The mirror is the one selection game with an answer that does not come from the model. Both
sides hold the same six with the same spreads, so ``M[i][j] = 1 - M[j][i]`` by the value
function's antisymmetry, the game is symmetric and its value is forced to exactly 0.500.
Anything else is a wiring bug. That makes it the right place to check a claim from outside
the model -- and ``configs/knowledge/*.json`` records one: three selections that players
know beat each other in a ring.

Two things follow, and both are checkable:

- **a genuine three-cycle rules out a pure equilibrium.** A pure strategy is a claim that
  one selection is unbeaten; a ring says every selection has an answer. The mirror solve
  printing a pure recommendation and the cycle being real cannot both be true.
- **the ring's direction is measurable.** The report says which way the model has it,
  rather than assuming the direction the note was written in -- "①←②" does not fix the
  arrow, and reading it wrongly would turn a confirmation into a contradiction.

    uv run --group learn python tools/mirror_cycle.py --text /c/tmp/mirror.txt

Only the three cited selections are examined closely, but the whole 90x90 matrix is solved
anyway: the cyclic share of the full matrix says whether a ring is a local curiosity or the
shape of the entire game, and the equilibrium over all 90 is what the tool would actually
recommend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.equilibrium import solve
from pokeuraou.names import localiser
from pokeuraou.regulation import to_id
from pokeuraou.selection import SpreadClass, cyclic_share, solve_selection
from pokeuraou.teams import load_roster
from pokeuraou.value import BatchedValue, load_model


def knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "knowledge"


def selection_index(
    roster: object, selections: list[tuple[int, ...]], leads: list[str], bench: list[str]
) -> int:
    """The index in the ordered 90 of a selection written as species names.

    Within a pair the order is not modelled -- both leads face both foes in doubles -- so
    the pairs are sorted before the lookup, exactly as :func:`all_selections` builds them.
    """
    where = {to_id(entry.species): i for i, entry in enumerate(roster.sets)}
    missing = [n for n in (*leads, *bench) if to_id(n) not in where]
    if missing:
        raise SystemExit(f"{missing} are not in the roster: {sorted(where)}")
    front = tuple(sorted(where[to_id(n)] for n in leads))
    back = tuple(sorted(where[to_id(n)] for n in bench))
    target = front + back
    if target not in selections:
        raise SystemExit(f"{target} is not one of the {len(selections)} selections")
    return selections.index(target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--model", type=Path, default=Path("data/models/value-gen2.pt"))
    ap.add_argument("--knowledge", type=Path, default=None)
    ap.add_argument("--text", type=Path, default=None, help="write the report as UTF-8")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    path = args.knowledge or (knowledge_dir() / f"{args.roster}-mirror.json")
    knowledge = json.loads(path.read_text(encoding="utf-8"))
    if knowledge.get("roster") != args.roster:
        raise SystemExit(f"{path} is about roster {knowledge.get('roster')!r}")

    encoder = Encoder(reg)
    net, meta = load_model(args.model, encoder)
    device = torch.device(args.device)
    value = BatchedValue(net.to(device), encoder, device=device)
    loc = localiser(reg, "ja")

    def name(species_id: str) -> str:
        return loc.species(species_id) if loc else reg.species[species_id].name

    # The mirror: our six against itself, spreads included. One spread class, because
    # there is nothing hidden -- we are both sides.
    classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]
    analysis = solve_selection(reg, roster.sets, classes, value)
    selections = list(analysis.ours)
    matrix = np.asarray(analysis.matrices[0], dtype=np.float64)

    out: list[str] = []
    out.append(f"■ ミラーの均衡（{args.model.name}, {len(selections)}x{len(selections)}）")
    out.append(
        f"  均衡値 {analysis.value * 100:.4f}%（ミラーなので 50.0000% が要件、"
        f"誤差 {abs(analysis.value - 0.5):.2e}）"
    )
    out.append(
        f"  反対称性 |V(x)+V(鏡像x)-1| = {analysis.antisymmetry_error:.2e}、"
        f"双対ギャップ {analysis.equilibrium.duality_gap:.2e}"
    )
    support = analysis.equilibrium.row_support()
    out.append(f"  均衡の支持 {len(support)} 通り / {len(selections)}")
    for index in support[np.argsort(-analysis.equilibrium.row_strategy[support])][:8]:
        out.append(
            f"    {analysis.equilibrium.row_strategy[index] * 100:5.1f}%  "
            f"#{int(index):>2} {analysis.label(0, selections[index], name)}"
        )
    t_norm, c_norm, share = cyclic_share(matrix)
    out.append(
        f"  行列の形: 推移的 |T|={t_norm:.3f}、巡回 |C|={c_norm:.3f} "
        f"→ 巡回の割合 {share * 100:.1f}%（じゃんけん=100%、はしご=0%）"
    )

    for cycle in knowledge.get("cycles", []):
        ring = cycle["ring"]
        indices = [
            selection_index(roster, selections, node["leads"], node["bench"])
            for node in ring
        ]
        out.append("")
        out.append(f"■ 報告された{len(ring)}すくみ「{cycle['id']}」の検証")
        if cycle.get("note"):
            out.append(f"  {cycle['note']}")
        for node, index in zip(ring, indices, strict=True):
            out.append(
                f"  {node['label']} #{index:>2} "
                f"{'+'.join(name(n) for n in node['leads'])} / "
                f"{'+'.join(name(n) for n in node['bench'])}"
                f"   均衡頻度 {analysis.equilibrium.row_strategy[index] * 100:5.1f}%、"
                f"EV損 {analysis.equilibrium.row_ev_loss[index]:.4f}"
            )
        out.append("")
        out.append("  総当たり（行が先手の勝率。ミラーなので対角は 50.0%）")
        header = "        " + "".join(f"{node['label']:>9}" for node in ring)
        out.append(header)
        for node, i in zip(ring, indices, strict=True):
            row = "".join(f"{matrix[i][j] * 100:8.1f}%" for j in indices)
            out.append(f"    {node['label']:>4}{row}")

        # Which way round the ring goes, measured rather than assumed. A ring exists if
        # one of the two orientations wins every edge.
        labels = [node["label"] for node in ring]
        forward = [
            matrix[indices[k]][indices[(k + 1) % len(ring)]] for k in range(len(ring))
        ]
        out.append("")
        for k, edge in enumerate(forward):
            a, b = labels[k], labels[(k + 1) % len(ring)]
            verdict = "勝ち" if edge > 0.5 else "負け" if edge < 0.5 else "互角"
            out.append(f"    {a} vs {b}: {edge * 100:5.1f}% → {a} の{verdict}")
        if all(edge > 0.5 for edge in forward):
            out.append(
                f"  → {'→'.join(labels)}→{labels[0]} の向きで3すくみを再現している"
            )
        elif all(edge < 0.5 for edge in forward):
            out.append(
                f"  → {'←'.join(labels)}←{labels[0]} の向き（報告と逆順）で"
                "3すくみを再現している"
            )
        else:
            wins = [labels[k] for k, e in enumerate(forward) if e > 0.5]
            out.append(
                f"  → 3すくみになっていない。環の一部だけが成立（{wins} が勝ち側）。"
                "価値関数はこの巡回構造を持っていない"
            )
        # The ring on its own: if it is a real cycle, the restricted game's equilibrium is
        # mixed over all three, and its value is 0.5 like every symmetric game.
        sub = matrix[np.ix_(indices, indices)]
        restricted = solve(sub)
        out.append(
            "  この3通りだけに制限したゲームの均衡: "
            + "、".join(
                f"{node['label']} {restricted.row_strategy[k] * 100:.1f}%"
                for k, node in enumerate(ring)
            )
            + f"（値 {restricted.value * 100:.1f}%）"
        )
        _, sub_c, sub_share = cyclic_share(sub)
        out.append(
            f"  3x3 の巡回の割合 {sub_share * 100:.1f}%"
            "（3すくみなら 100% に近い。0% ならただの強弱）"
        )
        if len(support) == 1:
            out.append(
                "  注: 90 通り全体の均衡は純戦略。報告された3すくみが実在するなら、"
                "その純戦略には答えがあるはず — つまり両方は成り立たない。"
                "どちらが誤りかを決めるのは実際の対戦（tools/book_check.py の形）"
            )

    # The three cited selections are a sample of the 90; the pure recommendation is only
    # credible if *nothing* beats it, so the best replies to it are worth naming.
    if len(support) == 1:
        best = int(support[0])
        column = matrix[:, best]
        out.append("")
        out.append(
            f"■ 均衡が指す #{best} {analysis.label(0, selections[best], name)} への最善反応"
        )
        for index in np.argsort(-column)[:5]:
            out.append(
                f"    {column[index] * 100:5.1f}%  #{int(index):>2} "
                f"{analysis.label(0, selections[index], name)}"
            )
        out.append(
            "  行が 50% を超える選出が一つも無ければ、その純戦略は「無敗」の主張として"
            "整合している（推定誤差の範囲内での話）"
        )
        out.append(
            f"  最大 {column.max() * 100:.1f}%"
            + (
                "（50% 超えが存在するので、純戦略は矛盾している）"
                if column.max() > 0.5
                else "（50% 超えは無い）"
            )
        )

    # A pure equilibrium in a symmetric game means one selection weakly beats all 90. How
    # close the runners-up are decides whether that survives the model's own error.
    out.append("")
    out.append("■ 参考: 上位の選出と、そのEV損")
    order = np.argsort(analysis.equilibrium.row_ev_loss)
    for index in order[:6]:
        out.append(
            f"    EV損 {analysis.equilibrium.row_ev_loss[index]:.4f}  "
            f"#{int(index):>2} {analysis.label(0, selections[index], name)}"
        )
    out.append(
        f"  検証 AUC {meta.get('val_auc', float('nan')):.4f} のモデルなので、"
        "0.01 未満の差は推定誤差の内側として読むべき"
    )
    text = "\n".join(out)
    print(text, file=sys.stderr)
    if args.text is not None:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(text + "\n", encoding="utf-8")
        print(f"（読める形: {args.text}）", file=sys.stderr)


if __name__ == "__main__":
    main()
