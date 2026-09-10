"""Usage-weighted mechanics coverage report.

Answers "what should be implemented next" with a measurement instead of a guess: every
ability, item and move in the regulation is checked against what the resolver models, and
the gaps are ranked by how much of a team slot they actually occupy in the metagame.

    uv run python tools/coverage.py
    uv run python tools/coverage.py --top 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.effects import (  # noqa: E402
    all_modelled_abilities,
    all_modelled_items,
)
from pokeuraou.moveinfo import (  # noqa: E402
    ENDEAVOR_MOVES,
    FRACTIONAL_HP_MOVES,
    HANDLED_VARIABLE_BP,
    KNOWN_UNIMPLEMENTED_VARIABLE_BP,
    USER_HP_DAMAGE_MOVES,
)
from pokeuraou.priors import MetagamePrior, find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


def usage_weights(prior: MetagamePrior) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Share of a team slot each ability / item / move occupies."""
    abilities: dict[str, float] = {}
    items: dict[str, float] = {}
    moves: dict[str, float] = {}
    for sp in prior.species.values():
        for table, out in ((sp.abilities, abilities), (sp.items, items), (sp.moves, moves)):
            for key, share in table.items():
                out[key] = out.get(key, 0.0) + share * sp.usage
    return abilities, items, moves


def _section(
    title: str, gaps: dict[str, float], total: float, top: int, note: str = ""
) -> list[str]:
    covered = total - sum(gaps.values())
    pct = 100.0 if total <= 0 else covered / total * 100
    lines = [f"{title}: {pct:.2f}% of usage covered" + (f"  ({note})" if note else "")]
    if not gaps:
        lines.append("  nothing unmodelled")
        return lines
    for key, weight in sorted(gaps.items(), key=lambda kv: -kv[1])[:top]:
        lines.append(f"  {weight / max(total, 1e-9) * 100:6.3f}%  {key}")
    if len(gaps) > top:
        rest = sum(sorted(gaps.values(), reverse=True)[top:])
        lines.append(f"  {rest / max(total, 1e-9) * 100:6.3f}%  ... {len(gaps) - top} more")
    return lines


def report(reg: Regulation, prior: MetagamePrior, top: int) -> str:
    ability_use, item_use, move_use = usage_weights(prior)

    modelled_abilities = all_modelled_abilities()
    mega_stones = frozenset(reg.mega_map)
    modelled_items = all_modelled_items(mega_stones)

    # A move needs modelling only if Showdown gives it a callback the dump cannot carry.
    variable_bp_moves = {
        m.id
        for m in reg.moves.values()
        if "basePowerCallback" in m.custom_hooks or "damageCallback" in m.custom_hooks
    }
    handled_moves = (
        HANDLED_VARIABLE_BP | set(FRACTIONAL_HP_MOVES) | USER_HP_DAMAGE_MOVES | ENDEAVOR_MOVES
    )

    ability_gaps = {k: v for k, v in ability_use.items() if k not in modelled_abilities}
    item_gaps = {k: v for k, v in item_use.items() if k and k not in modelled_items}
    move_gaps = {
        k: v for k, v in move_use.items() if k in variable_bp_moves and k not in handled_moves
    }

    out: list[str] = [
        f"coverage against {prior.format_id} usage (cutoff {prior.cutoff}, "
        f"{prior.battles:,} battles), evaluated for {reg.meta.format_id}",
        "",
    ]
    out += _section("abilities", ability_gaps, sum(ability_use.values()), top)
    out.append("")
    out += _section("items", item_gaps, sum(item_use.values()), top)
    out.append("")
    out += _section(
        "moves with a base-power or damage callback",
        move_gaps,
        sum(move_use.get(m, 0.0) for m in variable_bp_moves),
        top,
        note="share of the callback-move usage only",
    )
    out.append("")
    declared = sorted(KNOWN_UNIMPLEMENTED_VARIABLE_BP & variable_bp_moves)
    out.append(
        f"declared unimplemented and skipped by the differential test: {len(declared)} moves"
    )
    out.append("  " + ", ".join(declared))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--format", dest="format_id", default=FORMAT_ID)
    args = ap.parse_args()
    reg = load_regulation(args.format_id)
    chaos = find_cached_chaos(args.format_id)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    print(report(reg, load_chaos(chaos, reg), args.top))


if __name__ == "__main__":
    main()
