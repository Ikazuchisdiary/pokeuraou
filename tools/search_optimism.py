"""Is the search's optimism the narrowing's bill?

`leaf_calibration` found the leaf unbiased at every turn and the SEARCH optimistic by
13.3 points at turn 1, decaying to 0.2 by turn 9. That combination is the interesting
part: searchValue is an equilibrium over leaf evaluations one ply ahead, so if the leaf
is unbiased the bias has to come from the solve.

`narrow` is the obvious suspect. It cuts BOTH sides to `limit` candidates ranked by our
own leaf, and a best reply of theirs that falls outside the menu is a reply the matrix
never prices -- which flatters us. It also fits the decay: turn 1 has the largest legal
set and the deepest cut, and by turn 9 there is little left to drop.

Turn and legal-action count fall together, so the decay alone proves nothing. This holds
the turn fixed and buckets by how many of the opponent's actions the menu had to drop. If
the optimism still grows with the cut inside a single turn, narrowing is paying for it;
if it is flat, the bias is somewhere else and the menu is exonerated.

No model is loaded: `searchValue` is in the log and the legal pool is combinatorial.

    uv run python tools/search_optimism.py --dir data/selfplay-gen11L --turn 1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

DROP_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("dropped 0", 0, 0),
    ("1-20", 1, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101+", 101, 10**9),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/selfplay-gen11L"))
    ap.add_argument("--turn", type=int, default=1, help="hold the turn fixed at this")
    ap.add_argument("--positions", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    args = ap.parse_args()

    reg = load_regulation(args.format)
    register_mega_stones(reg)
    rng = np.random.default_rng(args.seed)

    picked: list[tuple[dict, float, int]] = []
    seen = 0
    for path in sorted(args.dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                game = json.loads(line)
                if not game.get("targetIsRealOutcome") or game.get("outcome") is None:
                    continue
                limit = int(game.get("searchLimit") or 0)
                for decision in game["decisions"]:
                    if decision["kind"] != "move" or decision["turn"] != args.turn:
                        continue
                    if decision.get("searchValue") is None:
                        continue
                    seen += 1
                    item = (decision, float(game["outcome"]), limit)
                    if len(picked) < args.positions:
                        picked.append(item)
                    else:
                        j = int(rng.integers(0, seen))
                        if j < args.positions:
                            picked[j] = item
    print(f"  turn {args.turn}: {seen} decisions with a search value, {len(picked)} sampled")
    if not picked:
        raise SystemExit("nothing to score")

    rows: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    limits: set[int] = set()
    for decision, outcome, limit in picked:
        pos = Position.from_json(decision["position"])
        # Theirs, because the claim is about a reply the menu did not price. Ours being
        # cut costs us candidates and would make the search PESSIMISTIC if anything.
        theirs = len(side_actions(reg, pos, 1))
        dropped = max(0, theirs - limit) if limit else 0
        limits.add(limit)
        name = next(
            (n for n, lo, hi in DROP_BUCKETS if lo <= dropped <= hi), DROP_BUCKETS[-1][0]
        )
        rows[name].append((float(decision["searchValue"]), outcome, theirs))

    print(f"  search limits in this pool: {sorted(limits)}")
    print(
        f"\n  {'their actions cut':<18} {'n':>5}  {'search':>7} {'actual':>7}  "
        f"{'optimism':>9}  {'±':>5}  {'legal':>6}"
    )
    for name, _lo, _hi in DROP_BUCKETS:
        items = rows.get(name) or []
        if not items:
            continue
        gap = np.asarray([s - o for s, o, _t in items], dtype=np.float64)
        legal = np.mean([t for _s, _o, t in items])
        half = 1.96 * float(np.std(gap, ddof=1)) / np.sqrt(len(gap)) if len(gap) > 1 else 0.0
        print(
            f"  {name:<18} {len(items):>5}  "
            f"{np.mean([s for s, _o, _t in items]):>6.1%} "
            f"{np.mean([o for _s, o, _t in items]):>6.1%}  "
            f"{gap.mean():>+8.1%}  {half:>5.1%}  {legal:>6.0f}"
        )
    print(
        "\n  Turn is held fixed, so a trend down this column is the menu and not the\n"
        "  stage of the game. Flat means narrowing is not what makes the search\n"
        "  optimistic and the next suspect is the solve over a noisy matrix."
    )


if __name__ == "__main__":
    main()
