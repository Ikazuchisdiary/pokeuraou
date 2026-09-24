"""What the port refuses on the nodes a search already met, and why.

IKA-29 counted refused cells by reason from inside generation -- `timing.count("refused:
...")` in `batched_payoffs`, printed by `tools/profile_stages.py` -- which takes a
generation run to answer. A change to the port's gate asks the same question of a
different binary, and the nodes it would be asked about are already on disk: every
recorded move decision carries the root position and both menus the search filled its
matrix from. So this replays them through the port's node and counts what comes back
refused, by the same reason strings. No game is played and Python fills nothing, which is
what makes it cheap enough to run once per binary:

    uv run python tools/refusal_replay.py --games-dir data/selfplay-gen11L
    POKEURAOU_RUST_NODE_BIN=before.exe uv run python tools/refusal_replay.py ...

The menus are the recorded ones, matched back to legal actions by their choice string, so
a cell here is a cell that search actually asked for. A decision whose menu no longer
matches -- a choice the current action generator does not produce -- is skipped and
counted rather than rebuilt some other way, because a rebuilt menu is a different node.

What this does not see: the foe's own matrix when it was built from a different menu (only
its column menu is recorded), and a hidden bench's sampled spreads (the recorded position
is the true one). Neither changes which items and abilities a turn can involve.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402


def candidates(args: argparse.Namespace) -> Iterator[dict]:
    """Every decision the sample may draw from, one game in memory at a time.

    Move decisions spread over the files and over games within them: at most `--per-game`
    from one game, so a few hundred decisions come from about a hundred games rather than
    from the first ten, and at most ceil(`--decisions` / files) from one file.
    """
    files = [p for d in args.games_dir for p in sorted(Path(d).glob("*.jsonl"))]
    if not files:
        raise SystemExit(f"no *.jsonl under {', '.join(map(str, args.games_dir))}")
    per_file = -(-args.decisions // len(files))
    for path in files:
        here = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if here >= per_file:
                    break
                if args.holding and not any(item in line for item in args.holding):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                from_game = 0
                for decision in record.get("decisions", ()):
                    if decision.get("kind") != "move":
                        continue
                    if int(decision.get("turn", 0)) < args.min_turn:
                        continue
                    yield decision
                    here += 1
                    from_game += 1
                    if from_game >= args.per_game or here >= per_file:
                        break
                del record


def sample(args: argparse.Namespace) -> Iterator[dict]:
    """`--decisions` of the candidates, streamed: never more than one game in memory.

    The draw is the one this tool always made -- `Random(seed).shuffle` over the candidate
    list, first `--decisions` kept -- but a shuffle's permutation depends only on the
    list's length, so it is taken over indices: one pass counts the candidates, the
    permutation picks which indices survive, and a second pass yields those, in file order.
    Until IKA-154 the candidates themselves were held and shuffled, which on a whole pool
    (`--decisions 1000000` over data/ika73/w12) reached 13 GB. The set is the same; only
    the order differs, and nothing downstream depends on order.
    """
    total = sum(1 for _ in candidates(args))
    if total <= args.decisions:
        yield from candidates(args)
        return
    order = list(range(total))
    random.Random(args.seed).shuffle(order)
    keep = bytearray(total)
    for index in order[: args.decisions]:
        keep[index] = 1
    del order
    # A file still being written can have grown between the passes; what it grew by was
    # not counted, so it is not drawn.
    for index, decision in zip(range(total), candidates(args), strict=False):
        if keep[index]:
            yield decision


def menu(reg, pos: Position, side: int, choices: list[str]) -> list | None:  # noqa: ANN001
    """The recorded menu as action objects, or None if a choice is no longer legal here."""
    legal = {action.to_choice(): action for action in side_actions(reg, pos, side)}
    out = []
    for choice in choices:
        action = legal.get(choice)
        if action is None:
            return None
        out.append(action)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-dir", action="append", type=Path, required=True)
    ap.add_argument("--decisions", type=int, default=300)
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--min-turn", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--holding",
        type=lambda s: frozenset(i for i in s.split(",") if i),
        default=frozenset(),
        help="comma-separated ids; read only games whose record names one of them",
    )
    args = ap.parse_args()

    build = rustnode.require_current_binary()
    print(f"binary {build['sha256']} built {build['built']}  ({build['path']})")
    decisions = sample(args)

    regs: dict = {}
    nodes: dict = {}
    counted = cells = refused_cells = skipped = 0
    reasons: Counter = Counter()
    nodes_refusing: Counter = Counter()
    budget = Budget.matrix()
    for decision in decisions:
        raw = decision["position"]
        fmt = raw.get("format")
        if fmt not in regs:
            reg = load_regulation(fmt)
            register_mega_stones(reg)
            regs[fmt] = reg
            nodes[fmt] = rustnode.RustNode(reg)
        reg, node = regs[fmt], nodes[fmt]
        pos = Position.from_json(raw)
        ours = menu(reg, pos, 0, decision.get("ownActions") or [])
        theirs = menu(reg, pos, 1, decision.get("foeActions") or [])
        if not ours or not theirs:
            skipped += 1
            continue
        filled = node.fill(pos, ours, theirs, ["hp-share"], budget)
        counted += 1
        cells += len(ours) * len(theirs)
        refused_cells += len(filled.refused)
        why_here = Counter(why for _i, _j, why in filled.refused)
        reasons.update(why_here)
        nodes_refusing.update(why_here.keys())
    for node in nodes.values():
        node.close()

    print(
        f"{counted} decisions from {', '.join(map(str, args.games_dir))}, {cells:,} cells"
        + (f"; {skipped} skipped, their recorded menu no longer legal" if skipped else "")
    )
    share = 100.0 * refused_cells / max(cells, 1)
    print(f"  refused {refused_cells:,} cells ({share:.2f}%), {len(reasons)} distinct reasons")
    for why, times in reasons.most_common(12):
        print(
            f"   {100.0 * times / max(refused_cells, 1):5.1f}%  {times:>8,}  "
            f"in {nodes_refusing[why]:>4} nodes  {why}"
        )


if __name__ == "__main__":
    main()
