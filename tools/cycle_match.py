"""Plays the mirror's reported three-way cycle instead of asking the value function.

``mirror_cycle.py`` asks the model, and the model says there is no cycle: it has the three
cited selections as a ladder, one edge of the ring pointing the wrong way. There is a good
reason to distrust that answer -- **self-play has never played a mirror.** Opponents are
drawn from the tournament field, our six are ours, so the mirror's 8,100 cells are
extrapolation from a function calibrated on something else entirely.

The scope is narrow on purpose: the ring is a claim about these three against *each other*,
in the mirror. It does not say they are strong selections, and the model preferring a
different selection over the whole 90 is not evidence either way.

So this settles it by playing. Each edge of the ring is played with the real search on both
sides, and the measured win rate is printed next to the cell the value function claimed.
Three outcomes, three different conclusions:

- **the cycle appears in play** -> the value function is wrong about the mirror, and the fix
  is to put mirror games in the training data rather than to argue with the players;
- **no cycle in play either** -> either our search does not reach the depth the cycle lives
  at, or the reported knowledge is about a different level of play. Both are worth knowing
  and neither is a value-function defect;
- **an edge lands inside its interval** -> not enough games. The interval is printed for
  exactly this reason.

Every edge is played in **both seats** with the same seed, because the mirror is where seat
bias shows up: with identical teams on both sides, ``win(i as side 0) + win(i as side 1)``
must come to 1.000, and a speed-tie bug in the resolver was once caught by precisely this
sum. The reported estimate averages the two seats, so any residual bias cancels instead of
being attributed to a selection.

    bash tools/cycle_match_parallel.sh
    uv run --group learn python tools/cycle_match.py --merge --text /c/tmp/cycle.txt

**Open game only: a reference** (IKA-123). This tool has no hidden-bench path, so its
search is shown the opponent's four -- not the game that ships. It says so on stderr
when it runs, and `tools/agent_drift.py` lists it under KNOWN_DRIFT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mirror_cycle import knowledge_dir, selection_index
from pool_matches import wilson

from pokeuraou.benchflags import say_open_reference
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.names import localiser
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.selfplay import play_game
from pokeuraou.teams import all_selections, load_roster
from pokeuraou.value import BatchedValue, load_model


def edges_from(knowledge: dict, roster: object, selections: list[tuple[int, ...]]) -> list[dict]:
    """The ring's ordered edges, as (label, selection) pairs to play against each other."""
    out: list[dict] = []
    for cycle in knowledge.get("cycles", []):
        ring = cycle["ring"]
        indices = [
            selection_index(roster, selections, node["leads"], node["bench"])
            for node in ring
        ]
        for k, node in enumerate(ring):
            nxt = (k + 1) % len(ring)
            out.append(
                {
                    "cycle": cycle["id"],
                    "a": node["label"],
                    "b": ring[nxt]["label"],
                    "ai": indices[k],
                    "bi": indices[nxt],
                }
            )
    return out


def merge(out: Path, text: Path | None) -> None:
    rows: list[dict] = []
    for part in sorted(out.parent.glob(f"{out.stem}.part*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in part.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not rows:
        raise SystemExit(f"no rows in {out.parent}/{out.stem}.part*.jsonl")
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    lines: list[str] = []
    by_edge: dict[str, list[dict]] = {}
    for row in rows:
        by_edge.setdefault(f"{row['a']} vs {row['b']}", []).append(row)

    unfinished = sum(1 for r in rows if r["outcome"] is None)
    lines.append(f"■ 報告された3すくみを実際に対戦させた（{len(rows)} 戦、打ち切り {unfinished}）")
    lines.append(
        f"  {'edge':>12} {'games':>6} {'実測(平均)':>11} {'95%':>16} "
        f"{'価値関数':>9} {'座席差':>8}"
    )
    verdicts: dict[str, float] = {}
    for edge, group in by_edge.items():
        # `seat` is which side the first selection sat on. A's win probability from seat 1
        # is one minus the recorded outcome, because the outcome is always side 0's.
        seat0 = [r["outcome"] for r in group if r["seat"] == 0 and r["outcome"] is not None]
        seat1 = [r["outcome"] for r in group if r["seat"] == 1 and r["outcome"] is not None]
        a_wins = seat0 + [1.0 - o for o in seat1]
        if not a_wins:
            continue
        wins = int(sum(a_wins))
        rate, low, high = wilson(wins, len(a_wins))
        claimed = group[0]["claimed"]
        bias = (
            (sum(seat0) / len(seat0) + sum(seat1) / len(seat1)) - 1.0
            if seat0 and seat1
            else float("nan")
        )
        verdicts[edge] = rate
        lines.append(
            f"  {edge:>12} {len(a_wins):>6} {rate * 100:10.1f}% "
            f"[{low * 100:5.1f}, {high * 100:5.1f}] {claimed * 100:8.1f}% "
            f"{bias:+8.3f}"
        )
    lines.append(
        "  座席差は win(side0)+win(side1)-1。ミラーなので 0 が要件で、"
        "ずれていれば解決器か探索の座席バイアス"
    )

    if verdicts:
        lines.append("")
        wins_all = all(rate > 0.5 for rate in verdicts.values())
        loses_all = all(rate < 0.5 for rate in verdicts.values())
        significant = {
            edge: rate
            for edge, rate in verdicts.items()
            if abs(rate - 0.5) > 1.96 * (0.25 / max(len(by_edge[edge]), 1)) ** 0.5
        }
        if wins_all or loses_all:
            direction = "報告どおりの向き" if wins_all else "報告と逆の向き"
            lines.append(f"  → 実際の対戦では3すくみが成立している（{direction}）")
        else:
            lines.append(
                "  → 実際の対戦でも3すくみになっていない。"
                "価値関数の誤りではなく、この探索深さでは巡回が現れないということ"
            )
        undecided = [edge for edge in verdicts if edge not in significant]
        if undecided:
            lines.append(
                f"  ただし {', '.join(undecided)} は区間が 50% を跨いでいるので"
                "まだ判定できない（ゲーム数を増やす）"
            )

    text_out = "\n".join(lines)
    print(text_out, file=sys.stderr)
    if text is not None:
        text.parent.mkdir(parents=True, exist_ok=True)
        text.write_text(text_out + "\n", encoding="utf-8")
        print(f"（読める形: {text}）", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--model", type=Path, default=Path("data/models/value-gen2.pt"))
    ap.add_argument("--knowledge", type=Path, default=None)
    ap.add_argument("--games", type=int, default=210, help="games per edge per seat")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data/matches/cycle-match.jsonl"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--text", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()
    say_open_reference()

    if args.merge:
        merge(args.out, args.text)
        return

    torch.set_num_threads(args.torch_threads)
    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    path = args.knowledge or (knowledge_dir() / f"{args.roster}-mirror.json")
    knowledge = json.loads(path.read_text(encoding="utf-8"))

    encoder = Encoder(reg)
    net, _meta = load_model(args.model, encoder)
    device = torch.device(args.device)
    evaluate = BatchedValue(net.to(device), encoder, device=device)
    loc = localiser(reg, "ja")
    selections = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
    edges = edges_from(knowledge, roster, selections)
    if not edges:
        raise SystemExit(f"{path} lists no cycles")

    # The value function's own answer for each edge, so the measured rate has something to
    # be compared against on the same line. One forward pass, not a solve.
    from pokeuraou.selfplay import position_from_sets

    for edge in edges:
        position = position_from_sets(
            reg,
            [roster.sets[i] for i in selections[edge["ai"]]],
            [roster.sets[j] for j in selections[edge["bi"]]],
        )
        edge["claimed"] = float(evaluate([position])[0])

    target = (
        args.out
        if args.shards == 1
        else args.out.parent / f"{args.out.stem}.part{args.shard}.jsonl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    mine = list(range(args.games))[args.shard :: args.shards]
    print(
        f"shard {args.shard}/{args.shards}: {len(edges)} 辺 x {len(mine)} ゲーム x 2 座席",
        file=sys.stderr,
    )

    def label(index: int) -> str:
        selection = selections[index]

        def name(i: int) -> str:
            species = roster.sets[i].species
            return loc.species(species) if loc else reg.species[species].name

        return (
            "+".join(name(i) for i in selection[:2])
            + "/"
            + "+".join(name(i) for i in selection[2:])
        )

    with target.open("w", encoding="utf-8") as handle:
        for done, game in enumerate(mine, start=1):
            for edge in edges:
                for seat in (0, 1):
                    first, second = (
                        (edge["ai"], edge["bi"]) if seat == 0 else (edge["bi"], edge["ai"])
                    )
                    # Same seed for both seats: the two games differ only in which side of
                    # the field each selection stands on.
                    rng = np.random.default_rng([args.seed, game, edge["ai"], edge["bi"]])
                    record = play_game(
                        reg,
                        rng,
                        [roster.sets[i] for i in selections[first]],
                        [roster.sets[j] for j in selections[second]],
                        f"mirror:{edge['a']}v{edge['b']}",
                        objective=OBJECTIVES["hp-share"],
                        search_limit=args.limit,
                        max_turns=args.max_turns,
                        evaluate=evaluate,
                        open_information=True,
                    )
                    handle.write(
                        json.dumps(
                            {
                                "cycle": edge["cycle"],
                                "a": edge["a"],
                                "b": edge["b"],
                                "seat": seat,
                                "game": game,
                                "outcome": record.outcome,
                                "turns": record.turns,
                                "claimed": edge["claimed"],
                                "aLabel": label(edge["ai"]),
                                "bLabel": label(edge["bi"]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
            if done % 5 == 0:
                print(f"  {done}/{len(mine)}", file=sys.stderr)


if __name__ == "__main__":
    main()
