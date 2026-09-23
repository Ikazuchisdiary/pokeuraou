"""Generates games from recorded late positions, because self-play rarely reaches them.

Measured over the 53,604-game pool: games average 9.4 turns, 5.4%% reach turn 15 and 0.7%
reach turn 20. Nothing is being truncated -- the generator discards no unfinished games --
the play simply ends. So the phase where a slow plan finally pays is one position in
twenty, and a value function prices patience at what it has seen.

The human repertoire for this roster is built on that phase. The reported win condition
against BIG6 is "keep Toxapex and Venusaur, face Basculegion at one and lock it out",
which is a turn-15 game; the solved book leads Toxapex in 2.5%% of its mass where the
humans lead it in 47%.

More self-play of the same shape will not fix it -- generations 2 through 8 all reach
turn 15 at about the same rate. This changes which positions get labelled rather than how
many: every recorded decision at or past a turn threshold becomes the start of a fresh
game, played to a real result with the current leaf. The label is as sound as any other
in the pool, because the game is still played to a win or a loss.

What it does *not* do is find positions self-play never visits. The starts come from the
same lineage, so this oversamples a thin tail rather than inventing a new one. That is the
intended trade: the tail is real and there is not enough of it.

**The games are OPEN** (IKA-123). A resumed start is a recorded position with no record of
which of the opponent's Pokemon each side had seen by then, so there is nothing to hand
`play_game` as `sheets` and both searches are shown the opponent's four -- teacher data from
the easier game, not the one that ships. It says so: the run stops unless `--open-bench` is
on the command line, and every game records `information: open`. Before IKA-123 it made
such games without a word.

    uv run --group learn python tools/resume_generate.py --open-bench \
        --dir data/selfplay-gen8 --from-turn 12 --games 2000 \
        --value data/models/value-all.pt --limit 48 --rank-leaf \
        --out data/selfplay-resume8/games-seed1.jsonl --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.benchflags import say_open_reference
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.position import Position
from pokeuraou.provenance import open_games, provenance, write_game
from pokeuraou.selfplay import play_game
from pokeuraou.teams import load_roster
from pokeuraou.value import BatchedValue, load_model


def collect(paths: list[Path], from_turn: int, to_turn: int, limit: int) -> list[dict]:
    """Every recorded move decision at or past `from_turn`, with its game's context.

    Move decisions only. A replacement or a self-switch node is mid-turn: resuming from
    one would hand `play_game` a position whose pending choice has been forgotten, and the
    game would continue from a state no rule produced.
    """
    out: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("outcome") is None:
                    continue
                for decision in record["decisions"]:
                    if decision["kind"] != "move":
                        continue
                    if not from_turn <= decision["turn"] <= to_turn:
                        continue
                    out.append(
                        {
                            "position": decision["position"],
                            "turn": decision["turn"],
                            "foeArchetype": record.get("foeArchetype", "?"),
                            "source_outcome": record["outcome"],
                        }
                    )
                    if limit and len(out) >= limit:
                        return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, nargs="+", default=[Path("data/selfplay-gen8")])
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--from-turn", type=int, default=12)
    ap.add_argument(
        "--to-turn",
        type=int,
        default=25,
        help="upper bound on the starting turn. Needed because --flat-turns spreads the "
        "budget over every band that exists, and the pool has bands out to turn 75: an "
        "unbounded run put 27%% of its games past turn 70 and only 22%% in the 15-25 range "
        "the human plans actually live in. Turn-70 doubles is a degenerate one-on-one "
        "stall, far enough outside the deployment distribution to risk the harm measured "
        "when generation 8's narrow data was added to the pool.",
    )
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--pool", type=int, default=0, help="cap on positions collected (0 = all)")
    ap.add_argument("--value", type=Path, default=Path("data/models/value-all.pt"))
    ap.add_argument("--objective", default="hp-share")
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--rank-leaf", action="store_true")
    ap.add_argument("--solve-sparsely", action="store_true")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument(
        "--flat-turns",
        dest="flat_turns",
        action="store_true",
        default=True,
        help="draw the turn uniformly first, then a position within it. On by default, "
        "because the recorded tail decays steeply -- 12,232 decisions at turn 12 against "
        "408 at turn 20 and 101 at turn 25 -- so drawing positions uniformly would spend "
        "a third of the budget on the turn that is already the least scarce. The rarest "
        "bands get resampled several times as a result, which is the intended trade: a "
        "repeated start still plays out differently, and the alternative is not sampling "
        "that phase at all.",
    )
    ap.add_argument("--no-flat-turns", dest="flat_turns", action="store_false")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument(
        "--open-bench",
        action="store_true",
        help="required: the resumed games show both searches the opponent's four, because "
        "a recorded position does not say which of them each side had seen. There is no "
        "--hide-bench here; this flag is the statement that open teacher data is meant.",
    )
    args = ap.parse_args()
    if not args.open_bench:
        raise SystemExit(
            "resume_generate.py can only play the open game -- a recorded start does not "
            "say which of the opponent's Pokemon each side had seen -- so its games are "
            "open teacher data, not the hidden-bench condition generation ships. Pass "
            "--open-bench to mean that. Before IKA-123 it was the silent default, so a "
            "recorded command without the flag replays with it."
        )
    say_open_reference()
    torch.set_num_threads(args.torch_threads)

    paths = sorted(p for d in args.dir for p in d.glob("games-seed*.jsonl"))
    if not paths:
        raise SystemExit(f"no recorded games under {args.dir}")
    starts = collect(paths, args.from_turn, args.to_turn, args.pool)
    if not starts:
        raise SystemExit(
            f"no move decisions between turns {args.from_turn} and {args.to_turn}"
        )
    print(
        f"{len(starts):,} starting positions at turn {args.from_turn}-{args.to_turn} "
        f"from {len(paths)} files",
        file=sys.stderr,
    )

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    encoder = Encoder(reg)
    net, _meta = load_model(args.value, encoder)
    device = torch.device(args.device)
    value = BatchedValue(net.to(device), encoder, device=device)
    objective = OBJECTIVES[args.objective]

    by_turn: dict[int, list[dict]] = {}
    for entry in starts:
        by_turn.setdefault(entry["turn"], []).append(entry)
    turns = sorted(by_turn)
    if args.flat_turns:
        print(
            "  turns "
            + ", ".join(f"{t}:{len(by_turn[t])}" for t in turns[:8])
            + (" ..." if len(turns) > 8 else "")
            + f"  ({len(turns)} bands, drawn evenly)",
            file=sys.stderr,
        )

    rng = np.random.default_rng(args.seed)
    games_file = open_games(args.out)
    finished = unfinished = 0
    started = time.perf_counter()
    reached: list[int] = []
    starts_used: list[int] = []
    for _index in range(args.games):
        if args.flat_turns:
            band = by_turn[turns[int(rng.integers(len(turns)))]]
            pick = band[int(rng.integers(len(band)))]
        else:
            pick = starts[int(rng.integers(len(starts)))]
        position = Position.from_json(pick["position"])
        record = play_game(
            reg,
            rng,
            [],
            [],
            pick["foeArchetype"],
            objective=objective,
            search_limit=args.limit,
            max_turns=args.max_turns,
            evaluate=value,
            rank_by_leaf=args.rank_leaf,
            solve_sparsely=args.solve_sparsely,
            start=position,
            open_information=True,
        )
        if record.outcome is None:
            unfinished += 1
            continue
        finished += 1
        starts_used.append(pick["turn"])
        if record.decisions:
            reached.append(max(d.turn for d in record.decisions))
        write_game(
            games_file,
            record,
            objective=f"value:{args.value.stem}",
            search_limit=args.limit,
            source=provenance(
                "resumed",
                seat="resumed from a recorded position",
                leaves=(args.value.stem, args.value.stem),
                limits=(args.limit, args.limit),
                rankings=tuple("leaf" if args.rank_leaf else "damage" for _ in range(2)),
                solvers=tuple("sparse" if args.solve_sparsely else "full" for _ in range(2)),
                information=("open", "open"),
                note=f"start turn {pick['turn']}, from-turn {args.from_turn}",
            ),
        )
        if finished % 50 == 0:
            print(
                f"  {finished}/{args.games} "
                f"{(time.perf_counter() - started) / max(finished, 1):.2f}s each",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(f"\n{finished} finished, {unfinished} unfinished, {elapsed / 60:.1f} min")
    if starts_used:
        arr = np.array(starts_used)
        print(f"  start turn: mean {arr.mean():.1f}, median {np.median(arr):.0f}, max {arr.max()}")
    if reached:
        arr = np.array(reached)
        print(f"  final turn: mean {arr.mean():.1f}, median {np.median(arr):.0f}, max {arr.max()}")


if __name__ == "__main__":
    main()
