"""Is the equilibrium's heavier selection actually the better one?

The book's job is not to print a win probability, it is to say which four to bring. On six
opponents measured with the corrected harness, the selection the equilibrium weighted most
heavily was the WEAKER one on the board in four of five -- and on place 34 a selection it
gave zero mass to won 40% against the heavy one's 32%.

The rating cannot see this. `generation_match` draws both arms' selections from the same
book, so an ordering error both arms share cancels; book-against-uniform was +18.9 points
because discarding the bad 87 of 90 is a different and much easier question than ordering
the top three. So this measures the ordering directly, on its own.

**Which arm the second one is decides what can be falsified, and `--second-by mass` --
the original and still the default -- cannot falsify the ordering.** In a zero-sum game
every action in the support of an equilibrium strategy earns exactly the game value
against the opponent's equilibrium, by complementary slackness. Mass within the support
is set by the OPPONENT's indifference conditions and is not a ranking of our payoffs. So
picking the two heaviest, with `--min-second 0.05` guaranteeing both are in the support,
asks a question whose answer the theory already fixes at zero -- and the motivating
example above was a selection the book gave ZERO mass, which that filter excludes.

What `--second-by mass` does measure is worth having and is a different thing: the gap
between the solved matrix and the board. A correct matrix makes the difference zero and
the signs a coin flip, so a systematic departure is the leaf mispricing selections. There
is even a mechanism that would make it negative -- a selection gets heavy mass when the
LP leans on it to keep the opponent indifferent, and the LP leans hardest where the matrix
overrates it -- but that is a hypothesis this panel can suggest and not confirm.

`--second-by evloss` is the falsifiable one. It plays the heaviest against a selection the
book REJECTED, chosen so its `our_ev_loss` is near a target, and the equilibrium's claim
is then a number: the rejected arm should lose by about that much. The board can contradict
it in size as well as in sign, which is what turns a sign test into a calibration curve.

For each opponent it emits the two `--force-selection` arms `selection_check` already
supports: our four is fixed, theirs is drawn from their equilibrium, and both arms see the
same opponent in the same game index because each game seeds from `[seed, index]`. The
difference is then paired per game and the only thing that differs is which four we
brought.

Printing the commands rather than running them, because each opponent is about twenty
minutes of eight-way sharded play and the caller should decide how many to buy.

    uv run python tools/ordering_check.py --opponents 20
    uv run python tools/ordering_check.py --opponents 20 --games 60 > run.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.selection_book import SelectionBook  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--book",
        type=Path,
        default=Path("data/selection/rizabanadohido-value-gen11L.jsonl.gz"),
    )
    ap.add_argument("--model", type=Path, default=Path("data/models/value-gen11L.pt"))
    ap.add_argument("--opponents", type=int, default=20)
    ap.add_argument("--games", type=int, default=60, help="games per arm per opponent")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument(
        "--min-second",
        type=float,
        default=0.05,
        help="--second-by mass only: skip an opponent whose second selection carries less "
        "mass than this, because with a pure equilibrium there is no ordering to test.",
    )
    ap.add_argument(
        "--second-by",
        choices=("mass", "evloss"),
        default="mass",
        help="how to choose the arm to play against the heaviest one. See the note in "
        "this module's docstring about what each one can falsify.",
    )
    ap.add_argument(
        "--evloss-target",
        type=float,
        default=0.05,
        help="--second-by evloss: aim for a rejected selection the book says loses about "
        "this much win probability, so the board has a NUMBER to disagree with.",
    )
    ap.add_argument(
        "--rejected-mass",
        type=float,
        default=1e-6,
        help="--second-by evloss: mass at or below which a selection counts as rejected.",
    )
    args = ap.parse_args()

    book = SelectionBook.read(args.book)
    roster = load_roster("rizabanadohido")
    reg = roster.reg
    species = [reg.species[s.species].name.lower().replace(" ", "") for s in roster.sets]

    entries = sorted(book.entries.values(), key=lambda e: e.place)
    rng = np.random.default_rng(args.seed)
    picked = rng.permutation(len(entries))

    chosen: list[tuple[int, str, str, float, float, float]] = []
    for index in picked:
        entry = entries[int(index)]
        x = np.asarray(entry.our_strategy, dtype=np.float64)
        loss = np.asarray(entry.our_ev_loss, dtype=np.float64)
        order = np.argsort(-x)
        top = int(order[0])
        if args.second_by == "mass":
            if float(x[order[1]]) < args.min_second:
                continue
            other = int(order[1])
        else:
            # A rejected selection with a predicted loss near the target. Not the nearest
            # to the boundary and not the worst on the board: the nearest is where the
            # equilibrium's claim is weakest, and the worst is a selection so bad that
            # beating it says nothing. A target in the middle is where the book stakes a
            # number big enough to see through 60 games of noise.
            rejected = np.flatnonzero(x <= args.rejected_mass)
            if rejected.size == 0:
                continue
            other = int(rejected[int(np.argmin(np.abs(loss[rejected] - args.evloss_target)))])
            if not loss[other] > 0.0:
                continue
        first, second = entry.selections[top], entry.selections[other]
        chosen.append(
            (
                entry.place,
                ",".join(species[i] for i in first),
                ",".join(species[i] for i in second),
                float(x[top]),
                float(x[other]),
                float(loss[other] - loss[top]),
            )
        )
        if len(chosen) >= args.opponents:
            break

    if args.second_by == "mass":
        print(f"# {len(chosen)} opponents with a second selection carrying >= "
              f"{args.min_second:.0%}, drawn with seed {args.seed}")
        print("# BOTH ARMS ARE IN THE SUPPORT, so the equilibrium predicts NO difference")
        print("# between them and this panel measures the matrix's calibration, not the")
        print("# ordering. See the docstring.")
    else:
        print(f"# {len(chosen)} opponents, second arm rejected by the book and predicted "
              f"to lose about {args.evloss_target:.1%}, drawn with seed {args.seed}")
        print("# The rejected arm is OUTSIDE the support, so the book stakes a number the")
        print("# board can contradict.")
    print("# each pair is paired per game: same opponent, same spread class, same index")
    print("set -uo pipefail")
    # Every path below is relative, so the generated script needs the repository root as
    # its working directory and nothing more. It used to print `cd` with this machine's
    # checkout baked in, which is a script that runs nowhere else and, in a worktree,
    # runs in the wrong tree without saying so.
    print("# run this from the repository root")
    # The named arms are labelled 'a+b / c+d'. The obvious selector is '/', and it does
    # not survive the shell: MSYS rewrites a lone slash as a Windows path, so the worker
    # was handed --only-arm 'C:/Program Files/Git/' and refused every opponent before
    # playing a single game. Sixteen opponents of nothing, about five hours, caught by
    # running one of them for two games first.
    #
    # '+' selects the same two arms -- it is in every forced label and in none of the
    # three standard ones -- and is not a path character.
    print(f"DIR=data/matches/ordering{'' if args.second_by == 'mass' else '-ev'}")
    print('mkdir -p "$DIR/logs"')
    for place, first, second, p1, p2, predicted in chosen:
        print(f"\n# place {place}: {p1:.1%} vs {p2:.1%}, book predicts {predicted:+.1%}")
        print(f'out="$DIR/place{place}.jsonl"')
        print('[ -f "$out" ] || {')
        print("  pids=()")
        print(f"  for i in $(seq 0 {args.shards - 1}); do")
        print("    uv run --group learn python -u tools/selection_check.py \\")
        print(f"      --place {place} --model {args.model.as_posix()} \\")
        # Eight classes, not the four an earlier panel used, because `selection_check`
        # does not read the book -- it re-solves the selection game itself -- and the book
        # whose ordering is on trial was solved with eight. The two arms are fixed by the
        # book either way and both face the same opponent, so four would still be a valid
        # paired comparison; it would just be a comparison against a different opponent
        # model than the one the book assumed. One extra solve per opponent, about a
        # quarter of an hour over the whole panel.
        print(f"      --games {args.games} --limit 16 --classes 8 "
              "--hide-bench --rank-by-leaf \\")
        print(f'      --force-selection "{first}" --force-selection "{second}" \\')
        print('      --only-arm "+" \\')
        print(f'      --shard "$i" --shards {args.shards} --out "$out" \\')
        print(f'      > "$DIR/logs/{place}-$i.log" 2>&1 &')
        print("    pids+=($!)")
        print("  done")
        print('  for pid in "${pids[@]}"; do wait "$pid" || true; done')
        print(f'  uv run --group learn python -u tools/selection_check.py --place {place} '
              '--merge --out "$out" 2>&1 | tail -6')
        print("}")
    print('\necho "=== ordering panel done ==="')


if __name__ == "__main__":
    main()
