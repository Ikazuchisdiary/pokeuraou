"""Is the equilibrium's heavier selection actually the better one?

The book's job is not to print a win probability, it is to say which four to bring. On six
opponents measured with the corrected harness, the selection the equilibrium weighted most
heavily was the WEAKER one on the board in four of five -- and on place 34 a selection it
gave zero mass to won 40% against the heavy one's 32%.

The rating cannot see this. `generation_match` draws both arms' selections from the same
book, so an ordering error both arms share cancels; book-against-uniform was +18.9 points
because discarding the bad 87 of 90 is a different and much easier question than ordering
the top three. So this measures the ordering directly, on its own.

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
        help="skip an opponent whose second selection carries less mass than this: with a "
        "pure equilibrium there is no ordering to test.",
    )
    args = ap.parse_args()

    book = SelectionBook.read(args.book)
    roster = load_roster("rizabanadohido")
    reg = roster.reg
    species = [reg.species[s.species].name.lower().replace(" ", "") for s in roster.sets]

    entries = sorted(book.entries.values(), key=lambda e: e.place)
    rng = np.random.default_rng(args.seed)
    picked = rng.permutation(len(entries))

    chosen: list[tuple[int, str, str, float, float]] = []
    for index in picked:
        entry = entries[int(index)]
        x = np.asarray(entry.our_strategy, dtype=np.float64)
        order = np.argsort(-x)
        if float(x[order[1]]) < args.min_second:
            continue
        first, second = entry.selections[int(order[0])], entry.selections[int(order[1])]
        chosen.append(
            (
                entry.place,
                ",".join(species[i] for i in first),
                ",".join(species[i] for i in second),
                float(x[order[0]]),
                float(x[order[1]]),
            )
        )
        if len(chosen) >= args.opponents:
            break

    print(f"# {len(chosen)} opponents with a second selection carrying >= "
          f"{args.min_second:.0%}, drawn with seed {args.seed}")
    print("# each pair is paired per game: same opponent, same spread class, same index")
    print("set -uo pipefail")
    print("cd /c/Users/Ikazuchi/repos/pokeuraou")
    # The named arms are labelled 'a+b / c+d'. The obvious selector is '/', and it does
    # not survive the shell: MSYS rewrites a lone slash as a Windows path, so the worker
    # was handed --only-arm 'C:/Program Files/Git/' and refused every opponent before
    # playing a single game. Sixteen opponents of nothing, about five hours, caught by
    # running one of them for two games first.
    #
    # '+' selects the same two arms -- it is in every forced label and in none of the
    # three standard ones -- and is not a path character.
    print("DIR=data/matches/ordering")
    print('mkdir -p "$DIR/logs"')
    for place, first, second, p1, p2 in chosen:
        print(f"\n# place {place}: {p1:.1%} vs {p2:.1%}")
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
