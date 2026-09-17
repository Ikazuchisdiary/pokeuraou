"""Is the printed selection advice a property of the data, or of the training seed?

The book is the largest measured lever in the project (+13.6 points on the board) and it is
the product's headline output: for each opponent sheet it prints a mixture over 90 ordered
selections, and a person reads it and brings four Pokemon. Nobody has asked whether that
mixture is reproducible.

There is a reason to think it might not be. The book's cells are turn-1 positions
exclusively, and turn 1 is where the value function is weakest by a wide margin -- AUC 0.681
at turn 1 against 0.958 by turn 10, from `data/train-gen10-s0.log`. And the record already
measures what a training seed is worth at that scale: held-out AUC scatters by 0.0240
between seeds of one configuration, against 0.004 between configurations.

So this compares books solved from two or more **seeds of the same configuration** -- not
two generations, which would confound the seed with the data. What it reports is in the
product's own units:

- **total variation** between the two printed mixtures, per sheet;
- **how much lead mass moves** -- summed over the 15 lead pairs, which is what a reader
  acts on;
- **whether the top selection agrees at all**, which is the coarsest possible reading;
- **the claimed value**, whose movement is the model disagreeing with itself;
- **the claimed value against a uniform drawer**, which is the product's headline number
  (`tools/book_against_uniform.py`) and the one a person would quote;
- **what following the other seed's advice costs**, which is the only line that says
  whether any of the rest matters.

That last one is the point. A near-pure equilibrium is a hair trigger: two selections
within a tenth of a point of each other swap places under any noise at all, and the
mixtures then score a total variation of 1.0 while the choice between them is worth
almost nothing. So the cost is read off the loser's own book. `our_ev_loss[i]` is
`value - row_ev[i]`, what our selection i gives up against the opponent's equilibrium,
so `x_B . our_ev_loss_A` is what seed B's advice gives up **in seed A's own game**, and
the pure version is what a person loses by having read the other book. It is a lower
bound on the damage: a real opponent would best-reply to the mixture in front of them
rather than keep playing the equilibrium of a game they are not in.

**What it found**, over 394 opponent sheets, across three seeds of `value-gen8` -- the same
690,840 decisions, the same configuration, held-out AUC 0.8745, 0.8807 and 0.8843:

    pair              same four   the other's four ranks   following it costs   of the book
    gen8 vs s1            18.0%           14th of 90            5.86 points          38%
    gen8 vs s2            32.7%            5th                  3.09                 19%
    s1   vs s2            38.3%            4th                  2.88                 17%
                                                   having no book at all: 15.3 to 16.8

**Report the range.** The first pair measured was the worst by a factor of two, and quoting
it alone would have been the error this tool exists to expose. `value-gen8` is also the
weakest of the three -- lowest AUC, stopped an epoch earlier -- so the pairs that include
it carry some model quality on top of the seed. The floor, from the two that do not, is
still that **two seeds of one configuration bring a different four 62% of the time, at a
cost of 2.88 points against a 16.8-point scale.**

Pointed at two generations instead, the same lines give that a scale: `value-gen9` against
`value-all` costs 4.56 points, `value-gen2` against `value-gen9` 10.86. So the seed's
contribution is the same order as the distance between neighbouring models -- enough to
say a single-seed comparison of neighbours cannot be read, not enough to say every
generation comparison was noise.

Two readings are ruled out by the same run. The book cannot tell which of its own answers
are seed luck -- the cost is flat across quartiles of how decided the matchup looks (4.87,
6.25, 6.25, 6.12) and runs the wrong way against the book's own margin between its first
and second choice, 5.36 where the call was close and 6.86 where it was clear. And the
seeds are not merely swapping first and second place: within a book the runner-up is at
most 2.14 points behind, while the other seed's choice is 5.88 behind at median rank 14.

The remedy is `--model a.pt b.pt` on the solver; see T5 in TODO.md for the prediction and
`tools/lead_profile.py` for what the instability does to a book's team.

    uv run python tools/book_seed_spread.py data/selection/seedcheck/*.jsonl.gz
    uv run python tools/book_seed_spread.py data/selection/rizabanadohido-value-{gen9,all}.jsonl.gz
"""

from __future__ import annotations

import glob
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.selection_book import SelectionBook  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from book_against_uniform import against  # noqa: E402


def lead_mass(entry, strategy: np.ndarray) -> dict[frozenset, float]:
    """Probability on each unordered lead pair, which is what a reader acts on."""
    out: dict[frozenset, float] = {}
    for selection, weight in zip(entry.selections, strategy, strict=True):
        key = frozenset(selection[:2])
        out[key] = out.get(key, 0.0) + float(weight)
    return out


def main() -> None:
    paths = sys.argv[1:] or sorted(glob.glob("data/selection/seedcheck/*.jsonl.gz"))
    paths = [p for p in paths if ".part" not in p]
    if len(paths) < 2:
        raise SystemExit(
            f"need at least two books solved from different seeds; got {len(paths)}"
        )
    books = {Path(p).stem.replace(".jsonl", ""): SelectionBook.read(Path(p)) for p in paths}
    for name, book in books.items():
        print(f"  {name}: {len(book.entries)} sheets")

    for left, right in itertools.combinations(books, 2):
        a, b = books[left], books[right]
        shared = sorted(set(a.entries) & set(b.entries))
        if not shared:
            print(f"\n  {left} vs {right}: no sheets in common")
            continue
        tvs, leads, tops, insides, values, uniforms, supports = [], [], [], [], [], [], []
        costs, pure_costs, randoms = [], [], []
        rankings, same_four = [], []
        for key in shared:
            ea, eb = a.entries[key], b.entries[key]
            xa = np.asarray(ea.our_strategy, dtype=np.float64)
            xb = np.asarray(eb.our_strategy, dtype=np.float64)
            tvs.append(0.5 * float(np.abs(xa - xb).sum()))
            la, lb = lead_mass(ea, xa), lead_mass(eb, xb)
            keys = set(la) | set(lb)
            leads.append(0.5 * sum(abs(la.get(k, 0.0) - lb.get(k, 0.0)) for k in keys))
            tops.append(int(np.argmax(xa) == np.argmax(xb)))
            # Weaker than agreement and worth reporting beside it: with a support of two,
            # two seeds can name different selections and still both be inside the other's
            # mixture, which is a different failure from naming a selection the other
            # gives no weight at all.
            insides.append(int(xb[int(np.argmax(xa))] > 1e-6))
            values.append(abs(float(ea.value) - float(eb.value)))
            uniforms.append(abs(against(ea) - against(eb)))
            # Each seed's advice, priced in the other's game. Both directions, because
            # neither book is the truth and an asymmetry would itself be worth seeing.
            loss_a = np.asarray(ea.our_ev_loss, dtype=np.float64)
            loss_b = np.asarray(eb.our_ev_loss, dtype=np.float64)
            costs.append(0.5 * (float(xb @ loss_a) + float(xa @ loss_b)))
            # The scale the cost is read against: the same regret vector, under a
            # player who has no book at all and picks one of the 90 at random.
            randoms.append(0.5 * (float(loss_a.mean()) + float(loss_b.mean())))
            # Where the other book's choice sits in this one's own ordering, and
            # whether it is even the same four Pokemon once the lead order is
            # ignored. These are the two lines that need no units explained.
            top_a, top_b = int(np.argmax(xa)), int(np.argmax(xb))
            order_a = np.argsort(loss_a).tolist()
            order_b = np.argsort(loss_b).tolist()
            rankings.append(order_a.index(top_b) + 1)
            rankings.append(order_b.index(top_a) + 1)
            same_four.append(
                int(set(ea.selections[top_a]) == set(eb.selections[top_b]))
            )
            pure_costs.append(
                0.5 * (float(loss_a[int(np.argmax(xb))]) + float(loss_b[int(np.argmax(xa))]))
            )
            supports.append((int((xa > 1e-6).sum()), int((xb > 1e-6).sum())))

        n = len(shared)
        tvs, leads = np.array(tvs), np.array(leads)
        values, uniforms = np.array(values), np.array(uniforms)
        costs, pure_costs = np.array(costs), np.array(pure_costs)
        randoms = np.array(randoms)
        rankings, same_four = np.array(rankings), np.array(same_four)
        sup = np.array(supports, dtype=np.float64)
        print(f"\n  {left}  vs  {right}   ({n} sheets in common)")
        print(f"    mixture total variation     mean {tvs.mean():.3f}   "
              f"median {np.median(tvs):.3f}   1.0 on {int((tvs > 0.999).sum())}/{n}")
        print(f"    lead mass moved             mean {100 * leads.mean():5.1f} points   "
              f"median {100 * np.median(leads):5.1f}   max {100 * leads.max():5.1f}")
        print(f"    same top selection          {sum(tops)}/{n} = {sum(tops) / n:.1%}")
        print(f"    its top is in the other's   {sum(insides)}/{n} = {sum(insides) / n:.1%}"
              f"   support")
        print(f"    claimed value moved         mean {100 * values.mean():5.2f} points   "
              f"median {100 * np.median(values):5.2f}   max {100 * values.max():5.2f}")
        print(f"    against a uniform drawer    mean {100 * uniforms.mean():5.2f} points   "
              f"median {100 * np.median(uniforms):5.2f}   max {100 * uniforms.max():5.2f}")
        print(f"    support size                {sup[:, 0].mean():.2f} vs {sup[:, 1].mean():.2f} "
              f"of 90")
        print(f"    the other book's advice     mean {100 * costs.mean():5.2f} points   "
              f"median {100 * np.median(costs):5.2f}   max {100 * costs.max():5.2f}"
              f"   costs us")
        print(f"      its top selection alone   mean {100 * pure_costs.mean():5.2f} points   "
              f"median {100 * np.median(pure_costs):5.2f}   max {100 * pure_costs.max():5.2f}")
        print(f"    having no book at all       mean {100 * randoms.mean():5.2f} points   "
              f"median {100 * np.median(randoms):5.2f}   max {100 * randoms.max():5.2f}"
              f"   <- the scale")
        share = float(costs.mean() / randoms.mean()) if randoms.mean() else float("nan")
        print(f"    so the difference costs     {share:.0%} of what the book is for")
        print(f"    its four ranks, in ours     median {int(np.median(rankings))} of 90   "
              f"mean {rankings.mean():.1f}   in our top 10 {(rankings <= 10).mean():.0%}")
        print(f"    the SAME four, order aside  {int(same_four.sum())}/{n} = "
              f"{same_four.mean():.1%}   <- what a person actually brings")

    print("\n  Two seeds of one configuration differ only in which random numbers\n"
          "  initialised and shuffled the training, so whatever moves between two such\n"
          "  books is not a property of the team, the opponent or the data. Pointed at\n"
          "  two generations instead, the same lines give that quantity a scale, which\n"
          "  is the only way to read it -- so what was compared has to be said, not\n"
          "  assumed from the file names.")
    print("  Read the last two lines first. They are the only ones denominated in the\n"
          "  game rather than in the mixture, and a book that disagrees at no cost is\n"
          "  one choosing between selections that are worth the same.")


if __name__ == "__main__":
    main()
