"""What the selection advice is worth against an opponent who does not have it.

The book prints one number: the equilibrium value, which is what our mixture gets against
an opponent who also solves the game -- with our own value function. That is the
guaranteed floor and it is the right number to quote when nothing is known about who is on
the other side. It is not the number anyone's next game is played at, and the record
already knows the difference is large: +18.9 points measured on the board between a side
that drew from the book and one that drew uniformly.

That difference is recoverable from the solved book with no solve and no games. The column
player's EV loss per selection is, by construction, how much less that selection gets them
than their best reply does, so our value against a distribution q over their selections is

    v(q) = value + sum_k w_k * <q, their_ev_loss[k]>

and with q uniform that is one mean per spread class. The identity is checked here rather
than asserted: q = their own equilibrium has to return the equilibrium value, because the
EV loss is zero on every selection their equilibrium plays.

Only uniform ships. It is the only opponent distribution the repository has a source for --
"an opponent who does not have the book" is not the book's own opponent mixture, and
choosing any other q would be inventing the claim the number is supposed to carry.

**It ships with its error, not with a correction.** Two board measurements exist and they
miss in opposite directions, so there is no single factor to apply:

    value-gen2 book   claims 80.7%   E1 measured 71.4% [65.9, 76.4]   9.3 too high
    value-all  book   claims 67.0%   E2 measured 71.1%                4.1 too low

    uv run python tools/book_against_uniform.py
    uv run python tools/book_against_uniform.py data/selection/some-book.jsonl.gz
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.selection_book import SelectionBook  # noqa: E402

#: Board anchors for this quantity, as (what a book claimed, what was measured, the note).
#: Printed with every report, because a number whose only two checks disagree in sign is a
#: number that has to carry them.
ANCHORS = (
    ("value-gen2", 80.7, "71.4 [65.9, 76.4] (E1)", "9.3 too high"),
    ("value-all", 67.0, "71.1 (E2)", "4.1 too low"),
)


def against(entry, q=None) -> float:  # noqa: ANN001
    """Our value when they play `q` over their 90 ordered selections, uniform by default."""
    weights = np.asarray(entry.class_weights, dtype=np.float64)
    weights = weights / weights.sum()
    total = 0.0
    for index, (weight, loss) in enumerate(zip(weights, entry.their_ev_loss, strict=True)):
        losses = np.asarray(loss, dtype=np.float64)
        theirs = (
            np.full(len(losses), 1.0 / len(losses))
            if q is None
            else np.asarray(q[index], dtype=np.float64)
        )
        total += float(weight) * float(theirs @ losses)
    return float(entry.value) + total


def main() -> None:
    paths = sys.argv[1:] or sorted(glob.glob("data/selection/rizabanadohido-*.jsonl.gz"))
    paths = [p for p in paths if ".part" not in p]
    if not paths:
        raise SystemExit("no solved books found under data/selection/")

    for path in paths:
        book = SelectionBook.read(Path(path))
        rows, drift = [], []
        for entry in book.entries.values():
            rows.append((float(entry.value), against(entry)))
            # The identity, checked on every entry: their own equilibrium gives up nothing.
            drift.append(abs(against(entry, entry.their_strategies) - float(entry.value)))
        if not rows:
            print(f"  {Path(path).name}: no entries")
            continue
        arr = np.array(rows)
        gain = arr[:, 1] - arr[:, 0]
        print(f"\n  {Path(path).name}  ({len(rows)} opponent sheets)")
        print(f"    equilibrium value          {arr[:, 0].mean():6.2%}")
        print(f"    against a uniform drawer   {arr[:, 1].mean():6.2%}"
              f"   {100 * gain.mean():+.2f} points"
              f"  [{100 * gain.min():+.2f}, {100 * gain.max():+.2f}]")
        worst = max(drift)
        print(f"    identity check             {worst:.2e}"
              f"   (their own equilibrium must return the equilibrium value)")
        if worst > 1e-6:
            print("    ^ that is too large; the stored EV losses do not match the value")

    print("\n  Board anchors for this quantity -- they miss in opposite directions, so it\n"
          "  ships with an error bar and not with a correction:")
    for name, claimed, measured, note in ANCHORS:
        print(f"    {name:<12} claimed {claimed:5.1f}%   measured {measured:<22} {note}")


if __name__ == "__main__":
    main()
