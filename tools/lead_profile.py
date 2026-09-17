"""Which of our six each solved book actually brings, and which two it leads with.

A selection book prints 90 ordered selections per opponent sheet, which is unreadable as a
statement about our team. Folded over the field it becomes six numbers a person can check
against their own opinion: how often each of our six is brought at all, and how often it
is one of the two on the floor at turn 1.

This exists because a recorded finding turned out to be a property of one book. E4 read
three numbers off `value-all` -- Incineroar led 6.6% of the time, Toxapex 2.5%, Venusaur
35.3% -- and concluded that the model brings Toxapex but will not lead it, against a human
article that leads it in 47% of its fifteen plans. Other books disagree flatly:

    lead mass        value-gen8  gen8-s1  gen8-s2  value-gen9  value-all
      Charizard           98.5%    99.9%   100.0%       75.2%      77.4%
      Venusaur            35.0%     3.0%    25.2%        2.5%      35.3%
      Garchomp             4.9%    32.9%    18.3%        7.0%      46.3%
      Sylveon             13.4%    61.0%    39.9%       64.8%      31.9%
      Toxapex             28.1%     3.1%     1.7%       25.7%       2.5%
      Incineroar          20.1%     0.0%    14.8%       24.8%       6.6%

The first three columns differ only in the training seed -- the same 690,840 decisions,
the same configuration, held-out AUC within 0.010 of each other. Across those three,
Toxapex's lead mass spans 1.7% to 28.1% and Sylveon's selection mass 14.3% to 86.1%.
Leading Charizard is the only line stable across all five, and that agrees with T9, which
measured on the board that the cost of a forced Incineroar+Toxapex lead is benching
Charizard.

So no row here means anything without the book it came from named beside it. See
`tools/book_seed_spread.py` for what that instability costs.

    uv run python tools/lead_profile.py
    uv run python tools/lead_profile.py data/selection/rizabanadohido-value-all.jsonl.gz
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.names import localiser  # noqa: E402
from pokeuraou.selection_book import SelectionBook  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402


def profile(book: SelectionBook, slots: int) -> tuple[np.ndarray, np.ndarray]:
    """(lead mass, selection mass) per party slot, averaged over the sheets in the book.

    Lead mass sums to 2 and selection mass to 4, because every selection names two leads
    and four brought -- printing them as percentages of the field keeps each row readable
    on its own rather than as a share of the other five.
    """
    lead, picked = np.zeros(slots), np.zeros(slots)
    for entry in book.entries.values():
        strategy = np.asarray(entry.our_strategy, dtype=np.float64)
        for selection, weight in zip(entry.selections, strategy, strict=True):
            for index in selection[:2]:
                lead[index] += float(weight)
            for index in selection:
                picked[index] += float(weight)
    n = max(len(book.entries), 1)
    return lead / n, picked / n


def main() -> None:
    paths = sys.argv[1:] or sorted(glob.glob("data/selection/rizabanadohido-*.jsonl.gz"))
    paths = [p for p in paths if ".part" not in p]
    if not paths:
        raise SystemExit("no solved books found under data/selection/")

    roster = load_roster("rizabanadohido")
    reg = roster.reg
    loc = localiser(reg, "ja")
    names = [
        loc.species(s.species) if loc else reg.species[s.species].name for s in roster.sets
    ]

    books = {}
    for path in paths:
        book = SelectionBook.read(Path(path))
        book.require_roster("rizabanadohido")
        books[Path(path).stem.replace(".jsonl", "")] = (book, profile(book, len(names)))

    width = max(len(label) for label in books) + 2
    header = f"  {'':<12}" + "".join(f"{label:>{width}}" for label in books)
    for title, which in (("LEAD mass (sums to 2.00)", 0), ("SELECTION mass (sums to 4.00)", 1)):
        print(f"\n  {title}")
        print(header)
        for index, name in enumerate(names):
            cells = "".join(
                f"{100 * books[label][1][which][index]:>{width - 1}.1f}%" for label in books
            )
            print(f"  {name:<12}{cells}")
    print("\n  sheets: " + "   ".join(f"{label} {len(b)}" for label, (b, _) in books.items()))
    print("\n  No row here means anything without the book beside it: three of the books\n"
          "  on disk differ only in the training seed and disagree about half the team.")


if __name__ == "__main__":
    main()
