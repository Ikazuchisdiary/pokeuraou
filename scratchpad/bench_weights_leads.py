# 出どころ: IKA-116 の棚卸しセッション（9/23）の一時 scratchpad にあった bench_weights_leads.py （sha256 d5379cea1a09e198、CRLF）を、改行を LF にしただけで写した。IKA-117/118 の受け入れの再生に使う。
"""Does the bench belief condition on which two LED, or only on which were seen?

`bench_weights` keeps every selection whose four contain the seen species. At turn 1 the
seen species are exactly the two leads, so a selection that planned one of them for the
BACK -- and led something that is not on the field -- is kept, and its back pair enters
the belief. The consistent set is the selections whose lead pair IS the observed pair.

For each opponent in the book and each lead pair the opponent's (softened) mixture puts
mass on, compare the belief `bench_weights` returns with the lead-conditioned one:
total variation, and the mass it puts on back pairs no lead-consistent selection has.
"""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
BOOK = Path(sys.argv[2])
EPS = float(sys.argv[3])
TEMP = float(sys.argv[4])
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from pokeuraou.regulation import to_id  # noqa: E402
from pokeuraou.selection_book import BenchPrior, SelectionBook, bench_weights  # noqa: E402

import inspect  # noqa: E402

# IKA-118 acceptance: a tree whose `bench_weights` takes `leads` is asked with the observed
# lead pair, the way `play_game` asks it; an older tree is asked the way it always was.
WITH_LEADS = "leads" in inspect.signature(bench_weights).parameters
print(f"tree {ROOT}: bench_weights {'with' if WITH_LEADS else 'without'} leads; "
      f"epsilon {EPS} temperature {TEMP}")

book = SelectionBook.read(BOOK)
from pokeuraou.teams import load_roster  # noqa: E402
OURS = [s.species for s in load_roster(book.roster or "rizabanadohido").sets]
rows = []
for entry in book.entries.values():
    for side in (0, 1):
        species = OURS if side == 0 else [s.species for s in entry.class_sets[0]]
        prior = BenchPrior.of(entry, side, species, epsilon=EPS, temperature=TEMP)
        probs = np.asarray(prior.probabilities)
        sels = prior.selections
        # lead pairs this side actually plays, weighted by how often
        lead_mass: dict[frozenset, float] = {}
        for sel, p in zip(sels, probs, strict=True):
            lead = frozenset(to_id(species[i]) for i in sel[:2])
            lead_mass[lead] = lead_mass.get(lead, 0.0) + float(p)
        for lead, mass in lead_mass.items():
            if mass <= 0:
                continue
            got = (
                bench_weights(sels, probs, species, lead, leads=lead)
                if WITH_LEADS
                else bench_weights(sels, probs, species, lead)
            )
            want: dict[tuple, float] = {}
            for sel, p in zip(sels, probs, strict=True):
                if frozenset(to_id(species[i]) for i in sel[:2]) != lead or p <= 0:
                    continue
                key = tuple(sorted(to_id(species[i]) for i in sel[2:]))
                want[key] = want.get(key, 0.0) + float(p)
            total = sum(want.values())
            want = {k: v / total for k, v in want.items()}
            keys = set(got) | set(want)
            tv = 0.5 * sum(abs(got.get(k, 0.0) - want.get(k, 0.0)) for k in keys)
            impossible = sum(v for k, v in got.items() if k not in want)
            rows.append((side, mass, tv, impossible))
    else:
        continue

if not rows:
    raise SystemExit("could not read sides from this book's entries; adjust the script")
rows_arr = np.array([(s, m, tv, imp) for s, m, tv, imp in rows])
for side in (0, 1):
    part = rows_arr[rows_arr[:, 0] == side]
    w = part[:, 1] / part[:, 1].sum()
    print(f"side {side} ({'ours' if side == 0 else 'theirs'}): {len(part)} (entry, lead) cells; "
          f"lead-mass-weighted TV {float(w @ part[:, 2]):.3f}, "
          f"mass on back pairs no consistent selection has {float(w @ part[:, 3]):.3f}; "
          f"max TV {part[:, 2].max():.3f}")
