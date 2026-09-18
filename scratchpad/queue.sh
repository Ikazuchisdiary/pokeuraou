set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou

# The block's queue, chained so the machine is never idle between steps. The anchor idled
# fifteen minutes tonight waiting on a file nothing wrote; a chain that hands the machine
# straight on is the fix for that class of loss, not a better gate.

echo "=== step 2: gen10's own selection book ==="
date
MODEL=data/models/value-gen10.pt LOGS=data/selection/logs-gen10 \
  bash tools/solve_book_parallel.sh 2>&1 | tail -20
echo "=== step 2 done ==="
date

# G10-b. Every rating row since generation 10 drew BOTH arms' selections from the same
# book, and that book was value-gen9 -- two generations stale. An ordering error both arms
# share cancels exactly, and tonight's panel found the ordering is where the error is: in
# four of five opponents the heavier-weighted selection was the weaker one on the board.
# So this is the same comparison with each arm drawing from its OWN book. The pairing is
# looser, so the interval is wider for the same games; that is the price of measuring the
# thing the rating was defined to measure.
echo "=== step 3: gen11L(own book) vs gen10(own book) ==="
date
uv run --group learn python -u tools/match_queue.py \
  --out data/matches/gen11L-vs-gen10-ownbooks --games 848 --seed 20260919 --served \
  --value data/models/value-gen11L.pt --baseline data/models/value-gen10.pt \
  -- --limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz \
     --baseline-selection-book data/selection/rizabanadohido-value-gen10.jsonl.gz 2>&1 | tail -20
echo "=== step 3 done ==="
date

# The same match with ONE book on both arms, so the two rows differ in exactly one thing
# and the cancelled amount is the difference between them. Without this row, "own books
# moved the number" cannot be told from "gen10's book is worse than gen9's".
echo "=== step 3b: same pair, one shared book (the old convention) ==="
date
uv run --group learn python -u tools/match_queue.py \
  --out data/matches/gen11L-vs-gen10-samebook --games 848 --seed 20260919 --served \
  --value data/models/value-gen11L.pt --baseline data/models/value-gen10.pt \
  -- --limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz 2>&1 | tail -20
echo "=== step 3b done ==="
date
echo "=== queue done ==="
