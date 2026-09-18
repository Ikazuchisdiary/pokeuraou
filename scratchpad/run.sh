set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou

# The whole block in one chain, rebuilt after two defects were found in the tools it uses:
#
#   --baseline-selection-book consulted one book per game, so no match has ever had each
#   arm drawing its own selection. Fixed by `draw_across`.
#
#   DEFAULT_TEMPERATURE was 0.05, the value an unused shell driver had overridden to 0.5
#   for a measured reason. Generations 10, 11h and 11L were all made at 0.05, where 34 of
#   90 selections get under 30 games in 24,000.
#
# Short matches first: they answer questions already on the table and each is under ten
# minutes. The three-hour generation is the long pole and goes after them.

# ---------------------------------------------------------- 1. G10-b, taken again, right
# The recorded -23.4 is void: it was two runs that each used ONE book, alternating which.
for pair in \
  "ownbooks|data/selection/rizabanadohido-value-gen10.jsonl.gz" \
  ; do
  other="${pair#*|}"
  OUT="data/matches/gen11L-vs-gen10-ownbooks-fixed"
  if [ ! -f "$OUT/DONE" ]; then
    echo "=== 1: gen11L(own book) vs gen10(own book), each arm its own ==="
    date
    uv run --group learn python -u tools/match_queue.py \
      --out "$OUT" --games 848 --seed 20260951 --served \
      --value data/models/value-gen11L.pt --baseline data/models/value-gen10.pt \
      -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
         --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz \
         --baseline-selection-book "$other" 2>&1 | tail -10
    echo "each arm its own book, seed 20260951, after the draw_across fix" > "$OUT/DONE"
  fi
done
echo "=== 1 done ==="
date

# -------------------------------------- 2. which book, with the leaf held fixed on both
# Now meaningful: before the fix this was guaranteed to return 50%, because one book
# governed both arms and the arms were the same network.
OUT=data/matches/gen11Lbook-vs-gen10book-sameleaf
if [ ! -f "$OUT/DONE" ]; then
  echo "=== 2: gen11L's book vs gen10's book, leaf gen11L on both ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20260952 --served \
    --value data/models/value-gen11L.pt --baseline data/models/value-gen11L.pt \
    -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
       --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz \
       --baseline-selection-book data/selection/rizabanadohido-value-gen10.jsonl.gz \
       2>&1 | tail -10
  echo "book comparison, leaf value-gen11L both arms, open, seed 20260952" > "$OUT/DONE"
fi
echo "=== 2 done ==="
date

# ------------------------------------------------------------- 3. the hidden-bench anchor
# 20,352 recorded games in the shipping condition have never had a zero in them. The three
# agents in that component play each other, so one edge to a parameter-free opponent gives
# all of them a position. Spelled exactly like the corpus, gen9's book included -- a bridge
# named differently from the thing it bridges connects nothing.
OUT=data/matches/anchor-gen11Lx2-book9-hidden-vs-hpshare
if [ ! -f "$OUT/DONE" ]; then
  echo "=== 3: hidden bridge, gen11Lx2 + gen9's book, vs hp-share ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20260953 --served --hide-bench \
    --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
    -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
       --selection-book data/selection/rizabanadohido-value-gen9.jsonl.gz \
       --baseline-uniform-selection 2>&1 | tail -10
  echo "hidden bridge with gen9's book, seed 20260953" > "$OUT/DONE"
fi
echo "=== 3 done ==="
date

# ------------------------------------------- 4. the same anchor, drawing its own book
# What would actually ship. Its own seed, because both rows place an agent on the scale and
# a corpus whose rows share a draw stops gaining independent opponents as rows are added.
OUT=data/matches/anchor-gen11Lx2-ownbook-hidden-vs-hpshare
if [ ! -f "$OUT/DONE" ]; then
  echo "=== 4: hidden anchor, gen11Lx2 with ITS OWN book, vs hp-share ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20260954 --served --hide-bench \
    --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
    -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
       --selection-book data/selection/rizabanadohido-value-gen11L-ens2.jsonl.gz \
       --baseline-uniform-selection 2>&1 | tail -10
  echo "hidden anchor with the ensemble's own book, seed 20260954" > "$OUT/DONE"
fi
echo "=== 4 done ==="
date

# ------------------------------------ 5. what the borrowed book costs, in one match
OUT=data/matches/gen11Lx2-ownbook-vs-gen9book-hidden
if [ ! -f "$OUT/DONE" ]; then
  echo "=== 5: own book vs gen9's book, same ensemble, hidden bench ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20260955 --served --hide-bench \
    --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
    --baseline data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
    -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
       --selection-book data/selection/rizabanadohido-value-gen11L-ens2.jsonl.gz \
       --baseline-selection-book data/selection/rizabanadohido-value-gen9.jsonl.gz \
       2>&1 | tail -10
  echo "own book vs gen9's, same ensemble both arms, seed 20260955" > "$OUT/DONE"
fi
echo "=== 5 done ==="
date

# --------------------------------------------- 6. the open ladder, reference but fixed
# Reference -- the search sees the opponent's four -- and the only axis that reaches back
# to 2026-09-12, whose conditions were checked against it by commit timestamp.
LADDER_SEED=20260960
for M in value-gen234 value-gen2345 value-gen8 value-gen9 value-gen10 value-gen11h; do
  LADDER_SEED=$((LADDER_SEED + 1))
  OUT="data/matches/anchor-$M-vs-hpshare"
  [ -f "$OUT/DONE" ] && { echo "skip $M"; continue; }
  echo "=== 6: ladder $M vs hp-share (open, reference), seed $LADDER_SEED ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed "$LADDER_SEED" --served \
    --value "data/models/$M.pt" \
    -- --objective hp-share --limit 24 --baseline-limit 24 2>&1 | tail -6
  echo "$M vs hp-share: w24, damage, uniform, OPEN bench, seed $LADDER_SEED" > "$OUT/DONE"
done
echo "=== 6 done ==="
date

# ----------------------------------------- 7. a pool played entirely in the dark, at 0.5
# The first attempt ran 27 minutes at temperature 0.05 and was discarded: 34 of 90
# selections would have taken under 30 games of 24,000. Passed explicitly here even though
# it is now the default, so the record of this pool says what it used.
OUT=data/selfplay-hidden2
if [ ! -f "$OUT/DONE" ]; then
  echo "=== 7: 24,000 hidden-bench games, leaf gen11L, seed 2001, T=0.5 ==="
  date
  uv run --group learn python -u tools/generate_queue.py \
    --out "$OUT" --games 24000 --seed 2001 --served --limit 24 \
    --value data/models/value-gen11L.pt --hide-bench \
    -- --rank-leaf --explore-temperature 0.5 --explore-epsilon 0.25 2>&1 | tail -10
  echo "24000 hidden games, leaf+book value-gen11L, seed 2001, --rank-leaf, e=0.25 T=0.5" \
    > "$OUT/DONE"
fi
echo "=== 7 done ==="
date
echo "=== run done ==="
