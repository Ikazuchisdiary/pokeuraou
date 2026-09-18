set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou
until grep -q "=== run done ===" "$RUNOUT"; do sleep 30; done

# ------------------------------------------- 8. hidden-only, and a control of equal size
#
# A model trained on 36,000 hidden games against value-gen11L's 70,603 mostly-open ones
# would confound the condition with the pool size, so the same trainer runs twice and only
# the condition differs.
#
# The hidden pool is gen11L's 12,000 plus the 24,000 from step 7. NOT gen11h's: it and
# gen11L are the same 12,000 setups at seed 1001, differing only in the candidate order,
# so adding it would add games without adding opponents.
if [ ! -f data/models/value-hidden36.pt ]; then
  echo "=== 8a: encode + train hidden-only (36,000) ==="
  date
  uv run --group learn python -u tools/encode_dataset.py \
    --dir data/selfplay-gen11L data/selfplay-hidden2 \
    --out data/selfplay-hidden36-encoded.npz 2>&1 | tail -6
  uv run --group learn python -u tools/train_value.py \
    --data data/selfplay-hidden36-encoded.npz \
    --out data/models/value-hidden36.pt --split-seed 0 2>&1 | tail -14
fi
echo "=== 8a done ==="
date

# gen7 + gen8 + gen9 = 30,982, the three most recent open pools that stay under the hidden
# one. The direction is stated before the result exists: the control has FEWER games, so a
# hidden model that loses cannot blame its pool size, and one that wins is not settled by
# this comparison alone.
if [ ! -f data/models/value-open31.pt ]; then
  echo "=== 8b: encode + train open control (30,982) ==="
  date
  uv run --group learn python -u tools/encode_dataset.py \
    --dir data/selfplay-gen7 data/selfplay-gen8 data/selfplay-gen9 \
    --out data/selfplay-open31-encoded.npz 2>&1 | tail -6
  uv run --group learn python -u tools/train_value.py \
    --data data/selfplay-open31-encoded.npz \
    --out data/models/value-open31.pt --split-seed 0 2>&1 | tail -14
fi
echo "=== 8b done ==="
date

# ------------------------------------------------ 9. a book each, then the two of them
# A model without a solved book does not get matched: an agent is its model together with
# the book it draws from, and the book is worth about 141 Elo.
for M in value-hidden36 value-open31; do
  B="data/selection/rizabanadohido-$M.jsonl.gz"
  [ -f "$B" ] && { echo "skip book for $M"; continue; }
  echo "=== 9a: selection book for $M ==="
  date
  MODEL="data/models/$M.pt" LOGS="data/selection/logs-$M" \
    bash tools/solve_book_parallel.sh 2>&1 | tail -6
done
echo "=== 9a done ==="
date

echo "=== 9b: value-hidden36 vs value-open31, own books, hidden bench ==="
date
uv run --group learn python -u tools/match_queue.py \
  --out data/matches/hidden36-vs-open31 --games 848 --seed 20260971 --served --hide-bench \
  --value data/models/value-hidden36.pt --baseline data/models/value-open31.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-hidden36.jsonl.gz \
     --baseline-selection-book data/selection/rizabanadohido-value-open31.jsonl.gz \
     2>&1 | tail -10
echo "=== 9b done ==="
date

# Each against the hidden-bench parameter-free opponent too, so both land on the scale
# rather than only against each other. Own seeds: these place agents.
S=20260980
for M in value-hidden36 value-open31; do
  S=$((S + 1))
  OUT="data/matches/anchor-$M-hidden-vs-hpshare"
  [ -f "$OUT/DONE" ] && { echo "skip anchor $M"; continue; }
  echo "=== 9c: $M vs hp-share, hidden bench, own book, seed $S ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed "$S" --served --hide-bench \
    --value "data/models/$M.pt" \
    -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
       --selection-book "data/selection/rizabanadohido-$M.jsonl.gz" \
       --baseline-uniform-selection 2>&1 | tail -8
  echo "$M vs hp-share, hidden, own book, seed $S" > "$OUT/DONE"
done
echo "=== 9c done ==="
date

# ------------------------------------------------------------ 10. the ordering panel
# The book's job is to say which four to bring, and on six opponents the selection the
# equilibrium weighted most was the WEAKER one on the board in four of five. A rating
# cannot see that when both arms draw from one book. Sixteen opponents, paired per game
# index on the two forced selections; `tools/ordering_result.py` reads it back.
echo "=== 10: ordering panel, 16 opponents ==="
date
bash "$ORDER"
echo "=== 10 done ==="
date
echo "=== run2 done ==="
