set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou
until grep -q "=== queue done ===" "$QOUT"; do sleep 20; done

# ------------------------------------- A0. is it the leaf or the book? Leaf held fixed.
#
# gen11L drawing its own book lost to gen10 drawing its own: 46.64% +-2.37%, which does
# exclude 50%. Two things differ between those arms, so the number does not say which one
# moved -- and the answer changes what to fix.
#
#   step 3b   book fixed (gen11L's on both), leaf varies   -> the leaf's contribution
#   A0 here   leaf fixed (gen11L on both), book varies     -> the book's contribution
#
# The arms are the same network on both sides, so anything that is not the solved
# selection is identical between them, down to the search width and the candidate order.
OUT=data/matches/gen11Lbook-vs-gen10book-sameleaf
if [ ! -f "$OUT/DONE" ]; then
  echo "=== A0: gen11L's book vs gen10's book, leaf gen11L on both ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20260918 --served \
    --value data/models/value-gen11L.pt --baseline data/models/value-gen11L.pt \
    -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
       --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz \
       --baseline-selection-book data/selection/rizabanadohido-value-gen10.jsonl.gz \
       2>&1 | tail -8
  echo "gen11L book vs gen10 book, leaf value-gen11L both arms, open, seed 20260918" \
    > "$OUT/DONE"
fi
echo "=== A0 done ==="
date

# ---------------------------------------------------------------- A. hidden-bench anchor
#
# A match whose search can see the opponent's actual four is solving a game a real one does
# not offer, so those rows are reference. The condition that ships holds 20,352 recorded
# games -- gen10x2, gen11hx2, gen11Lx2, all width 24 with leaf ordering and gen9's book --
# and has never had a zero in it, so every number in it is a difference against another
# unanchored number.
#
# Those three already play each other, so ONE edge to a parameter-free opponent gives the
# whole component a position. The arm is spelled to match what is already recorded, down to
# the book's file stem: a bridge named differently from the thing it bridges connects
# nothing, which is the defect repaired in 6ecbf3f, and a second one would be worse than
# none.
#
# hp-share here is NOT the open-bench zero. Different game, so a second scale, read with
# --anchor hp-share/w24/hidden-bench.
OUT=data/matches/anchor-gen11Lx2-book9-hidden-vs-hpshare
if [ ! -f "$OUT/DONE" ]; then
  echo "=== A: hidden-bench bridge, gen11Lx2+gen9book vs hp-share ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20260919 --served --hide-bench \
    --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
    -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
       --selection-book data/selection/rizabanadohido-value-gen9.jsonl.gz \
       --baseline-uniform-selection 2>&1 | tail -8
  echo "hidden-bench anchor bridge, seed 20260919" > "$OUT/DONE"
fi
echo "=== A done ==="
date

# --------------------------------------------- A2. the same anchor, drawing its own book
#
# A uses value-gen9's book because that is what the 20,352 recorded games used and a bridge
# has to be spelled like the thing it bridges. But a book is not an opening repertoire --
# it is the equilibrium of the selection game AS THAT MODEL'S VALUE FUNCTION DEFINES IT --
# so an agent drawing from another model's book selects by one evaluation and plays by
# another. Not stale: incoherent. Three generations of training were all made to select
# with generation 9's opinion.
#
# So the same anchor again with the ensemble's own book, which is the configuration that
# would actually ship. The difference between A and A2 is what using someone else's book
# has been costing.
OUT=data/matches/anchor-gen11Lx2-ownbook-hidden-vs-hpshare
if [ ! -f "$OUT/DONE" ]; then
  echo "=== A2: hidden anchor, gen11Lx2 with ITS OWN book, vs hp-share ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20260920 --served --hide-bench \
    --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
    -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
       --selection-book data/selection/rizabanadohido-value-gen11L-ens2.jsonl.gz \
       --baseline-uniform-selection 2>&1 | tail -8
  echo "hidden anchor with the ensemble's own book, seed 20260920" > "$OUT/DONE"
fi
echo "=== A2 done ==="
date

# ------------------------------------ A3. what the borrowed book costs, measured directly
#
# A and A2 each place an agent on the scale and they use different seeds, so differencing
# them carries both intervals. The question they are differenced for -- what did drawing
# from generation 9's book cost -- is a contrast between two configurations, and a contrast
# is measured by putting them in one match. Same leaf on both arms, same width, same
# ordering, same hidden bench; the only difference is which solved selection each draws
# from. Both arms then necessarily see the same games, which is what a match is, rather
# than two runs made to agree by seed.
OUT=data/matches/gen11Lx2-ownbook-vs-gen9book-hidden
if [ ! -f "$OUT/DONE" ]; then
  echo "=== A3: own book vs gen9's book, same leaf, hidden bench ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20260921 --served --hide-bench \
    --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
    --baseline data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
    -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
       --selection-book data/selection/rizabanadohido-value-gen11L-ens2.jsonl.gz \
       --baseline-selection-book data/selection/rizabanadohido-value-gen9.jsonl.gz \
       2>&1 | tail -8
  echo "own book vs gen9 book, same leaf both arms, seed 20260921" > "$OUT/DONE"
fi
echo "=== A3 done ==="
date

# ------------------------------------------------- B. a pool played entirely in the dark
#
# Asked for directly: a model built purely under the hidden rule. The two hidden pools that
# exist are not 24,000 games of it -- gen11h and gen11L are both seed 1001 over the same
# indices, gen11L being gen11h with leaf ordering and nothing else, so game i draws the
# same team and the same selection in both. Twelve thousand setups, played twice.
#
# So: 24,000 fresh ones at seed 2001, leaf ordering (the pools agree it is the better
# generator: it keeps 93.3% of the full equilibrium's mass against damage ordering's
# 58.1%), the current best leaf and its own book. With gen11L's 12,000 that is 36,000
# distinct hidden setups to train on.
#
# 135 games/min was gen11h's measured rate for the same shape, so about three hours.
OUT=data/selfplay-hidden2
if [ ! -f "$OUT/DONE" ]; then
  echo "=== B: 24,000 hidden-bench games, leaf gen11L, seed 2001 ==="
  date
  uv run --group learn python -u tools/generate_queue.py \
    --out "$OUT" --games 24000 --seed 2001 --served --limit 24 \
    --value data/models/value-gen11L.pt --hide-bench \
    -- --rank-leaf 2>&1 | tail -8
  echo "24000 hidden games, leaf value-gen11L, book value-gen11L, seed 2001, --rank-leaf" \
    > "$OUT/DONE"
fi
echo "=== B done ==="
date

# ------------------------------------------- C. hidden-only, and a control of equal size
#
# A model trained on 36,000 hidden games against value-gen11L's 70,603 mostly-open ones
# would confound two things: the condition and the pool size. So the same trainer runs
# twice on the same number of games, and only the condition differs. Whichever way it
# comes out, the comparison is about the condition.
if [ ! -f data/models/value-hidden36.pt ]; then
  echo "=== C1: encode + train hidden-only (36,000) ==="
  date
  uv run --group learn python -u tools/encode_dataset.py \
    --dir data/selfplay-gen11L data/selfplay-hidden2 \
    --out data/selfplay-hidden36-encoded.npz 2>&1 | tail -5
  uv run --group learn python -u tools/train_value.py \
    --data data/selfplay-hidden36-encoded.npz \
    --out data/models/value-hidden36.pt 2>&1 | tail -12
fi
echo "=== C1 done ==="
date

# gen7 + gen8 + gen9 = 30,982 and gen10 brings it to 42,982; the three closest to 36,000
# are taken, which lands at 30,982 against 36,000. Not equal, and the direction is stated
# rather than buried: the control has FEWER games, so if the hidden model loses, the size
# argument cannot explain it, and if it wins, this comparison alone does not settle it.
if [ ! -f data/models/value-open31.pt ]; then
  echo "=== C2: encode + train open control (30,982) ==="
  date
  uv run --group learn python -u tools/encode_dataset.py \
    --dir data/selfplay-gen7 data/selfplay-gen8 data/selfplay-gen9 \
    --out data/selfplay-open31-encoded.npz 2>&1 | tail -5
  uv run --group learn python -u tools/train_value.py \
    --data data/selfplay-open31-encoded.npz \
    --out data/models/value-open31.pt 2>&1 | tail -12
fi
echo "=== C2 done ==="
date

# ------------------------------------------- C3. a book for each of the two new models
#
# A new model needs its own book before it plays. An agent is its model together with the
# book it draws from, and a book is worth about 141 Elo -- the whole distance from the
# parameter-free baseline to the best agent on the scale -- so a match between two new
# leaves with no books measures the leaves and reports the answer as the agents'. Worse
# here than usual: tonight's finding is that the ordering INSIDE the book is where the
# error lives, so a comparison that removes the book removes the thing under test.
#
# About 35 minutes each at 8 shards.
for M in value-hidden36 value-open31; do
  B="data/selection/rizabanadohido-$M.jsonl.gz"
  [ -f "$B" ] && { echo "skip book for $M"; continue; }
  echo "=== C3: selection book for $M ==="
  date
  MODEL="data/models/$M.pt" LOGS="data/selection/logs-$M" \
    bash tools/solve_book_parallel.sh 2>&1 | tail -6
done
echo "=== C3 done ==="
date

# ------------------------------------------------------ D. the two of them, in the dark
#
# Each arm draws from its OWN book. That is the comparison a rating is defined to make and
# the one every row since generation 10 failed to make: those drew both arms from
# value-gen9's book, two generations stale, so an ordering error the arms shared cancelled
# exactly and the rating could not see it.
echo "=== D: value-hidden36 vs value-open31, own books, hidden bench ==="
date
uv run --group learn python -u tools/match_queue.py \
  --out data/matches/hidden36-vs-open31 --games 848 --seed 20260941 --served --hide-bench \
  --value data/models/value-hidden36.pt --baseline data/models/value-open31.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-hidden36.jsonl.gz \
     --baseline-selection-book data/selection/rizabanadohido-value-open31.jsonl.gz \
     2>&1 | tail -8
echo "=== D done ==="
date

# ------------------------------------------ E. the open ladder, reference but comparable
#
# Reference by the rule above, and kept anyway: the only measurement that survives from six
# generations back is in this condition, and a paired curve against a parameter-free
# opponent is the one thing here that cannot drift. Same seed everywhere, so game i is the
# same opponent, the same spread class and the same hp-share arm in every row and only our
# leaf changes -- paired, which is worth more than more games (gen234 against gen11L
# unpaired was +1.84% +-4.02%).
# One seed per row, not one for the ladder. Sharing it would tighten the differences
# between these six and buy that with a corpus that stops gaining independent opponent
# draws as rows are added -- and the rating pools every match ever recorded, so that
# cost lands on every future comparison rather than on this table.
LADDER_SEED=20260930
for M in value-gen234 value-gen2345 value-gen8 value-gen9 value-gen10 value-gen11h; do
  LADDER_SEED=$((LADDER_SEED + 1))
  OUT="data/matches/anchor-$M-vs-hpshare"
  [ -f "$OUT/DONE" ] && { echo "skip $M"; continue; }
  echo "=== E: ladder $M vs hp-share (open, reference), seed $LADDER_SEED ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed "$LADDER_SEED" --served \
    --value "data/models/$M.pt" \
    -- --objective hp-share --limit 24 --baseline-limit 24 2>&1 | tail -6
  echo "$M vs hp-share: w24, damage, uniform, OPEN bench (reference), seed $LADDER_SEED" \
    > "$OUT/DONE"
done
echo "=== ladder done ==="
date
