set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou

# Everything after the menu-ownership fix (e8d5fd1). The matches run before the training,
# because they answer questions already on the table and each is under twenty minutes,
# while the pool -> model -> book -> match sequence is four hours whatever order it is in.

# --------------------------------------------------- 0. stop the pool at 12,000 and use it
#
# Measured at 70 games a minute rather than the 135 gen11h managed at temperature 0.05 --
# a wider exploration draw reaches costlier positions -- so 24,000 is 5.7 hours and would
# eat the afternoon. 12,000 fresh setups at seed 2001 joins gen11L's 12,000 at seed 1001
# for 24,000 DISTINCT setups, which is what the pool was for; gen11h adds none, being
# gen11L's own setups played with the other candidate ordering.
until [ "$(cat data/selfplay-hidden2/games-worker*.jsonl 2>/dev/null | wc -l)" -ge 12000 ]; do
  sleep 60
done
echo "=== 0: pool reached 12,000, stopping generation ==="
date
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'genonly.sh|generate_queue|selfplay.py|inference_server' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
sleep 20
GAMES=$(cat data/selfplay-hidden2/games-worker*.jsonl 2>/dev/null | wc -l)
echo "$GAMES hidden games, leaf+book value-gen11L, seed 2001, --rank-leaf, e=0.25 T=0.5, stopped early" \
  > data/selfplay-hidden2/DONE
echo "=== 0 done: $GAMES games ==="
date

# ------------------------------------------------------------------- 1. the matches again
# Every one of these was measured before the fix and is void: with --rank-leaf on both arms
# and different models, the column player chose from the row player's menu.
run_match() {
  local name="$1"; shift
  local out="data/matches/$name"
  # A DONE that does not say whether the run WORKED is a DONE that makes the next attempt
  # skip a failure. Three directories in this project carry "FAILED" markers whose games
  # went into the rating anyway, and one of them says "0 games" over 107; the marker is
  # the only thing that could have stopped that and it was being written unconditionally.
  # `set -o pipefail` is on, so PIPESTATUS[0] is the queue's own exit.
  if grep -q "^ok" "$out/DONE" 2>/dev/null; then echo "skip $name (ok)"; return; fi
  echo "=== match: $name ==="
  date
  uv run --group learn python -u tools/match_queue.py --out "$out" "$@" 2>&1 | tail -8
  local status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "ok: $name, after the menu-ownership fix e8d5fd1" > "$out/DONE"
  else
    echo "FAILED (exit $status): $name" > "$out/DONE"
    echo "!! $name exited $status -- leaving it unmarked so a rerun retries it"
  fi
  uv run --group learn python tools/match_result.py "$out" 2>&1 | tail -16
  uv run --group learn python tools/paired_result.py "$out" 2>&1 | tail -8
}

# G10-b: each arm from its own book. The headline of the morning, re-measured.
run_match gen11L-vs-gen10-ownbooks-m2 \
  --games 848 --seed 20261001 --served \
  --value data/models/value-gen11L.pt --baseline data/models/value-gen10.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz \
     --baseline-selection-book data/selection/rizabanadohido-value-gen10.jsonl.gz

# The same pair with one book, so the two rows differ in exactly one thing.
run_match gen11L-vs-gen10-samebook-m2 \
  --games 848 --seed 20261002 --served \
  --value data/models/value-gen11L.pt --baseline data/models/value-gen10.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz

# Which book, with the leaf held fixed on both arms.
run_match gen11Lbook-vs-gen10book-sameleaf-m2 \
  --games 848 --seed 20261003 --served \
  --value data/models/value-gen11L.pt --baseline data/models/value-gen11L.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz \
     --baseline-selection-book data/selection/rizabanadohido-value-gen10.jsonl.gz

# The hidden-bench bridge: gives 20,352 recorded games a zero. Spelled like the corpus.
run_match anchor-gen11Lx2-book9-hidden-vs-hpshare-m2 \
  --games 848 --seed 20261004 --served --hide-bench \
  --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
  -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
     --selection-book data/selection/rizabanadohido-value-gen9.jsonl.gz \
     --baseline-uniform-selection

# The same anchor with the ensemble's own book: the configuration that would ship.
run_match anchor-gen11Lx2-ownbook-hidden-vs-hpshare-m2 \
  --games 848 --seed 20261005 --served --hide-bench \
  --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
  -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
     --selection-book data/selection/rizabanadohido-value-gen11L-ens2.jsonl.gz \
     --baseline-uniform-selection

# What the borrowed book costs, as one match rather than a difference of two anchors.
run_match gen11Lx2-ownbook-vs-gen9book-hidden-m2 \
  --games 848 --seed 20261006 --served --hide-bench \
  --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
  --baseline data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-gen11L-ens2.jsonl.gz \
     --baseline-selection-book data/selection/rizabanadohido-value-gen9.jsonl.gz

# The open ladder. Reference -- the search sees the opponent's four -- and the only axis
# that reaches back to 2026-09-12, whose conditions were checked against it by commit
# timestamp. One seed per row: rows that share a draw stop the corpus gaining independent
# opponents as rows are added, and the rating pools every match ever recorded.
LADDER_SEED=20261010
for M in value-gen234 value-gen2345 value-gen8 value-gen9 value-gen10 value-gen11h; do
  LADDER_SEED=$((LADDER_SEED + 1))
  run_match "anchor-$M-vs-hpshare-m2" \
    --games 848 --seed "$LADDER_SEED" --served --uniform-selection \
    --value "data/models/$M.pt" \
    -- --objective hp-share --limit 24 --baseline-limit 24
done
echo "=== 1 done ==="
date

# ------------------------------------------------ 2. the twin pool: one difference
#
# The control was gen8 + gen9, and it differs from the hidden pool in three things:
# temperature (0.05 against half of it at 0.5), candidate ordering (gen11L is the only
# pool ever generated with --rank-leaf), and the bench. An experiment that wants to
# attribute a result to the bench cannot use it.
#
# So: the same seed, the same leaf, the same book, the same width, the same ordering, the
# same exploration -- and the bench open. Seed 2001 means game i draws the same team and
# the same selection as its hidden twin, as far as the draw is concerned; whether hiding
# the bench also consumes randomness is NOT assumed here, because the same assumption
# about --mirror-share is what desynchronised generation 11h from generation 10.
OUT=data/selfplay-open2
if [ ! -f "$OUT/DONE" ]; then
  echo "=== 2: the open twin, 12,000 games, seed 2001 ==="
  date
  uv run --group learn python -u tools/generate_queue.py \
    --out "$OUT" --games 12000 --seed 2001 --served --limit 24 \
    --value data/models/value-gen11L.pt \
    -- --rank-leaf --explore-temperature 0.5 --explore-epsilon 0.25 2>&1 | tail -8
  echo "ok: 12000 OPEN games, twin of selfplay-hidden2, seed 2001, --rank-leaf, e=0.25 T=0.5" \
    > "$OUT/DONE"
fi
echo "=== 2 done ==="
date

# ------------------------------------------------------------------ 3. three models
#
# `h12` and `o12` are the experiment: twin pools, one difference, so whichever way it
# comes out the comparison is about the bench. `hidden24` is the deliverable -- a model
# built purely under the hidden rule, on everything hidden that exists -- and it is not
# the control, because its pool is twice the size and mixes two exploration temperatures.
train_one() {
  local name="$1"; shift
  [ -f "data/models/$name.pt" ] && { echo "skip $name"; return; }
  echo "=== 3: encode + train $name ==="
  date
  uv run --group learn python -u tools/encode_dataset.py \
    --dir "$@" --out "data/$name-encoded.npz" 2>&1 | tail -6
  uv run --group learn python -u tools/train_value.py \
    --data "data/$name-encoded.npz" --out "data/models/$name.pt" \
    --split-seed 0 2>&1 | tail -14
}
train_one value-h12 data/selfplay-hidden2
train_one value-o12 data/selfplay-open2
train_one value-hidden24 data/selfplay-gen11L data/selfplay-hidden2
echo "=== 3 done ==="
date

# --------------------------------------- 3b. the twins, in the condition that ships
#
# Uniform selection on BOTH arms, acknowledged. Neither twin has a book and solving two
# would cost seventy minutes to answer a question about the training condition, not about
# selection; both arms are the agent (model, uniform), which is a fair pair.
run_match h12-vs-o12-hidden \
  --games 848 --seed 20261041 --served --hide-bench --uniform-selection \
  --value data/models/value-h12.pt --baseline data/models/value-o12.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf

# The same pair in the OPEN game, which is the other half of the question: a model taught
# in the dark should lose less by being put in the light than one taught in the light
# loses by being put in the dark.
run_match h12-vs-o12-open \
  --games 848 --seed 20261042 --served --uniform-selection \
  --value data/models/value-h12.pt --baseline data/models/value-o12.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf

# ------------------------------------------- 3c. the deliverable, on the hidden scale
B="data/selection/rizabanadohido-value-hidden24.jsonl.gz"
if [ ! -f "$B" ]; then
  echo "=== 3c: selection book for value-hidden24 ==="
  date
  MODEL=data/models/value-hidden24.pt LOGS=data/selection/logs-value-hidden24 \
    bash tools/solve_book_parallel.sh 2>&1 | tail -6
fi
run_match anchor-value-hidden24-hidden-vs-hpshare \
  --games 848 --seed 20261043 --served --hide-bench \
  --value data/models/value-hidden24.pt \
  -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
     --selection-book "$B" --baseline-uniform-selection
echo "=== 3c done ==="
date

# ---------------------------------------------------------------- 4. the ordering panel
echo "=== 4: ordering panel, 12 opponents ==="
date
bash "$ORDER"
echo "=== 4 done ==="
date
uv run --group learn python tools/ordering_result.py 2>&1 | tail -30
echo "=== run3 done ==="
date
