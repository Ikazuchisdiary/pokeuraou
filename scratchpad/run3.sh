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

# -------------------------------------------- 2. hidden-only, and a control of equal size
if [ ! -f data/models/value-hidden24.pt ]; then
  echo "=== 2a: encode + train hidden-only ==="
  date
  uv run --group learn python -u tools/encode_dataset.py \
    --dir data/selfplay-gen11L data/selfplay-hidden2 \
    --out data/selfplay-hidden24-encoded.npz 2>&1 | tail -6
  uv run --group learn python -u tools/train_value.py \
    --data data/selfplay-hidden24-encoded.npz \
    --out data/models/value-hidden24.pt --split-seed 0 2>&1 | tail -14
fi
echo "=== 2a done ==="
date

# gen9 alone is 11,999 and gen8+gen9 is 18,999; with gen7 it is 30,982. The hidden pool is
# about 24,000, so gen8+gen9+gen7 overshoots and gen8+gen9 undershoots. Taking the smaller
# one states the direction: the control has FEWER games, so a hidden model that loses
# cannot blame its pool size, and one that wins is not settled by this comparison alone.
if [ ! -f data/models/value-open19.pt ]; then
  echo "=== 2b: encode + train open control ==="
  date
  uv run --group learn python -u tools/encode_dataset.py \
    --dir data/selfplay-gen8 data/selfplay-gen9 \
    --out data/selfplay-open19-encoded.npz 2>&1 | tail -6
  uv run --group learn python -u tools/train_value.py \
    --data data/selfplay-open19-encoded.npz \
    --out data/models/value-open19.pt --split-seed 0 2>&1 | tail -14
fi
echo "=== 2b done ==="
date

# ------------------------------------------------------- 3. a book each, then the match
for M in value-hidden24 value-open19; do
  B="data/selection/rizabanadohido-$M.jsonl.gz"
  [ -f "$B" ] && { echo "skip book for $M"; continue; }
  echo "=== 3a: selection book for $M ==="
  date
  MODEL="data/models/$M.pt" LOGS="data/selection/logs-$M" \
    bash tools/solve_book_parallel.sh 2>&1 | tail -6
done
echo "=== 3a done ==="
date

run_match hidden24-vs-open19 \
  --games 848 --seed 20261021 --served --hide-bench \
  --value data/models/value-hidden24.pt --baseline data/models/value-open19.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf \
     --selection-book data/selection/rizabanadohido-value-hidden24.jsonl.gz \
     --baseline-selection-book data/selection/rizabanadohido-value-open19.jsonl.gz

S=20261030
for M in value-hidden24 value-open19; do
  S=$((S + 1))
  run_match "anchor-$M-hidden-vs-hpshare" \
    --games 848 --seed "$S" --served --hide-bench \
    --value "data/models/$M.pt" \
    -- --objective hp-share --limit 24 --baseline-limit 24 --rank-leaf \
       --selection-book "data/selection/rizabanadohido-$M.jsonl.gz" \
       --baseline-uniform-selection
done
echo "=== 3 done ==="
date

# ---------------------------------------------------------------- 4. the ordering panel
echo "=== 4: ordering panel, 16 opponents ==="
date
bash "$ORDER"
echo "=== 4 done ==="
date
uv run --group learn python tools/ordering_result.py 2>&1 | tail -30
echo "=== run3 done ==="
date
