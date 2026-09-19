set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou

# Does the generation ordering survive a different six?
#
# Every number this project has is conditional on `rizabanadohido`. The anchor, the
# ladder, the +141 the book is worth -- all of it is "this agent, playing that team", and
# nothing has ever asked whether gen11L beats gen10 IN GENERAL. `configs/teams/` held one
# file until today.
#
# No new training and no new generation: the same two models, on place 1's six, against
# the same field. If the ordering flips, the fixed-six scale does not generalise and the
# roster has to become an axis. If it holds, the fixed six is a defensible simplification
# for now, which is worth knowing before paying for the alternative.
#
# Uniform selection on both arms, acknowledged: there is no solved book for place1 and
# solving one costs 35 minutes to answer a question that is not about selection. Both arms
# are the agent (model, uniform), which is a fair pair -- and it matches the open ladder,
# which is also uniform.
#
# Waits for the twin pool rather than the ladder, because two 24-worker jobs on eight
# cores is slower than either alone; a fifteen-minute match is not worth slowing a
# two-and-a-half-hour generation for.
until grep -q "=== 2 done ===" "$RUNOUT"; do sleep 60; done
echo "=== G: does the ordering survive another six? ==="
date

OUT=data/matches/gen11L-vs-gen10-place1roster
if ! grep -q "^ok" "$OUT/DONE" 2>/dev/null; then
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20261051 --served --uniform-selection \
    --value data/models/value-gen11L.pt --baseline data/models/value-gen10.pt \
    -- --roster place1 --limit 24 --baseline-limit 24 \
       --rank-leaf --baseline-rank-leaf 2>&1 | tail -8
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "ok: gen11L vs gen10 on place1's six, uniform selection, seed 20261051" > "$OUT/DONE"
  else
    echo "FAILED (exit $status)" > "$OUT/DONE"
  fi
fi
uv run --group learn python tools/match_result.py "$OUT" 2>&1 | tail -16
uv run --group learn python tools/paired_result.py "$OUT" 2>&1 | tail -8

# The same pair on OUR six with the same settings, so the two rows differ in one thing.
# The open ladder's own rows are against hp-share, not against each other, so neither of
# them answers this.
OUT=data/matches/gen11L-vs-gen10-ownroster-uniform
if ! grep -q "^ok" "$OUT/DONE" 2>/dev/null; then
  echo "=== G2: the same pair on our six, same settings ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20261052 --served --uniform-selection \
    --value data/models/value-gen11L.pt --baseline data/models/value-gen10.pt \
    -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf 2>&1 | tail -8
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "ok: gen11L vs gen10 on our six, uniform selection, seed 20261052" > "$OUT/DONE"
  else
    echo "FAILED (exit $status)" > "$OUT/DONE"
  fi
fi
uv run --group learn python tools/match_result.py "$OUT" 2>&1 | tail -16
uv run --group learn python tools/paired_result.py "$OUT" 2>&1 | tail -8
echo "=== G done ==="
date
