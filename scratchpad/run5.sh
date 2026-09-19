set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou
until grep -q "=== G done ===" "$G4OUT"; do sleep 60; done

# gen11L's open anchor, after the fix.
#
# The ladder runs gen234 through gen11h and NOT gen11L, because gen11L's anchor was taken
# this morning -- before the menu and replacement-node repairs. Six post-fix rows and one
# pre-fix row is not a curve. The hp-share arm's own replacements are now chosen by
# hp-share rather than by the model, and the first ladder row shows what that is worth:
# gen234 reads +63.4 today against the +77.4 recorded on 2026-09-12.
#
# Same settings as every other ladder row, and its own seed.
OUT=data/matches/anchor-value-gen11L-vs-hpshare-m2
if ! grep -q "^ok" "$OUT/DONE" 2>/dev/null; then
  echo "=== H: value-gen11L vs hp-share, open, after the fix ==="
  date
  uv run --group learn python -u tools/match_queue.py \
    --out "$OUT" --games 848 --seed 20261017 --served --uniform-selection \
    --value data/models/value-gen11L.pt \
    -- --objective hp-share --limit 24 --baseline-limit 24 2>&1 | tail -8
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "ok: value-gen11L vs hp-share, w24, damage, uniform, open, seed 20261017" \
      > "$OUT/DONE"
  else
    echo "FAILED (exit $status)" > "$OUT/DONE"
  fi
fi
uv run --group learn python tools/match_result.py "$OUT" 2>&1 | tail -14
uv run --group learn python tools/paired_result.py "$OUT" 2>&1 | tail -6
echo "=== H done ==="
date
uv run --group learn python tools/ratings.py 2>&1 | head -50
echo "=== run5 done ==="
date
