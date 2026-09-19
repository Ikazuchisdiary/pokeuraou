set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou

# Wait for the MACHINE, not for a marker.
#
# Gating on another script's end marker put two 24-worker matches and four inference
# servers on eight cores at once, and three workers died of 0xC0000374 -- heap corruption,
# which is what this box does when it runs out. The run survived: every one of its 1,696
# games is on disk, the queue replayed the three it lost and wrote no duplicates. It cost
# 57 minutes instead of 25 and left a directory marked FAILED.
#
# A marker says "that script finished". It does not say "nothing else is running", and
# the chains were never meant to interleave. So this one waits until no match, no
# generation and no book solve is running anywhere, and then takes its turn.
busy() {
  powershell -NoProfile -Command \
    "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'match_queue|generate_queue|solve_selection_book|train_value' } | Measure-Object).Count" \
    2>/dev/null | tr -d '\r '
}
echo "=== H: waiting for the machine ==="
date
quiet=0
while true; do
  n=$(busy)
  # The query's own command line contains the pattern, so it counts itself: anything at
  # or under two is an idle machine. Two consecutive quiet samples, because a chain
  # between two steps is briefly idle and is not free.
  if [ "${n:-9}" -le 2 ]; then
    quiet=$((quiet + 1))
    [ "$quiet" -ge 2 ] && break
  else
    quiet=0
  fi
  sleep 60
done
echo "=== H: machine free, starting ==="
date

# gen11L's open anchor, after the fix. The ladder runs gen234 through gen11h and not
# gen11L, whose anchor was taken before the menu and replacement-node repairs -- six
# post-fix rows and one before is not a curve. gen234 reads +63.4 today against the
# +77.4 recorded on 2026-09-12, so the repair reaches this comparison even though no
# menu does: the hp-share arm's replacements used to be chosen by the value function.
OUT=data/matches/anchor-value-gen11L-vs-hpshare-m2
if ! grep -q "^ok" "$OUT/DONE" 2>/dev/null; then
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

# The two cases a person adjudicated, against the two generations whose ordering is in
# question. No games: it solves one position per case.
for M in "data/models/value-gen11L.pt data/models/value-gen11L-s1.pt" \
         "data/models/value-gen10.pt" ; do
  echo "--- human baseline: $M"
  uv run --group learn python tools/human_baseline.py --value $M 2>&1 | tail -14
done
echo "=== run5 done ==="
date
