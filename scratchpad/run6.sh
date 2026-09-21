set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou

# Overnight, after the ordering panel. Everything here waits for an idle machine first:
# running two 24-worker jobs at once killed three workers with 0xC0000374 this afternoon.
#
# The occupancy pattern lists `selection_check` too. run5's did not, so it started while
# the panel was running -- the second time in one day that an enumerated list of process
# names was short by one. An enumeration is the wrong shape for "is anything running",
# and the right one is a shared function with every name in it, which is at least one
# place to fix instead of three.
wait_for_machine() {
  local quiet=0 n
  echo "--- waiting for an idle machine ($(date +%H:%M))"
  while true; do
    n=$(powershell -NoProfile -Command \
      "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'match_queue|generate_queue|solve_selection_book|selection_check|train_value|encode_dataset' } | Measure-Object).Count" \
      2>/dev/null | tr -d '\r ')
    # The query's own command line matches the pattern, so it counts itself.
    if [ "${n:-9}" -le 2 ]; then
      quiet=$((quiet + 1))
      [ "$quiet" -ge 2 ] && break
    else
      quiet=0
    fi
    sleep 60
  done
  echo "--- machine free ($(date +%H:%M))"
}

run_match() {
  local name="$1"; shift
  local out="data/matches/$name"
  if grep -q "^ok" "$out/DONE" 2>/dev/null; then echo "skip $name (ok)"; return; fi
  # A failed run left its games behind, and `open_games` appends -- re-running into the
  # same directory would write every game twice and the duplicate check would then be
  # reporting a bug this script created. Moved aside, not deleted: they are real games.
  if [ -d "$out" ]; then
    mv "$out" "${out}-crashed-$(date +%H%M%S)"
    echo "  moved the crashed run aside"
  fi
  wait_for_machine
  echo "=== match: $name ==="
  date
  uv run --group learn python -u tools/match_queue.py --out "$out" "$@" 2>&1 | tail -8
  local status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "ok: $name, rerun clean after the afternoon's contention" > "$out/DONE"
  else
    echo "FAILED (exit $status): $name" > "$out/DONE"
    echo "!! $name exited $status"
  fi
  uv run --group learn python tools/match_result.py "$out" 2>&1 | tail -16
  uv run --group learn python tools/paired_result.py "$out" 2>&1 | tail -8
}

# ------------------------------------------------- 1. the two runs that lost workers
# Both are usable as they stand -- every game is on disk and the pairs are complete --
# but both are marked FAILED, and a number whose run says it failed should not be the
# one that goes in the record.
run_match h12-vs-o12-hidden \
  --games 848 --seed 20261041 --served --hide-bench --uniform-selection \
  --value data/models/value-h12.pt --baseline data/models/value-o12.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf

run_match gen11L-vs-gen10-ownroster-uniform \
  --games 848 --seed 20261052 --served --uniform-selection \
  --value data/models/value-gen11L.pt --baseline data/models/value-gen10.pt \
  -- --limit 24 --baseline-limit 24 --rank-leaf --baseline-rank-leaf

# ----------------------------------- 2. the generalisation test with a real difference
#
# The first one used gen11L against gen10, whose difference is about zero on our six --
# so it could only show that "no difference" generalises. gen234 against gen10 is +43 Elo
# on our six and not in doubt. If THAT survives the change of team, the fixed six is a
# defensible simplification; if it does not, the roster has to become an axis.
run_match gen234-vs-gen10-place1roster \
  --games 848 --seed 20261061 --served --uniform-selection \
  --value data/models/value-gen234.pt --baseline data/models/value-gen10.pt \
  -- --roster place1 --limit 24 --baseline-limit 24

run_match gen234-vs-gen10-ownroster \
  --games 848 --seed 20261062 --served --uniform-selection \
  --value data/models/value-gen234.pt --baseline data/models/value-gen10.pt \
  -- --limit 24 --baseline-limit 24

# --------------------------- 3. what it costs that the opponent can see our spreads
# No games. Two solves per opponent, and the gap between them is the price of the
# assumption the user named this morning.
wait_for_machine
echo "=== blind opponent ==="
date
uv run --group learn python -u tools/blind_opponent.py \
  --model data/models/value-gen11L.pt data/models/value-gen11L-s1.pt \
  --teams 8 2>&1 | tail -24
echo "=== run6 done ==="
date
