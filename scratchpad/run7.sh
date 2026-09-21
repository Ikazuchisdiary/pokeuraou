set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou
SP="/c/Users/Ikazuchi/AppData/Local/Temp/claude/C--Users-Ikazuchi-repos-pokeuraou/5db79e9c-c9bf-45e4-9093-199d908435df/scratchpad"

# Does the Mega fix remove the hidden-bench search's optimism? (G30)
#
# The twin pools say the search is +9.9 points optimistic at turn 1 under --hide-bench
# and +1.0 without it, same model, out of sample. Both were generated before the fix that
# stopped `completions` from putting a second Charizard on the bench while the first was
# on the field -- 40% of the belief on an opponent weaker than the real one, which is the
# right direction to produce exactly this.
#
# 600 games is enough: the twin measurement had n=2,034 at turn 1 for +-2.4 points, and
# 600 gives one turn-1 decision per game, so about +-4 points against an effect of 10.
#
# The wait is the same shared function as run6's, and it counts run6's own workers, so
# this cannot start on top of it. `why_action` is deliberately NOT in the pattern: putting
# it there is how the watcher got killed along with the job it was watching at 15:41.
wait_for_machine() {
  local quiet=0 n
  echo "--- waiting for an idle machine ($(date +%H:%M))"
  while true; do
    n=$(powershell -NoProfile -Command \
      "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' } | Measure-Object).Count" \
      2>/dev/null | tr -d '\r ')
    if [ "${n:-99}" -le 1 ]; then
      quiet=$((quiet + 1))
      [ "$quiet" -ge 2 ] && break
    else
      quiet=0
    fi
    sleep 60
  done
  echo "--- machine free ($(date +%H:%M))"
}

OUT=data/selfplay-hidden-fixed
if [ -d "$OUT" ]; then
  echo "!! $OUT exists; move it aside first so generation does not append"
  exit 1
fi

wait_for_machine
echo "=== generating 600 hidden-bench games under the fixed hidden.py ==="
date
# `--limit` and `--hide-bench` are the DRIVER's own flags; only what the driver does not
# name goes after `--`. Checked against --help rather than assumed, and smoke-tested at
# two games, because a flag that lands on the wrong side of `--` produces a run that
# looks fine and answers a different question.
uv run --group learn python -u tools/generate_queue.py \
  --out "$OUT" --games 600 --seed 7001 --limit 24 --hide-bench \
  --value data/models/value-gen11L.pt \
  -- --rank-leaf 2>&1 | tail -8
echo "=== generation exit $? ($(date +%H:%M)) ==="

echo "=== search optimism, fixed code ==="
uv run --group learn python -u tools/leaf_calibration.py \
  --value data/models/value-gen11L.pt --dir "$OUT" --positions 6000 2>&1 | tail -12
echo
echo "for comparison, the same measurement before the fix:"
echo "  hidden2 turn1 srch-act +9.9%   open2 turn1 srch-act +1.0%"
echo "=== run7 done ==="
date
