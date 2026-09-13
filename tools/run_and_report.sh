#!/usr/bin/env bash
# Runs a long job and prints one line when it ends, whatever the ending.
#
# Three times in one night a job finished with nothing armed to notice. Once a match died
# on a TypeError and wrote zero games; once a 97-second generation finished and the machine
# then sat idle for three hours; once a 1,696-game match ran unwatched because the launch
# command had grown long enough that arming the watch got forgotten after it.
#
# The common cause is that starting and watching were two steps, and the second one is the
# one that goes missing. This makes them one. The final line is written to stdout *and* to
# `$OUT_DIR/DONE`, so a watcher can tail one file rather than guess at process names.
#
# Exit status is the job's. A job that fails says so in the line.
#
#   bash tools/run_and_report.sh data/matches/foo "bash tools/genmatch_parallel.sh"
set -uo pipefail

OUT_DIR="${1:?usage: run_and_report.sh OUT_DIR COMMAND...}"
shift
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/DONE"

started=$(date +%s)
"$@"
status=$?
elapsed=$(($(date +%s) - started))

# What the job produced, in the units this project actually counts.
# Both layouts: a fixed split names its files by seed, a queued run by worker. Counting
# only one of them reported a 107-game run as zero games, from the very tool written to
# stop an empty run being called a success.
games=$(cat "$OUT_DIR"/games-*.jsonl 2>/dev/null | wc -l)
games=${games:-0}
errors=$(grep -lE "Traceback" "$OUT_DIR"/*.log 2>/dev/null | wc -l)
errors=${errors:-0}

verdict="ok"
if [ "$status" -ne 0 ]; then
	verdict="FAILED (exit $status)"
elif [ "$errors" -gt 0 ]; then
	verdict="FAILED ($errors log(s) with a traceback)"
elif [ "$games" -eq 0 ]; then
	# The failure mode that cost a match: workers exit cleanly having written nothing.
	verdict="FAILED (no games written)"
fi

line="$(basename "$OUT_DIR"): $verdict, $games games, $((elapsed / 60))m$((elapsed % 60))s"
echo "$line"
echo "$line" > "$OUT_DIR/DONE"
exit "$status"
