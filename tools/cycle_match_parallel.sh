#!/usr/bin/env bash
# The mirror's reported three-way cycle, played rather than evaluated.
#
# 3 edges x GAMES games x 2 seats at roughly 7s a game: 210 games is 1,260 battles, which
# is two and a half hours in one process and ten minutes in fourteen.
#
#   bash tools/cycle_match_parallel.sh
#   GAMES=420 bash tools/cycle_match_parallel.sh
set -uo pipefail

MODEL="${MODEL:-data/models/value-gen2.pt}"
ROSTER="${ROSTER:-rizabanadohido}"
WORKERS="${WORKERS:-14}"
GAMES="${GAMES:-210}"
LIMIT="${LIMIT:-16}"
SEED="${SEED:-23}"
OUT="${OUT:-data/matches/cycle-match.jsonl}"
LOGS="${LOGS:-data/matches/logs}"
TEXT="${TEXT:-data/matches/cycle-match.txt}"

mkdir -p "$LOGS" "$(dirname "$OUT")"
echo "cycle match: $WORKERS shards x $GAMES games x 3 edges x 2 seats, leaf $MODEL"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	uv run --group learn python tools/cycle_match.py \
		--roster "$ROSTER" --model "$MODEL" \
		--games "$GAMES" --limit "$LIMIT" --seed "$SEED" \
		--shard "$i" --shards "$WORKERS" \
		--out "$OUT" --device cpu --torch-threads 1 \
		>"$LOGS/cycle$i.log" 2>&1 &
	pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
	wait "$pid" || failed=$((failed + 1))
done
elapsed=$(($(date +%s) - started))
echo "edges done in $((elapsed / 60))m$((elapsed % 60))s, $failed worker(s) failed"

uv run --group learn python tools/cycle_match.py --merge --out "$OUT" --text "$TEXT"
