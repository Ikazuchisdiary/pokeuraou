#!/usr/bin/env bash
# The selection book's advice, measured against the whole field, on every core.
#
# Four arms x GAMES games, paired on the opponent, at roughly 7s a game: 280 games is 1,120
# battles, which is two hours in one process and ten minutes in fourteen.
#
#   bash tools/book_check_parallel.sh
#   GAMES=560 bash tools/book_check_parallel.sh
set -uo pipefail

MODEL="${MODEL:-data/models/value-gen2.pt}"
ROSTER="${ROSTER:-rizabanadohido}"
WORKERS="${WORKERS:-14}"
GAMES="${GAMES:-280}"
LIMIT="${LIMIT:-16}"
SEED="${SEED:-11}"
EPSILON="${EPSILON:-0.25}"
TEMPERATURE="${TEMPERATURE:-0.05}"
OUT="${OUT:-data/matches/book-check.jsonl}"
LOGS="${LOGS:-data/matches/logs}"

mkdir -p "$LOGS" "$(dirname "$OUT")"
echo "book check: $WORKERS shards x $GAMES games x 4 arms, leaf $MODEL, eps=$EPSILON T=$TEMPERATURE"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	uv run --group learn python tools/book_check.py \
		--roster "$ROSTER" --model "$MODEL" \
		--games "$GAMES" --limit "$LIMIT" --seed "$SEED" \
		--epsilon "$EPSILON" --temperature "$TEMPERATURE" \
		--shard "$i" --shards "$WORKERS" \
		--out "$OUT" --device cpu --torch-threads 1 \
		>"$LOGS/check$i.log" 2>&1 &
	pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
	wait "$pid" || failed=$((failed + 1))
done
elapsed=$(($(date +%s) - started))
echo "arms done in $((elapsed / 60))m$((elapsed % 60))s, $failed worker(s) failed"

uv run --group learn python tools/book_check.py --merge --out "$OUT"
