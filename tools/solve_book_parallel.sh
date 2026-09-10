#!/usr/bin/env bash
# Selection equilibria for the whole field, one process per shard.
#
# The solve is CPU-bound and single-threaded -- per class of 8,100 cells, building the
# positions costs 2.23s, encoding them 0.49s and the forward pass 0.77s -- so a single
# process leaves fifteen threads and the whole GPU idle. Sharded, the field takes minutes.
#
# 14 workers, --device cpu, one torch thread each: the same measured setting as
# generate_parallel.sh, for the same reasons (16 workers collapse; a CUDA context per
# worker buys nothing when the forward pass is a fifth of the cost).
#
# Spread classes are seeded per team, so the merged book does not depend on the shard count.
#
#   bash tools/solve_book_parallel.sh
#   CLASSES=16 WORKERS=14 bash tools/solve_book_parallel.sh
set -uo pipefail

MODEL="${MODEL:-data/models/value-gen2.pt}"
ROSTER="${ROSTER:-rizabanadohido}"
WORKERS="${WORKERS:-14}"
CLASSES="${CLASSES:-8}"
POOL="${POOL:-all}"
SEED="${SEED:-0}"
LOGS="${LOGS:-data/selection/logs}"

mkdir -p "$LOGS"
echo "selection book: $WORKERS shards x $CLASSES spread classes, pool $POOL, leaf $MODEL"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	uv run --group learn python tools/solve_selection_book.py \
		--roster "$ROSTER" \
		--model "$MODEL" \
		--classes "$CLASSES" \
		--pool "$POOL" \
		--seed "$SEED" \
		--shard "$i" \
		--shards "$WORKERS" \
		--device cpu \
		>"$LOGS/shard$i.log" 2>&1 &
	pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
	wait "$pid" || failed=$((failed + 1))
done
elapsed=$(($(date +%s) - started))
echo "shards done in $((elapsed / 60))m$((elapsed % 60))s, $failed failed"

uv run --group learn python tools/solve_selection_book.py \
	--roster "$ROSTER" --model "$MODEL" --merge
