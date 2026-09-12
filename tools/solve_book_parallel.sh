#!/usr/bin/env bash
# Selection equilibria for the whole field, one process per shard.
#
# The solve is CPU-bound and single-threaded -- per class of 8,100 cells, building the
# positions costs 2.23s, encoding them 0.49s and the forward pass 0.77s -- so a single
# process leaves fifteen threads and the whole GPU idle. Sharded, the field takes minutes.
#
# 8 workers on the GPU, which reverses what this script used to say. "16 workers collapse"
# and "a CUDA context buys nothing when the forward pass is a fifth of the cost" were both
# measured before the resolver moved to Rust, and the port changed every share measured
# against it: at width 48 torch is 35.4% of a cpu process and 9.4% of a cuda one, and
# whole-machine throughput is 307.9 games/min on cuda against 190.6 on cpu. The same
# correction was applied to generate_parallel.sh.
#
# RUST_NODE defaults to 1 here too. It was not passed at all, which is the trap rather
# than the default: a run launched without it pays about fifteen times over in silence.
#
# Spread classes are seeded per team, so the merged book does not depend on the shard count.
#
#   bash tools/solve_book_parallel.sh
#   CLASSES=16 WORKERS=14 bash tools/solve_book_parallel.sh
set -uo pipefail

MODEL="${MODEL:-data/models/value-all.pt}"
ROSTER="${ROSTER:-rizabanadohido}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
RUST_NODE="${RUST_NODE:-1}"
CLASSES="${CLASSES:-8}"
POOL="${POOL:-all}"
SEED="${SEED:-0}"
LOGS="${LOGS:-data/selection/logs}"

mkdir -p "$LOGS"
echo "selection book: $WORKERS shards x $CLASSES spread classes, pool $POOL, leaf $MODEL"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	POKEURAOU_RUST_NODE="$RUST_NODE" uv run --group learn python tools/solve_selection_book.py \
		--roster "$ROSTER" \
		--model "$MODEL" \
		--classes "$CLASSES" \
		--pool "$POOL" \
		--seed "$SEED" \
		--shard "$i" \
		--shards "$WORKERS" \
		--device "$DEVICE" \
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
