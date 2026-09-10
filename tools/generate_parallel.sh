#!/usr/bin/env bash
# Generation 2 data: one process per worker, because generation is the whole bottleneck.
#
# 14 workers, not 16. Measured on this machine (8 physical cores / 16 threads,
# tools/bench_scaling.py): 14 workers give 112.3 games/min and 16 collapse to 80.2, because
# sixteen workers on sixteen logical threads leave nothing for the OS. "workers = logical
# cores" is the wrong rule here.
#
# --device cpu --torch-threads 1 is deliberate. CPU inference is only 8% slower per process
# and removes the ~1 GB CUDA context per worker that caps a 12 GB card at nine; the forward
# pass is 3% of the per-leaf cost, so the GPU has almost nothing to win. And torch defaults
# to a machine-sized thread pool per process, so without the thread cap fourteen processes
# spawn a hundred threads onto eight cores and the run ends up slower than a serial one.
#
#   bash tools/generate_parallel.sh data/selfplay-gen2 1000 601
set -uo pipefail

OUT_DIR="${1:-data/selfplay-gen2}"
GAMES_PER_WORKER="${2:-1000}"
FIRST_SEED="${3:-601}"
WORKERS="${WORKERS:-14}"
VALUE="${VALUE:-data/models/value-worlds.pt}"
LIMIT="${LIMIT:-16}"

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

echo "generation: $WORKERS workers x $GAMES_PER_WORKER games -> $OUT_DIR"
echo "  leaf: $VALUE (cpu, 1 torch thread), search limit $LIMIT"
echo "  seeds $FIRST_SEED..$((FIRST_SEED + WORKERS - 1))"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	seed=$((FIRST_SEED + i))
	uv run --group learn python tools/selfplay.py \
		--games "$GAMES_PER_WORKER" \
		--seed "$seed" \
		--limit "$LIMIT" \
		--opponents worlds \
		--value "$VALUE" \
		--device cpu \
		--torch-threads 1 \
		--out "$OUT_DIR/games-seed$seed.jsonl" \
		>"$OUT_DIR/logs/seed$seed.log" 2>&1 &
	pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
	wait "$pid" || failed=$((failed + 1))
done

elapsed=$(($(date +%s) - started))
total=$(cat "$OUT_DIR"/games-seed*.jsonl 2>/dev/null | wc -l)
echo "done in $((elapsed / 60))m$((elapsed % 60))s: $total games written, $failed worker(s) failed"
