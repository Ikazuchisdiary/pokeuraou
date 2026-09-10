#!/usr/bin/env bash
# One candidate width against another, on every core, pooled by tools/pool_matches.py.
#
# `width_match.py` plays both seats in one process and appends a JSON line per seat, so
# parallelism is independent seeds pooled afterwards -- the same shape as the generation
# match. Pooling the counts rather than averaging the rates matters: a run that discarded
# more games must not weigh the same as one that discarded fewer.
#
# Cost scales with the *wider* side, because one seat pays it every game. Measured per
# process on this machine with the gen-2 leaf on CPU: 9.2 s/game at 16, 13.3 at 24, 32.5
# at 32. Budget accordingly -- 48 is roughly eight times 16.
#
#   WIDE=24 NARROW=16 GAMES=40 bash tools/width_match_parallel.sh
set -uo pipefail

MODEL="${MODEL:-data/models/value-gen2.pt}"
ROSTER="${ROSTER:-rizabanadohido}"
WORKERS="${WORKERS:-14}"
WIDE="${WIDE:-24}"
NARROW="${NARROW:-16}"
GAMES="${GAMES:-40}"
FIRST_SEED="${FIRST_SEED:-770}"
OUT_DIR="${OUT_DIR:-data/matches/width-$WIDE-vs-$NARROW}"

mkdir -p "$OUT_DIR"
echo "width match: $WIDE vs $NARROW, $WORKERS workers x $GAMES games per seat, leaf $MODEL"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	seed=$((FIRST_SEED + i))
	uv run --group learn python tools/width_match.py \
		--roster "$ROSTER" --value "$MODEL" \
		--wide "$WIDE" --narrow "$NARROW" \
		--games "$GAMES" --seed "$seed" \
		--out "$OUT_DIR/seed$seed.jsonl" \
		--device cpu --torch-threads 1 \
		>"$OUT_DIR/seed$seed.log" 2>&1 &
	pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
	wait "$pid" || failed=$((failed + 1))
done
elapsed=$(($(date +%s) - started))
echo "done in $((elapsed / 60))m$((elapsed % 60))s, $failed worker(s) failed"

uv run python tools/pool_matches.py --dir "$OUT_DIR"
