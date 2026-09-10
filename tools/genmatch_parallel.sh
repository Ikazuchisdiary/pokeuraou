#!/usr/bin/env bash
# One generation's leaf against another's, on every core, pooled by tools/pool_matches.py.
#
# This is the only fair comparison between generations: validation AUC cannot make it,
# because each generation plays differently and so creates a different distribution of
# positions to be judged. Both seats are played -- side 0 always holds our roster and side 1
# always a tournament team, so swapping which *generation* sits where is the only way to
# separate the roster's edge R from the new generation's gain a:
#
#   seat A: roster+new vs field+old  -> R + a
#   seat B: roster+old vs field+new  -> (1 - R) + a
#   mean                             -> 0.5 + a
#
# R comes back out of the difference, and it has been measured independently at 60.9-62.1%
# -- so it doubles as a check on the setup rather than only a nuisance parameter.
#
# Model against model costs about twice a normal game: with different leaves the two sides
# stop solving one game and each solves its own matrix.
#
#   NEW=data/models/value-gen3.pt OLD=data/models/value-gen2.pt bash tools/genmatch_parallel.sh
set -uo pipefail

NEW="${NEW:-data/models/value-gen3.pt}"
OLD="${OLD:-data/models/value-gen2.pt}"
ROSTER="${ROSTER:-rizabanadohido}"
WORKERS="${WORKERS:-14}"
GAMES="${GAMES:-50}"
LIMIT="${LIMIT:-16}"
FIRST_SEED="${FIRST_SEED:-880}"
OUT_DIR="${OUT_DIR:-data/matches/genmatch-$(basename "$NEW" .pt)-vs-$(basename "$OLD" .pt)}"

mkdir -p "$OUT_DIR"
echo "generation match: $(basename "$NEW") vs $(basename "$OLD")"
echo "  $WORKERS workers x $GAMES games per seat, search limit $LIMIT"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	seed=$((FIRST_SEED + i))
	uv run --group learn python tools/generation_match.py \
		--roster "$ROSTER" --value "$NEW" --baseline "$OLD" \
		--games "$GAMES" --limit "$LIMIT" --seed "$seed" \
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
