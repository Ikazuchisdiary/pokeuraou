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
# Every game is also written out with its provenance. A match is thousands of real games
# played to a real outcome, and keeping only the win rate throws away training data that
# has already been paid for -- an afternoon of these comes to about a tenth of a
# generation. They are not self-play and the provenance block says so per side, so a
# dataset builder can include, exclude or reweight them and the effect can be measured.
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
# The leaf, the width and the candidate ranking are all part of what an agent *is*, and
# the rating fit reads them back out of the recorded provenance. A match run with a
# different ranking than the generation it is meant to describe is a different agent
# wearing the same name, so these are knobs rather than constants.
DEVICE="${DEVICE:-cpu}"
# Set apart, because the two arms are allowed to differ and the interesting matches do.
# An agent already on the rating scale has a fixed configuration; to attach a new one to
# the scale, the new arm plays the way it generates and the old arm plays the way it was
# rated, mismatched on purpose.
RANK_LEAF="${RANK_LEAF:-0}"
OLD_RANK_LEAF="${OLD_RANK_LEAF:-$RANK_LEAF}"
RUST_NODE="${RUST_NODE:-0}"
rank_args=()
if [ "$RANK_LEAF" != "0" ]; then
	rank_args+=(--rank-leaf)
fi
if [ "$OLD_RANK_LEAF" != "0" ]; then
	rank_args+=(--baseline-rank-leaf)
fi

mkdir -p "$OUT_DIR"
echo "generation match: $(basename "$NEW") vs $(basename "$OLD")"
echo "  $WORKERS workers x $GAMES games per seat, search limit $LIMIT"
echo "  device $DEVICE, rust node $RUST_NODE, leaf ranking $RANK_LEAF (baseline $OLD_RANK_LEAF)"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	seed=$((FIRST_SEED + i))
	POKEURAOU_RUST_NODE="$RUST_NODE" uv run --group learn python tools/generation_match.py \
		--roster "$ROSTER" --value "$NEW" --baseline "$OLD" \
		--games "$GAMES" --limit "$LIMIT" --seed "$seed" \
		"${rank_args[@]}" \
		--open-bench \
		--out "$OUT_DIR/seed$seed.jsonl" \
		--games-out "$OUT_DIR/games-seed$seed.jsonl" \
		--device "$DEVICE" --torch-threads 1 \
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
