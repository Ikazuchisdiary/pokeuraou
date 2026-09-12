#!/usr/bin/env bash
# Endgame data, one process per worker.
#
# The pool reaches turn 15 in 5.4% of its games and turn 20 in 0.7%, because self-play
# ends at 9.4 turns on average and nothing truncates it. This resamples that tail: each
# worker starts games from recorded decisions at or past --from-turn and plays them out.
#
# Seeds differ per worker so the draws differ; the *positions* are the same pool, which is
# the point -- the tail is thin and gets sampled more than once on purpose.
#
#   FROM_TURN=12 GAMES=250 bash tools/resume_parallel.sh data/selfplay-resume8
set -uo pipefail

OUT_DIR="${1:-data/selfplay-resume8}"
GAMES="${2:-250}"
FIRST_SEED="${3:-20001}"
SRC="${SRC:-data/selfplay-gen8}"
WORKERS="${WORKERS:-8}"
VALUE="${VALUE:-data/models/value-all.pt}"
LIMIT="${LIMIT:-48}"
FROM_TURN="${FROM_TURN:-12}"
# Bounded on both sides. The first unbounded run spread its budget over every band the
# pool has, out to turn 75, and put 27% of its games past turn 70 against 22% in the
# 15-25 range the human plans live in. Turn-70 doubles is a degenerate stall.
TO_TURN="${TO_TURN:-25}"
RANK_LEAF="${RANK_LEAF:-1}"
DEVICE="${DEVICE:-cuda}"
RUST_NODE="${RUST_NODE:-1}"
rank_args=()
if [ "$RANK_LEAF" != "0" ]; then
	rank_args=(--rank-leaf)
fi

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"
echo "resume generation: $WORKERS workers x $GAMES games -> $OUT_DIR"
echo "  starts: $SRC, turn $FROM_TURN-$TO_TURN"
echo "  leaf: $VALUE ($DEVICE), width $LIMIT, rust node $RUST_NODE"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	seed=$((FIRST_SEED + i))
	POKEURAOU_RUST_NODE="$RUST_NODE" uv run --group learn python tools/resume_generate.py \
		--dir $SRC \
		--from-turn "$FROM_TURN" \
		--to-turn "$TO_TURN" \
		--games "$GAMES" \
		--value "$VALUE" \
		--limit "$LIMIT" \
		"${rank_args[@]}" \
		--seed "$seed" \
		--device "$DEVICE" \
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
echo "done in $((elapsed / 60))m$((elapsed % 60))s: $total games, $failed worker(s) failed"
