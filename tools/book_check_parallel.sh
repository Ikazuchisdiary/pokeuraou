#!/usr/bin/env bash
# The selection book's advice, measured against the whole field, on every core.
#
# Four arms x GAMES matchups x 2 seats, paired on the opponent AND on the seat, at roughly
# 7s a game: 280 matchups is 2,240 battles, which is four hours in one process and twenty
# minutes in fourteen. It was half that before every matchup was played in both seats; the
# second seat is what takes the seat term out of the calibration line, and `tools/seats.py`
# says why that line could not be read without it.
#
#   bash tools/book_check_parallel.sh
#   GAMES=560 bash tools/book_check_parallel.sh
set -uo pipefail

MODEL="${MODEL:-data/models/value-gen2.pt}"
ROSTER="${ROSTER:-rizabanadohido}"
WORKERS="${WORKERS:-8}"
GAMES="${GAMES:-280}"
LIMIT="${LIMIT:-16}"
SEED="${SEED:-11}"
EPSILON="${EPSILON:-0.25}"
# 0.5, matching generate_parallel.sh. It was 0.05 here, which only moves the gen/gen arm
# -- but that arm exists to be "what generation actually plays", and the coverage table
# gives the thinnest selection 0.0 games in a generation at 0.05 against 11.9 at 0.5. An
# arm that answers for a setting nobody runs answers nothing.
TEMPERATURE="${TEMPERATURE:-0.5}"
# The port, which this launcher never asked for: without it every arm runs the Python
# resolver at about fifteen times the cost.
RUST_NODE="${RUST_NODE:-1}"
# RANK=leaf orders the narrowing by the leaf, which is what generation does. Empty keeps
# the damage ordering every board number in the record was taken with.
RANK="${RANK:-}"
DEVICE="${DEVICE:-cuda}"
# Which book, separately from which model plays. They default together -- a book solved by
# a model is the one that model's name finds -- and the interesting comparison breaks that
# pairing on purpose: the same model playing two books says what re-solving bought, where
# changing both at once says only that something changed.
BOOK="${BOOK:-}"
book_args=()
if [ -n "$BOOK" ]; then
	book_args=(--book "$BOOK")
fi
OUT="${OUT:-data/matches/book-check.jsonl}"
LOGS="${LOGS:-data/matches/logs}"

mkdir -p "$LOGS" "$(dirname "$OUT")"
echo "book check: $WORKERS shards x $GAMES matchups x 4 arms x 2 seats, leaf $MODEL, eps=$EPSILON T=$TEMPERATURE"
echo "  device $DEVICE, rust node $RUST_NODE, width $LIMIT"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	# MODEL is deliberately unquoted: it may name several nets to average as one leaf,
	# and that leaf has to be the one that solved the book being checked.
	# shellcheck disable=SC2086
	POKEURAOU_RUST_NODE="$RUST_NODE" uv run --group learn python tools/book_check.py \
		--roster "$ROSTER" --model $MODEL \
		${RANK:+--rank-by-leaf} \
		"${book_args[@]}" \
		--games "$GAMES" --limit "$LIMIT" --seed "$SEED" \
		--epsilon "$EPSILON" --temperature "$TEMPERATURE" \
		--shard "$i" --shards "$WORKERS" \
		--out "$OUT" --device "$DEVICE" --torch-threads 1 \
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
