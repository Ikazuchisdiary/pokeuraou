#!/usr/bin/env bash
# Generation 2 data: one process per worker, because generation is the whole bottleneck.
#
# 14 workers, not 16. Measured on this machine (8 physical cores / 16 threads,
# tools/bench_scaling.py): 14 workers give 112.3 games/min and 16 collapse to 80.2, because
# sixteen workers on sixteen logical threads leave nothing for the OS. "workers = logical
# cores" is the wrong rule here.
#
# --device cpu --torch-threads 1 is deliberate. CPU inference is only 8% slower per process
# and removes the ~1 GB CUDA context per worker that caps a 12 GB card at nine, and the GPU
# has almost nothing to win: re-measured at width 24 with the learned leaf
# (tools/profile_generation.py), the forward pass is 1.3% of a game and building the batch
# in Python is ten times that. Moving 1.3% to a card cannot pay for the contexts.
# And torch defaults to a machine-sized thread pool per process, so without the thread cap
# fourteen processes spawn a hundred threads onto eight cores and the run ends up slower
# than a serial one.
#
#   bash tools/generate_parallel.sh data/selfplay-gen2 1000 601
set -uo pipefail

OUT_DIR="${1:-data/selfplay-gen2}"
GAMES_PER_WORKER="${2:-1000}"
FIRST_SEED="${3:-601}"
WORKERS="${WORKERS:-14}"
VALUE="${VALUE:-data/models/value-worlds.pt}"
LIMIT="${LIMIT:-16}"
# Optional: draw both sides' 4-of-6 from cached selection equilibria instead of uniformly.
# BOOK=data/selection/rizabanadohido-value-gen2.jsonl.gz bash tools/generate_parallel.sh ...
BOOK="${BOOK:-}"
EPSILON="${EPSILON:-0.25}"
# 0.5, not 0.05: the coverage table over all 394 solved teams says the rarest of our 90
# selections gets 0.0 games in a generation at 0.05 and 11.9 at 0.5, with three quarters of
# games still on the equilibrium either way. A setting that leaves a selection with no games
# recreates a failure this project already had.
TEMPERATURE="${TEMPERATURE:-0.5}"
# Games played against our own six, spreads included. The mirror is the one case with
# knowledge from outside the model (configs/knowledge/), and no generation had ever
# contained a single mirror game -- so every mirror answer was extrapolation. It also
# asserts 50%: a true mirror is antisymmetric, so its win rate is a calibration check on
# search, resolver and evaluator together.
MIRROR_SHARE="${MIRROR_SHARE:-0.1}"
# Order candidates by the leaf rather than by expected damage. The two disagree, and the
# disagreement is measured: the menu drops the leaf's preferred reply in 37% of decisions,
# playing the dropped one instead is worth +14.3 points where they disagree, and the wider
# menu that contains them wins whole games by +5.5. Costs about 1.4x per decision.
# The Rust node fills the payoff matrix and encodes the leaves; only the forward pass
# stays in torch. Measured here, identical games either way, learned leaf at width 24:
# 8.33 s/game in Python, 0.84 with the node on the CPU, 0.51 on the GPU.
#
# `DEVICE` matters now in a way it did not before. The comment above about the GPU having
# nothing to win was true when it was written: with the resolver and the encoder in Python
# the forward pass was 1.3% of a game. With both of those in Rust it is most of what is
# left, and the balance turns over.
RUST_NODE="${RUST_NODE:-0}"
DEVICE="${DEVICE:-cpu}"
RANK_LEAF="${RANK_LEAF:-0}"
rank_args=()
if [ "$RANK_LEAF" != "0" ]; then
	rank_args=(--rank-leaf)
fi
book_args=()
if [ -n "$BOOK" ]; then
	book_args=(--selection-book "$BOOK" --explore-epsilon "$EPSILON" --explore-temperature "$TEMPERATURE")
fi

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

echo "generation: $WORKERS workers x $GAMES_PER_WORKER games -> $OUT_DIR"
echo "  leaf: $VALUE ($DEVICE, 1 torch thread), search limit $LIMIT"
if [ "$RUST_NODE" != "0" ]; then
	echo "  node: the Rust port (POKEURAOU_RUST_NODE=1)"
else
	echo "  node: Python"
fi
if [ -n "$BOOK" ]; then
	echo "  selection: $BOOK (eps=$EPSILON, T=$TEMPERATURE) -- both sides drawn from the equilibrium"
else
	echo "  selection: uniform 4-of-6 on both sides"
fi
echo "  mirror share: $MIRROR_SHARE (its win rate must come out at 50%)"
if [ "$RANK_LEAF" != "0" ]; then
	echo "  candidate ranking: the leaf"
else
	echo "  candidate ranking: expected damage"
fi
echo "  seeds $FIRST_SEED..$((FIRST_SEED + WORKERS - 1))"
started=$(date +%s)

pids=()
for i in $(seq 0 $((WORKERS - 1))); do
	seed=$((FIRST_SEED + i))
	POKEURAOU_RUST_NODE="$RUST_NODE" uv run --group learn python tools/selfplay.py \
		--games "$GAMES_PER_WORKER" \
		--seed "$seed" \
		--limit "$LIMIT" \
		--opponents worlds \
		--value "$VALUE" \
		--device "$DEVICE" \
		--torch-threads 1 \
		--mirror-share "$MIRROR_SHARE" \
		"${rank_args[@]}" \
		"${book_args[@]}" \
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
