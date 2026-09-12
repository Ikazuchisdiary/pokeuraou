#!/usr/bin/env bash
# Generation 2 data: one process per worker, because generation is the whole bottleneck.
#
# 8 workers on the GPU, which reverses two settings this script used to argue for at
# length. Both arguments were correct when they were made and both were invalidated by the
# same event: the resolver moved to Rust and got twenty times faster, so every share
# measured against it changed.
#
#   "the forward pass is 1.3% of a game, and 1.3% cannot pay for a 1 GB CUDA context per
#   worker" -- that 1.3% was of a game whose resolver was Python. Re-measured against the
#   port: width 48, cpu, torch is 35.4% of a process; on cuda it is 9.4% and the process
#   takes 4.85 s instead of 10.68. Whole machine, width 48, --rank-leaf, idle: cuda at 8
#   workers 263.8 games/min against cpu at 24 workers 146.3. The nine-context cap is real
#   and no longer binds, because nine beats twenty-four.
#
#   "14 workers, not 16; 16 collapse to 80.2 games/min" -- also pre-port. A worker is much
#   lighter now and the cpu curve no longer collapses at 16 or 24. On cuda the knee is 8
#   to 9, which is where the contexts run out anyway.
#
# --torch-threads 1 stays. torch defaults to a machine-sized pool per process, so without
# the cap eight processes spawn a hundred threads onto eight cores.
#
# Two things that are *not* the machine, measured separately: a game's cost varies
# eight-fold (1.77 to 13.75 s/game measured strictly serially on an idle machine), because
# a position needing an exact budget expands one matrix cell into many leaves; and the
# cuda-over-cpu margin depends on which games are in the mix, since the GPU wins the cheap
# games by about 2x and the CPU wins the dear ones by about 14%. A benchmark on a light
# seed therefore flatters cuda and one on a heavy seed flatters cpu. Neither ever says cpu
# is the better default.
#
#   bash tools/generate_parallel.sh data/selfplay-gen2 1000 601
set -uo pipefail

OUT_DIR="${1:-data/selfplay-gen2}"
GAMES_PER_WORKER="${2:-1000}"
FIRST_SEED="${3:-601}"
WORKERS="${WORKERS:-8}"
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
# `RUST_NODE` defaults to 1. It defaulted to 0, which meant a run launched without the
# variable quietly used the Python resolver and paid twenty times over for it -- a default
# that costs 20x when you forget it is not a default, it is a trap.
RUST_NODE="${RUST_NODE:-1}"
DEVICE="${DEVICE:-cuda}"
RANK_LEAF="${RANK_LEAF:-0}"
# Prove the equilibrium from the fifth of the matrix that settles it, instead of filling
# all of it. Not an approximation: the round ends by asking both sides for a better reply
# over *every* action and getting none, which is a proof about the cells that were never
# solved (exploitability 7.7e-08). It does pick a different vertex of a degenerate optimum
# in six nodes out of eight, and that is what was measured rather than assumed: 2,544
# games, same model and width on both sides, 49.8% [47.9, 51.8], a = -0.2.
#
# Off by default, because the speed-up it is traded against is 1.14x, not the 1.94x first
# reported. Measured properly -- same seeds, generation's own shape, one flag: 3.62 s/game
# against 4.13. The 1.94x came from comparing a sparse match against a *depth-2* match,
# which is not a like-for-like baseline; that was an error in the comparison, not in the
# solver. At 1.14x the gain is 0.19 of a doubling, worth +0.3 to +0.7 points, which is
# inside the interval of the -0.2 it costs. Turn it on when the boundary gets cheaper:
# skipping 80% of the cells buys only 12% because the rounds each pay a round trip, and
# the round trip is what is expensive.
SOLVE_SPARSELY="${SOLVE_SPARSELY:-0}"
rank_args=()
if [ "$RANK_LEAF" != "0" ]; then
	rank_args=(--rank-leaf)
fi
if [ "$SOLVE_SPARSELY" != "0" ]; then
	rank_args+=(--solve-sparsely)
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
