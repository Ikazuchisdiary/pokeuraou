#!/usr/bin/env bash
# IKA-99 / IKA-100: the timing table, every arm interleaved, minimum of REPS passes.
#
#     PYTHON=<interpreter> rust/experiments/ika99_bench.sh <out-dir> <turns.json> <games-dir> \
#         [reps] [positions] [cases.json ...]
#
# <out-dir> is what ika99_build.sh and pgo.sh were given: the four target-cpu / integer arms
# at <out-dir>/pokeuraou-damage-<arm>.exe and the two PGO arms under <out-dir>/pgo/. An arm
# whose binary is missing is left out. The node positions are the ones after the 36 that
# pgo.sh trained on. Idle machine only: run it under the machine lock.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?usage: ika99_bench.sh <out-dir> <turns.json> <games-dir> [reps] [positions] [cases ...]}"
turns="${2:?turns.json}"
games="${3:?games-dir}"
reps="${4:-5}"
positions="${5:-48}"
shift 3
[ "$#" -gt 0 ] && shift
[ "$#" -gt 0 ] && shift
python="${PYTHON:-python}"

arms=()
for arm in base v3 native introll; do
    bin="$out/pokeuraou-damage-$arm.exe"
    [ -e "$bin" ] && arms+=(--arm "$arm=$bin")
done
for arm in llvm pgo; do
    bin="$out/pgo/pokeuraou-damage-$arm.exe"
    [ -e "$bin" ] && arms+=(--arm "$arm=$bin")
done
cases=()
for fixture in "$@"; do
    cases+=(--cases "$fixture")
done

PYTHONPATH="$repo/src" "$python" "$repo/tools/bench_ika99.py" bench "${arms[@]}" "${cases[@]}" \
    --turns "$turns" --games-dir "$games" --positions "$positions" --train-positions 36 \
    --reps "$reps" --turn-repeats 20 --damage-repeats 20
