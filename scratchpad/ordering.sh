# 12 opponents with a second selection carrying >= 5%, drawn with seed 20260919
# each pair is paired per game: same opponent, same spread class, same index
set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou
DIR=data/matches/ordering
mkdir -p "$DIR/logs"

# place 121: 57.4% vs 42.6%
out="$DIR/place121.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 121 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,garchomp,toxapex,incineroar" --force-selection "charizard,sylveon,garchomp,toxapex" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/121-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 121 --merge --out "$out" 2>&1 | tail -6
}

# place 170: 92.0% vs 8.0%
out="$DIR/place170.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 170 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,garchomp,venusaur,toxapex" --force-selection "toxapex,incineroar,venusaur,garchomp" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/170-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 170 --merge --out "$out" 2>&1 | tail -6
}

# place 353: 79.7% vs 20.3%
out="$DIR/place353.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 353 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,sylveon,venusaur,incineroar" --force-selection "charizard,garchomp,venusaur,toxapex" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/353-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 353 --merge --out "$out" 2>&1 | tail -6
}

# place 173: 73.0% vs 27.0%
out="$DIR/place173.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 173 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,garchomp,toxapex,incineroar" --force-selection "charizard,toxapex,garchomp,incineroar" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/173-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 173 --merge --out "$out" 2>&1 | tail -6
}

# place 277: 66.8% vs 24.6%
out="$DIR/place277.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 277 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,garchomp,toxapex,incineroar" --force-selection "charizard,toxapex,venusaur,garchomp" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/277-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 277 --merge --out "$out" 2>&1 | tail -6
}

# place 284: 61.8% vs 30.2%
out="$DIR/place284.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 284 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,garchomp,toxapex,incineroar" --force-selection "charizard,toxapex,venusaur,garchomp" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/284-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 284 --merge --out "$out" 2>&1 | tail -6
}

# place 202: 46.4% vs 34.8%
out="$DIR/place202.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 202 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,garchomp,venusaur,toxapex" --force-selection "charizard,garchomp,toxapex,incineroar" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/202-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 202 --merge --out "$out" 2>&1 | tail -6
}

# place 361: 39.0% vs 34.6%
out="$DIR/place361.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 361 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,sylveon,venusaur,incineroar" --force-selection "charizard,venusaur,garchomp,incineroar" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/361-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 361 --merge --out "$out" 2>&1 | tail -6
}

# place 82: 72.0% vs 28.0%
out="$DIR/place82.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 82 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,sylveon,venusaur,garchomp" --force-selection "charizard,sylveon,garchomp,toxapex" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/82-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 82 --merge --out "$out" 2>&1 | tail -6
}

# place 137: 75.1% vs 24.9%
out="$DIR/place137.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 137 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,garchomp,venusaur,toxapex" --force-selection "charizard,garchomp,toxapex,incineroar" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/137-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 137 --merge --out "$out" 2>&1 | tail -6
}

# place 37: 76.1% vs 17.7%
out="$DIR/place37.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 37 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,garchomp,toxapex,incineroar" --force-selection "charizard,sylveon,venusaur,garchomp" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/37-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 37 --merge --out "$out" 2>&1 | tail -6
}

# place 189: 69.3% vs 30.7%
out="$DIR/place189.jsonl"
[ -f "$out" ] || {
  pids=()
  for i in $(seq 0 7); do
    uv run --group learn python -u tools/selection_check.py \
      --place 189 --model data/models/value-gen11L.pt \
      --games 60 --limit 16 --classes 8 --hide-bench --rank-by-leaf \
      --force-selection "charizard,sylveon,venusaur,garchomp" --force-selection "charizard,sylveon,venusaur,incineroar" \
      --only-arm "+" \
      --shard "$i" --shards 8 --out "$out" \
      > "$DIR/logs/189-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  uv run --group learn python -u tools/selection_check.py --place 189 --merge --out "$out" 2>&1 | tail -6
}

echo "=== ordering panel done ==="
