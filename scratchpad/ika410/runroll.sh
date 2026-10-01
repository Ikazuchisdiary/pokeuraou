#!/bin/bash
N=$1
export PYTHONPATH=C:/tmp/ika410/eng/src
export POKEURAOU_RUST_NODE_BIN=C:/tmp/ika401/wt/rust/target/release/pokeuraou-damage.exe
PY=C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe
cd C:/tmp/ika410
for i in $(seq 0 $((N-1))); do
  $PY rollout2.py picks.jsonl games.jsonl 150 $i $N roll/r$i.jsonl 2> roll/r$i.log &
done
wait
echo done
