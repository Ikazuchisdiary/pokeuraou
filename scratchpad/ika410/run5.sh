#!/bin/bash
# IKA-410 stage 2: N shards in parallel (N = cores granted), then the negative control.
N=$1
export PYTHONPATH=C:/tmp/ika410/wt/src
export POKEURAOU_RUST_NODE_BIN=C:/tmp/ika406/wt/rust/target/release/pokeuraou-damage.exe
PY=C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe
cd C:/tmp/ika410
for i in $(seq 0 $((N-1))); do
  $PY stage2.py examples.jsonl games.jsonl $i $N out5/s$i.jsonl 2> out5/s$i.log &
done
wait
$PY stage2.py smoke_ex.jsonl neg_games.jsonl 0 1 out5/neg.jsonl 2> out5/neg.log
$PY stage2.py smoke_ex.jsonl smoke_games.jsonl 0 1 out5/smoke.jsonl 2> out5/smoke.log
echo done
