#!/bin/sh
# IKA-66 の2番目 — 等しい壁時計で、幅24 と幅48 のプールを1本ずつ作る。
#
#   sh tools/ika66_generate.sh <A腕の局数> <B腕の局数>
#
# `--limit` 以外は1文字も変えない。逐次に回す: 同時に回すと2腕が機械を分け合い、
# 「等しい壁時計」という独立変数そのものが壊れる。
# 局は添字から種を取るので、両腕とも添字 0 から始めれば局 i は同じ6匹・同じ選出になる。
set -e
cd "$(dirname "$0")/.."

A_GAMES="$1"
B_GAMES="$2"
[ -n "$A_GAMES" ] && [ -n "$B_GAMES" ] || { echo "usage: $0 <A games> <B games>" >&2; exit 2; }

run_arm() {
  W="$1"; N="$2"; OUT="data/ika66/w$W"
  echo "=== arm width $W : $N games -> $OUT ==="
  rm -rf "$OUT"
  START=$(date +%s)
  uv run --group learn python tools/generate_queue.py \
    --out "$OUT" --games "$N" --seed 6601 --served --servers 2 --workers 24 \
    --limit "$W" --value data/models/value-gen11L.pt \
    --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz \
    --hide-bench -- --rank-leaf 2>&1
  END=$(date +%s)
  echo "ARM w$W WALL $((END-START))s FOR $N GAMES"
}

run_arm 24 "$A_GAMES"
run_arm 48 "$B_GAMES"
echo "GENERATE DONE"
