#!/bin/sh
# IKA-73 —— 狭い幅の腕を1本、armA と同じ壁時計で作る。
#
#   sh tools/ika73_generate.sh <幅> <局数>
#
# armA（幅24・19,800局・6,022秒、`data/ika66/w24`）がそのまま基準になるので、生成するのは
# この1本だけ。`--limit` 以外は armA と1文字も変えない —— 葉・book・控え隠蔽・葉順位・
# 2サーバ24ワーカー・run seed 6601・添字0から。局は添字から種を取るので、局 i は
# armA と同じ6匹・同じ選出になる。
#
# 局数は多めに頼んで、あとで実測の壁時計から添字で切って揃える（IKA-66 と同じ手順）。
set -e
cd /c/Users/Ikazuchi/repos/pokeuraou

W="$1"; N="$2"
[ -n "$W" ] && [ -n "$N" ] || { echo "usage: $0 <width> <games>" >&2; exit 2; }
OUT="data/ika73/w$W"

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
echo "GENERATE DONE"
