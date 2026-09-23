#!/bin/sh
# IKA-68 —— 深さ2の戦略を「制限ゲームの均衡」として読む腕を、盤の上で測る。
#
#   sh tools/ika68_match.sh <mode> <games per seat> [name suffix]
#   mode  vs-d1     制限深さ2 対 深さ1        採用の判定（IKA-12 の 48.28% ±0.71 が基準線）
#         vs-mixed  制限深さ2 対 混合深さ2    修正そのものの判定（深さ1を挟まない直接の対）
#         null      深さ1 対 深さ1            帰無対照。発火率 0%・50.00% ±0.00 に戻ること
#         null-r    制限深さ2 対 制限深さ2    新しい読みそのものの帰無対照（決定的か）
#
# 条件は IKA-12 に合わせる: 葉 value-gen11L 2本アンサンブル、葉順位、幅は両腕24、
# book は葉と同じ `-ens2`、seed 77、公開情報（⚠ 深さ2は控え隠蔽と同時に使えない ——
# `belief_solve` に深さの引数が無く、`play_game` が例外を投げる）。
set -e
cd "$(dirname "$0")/.."

MODE="$1"; GAMES="$2"; SUFFIX="$3"
[ -n "$GAMES" ] || { echo "usage: $0 <vs-d1|vs-mixed|null|null-r> <games per seat> [suffix]" >&2; exit 2; }

case "$MODE" in
  vs-d1)    ARM="--depth 2 --solve-restricted --baseline-depth 1" ;;
  vs-mixed) ARM="--depth 2 --solve-restricted --baseline-depth 2" ;;
  null)     ARM="--depth 1 --baseline-depth 1" ;;
  null-r)   ARM="--depth 2 --solve-restricted --baseline-depth 2 --baseline-solve-restricted" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

NAME="ika68-$MODE${SUFFIX:+-$SUFFIX}"
OUT="data/matches/$NAME"
MODEL="data/models/value-gen11L.pt data/models/value-gen11L-s1.pt"

START=$(date +%s)
# shellcheck disable=SC2086
uv run --group learn python tools/match_queue.py \
  --out "$OUT" --games "$GAMES" --served --servers 2 --workers 24 \
  --value $MODEL --baseline $MODEL --open-bench \
  -- --limit 24 --rank-leaf --baseline-rank-leaf $ARM \
  --selection-book data/selection/rizabanadohido-value-gen11L-ens2.jsonl.gz 2>&1
END=$(date +%s)
echo "MATCH $NAME WALL $((END-START))s FOR $((2*GAMES)) GAMES"

echo "=== 勝率と条件（席ごとに出る） ==="
uv run python tools/match_result.py "$OUT" 2>&1
echo "=== 対の区間 ==="
uv run python tools/paired_result.py "$OUT" 2>&1
echo "=== 発火率 —— 設定が実際に手を変えたか ==="
uv run python tools/pair_divergence.py "$OUT" 2>&1
echo "MATCH DONE"
