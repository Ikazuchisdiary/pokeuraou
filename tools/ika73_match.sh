#!/bin/sh
# IKA-73 —— 腕どうしを打たせる。腕はモデル名の接頭辞で指定する（IKA-66 の腕とも混ぜられる）。
#
#   sh tools/ika73_match.sh <tested prefix> <baseline prefix> <games per seat> <name>
#   例: sh tools/ika73_match.sh ika73-armDg ika66-armA 6000 dg-vs-a
#
# 幅は両腕 24（現行の出荷幅）。差は葉だけ。book は1つを共有するので、両腕は同じ6匹・
# 同じ選出・同じ種で打ち、対応が残る。プールが隠蔽生成なので判定も `--hide-bench`。
set -e
cd /c/Users/Ikazuchi/repos/pokeuraou

TESTED="$1"; BASE="$2"; GAMES="$3"; NAME="$4"
[ -n "$NAME" ] || { echo "usage: $0 <tested prefix> <baseline prefix> <games per seat> <name>" >&2; exit 2; }
OUT="data/matches/ika73-$NAME"

START=$(date +%s)
uv run --group learn python tools/match_queue.py \
  --out "$OUT" --games "$GAMES" --served --servers 2 --workers 24 \
  --value "data/models/$TESTED.pt" "data/models/$TESTED-s1.pt" \
  --baseline "data/models/$BASE.pt" "data/models/$BASE-s1.pt" \
  --hide-bench \
  -- --limit 24 --rank-leaf --baseline-rank-leaf \
  --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz 2>&1
END=$(date +%s)
echo "MATCH $NAME WALL $((END-START))s"

echo "=== 勝率と条件（席ごとに出る） ==="
uv run python tools/match_result.py "$OUT" 2>&1
echo "=== 対の区間 ==="
uv run python tools/paired_result.py "$OUT" 2>&1
echo "=== 発火率 —— 設定が実際に手を変えたか ==="
uv run python tools/pair_divergence.py "$OUT" 2>&1
echo "MATCH DONE"
