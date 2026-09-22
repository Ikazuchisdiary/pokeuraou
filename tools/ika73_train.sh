#!/bin/sh
# IKA-73 —— 狭い幅の腕を2通りに切り出し、符号化して、それぞれ2シードで学習させる。
#
#   sh tools/ika73_train.sh <幅> <N_equal_time> <N_equal_games>
#
#     armD   幅W・等時間ぶん   = 実測レート × armA の壁時計 6,022秒
#     armDg  幅W・等局数ぶん   = 19,800局（armA と同じ）
#
# 基準の armA（幅24・19,800局）は IKA-66 のものをそのまま使う —— 木は動いたが
# `sources` が変わったのは docstring だけで、armA の局 0〜23 を今日の木で引き直すと
# 24/24 が完全一致した（同じ検査は幅16 では 0/24 に落ちるので、検査自体は効いている）。
#
# 学習設定は IKA-66 の3腕と同一（既定のまま、--seed 0 と 1 だけ動かす）。
set -e
cd "$(dirname "$0")/.."

W="$1"; NT="$2"; NG="$3"
[ -n "$W" ] && [ -n "$NT" ] && [ -n "$NG" ] || { echo "usage: $0 <width> <N equal-time> <N equal-games>" >&2; exit 2; }

echo "=== 切り出す ==="
uv run python tools/ika66_truncate.py --src "data/ika73/w$W" --out data/ika73/armD  --games "$NT"
uv run python tools/ika66_truncate.py --src "data/ika73/w$W" --out data/ika73/armDg --games "$NG"

for ARM in armD armDg; do
  echo "=== encode $ARM ==="
  uv run --group learn python tools/encode_dataset.py \
    --dir "data/ika73/$ARM" --out "data/ika73/$ARM-encoded.npz" 2>&1 | tail -22
done

for ARM in armD armDg; do
  for S in 0 1; do
    if [ "$S" = "0" ]; then OUT="data/models/ika73-$ARM.pt"; else OUT="data/models/ika73-$ARM-s$S.pt"; fi
    echo "=== train $ARM seed $S -> $OUT ==="
    uv run --group learn python tools/train_value.py \
      --data "data/ika73/$ARM-encoded.npz" --out "$OUT" --seed "$S" 2>&1 | tail -6
  done
done
echo "TRAIN DONE"
