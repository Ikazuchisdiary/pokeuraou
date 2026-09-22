#!/bin/sh
# IKA-66 の2番目 —— 3腕を等時間に揃えて切り出し、符号化して、それぞれ2シードで学習させる。
#
#   sh tools/ika66_train.sh <N_A> <N_B>
#
#     N_A  A腕（幅24）から取る局数   = 19800 * 等時間 / A腕の実測壁時計
#     N_B  B腕（幅48）から取る局数   =  8400 * 等時間 / B腕の実測壁時計
#            等時間 = min(A腕の壁時計, B腕の壁時計)
#
# 局数を注文どおりに使わず実測から切り直すのは、**独立変数が壁時計だから**。多くもらった
# 腕を添字で切る。局は添字から種を取るので、先頭 N 局は「N 局を注文していたら得られた
# プール」そのもので、これが正しい切り方になる。
#
#   A腕  幅24・N_A局（等時間）
#   B腕  幅48・N_B局（等時間）
#   C腕  幅24・N_B局（等局数、時間は 1/2.4）  ← 生成不要。A腕を切るだけ
#
# 学習設定は3腕とも既定のまま同一（30エポック・batch 1024・lr 2e-3・holdout 0.15・
# split-seed 0）。最良エポックの重みが保存されるので、局数の違う腕がそれぞれの最適で
# 止まる。動かすのは `--seed` だけ（0 と 1）—— 同じ設定の6シードが保留AUC 0.024 に散り、
# 設定差は 0.004 なので、単独1本はシード運を測ってしまう。
set -e
cd "$(dirname "$0")/.."

N_A="$1"; N_B="$2"
[ -n "$N_A" ] && [ -n "$N_B" ] || { echo "usage: $0 <N_A> <N_B>" >&2; exit 2; }
TRUNC=tools/ika66_truncate.py

echo "=== 等時間に揃えて切り出す ==="
uv run python "$TRUNC" --src data/ika66/w24 --out data/ika66/armA --games "$N_A"
uv run python "$TRUNC" --src data/ika66/w48 --out data/ika66/armB --games "$N_B"
uv run python "$TRUNC" --src data/ika66/w24 --out data/ika66/armC --games "$N_B"

for ARM in armA armB armC; do
  echo "=== encode $ARM ==="
  uv run --group learn python tools/encode_dataset.py \
    --dir "data/ika66/$ARM" --out "data/ika66/$ARM-encoded.npz" 2>&1 | tail -22
done

for ARM in armA armB armC; do
  for S in 0 1; do
    if [ "$S" = "0" ]; then OUT="data/models/ika66-$ARM.pt"; else OUT="data/models/ika66-$ARM-s$S.pt"; fi
    echo "=== train $ARM seed $S -> $OUT ==="
    uv run --group learn python tools/train_value.py \
      --data "data/ika66/$ARM-encoded.npz" --out "$OUT" --seed "$S" 2>&1 | tail -18
  done
done
echo "TRAIN DONE"
