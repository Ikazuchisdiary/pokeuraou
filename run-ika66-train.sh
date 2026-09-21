#!/bin/sh
# IKA-66 の2番目 —— 3腕を符号化して、それぞれ2シードで学習させる。
#
#   sh run-ika66-train.sh <C腕の局数 = B腕の局数>
#
# 学習設定は3腕とも既定のまま同一（30エポック・batch 1024・lr 2e-3・holdout 0.15・
# split-seed 0）。最良エポックの重みが保存されるので、局数の違う腕がそれぞれの最適で
# 止まる。動かすのは `--seed` だけ（0 と 1）—— 同じ設定の6シードが保留AUC 0.024 に散り、
# 設定差は 0.004 なので、単独1本はシード運を測ってしまう。
set -e
cd /c/Users/Ikazuchi/repos/pokeuraou

B_GAMES="$1"
[ -n "$B_GAMES" ] || { echo "usage: $0 <B arm game count>" >&2; exit 2; }

echo "=== C腕を切り出す（生成なし） ==="
uv run python .claude/worktrees/ika-66-width48-pool/run-ika66-truncate.py \
  --src data/ika66/w24 --out data/ika66/w24trunc --games "$B_GAMES"

for ARM in w24 w48 w24trunc; do
  echo "=== encode $ARM ==="
  uv run --group learn python tools/encode_dataset.py \
    --dir "data/ika66/$ARM" --out "data/ika66/$ARM-encoded.npz" 2>&1 | tail -20
done

for ARM in w24 w48 w24trunc; do
  for S in 0 1; do
    if [ "$S" = "0" ]; then OUT="data/models/ika66-$ARM.pt"; else OUT="data/models/ika66-$ARM-s$S.pt"; fi
    echo "=== train $ARM seed $S -> $OUT ==="
    uv run --group learn python tools/train_value.py \
      --data "data/ika66/$ARM-encoded.npz" --out "$OUT" --seed "$S" \
      2>&1 | tail -18
  done
done
echo "TRAIN DONE"
