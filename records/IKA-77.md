# IKA-77: M-C gen-0 を生成した —— 40,000 局中 39,998 局（2 局は決着せず捨てた）、68.7 分、port の拒否 0

2026-09-25（9/24 23:41〜9/25 00:50）。調整役のセッション。これより前の IKA-77 の記録は TODO.md「9/23 — IKA-77」「9/24 — IKA-77」と
records/2026-09-24-coordination.md（試走のやり直し）にある。

## コマンド

`data/selfplay-mc0/CMD` に書いたとおり（TODO.md「9/24 — IKA-77」§5 と同じ）。master 1adf369（IKA-257 を取り込んだ後）、
exe と sim-bridge の dist はその master で作り直した。始める前に動いていた python は 0 本。16 コアを専有した（heavy.py `--agent IKA-77`）。

```
generate_queue.py --out data/selfplay-mc0 --games 40000 --seed 7701 --served --servers 2 --workers 24 --limit 12
  --value data/models/value-gen11L.pt --pool regmc-matchupweb -- --rank-leaf
```

* 24 本のワーカーのログには、すべて `pool vs pool / gen9championsvgc2026regmc / search 12x12 / leaf value:value-gen11L / selection solved (eps=0.25, T=0.5) / bench hidden` が出ていた
* 推論サーバ 2 本は、どちらも `cuda waits: blocking sync` だった

## 結果

```
壁時計           68.7 分（見積もり 75〜85 分）。全対が解けた後は約 680 局/分（朝の試走の r600 は 554 局/分）
局               39,998 / 40,000。欠けたのは 18763・30238 の 2 局で、ワーカーが「決着しなかった局」として捨てた
                 （discarded unfinished 1 が 2 本）。局は種で決まるので、打ち直しても同じになる
選出の解         2,145 / 2,145 対（selection-solved/）
大きさ           4.6 GB
記録             selectionSource solved・information hidden-bench・endReason wipeout・finalPosition が全局にある
                 engine commit 1adf369、dirty false が全局
ミラー           1,258 局（3.15%、期待値 3.03%）
席0 の勝ち       20,101 / 39,998 = 50.3%
長さ             9.46 ターン、11.69 決定（1 局あたり）
port の拒否      refusal_replay 2,000 決定・285,144 セルで 0
```

注記（局の割合）は試走とほぼ同じ:
* 同速の残差 68.5%、こおりの解凍 33.4%、のろわれボディ 25.5%、途中交代の選択 23.0%、ねむりのターン数 19.3%、アンコールとふいうち 13.4%
* 未実装の変化技: ソウルビート 7.5%・ふういん 5.1%・じこあんじ 2.7%（IKA-256）、スキルスワップ 2.6%（IKA-255）、さいきのいのり 2.8%（IKA-221）。
  ユーザーの判断で、注記つきのまま本番に入った
* 追加効果の確率は枝にしない（各 0.2〜5.4%）。M-B でも同じ形

## 機械の使われ方（ユーザーの指摘、IKA-258）

本番の途中（0:06・0:24）で測った。CPU は 100%、GPU は 50% 前後だった（以前の生成は GPU 90% 前後）。選出の解がすべて済んだ後も変わらない。
開始から 43 分のプロセスごとの累計 CPU 時間は、Python のワーカー 18,761 s（51%）、port 14,094 s（39%）、推論サーバ 3,620 s（10%）。
GPU は CPU 側を待っている。内訳の計測は IKA-258。

## 次

gen-0 の学習（ユーザーの判断、9/24）: gen-0 の記録で「温間（value-gen11L から）・ゼロから × 今のレシピ・SWA 系」を検証の数字で比べる。
上位を盤（`match_queue.py --sprt 0 10`、M-C のプールで両席、null の対戦を先に）で対戦させる。IKA-82 の「温間 対 ゼロから」と、
IKA-86 の盤の確認を兼ねる。GPU を専有するので、生成とは並べない。
