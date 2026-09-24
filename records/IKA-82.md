# IKA-82: M-C gen-0 では温間始動（value-gen11L から 2 エポック）がゼロからより強い —— 盤で x2 どうし 53.5%（Elo +24 [+8, +41]、SPRT H1）。原点 hp-share/w12 の上に +230、ゼロからは +207、value-gen11L は +151。出荷は value-mc0（温間 x2）

2026-09-25。ワーカー（IKA-82・IKA-86 を 1 つの手順で答える回）。master c6ec705、ブランチ `ika-82-mc0-training`。
これより前の IKA-82 の記録は TODO.md「9/23 — IKA-82」「9/24 — IKA-82」（語彙の延長と、温間 対 ゼロからの手順 1〜6）にある。
学習率スケジュールの側（IKA-86）は records/IKA-86.md。

## 1. 符号化

`tools/encode_dataset.py --dir $M/data/selfplay-mc0 --out $M/data/selfplay-mc0-encoded.npz`（この木の src、1 コア、130 秒）。

```
局             39,998（IKA-77 と一致）
決定（行）     467,575（IKA-77 の 11.69 決定/局 × 39,998 ≈ 467,577 と合う）
規則・語彙     gen9championsvgc2026regmc、指紋 c2557340f3ed460f、encoding_revision 2
席0 の勝ち     50.2%
葉・情報・選出  value:value-gen11L・hidden-bench・solved が全局。engine 545f964b20a41d87 が全局
1 手しか無い決定 86,548（18.5%）
名前の無い揮発  flinch 758・ragepowder 63・detect 39・spikyshield 4
ミラー         15,843 決定
大きさ         53 MB
```

## 2. 検証の比較

同じ shard、`--split-seed 0`・holdout 0.15（学習 397,433 行 / 検証 70,142 行・6,000 局）。各腕 `--seed` 0・1 の 2 本、`tools/train_value.py` で学習。
駆動は `C:/tmp/ikamc0/train_arms.py`、評価は `C:/tmp/ikamc0/eval_arms.py`（どちらもコミットしない）。
x2 = 2 本の logit の平均（served のアンサンブルと同じ）、|dp| = 2 本の確率の差の平均、差 = 2 本の差の絶対値。

```
  腕                                         単体 log loss       単体 AUC          x2 loss   x2 AUC   x2 t1 AUC  |dp|    エポック
                                             (s0 / s1)           (s0 / s1)
  生成した探索（参考）                        0.4842              0.8537                                0.6043(t1)
  value-gen11L そのまま（温間の epoch 0）      0.5357              0.8202                                0.5456(t1)
  S-A  ゼロから・今のレシピ                   0.4630 / 0.4620     0.8588 / 0.8566   0.4524   0.8634   0.6571     0.083   8 / 7（最良 ep4）
  S-B8 ゼロから・8ep SWA(0.25)                0.4616 / 0.4622     0.8574 / 0.8567   0.4602   0.8580   0.6372     0.033   8
  W3   温間・3ep lr 2.5e-4・last              0.4530 / 0.4530     0.8636 / 0.8638   0.4528   0.8638   0.6568     0.012   3
  W2   温間・2ep lr 5e-4・last                0.4514 / 0.4517     0.8648 / 0.8648   0.4510   0.8651   0.6602     0.019   2
  WA   温間・今のレシピ（--init-from だけ）    0.4538 / 0.4547     0.8621 / 0.8621   0.4504   0.8645   0.6565     0.051   6 / 6（最良 ep2）
  W4e  温間・4ep lr 5e-4・EMA                 0.4555 / 0.4564     0.8631 / 0.8627   0.4557   0.8630   0.6523     0.011   4
```

シード間の差は、どの腕も log loss 0.0011 以下・AUC 0.0022 以下。

検証損失の推移（seed 0、エポックごと）:

```
  S-A    0.4866 0.4788 0.4690 0.4630 0.4677 0.4696 0.4689 0.4729            （ep4 が最良、ep8 で早期終了）
  S-B8   0.4815 0.4678 0.4651 0.4675 0.4656 0.4724 0.4805 0.4906            （最後の重みは 0.49、SWA で 0.4616）
  WA     0.4586 0.4538 0.4628 0.4550 0.4714 0.4749                          （ep2 が最良）
  W2     0.4574 0.4514
  W3     0.4591 0.4520 0.4530
  W4e    0.4560 0.4519 0.4528 0.4568（生の重み。EMA の重みは 0.4555）
```

読めること:

* **温間の学習前（epoch 0）は 0.5357 / AUC 0.8202。** M-B の知識だけで、ゼロからの 1 エポック目（0.4866）より悪い。M-C の新しい種族・道具の行が 0 から始まることと、分布の違い
* **温間は 1 エポックでゼロからの最良を越える**（WA ep1 0.4586、W2 ep1 0.4574 に対し S-A の最良 0.4630）。単体どうしでは W2 が S-A より −0.011、S-B8 より −0.010
* **x2 どうしでは差が縮む。** S-A x2 0.4524 に対し W2 x2 0.4510（−0.0014）、WA x2 0.4504（−0.0020）。S-A は 2 本がよく違う（|dp| 0.083）のでアンサンブルの上積みが −0.010 と大きく、
  温間はほぼ同じ 2 本（|dp| 0.012〜0.019）なので上積みが −0.0005 しかない。AUC では W2 x2 0.8651・WA x2 0.8645・S-A x2 0.8634（差 0.0017、雑音の床 0.024 よりずっと小さい）
* **M-C は M-B より過学習が早い。** 学習行が M-B のプール（784,730）の約半分で、ゼロからは ep3〜4、温間は ep2 で検証損失が最小になり、その後は上がる
* ⚠ M-B の表（TODO.md「9/24 — IKA-194・IKA-86」）とは分布が違うので、数字どうしは比べない。比べたのは腕の並び方だけ（records/IKA-86.md）

### 盤にかける 2 案（選んだ理由）

「温間の最良」対「ゼロからの最良」。1 案が明らかに勝ってはいない（x2 どうしの差は log loss 0.001〜0.002、AUC 0.002）ので、「その案 対 value-gen11L」には替えない。

* 温間の最良: **W2 x2**。x2 の log loss は WA（0.4504）とほぼ同じ（+0.0006）で、単体・AUC・turn-1 AUC は W2 が上。学習が 2 エポックと短く、SA の早期終了が選ぶエポックに左右されない
* ゼロからの最良: **S-A x2**。S-B8 は単体では S-A と同じだが、x2 では 0.4602 と +0.008 悪い（2 本が似すぎてアンサンブルが効かない）

## 3. 盤: 未実施（道具が無い）→ IKA-259 で作った。盤の結果は §4

`tools/match_queue.py`（→ `tools/generation_match.py`）には **M-C のプールから両席を引く口が無い**。`generation_match.py` は `--roster`（自陣の 1 構築）対 `standings.pool("all")`
で、`--selection-book` か `--uniform-selection` を要る。M-C の生成の形（`--pool regmc-matchupweb`・選出はその場で解く・両席の控えの推定分布も同じ解から）は
`tools/selfplay.py` → `pokeuraou.poolplay.generate_pool` の 1 腕の自己対戦にしかない。`grep -rln 'generate_pool\|poolplay' tools/ src/` は
poolplay.py・selfplay.py・agent_drift.py・replacement_audit.py だけ。

2 つの腕を M-C のプールで対戦させるには、次を決めて作る必要がある（課題の範囲外なので、調整役に返した）:

1. 選出をどの葉で解くか。各腕が自分の葉で解き、自分の席の 4 匹と相手の控えの推定分布を自分の解から取るのが、生成の形（「その場で考える」）をそのまま腕ごとにした形。
   解の共有ディレクトリ（`SolvedSelections.store`）は腕ごとに分ける（tag が葉を含むので、混ぜると止まる）。gen11L の解は selfplay-mc0/selection-solved/ にある
2. 対の引き方。index `i` の局 `i // 2` で対を引き、席だけ入れ替える（generation_match の対の形）。ミラーの扱い
3. 生成と同じ旗（幅 12・`--rank-leaf`・隠蔽・served）。`tools/agent_drift.py --check` に通す（measurement-lags-generation）
4. ワーカーの echo に、腕ごとの葉・選出の出どころ・控えの扱いを両席分出す（a-menu-belongs-to-the-agent-that-built-it）

## 4. 盤（9/25 — 続き、master eb1ed5a = IKA-259 を取り込んだ後）

道具は `tools/match_queue.py --pool regmc-matchupweb`（IKA-259）。全対戦で生成と同じ条件（幅 12・`--rank-leaf`・隠蔽・served 24 ワーカー / 2 サーバ）。
各腕は自分の葉で選出を解き、推定分布も自分の解から取る（ε 0・T 1）。hp-share の腕は一様。駆動は `C:/tmp/ikamc0/board.py`（コミットしない）。
記録は `$M/data/matches-mc/<名前>/`（M-B の `data/matches` と分けた。理由は §4.4）。腕のモデルは `C:/tmp/ikamc0/arms/`:

* `value-mc0-warm2-s{0,1}.pt` = 温間・2ep lr 5e-4（W2）の 2 本。盤では x2 = `value-mc0-warm2x2`
* `value-mc0-scratch-s{0,1}.pt` = ゼロから・今のレシピ（S-A）の 2 本。盤では x2 = `value-mc0-scratchx2`
* 選出の解は葉ごとに `C:/tmp/ikamc0/stores/<葉>/` に貯めて使い回した（gen11L は gen-0 の生成の解の写し、2,145 対）

### 4.1 null: warm2x2 対 warm2x2（`--sprt 0 10`、seed 8201）

```
SPRT        H0 after 102 pairs (204 games), LLR -2.970
対の得点    lost 0 / split 102 / won 0（50.0%、全対が引き分け）
壁時計      1.9 分
echo        両席とも tested・other の葉 value-mc0-warm2x2、selection solved・belief solved。推論サーバの腕 value・baseline とも
            「value-mc0-warm2-s0.pt, value-mc0-warm2-s1.pt (ensemble, logits averaged)」
```

### 4.2 本番: warm2x2 対 scratchx2（`--sprt 0 10`、上限 6,000 局/席、seed 8202）

```
SPRT        H1 after 795 pairs (1,590 games), LLR +2.983
対の得点    won 221 / split 414 / lost 160（温間が勝ち越し）
全 1,660 局 温間 53.49% ±2.35（対で見た区間）、Elo +24.3 [+7.9, +40.8]、対の 52% が引き分け（sprt_replay。止まるのはライブと同じ対 795）
壁時計      14.5 分（両腕の選出を初めて解く分を含む。scratchx2 は 1,099 対、warm2x2 は 864 対を解いた）
echo        worker7: 席 0・席 1 とも tested = warm2x2、other = scratchx2。どちらも selection solved・belief solved、store は葉ごとに別
```

⚠ 24 本のワーカーのうち 1 本（worker4）が 341 秒で落ちた。store のファイルを読もうとした瞬間に、別のワーカーがそのファイルを置き換えていた
（Windows の `PermissionError`。IKA-259 で直したのは書く側だけで、これは読む側）。キューがその 1 局を配り直し、局は番号から種を取るので、局も答えも変わらない。
match_queue の rc は 1。読む側を直した（`poolplay._read_shared`: `PermissionError` なら 20 ms 待って最大 50 回読み直す。テスト 1 件と、ずっと断られるときは例外のままという対照）。
直した後の 2 本（§4.3 の warm2x2・scratchx2 の行）で落ちたワーカーは 0。

### 4.3 表の行: 原点 `hp-share/w12/hidden-bench` に対する大きさ（固定局数、各 1,000 局/席）

大きさなので SPRT ではなく固定局数（register-the-stop-before-the-match）。

```
  腕                         seed   原点に対する勝率（対で見た区間）   Elo（sprt_replay）          壁時計
  value-gen11L（x1）         8211   70.40% ±1.93                     +150.5 [+134.7, +166.9]    4.1 分
  value-mc0-warm2x2          8212   78.90% ±1.80                     +229.1 [+210.9, +248.5]    10.1 分（新しい対の解を含む。1 コアのテストと重なった）
  value-mc0-scratchx2        8213   76.90% ±1.86                     +208.9 [+191.2, +227.7]    10.0 分
```

`tools/ratings.py --matches $M/data/matches-mc --anchor hp-share/w12/hidden-bench`（null・本番・3 行、7,895 局、4 腕、全部隠蔽）:

```
  agent                                                                  Elo      +-    games
  value-mc0-warm2x2/w12/leaf/book:solved/hidden-bench/belief:solved     230.3   14.9    4129
  value-mc0-scratchx2/w12/leaf/book:solved/hidden-bench/belief:solved   207.1   14.6    3661
  value-gen11L/w12/leaf/book:solved/hidden-bench/belief:solved          150.8   16.7    2000
  hp-share/w12/hidden-bench                                               0.0    0.0    6000
  席の有利 −0.023 logit（席 0 の勝ち 49.4%）。対戦ごとの fit と観測の差は ±0.8 ポイント以内（循環なし）
```

⚠ ratings は engine の指紋が 2 つある（2574b487 = 6,000 局・6b30e42d = 1,895 局）と言う。指紋はソースのハッシュで、§4.2 の後に `_read_shared` を入れたので分かれた。
違いは読むときの例外処理だけで、局は変わらない。

### 4.4 `ratings.py` は `pool-match` の記録を読むか

読む。`ratings.py` は `games-worker*.jsonl` の `provenance` と `outcome` があれば数え、`provenance.kind` で絞らない。名前は `agent_name` が作り、
原点は `hp-share/w12/hidden-bench` になる（上の表）。

ただし **規則（M-B / M-C）で分けない**。`--matches` の下を全部 1 つの fit に入れるので、M-C の対戦を `data/matches` に置くと M-B の表と混ざる。
今は名前が重ならない（`data/matches/.ratings-cache.json` に `hp-share/w12/hidden-bench` は 0 件）ので、つながらない別の成分になるだけ。
それでも `--anchor` の既定は M-B の `hp-share/w24/hidden-bench` なので、M-C の行は原点の無い成分として並ぶ。今回は `data/matches-mc` に分けて `--matches` で指した。
→ 別課題の候補（直していない）

## 5. 結論（IKA-82）

* **温間始動はゼロからより強い。** 同じ記録（gen-0）・同じ分割・同じ評価の盤で、温間 x2 がゼロから x2 に 53.5%（Elo +24 [+8, +41]）、SPRT(0, 10) は H1。
  原点に対しても 78.9% 対 76.9%（+229 対 +209、ratings では +230 対 +207）で、向きが同じ
* 検証の数字（x2 の log loss 差 0.0014、AUC 差 0.0017）は雑音の床より小さかったが、盤でははっきり差が出た。学習の数字だけで「差が無い」としなかったのは正しかった
* **value-gen11L をそのまま M-C で使うより、gen-0 で学習し直すほうがずっと強い**: +151 → +230（温間）・+207（ゼロから）
* **gen-0 の出荷モデル: 温間・2ep lr 5e-4 の 2 本**（盤で勝った側）
  * `$M/data/models/value-mc0.pt` = `C:/tmp/ikamc0/models/W2lr5e-4-s0.pt`（sha256 25d4c030c0a389fb…）
  * `$M/data/models/value-mc0-s1.pt` = `W2lr5e-4-s1.pt`（sha256 384f2e495e4e6a5f…）
  * `train_value.py --data $M/data/selfplay-mc0-encoded.npz --init-from $M/data/models/value-gen11L.pt --epochs 2 --lr 5e-4 --keep last --split-seed 0 --seed {0,1}`
  * 検証（6,000 局の held-out）: 単体 log loss 0.4514 / 0.4517、AUC 0.8648 / 0.8648、x2 0.4510 / 0.8651
  * 盤の名前は `value-mc0-warm2x2` だった。同じ重みを `value-mc0.pt`・`value-mc0-s1.pt` の名で使うと、葉の名前は `value-mc0x2` になる。
    選出の解の store は tag に葉の名前が入るので、`C:/tmp/ikamc0/stores/value-mc0-warm2x2`（約 1,700 対）をそのままは読めない（答えは同じ）

## 機械

heavy.py の行（`--agent IKA-82`）:

```
  00:58:25–01:00:36  1 コア    符号化                           131 s
  01:01:05–01:04:42  16 コア   12 本の学習（GPU 専有）           217 s（1 本 9〜28 s、1 エポック 2.6〜3.2 s）
  01:04:53–01:04:58  16 コア   12 本と gen11L の held-out 評価    5 s
  01:35:43–01:37:37  16 コア   盤 null warm2x2                  114 s
  01:37:50–01:52:23  16 コア   盤 warm2x2 対 scratchx2          873 s（rc 1 = ワーカー 1 本が落ちた）
  01:53:20–01:57:30  16 コア   行 gen11L 対 hp-share            250 s
  01:58:05–02:08:14  16 コア   行 warm2x2 対 hp-share           609 s
  01:58:00–02:07:43   1 コア   読む側の直しのテスト              582 s（上の行と重なった）
  02:08:14–02:18:13  16 コア   行 scratchx2 対 hp-share         599 s
```

合計: 16 コア 2,667 s（44.5 分。うち GPU の学習・評価 222 s、盤 2,445 s）、1 コア 713 s。
