# 9/24 — 調整: gen-0 の前の 4 件の第 1 組（IKA-222・IKA-223）と、記録の置き場所の変更（IKA-228）

調整役のセッション（9/24、worktree `reinforcement-learning-improvements-782a96`）。IKA-210 の後半と IKA-212 を取り込んだ後の
記録は `TODO.md` の「9/24 — IKA-204 の調整（続き）」にある。ここはその後。

## 取り込んだもの

子は worker-medium に `isolation: "worktree"` で渡した。家の規則は `C:/tmp/pokeuraou-machine/house_rules_ika204.md`。

| 課題 | 中身 | master |
|---|---|---|
| IKA-223 | 交代の行列のずれ（d959ab0〜IKA-209）は、記録された searchValue（平均 0.004〜0.017）と方策（1,686 決定のうち 11）だけを動かした。学習の目標は勝敗で入力は本当の局面なので放置でよい。tools/replacement_audit.py | 7f8c356 |
| IKA-222 | わざわいの 4 特性・sniper・supremeoverlord・snowcloak を port に入れた。範囲技の途中で倒れたわざわいの持ち主は技が終わるまで残す。生成の壁時計と局は変わらない | 46bdf61 |

取り込み前の確認（調整役の worktree、`--agent IKA-204`）:

```
                      IKA-223                          IKA-222
cargo test --release  6 passed                         6 passed
テスト一式 (-n 8)     1,539 件 失敗 0・skip 11・xfail 1  1,615 件 失敗 0・skip 11・xfail 1
                      （68 s）                          （69 s）
ci_skip_audit         standings=9・vendor=2 ok         同じ
port_coverage/gate    ok / ok                          ok / ok
```

## 起票したもの

* IKA-223 から: IKA-229（決定の種類で教材の行を外す旗。td-lambda を使う日のため、future work）
* IKA-222 から: IKA-230（gate の「No effect a turn can observe」の群 9 id が注記も出さずに違う答え。IKA-210 §6 の「今は注記が出る」は誤り）、
  IKA-231（命中率の固定小数点）、IKA-232（範囲技の途中で倒れた持ち主の他の特性）、IKA-233（へんげんじざい・リベロの印、疑い）、
  IKA-234（IKA-222 の残り: 1% 未満の特性 22・道具 11、こだいかっせい・クォークチャージ）

## 記録の置き場所（IKA-228）

ユーザーの提案で、記録を `TODO.md` の末尾への追記から `records/` に移した。`TODO.md` の先頭に凍結の 1 行を足した
（他の行は変えていない）。`tests/test_no_machine_specific_paths.py` の記録の除外に `records/` を足した
（`test_line_endings` は追跡しているファイルを全部見るので、何もしなくても records/ を見る）。家の規則と auto memory も
records/ に替えた。

## 次

gen-0 に効く 4 件の第 2 組: IKA-213（damageCallback の技が素通り）と IKA-219（へんしん・かわりもの）。どちらも記録は
`records/IKA-213.md`・`records/IKA-219.md` に書く。

## IKA-77（M-C gen-0）の段取り —— 第 2 組（IKA-213・IKA-219）の後

ユーザーの指示（9/24）: 今の課題が一段落したら IKA-77 系を進める。引き継ぎは Linear の IKA-77 のコメント（9/24 07:13）と
TODO.md「9/24 — IKA-77: M-C gen-0 の生成の手順」「9/24 — IKA-194・IKA-86」。

引き継ぎの 5 項目と今の状態:

| # | 引き継ぎ | 今 |
|---|---|---|
| 1 | IKA-205（Urgent）ワイドフォースがサイコフィールドで範囲化しない（試走で 227/600 局）、ふうせん | **未着手（Todo）**。port の moveinfo.rs は 1.5 倍だけで、範囲化・0.75 倍・接地条件が無い |
| 2 | IKA-206 を入れるか | 入った（40746e0）が、diff_node は IKA-212 で Python ごと消えた。今は不要 |
| 3 | 試走 g600 をもう一度 | 未。試走の後に入った規則: IKA-201・202・203・205（未）・208・213（作業中）・219（作業中）・222 と、resolver の廃止 |
| 4 | 本番 40,000 局（75〜85 分、16 コア専有、seed 7701、data/selfplay-mc0） | 未。**9/23 のユーザーの決定で、生成に入る時点でもう一度相談する** |
| 5 | gen-0 の学習（IKA-86 のレシピ案） | 未。案は TODO「IKA-194・IKA-86」の末尾: ゼロから `--epochs 8 --keep last --average swa --swa-from 0.25`、温間 `--init-from value-gen11L --epochs 3 --lr 2.5e-4 --keep last`。温間の案 2〜3 通りを検証で比べてから決める |

走らせる順（各段の問い）:

1. IKA-213・IKA-219 を取り込む
2. **IKA-205** を子に渡す（問い: ワイドフォースはサイコフィールドで範囲技になり 0.75 倍・ワイドガードが効くか、1.5 倍は使い手が接地しているときだけか、ふうせんは割れるか。オラクルと正の対照）
3. **試走 g600 をもう一度**（今の master、TODO の §1 と同じコマンド・seed 7700、16 コア専有、約 4 分）。問い: 記録の注記と拒否の上位、注記の出ない未実装（C:/tmp/ika77/silent.py・terrain.py・read_games.py）に、本番の前に直す穴が残っているか。前回（9/24 早朝）の表と並べる
4. 残った穴と見積もりを持って **ユーザーに相談**（本番に入るか）
5. 本番 40,000 局（TODO §5 のコマンド。止まったら §6 と tools/queue_restart.py）
6. gen-0 の学習: 温間 2〜3 案・ゼロから 1 案を検証で比べる（GPU 専有、生成と並走させない）

M-C gen-0 に影響しない: IKA-223（交代の行列のずれ）は放置でよい（新しい port で作るので入らない）。

## その後（9/24 夜）: 取り込んだものと、IKA-86 の判断

取り込んだ（どれも取り込み前にテスト一式・skip の監査・port_coverage・port_gate_audit を読んだ。sim-bridge を変えた枝は dist を作り直した）:

| 課題 | master | 中身 |
|---|---|---|
| IKA-231 | f86f43c | 命中率を Showdown の整数の計算に |
| IKA-218・227 | 9310aad | diff_turn・diverge_report の PYTHONHASHSEED 依存、diverge_report が途中交代の先も比べる |
| IKA-213 | c872ce0 | カウンター・ミラーコート・メタルバースト・ほうふく、`damaging_move_is_unmodelled` |
| IKA-205 | 0368a02 | ワイドフォースの範囲化・ふうせん・フィールドパルス |
| IKA-235 | cd7e2b6 | 連続技の 2 発目以降の命中（dump に multiaccuracy） |
| IKA-239・238・240 | e7f7e32 | ナイトヘッド・わるあがき、同速の並び（選択ソート）とミラーアーマー、壁破り・ポルターガイスト・むしくい |
| IKA-244 | e11b337 | 生成の開始 PP を Showdown に（dump に startPP。ENCODING_REVISION は 2 のまま、ユーザーの判断） |
| IKA-246 | ff26ce2 | 門の field の残りを dump に |

保留: IKA-219（へんしん）。hp-share の壁時計が混んだ窓で 1.015〜1.016。静かな窓で ABBA×2 を測り直してから決める（M-C のプールにメタモンは 0）。

**IKA-86 の判断（ユーザー、9/24）**: 盤での確認は M-B でやらず、M-C gen-0 の学習で兼ねる。段取りの 6 は次の形にする。
gen-0 の記録で「温間・ゼロから × 今のレシピ・SWA 系」を検証で比べ、上位を盤（`--sprt 0 10`、null 対戦を先に）にかける。
IKA-82 の「温間 対 ゼロから」の盤と同じ回。`ValueConfig` の既定はその結果で変える。生成（試走・本番）は IKA-86 を待たない。

次: IKA-248（マジックガード）を取り込む → IKA-219 の壁時計の測り直し → 試走 g600 → 本番の相談。
