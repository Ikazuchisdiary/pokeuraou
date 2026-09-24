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
