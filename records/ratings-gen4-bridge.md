# ratings.py の gen-4 の行を群 0 につなぐ

作業: 2026-10-02、ブランチ ratings-gen4-bridge（origin/master 05b917d から）。

## 問題

`tools/ratings.py --matches data/matches-mc --anchor hp-share/w12/hidden-bench` で、gen-4 の行（value-mc4x2 の q-mc4・q-mc3、value-mc3x2 の q-mc3）が群 1 として、ほかの打ち手（群 0）から切れていた（±216）。

原因は 2 つ。

1. 名前。`data/models/value-mc3.pt`・`value-mc3-s1.pt` は `C:/tmp/ika405/models/value-mc3e6-s0.pt`・`-s1.pt` とバイト単位で同じ（sha1 506faff7…・9fa83bbd…）。本番にしたときに名前を替えた（コミット c0be672）。IKA-405 の記録は `value-mc3e6x2`、IKA-409 の記録は `value-mc3x2` と書いている。
2. Q。群 0 の 6 エポック版は Q が q-mc2 の組だけ、群 1 は q-mc3 の組だけ。名前をそろえても Q が違う 2 つの打ち手は別のまま。

## A. 名前の表

`src/pokeuraou/provenance.py` に `LEAF_ALIASES` を足し、`agent_name` が `.pt` を落としたあとに引く。
data/matches-mc の各対戦の先頭の局から provenance.leaves を読み、実際の書き方は `value-mc3e6x2`（IKA-405）と `value-mc3x2`（IKA-409）の 2 つと確かめた。単独の `value-mc3e6` は記録に無いが、x2 でない名前でも効くよう表に入れた。`value-mc3e4x2`（4 エポック版）は別の重みなので表に入れない。

`tools/ratings.py` の `NAMES`（名前の作り方の版）を 2 から 3 に上げた。上げないと、古い名前で作った `.ratings-cache.json` を読み直さず、表の変更が効かない。

テスト: `tests/test_provenance.py::test_promoted_gen3_leaf_has_one_name`。表を外すと落ちることを確かめた（`leaf = LEAF_ALIASES.get(leaf, leaf)` を `leaf = leaf` に替えると 1 本落ち、戻すと 9 本通る）。

## B. つなぎの対戦評価

value-mc3x2 の上で Q だけを違える。試す側の Q は q-mc3、相手の Q は q-mc2。

* 形は IKA-409 の board.py と IKA-402 の q 対戦評価（run.py）の写し: `match_queue --pool regmc-matchupweb --served --servers 2 --hide-bench`、幅 12、`--rank-leaf`、`--q-model q-mc2.pt --q-model-named new q-mc3.pt`、試す側 `--rank-fill q-nocover.new`・相手 `q-nocover`、選出の store は IKA-409 の `store-mc3x2`（2,145 対、新しく解く対は 0）を両腕で共有
* 両腕とも value-mc3.pt・value-mc3-s1.pt（x2）
* 固定 400 対（800 局）、SPRT なし、seed 91002、出力 `data/matches-mc/bridge-mc3-q3-vs-q2`。起動の script は `C:/tmp/ika-rbridge/bridge.py`
* 先に `--games 2` で試走（4 局、6 秒）。worker のログに `Q arm new: q-mc3.pt`・`Q arm q: q-mc2.pt` と、試す側の rank fill が `q-nocover.new`、相手が `q-nocover` と出ることを確かめてから本走。本走は heavy.py（`--cores 16 --cores-min 8 --gpu`）の待ちに従い、実行 99 秒
* 局は 800 局が書かれ、失敗した worker は 0

### 結果

`tools/paired_result.py` で読んだ。

| 項目 | 値 |
| --- | --- |
| 対数 | 400 対（800 局） |
| 同じ側が 2 席とも勝った対 | 247 対（62%） |
| 試す側（q-mc3）の得点率 | 49.38% ±3.03% |
| Elo（q-mc3 − q-mc2） | −4.3 [−25.5, +16.8] |

q-mc3 と q-mc2 の差は、この対数では 0 と区別できない。ratings.py の当てはめでは同じ 2 つの行が 363.8 と 369.8 で、差 −6.0 は直接の測定と同じ向き・大きさ。

この読み方が動くことの対照として、同じコマンドを IKA-409 の i409-v4-vs-v3 に使い、記録の +51.1 と同じ値が出ることを確かめた（+51.1 [+28.5, +74.2]。記録の [+25.8, +77.0] は SPRT の停止時点の区間）。

## C. 当てはめ直し

A の入った worktree の src を PYTHONPATH に、メインの .venv の python で、同じコマンドを実行（出力は `C:/tmp/ika-rbridge/ratings-final.txt`）。

* 59,228 局、18 打ち手、群は 1 つ（以前は群 0 と群 1 の 2 つ）
* 対照 1（表あり・つなぎなし。つなぎの dir を除いた複製で当てはめ）: 58,428 局、18 打ち手、gen-4 の行は群 1（±216）のまま。名前だけでは足りない
* 対照 2（つなぎあり・表なし。`LEAF_ALIASES.clear()` で当てはめ、新しい cache）: 19 打ち手、gen-4 の行は群 1（±193）のまま。つなぎだけでも足りない
* 表あり・つなぎあり: 18 打ち手、群 0 のみ

| 打ち手（幅 12・rank-leaf・q-nocover） | Elo | ± | 局数 |
| --- | --- | --- | --- |
| value-mc4x2（Q q-mc4、gen-4 の本番） | +415.7 | 48.5 | 3,782 |
| value-mc4x2（Q q-mc3） | +413.4 | 47.3 | 5,308 |
| value-mc3x2（Q q-mc2、gen-3 の 6 エポック版） | +369.8 | 32.5 | 4,174 |
| value-mc3x2（Q q-mc3） | +363.8 | 40.1 | 1,526 |
| value-mc3e4x2（Q q-mc3） | +358.2 | 32.2 | 3,443 |
| value-mc3e4x2（Q q-mc2） | +355.0 | 30.1 | 21,150 |
| value-mc2x2（Q q-mc2、gen-2 の本番） | +299.5 | 17.5 | 8,764 |
| value-mc1x2（Q q-mc0、gen-1 の本番） | +254.2 | 11.3 | 12,476 |
| value-mc0x2（Q q-mc0、gen-0） | +229.7 | 14.5 | 3,255 |
| value-gen11L（M-B の重みのまま） | +150.8 | 16.7 | 2,000 |
| hp-share/w12/hidden-bench | 0（原点） | 0 | 14,000 |

value-mc3x2 の q-mc2 の行（4,174 局）は、つなぎを除いた当てはめの 3,374 局（IKA-405 の `value-mc3e6x2` と IKA-402 の分）につなぎの 800 局を足したもの。q-mc3 の行の 1,526 局は、IKA-409 の value-mc4x2 対 value-mc3x2 の 726 局につなぎの 800 局を足したもの。

## 見ていないこと

* gen-3 と gen-4 の差（+46〜+52）は、行の ± が 40〜48 あるので、この表からは精密に言えない。直接の対戦評価は IKA-409 の +51.1 [+25.8, +77.0]。
* つなぎは q-mc3 対 q-mc2 の 1 本だけ。q-mc4 の行と q-mc3 の行は、IKA-409 の Q の対戦評価（q-mc4 対 q-mc3、value-mc4x2 の上）が元からつないでいる。
* 実行は C:/tmp/ika409/wt（IKA-409 の worktree。ビルド済みの exe と data/pool がある）から。match_queue の中身は origin/master のあとの IKA-413 の変更より前のコミット（250be8b）。別の worktree の exe を使ったことを断る。
