# 展示対局 1001: 今できる最高の条件で AI 同士を 1 局打つ

2026-10-01。課題番号なし。ユーザーの依頼「今できる最高の条件で 1 回対局してログを見せて」。master 085fb69 から切った `showcase-game-1001`。

## 結果

- 席 0（Garde Psyspam）の勝ち。種 1001、組 0、局 0。5 ターンの決定、6 ターン目の途中で全滅。
- 勝率（席 0 側、2 席の読みの平均）: T1 62% → T2 45% → T3 61% → T4 99% → T5 100%。
- 時間: 選出 92.3 秒、対局 450.5 秒、全体の専有 550 秒（heavy.py の窓 21:19〜21:28）。1 手 45.0〜45.1 秒で、予算どおり。
- 到達した段（L6）: T1 深さ 2、T2 深さ 2、T3 深さ 3、T4 深さ 5/4、T5 深さ 6/5。
- 出力: `C:/tmp/showcase/game/`（games・transcripts・logs・settings）、`C:/tmp/showcase/game.html`、`game-open.html`（読みの表を開いた版）、`summary.txt`。

## 条件（両席とも同じ）

- 評価モデル value-mc4 + value-mc4-s1（平均）、Q q-mc4、候補集合 q-nocover、構築プール regmc-matchupweb（組は種から無作為）。
- 1 手: L6（ladder）、実時間 45 秒、16 スレッド（ladder の worker 15 本 + 本体）、神経網の推論サーバ 2 本（`--ladder-pool --ladder-servers 2`: 本体は 1 本目に、worker は交互に 2 本に依頼）。
- 選出: `selection=default`、実時間 90 秒、全論理コアの読み手 16 本（`selection_deep.LazyPoolReader`、読み手も 2 本のサーバに交互）。
- 裏は非公開（`bench_drop` none、各席は相手の裏の信念で読む）。
- 道具: `tools/time_match.py`（`--transcript`、`--single-game`）、図は `tools/show_game.py --transcript`。

## 本番（play_human）との違い

- 本番の人の席は人が打つ。ここは同じ条件の 2 席で、1 つの過程の中で順に読む（片方が読む間もう片方は待つ。人と打つときの「AI が読む間は全コアが AI のもの」と同じ）。
- port の旗（`POKEURAOU_LADDER_PORT_*`）は本番の既定の off のまま。IKA-389・397 が速い形として測った on は使っていない。
- 推論サーバ 2 本は本番の play_human の既定ではない（IKA-390 の段 A、`position_set` にだけある形）。
- 選出は A・B が同じ条件なので 1 回だけ解き、両席が同じ解から 4 体を引いた（A=B の対局では当然）。選出の読みは 90 秒の予算を 92.0 秒で打ち切り、完了した段は stage 3 (rect 24)。人の選出を待たない。
- 予備の先読み（ponder）は本番どおり off。

## コードの変更（time_match だけ。既定の経路は変えない）

- `tools/time_match.py`: `--ladder-pool`（ladder の worker と選出の読み手を全コアで）、`--ladder-servers N`、`--ladder-inference`、`--single-game`。
- `src/pokeuraou/timematch.py`: 条件の鍵 `selection_seconds`、`play_pair(games=)`、選出の後に読み手を返す。
- `src/pokeuraou/selection_deep.py`: `PoolReader` が `ladder._worker_args` で、サーバの一覧の k 番目を k 番目の読み手に。
- 確かめたのは 3 秒・5 秒の短い 1 局のスモークテスト（15 worker が立ち、局が終わる）と本番の 1 局だけ。テスト・ruff・agent_drift は回していない。旗が既定 off の経路が変わらないことのビット一致は見ていない。

## 局 2 から 4

2026-10-02。origin/master の取り込み後、port の exe を作り直し、同じ設定で種 1002、1003、1004 の 3 局を 1 局ずつ打った。構築の組は局 1 と別。局ごとの出力は `C:/tmp/showcase/game2/` から `game4/`、HTML は `C:/tmp/showcase/game2.html` から `game4.html`。勝率は席 0 側で、2 席の読みの平均。

| 局 | 構築（席 0 対 席 1） | 結果 | 勝率の推移（席 0 側） | かかった時間 |
|---|---|---|---|---|
| 2（種 1002） | Brady Perish 対 Standard Zard Y | 席 1 の勝ち、12 ターン | T1 60%、T2 71%、T3 65%、T4 16%、T5 6%、T6 以降 0% | 1091 秒。選出 92.7 秒、対局 991 秒 |
| 3（種 1003） | Gengar + Swamp Rain 対 Mence + Metagross | 席 1 の勝ち、9 ターン | T1 62%、T2 62%、T3 65%、T4 35%、T5 42%、T6 19%、T7 以降 0% | 820 秒。選出 91.7 秒、対局 721 秒 |
| 4（種 1004） | Gengar + Swamp (Kommo) 対 Phox + Floette (Rilla) | 席 1 の勝ち、12 ターン | T1 25%、T2 17%、T3 10%、T4 5%、T5 2%、T6 2%、T7 以降 0% | 1058 秒。選出 91.0 秒、対局 960 秒 |

## 旗のテスト

`tests/test_timematch.py` に 3 本を足した。

- `games=(0,)` は game 0 だけを打ち、既定で 2 局打つ場合の game 0 と時刻以外が同じ（陽性対照は既定の 2 局）。
- `selection_seconds` は条件の鍵として読まれ、`describe` に出て、エージェントに届く。
- サーバの一覧は、k 番目の worker に k 番目の住所を渡す。1 本のときは変わらない。

通した検査は ruff（変更した 4 ファイル）、`agent_drift.py --check`（終了コード 0、time_match は ok）、`test_timematch.py`・`test_selection_deep.py`・`test_ladder_servers.py`、`test_line_endings.py`、`test_no_machine_specific_paths.py`。

