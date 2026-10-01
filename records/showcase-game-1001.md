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
