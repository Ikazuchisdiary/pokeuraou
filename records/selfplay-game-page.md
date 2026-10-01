# 生成の局を HTML で見る（課題番号なし）

ユーザーの依頼（10/1）: 生成の局（`data/selfplay-mc3/games-*worker*.jsonl`）の 1 局を、対戦評価の局と同じ HTML ページで見たい。

## 変えたもの

- `tools/show_game.py --file <jsonl> --game N --html --out game.html`。`--html` を付けないテキスト出力と `--transcript` の HTML は変えていない。
- `show_game.turn_events` の前半を `turn_trace` に切り出した（port で解き直した枝の選び方は同じ。
  結果は `_Trace`：trace の行・行動ごとの区切り・注記・途中交代の有無）。`turn_events` は `turn_trace` を言葉にするだけ。
- `game_page.selfplay_line`: 生成の局の記録から、`render_html` が読む transcript の行を組み立てる。
- 読みは席ごとに、席 0 は `searchValue` と `ownPolicy`、席 1 は `foeSearchValue`（席 0 の単位）と `foePolicy`。
  裏を隠す局は席ごとに別の局を解いているので、`foeSearchValue` が無い記録の席 1 は「—」（席 0 の値を写さない）。
- 記録に無いもの（読みの段・時計・手ごとの値・席が相手に見ていた混合・席の条件・乱数の種・組）は、
  ページが「生成の 1 局」「記録にない（—）」と書き、空の表は出さない。

## 確かめたこと

- data/selfplay-mc3/games-b07-worker3.jsonl の 5 局目。テキスト版は切り出しの前後でバイト一致（`cmp`）し、
  ユーザーが見たテキスト（gamelogs/gen3-b07-w3-g5.txt）とも一致。
- HTML とテキストの突き合わせ（9 手番）: ターン番号、各ターンの席 0 の読み（%）、引いた手の均衡確率がすべて一致。
  ターン 3 だけ、テキストが 3 桁に丸めた値（0.665）を再度丸めると 66、元の値では 67 で、ページは元の値から丸めている。
- テスト `tests/test_selfplay_game_page.py`（4 本）。対照として、値を変えるとページが変わること、`foeSearchValue` を落とすと席 1 が「—」になることを見ている。
- スクリーンショット: Edge のヘッドレスでデスクトップ幅（1100）と 375 px（iframe で幅を固定）。崩れなし。

## 10/1 読みの表に技の対象を出す

- 原因: `game_page.pair_of`（読み・均衡の表の手のラベル）が、`name_action` の「→ 敵…」を正規表現で削っていた（交代だけ残す）。played-move 側の `hands` は残していたので、読みの表だけ対象が消えていた。`--reread` の表も同じ `mixtures`→`pair_of` を通るので同時に直った。
- 直し: 対象を残し、`hands` と同じく「→ 敵」を「→ 相手の」、「→ 味方」を「→ 味方の」にする。全体技・交代・対象の無い技は `name_action` が対象を付けないので今のまま（例: ハイパーボイス）。
- テスト: `test_a_mixture_row_names_the_target_of_a_single_target_move`。直す前のコードで落ち、直して通る。単体技と対象なしの技の両方が出たことも確かめる。
- 例の局（gen4-b01-w7-g12）のターン 1 席 0: 「でんじほう → 相手のウインディ (Hisui) + メガ」「ねこだまし → 相手のムクホーク」。スクリーンショット C:/tmp/sptarget/turn1-open.png。
