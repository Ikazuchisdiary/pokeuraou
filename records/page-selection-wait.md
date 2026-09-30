# 人と打つ画面: 選出を人と AI が同時に行う・日本語名が無い種族の表記（枝 `page-selection-wait`、master 717bb23 の上）

課題番号は未採番（仮名）。対象は `tools/play_human.py --person web --view`。

## 1. 選出を同時にする

調整役（ユーザーの指摘）: 選出は実際の対戦と同じく、お互いが同じ時間の中で選ぶ。

* 選出の解（`humanplay.solve_entry`）は両チームの 6 体だけを使い、人の選出を入力に持たない（`selection_deep.solve_selection_deep(reg, row.sets, col.sets, ...)`）。だから並べても局は変わらない。
* `humanplay.play`: 読みを持つ実時間の局で listener がある（画面がある）とき、`sheets` → `selecting`（秒数）→ `select` を続けて出し、人の入力待ち（`person.select`）を補助スレッドに移し（`_Asked`。デーモン。HiGHS はこのスレッドでは動かさない: 短命のスレッドで HiGHS を動かすと止まる）、メインスレッドで選出を読む。読み終えたら中立の `selected`（AI の 4 体は載せない）を出し、人の答えを待つ。人の答えが例外なら読みのあとで再送出する。読みを持たない局・ノード時間・listener なし（端末・script）は今までのまま。
* AI は人が早く決めても、90 秒（`--selection-seconds`）読み切る。人が 90 秒を過ぎても決めないときの扱いは今も無く、そのまま（AI の読みは期限で止まる）。
* 画面（`live-view.js`）:
  * 選んでいる間の状態欄: 「4 体を選んでください（AI も選出を読んでいます・残り N 秒）」。スマホの短い版は「選出の番」。
  * 送ったあと: 「AI の選出を待っています（残り N 秒）」（短い版「AI を待つ」）。入力の枠は「選出を送りました。AI の選出を待っています」。
  * スマホの状態欄には秒数が入らないので（デザイン担当の指摘）、入力の枠にも出す: 選ぶ間は見出しの下に灰色の小さな文字「AI も選出を読んでいます・残り N 秒」、送ったあとは枠の文言の末尾に「（残り N 秒）」。状態欄と同じ 1 秒ごとの更新で、文字だけを書き換える。
  * `selected`: AI の選出は出さず、選んでいる途中なら状態欄が「4 体を選んでください」に戻る。送っていたら「AI の選出が決まりました。まもなく始まります」。最初の `board` で待ちの時計を止める（そこで両者の選出が公開される）。
  * 1 秒ごとの更新は文字と title だけ（点滅を戻さない。IKA-392 §12 と同じ）。
* 前の直し（待ちの間に「あなたの手」の枠の文言を替え、チームの枠を開く）は、この形では待ちが無いので入れていない。

## 2. 日本語名が無い種族

* 原因: Showdown の日本語 text（`data/text/ja/pokedex.ts`）に Floette・Alcremie・Toxtricity などの元の種族の日本語が無い（`name: null`）。`Localiser.species` は元の日本語が無いとき英語名の `Floette-Eternal` をそのまま返した。手書きの対応表は作らない。
* 直し（`names.py`）: 元の日本語が無いフォルムは `Floette (Eternal)` 、`Alcremie (Caramel-Swirl)` の形（元の英語名＋括弧のフォルム）にした。日本語がある元のフォルムの `ウインディ (Hisui)` と同じ形なので、画面が同じ関数（`nameHtml`）でフォルムを小さな文字にできる。`tools/game_page.py` の `name_html` も同じ形を読む。
* 画面（`live-view.js` の `nameHtml`、`live.css` の `.forme`）で使う場所: 場の札・控え・選出の札・チーム欄・経過の変化の行。手のラベルや title など Python が文字列で組む所は、同じ `Localiser.species` の出力（`ウインディ (Hisui)` の形の平文）。
* 洗い出し: 規則 M-C の使える種族とその変化（`team_legal` か `changes_from`）のうち、変更前に括弧の無い英語のフォルム名（id 的な `X-Y`）を出していたのは、Alcremie の 7 フォルム・Floette-Eternal・Floette-Mega・Toxtricity-Low-Key。変更後は、英語の元名のまま残るのは Alcremie・Toxtricity の 2 つ（元の種族そのもの）と、その 10 フォルム（`Alcremie (Ruby-Cream)`、`Floette (Eternal)`、`Floette (Mega)`、`Toxtricity (Low-Key)` など。`C:/tmp/pagewait/probe2.py` で全数を数えた）で、どれも「元の名前 (フォルム)」の形。id 的な `X-Y` の形の英語名は 0。
* 直していない: 持ち物の英語名（メガストーンなど。`ja-extra.json` の値が空）、Floette・Alcremie・Toxtricity の元の名前自体（日本語の出典が無いため英語のまま）。

## 3. 確かめ

* テスト: `tests/test_selection_deep.py` に 3 本（`test_the_person_picks_while_the_ai_reads_the_selection` は、人への問いが読みの終了より先に来ること。読みの前に問う形に戻すと落ちる。`test_a_failing_person_stops_the_game_after_the_reading`、事象の順 `sheets, selecting, select, selected`）。ワーカーの作成・閉じる順のテストは新しい順に直した。`tests/test_names.py` に 2 行。
* 陽性対照: 並行の条件を `and False` で外すと、新しい 2 本が落ちる（人への問いが読みのあとになる: `reading_finished_while_asking: False`）。外した状態は戻した。
* 関係テスト `test_selection_deep`・`test_names`・`test_humanplay`・`test_liveview`: 通る。ruff・`node --check`・`agent_drift --check`: 通る。
* 実物: `play_human --person web --view`（推論サーバ付き、`--selection-seconds 40`）を `heavy.py --gpu` で起動し、ヘッドレス Edge を CDP で動かして、選ぶ間・4 体を押した後・送った後・対局の画面を、デスクトップ（1280）と 375 px のライト・ダークで撮った（最終は `C:/tmp/pagewait/run3/shots/`。名前は `1-pick`・`2-picked`・`3-wait`・`4-board` の 4 局面 × 1280/375 × light/dark）。デザイン担当: 状態欄と英語名は可、秒数を入力の枠にも出すことを条件に確定と返答 → 条件を入れて撮り直し（run3）、撮り直しの画像を見せた（再返答は受けていない）。状態欄と入力の文字の実測は `text-*.json`。起動したプロセスは `heavy.py` の rc=0 で終わった（設定した時間で推論サーバを止めてから終了）。
* 確かめていない: rust の exe は本枝でビルドせず、master と同じソースの主チェックアウトの exe を `POKEURAOU_RUST_NODE_BIN` で使った（Rust は変えていない）。90 秒の本番の長さでは撮っていない（40 秒）。後から開いたページは再生で残りを最初から数え直す（IKA-392 §12 と同じ）。スマホは秒数を出せない（短い版は固定文言）。
