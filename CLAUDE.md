# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Pokémon Champions（VGC 2026 Reg M-C）のダブルバトル検討ソルバ。同時手番・確率的・不完全情報のゲームとして解き、
均衡の混合戦略を出す。価値関数は自己対戦で学ぶ。説明と記録は日本語（README.md・rust/README.md・records/）。

## コマンド

```bash
# 初回（vendor の Showdown、sim-bridge、Rust の port、Python）
git submodule update --init
cd vendor/pokemon-showdown && npm ci && node build decl && cd -
npm install && npm run build                       # packages/sim-bridge/dist（オラクル。テストが使う）
cd rust && cargo build --release && cd -           # port の exe（rust/target/release/pokeuraou-damage.exe）
uv sync                                            # 価値関数を使うなら uv sync --group learn

# テスト（既定で -n auto --dist worksteal）
uv run pytest
uv run pytest tests/test_foo.py::test_bar -n 0     # 1 本。pdb を使うなら -n 0 が要る
uv run pytest -m "not slow"
cd rust && cargo test --release

# CI と同じ検査
uv run ruff check .
uv run python tools/agent_drift.py --check         # 対戦・計測の道具が生成と同じ打ち手を作っているか
uv run python tools/port_gate_audit.py --check     # port の門の一覧
uv run python tools/port_coverage.py --check       # 生成物 rust/src/inert.rs・modelled.rs が source と一致するか
uv run pytest --junitxml=reports/pytest.xml && uv run python tools/ci_skip_audit.py --junit reports/pytest.xml --absent standings=9 --absent vendor=2
```

- port の exe が無い・ソースより古いと、テストも本番の経路も skip せずに止まる（`rustnode.PortUnavailable`）。Rust を触ったら必ずビルドし直す。
- sim-bridge の `src/` を変えたら `npm run build` で dist を作り直す（dist は git に無い）。レギュレーションの dump（`configs/regulations/*.json`）は `npm run dump-regulation` で作る生成物で、コミットする。
- 規則の門や特性の表を変えたら `tools/port_coverage.py --rust` / `--rust-modelled` で inert.rs・modelled.rs を作り直す。

## 構成（複数のファイルを読まないと分からないこと）

**正しさの鎖は Showdown → port の 1 段。**
- `vendor/pokemon-showdown` は固定した submodule で、`packages/sim-bridge`（TypeScript）だけがそれを使う。sim-bridge の役目は次の 3 つ。
  - dex からレギュレーションの dump を作る
  - 局面の正規形を定める（`position.ts` ⇔ `src/pokeuraou/position.py`）
  - 乱数を固定できる決定論のオラクル（JSONL over stdio）
- 1 ターンの解決は Rust の port（`rust/`）だけが行う。Python の resolver は IKA-212 で消えた。
- 規則の正しさは Showdown で決める。port の規則を足したら、`tests/test_*_oracle.py` の形（port 対 Showdown）のテストを付け、直す前の exe で落ちることを正の対照として示す。
- 広い確認は `tools/diff_turn.py`（port 対 Showdown のターンごとの乖離）と `tools/diverge_report.py`。

**port が扱わない規則の扱い:**
- **注記:** 答えつつ unmodelled の注記を出す。
- **拒否:** `PortRefused` で止める。
- **門:** `rust/src/resolve.rs` の `UNHANDLED_MOVE_FIELDS` は、dump に出ている field しか見えない。dump に無い field は黙って素通りする（IKA-235・246 の見落とし）。
- **表:** inert.rs（port が名前を出さず、通してよい id）と modelled.rs（注記を出さない id）は、dex の `customHooks` と port の source から生成する。

**Python（`src/pokeuraou/`）と port の境界はノード単位。**
- `port.py` / `rustnode.py` は、局面と両側の行動リストを渡し、利得の行列を受け取る。
- 学習した葉は、葉を `encode.py` 形の配列にして Python に返し、torch で順伝播する。順伝播は torch に残す（加算順が変わると均衡が変わるため）。
- 試合を進める exact の解決は、重みだけを返す。Python が自分の numpy の乱数で 1 つ引き、その枝の局面を取りに行く。これで同じ種の対局がビット単位で再現する。
- 既存の経路の純粋な高速化・リファクタは「前後で同じ種の局がビット一致すること」で確かめる（バグを見つける道具）。
- 決定性は検証の道具で、製品の制約ではない（ユーザー、9/28）。局が少し変わる高速化（浮動小数の足し順・順伝播の積み上げ・スレッドの並べ方など）は、一致しないことでは捨てず、変化の大きさを記録して同じ費用で判定する。テストと A/B のためにノード時間・固定スレッドの決定的な読み方は残し、人と打つ・検討は実時間・全コア・非決定の並列を使ってよい。

**探索:**
- `search.py`（narrow で候補を絞り、葉の順位付けでメニューを作り、行列を埋める）
- `equilibrium.py`（HiGHS の LP で均衡を解く）
- 控えを隠すノードは `beliefnode.py` / `hidden.py`（控えの完成形ごとに解いて畳む）
- 選出（6→4）は、順序付きの 90×90 のベイズ型ゲーム（`selection.py`）

**生成と学習の流れ:**
1. `tools/generate_queue.py` がワーカー（`tools/selfplay.py`）と推論サーバ（`--served`）を立て、局をキューで配る。局の種は局の番号で決まるので、再開は `tools/queue_restart.py` で行う。
2. 局を打つ:
   - M-C は `poolplay.py`。両席を 65 構築のプールから引き、選出をその場で葉で解いて `selection-solved/` に共有する。
   - M-B は `selfplay.py`。自陣のロスタ対プール。
3. 記録（jsonl）を `tools/encode_dataset.py` で符号化し（局ごとの処理は port の `encode-games`、並べるなら `--jobs N`。IKA-347）、`tools/train_value.py` で学ぶ（温間始動は `--init-from`）。
4. 盤は `tools/match_queue.py`（M-C は `--pool`、止めるのは `--sprt 0 10`）。レーティングは `tools/ratings.py`。
5. 今の M-C の本番の評価モデルは `data/models/value-mc1.pt`・`value-mc1-s1.pt`（gen-1、2 本の平均。IKA-346）。次の世代は、gen-0 + gen-1 のように前の世代のデータに足して、前の世代の評価モデルの種ごとに温間始動で学ぶ（`--init-from <前の種 k> --epochs 2 --lr 5e-4 --keep last`）。
- Q（`data/models/q-mc0.pt`）は評価モデルを替えるたびに確かめる（IKA-346・IKA-348）: 新しい世代の局面 3,000（`q_teach.py index` の先頭）を、前と新しい評価モデルで `q_teach.py fill` し、`tools/q_drift.py` で Q のセルの MAE の増え方 r を出す。r が 11% 未満なら Q はそのまま。以上なら Q の教材を新しい評価モデルで作り直し（`q_train.py --checkpoint --stop-after` で 30 分以内の呼び出しに分ける）、同じ評価モデルの対戦評価（SPRT(0,10)）で前の Q と比べて、H1 のときだけ替える。gen-1 は r = +55.7% だったが、作り直した q-mc1 は q-mc0 に H0（−7.6 [−20.1, +5.0]）で、本番の Q は q-mc0 のまま（records/IKA-348.md）。r は「作り直しを試す」合図で、Q を替える理由にはならない。
- M-C のデータ生成と対戦評価の候補集合は、葉の順位付けで `--rank-fill` を名指ししなければ `q-nocover`（Q は `data/models/q-mc0.pt`、無ければ止まる。IKA-338）。`refs2` は名指しすれば打てる。IKA-341 からライブラリ（`play_game`・`_menus`・`generate_pool`・`PoolArm`）も同じで、名指しが無い葉の順位付けは `q-nocover`（Q が入っていなければ止まる）。M-B（Q が無い）は `search.ROSTER_RANK_FILL`（`refs2`）を名指しで打つ。人と打つ道具・検討は、読める Q が無ければ注記して `refs2`。refs の仕組み（`leaf_ranking`・`refs<N>`）は名指しでだけ打つ研究用に残す。
- 生成の設定と、対戦・計測の道具の設定がずれていないかは `agent_drift.py` が見る。対戦で打つ手は、生成と同じ引数で `play_game` を呼んでいなければならない。
- 速さの内訳は `tools/profile_stages.py`（段の計時・二度呼びの数え・標本採り。既定で off）で測る。

## 守ること

- **記録:** 作業の記録は `records/IKA-NNN.md`（1 課題 1 ファイル、日本語）。`TODO.md` は 9/24 から凍結で、読むだけ（1.3 MB あるので grep で位置を出して読む）。記録を探すときは `grep -rn '<語>' records/ TODO.md GENERATIONS.md`。世代とレーティングの表は `GENERATIONS.md`。M-B と M-C は別の尺度で、M-C の原点は `hp-share/w12/hidden-bench`。
- **課題:** 管理は Notion のデータベース「Issues」（ページ「pokeuraou」の下、https://app.notion.com/p/f58fae0e4bf843e5b00c03873f17b328）。番号は IKA-NNN を手で振る（最大の No + 1）。9/25 に Linear から移った。Linear に残っているのは、未完了の子を持たない Done の課題。
  - ページのアイコンは Status に合わせる: Backlog ⚪・Todo 🔵・In Progress 🟡・Done ✅・Canceled ❌。課題を作るときも Status を変えるときも、同じ更新でアイコンを付け替える（`notion-update-page` の `icon`）。
  - ユーザーへの報告と Notion の本文で課題に触れるときは、番号をその課題のページへのリンクにする（例: `[IKA-372](https://app.notion.com/p/3e94b8acaa6c819788e7c10aef4379a7)`）。URL は Issues を `ID` で引いて得る。
  - 結果（取り込み・段の報告など）は Notion のコメントにせず、本文の先頭に地の文で書く: `notion-update-page` の `insert_content`・`position: {type: start}` で、`### YYYY-MM-DD HH:MM UTC — 書き手` の見出し、結果、`---`（区切り線）の順。区切り線で下の課題の本文と分ける。新しい結果ほど上に来る。
  - 起票も報告も、読んで分かることを優先する。数字の比較は表に、流れや依存は図（Notion の mermaid のコードブロック）に、曲線や分布は必要なら画像にする。タイトルは短く（目安 40 字以内）、何をするか・何を問うかだけを書く。経緯・条件・数字は本文に回す。
- **用語:** 同じページの下の「用語集」（https://app.notion.com/p/3e64b8acaa6c8195a66ce7bc42f24256）に従う。たとえば「控え」は「裏」、「完成形」は「裏の決定化」、「盤」は「対戦評価」と書く。コードの識別子・旗・JSON のキーは変えず、古い記録は旧名のまま読む。
- **ファイル:** すべて LF（`tests/test_line_endings.py`）。Python の `write_text` は Windows で CRLF になるので、bytes で書く。`tools/` と `src/` に機械ごとの絶対パスを書かない（`tests/test_no_machine_specific_paths.py`）。`scratchpad/` は当時のコードをそのまま残す記録なので、lint しない・書き換えない。
- **本番の経路:** 生成を遅くする変更は入れない。前後を交互（ABBA）に回し、同じ窓で壁時計を比べる（同じ exe、同じ長さのパスで）。規則を変えて局が変わるなら、盤で弱くならないことを確かめる。
- **速さと強さの引き換え:** 探索を安くして局が変わる変更（順位付けや予算を安くするなど）は、同じ費用で判定する。生成向けは、同じ壁時計で作ったプールから学んだ葉の強さで比べる（IKA-73 の形）。打ち手向けは、同じ時間の盤で比べる。設定を固定した盤で弱くなるだけでは捨てない（IKA-268・IKA-270 は捨てずに枝に残してある）。
- **データ:** `data/` は git の外。生成データ・モデル・大会データ（standings）・使用率（priors）はここに置く。
- **環境:** Windows（PowerShell と Git Bash）。Git Bash の MSYS は `/` で始まる引数をパスに書き換える。1 コアや 30 秒を超える仕事を並べるときは、機械（8 物理・16 論理コア、31 GB、RTX 5070）を取り合わないようにする。
