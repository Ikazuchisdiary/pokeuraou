# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Pokémon Champions（VGC 2026 Reg M-C）のダブルバトル検討ソルバ。同時手番・確率的・不完全情報のゲームとして解き、
均衡の混合戦略を出す。価値関数は自己対戦で学ぶ。説明と記録は日本語（README.md・rust/README.md・records/）。

## コマンド

```bash
# 初回（vendor の Showdown、sim-bridge、Rust の port、Python）
git submodule update --init
cd vendor/pokemon-showdown && npm ci && node build && cd -
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
- 高速化の不変条件は「前後で同じ種の局がビット一致すること」。

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
3. 記録（jsonl）を `tools/encode_dataset.py` で符号化し、`tools/train_value.py` で学ぶ（温間始動は `--init-from`）。
4. 盤は `tools/match_queue.py`（M-C は `--pool`、止めるのは `--sprt 0 10`）。レーティングは `tools/ratings.py`。
5. 今の M-C の出荷の葉は `data/models/value-mc0.pt`・`value-mc0-s1.pt`（2 本の平均）。
- 生成の設定と、対戦・計測の道具の設定がずれていないかは `agent_drift.py` が見る。対戦で打つ手は、生成と同じ引数で `play_game` を呼んでいなければならない。
- 速さの内訳は `tools/profile_stages.py`（段の計時・二度呼びの数え・標本採り。既定で off）で測る。

## 守ること

- **記録:** 作業の記録は `records/IKA-NNN.md`（1 課題 1 ファイル、日本語）。`TODO.md` は 9/24 から凍結で、読むだけ（1.3 MB あるので grep で位置を出して読む）。記録を探すときは `grep -rn '<語>' records/ TODO.md GENERATIONS.md`。世代とレーティングの表は `GENERATIONS.md`。M-B と M-C は別の尺度で、M-C の原点は `hp-share/w12/hidden-bench`。
- **課題:** 管理は Notion のデータベース「Issues」（ページ「pokeuraou」の下、https://app.notion.com/p/f58fae0e4bf843e5b00c03873f17b328）。番号は IKA-NNN を手で振る（最大の No + 1）。9/25 に Linear から移った。Linear に残っているのは、未完了の子を持たない Done の課題。
- **用語:** 同じページの下の「用語集」（https://app.notion.com/p/3e64b8acaa6c8195a66ce7bc42f24256）に従う。たとえば「控え」は「裏」、「完成形」は「裏の決定化」、「盤」は「対戦評価」と書く。コードの識別子・旗・JSON のキーは変えず、古い記録は旧名のまま読む。
- **ファイル:** すべて LF（`tests/test_line_endings.py`）。Python の `write_text` は Windows で CRLF になるので、bytes で書く。`tools/` と `src/` に機械ごとの絶対パスを書かない（`tests/test_no_machine_specific_paths.py`）。`scratchpad/` は当時のコードをそのまま残す記録なので、lint しない・書き換えない。
- **本番の経路:** 生成を遅くする変更は入れない。前後を交互（ABBA）に回し、同じ窓で壁時計を比べる（同じ exe、同じ長さのパスで）。規則を変えて局が変わるなら、盤で弱くならないことを確かめる。
- **速さと強さの引き換え:** 探索を安くして局が変わる変更（順位付けや予算を安くするなど）は、同じ費用で判定する。生成向けは、同じ壁時計で作ったプールから学んだ葉の強さで比べる（IKA-73 の形）。打ち手向けは、同じ時間の盤で比べる。設定を固定した盤で弱くなるだけでは捨てない（IKA-268・IKA-270 は捨てずに枝に残してある）。
- **データ:** `data/` は git の外。生成データ・モデル・大会データ（standings）・使用率（priors）はここに置く。
- **環境:** Windows（PowerShell と Git Bash）。Git Bash の MSYS は `/` で始まる引数をパスに書き換える。1 コアや 30 秒を超える仕事を並べるときは、機械（8 物理・16 論理コア、31 GB、RTX 5070）を取り合わないようにする。
