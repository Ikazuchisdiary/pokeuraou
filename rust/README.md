# Rust 移植は「意外と楽」か（ダメージ層のスパイク）

結論から: **ダメージ層に限れば楽でした。** Python 2,039 行を Rust 1,664 行に移し、
**30,640 ケースで 1 件も食い違わず**、1 呼び出しあたり **35〜78 倍**速い。
ただし楽だった理由は移植そのものではなく、**この計画が既に持っていた 2 つの資産**です。

## 楽だった理由（＝再利用できた資産）

1. **dex がもう JSON になっている。** `configs/regulations/*.json` は種族・技・タイプ相性・
   メガ対応表を全部含んだ素のデータで、Showdown の TypeScript に触る必要が一切ない。
   Rust 側は serde で読むだけ（`src/reg.rs`、155 行）。移植でいちばん重いのは普通ここです。
2. **Python が差分オラクルになる。** 解決器は乱数をサンプルせず列挙するので決定論で、
   `damage.calculate` は純関数。だから「入力と Python の答え」を大量に吐いて突き合わせられる。
   Showdown に対する検証（`tools/diff_damage.py`）はそのまま Python が担い、
   Rust は **Python に対して**検証されます。検証の鎖が切れない。

## 測定

移植したのは `damage.py` + `effects.py` + `moveinfo.py` + `fixedpoint.py` + `battler.py`。

| ケース集合 | 件数 | 一致 | Python | Rust | 比 |
|---|---|---|---|---|---|
| `cases.json`（scenario-turn5 の 24x24 x 4クラス） | 1,018 | **1,018 / 1,018** | 48.7 us | 1.38 us | **35x** |
| `cases-field.json` の先頭 1,018（自己対戦・大会プール） | 1,018 | — | 128.5 us | 1.64 us | **78x** |
| `cases-field.json` 全体 | 5,992 | **5,992 / 5,992** | 149.4 us | 14.7 us | 10x |
| `cases-synthetic.json`（表の全エントリを発火させる格子） | 23,630 | **23,630 / 23,630** | 92.1 us | 4.17 us | 22x |

一致は rolls 16 本・effectiveness・typeMod・immune の**完全一致**で、許容誤差は置いていません。

**下 2 行の比が小さいのはハーネスの都合です。** 同じ 1,018 ケースでも、5,992 件の配列を
歩くと Rust は 1.64 → 14.7 us に落ちる（計算ではなくキャッシュミス）。Python は元が遅いので
相対的に効きません。実際の統合では入力は JSON から作り直すのではなく解決器がその場で持つので、
比較すべきは上 2 行です。

## 検証がどれだけ当てになるか

自己対戦 6 ゲーム分（40 技 / 24 種族 / 22 特性）でも、**ダメージ修正子を持つ特性 48 個のうち
実際に発火したのは 4 個**でした。残り 44 個は「書いたが一度も動いていない」状態になる。
そこで `tools/dump_damage_cases_synthetic.py` が、全 66 特性 × 全 49 道具を攻撃側・防御側の
両方に置き、フラグ（接触・音・パンチ・噛む・斬る・波動・反動・追加効果・低威力）を網羅する
技グリッドと天候・フィールド・状態異常・壁・急所・範囲・ランクの組み合わせで 23,630 ケースを作ります。
**実在しない組み合わせを含むのは意図的**で、ここで問うているのは「同じ表の 2 実装が同じ入力で
一致するか」だけです。到達可能性は Showdown に対する `diff_damage.py` の仕事。

**このケース集合が失敗しうることも確認しました**（変異テスト）。technician を 1.5 → 1.4、
オーラを 5448 → 5449 に変えると **48 / 23,630 件が不一致**として出ます。

## 楽ではなかったところ

- **このマシンには C ツールチェインが無い。** MSVC も Windows SDK も入っておらず、
  PATH の `link.exe` は coreutils のものです。`x86_64-pc-windows-gnu`（rustup が
  リンカごと配る）に切り替えて解決しましたが、**PyO3 拡張を作る道はこれで塞がります**
  ——CPython は MSVC ビルドなので、拡張モジュールも MSVC で作るのが筋。
  つまり PyO3 で統合するなら Build Tools（数 GB）の導入が前提条件です。
- **統合境界は「セルごと」では割に合わない。** PyO3 の呼び出しオーバーヘッドは
  1 呼び出し数 us で、Rust 側の計算 1.6 us と同じ桁。ダメージ計算ごとに往復すると
  35 倍が消えます。境界を引くなら `batched_payoffs` と同じ**ノード単位**
  （局面 + 両側の行動リストを渡して行列を返す）。これは既存の sim-bridge と同じ形なので、
  stdio 越しの別プロセスにすれば ABI 問題も消えます。
- **粒子軸（n>1）は移していません。** CLI の行列経路は計測上 100% が n=1 ですが、
  `narrow` と信念層は n が数百〜数千で呼びます。そこは NumPy のベクトル化が実際に効いている
  ので、Rust 側にはバッチ経路が別途必要です（未測定）。
- **移していない機能**: `_unmodelled` の申告、multiscale の粒子混在経路、`effective_damage`、
  `crit_stage` / `crit_probability`。

## 全体を移すとどれくらいか（外挿、測定ではない）

今回の実績は **Python 2,039 行 → Rust 1,664 行（0.82 倍）**。同じ比率なら残りは

| 残り | Python 行 | Rust 見込み |
|---|---|---|
| `resolve.py` | 4,114 | 約 3,400 |
| `speed.py` / `actions.py` / `position.py` / `view.py` / `narrow.py` | 2,380 | 約 1,950 |

で **約 5,300 行**。ダメージ層より難しいのは、解決器が**状態を書き換える**こと（Python の
`Pokemon.copy` が行列充填時間の 25% を占めているのは、まさにその書き換えのためです）で、
所有権が絡む分だけ行あたりの手間は増えます。一方で**ここが Rust のいちばんの取り分**でもあります。

## 走らせ方

```bash
uv run python tools/dump_damage_cases.py --out rust/cases.json
uv run python tools/dump_damage_cases_selfplay.py --games 6 --out rust/cases-field.json
uv run python tools/dump_damage_cases_synthetic.py --out rust/cases-synthetic.json
cd rust && cargo run --release -- ../configs/regulations/gen9championsvgc2026regmc.json cases-synthetic.json
uv run python tools/bench_damage_cases.py rust/cases.json --repeats 20
```
