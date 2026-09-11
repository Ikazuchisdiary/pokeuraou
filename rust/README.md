# Rust 移植：局面生成の 96.7% を占める解決器を Rust に移す

**結論**: 生成は **19〜24 倍**速くなり、**同じ乱数種で生成した対戦は 1 バイトも変わりません**。

```
3 ゲーム / 種 5      python 41.4 s -> rust 2.1 s   19.9x   試合一致 3/3
3 ゲーム / 種 1001   python 47.9 s -> rust 2.5 s   18.9x   試合一致 3/3
3 ゲーム / 種 77     python 59.1 s -> rust 2.5 s   24.0x   試合一致 3/3
```

再現は `uv run python tools/diff_generation.py --games 3 --seed 5`。

## なぜ解決器だったのか

移植先は推測ではなく実測で決めました。**カウンタで測った**生成 1 ゲームの内訳:

| | 秒 | 割合 |
|---|---|---|
| `resolve_turn` | 35.04 | **96.7%** |
| うち `damage.calculate` | 7.49 | 20.7% |
| うち `view.battler` | 3.74 | 10.3% |
| `narrow` | 0.43 | 1.2% |
| `equilibrium.solve`（LP） | 0.02 | **0.1%** |

**cProfile はこれに同意しません。** 同じワークロードで実行時間の 16% を scipy の HiGHS
オプション検証（`_highs_wrapper.check_option`）に配分します。単体ベンチでは 1 回の solve が
4.5 ms、つまり 0.1% です。C 拡張の呼び出しを歪めないほうのプロファイラを信じます。

## 推測せず拒否する

移植の契約は「**Python と同じ答え**」です。だから未対応に出会ったら推測せず `Err(reason)` を返し、
呼び出し側は Python の答えをそのまま使います。これが部分移植を「測れるだけ」ではなく
「使える」ものにします:

- Rust が解決したターンは Python の答えそのもの
- 拒否したターンは Python の時間がかかるだけ（余計な損はない）
- **拒否理由は数えられる**ので、次に実装すべきものは推測ではなく影響の大きさで決まる

実際この数えが次を順に名指ししました: `pixilate`(730) → `poisontouch`(177) →
`thawsTarget`(96) → `mustrecharge`（1 ノードの 45% のセル）→ `recharge` 疑似技。
最後の 2 つは「この移植が実装済みの効果」を白リストに書き忘れていただけで、
**白リストを反転させて解決しました**——列挙するのは「Python が名指しで扱っていて、こちらが
実装していないもの」だけ（揮発性状態では `disable` 1 つ）。

## 検証

Python がオラクル、Showdown は Python のオラクル（`tools/diff_*.py`）。鎖は切れていません。

| 対象 | 母数 | 結果 |
|---|---|---|
| ダメージ計算（実戦 + 合成） | 30,640 ケース | **0 乖離** |
| 局面 JSON の往復 | 6,200 局面 | **全一致** |
| 1 ターン解決（狭いサンプル） | 1,190 ターン | **1,190 一致 / 誤り 0 / 拒否 0** |
| 1 ターン解決（広いサンプル） | 1,863 ターン | **1,863 一致 / 誤り 0 / 拒否 0** |
| ノード（行列）充填 | 25 ノード 12,192 セル | **拒否 0**、最大差 6.7e-16 |
| 生成した対戦 | 9 ゲーム / 3 種 | **全て同一** |

「一致」は分岐・確率・結果局面の全フィールド・**申告した未対応効果**まで含みます。
最後の 1 つを比較していなかった時点で `noguard` の申告漏れが見つかったので、
比較対象に入れる価値がありました。

**カバー率はサンプルの性質です。** 最初の 2 ゲームのサンプルでは 87.2% でしたが、
別の 10 ゲームでは同じビルドが 42.4% でした。退行ではなく、最初のサンプルに
メガサーナイトが入っていなかっただけです（`pixilate` だけで 730 ターンが拒否）。
1 つのサンプルのカバー率は、そのサンプルについての数字でしかありません。

### 1 ULP の差について

ノードの差分で、採点されたセルの 73% は**ビット単位で一致**し、残りは最大 6.7e-16 ずれます。
これは numpy の `values @ weights`（内積）と逐次加算の**加算順序の違い**です。
解決器そのものはビット単位で一致しています（局面・分岐・確率すべて）。
そして実際に読まれる答えである均衡は:

```
均衡値が動いた最大   6.7e-16
採用頻度が動いた最大 1.2e-14
```

CLI が印字する最下位桁より 12 桁下です。生成した対戦が完全一致することが、
この差が何も変えないことの実地の証拠になっています。

## 速度（交互実行・各 5 回の最小値）

絶対値は負荷で 2 倍動くので、比は必ず両側を同条件で交互に測ります。

| | Python | Rust | 比 |
|---|---|---|---|
| `damage.calculate` | 103.8 us | 1.54 us | **67.2x** |
| `resolve_turn`（狭いサンプル） | 3482.9 us | 82.9 us | **42.0x** |
| `resolve_turn`（広いサンプル） | 3378.3 us | 89.6 us | **37.7x** |
| ノード充填（24x24、2 評価軸） | 56.0 s / 25 ノード | 1.4 s | **40.3x** |
| 生成（エンドツーエンド） | 1.1〜1.8 s/ターン | 0.056〜0.079 s/ターン | **19〜24x** |

## 何が速いのか

**コピーです。** Python の `Pokemon.copy` は生成時間の 16% で、解決器は分岐ごと・行動ごとに
局面を複製します。だから表現をコピーのために選びました:

- 識別子はインライン（`Id`、31 バイト、**切り詰めずにパニック**する）
- ランクは固定配列、技は固定 4 枠
- 解決器が読むだけの 2 つの JSON（`abilityState`、効果の `extra`）は `Rc` の後ろ

`Position` の複製は memcpy と参照カウントの加算です。

**そして粒子軸がありません。** CLI の行列経路は実測で 100% が n=1（`damage.calculate` の
呼び出し 3,855 件すべて）。Python が運ぶベクトル化は、この経路では幅 1 の NumPy 配列を作る
純粋なオーバーヘッドでした。

## 使い方

既定では**オフ**です。何も設定しなければ何も変わりません。

```bash
cd rust && cargo build --release

export POKEURAOU_RUST_NODE=1          # 使う
uv run python tools/selfplay.py --games 100
```

壊れていたら黙って Python に戻り、理由を 1 度だけ表示します。実行を失敗させません。

### 境界はノード単位

`batched_payoffs` はもともとこの計画のノード境界（局面と両側の行動リストが入り、行列が出る）
なので、プロセス境界を同じ場所に置けば**葉は一切渡りません**。渡るのは入りに局面 1 つ、
帰りに 576 個の float だけです。

試合を進める `resolve_turn`（exact 予算）は別扱いで、**重みだけ**を先に返して
Python 側が自分の乱数生成器で 1 つ引き、その 1 局面だけを取りに行きます。
これが「対戦がビット単位で同じ」を保つ仕組みです——
`numpy.random.Generator.choice` が乱数列そのものなので、引く回数と引き方を変えられません。

### 渡らないもの

**学習した価値関数は渡りません。** 入力が葉そのもので、1 ノードの葉は数十 MB の JSON です。
これを渡すには `encode.py` も移植して特徴ベクトルだけを渡す必要があります。
今日の生成が `hp-share` で動いているのは、このレギュレーションにまだ学習済みモデルが
無いからで、そこは移植の限界ではなく現状の設定です。

## まだ Python にあるもの

ブリッジを入れたあとの生成の内訳（3 ゲーム、2.5 秒）:

| | 秒 | 割合 |
|---|---|---|
| ノード充填（Rust への往復込み） | 1.43 | 56% |
| `narrow`（候補の絞り込み） | 0.45 | 18% |
| `resolve_turn`（中断ターンの戻り） | 0.26 | 10% |
| LP | 0.15 | 6% |

次に効くのは `narrow` で、これは Python のままです。

## 走らせ方

```bash
# 差分の材料を作る
uv run python tools/dump_damage_cases.py --out rust/cases.json
uv run python tools/dump_damage_cases_selfplay.py --games 6 --out rust/cases-field.json
uv run python tools/dump_damage_cases_synthetic.py --out rust/cases-synthetic.json
uv run python tools/dump_turn_cases.py --games 2 --out rust/turns.json

# 差分を取る
cd rust && cargo build --release
./target/release/pokeuraou-damage damage    ../configs/regulations/gen9championsvgc2026regmc.json cases-synthetic.json
./target/release/pokeuraou-damage roundtrip turns.json
./target/release/pokeuraou-damage turns     ../configs/regulations/gen9championsvgc2026regmc.json turns.json

# ノードと対戦の差分（こちらが実使用の形）
POKEURAOU_RUST_NODE=1 uv run python tools/diff_node.py --games 2 --nodes 20
uv run python tools/diff_generation.py --games 3 --seed 5

# 白リストの再生成（エンジンが効果を覚えたら必ず）
uv run python tools/port_coverage.py --rust
uv run python tools/port_coverage.py --rust-modelled
```

## 環境の注意

このマシンには C ツールチェインがありません（MSVC も Windows SDK も無く、PATH の
`link.exe` は coreutils のもの）。`x86_64-pc-windows-gnu`（rustup がリンカごと配る）で
解決しています。**この道では PyO3 拡張は作れません**——CPython は MSVC ビルドなので、
拡張も MSVC で作るのが筋です。だから境界はプロセス（JSONL over stdio）で、
これは既存の sim-bridge と同じ形でもあります。
