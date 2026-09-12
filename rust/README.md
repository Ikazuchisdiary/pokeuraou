# Rust 移植：局面生成の 96.7% を占める解決器を Rust に移す

**結論**: 生成は **20〜28 倍**速くなり、**同じ乱数種で生成した対戦は 1 バイトも変わりません**。

```
3 ゲーム / 種 5      python 21.7 s -> rust 1.0 s   22.2x   試合一致 3/3
3 ゲーム / 種 1001   python 24.4 s -> rust 1.2 s   20.4x   試合一致 3/3
3 ゲーム / 種 77     python 30.2 s -> rust 1.1 s   27.5x   試合一致 3/3
```

**学習済みの価値関数でも同じです。** `data/models/*.pt` を評価軸にした生成は
GPU で **15〜18 倍**、CPU で順伝播しても **8〜10 倍**速くなり、
対戦はやはり 1 バイトも変わりません。比が hp-share より小さいのは、
解決器が速くなったあとに**残るのが torch** だからです。

```
10 ゲーム / 種 404 (cuda)  python 113.0 s -> rust 6.3 s   17.8x   試合一致 10/10
 6 ゲーム / 種 77  (cuda)  python  71.7 s -> rust 4.8 s   14.9x   試合一致 6/6
 2 ゲーム / 種 11  (cpu)   python  19.6 s -> rust 2.6 s    7.6x   試合一致 2/2
```

再現は `uv run python tools/diff_generation.py --games 3 --seed 5`、
価値関数なら `--value data/models/value-gen234.pt` を足します。

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
`thawsTarget`(96) → `mustrecharge`（1 ノードの 45% のセル）→ `recharge` 疑似技 →
`electroshot`(1.5%) → `afteryou`(1.9%) → `cursedbody`(3.3%)。
**白リストは 2 つとも反転させました**——列挙するのは「Python が名指しで扱っていて、こちらが
実装していないもの」だけ:

- 揮発性状態: 以前は `disable` 1 つ。`disable` を実装したので**今は空**です。
- 状態技: Python が完全にモデル化していない技は、Python 自身も宣言的フィールドを適用して
  申告するだけです。こちらも同じことをするので、**拒否が正しいのは Python が独自コードを
  モデル化していてこちらが未実装のときだけ**。112 技が拒否から解決に変わりました。

学習済み価値関数の生成 10 ゲーム・34,959 セルで、**拒否は 0** です。

## 検証

Python がオラクル、Showdown は Python のオラクル（`tools/diff_*.py`）。鎖は切れていません。

| 対象 | 母数 | 結果 |
|---|---|---|
| ダメージ計算（実戦 + 合成） | 30,640 ケース | **0 乖離** |
| 局面 JSON の往復 | 6,200 局面 | **全一致** |
| 1 ターン解決（狭いサンプル） | 1,190 ターン | **1,190 一致 / 誤り 0 / 拒否 0** |
| 1 ターン解決（広いサンプル） | 1,863 ターン | **1,863 一致 / 誤り 0 / 拒否 0** |
| 1 ターン解決（二段技だけを集めた） | 3,495 ターン | **3,495 一致 / 誤り 0 / 拒否 0** |
| 1 ターン解決（`cursedbody` を場に含む） | 1,154 ターン | **1,154 一致 / 誤り 0 / 拒否 0** |
| 1 ターン解決（新たに解決する状態技） | 667 ターン | **667 一致 / 誤り 0 / 拒否 0** |
| 符号化（`encode.py` の 8 配列） | 9,730 局面 | **全配列ビット一致** |
| ノード（行列）充填 | 25 ノード 12,192 セル | **拒否 0**、最大差 6.7e-16 |
| ノード充填（学習済み価値関数 + 対照軸） | 10 ノード 9,696 セル | **拒否 0**、最大差 3.9e-16 |
| 生成した対戦 | 9 ゲーム / 3 種 | **全て同一**（申告リスト含む） |
| 生成した対戦（学習済み価値関数） | 12 ゲーム / 3 種 | **全て同一** |
| 既存のテスト一式（オラクル差分込み） | 343 件 | **全通過**（ブリッジの有無どちらでも） |

「一致」は分岐・確率・結果局面の全フィールド・**申告した未対応効果**まで含みます。
最後の 1 つを比較していなかった時点で `noguard` の申告漏れが見つかったので、
比較対象に入れる価値がありました。

### 到達しえない局面は答えずに拒否する

`validate_position` が弾く局面（hp が 0..maxhp の外、`fainted` と hp の不一致）は
Rust 側も拒否します。2 実装が一致するのは関数が定義されている入力の上だけで、
ゲームが到達しない状態では Python 自身が負のダメージを出します。
**別の間違った答えを黙って返すより、拒否するほうが正しい。**
これはテストで見つけました——隠れ個体に勝手な SP を入れた手作りの局面で
0.095 の差が出て、原因は移植ではなくテスト側の局面でした。

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
| ノード充填（学習済み価値関数 + 対照軸） | 20.7 s / 10 ノード | 2.2 s | **9.4x** |
| 符号化（`encode.py`） | 66.1 us | 3.5 us | **18.9x** |
| 生成（エンドツーエンド） | 0.59〜0.92 s/ターン | 0.026〜0.037 s/ターン | **20〜28x** |
| 生成（学習済み価値関数・cuda） | 0.99〜1.20 s/ターン | 0.067〜0.071 s/ターン | **15〜18x** |
| 生成（学習済み価値関数・cpu） | 1.11〜1.22 s/ターン | 0.115〜0.160 s/ターン | **8〜10x** |

学習済み価値関数の比が小さいのは、解決器が速くなった後に**残るのが torch** だからです。
順伝播と符号化の受け取りは Python 側の仕事で、そこは移植の対象ではありません。

## CLI の 1 局面：15.95 秒 → 2.38 秒

`uv run python -m pokeuraou.cli examples/scenario-turn5.json` の内訳:

| 段階 | 前 | 後 |
|---|---|---|
| import | 0.86 | 0.94 |
| build_beliefs | 0.88 | 0.92 |
| reduce_for_position | 0.14 | 0.15 |
| **joint_classes** | **4.62** | **0.05** |
| narrow | 0.03 | 0.03 |
| **行列充填** | **9.41** | **0.27** |
| **合計** | **15.95** | **2.38** |

`joint_classes` は Rust とは無関係で、Python 側のアルゴリズムの問題でした——
`itertools.product` で 1,359,156 通りを作ってソートし、4 つだけ残していました。
各スロットが既に重み降順で結合重みが積なので、ヒープで格子を歩けば作る必要がありません。
分母も「積の和 = 和の積」で列挙不要です。同点の並びは `itertools.product` 順に揃えてあり
（報告に出るクラス名が変わらないように）、`tests/test_joint_classes.py` が総当たりと突合します。

**印字される内容は 1 行も変わりません。** 変わるのは自己申告の所要時間と、
LP の双対ギャップ（3.4e-15 対 6.2e-15、どちらも「解けた」の意味）だけです。

## 何が速いのか

**コピーです。** Python の `Pokemon.copy` は生成時間の 16% で、解決器は分岐ごと・行動ごとに
局面を複製します。だから表現をコピーのために選びました:

- 識別子はインライン（`Id`、31 バイト、**切り詰めずにパニック**する）
- ランクは固定配列、技は固定 4 枠
- 解決器が読むだけの 2 つの JSON（`abilityState`、効果の `extra`）は `Rc` の後ろ

`Position` の複製は memcpy と参照カウントの加算です。

### 複製は移植後も一番大きい

移植したあと、**Rust の解決器の中で一番大きいのも複製でした**。カウンタで測ると
1 ターンあたり 44.3 回、1 回 0.90 us——**解決器 55.2 us の 65%** です。
`cargo run --release -- clones <turns.json>` と `turns` の出力がこの 2 つを直接出します。

内訳は 2 つあり、順に潰しました:

| | 複製 1 回 | 1 ターン |
|---|---|---|
| 最初 | 0.90 us | 55.2 us |
| 控えのポケモンを `Rc` で共有（11 KB の memcpy → 参照カウント） | 0.36 us | 49.3 us |
| 文字列 5 つを `Rc<str>` に、`sides` を固定長 2 に（確保 6 回削減） | **0.22 us** | **44.5 us** |

1 段目は memcpy が理由でした——`Pokemon` は 928 バイトで両側 12 体。ターンが触るのは
場に出ている 2 体だけなので、`Rc::make_mut` は書いたものだけを複製します。
2 段目で複製は memcpy 律速ではなく**確保律速**に変わっていたので、確保を減らしました。
生成全体では 15.4x → 17.8x（種 404・10 ゲーム・学習済み価値関数）、
hp-share では 19.1x → 22.3x です。

カウンタは製品コードに残してあります。1% ほどの負担で、**次に何を直すべきかを
推測ではなく測定で決められる**ことのほうが高くつきません。

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

### 学習済み価値関数：渡るのは葉の「符号化」

パラメータ無しの評価軸（`hp-share`、`faints`）は Rust 側で計算され、**行列だけ**が帰ります。
学習した価値関数はそれができません——**入力が葉そのもの**だからです。
そこで境界の向きを変えて、葉を `encode.py` の配列にしてから渡します:

| | 渡るもの | 1 ノードあたり |
|---|---|---|
| 局面 JSON を渡す（却下） | 葉の局面 | 約 2,000 × 15 KB + パース |
| **符号化を渡す（採用）** | 8 本の配列 | 約 2,000 × 3.7 KB、パース無し |

**順伝播は torch に残します。** float32 の行列演算をもう 1 つ実装すれば加算順序が変わり、
勝率の下位桁が変われば利得が変わり、利得が変われば均衡が変わります。渡るのは入力だけです。
`encode.rs` は `encode.py` の移植で、9,730 局面 × 8 配列を**ビット単位で**突合済み
（`tools/diff_encode.py`）。速度は 3.5 us 対 66.1 us で **18.9x**。

折り畳みは Python のままです。Rust が返すのは「葉」と「畳み方」——
ふつうのセルは (開始位置, 重み列) の内積、途中で交代が入ったセルは `turn_leaves` が作る木
（偶然は平均、交代は選ぶ側の最大）——で、`fold_value` はすでにある関数です。

**対照軸も同じ往復に乗ります。** 解析は学習軸の隣にパラメータ無しの軸をもう 1 本置きますが、
葉は向こうにあるので**そちらで葉ごとに採点して**値を一緒に返します。
2 本目の列のためにノードごと帰すより安く、実測で 2.11 s 対 2.19 s（差はほぼ 0）。

## まだ Python にあるもの

ブリッジを入れたあとの生成の内訳（3 ゲーム、2.5 秒）:

| | 秒 | 割合 |
|---|---|---|
| ノード充填（Rust への往復込み） | 1.43 | 56% |
| `narrow`（候補の絞り込み） | 0.45 | 18% |
| `resolve_turn`（中断ターンの戻り） | 0.26 | 10% |
| LP | 0.15 | 6% |

次に効くのは `narrow` で、これは Python のままです。

### 拒否は 0 になりました

学習済み価値関数の生成 10 ゲーム・34,959 セルで拒否 0。そこに至るまでの 3 つ:

| 理由 | 測ったときの割合 | 中身 |
|---|---|---|
| `two-turn move: electroshot` | 1.5% | 溜め技 12 種すべて |
| `status move: afteryou` | 1.9% | 状態技の白リスト反転（112 技） |
| `ability: cursedbody` | 3.3% | `disable` 条件そのもの |

拒否されたセルは Python が埋めるので答えは変わりませんが、**そのセルだけ速くありません**。
学習済み価値関数ではもう 1 つ副作用があります——拒否されたセルの葉は、
ノード全体とは別の小さな順伝播で採点されるので、float32 の行列演算が
形に依存する分だけ値が動きます。実際 1 つの探索値が 1e-8 動いていました。
**移植の穴が数値にも出る**ということです。埋めたら消えました。

`cursedbody` は 30% で Disable を分岐します。だから Disable 条件——どの技を止めたかを
覚える揮発性状態、技スロットの `disabled` フラグ、残留処理での解除——を実装するまで
実装できませんでした。43/394 のチームが持っています。

サンプルの作り方も変えました。普通のサンプルには `electroshot` が 1 件も入っていません。
`tools/dump_turn_cases.py --only-move` / `--only-ability` は「覚えたばかりのものだけ」を
集めます。**カバー率はサンプルの性質**なので、実装した直後に必要なのは
そのサンプルです。

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

# 符号化の差分（学習済み価値関数を使うなら必須）
./target/release/pokeuraou-damage encode ../configs/regulations/gen9championsvgc2026regmb.json turns.json encoded.bin
uv run python tools/diff_encode.py rust/turns.json rust/encoded.bin

# ノードと対戦の差分（こちらが実使用の形）
POKEURAOU_RUST_NODE=1 uv run python tools/diff_node.py --games 2 --nodes 20
uv run python tools/diff_generation.py --games 3 --seed 5

# 学習済み価値関数で同じことをする
POKEURAOU_RUST_NODE=1 uv run --group learn python tools/diff_node.py \
    --games 2 --nodes 10 --value data/models/value-gen234.pt
POKEURAOU_RUST_NODE=1 uv run --group learn python tools/diff_generation.py \
    --games 6 --seed 77 --device cuda --value data/models/value-gen234.pt

# 覚えたばかりのものだけを集めて突合する（普通のサンプルには 1 件も入らない）
uv run python tools/dump_turn_cases.py --games 8 --seed 404 \
    --only-move fly,dig,dive,bounce,phantomforce,shadowforce,skyattack,meteorbeam \
    --out rust/turns-twoturn.json
uv run python tools/dump_turn_cases.py --games 10 --seed 404 \
    --value data/models/value-gen234.pt --only-ability cursedbody \
    --out rust/turns-cursedbody.json

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
