# tools/ の棚卸し 〔9/19 制定〕

`tools/*.py` は測定の出典である。数字だけを残して道具を消すと、その数字が何を測ったか
検証できなくなる。**この表は削除の根拠ではない。** 退避であって削除ではなく、退避した
ものも `tools/oneshot/` にそのまま残っている。

> **9/24 IKA-212 で消した道具（下の表の行はそのまま残す）。** Python の解決器
> （`src/pokeuraou/resolve.py`）を消したので、それを前提にする道具を消した。消した本文は
> git の 471b98e で読める。理由は TODO.md の「9/24 — IKA-212」の節の表:
> `bench`・`bench_turn_cases`・`bench_damage_cases`・`branch_dedup`・`cells_needed`・
> `count_resolver_calls`・`diff_commands`・`diff_narrow`・`diff_node`・`diff_solve_node`・
> `dump_damage_cases`・`dump_damage_cases_selfplay`・`dump_damage_cases_synthetic`・
> `dump_turn_cases`・`hidden_dominance`・`ko_branch_count`・`narrow_effect`・`narrow_regret`・
> `oneshot/depth_effect`・`profile_resolve`・`regret_playout`・`seat_bias`・`selfplay_budget`・
> `width_vs_ranking`。port に向け直したもの（`diff_turn`・`diverge_report`・`diff_replacement`・
> `show_game`・`why_action` ほか）は残っている。

## 数えた結果（IKA-25 で再測定）

| 数えたもの | 本数 |
|---|---|
| `tools/*.py`（`oneshot/` を含む） | 102 |
| 課題本文の定義（`TODO.md` / `GENERATIONS.md` / `scratchpad/` / `tools/*.sh` のみ）で参照なし | 65 |
| 記録全体（上記 + `README.md` / `docs/` / `src/` / `tests/` / `configs/`）で参照なし | 40 |
| ↑のうち他のツールから import または名指しされているもの | 17 |
| ↑のうちどこからも参照が無いもの | 23 |
| **`tools/oneshot/` へ退避した** | **6** |
| **据え置き（判断保留）** | **17** |

課題本文は「102本のうち66本」と書いている。総数 102 は一致。未参照は本文と同じ
定義で数えて **65本** で、本文の66本と1本食い違う。食い違いの向きも幅も
小さいが、**この表の数字は上の定義で自分で数え直したもの**であって、本文の転記ではない。
`scripts/` はこのリポジトリに存在しない。

⚠ **「参照が無い＝死んでいる」ではない。** その場で叩いただけの道具は記録に参照が残らない。
この数字は棚卸しの必要性を示すだけで、削除の根拠にはならない。

## 凡例

**日付の出典**: 「最終更新」は `git log -1 --date=short` の日付、つまり**そのファイルを最後に
編集した日**である。**「最後に使った日」ではない。** このリポジトリは実行日を残す記録を
持っていないので、どの道具についても「最後に走らせた日」は機械的には取れない。取れない
ものを埋めるより空けておくほうがよいので、実行日の列は作っていない。`TODO.md` の節見出し
（〔9/19〕など）から言及日を拾うことは機械的には可能だが、節の日付は節のものであって実行の
ものではなく、実際に作成日より前の日付が付く例が出たので採らなかった。退避した6本の日付は
**退避前の最終編集日**である（退避そのもののコミットで `git log -1` は 9/19 に動く）。

**参照**: その道具の名前（`<名前>.py` または `tools/<名前>`）が現れる場所。

| 記号 | 場所 |
|---|---|
| T | `TODO.md` |
| G | `GENERATIONS.md` |
| s | `scratchpad/*.sh` |
| h | `tools/*.sh` |
| R | `README.md` |
| W | `docs/wf-summary-0917.txt` |
| S | `src/` |
| X | `tests/` |
| C | `configs/` |
| — | どこにも無い |

**判定**:

| 判定 | 意味 |
|---|---|
| 現役 | 記録のどこかから参照されている。動かさない |
| 据え置き（他ツール依存） | 記録からの参照は無いが、他のツールが import または名指ししている。動かすと壊れる |
| 判断保留 | 記録からも他のツールからも参照が無い。**退避の根拠が弱いので動かさなかった** |
| **退避** | 一度きりの分析と判断し、`tools/oneshot/` へ移した |

**「問い」の出典**: 各ファイルの docstring 冒頭と、実際のコードから取った。ファイル名から
推測したものは無い。docstring が問いを書いていないものは無かった（102本すべてが冒頭1行に
問いか役割を書いている）。

## 表

「使う側」は、そのツールを import または本文・ヘルプ文字列で名指ししている他のツール。

⚠ **上の「数えた結果」は 9/19 の数字**であって、この表の行数ではない。9/19 以降に増えた
道具は表に足しているが、**102 という総数は数え直していない**。9/22 時点で表に無いものが
1本ある: `ci_skip_audit.py`（CI が何を飛ばしたかを見る。9/19 以降に追加）。

| ツール | 何の問いに答えるか | 最終更新 | 参照 | 使う側 | 判定 |
|---|---|---|---|---|---|
| `agent_drift.py` | 生成と違うエージェントを組み立てている道具はどれか | 9/19 | T | — | 現役 |
| `archetype_audit.py` | アーキタイプ一覧は、使用率データが測ったメタを説明できているか | 9/10 | RC | — | 現役 |
| `asymmetry.py` | 勝率はチームの差か、各席の探索のかけ方の差か | 9/10 | R | — | 現役 |
| `bayes_auc.py` | モデルが完璧なら turn-1 の AUC はどこまで上がりうるか | 9/18 | T | — | 現役 |
| `belief_reach.py` | 配分の推定分布は、実在の人が持ち込んだ構築を表現できるか | 9/18 | T | — | 現役 |
| `belief_report.py` | 推定分布の層が実際いくら稼ぐか（主張ではなく測定で） | 9/10 | R | — | 現役 |
| `bench.py` | 速度が道具の出来ることを決めている部品の、素の所要時間 | 9/10 | R | — | 現役 |
| `bench_damage_cases.py` | Rust 移植の比較対象ケースでの Python 側の所要時間 | 9/12 | — | — | 判断保留 |
| `bench_generation.py` | この機械は何プロセスでどれだけ速く対局を生成できるか | 9/17 | — | `agent_drift`、`bench_scaling`、`bench_workers` | 据え置き（他ツール依存） |
| `bench_scaling.py` | この機械が実際に欲しいワーカー本数はいくつか（推測ではなく測定） | 9/10 | R | — | 現役 |
| `bench_turn_cases.py` | Rust 移植の比較対象ターンでの Python 側の所要時間 | 9/12 | — | — | 判断保留 |
| `bench_workers.py` | 生成プロセスを同時に何本走らせるべきか | 9/17 | G | — | 現役 |
| `blind_opponent.py` | 選出解が相手にこちらの配分を見せていることの代償はいくらか | 9/19 | Ts | — | 現役 |
| `blunders.py` | ルール上通らない手を、記録した対局からモデル抜きで数える | 9/18 | — | — | 判断保留 |
| `book_against_uniform.py` | 選出キャッシュを持たない相手に対して、その助言はいくらの価値か | 9/18 | T | `book_seed_spread` | 現役 |
| `book_check.py` | 選出キャッシュの均衡は、一様抽選より多く勝つか（フィールド全体で） | 9/19 | TGhW | `counterplay` | 現役 |
| `book_seed_spread.py` | 印字される選出助言はデータの性質か、学習シードの性質か | 9/18 | TX | `lead_profile` | 現役 |
| `budget_effect.py` | 探索予算をどこで削ると何を失い、何が買えるか | 9/10 | RS | `narrow_as_built`、`narrow_effect`、`policy_ceiling`、`policy_pool_recall`、`seat_bias`、`width_vs_ranking` | 現役 |
| `cell_noise.py` | 同じ局面の勝率を、2つの学習シードはどれだけ離して置くか | 9/18 | T | — | 現役 |
| `cells_needed.py` | 均衡は実際いくつのセルを必要とするか | 9/12 | S | `agent_drift` | 現役 |
| `cooc_report.py` | 2体同時出現のチームモデルは、作成元のメタを再現できているか | 9/10 | RSX | — | 現役 |
| `counterplay.py` | 解いた選出キャッシュは、人が想定する対策について何を信じているか | 9/11 | — | — | 判断保留 |
| `coverage.py` | 使用率で重み付けした仕様カバレッジはどれだけか | 9/10 | RS | — | 現役 |
| `cycle_match.py` | ミラーの三すくみを、価値関数に聞かず実際に打って確かめる | 9/11 | h | `counterplay` | 現役 |
| `tools/oneshot/depth_effect.py` | 2手目の読みは答えを実際に変えるか、どのくらいの頻度で | 9/12 | — | — | **退避** |
| `diff_damage.py` | ダメージ計算は Showdown と一致するか | 9/11 | R | `dump_damage_cases`、`dump_damage_cases_synthetic` | 現役 |
| `diff_encode.py` | Rust のエンコーダは Python と同じ配列を出すか | 9/12 | — | — | 判断保留 |
| `diff_generation.py` | Rust ノードの有無で、生成した対局は同一になるか | 9/13 | — | `agent_drift` | 据え置き（他ツール依存） |
| `diff_narrow.py` | 移植側が採点しても、候補手は同一に出るか | 9/12 | — | `agent_drift` | 据え置き（他ツール依存） |
| `diff_node.py` | ノード全体が Rust 移植経由でも同一に出るか | 9/12 | S | `agent_drift` | 現役 |
| `diff_order.py` | 行動順は Showdown と一致するか | 9/10 | RX | — | 現役 |
| `diff_replacement.py` | 交代出しフェーズは Showdown と一致するか | 9/10 | R | — | 現役 |
| `diff_solve_node.py` | ノードを埋めずに解いても同じ答えに届くか | 9/13 | S | `agent_drift` | 現役 |
| `diff_speed.py` | 実効素早さは Showdown と一致するか | 9/10 | RX | — | 現役 |
| `diff_turn.py` | 1ターン解決器の全体が Showdown と一致するか | 9/19 | TR | `diff_replacement`、`diverge_report` | 現役 |
| `diverge_report.py` | 解決器の無言の乖離の原因を、統計的リフトで順に並べる | 9/10 | RX | — | 現役 |
| `tools/oneshot/does_the_sweep_happen.py` | 人が実際に打つ筋書き（削り→抜き）はデータに現れるか | 9/17 | — | — | **退避** |
| `dump_damage_cases.py` | 実際の行列充填が行う全ダメージ呼び出しと、Python が返した答え | 9/12 | — | `dump_damage_cases_selfplay`、`dump_damage_cases_synthetic` | 据え置き（他ツール依存） |
| `dump_damage_cases_selfplay.py` | 同じダンプを、フィールド相手の自己対戦から取る | 9/12 | — | — | 判断保留 |
| `dump_damage_cases_synthetic.py` | 効果表の全修正子を最低1回発火させる合成ケースを作る | 9/12 | — | — | 判断保留 |
| `dump_turn_cases.py` | 生成1本の全 resolve_turn 呼び出しと、Python が返した分岐 | 9/12 | — | `agent_drift` | 据え置き（他ツール依存） |
| `encode_dataset.py` | 自己対戦を一度だけ配列に符号化する（学習が JSON を再解析しないように） | 9/19 | sRS | — | 現役 |
| `fetch_priors.py` | Smogon 使用率統計を取ってくる | 9/10 | RSX | `archetype_audit`、`asymmetry`、`belief_reach`、`belief_report`、`cooc_report`、`coverage`、`diff_damage`、`diff_order`、`diff_replacement`、`diff_speed`、`diff_turn`、`diverge_report`、`dump_damage_cases_selfplay`、`matchup`、`selection`、`selfplay`、`solve_selection_book` | 現役 |
| `fetch_standings.py` | 大会順位表を、チーム込みで Reportworm の公開 API から取ってくる | 9/10 | RX | `matchup`、`selection`、`selfplay`、`solve_selection_book`、`standings_report` | 現役 |
| `forced_handoff.py` | 均衡が受け渡しをゼロと値付けしているのは正しいか | 9/17 | T | — | 現役 |
| `generate_queue.py` | 対局をブロック配分ではなく1本のキューで生成する | 9/17 | TshS | `match_queue`、`selfplay` | 現役 |
| `generation_match.py` | ある世代の探索を別世代の探索と戦わせる（唯一公平な比較） | 9/19 | TGhRSX | `match_queue`、`paired_match`、`served_matches_direct` | 現役 |
| `hidden_dominance.py` | 同じ優越の問いを、プレイヤーが実際に持っていた情報の下で問う | 9/19 | — | `human_baseline` | 据え置き（他ツール依存） |
| `human_baseline.py` | 人が答えを知っている局面を、葉と突き合わせる | 9/19 | Ts | `hidden_dominance`、`ko_branch_count`、`why_action` | 現役 |
| `inference_server.py` | 葉をワーカーの代わりに持つ（ワーカーがモデルを抱えずに済むように） | 9/17 | — | `generate_queue`、`match_queue`、`served_matches_direct`、`served_throughput`、`worker_growth` | 据え置き（他ツール依存） |
| `ko_branch_count.py` | 確定KOでも分岐は16本あるのか | 9/19 | — | — | 判断保留 |
| `lead_profile.py` | 解いた各選出キャッシュが六体のどれを連れ、どの2体で先発するか | 9/18 | T | `book_seed_spread` | 現役 |
| `leaf_calibration.py` | 静的な葉の見積もりは、探索の答えからどこでどれだけ離れているか | 9/19 | Ts | — | 現役 |
| `match_queue.py` | 対局を配分ではなく手渡しする両席対戦 | 9/19 | Ts | `generate_queue`、`served_throughput` | 現役 |
| `match_result.py` | 1つの対戦を読み戻す（席別勝率・対・どの条件で打たれたか） | 9/19 | s | — | 現役 |
| `tools/oneshot/matchup.py` | 勝率は相性か、機械側の偏りか | 9/10 | — | — | **退避** |
| `mirror_check.py` | 各プールのミラー行を、記憶ではなく対局から再計算する | 9/18 | — | — | 判断保留 |
| `mirror_cycle.py` | 価値関数はミラーの既知の三すくみを再現するか | 9/11 | — | `cycle_match` | 据え置き（他ツール依存） |
| `names_report.py` | 日本語名表がどこまで被覆していて、何が英語のまま残っているか | 9/10 | RSC | — | 現役 |
| `narrow_as_built.py` | 絞り込みの代価を、`narrow` が実際に作る候補手の上で測る | 9/16 | T | — | 現役 |
| `narrow_effect.py` | 絞り込みが捨てるものを、均衡質量として測る | 9/11 | — | `width_match` | 据え置き（他ツール依存） |
| `narrow_regret.py` | 候補手の代価を、探索自身の評価に照らして測る | 9/12 | — | `regret_playout`、`selfplay` | 据え置き（他ツール依存） |
| `noise_cost_model.py` | セル誤差±11点は勝率でいくらか（測定ではなくモデル） | 9/19 | — | — | 判断保留 |
| `ordering_check.py` | 均衡が重く置いた選出は、本当に良いほうか（対局を打たずに） | 9/19 | — | `ordering_result` | 据え置き（他ツール依存） |
| `ordering_result.py` | 均衡が重く置いた選出は、盤上で本当に多く勝つか | 9/19 | Ts | `match_result` | 現役 |
| `paired_match.py` | 両席対戦を対として読み、2つのソルバの答えを直接比べる | 9/13 | TW | — | 現役 |
| `paired_result.py` | キュー方式の対戦を、それが本来もつ対の設計として読む | 9/19 | Ts | — | 現役 |
| `policy_ceiling.py` | 並べ替えが完璧なら、候補手はどこまで狭くできるか | 9/13 | — | `narrow_as_built`、`policy_dataset`、`policy_pool_recall`、`policy_train` | 据え置き（他ツール依存） |
| `policy_dataset.py` | 記録した対局を、並べ替えの学習データに変える | 9/13 | SX | — | 現役 |
| `policy_pool_recall.py` | 学習した並べ替えは、全部を見せられても働くか | 9/13 | T | — | 現役 |
| `policy_train.py` | 記録済みの均衡から並べ替えを学習する | 9/13 | — | `generation_match`、`policy_ceiling` | 据え置き（他ツール依存） |
| `pool_games.py` | 対戦を、ワーカーの要約ではなく記録した対局から集計する | 9/16 | — | — | 判断保留 |
| `pool_matches.py` | 独立した1対1のランを、席別と全体の1つの判定に集計する | 9/11 | h | `book_check`、`cycle_match`、`pool_games` | 現役 |
| `port_coverage.py` | Rust 移植が無視してよい id と、Python が模擬していると主張する id | 9/12 | — | `port_gate_audit` | 据え置き（他ツール依存） |
| `port_gate_audit.py` | 移植が実装しているのにゲートが弾く効果と、ゲートは通すのに実装が無い名前 | 9/22 | TX | — | 現役 |
| `profile_generation.py` | 学習済みの葉を入れた生成対局で、時間はどこに行っているか | 9/11 | — | `agent_drift` | 据え置き（他ツール依存） |
| `profile_resolve.py` | ターン解決器の時間はどこに行っているか | 9/10 | R | `profile_generation` | 現役 |
| `ratings.py` | 記録した全対戦から、全エージェントを1つの尺度に載せる | 9/19 | TGW | — | 現役 |
| `regret_playout.py` | 絞り込みが落とした手は本当に良かったのか、良く見えただけか | 9/12 | — | `selfplay` | 据え置き（他ツール依存） |
| `resume_generate.py` | 記録した終盤局面から対局を生成する（自己対戦が滅多に届かないので） | 9/17 | TGh | — | 現役 |
| `roster_from_standings.py` | 大会の六体を自陣ロスターとして書き出す | 9/19 | — | — | 判断保留 |
| `scenario_from_book.py` | 選出キャッシュが想定する turn-1 局面を、解析器が読める形に組み立てる | 9/11 | — | — | 判断保留 |
| `search_optimism.py` | 探索の楽観は、絞り込みの請求書か | 9/19 | — | — | 判断保留 |
| `seat_bias.py` | どちらかの席が構造的に有利か（対局を打たずに厳密に） | 9/11 | — | — | 判断保留 |
| `selection.py` | 相手1チーム分の 6→4 選出を解く | 9/10 | R | — | 現役 |
| `selection_check.py` | 選出ソルバの助言は、本当に勝率を上げるか（1チーム分で） | 9/19 | TGsRC | `book_check`、`ordering_check` | 現役 |
| `selfplay.py` | 自己対戦を生成して JSONL で書く | 9/17 | TshRW | `agent_drift`、`generate_queue`、`profile_generation` | 現役 |
| `selfplay_analyse.py` | 自己対戦データに何が入っていて、学習した価値関数にどれだけ余地があるか | 9/10 | R | — | 現役 |
| `selfplay_budget.py` | 今の解決器で自己対戦をどれだけ賄えるか | 9/10 | RS | — | 現役 |
| `tools/oneshot/served_matches_direct.py` | 推論サーバ経由の対戦は、経由しない対戦と同一の対局を打つか | 9/16 | — | — | **退避** |
| `tools/oneshot/served_throughput.py` | サーバ経由と直接を、同じ幅・同じ機械・連続で比べたスループット | 9/17 | — | — | **退避** |
| `show_game.py` | 記録した対局を、日本語の読めるログとして描く | 9/19 | T | `solve_selection_book` | 現役 |
| `solve_selection_book.py` | フィールドの相手ごとに 6→4 選出を解いてキャッシュする | 9/18 | ThW | `selfplay` | 現役 |
| `standings_report.py` | 大会プールの中身と、ラダー使用率との違い | 9/10 | R | — | 現役 |
| `sweep_value.py` | 価値関数の設定をスイープする（学習は安く、生成は高いので） | 9/10 | TWS | — | 現役 |
| `train_value.py` | 価値関数を学習し、意味のある2つの基準線に照らして報告する | 9/19 | TsRWSX | `selection`、`selfplay` | 現役 |
| `what_the_data_lacks.py` | 学習データに、遅い筋書きが勝った対局は入っているか | 9/17 | T | `does_the_sweep_happen` | 現役 |
| `what_the_leak_buys.py` | 相手の控え2体を探索に教えるのをやめると、何が変わるか | 9/17 | T | `hidden_dominance` | 現役 |
| `why_action.py` | ある手の価値はどこから来ているか（列ごとの分解） | 9/19 | — | — | 判断保留 |
| `tools/oneshot/why_no_handoff.py` | 削りが終わったとき、抜き手への交代はそもそも候補手にあるか | 9/17 | — | — | **退避** |
| `width_match.py` | 広い候補手と狭い候補手を、他をすべて同一にして戦わせる | 9/19 | h | — | 現役 |
| `width_vs_ranking.py` | 広い候補手は、よく並べた狭い候補手が買えないものを買っているか | 9/13 | — | `policy_ceiling` | 据え置き（他ツール依存） |
| `worker_growth.py` | ワーカーは長く打つほど、なぜ重くなるのか | 9/17 | — | `agent_drift` | 据え置き（他ツール依存） |

## 退避したもの（6本）

`tools/oneshot/` へ移した。**消していない。** 結果の再現に要る。

退避の条件は4つ全部を満たすこととした。ひとつでも欠けたら動かしていない。

1. 記録全体（`TODO.md` / `GENERATIONS.md` / `README.md` / `docs/` / `scratchpad/` /
   `tools/*.sh` / `src/` / `tests/` / `configs/`）のどこからも参照が無い
2. 他のツールから import されていない（import があれば移動で壊れる）
3. 他のツールの本文・ヘルプ文字列でも名指しされていない（名指しがあれば追随修正が要る）
4. docstring が「一度きりの問い」を書いている —— 再実行する器具ではなく、一度答えて済んだ問い

| ツール | 退避の理由 |
|---|---|
| `depth_effect.py` | 「1時間の対戦を打つ前に、深さが効くかを切り分ける」と docstring 自身が書いている。その切り分けのための一回きりの診断 |
| `does_the_sweep_happen.py` | `what_the_data_lacks.py` が数え損ねた形を数え直した一回きりの訂正。後続（`what_the_data_lacks` 側）は `TODO.md` に残っている |
| `matchup.py` | 「勝率62.3%は相性か機械の偏りか」を一度切り分けるための道具。9/10 以降更新も参照もない |
| `served_matches_direct.py` | 推論サーバに葉を移すときの**受け入れ条件**。サーバは採用済み（`inference_server.py` は5本から使われている）で、受け入れ判定は済んでいる |
| `served_throughput.py` | 「比較できない数字が転がっている」ので同条件で並べ直した一回きりの計測。docstring 自身がそう書いている |
| `why_no_handoff.py` | 受け渡しが無い理由を3つに切り分けた一回きりの診断。後続の `forced_handoff.py` は `TODO.md` に残っている |

### 退避で直した参照

移動で壊れるのは**そのファイル自身が持っているパス**だけだった（外部からの参照は定義上ゼロ）。
移動後に `grep -rn "tools/<名前>"` をリポジトリ全体にかけ、見つかったのは自分自身の中の
使用例だけである。直したのは次の2種類。

| 直したもの | 内容 | 対象 |
|---|---|---|
| ルート解決 | `Path(__file__).resolve().parents[1]` → `parents[2]` | `depth_effect` / `matchup` / `served_matches_direct` / `served_throughput` / `why_no_handoff` |
| docstring の使用例 | `tools/<名前>.py` → `tools/oneshot/<名前>.py` | 退避した6本すべて |

`parents[1]` は `tools/` から見たリポジトリルートで、`src` も `configs` もそこから引いている。
1階層深くなったので放置すると `import pokeuraou` が全滅する。`does_the_sweep_happen.py` は
標準ライブラリしか使っていないので、直したのは使用例1行だけ。

`served_matches_direct.py` と `served_throughput.py` は `ROOT / "tools" / ...` で
`generation_match.py` / `match_queue.py` / `inference_server.py` を起動する。これらは
`tools/` に残したので、`ROOT` さえ正しければパスは正しいままである。

## 同じ問いの道具が複数あるもの

**名指しまでで、統合はしない。** どちらを残すかは測定の履歴を読む判断が要る。
いま同じ問いに複数の道具があること自体が、答えを引くときに取り違えを生む。

| 問い | 道具 | 違い |
|---|---|---|
| この機械は生成プロセスを何本走らせるべきか | `bench_generation` / `bench_scaling` / `bench_workers` | 3本が同じ問いに答える。記録に残っているのは `bench_workers`（G）だけで、残り2本は互いを名指ししているだけ |
| 対戦を1つの判定に集計する | `pool_matches` / `pool_games` | 前者はワーカーの要約行から、後者は記録した対局から。`pool_games` の docstring が「死んだワーカーの対局が見えなくなる」と違いを書いている |
| 両席対戦を対として読む | `paired_match` / `paired_result` / `match_result` | ブロック方式 / キュー方式 / 1対戦の読み戻し。3本とも `TODO.md` か `scratchpad/*.sh` から使われている（`match_result` は本課題では触っていない） |
| 絞り込みの代価 | `narrow_effect` / `narrow_as_built` | 同じ問いを違う候補手の上で測る（理論上の幅 / `narrow` が実際に作る幅） |
| 絞り込みが落とした手は良かったか | `narrow_regret` / `regret_playout` | 探索自身の評価で見積もる / 実際に打って確かめる |
| 学習シードで答えがどれだけ動くか | `cell_noise` / `book_seed_spread` | セル単位 / 印字される選出助言単位。同じ不安定さを2つの粒度で測る |
| 選出キャッシュの助言はいくらの価値か | `book_check` / `book_against_uniform` / `selection_check` | フィールド全体・盤上 / 一様抽選相手 / 1チーム分。`book_check` の docstring が `selection_check` との射程の違いを明記している |
| ミラーは 50% か（探索・解決器・評価器の較正） | `mirror_check` / `seat_bias` / `mirror_cycle` / `cycle_match` | 対局から再計算 / 対局を打たずに厳密に / 三すくみをモデルに聞く / 三すくみを実際に打つ |
| 隠れ情報はいくらの価値か | `what_the_leak_buys` / `hidden_dominance` / `blind_opponent` | 控え2体を教えるのをやめる / 同じ優越を実際の情報下で / 配分を相手に見せる代償。3本とも 9/17〜9/19 に立て続けに増えた |
| Rust 移植は同じ答えを出すか | `diff_node` / `diff_generation` / `diff_narrow` / `diff_solve_node` / `diff_encode` | 層ごとに1本ずつ。族としては意図的な分割で、重複ではない（記録しておく） |
| 削り→抜きの筋書き | `does_the_sweep_happen` / `what_the_data_lacks` / `why_no_handoff` / `forced_handoff` | データに現れるか / 遅い筋書きが勝った対局はあるか / 候補手に載っているか / 均衡のゼロは正しいか。4本で1つの話を追っている |

### 課題本文が挙げた取り違え

本文は「今日それが1回起きた: 順位パネルの問いの取り違え」と書いている。該当するのは
`ordering_check`（対局を打たずに、均衡が重く置いた選出は良いほうか）と
`ordering_result`（盤上で本当に多く勝つか）の対である。名前が近く、答える問いが違う。
`ordering_result` は `TODO.md` と `scratchpad/run2.sh` / `run3.sh` から使われており、
`ordering_check` はその入力を吐く側なので、どちらも動かしていない。

## 判断保留（17本）

記録からも他のツールからも参照が無いが、**退避の根拠が弱いので動かさなかった**もの。
迷ったら動かさない。

| ツール | 動かさなかった理由 |
|---|---|
| `bench_damage_cases` / `bench_turn_cases` | Rust 移植の比較対象。`rust/` の作業が続く限り再実行する器具で、一度きりの分析ではない |
| `port_coverage` / `diff_encode` | 同上。`port_coverage` は `rust/src/inert.rs` などを生成する側でもある |
| `dump_damage_cases_selfplay` / `dump_damage_cases_synthetic` | `dump_damage_cases` を import する族。族ごと `tools/` に残す |
| `blunders` / `mirror_check` / `seat_bias` | 「器具」であって分析ではない。モデルや解決器が変わるたびに再実行する種類のもの（`seat_bias` は `budget_effect` を import してもいる） |
| `pool_games` | `pool_matches` と重複するが、どちらを残すかは判断が要る。名指しにとどめる |
| `scenario_from_book` | 選出キャッシュと解析器のあいだの配管。分析の結果ではない |
| `roster_from_standings` | 大会の六体を自陣ロスターに変える下ごしらえ。再利用される形をしている |
| `counterplay` | `configs/knowledge/rizabanadohido-counters.json` の唯一の読み手。維持されている設定ファイルと読み手を離すのは筋が悪い |
| `ko_branch_count` / `noise_cost_model` / `search_optimism` / `why_action` | 9/19 に増えたばかり。「一度きりだった」と言えるだけの履歴がまだ無い |

## 動かさなかったもののうち、明示的に触らなかった道具

本課題の作業中、次の7本は他の作業と衝突するため一切触っていない（表には載っている）。

`selection_check` / `book_check` / `train_value` / `agent_drift` / `sweep_value` /
`match_result` / `ratings`
