# 評価モデルを入れたテストの KeyError（4 本の worker スレッドが 1 つの port を共有）

2026-10-07。課題番号なし。master 3fdaacc の CI（run 37523427714）で、`tests/test_inference.py::test_an_ensemble_arm_survives_several_workers_at_once[cpu]` が `a caller raised: KeyError('gen9championsvgc2026regmc')` で落ちた。

## 要点

| 項目 | 内容 |
| --- | --- |
| 原因 1（主） | このテストは 4 本のスレッドを、モジュールが持つ 1 本の port プロセスに載せていた。本番の worker は 1 プロセスに 1 本の port を持つので、この共有は本番に無い。 |
| 原因 2（落ち方） | 1 本の port プロセスは 1 行ずつしか答えられない。スレッドが共有すると答えが入れ違い、`bind` が「33 件の問いに 8 件」を返す。`port.ask` は `rustnode.disable` を呼び、`reset()` が走る。2 本のスレッドが同時に `reset()` に入ると、後の方が、先の方が取り除いた鍵を `_NODES.pop(key)` で取りに行き、鍵 `gen9championsvgc2026regmc` の KeyError になる。 |
| 直し 1 | テストの各スレッドを `rustnode.own_node(regulation)` に入れる（IKA-435 が同じ形のテスト 2 本を直したとき、このテストだけが残っていた）。 |
| 直し 2 | `rustnode.reset()` の `_NODES.pop(key)` を `pop(key, None)` にする。本番の答えは変わらない（鍵が無いときの扱いだけが変わる）。 |
| 陽性対照 | 直し 2 の新しいテストは、直す前の `pop(key)` で `KeyError: 'second'` になり、直したあとで通る。 |
| 再現 | CI の KeyError そのものは手元で再現できなかった。入れ違いと再起動は再現した（下の表）。 |

## 原因の経路

1. `RemoteValue.__call__` は符号化のときに `port.bind_columns` を呼び、これは `rustnode.require_node` で得たモジュールの 1 本のプロセスに問う（IKA-429 以降）。
2. `RustNode._exchange` は、書き込みと読み込みのあいだに鎖を持たない。2 本のスレッドが書くと、一方の答えがもう一方に返る。
3. `bind` の件数の検査が落ち、`port.ask` が `rustnode.disable(...)` を呼ぶ。`disable` は `reset()` を呼び、`_NODES` の全部のプロセスを閉じて取り除く。
4. 2 本が同時に 3 の段に入ると、`reset()` の `for key in list(_NODES): _NODES.pop(key)` で、後の方の写しに残った鍵が既に無い。
5. この KeyError は `ask` の `except` の中で出るので、worker スレッドの `failures` に入り、テストの `assert not failures` が 467 行目で落ちる。

IKA-435 の記録にある「同じ port プロセスを 2 本のスレッドが使うと答えが入れ違い、止まる」と同じ原因で、落ち方だけが違う。手元では止まる側が多く出た。

## 手元の測定

機械は IKA-438 の生成が 14 コアを使っている状態。exe は worktree で build した（`cargo build --release`、9 分 39 秒）。評価モデルのテストは CPU のみで回した（`CUDA_VISIBLE_DEVICES` を空にし、GPU は使っていない）。スレッド側は、テストの本体と同じ 4 本（8・20・33・41 行）が 8 回ずつ問う 1 ラウンドを繰り返す一時の道具で回した（コミットしない）。

| 条件 | プロセス数 × ラウンド | 結果 |
| --- | --- | --- |
| 共有（直す前の形）、1 プロセス | 1 × 6 | 6 ラウンドとも通る。入れ違いは出なかった |
| 共有（直す前の形）、4 並列 | 4 × 10 の予定 | 35 分で 17 ラウンドしか進まない。3 プロセスが 1・2・4 ラウンドで止まり、手で止めた。入れ違いの再起動が 2 プロセスで 7 回出た（`bind answered 33 of 8 positions`、`write to closed file` など）。1 プロセスは 10 ラウンドを再起動つきで通った |
| 各スレッドが own_node（直した形）、4 並列 | 4 × 10 | 40 ラウンド、落ち 0、再起動 0、250 秒 |
| 各スレッドが own_node（直した形）、4 並列 | 4 × 50 | 200 ラウンド、落ち 0、再起動 0（2,444 秒） |

直したテストそのもの（`tests/test_inference.py`、CPU のみ）は、`-n 0` で 32 本が通った。新しいテストは、直す前の `pop(key)` で落ち、直したあとで通る。ruff は通った。`tools/agent_drift.py --check` は終了コード 0 で、`DRIFTED` を 8 行出す。これは今回の変更に関わらない（worktree に `data/models/q-mc5.pt` が無い影響かどうかは確かめていない）。

共有の側で KeyError そのものが出なかったのは、2 本のスレッドが同時に `reset()` に入る時間の重なりが小さいためと見ている。この解釈を、同時の `reset()` を手元で起こして確かめてはいない。代わりに、同じ形の競合を 1 スレッドの中で正確に作るテスト（下）で確かめた。

## 直し

- `tests/test_inference.py`: `test_an_ensemble_arm_survives_several_workers_at_once` の `ask` を `with rustnode.own_node(regulation), RemoteValue(...)` にした。
- `src/pokeuraou/rustnode.py`: `reset()` の `_NODES.pop(key)` を `_NODES.pop(key, None)` にした。
- `tests/test_rust_node.py`: `test_a_reset_that_meets_a_reset_does_not_pop_a_key_twice` を足した。最初の偽ノードの `close()` が 2 回目の `reset()` を呼び、表を外側の写しの下で空にする。直す前の `pop(key)` では `KeyError: 'second'` で落ちる。

スレッドを使う他のテストも調べた。`tests/test_inference_merge.py` は符号化をスレッドの外で済ませ、スレッドの中では `from_encoded` だけを呼ぶので、port に問わない。`test_analysis*.py`・`test_liveview.py`・`test_timing.py`・`test_workqueue.py` は、スレッドの中身までは読んでいない。

## 確かめていないこと

- CI で落ちた run 37523427714 のログは、書いた時点で run が進行中で読めなかった。スタックの行番号は `tests/test_inference.py` の 467 行目と KeyError の文面を、依頼の文面から取った。スタックを読んでの確認ではなく、手元の入れ違いの再起動と `reset()` のコードからの推定である。
- 共有のまま止まった 3 プロセスの止まり方（どの行で待っていたか）は調べていない。
- `tests/test_rust_node.py` と `tests/test_port_threads.py` の全体は回せなかった。worktree に `data/` の使用率（priors）が無く、少なくとも 13 本（出力の末尾 15 行だけを見た）が `SystemExit: no cached usage stats` で落ち、`test_port_threads.py` の 1 本が setup の ERROR になった。直す前の tree での同じ結果とは突き合わせていない。
- `-n 4` で 3 つのファイルをまとめて回したとき、`test_two_workers_at_once_get_what_they_would_get_alone[cpu]` が 1 回落ちた。同じテストの単独の再実行と、`test_inference.py` 全体の `-n 0` は通った。落ちた理由は読んでいない。機械が 100% 使われている中の 1 回で、競合の残りかどうかは分からない。
- GPU の経路（`cuda`）は回していない。
