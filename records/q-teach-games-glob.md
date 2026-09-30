# q-teach-games-glob: 局の記録の glob を両方の名前に

gen-2 の局のファイルは `games-bNN-workerK.jsonl` で、`tools/q_teach.py index` などは `games-worker*.jsonl` だけを探していた（IKA-400 §9 はハードリンクで回避）。

読む側 17 か所の glob を `games-*worker*.jsonl` にした: src の luck.py・sprt.py（GAME_FILES）、tools の aivat・crit_effect・deep_targets・endgame_exact・error_decompose・legal_moves_cost・match_result・oneshot/served_throughput・paired_result・position_set（2 か所）・profile_stages・q_teach・ratings・refine_signals・time_match。

変えなかった所: 自分が書いて数える側（generate_queue.py・match_queue.py の `written()`、書き出し名）と、ファイルを動かす queue_restart.py（再開の挙動が変わるため）。

確認:
- data/selfplay-mc1 で旧 glob と新 glob の並びが一致（48 本、同一リスト）。よって既存の名前だけのディレクトリでは index の順も同じ。
- tests/test_games_file_glob.py: 両方の名前を拾い、rank-worker・workers-Q1.json・summary.jsonl は拾わない。変更前のコードで落ち、変更後に通る。
- ruff 通過。test_luck の 3 件は worktree に port の exe が無いため PortUnavailable（glob と無関係）。
- 注意: `games-*worker*` は再開の `games-r1-worker0.jsonl` も読む（読む側では望ましい）。
