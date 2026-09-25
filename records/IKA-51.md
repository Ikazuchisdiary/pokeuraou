# IKA-51: CI の走る場所ができた —— GitHub の公開リポジトリに繋ぎ、4 回の赤で 3 つの不具合（テスト同士の干渉 1、CI の組み立て 2）を直した。CPU で行の答えが束ね方に依存しない性質は、GitHub のランナーでは成り立たない

2026-09-25。担当は調整役のセッション。

答えた問い: **IKA-24 の `.github/workflows/ci.yml` を実際に走らせられるか。無料の範囲に収まるか。**

## 1. どこで走らせるか

* **非公開の場合:** Free プランの Actions は月 2,000 分。今の `ci.yml` は push するたびに（全ブランチ）3 つのジョブを走らせ、suite と learn はどちらも
  Showdown・Rust release のビルドとスイートの全体を回すので、1 回で 20〜40 分かかる見込みだった。master への first-parent のコミットは 1 週間で 264 本あり、
  数日で枠を使い切る。
* **公開の場合:** 標準ランナーの分数に上限はなく、1 ジョブに 4 vCPU・16 GB が付く。`data/` は `.gitignore` 済みなので、Smogon の使用率や大会の順位は
  最初から入らない。→ 公開にした: https://github.com/Ikazuchisdiary/pokeuraou （既定のブランチは master）。
* **Actions に回さないもの:** 速さの測定（共有ランナーでは CPU もほかの負荷も実行ごとに違う）、GPU、自己対戦の大量生成（規約がプロジェクトと関係ない計算や
  サーバーレスの計算基盤としての利用を禁じていて、グレー）。

## 2. 公開する前に直したもの: コミットの作者メール

* 全 930 コミットの作者と、`configs/knowledge/rizabanadohido-{counters,mirror,selection}.json` の `author` / `who` が
  職場のアドレスだった。→ `git filter-repo --mailmap … --replace-text …` で
  `16918436+Ikazuchisdiary@users.noreply.github.com` に置き換えた。リポジトリの `user.email` も同じアドレスに変えた。
* 名前・パス・秘密情報も走査した。
  * 名前（本名）は出てこない。
  * 絶対パスは `C:/Users/Ikazuchi/repos/pokeuraou` と `C:/tmp/ika*` だけ。
  * API キーやトークンは見つからなかった。
* **全コミットのハッシュが変わった。** records/ と `TODO.md` に書いてあるハッシュは、書き換える前の履歴を指す。
  対応表は [IKA-51-commit-map.txt](IKA-51-commit-map.txt)（`old new` の 930 行。filter-repo の `commit-map`）。
* 書き換える前のすべての ref は `C:/tmp/pokeuraou-backup-20260925b.bundle` にある（`git bundle verify` 済み）。
* worktree 144 個で、チェックアウトしているブランチの先の JSON 3 本だけが変わった。index が古い中身のままになったので、メールの行の差分だけであることを
  確かめてから HEAD の中身に戻した（144 個、残りは 0）。
* 手順の失敗: 最初は、master だけのクローンで書き換えて push した。filter-repo はコミットメッセージの中の
  ハッシュも書き換えるので、ほかのブランチのコミットを指すメッセージの書き換わり方がクローンと手元で違い、ハッシュが分かれた（同じ tree、違う parent）。
  手元（全 ref）で書き換えた履歴のほうを正とし、GitHub 側を force-push で置き換えた。**書き換えは全 ref のある場所で 1 回だけ行うこと。**

## 3. CI の赤とその直し方

| 回 | 赤の原因 | 直し方 |
|---|---|---|
| 1 | vendor の Showdown を `node build` で作っていて `.d.ts` が出ず、sim-bridge の strict tsc が TS7016 で止まった | `node build decl`（README は元からこれ。CLAUDE.md の初回手順も同じ誤りだったので直した） |
| 2 | `test_rust_node` の 5 本（不可能な局面を拒否しない、など）と、torch の無い suite ジョブでの収集エラー 2 本 | 下の (a)(b) |
| 3 | suite ジョブで torch の skip が 12 本あり、許した 8 本を超えた。learn ジョブで `test_inference` の CPU 版が 1 ulp ずれた | 許す本数を実際に数えた 12 本にした。CI の CPU 版 torch を lockfile と同じ 2.11.0 に固定した |
| 4 | torch を 2.11.0 に固定しても、CPU 版が同じ 5.96e-8 でずれた | (c) |

**(a) `hold_positions` が次のテストへ持ち越されていた（テスト同士の干渉）。**
* `tools/selfplay.py` は `rustnode.hold_positions()` を入れっぱなしにする。局面を書き換えない生成ワーカーのための設定（IKA-264）で、生成ではこれで正しい。
* ところが、`test_poolplay` と `test_rank_scores` は `run_pool` をプロセスの中で呼ぶ。同じ xdist ワーカーで後に走ったテストが、局面を送った後で書き換えると、
  書き換える前の JSON が送られた。
* 手元でも、`test_rank_scores.py` の後に同じプロセスで `test_rust_node.py` の 2 本を流すと 2 本とも落ちた（正の対照）。
  `tests/conftest.py` に、テストのたびに `hold_positions(False)` に戻す autouse の fixture を足すと、同じ順番で 10 本すべて通った。
  CI の 3 回目以降は 5 本とも通っている。
* 本番の生成は局面を書き換えないので、影響はテストだけ。

**(b)** `test_vocab_order.py` は torch を無条件に import し、`test_final_position.py` は `tools/encode_dataset.py` を経由して import していた。
既存の 4 本と同じ `pytest.importorskip("torch", reason="…learn group")` にした。

**(c) CPU で、行の答えが束ね方に依存しないという性質は、機械によって成り立たない。**
* `test_a_rows_answer_does_not_depend_on_what_it_was_batched_with[cpu]` は、同じ 24 行を逆順で送っても行ごとの答えがビット一致することを求める。
  GitHub のランナーでは最大 5.96e-8（float32 の 1 ulp）ずれる。torch 2.14.0 と 2.11.0 のどちらでも同じだった。手元の機械（AVX512）では通る。
* 出荷の推論は GPU のサーバなので直接の影響はないが、**CPU で葉を評価する経路は、機械が変わると同じ種の局がビット一致しない**。
* 対処（ユーザーの判断で案 1）: CI の learn ジョブは `POKEURAOU_CPU_BATCH_VARIES=1` を宣言し、CPU 版だけを skip する。
  `tools/ci_skip_audit.py` に `cpu-batch` の分類を足し、`--absent cpu-batch=1` で 1 本とちょうど数える。
* 宣言の無い機械では、これまでどおり CPU 版もビット一致を求める。手元で宣言なしは 2 本とも通り、宣言ありは CPU 版だけが skip されて audit も通ることを確かめた。

## 4. 別課題の候補

* CPU で推論する経路があるなら、そこでの局の再現性（機械をまたいだビット一致）を確かめる。
* Actions の `checkout@v4` などは Node 20 で動いていて、廃止の注記が出ている。v5 に上げる。
* suite と learn はビルドを 2 回ずつしている。`rust-cache` は効くが、Showdown と sim-bridge は毎回作り直している（1 回あたり約 1 分）。
