# 評価モデルの学習のメモリ（value-train-memory）

課題番号なし。10/1 03:50、mc0123（7,762,757 決定）の学習が 1 プロセスで約 29 GB を取り、他の仕事と重なってページングで止まりかけた。

## 変えたこと

学習の答えは変えない（値は 1 ビットも丸めない）。読み込み後の配列を小さく持つ。

- `src/pokeuraou/packed.py`（新）
  - `NpzMember`: npz の 1 配列を、全部を展開せず 64k 行ずつ読む。
  - `narrow_ints`: species/ability/item/moves を、値が入る最も狭い整数型（今は int16）で持つ。
  - `PackedFloat`: 最後の軸の列のうち、全行がちょうど 0.0 か 1.0（ビットで比べる。-0.0 と NaN は別扱い）の列を uint8 で持ち、残りは float32 のまま。`[index]` で float32 の行に戻す。途中で 0/1 でない値が出た列は、それまでの行を正確に広げて float32 の列へ移す。
- `value.py`: `load_dataset(path, *, packed=True)`（既定で packed。`packed=False` は従来の読み）。`Dataset.tensors` は id を int64 に戻して GPU に送る。`concat_datasets` は packed の部品を密な連結を作らずに繋ぐ。`save_dataset` は packed の入力を chunk ごとに書く（ファイルの並び・型は従来と同じ、id は int64）。
- `tools/train_value.py`: `--unpacked`（packing の A/B 用）。
- CI の torch skip の枠を 23 から 24（tests/test_packed.py が torch なしで丸ごと skip）。

mmap は使わない: npz の圧縮は mmap できず、非圧縮の .npy にすると mon だけで 22 GB をディスクに置くことになる。packed で足りた。float16 のように値が変わる案は試していない（答えが変わるので別の比べ）。厳密に狭められる残りとして、float16 で正確に表せる mon の 6 列（0.75 GB 相当）は未対応。

## 一致の確かめ

- mc01 の全 912,417 行で、packed と従来の `tensors()` の 8 本のテンソルがバイト一致（差のあるテンソル 0）。対照: 従来側の 1 値を 1e-6 動かすと不一致になる。packed は mon の 91 列のうち 67 列が uint8、24 列が float32。
- tests/test_packed.py: 合成データで同じ内容（遅れて 0.5 が出る列の降格、-0.0、NaN、int16 に入らない id、concat、streaming 保存の往復）。
- 学習 1 エポック（mc01、CPU、同じ種、4 スレッド）: CPU の学習は同じ設定でも走るたびに揺れるので、損失・重みのビット一致は取れなかった。
  | 走り | val loss | val AUC |
  |---|---|---|
  | packed | 0.4526 | 0.8613 |
  | unpacked | 0.4536 | 0.8606 |
  | unpacked（対照の再走） | 0.4528 | 0.8612 |
  | packed（対照の再走） | 0.4529 | 0.8610 |
  同じ型の再走どうしの差（0.4536 対 0.4528 など）が packed と unpacked の差と同じ大きさ。入力のバイト一致と合わせ、差は計算の非決定による。初期の反対称性の検査（最初の 256 検証行）の誤差は両方 1.19e-07 で同じ。
- mc0123、GPU、1 エポック、種 0、`--no-save`: val loss 0.4359（packed）対 0.4346（unpacked）、val AUC 0.8718 対 0.8726。GPU の非決定の範囲とみているが、mc0123 では同じ型の再走を取っていない（対照なし）。

## 確保量と時間（mc0123、1 エポック、--exclusive、GPU）

PrivatePageCount と作業セットは、走らせたプロセス自身を 0.2 秒ごとに測った値。

| | 読み込み後 private | 読み込み後 作業セット | ピーク private | ピーク 作業セット | 1 エポック | プロセス全体 |
|---|---|---|---|---|---|---|
| unpacked（従来） | 28.71 GB | 20.27 GB | 30.17 GB | 22.54 GB | 162.7 s | 199 s |
| packed（新・既定） | 12.91 GB | 11.87 GB | 14.38 GB | 12.96 GB | 45.5 s | 115 s |

- 従来の 1 エポック 162.7 s は、作業セット 20 GB に対し private 29 GB（約 8 GB が外に出ている）のときの値で、他の仕事との競合か OS のページングが入っている可能性がある。packed の 45.5 s と並べて「packed の方が速い」とは言わない。遅くはならなかった、までが言えること。
- mc01（小）の 1 エポックは CPU で 34 s 対 33 s（同じ）、読み込みは 6.7 s 対 3.0 s（packed の方が遅い。展開しながら詰めるため）。
- 残りの 12.9 GB の内訳はほぼ mon（uint8 67 列 + float32 24 列 = 1 行 1,304 B、7.76M 行で 約 10.1 GB）。

## 未確認

- mc0123 で同じ型の再走による損失の揺れ（対照）。
- encode_dataset.py の連結と streaming 保存を mc0123 の大きさで通していない（合成とテストのみ）。
- 降格（読み込み中に 0/1 でない値が出る列）が実データで起きたかは見ていない（mc01 では列 67 本が最後まで 0/1）。
