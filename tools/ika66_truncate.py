"""C腕を作る —— A腕（幅24）のプールを、B腕（幅48）と同じ局数だけ先頭から切る。

局は添字から種を取るので「先頭 N 局」は原理的な切り方で、A腕の局 i と B腕の局 i は
同じ6匹・同じ選出になっている。生成はいらない。符号化するだけ。

  B 対 A = 等時間（出荷判断そのもの）
  B 対 C = 等局数（プールの「質」だけ）
  A 対 C = 等幅  （プールの「量」だけ）

    uv run python tools/ika66_truncate.py --src data/ika66/w24 --out data/ika66/w24trunc --games 8400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, required=True, help="keep game indices [0, N)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.jsonl"):
        stale.unlink()

    kept = 0
    seen = 0
    indices: set[int] = set()
    # Binary throughout: this repository mixes LF and CRLF per file, and a re-encode on
    # the way through would change the bytes of games that are meant to be the same games.
    with (args.out / "games-truncated.jsonl").open("wb") as out:
        for path in sorted(args.src.glob("*.jsonl")):
            with path.open("rb") as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    seen += 1
                    index = json.loads(raw)["gameIndex"]
                    if index < args.games:
                        out.write(raw)
                        indices.add(index)
                        kept += 1

    print(f"{args.src} -> {args.out}")
    print(f"  {seen} games read, {kept} kept, {len(indices)} distinct indices")
    missing = sorted(set(range(args.games)) - indices)
    if missing:
        print(f"  ! {len(missing)} of the first {args.games} indices are absent: {missing[:10]}")
    else:
        print(f"  every index in [0, {args.games}) is present")


if __name__ == "__main__":
    main()
