"""Writes trained value nets (`.pt`) as the `.rnet` files the Rust inference server reads.

IKA-461. The Rust server (`pokeuraou-damage inference-server`) has its own forward pass, so
the weights cross as raw float32 and a header (`pokeuraou.rustnet`). One file in, one file
out; an existing output whose source hash is the input's is left alone.

    uv run --group learn python tools/export_rust_net.py data/models/value-mc7.pt \\
        --regulation gen9championsvgc2026regmc --out-dir data/models/rnet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.rustnet import ensure_exported, read_header  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--regulation", required=True, help="the format id the net's vocabulary is read against")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    for model in args.models:
        out = ensure_exported(model, args.regulation, args.out_dir)
        header = read_header(out)
        print(f"{model.name} -> {out} ({out.stat().st_size / 1e6:.2f} MB, mon_in {header['mon_in']}, "
              f"sha256 {header['sha256'][:12]})")


if __name__ == "__main__":
    main()
