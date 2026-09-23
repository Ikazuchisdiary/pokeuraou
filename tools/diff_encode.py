"""Does the Rust encoder produce the same arrays as the Python one?

The encoding is the network's input. A feature written one slot along, or rounded
differently, is a plausible-looking win probability that means nothing and that nothing
downstream would notice -- so this compares every array bit for bit, not within a
tolerance.

    cd rust && cargo run --release -- encode \\
        ../configs/regulations/gen9championsvgc2026regmc.json turns.json encoded.bin
    uv run python tools/diff_encode.py rust/turns.json rust/encoded.bin

`--mega-from-slots` compares revision 1's `can_mega` rule on both sides (IKA-141): pass it
here and to the Rust `encode` subcommand together. The Rust header says which rule it
applied, and a disagreement with this flag is refused before any array is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import Encoder, EncodingRules  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", help="a turns.json, for its positions")
    ap.add_argument("encoded", help="the binary the Rust `encode` subcommand wrote")
    ap.add_argument("--limit", type=int, default=0, help="compare only this many positions")
    ap.add_argument(
        "--mega-from-slots",
        action="store_true",
        help="encode with revision 1's can_mega rule (EncodingRules, IKA-141)",
    )
    args = ap.parse_args()

    doc = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    reg = load_regulation(doc["format_id"])
    positions = [Position.from_json(p) for p in doc["positions"]]
    if args.limit:
        positions = positions[: args.limit]

    raw = Path(args.encoded).read_bytes()
    newline = raw.index(b"\n")
    header = json.loads(raw[:newline])
    body = raw[newline + 1 :]

    n = header["positions"]
    m = header["monsPerSide"]
    if args.limit:
        assert n >= len(positions)
    rust_rule = bool((header.get("encoding") or {}).get("megaFromSlots", False))
    if rust_rule != args.mega_from_slots:
        raise SystemExit(
            f"the Rust binary encoded with megaFromSlots={rust_rule} and this comparison "
            f"asks for {args.mega_from_slots}; pass --mega-from-slots to both or to neither"
        )
    encoder = Encoder(reg, rules=EncodingRules(mega_from_slots=args.mega_from_slots))
    started = time.perf_counter()
    expected = encoder.encode_positions(positions)
    python_seconds = time.perf_counter() - started
    print(
        f"python encoded {len(positions)} positions in {python_seconds:.3f} s = "
        f"{python_seconds / len(positions) * 1e6:.1f} us each"
    )

    if (encoder.widths["mon"], encoder.widths["side"], encoder.widths["field"]) != (
        header["monWidth"],
        header["sideWidth"],
        header["fieldWidth"],
    ):
        raise SystemExit(
            f"widths differ: python {encoder.widths}, rust "
            f"{header['monWidth']}/{header['sideWidth']}/{header['fieldWidth']}"
        )
    if list(encoder.vocab.types) != list(header["types"]):
        raise SystemExit("the type vocabulary differs")

    layout = [
        ("species", np.int32, (n, 2, m)),
        ("ability", np.int32, (n, 2, m)),
        ("item", np.int32, (n, 2, m)),
        ("moves", np.int32, (n, 2, m, 4)),
        ("mon", np.float32, (n, 2, m, header["monWidth"])),
        ("mask", np.float32, (n, 2, m)),
        ("side", np.float32, (n, 2, header["sideWidth"])),
        ("field", np.float32, (n, header["fieldWidth"])),
    ]

    offset = 0
    problems = 0
    for name, dtype, shape in layout:
        count = int(np.prod(shape))
        got = np.frombuffer(body, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += count * np.dtype(dtype).itemsize
        want = getattr(expected, name)
        got = got[: len(positions)]
        want = want[: len(positions)]
        if want.dtype != dtype:
            got = got.astype(want.dtype)
        if np.array_equal(got, want):
            print(f"  {name:<8} {str(shape):<24} identical")
            continue
        problems += 1
        difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
        where = np.unravel_index(int(np.argmax(difference)), want.shape)
        print(
            f"  {name:<8} {str(shape):<24} DIFFERS at {where}: "
            f"python {want[where]!r} rust {got[where]!r} "
            f"({int((got != want).sum())} of {want.size} entries)"
        )

    if header["unknownVolatiles"] or expected.unknown_volatiles:
        print(f"  unknown volatiles python {expected.unknown_volatiles} rust {header['unknownVolatiles']}")
        if dict(expected.unknown_volatiles) != {
            k: v for k, v in header["unknownVolatiles"].items()
        }:
            problems += 1

    print(f"\n{'all arrays identical' if not problems else f'{problems} arrays differ'}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
