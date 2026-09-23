"""The committed order of every vocabulary, and the one way it is allowed to change.

`configs/vocab/<format_id>.json` lists, per table, the ids in index order: the first id
is index 1, and 0 stays "absent or unknown". Both encoders read it -- `encode.py`'s
`build_vocabulary` and `rust/src/reg.rs` -- so the integer an id encodes as no longer
depends on where it sorts among the ids of today's dump (IKA-82).

The order is append-only. A new id in a re-dumped regulation goes at the end, so every
index a trained model already knows keeps its meaning and the model loads by growing its
embedding tables (`value.load_model`). An id that leaves the dump keeps its slot; nothing
is ever removed or reordered. Neither encoder appends on its own: an id the dump has and
the list does not is refused at load, because an append made in memory would sort among
whatever else was new that day and could give the same id a different index next time.

    python tools/vocab_order.py --check     # every dump id is listed, nothing duplicated
    python tools/vocab_order.py --append    # add the dump's new ids at the end

`--append` adds each table's new ids in sorted order, which is only a tie-break within
one batch; after that the list is the order.

`--init` writes a list from a dump's sorted ids -- today's order, which is how the first
lists were made so that no existing index moved. It refuses to overwrite a list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import VOCAB_TABLES, dump_ids, read_vocab_order, vocab_order_path  # noqa: E402
from pokeuraou.regulation import load_regulation, regulation_dir  # noqa: E402


def render(format_id: str, order: dict[str, list[str]]) -> bytes:
    """The file's bytes: one id per line so an append is a one-line diff, LF only."""
    lines = ["{", f'  "formatId": {json.dumps(format_id)},']
    for n, table in enumerate(VOCAB_TABLES):
        ids = order[table]
        body = ",\n".join(f"    {json.dumps(i)}" for i in ids)
        end = "," if n + 1 < len(VOCAB_TABLES) else ""
        lines.append(f'  "{table}": [\n{body}\n  ]{end}' if ids else f'  "{table}": []{end}')
    lines.append("}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--append", action="store_true")
    mode.add_argument("--init", action="store_true")
    ap.add_argument("formats", nargs="*", help="format ids (default: every committed dump)")
    args = ap.parse_args()

    formats = args.formats or sorted(p.stem for p in regulation_dir().glob("*.json"))
    failed = False
    for format_id in formats:
        dump = regulation_dir() / f"{format_id}.json"
        reg = load_regulation(format_id)
        path = vocab_order_path(dump)
        ids = dump_ids(reg)
        if args.init:
            if path.exists():
                print(f"{format_id}: {path.name} exists; --init never overwrites a list")
                failed = True
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(render(format_id, {t: sorted(ids[t]) for t in VOCAB_TABLES}))
            print(f"{format_id}: wrote {path}")
            continue
        order = read_vocab_order(path)
        if order is None:
            print(f"{format_id}: no committed vocabulary order at {path}")
            failed = True
            continue
        missing = {t: sorted(set(ids[t]) - set(order[t])) for t in VOCAB_TABLES}
        retired = {t: sorted(set(order[t]) - set(ids[t])) for t in VOCAB_TABLES}
        for table in VOCAB_TABLES:
            if retired[table]:
                print(f"{format_id}: {table} keeps {len(retired[table])} retired slot(s): "
                      f"{', '.join(retired[table])}")
        if args.check:
            for table in VOCAB_TABLES:
                if missing[table]:
                    print(f"{format_id}: {table} missing from the order: "
                          f"{', '.join(missing[table])} (run --append)")
                    failed = True
            if not any(missing.values()):
                sizes = ", ".join(f"{t} {len(order[t])}" for t in VOCAB_TABLES)
                print(f"{format_id}: ok ({sizes})")
            continue
        if not any(missing.values()):
            print(f"{format_id}: nothing to append")
            continue
        grown = {t: order[t] + missing[t] for t in VOCAB_TABLES}
        path.write_bytes(render(format_id, grown))
        for table in VOCAB_TABLES:
            for n, i in enumerate(missing[table]):
                print(f"{format_id}: {table} + {i} at index {len(order[table]) + n + 1}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
