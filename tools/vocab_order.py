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

A list can extend another regulation's (IKA-82): M-C's begins with M-B's order, so every
id M-B knows has the same integer in M-C and a model trained on M-B loads onto M-C by
growing its tables (`value.load_model`). The file records it as

    "extends": {"formatId": "<base>", "sizes": {"species": 357, ...}}

-- the base and how many of its ids, per table, the list begins with. The base is
append-only too, so that prefix stays a prefix of it for good, and `--check` verifies it.

    python tools/vocab_order.py --extend gen9championsvgc2026regmb gen9championsvgc2026regmc

rewrites the named list as the base's order, then the list's own ids the base does not
have (in their existing order). That renumbers the list, so it is only for a regulation
with no model trained on its own numbering yet; a list that already extends is left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import (  # noqa: E402
    VOCAB_TABLES,
    dump_ids,
    read_vocab_extends,
    read_vocab_order,
    vocab_order_path,
)
from pokeuraou.regulation import load_regulation, regulation_dir  # noqa: E402


def _order_of(format_id: str) -> dict[str, list[str]]:
    order = read_vocab_order(vocab_order_path(regulation_dir() / f"{format_id}.json"), format_id)
    if order is None:
        raise SystemExit(f"{format_id}: no committed vocabulary order")
    return order


def extension_problems(
    format_id: str, order: dict[str, list[str]], extends: dict | None
) -> list[str]:
    """What breaks the recorded `extends`: the list must begin with the base's first ids."""
    if extends is None:
        return []
    base_id = extends["formatId"]
    base = _order_of(base_id)
    problems = []
    for table in VOCAB_TABLES:
        n = int(extends["sizes"][table])
        if len(base[table]) < n:
            problems.append(f"extends {base_id} by {n} {table}, but that list has {len(base[table])}")
        elif order[table][:n] != base[table][:n]:
            pairs = zip(order[table] + [None] * n, base[table][:n], strict=False)
            at = next(i for i, (a, b) in enumerate(pairs) if a != b)
            problems.append(
                f"{table} does not begin with {base_id}'s first {n} (differs at index {at + 1})"
            )
    return problems


def extend(base_id: str, format_id: str) -> bool:
    """Rewrite `format_id`'s list to begin with `base_id`'s; True when it failed."""
    path = vocab_order_path(regulation_dir() / f"{format_id}.json")
    order = _order_of(format_id)
    if read_vocab_extends(path) is not None:
        print(f"{format_id}: already extends {read_vocab_extends(path)['formatId']}; left alone")
        return True
    base = _order_of(base_id)
    grown = {}
    for table in VOCAB_TABLES:
        known = set(base[table])
        grown[table] = list(base[table]) + [i for i in order[table] if i not in known]
        moved = sum(1 for n, i in enumerate(order[table]) if grown[table][n] != i)
        retired = len(known - set(order[table]))
        print(f"{format_id}: {table} {len(order[table])} -> {len(grown[table])} "
              f"({len(base[table])} from {base_id}, {retired} of them not in this dump; "
              f"{moved} ids renumbered)")
    extends = {"formatId": base_id, "sizes": {t: len(base[t]) for t in VOCAB_TABLES}}
    path.write_bytes(render(format_id, grown, extends))
    print(f"{format_id}: wrote {path}")
    return False


def render(
    format_id: str, order: dict[str, list[str]], extends: dict | None = None
) -> bytes:
    """The file's bytes: one id per line so an append is a one-line diff, LF only."""
    lines = ["{", f'  "formatId": {json.dumps(format_id)},']
    if extends is not None:
        lines.append(f'  "extends": {json.dumps(extends)},')
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
    mode.add_argument("--extend", metavar="BASE", help="make the named lists begin with BASE's")
    ap.add_argument("formats", nargs="*", help="format ids (default: every committed dump)")
    args = ap.parse_args()

    if args.extend:
        if not args.formats:
            ap.error("--extend names the lists to rewrite")
        return 1 if any(extend(args.extend, f) for f in args.formats) else 0
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
        extends = read_vocab_extends(path)
        if args.check:
            for problem in extension_problems(format_id, order, extends):
                print(f"{format_id}: {problem}")
                failed = True
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
        path.write_bytes(render(format_id, grown, extends))
        for table in VOCAB_TABLES:
            for n, i in enumerate(missing[table]):
                print(f"{format_id}: {table} + {i} at index {len(order[table]) + n + 1}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
