"""What the Rust port may ignore, and what Python's calculator claims to model.

Two generated lists, both about the same question: the port's contract is "the same answer
as Python", and deciding by memory which ids Python acts on is how a port acquires a silent
wrong answer.

    uv run python tools/port_coverage.py                   # what the engine reads
    uv run python tools/port_coverage.py --rust            # -> rust/src/inert.rs
    uv run python tools/port_coverage.py --rust-modelled   # -> rust/src/modelled.rs
    uv run python tools/port_coverage.py --check           # both, compared, neither written

`inert.rs` is the ids the engine never acts on, found by scanning its source -- so the port
may ignore them. `modelled.rs` is `effects.all_modelled_abilities()` and
`all_modelled_items()`, which `damage._unmodelled` uses to name, on every hit, the ability
or item the calculator does *not* account for. Those notes reach the caller, so a port that
resolved turns correctly and skipped them would quietly shrink what the caller is told.

Regenerate both when the engine learns a new effect. `--check` is that sentence as a test,
and CI runs it: it builds both files the way the two flags would, writes neither, and fails
on any byte that differs from `rust/src`. Until 2026-09-23 it could not have been one.
`inert.rs` had been reworded by hand after it was generated, so regenerating it gave a diff
with not one id in it, and "regenerate and see whether the diff is empty" could not tell an
engine that had learned an effect from a header someone had improved (IKA-72). The file's
words now live here, so an edit to either file belongs in this tool -- and `--check` says so
on the day the edit is made, not on the day a regeneration throws it away.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Where the generated files live. Read at call time rather than bound into each function,
#: so a test can point `--check` at a copy with one line changed.
RUST_SRC = ROOT / "rust" / "src"

#: The files a turn goes through. `names.py`, the priors and the tools are not the engine.
ENGINE_FILES = [
    "resolve.py",
    "damage.py",
    "effects.py",
    "speed.py",
    "moveinfo.py",
    "view.py",
    "battler.py",
    "actions.py",
]

#: Functions whose bodies are inventories rather than behaviour. `all_modelled_abilities`
#: lists every ability the calculator accounts for, most of them precisely because they do
#: nothing to a damage roll -- so a mention there is the opposite of evidence that the
#: engine acts on the id.
INVENTORY_FUNCTIONS = ("all_modelled_abilities", "all_modelled_items")


def strip_inventories(source: str) -> str:
    out = source
    for name in INVENTORY_FUNCTIONS:
        start = out.find(f"def {name}(")
        if start < 0:
            continue
        rest = out[start:]
        end = len(rest)
        for marker in ("\ndef ", "\n@", "\nclass "):
            found = rest.find(marker, 1)
            if 0 < found < end:
                end = found
        out = out[:start] + out[start + end :]
    return out


def engine_text() -> str:
    return "\n".join(
        strip_inventories((ROOT / "src" / "pokeuraou" / name).read_text(encoding="utf-8"))
        for name in ENGINE_FILES
    )


def mentioned(text: str, identifier: str) -> bool:
    """Whether the engine names this id as a string literal."""
    return re.search(rf'["\']{re.escape(identifier)}["\']', text) is not None


def rust_predicate(name: str, values: list[str], doc: str) -> str:
    body = "\n".join(f'            | "{value}"' for value in values)
    body = body.replace("            | ", "        ", 1)
    return f"{doc}\npub fn {name}(id: &str) -> bool {{\n    matches!(\n        id,\n{body}\n    )\n}}\n"


def regulation_dump(regulation: str) -> dict:
    path = ROOT / "configs" / "regulations" / f"{regulation}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Generated:
    """One file under `rust/src`, as this tool writes it."""

    name: str
    #: The flag that writes it, for the message that says how to regenerate it.
    flag: str
    text: str
    #: "147 abilities, 102 items": the line each flag has always printed.
    counts: str


#: Written into the file by hand after it was first generated, and better than what this
#: tool used to say, so the tool says it now (IKA-72).
INERT_HEADER = (
    "//! Ids the Python engine never acts on.\n"
    "//!\n"
    "//! The port's contract is \"the same answer as Python\", so an ability or item Python never\n"
    "//! mentions outside its own inventory lists is one this port may ignore without diverging.\n"
    "//! Deciding that by memory is how a port acquires a silent wrong answer, so it is decided\n"
    "//! by scanning the engine's source: `tools/port_coverage.py --rust` regenerates this file,\n"
    "//! and it must be regenerated when the engine learns a new effect.\n\n"
)
INERT_DOC = (
    "/// Ids the Python engine never mentions, so ignoring them cannot diverge\n"
    "/// from it. Generated by `tools/port_coverage.py --rust`."
)


def inert(regulation: str, engine: str | None = None) -> Generated:
    """`rust/src/inert.rs`: the regulation's ids that the engine's source never names.

    `engine` stands in for `engine_text()`. It is how a test hands this an engine that has
    learned one more effect, which is the change `--check` exists to notice.
    """
    data = regulation_dump(regulation)
    text = engine_text() if engine is None else engine
    abilities = sorted(e["id"] for e in data["abilities"] if not mentioned(text, e["id"]))
    items = sorted(e["id"] for e in data["items"] if not mentioned(text, e["id"]))
    body = (
        INERT_HEADER
        + rust_predicate("ability_is_inert", abilities, INERT_DOC)
        + "\n"
        + rust_predicate("item_is_inert", items, INERT_DOC)
    )
    return Generated("inert.rs", "--rust", body, f"{len(abilities)} abilities, {len(items)} items")


MODELLED_HEADER = (
    "//! The ids Python's calculator accounts for, and therefore does not report.\n"
    "//!\n"
    "//! `damage._unmodelled` names every ability and item outside these sets, on every\n"
    "//! hit, and the resolver unions those notes into the turn's report. Generated by\n"
    "//! `tools/port_coverage.py --rust-modelled`.\n\n"
)


def modelled(regulation: str) -> Generated:
    """`rust/src/modelled.rs`: what `effects` says the calculator accounts for."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import pokeuraou
    from pokeuraou.effects import all_modelled_abilities, all_modelled_items
    from pokeuraou.regulation import load_regulation
    from pokeuraou.resolve import STATUS_MOVES_FULLY_MODELLED

    # `engine_text` reads this tree's files by path; the sets below come from whichever
    # `pokeuraou` was imported. In one process those can be two trees -- a worktree whose
    # test run forgot PYTHONPATH gets the main checkout's package -- and a comparison
    # between them is wrong without failing. So it fails.
    if not Path(pokeuraou.__file__).parent.samefile(src / "pokeuraou"):
        raise SystemExit(
            f"  port_coverage: pokeuraou was imported from {Path(pokeuraou.__file__).parent},\n"
            f"  not from this tree's {src / 'pokeuraou'}. Put this tree's src on PYTHONPATH."
        )

    reg = load_regulation(regulation)
    abilities = sorted(all_modelled_abilities())
    # The stones are passed, not registered: `register_mega_stones` sets a global in
    # `damage` that this output never reads, and `--check` also runs inside the suite.
    items = sorted(all_modelled_items(frozenset(reg.mega_map)))
    status_moves = sorted(STATUS_MOVES_FULLY_MODELLED)
    body = (
        MODELLED_HEADER
        + rust_predicate(
            "ability_is_modelled", abilities, "/// `effects.all_modelled_abilities()`."
        )
        + "\n"
        + rust_predicate(
            "item_is_modelled",
            items,
            "/// `effects.all_modelled_items(mega_stones)` for this regulation's stones.",
        )
        + "\n"
        + rust_predicate(
            "status_move_is_fully_modelled",
            status_moves,
            "/// `resolve.STATUS_MOVES_FULLY_MODELLED`: the status moves whose whole effect\n"
            "/// is the declarative fields, so the resolver does not report them.",
        )
    )
    return Generated(
        "modelled.rs", "--rust-modelled", body, f"{len(abilities)} abilities, {len(items)} items"
    )


def write(generated: Generated) -> None:
    # `newline`: text mode writes CRLF on Windows, and this is a tracked source that
    # `engine_fingerprint` hashes byte for byte. See .gitattributes.
    (RUST_SRC / generated.name).write_text(generated.text, encoding="utf-8", newline="\n")
    print(f"{generated.name}: {generated.counts}")


FN = re.compile(r"^pub fn (\w+)\(", re.MULTILINE)
#: One arm of a generated `matches!`: `"id"` alone on its line, or after a `|`.
ARM = re.compile(r'^[ \t]*(?:\|[ \t]*)?"([^"\s]+)"[ \t]*$', re.MULTILINE)


def arms(text: str) -> dict[str, set[str]]:
    """Each generated predicate's ids, by the function's name.

    By function and not pooled: an id that moved from one predicate to another is a moved
    id, and a pooled set would call it no change.
    """
    starts = list(FN.finditer(text))
    return {
        found.group(1): set(
            ARM.findall(text, found.end(), starts[i + 1].start() if i + 1 < len(starts) else len(text))
        )
        for i, found in enumerate(starts)
    }


def describe(on_disk: str, written: str, shown: str) -> list[str]:
    """What the reader of a red check needs first, and then the diff.

    First whether an id moved -- that is the part that can make the port answer differently
    from Python -- or only the words. The 9/22 diff was all words, and it took reading it
    line by line to know that.
    """
    if on_disk.replace("\r\n", "\n") == written:
        # Before IKA-114 this tool wrote CRLF itself. A diff would be every line of the
        # file, each looking the same as its pair.
        return ["    only the line endings differ: the file has CRLF, and this tool writes LF"]
    have, want = arms(on_disk), arms(written)
    lines = []
    for fn in sorted(set(have) | set(want)):
        dropped = sorted(have.get(fn, set()) - want.get(fn, set()))
        added = sorted(want.get(fn, set()) - have.get(fn, set()))
        if dropped:
            lines.append(f"    {fn}: the file lists, the tool would not: {', '.join(dropped)}")
        if added:
            lines.append(f"    {fn}: the tool would list, the file does not: {', '.join(added)}")
    if not lines:
        lines.append("    no id moved: only the words or the whitespace differ")
    diff = difflib.unified_diff(
        on_disk.splitlines(keepends=True),
        written.splitlines(keepends=True),
        fromfile=f"{shown} (on disk)",
        tofile=f"{shown} (as the tool writes it)",
        n=1,
    )
    return lines + ["    " + line.rstrip("\n") for line in diff]


def check(regulation: str, engine: str | None = None) -> int:
    """How many of the generated files differ from what this tool would write.

    Writes nothing, and compares bytes: `engine_fingerprint` hashes these files byte for
    byte, so a difference only in whitespace is still a different engine to it.
    """
    bad = 0
    for generated in (inert(regulation, engine), modelled(regulation)):
        path = RUST_SRC / generated.name
        shown = f"rust/src/{generated.name}"
        if not path.exists():
            bad += 1
            print(f"  MISSING  {shown}: `tools/port_coverage.py {generated.flag}` writes it")
            continue
        on_disk = path.read_bytes()
        if on_disk == generated.text.encode("utf-8"):
            print(f"  ok       {shown}  ({generated.counts})")
            continue
        bad += 1
        print(f"  DIFFERS  {shown} is not what `tools/port_coverage.py {generated.flag}` writes")
        for line in describe(on_disk.decode("utf-8", errors="replace"), generated.text, shown):
            print(line)
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regulation", default="gen9championsvgc2026regmc")
    ap.add_argument("--rust", action="store_true", help="write rust/src/inert.rs")
    ap.add_argument("--rust-modelled", action="store_true", help="write rust/src/modelled.rs")
    ap.add_argument(
        "--check",
        action="store_true",
        help="build both files, write neither, and exit 1 if either differs from rust/src",
    )
    args = ap.parse_args(argv)

    if args.check:
        if args.rust or args.rust_modelled:
            ap.error("--check writes nothing; give it without --rust / --rust-modelled")
        bad = check(args.regulation)
        if not bad:
            print("\n  both files are byte for byte what this tool writes.")
            return 0
        print(
            f"\n  {bad} generated file(s) are not what this tool writes. If the engine learned\n"
            "  or dropped an effect, regenerate with the flag named above and commit the\n"
            "  result. If someone improved the words in the file, move them into this tool:\n"
            "  the next regeneration throws a hand edit away (IKA-72)."
        )
        return 1

    if args.rust:
        write(inert(args.regulation))
    if args.rust_modelled:
        write(modelled(args.regulation))
    if args.rust or args.rust_modelled:
        return 0

    data = regulation_dump(args.regulation)
    text = engine_text()
    for kind in ("abilities", "items"):
        ids = sorted(entry["id"] for entry in data[kind])
        read = [i for i in ids if mentioned(text, i)]
        print(f"{kind}: {len(read)} read by the engine, {len(ids) - len(read)} inert")
        print("  read:", ", ".join(read))
    return 0


if __name__ == "__main__":
    sys.exit(main())
