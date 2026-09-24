"""What the Rust port may ignore, and which of those it has to say it ignored.

Two generated lists, both the port's own (IKA-210). They were Python's: `inert.rs` was the
ids Python's source never named and `modelled.rs` Python's calculator sets, because the
port's contract was "the same answer as Python". The contract is Showdown's now, so both are
decided by two facts that do not depend on Python -- whether Showdown has a handler for the
id (the regulation dump's `customHooks`, which the bridge reads off the dex) and whether the
port's engine names it:

    uv run python tools/port_coverage.py                   # what the port reads
    uv run python tools/port_coverage.py --rust            # -> rust/src/inert.rs
    uv run python tools/port_coverage.py --rust-modelled   # -> rust/src/modelled.rs
    uv run python tools/port_coverage.py --check           # both, compared, neither written

`inert.rs` is the ids the port's engine never names, found by scanning its source -- so its
gates take them without code of their own. `modelled.rs` says which of the regulation's ids
need no note: one with no Showdown handler (ignoring it cannot differ from Showdown), one the
port names (it acts on it), and a mega stone (the mega action owns it). Every other id is
one Showdown acts on and the port ignores, and `damage::unmodelled_effects` names it on every
hit -- so a port that learned an effect and was not regenerated would go on reporting a gap
it no longer has, and one that dropped an effect would stop reporting one it has.

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
#: Where the port's engine is read from: apart from `RUST_SRC`, so a test that points the
#: generated files at copies still scans the real engine.
PORT_SRC = ROOT / "rust" / "src"

#: The port's engine: the files a turn goes through (IKA-210). The loader (`reg.rs`, whose
#: JSON keys would read as names), the encoder, the node's plumbing and the generated files
#: themselves are not the engine.
PORT_ENGINE_FILES = [
    "resolve.rs",
    "moves.rs",
    "commands.rs",
    "damage.rs",
    "effects.rs",
    "battler.rs",
    "speed.rs",
    "terrain.rs",
    "moveinfo.rs",
    "damage_callback.rs",
    "level_struggle.rs",
    "move_hooks.rs",
    "magic_guard.rs",
]

#: What is left of Python's engine -- the damage calculator, speed, the menus -- read by
#: `tools/port_gate_audit.py`. `resolve.py` was the first entry until IKA-212 deleted it.
#: `names.py`, the priors and the tools are not the engine.
ENGINE_FILES = [
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


#: The port's gates, whose arms are grouped under comments saying why each is taken.
GATE_FUNCTIONS = ("ability_handled", "item_handled")
#: The one group that says the port takes the id and does nothing Showdown does with it. A
#: name there is not the port acting on the id, so it does not take the note away.
IGNORED_SECTION = "Python reports these and changes nothing"


def taken_and_ignored(text: str) -> set[str]:
    """The gates' ids under `IGNORED_SECTION`: taken by the port, not acted on."""
    out: set[str] = set()
    for name in GATE_FUNCTIONS:
        start = text.find(f"fn {name}(")
        if start < 0:
            continue
        section = ""
        for line in text[start : text.find("\n}\n", start)].splitlines():
            stripped = line.strip()
            if stripped.startswith("//"):
                section = stripped.lstrip("/ ")
                continue
            if section.startswith(IGNORED_SECTION):
                out |= set(re.findall(r'"([a-z0-9]+)"', line))
    return out


def port_text() -> str:
    """The port's engine source, the text `inert` and `modelled` scan."""
    return "\n".join((PORT_SRC / name).read_text(encoding="utf-8") for name in PORT_ENGINE_FILES)


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
    "//! Ids this port's engine never acts on.\n"
    "//!\n"
    "//! An ability or item the engine never names is one its gates may take without code of\n"
    "//! their own: whatever Showdown does with it, the port does nothing, and `modelled.rs` says\n"
    "//! whether that nothing needs a note. Deciding which ids those are by memory is how a port\n"
    "//! acquires a silent wrong answer, so it is decided by scanning the engine's source:\n"
    "//! `tools/port_coverage.py --rust` regenerates this file, and it must be regenerated when\n"
    "//! the engine learns a new effect (IKA-210; it was Python's source until then).\n\n"
)
INERT_DOC = (
    "/// Ids the port's engine never names. Generated by `tools/port_coverage.py --rust`."
)


def inert(regulation: str, engine: str | None = None) -> Generated:
    """`rust/src/inert.rs`: the regulation's ids that the port's engine never names.

    `engine` stands in for `port_text()`. It is how a test hands this an engine that has
    learned one more effect, which is the change `--check` exists to notice.
    """
    data = regulation_dump(regulation)
    text = port_text() if engine is None else engine
    abilities = sorted(e["id"] for e in _abilities(data) if not mentioned(text, e["id"]))
    items = sorted(e["id"] for e in data["items"] if not mentioned(text, e["id"]))
    body = (
        INERT_HEADER
        + rust_predicate("ability_is_inert", abilities, INERT_DOC)
        + "\n"
        + rust_predicate("item_is_inert", items, INERT_DOC)
    )
    return Generated("inert.rs", "--rust", body, f"{len(abilities)} abilities, {len(items)} items")


MODELLED_HEADER = (
    "//! The ids that need no note: Showdown has no handler for them, this port acts on them,\n"
    "//! or (an item) the mega action owns them.\n"
    "//!\n"
    "//! Every other ability and item on a hit is one Showdown acts on and this port ignores,\n"
    "//! and `damage::unmodelled_effects` names it so the caller is told. Generated from the\n"
    "//! regulation dump's `customHooks` and the port's engine source by\n"
    "//! `tools/port_coverage.py --rust-modelled` (IKA-210; it was Python's calculator sets).\n"
    "//! The last predicate is the other way round: the damaging moves that do need a note\n"
    "//! (IKA-213).\n\n"
)


def _abilities(data: dict) -> list[dict]:
    """The dump's abilities, and any a species carries that the dex list lacks.

    Aura Guard is the one (Mega Pokemon's champions ability): no dex entry, so no
    `customHooks` to read. It is taken to have a handler -- a note unless the port names it.
    """
    listed = {e["id"] for e in data["abilities"]}
    carried = {
        re.sub(r"[^a-z0-9]", "", name.lower())
        for species in data["species"]
        for name in (species.get("abilities") or [])
    }
    unlisted = [{"id": i, "customHooks": ["(no dex entry)"]} for i in sorted(carried - listed)]
    return list(data["abilities"]) + unlisted


#: Ids Showdown acts on without a handler of their own, so the dump's empty `customHooks`
#: does not mean "nothing happens": the simulator or another effect names them
#: (`pokemon.hasAbility('levitate')`, the weather rocks in `conditions.ts`'s
#: `durationCallback`s), or the entry carries a value where a handler would be (Battle
#: Armor's and Shell Armor's `onCriticalHit: false`). Read off vendor a5df827 by
#: `tests/test_port_coverage.py::test_showdowns_names_are_the_ones_listed` (IKA-210).
SHOWDOWN_ACTS_BY_NAME = {
    "abilities": frozenset(
        {
            "battlearmor", "corrosion", "dancer", "earlybird", "levitate", "multitype",
            "rkssystem", "shellarmor", "stall",
        }
    ),
    "items": frozenset(
        {"bindingband", "damprock", "heatrock", "icyrock", "lightclay", "smoothrock", "terrainextender"}
    ),
}
#: The two of those that are a value rather than a name, which the vendor scan cannot see.
SHOWDOWN_VALUE_NOT_HANDLER = frozenset({"battlearmor", "shellarmor"})


def showdown_acts(entry: dict, kind: str) -> bool:
    """Whether Showdown does anything with the id: a handler, or a name somewhere else."""
    return bool(entry.get("customHooks")) or entry["id"] in SHOWDOWN_ACTS_BY_NAME[kind]


def _without_gates(text: str) -> str:
    for name in GATE_FUNCTIONS:
        start = text.find(f"fn {name}(")
        if start >= 0:
            text = text[:start] + text[text.find("\n}\n", start) + 3 :]
    return text


def _needs_no_note(entry: dict, text: str, kind: str, ignored: set[str], behaviour: str) -> bool:
    """No note: Showdown does nothing with it, the port acts on it, or it is a mega stone.

    The port acts on an id it names -- in a gate's arm other than the ignored group, or
    anywhere outside the gates. An ignored-group id named outside them is one the port
    reports where it fires (`contact ability: static`), so a note on every hit would say it
    twice.
    """
    identifier = entry["id"]
    acts = mentioned(behaviour, identifier) or (
        mentioned(text, identifier) and identifier not in ignored
    )
    return not showdown_acts(entry, kind) or acts or bool(entry.get("megaStone"))


def modelled(regulation: str, engine: str | None = None) -> Generated:
    """`rust/src/modelled.rs`: what needs no note, from the dex and the port's source.

    A status move is fully modelled when its whole effect is its declarative fields (no
    custom code in the dex) or the port names it; any other status move with custom code
    is reported (`moves::apply_status_move`). A gate's arm under "Python reports these and
    changes nothing" is not the port acting on the id; every other name is. A damaging move
    with custom code the engine never names is listed in `damaging_move_is_unmodelled` and
    reported (`moves::use_move`, IKA-213).
    """
    data = regulation_dump(regulation)
    text = port_text() if engine is None else engine
    ignored = taken_and_ignored(text)
    behaviour = _without_gates(text)
    abilities = sorted(
        e["id"]
        for e in _abilities(data)
        if _needs_no_note(e, text, "abilities", ignored, behaviour)
    )
    items = sorted(
        e["id"] for e in data["items"] if _needs_no_note(e, text, "items", ignored, behaviour)
    )
    status_moves = sorted(
        e["id"]
        for e in data["moves"]
        if e.get("category") == "Status" and (not e.get("hasCustomCode") or mentioned(text, e["id"]))
    )
    # The damaging moves' side of the same question (IKA-213), as the short list: a hook such
    # as `damageCallback` is not a field, so nothing refused Counter, and it was answered as a
    # base-power-0 hit. A move with custom code the engine never names is reported.
    damaging_moves = sorted(
        e["id"]
        for e in data["moves"]
        if e.get("category") != "Status" and e.get("hasCustomCode") and not mentioned(text, e["id"])
    )
    body = (
        MODELLED_HEADER
        + rust_predicate(
            "ability_is_modelled",
            abilities,
            "/// No Showdown handler, or one the port's engine acts on.",
        )
        + "\n"
        + rust_predicate(
            "item_is_modelled",
            items,
            "/// No Showdown handler, one the port's engine acts on, or a mega stone.",
        )
        + "\n"
        + rust_predicate(
            "status_move_is_fully_modelled",
            status_moves,
            "/// The status moves whose whole effect is the declarative fields, or whose custom\n"
            "/// code the port's engine implements, so they are not reported.",
        )
        + "\n"
        + rust_predicate(
            "damaging_move_is_unmodelled",
            damaging_moves,
            "/// The damaging moves with custom code (a hook such as `damageCallback`) that the\n"
            "/// port's engine never names, so a turn that uses one reports it (IKA-213).",
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
    for generated in (inert(regulation, engine), modelled(regulation, engine)):
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
    text = port_text()
    for kind in ("abilities", "items"):
        ids = sorted(entry["id"] for entry in data[kind])
        read = [i for i in ids if mentioned(text, i)]
        print(f"{kind}: {len(read)} read by the port's engine, {len(ids) - len(read)} inert")
        print("  read:", ", ".join(read))
    return 0


if __name__ == "__main__":
    sys.exit(main())
