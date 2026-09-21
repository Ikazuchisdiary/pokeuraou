"""Effects the port implements but its own gate refuses, and names the gate passes that
nothing in the port implements.

`check_white_herb` and its four call sites landed with the port on 2026-09-12. The one
thing missing was the id in `item_handled`, so for eight days every position a holder
could appear in was refused -- 63% of the cells generation refused and 76% of a match's,
and the tail of filling those 1x1 nodes in Python was 44.5% of generation's wall clock
(IKA-29, 2026-09-20). One line nobody wrote, and it only became visible once something
counted refusals by reason.

`tools/port_coverage.py` already answers the mirror-image question by scanning source
rather than trusting memory: which ids the *Python* engine never mentions, so the port may
ignore them. This points the same scan at the port and asks two questions the gate cannot
answer about itself:

    referenced but not gated   the port knows the name, the gate refuses it. The answer
                               stays right -- Python fills the refused cell -- so nothing
                               is wrong except the bill. This is the whiteherb shape.

    gated but not referenced   the gate passes the name and no code acts on it. This is
                               the dangerous direction: nothing refuses, nothing
                               implements, and the port returns a quietly different
                               answer. Most of today's list is deliberate (an ability with
                               no effect a turn can observe is handled by ignoring it),
                               which is why the ids are recorded below rather than
                               required to be empty.

    uv run python tools/port_gate_audit.py            # the report
    uv run python tools/port_gate_audit.py --check    # and fail if either set moved

Like `agent_drift.py --check`, this fails in *both* directions: a new finding is the thing
to catch, and one that goes away has to be struck off, or the recorded set stops
describing anything.

A reference is not an implementation. A name can appear in the port only inside the
message that refuses it -- `stancechange` was in exactly that state until the forme change
moved into `use_move`. So the first question lists candidates and a person decides;
`ACKNOWLEDGED` below is where that decision is written down, with its reason.

What this must not become is a check that reads a stale copy of the gate. The gate is not
a list -- `item_handled` also passes anything `type_boost_item` or `resist_berry` knows,
and any mega stone by suffix -- so this parses those clauses out of the Rust source
instead of restating them here, and *refuses to run* if a gate grows a clause it does not
recognise. A check that silently ignored a new escape hatch would report findings that are
already handled, and be switched off.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import port_coverage  # noqa: E402  -- `engine_text`/`mentioned`: one definition of

# "Python's engine names this id", shared with the scan that writes `inert.rs`. Two
# definitions of that would be two answers to the question this file asks.

ROOT = Path(__file__).resolve().parents[1]
RUST_SRC = ROOT / "rust" / "src"

#: Written by `tools/port_coverage.py`. Inventories, not behaviour: a name here is the
#: statement that the engine never acts on it, which is the opposite of evidence that the
#: port does.
GENERATED = ("inert.rs", "modelled.rs")


@dataclass(frozen=True)
class Gate:
    """One `fn ..._handled` in the port, and which of the regulation's ids it judges."""

    fn: str
    #: The key in the regulation dump whose ids this gate is asked about.
    kind: str
    what: str


GATES = (
    Gate("ability_handled", "abilities", "abilities"),
    Gate("item_handled", "items", "items"),
    Gate("status_move_handled", "moves", "status moves"),
)

#: Candidates for the first question that a person has looked at and kept, because the
#: name being in the port is not the port acting on it. The reason is the point: without
#: it the next reader cannot tell a decided case from an unnoticed one.
ACKNOWLEDGED: dict[str, str] = {
    "slowstart": (
        "the only mention is the refusal itself: check_position_supported refuses it at "
        "resolve.rs:883, so a turn can never get half-way through it"
    ),
    "trick": "named only by the refusal at resolve.rs:796 -- this port swaps no items",
    "switcheroo": "named only by the refusal at resolve.rs:796 -- likewise",
}

#: Effects the port implements and its gate refuses, as of 2026-09-22. Every one of them
#: is a bill rather than a wrong answer, and taking one off the list is not a one-line
#: change: it makes the port answer positions it used to hand back, so it needs
#: `tools/diff_node.py` to say the two engines agree on them first. Recorded so `--check`
#: can fail on a fifth without first demanding these four be settled.
KNOWN_UNGATED: frozenset[str] = frozenset(
    {
        # speed.rs:160 branches the priority roll for it, beside `quickdraw`, which is in
        # the gate. Python does the same at speed.py:222.
        "quickclaw",
        # effects.rs:463 gives it the 1-in-10, read at resolve.rs:327, beside `focussash`,
        # which is in the gate. Python has it at effects.py:552.
        "focusband",
        # damage.rs:379-382 zeroes the hit for both. Not the same thing as Python, which
        # also busts the forme and takes Mimikyu's 1/8 (resolve.py:2952) -- so this pair
        # is a question about the port's damage layer, not a line for the gate.
        "disguise",
        "iceface",
    }
)

#: Names the gate passes that the port never mentions while Python does, as of
#: 2026-09-22. Read one by one, Python's mention of each turns out to be an inventory or a
#: quotation rather than behaviour -- which is the same trap as on the port's side, and the
#: reason this question lists candidates instead of declaring bugs:
#:
#:    shadowtag, arenatrap, magnetpull   actions.TRAPPING_ABILITIES, which nothing reads.
#:                                       Neither engine traps, so they agree.
#:    dancer                             speed.ACTION_OVERRIDING_EFFECTS, read by a
#:                                       helper that scans protocol lines for a
#:                                       differential, not by the resolver.
#:    truant                             resolve.py:1788, inside a docstring quoting
#:                                       Showdown's own TypeScript.
#:
#: Recorded rather than required to be empty so that `--check` can fail on the next one
#: without first demanding these five be re-argued.
KNOWN_UNREFERENCED: frozenset[str] = frozenset(
    {"shadowtag", "arenatrap", "magnetpull", "dancer", "truant"}
)


# ---------------------------------------------------------------------------
# Reading Rust source
# ---------------------------------------------------------------------------


def body_of(text: str, header: re.Pattern[str]) -> str:
    """The braced body of the first item whose header matches, braces balanced.

    Rust strings can hold braces, but none of the bodies this reads do; the one thing it
    must survive is the nested braces of a `matches!` guard and an `if`.
    """
    found = header.search(text)
    if found is None:
        raise SystemExit(f"  port_gate_audit: no `{header.pattern}` in the port")
    start = text.index("{", found.end() - 1)
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise SystemExit(f"  port_gate_audit: unbalanced braces after `{header.pattern}`")


def fn_body(text: str, name: str) -> str:
    return body_of(text, re.compile(rf"\bfn\s+{re.escape(name)}\s*\("))


LITERAL = re.compile(r'"([^"\\\n]*)"')
COMMENT_LINE = re.compile(r"^\s*//\s?(.*)$")


def matches_arms(body: str) -> tuple[dict[str, str], str]:
    """The literals of the body's `matches!`, each with the comment it sits under.

    Returns the arms and the body with that `matches!` call removed, so the caller can
    check what else the gate does.
    """
    at = body.find("matches!(")
    if at < 0:
        raise SystemExit("  port_gate_audit: a gate with no `matches!` -- read it by hand")
    depth = 0
    for index in range(at + len("matches!"), len(body)):
        if body[index] == "(":
            depth += 1
        elif body[index] == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    else:
        raise SystemExit("  port_gate_audit: unbalanced parens in a gate's `matches!`")

    arms: dict[str, str] = {}
    section = ""
    in_comment = False
    for line in body[at:end].splitlines():
        comment = COMMENT_LINE.match(line)
        if comment is not None:
            # A comment describes the run of arms that follows it, up to the next one.
            # Consecutive lines are one comment; a comment after arms starts a new group.
            section = f"{section} {comment.group(1)}".strip() if in_comment else comment.group(1)
            in_comment = True
            continue
        literals = LITERAL.findall(line)
        if not literals:
            continue
        in_comment = False
        for literal in literals:
            arms[literal] = section
    return arms, body[:at] + body[end:]


MATCH_KEY = re.compile(r'"([^"\\\n]+)"\s*(?:\||=>)')


def match_table_keys(text: str, name: str) -> set[str]:
    """The ids a `match id { "x" => ... }` helper answers to.

    Only the left-hand side: `type_boost_item` maps an id to a type name, and the type
    names are not ids. An empty answer is refused rather than returned -- a helper whose
    shape this does not fit would otherwise silently shrink the gate, and every id it
    passes would be reported as refused.
    """
    keys = set(MATCH_KEY.findall(fn_body(text, name)))
    if not keys:
        raise SystemExit(
            f"  port_gate_audit: `{name}` passes ids to a gate and this read no ids out\n"
            "  of it. Either it is empty or it is not a `match id` table, and the gate\n"
            "  would be read as narrower than it is."
        )
    return keys


#: The clauses a gate is allowed to be made of besides its own `matches!`, and what each
#: one contributes. A gate that grows anything else stops this check rather than being
#: quietly misread -- the whole point is that the gate's real membership is read from the
#: gate.
DELEGATES = re.compile(
    r"crate::(?P<module>inert|effects)::(?P<fn>\w+)\(\s*\w+\s*\)(?:\.is_some\(\))?"
)
SUFFIX = re.compile(r'\w+\.ends_with\("(?P<suffix>[^"]+)"\)')
#: Scaffolding with no membership of its own: the early return `item_handled` opens with.
SCAFFOLD = re.compile(r"^(if|\{|\}|return true;|;)$")


@dataclass
class Parsed:
    """What one gate passes, decomposed into the parts this check can name."""

    #: id -> the comment it sits under in the gate's own `matches!`.
    explicit: dict[str, str]
    #: Ids passed by a delegated list rather than named here. Being on one is not evidence
    #: that the port acts on the id, so these are outside the second question.
    delegated: set[str] = field(default_factory=set)
    suffixes: list[str] = field(default_factory=list)
    #: The source span the arms occupy, so the scan for references can skip it.
    inventory: str = ""

    def passes(self, identifier: str) -> bool:
        return (
            identifier in self.explicit
            or identifier in self.delegated
            or any(identifier.endswith(suffix) for suffix in self.suffixes)
        )


def parse_gate(gate: Gate, resolve: str, sources: dict[str, str]) -> Parsed:
    body = fn_body(resolve, gate.fn)
    arms, rest = matches_arms(body)
    parsed = Parsed(explicit=arms, inventory=body)

    rest = "\n".join(
        line for line in rest.splitlines() if COMMENT_LINE.match(line) is None
    )
    for clause in (c.strip() for c in rest.split("||")):
        if not clause or SCAFFOLD.match(clause.replace(" ", "")):
            continue
        cleaned = clause.replace("{", " ").replace("}", " ")
        cleaned = re.sub(r"\breturn true;|\bif\b", " ", cleaned).strip()
        if not cleaned:
            continue
        delegate = DELEGATES.fullmatch(cleaned)
        if delegate is not None:
            module, name = delegate.group("module"), delegate.group("fn")
            parsed.delegated |= match_table_keys(sources[f"{module}.rs"], name)
            continue
        suffix = SUFFIX.fullmatch(cleaned)
        if suffix is not None:
            parsed.suffixes.append(suffix.group("suffix"))
            continue
        raise SystemExit(
            f"  port_gate_audit: `{gate.fn}` has a clause this check cannot read:\n"
            f"      {clause.strip()}\n"
            "  It decides membership and this file would answer as if it were not there,\n"
            "  so the check stops. Teach it the clause in DELEGATES/SUFFIX, or say in\n"
            "  SCAFFOLD that the clause passes nothing."
        )
    return parsed


# ---------------------------------------------------------------------------
# The two questions
# ---------------------------------------------------------------------------


def port_sources() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8") for path in sorted(RUST_SRC.glob("*.rs"))
    }


def behaviour_text(sources: dict[str, str], inventories: list[str]) -> str:
    """The port's source minus every inventory in it.

    The generated files are lists of ids the port may ignore, and a gate's own arms are
    the list being audited; a name in either is not the port acting on it.
    """
    text = "\n".join(
        source for name, source in sources.items() if name not in GENERATED
    )
    for inventory in inventories:
        text = text.replace(inventory, "\n")
    return text


def fully_modelled_moves(sources: dict[str, str]) -> set[str]:
    """`resolve.STATUS_MOVES_FULLY_MODELLED`, by way of the file generated from it.

    Python's own statement about which status moves are nothing but their declarative
    fields. Both engines read those fields out of the regulation dump, so neither has to
    name the move -- which is why an arm for one of these is expected to be unreferenced.
    """
    return set(
        matches_arms(fn_body(sources["modelled.rs"], "status_move_is_fully_modelled"))[0]
    )


def universes(regulation: Path, sources: dict[str, str]) -> dict[str, list[str]]:
    """The ids each gate is actually asked about.

    Not simply "every id in the regulation". `check_move_supported` consults
    `status_move_handled` only for a Status move that Python fully models: a damaging move
    never reaches it, and a status move Python does not model either gets the declarative
    fields from both engines. Asking the gate about the rest would report a hundred moves
    it was never consulted about, and a report that long is one nobody reads.
    """
    data = json.loads(regulation.read_text(encoding="utf-8"))
    modelled = fully_modelled_moves(sources)
    return {
        "ability_handled": sorted(entry["id"] for entry in data["abilities"]),
        "item_handled": sorted(entry["id"] for entry in data["items"]),
        "status_move_handled": sorted(
            entry["id"]
            for entry in data["moves"]
            if entry.get("category") == "Status" and entry["id"] in modelled
        ),
    }


@dataclass
class Finding:
    gate: Gate
    identifier: str
    note: str


def audit(
    regulation: str, sources: dict[str, str] | None = None
) -> tuple[list[Finding], list[Finding], dict[str, int]]:
    """The two questions, plus a count of the arms the second one explained away.

    `sources` is how a test hands this a port with one line taken out of a gate: a check
    that has never been shown failing is a check nobody has reason to believe.
    """
    sources = port_sources() if sources is None else sources
    resolve = sources["resolve.rs"]
    parsed = {gate.fn: parse_gate(gate, resolve, sources) for gate in GATES}
    text = behaviour_text(sources, [p.inventory for p in parsed.values()])
    asked = universes(ROOT / "configs" / "regulations" / f"{regulation}.json", sources)
    named = set(LITERAL.findall(text))
    engine = port_coverage.engine_text()
    declarative = fully_modelled_moves(sources)

    ungated: list[Finding] = []
    unreferenced: list[Finding] = []
    explained: dict[str, int] = {}
    for gate in GATES:
        spec = parsed[gate.fn]
        for identifier in asked[gate.fn]:
            if identifier in named and not spec.passes(identifier):
                ungated.append(Finding(gate, identifier, where(sources, identifier)))
        # Over the gate's own arms rather than the regulation's ids: a name the gate
        # passes is a claim about the port, and it is a claim whether or not this
        # regulation happens to contain the id.
        for identifier, section in spec.explicit.items():
            if identifier in named:
                continue
            if gate.kind == "moves" and identifier in declarative:
                explained["the declarative fields are the whole move"] = (
                    explained.get("the declarative fields are the whole move", 0) + 1
                )
                continue
            if not port_coverage.mentioned(engine, identifier):
                # Neither engine names it. The port reproduces Python's answer by doing
                # the same nothing, which is what `inert.rs` says in the cases where the
                # regulation happens to contain the id.
                explained["python is silent too"] = (
                    explained.get("python is silent too", 0) + 1
                )
                continue
            unreferenced.append(Finding(gate, identifier, section or "no comment"))
    return ungated, unreferenced, explained


def where(sources: dict[str, str], identifier: str) -> str:
    """The first place outside a generated file that names this id, for the report."""
    quoted = f'"{identifier}"'
    for name, source in sources.items():
        if name in GENERATED:
            continue
        for number, line in enumerate(source.splitlines(), start=1):
            if quoted in line:
                return f"rust/src/{name}:{number}"
    return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regulation", default="gen9championsvgc2026regmc")
    ap.add_argument(
        "--check",
        action="store_true",
        help="fail if either set differs from the one recorded in this file",
    )
    args = ap.parse_args(argv)

    if not (ROOT / "configs" / "regulations" / f"{args.regulation}.json").exists():
        print(f"  no configs/regulations/{args.regulation}.json")
        return 2

    ungated, unreferenced, explained = audit(args.regulation)

    print("  the port names it, the gate refuses it -- the whiteherb shape")
    known = set(ACKNOWLEDGED) | KNOWN_UNGATED
    surprises = [f for f in ungated if f.identifier not in known]
    if not ungated:
        print("    none")
    for finding in ungated:
        mark = "    " if finding.identifier in known else "  ! "
        reason = ACKNOWLEDGED.get(finding.identifier) or finding.note
        print(f"{mark}{finding.gate.what:<13} {finding.identifier:<20} {reason}")

    print("\n  the gate passes it, Python acts on it, the port never names it")
    new = [f for f in unreferenced if f.identifier not in KNOWN_UNREFERENCED]
    gone = sorted(KNOWN_UNREFERENCED - {f.identifier for f in unreferenced})
    if not unreferenced:
        print("    none")
    for finding in sorted(unreferenced, key=lambda f: (f.gate.fn, f.identifier)):
        mark = "    " if finding.identifier in KNOWN_UNREFERENCED else "  ! "
        print(f"{mark}{finding.gate.what:<13} {finding.identifier:<20} {finding.note}")
    for why, count in sorted(explained.items()):
        print(f"    {count} more arm(s) the port does not name, set aside: {why}")

    if not args.check:
        return 0

    bad = 0
    if surprises:
        bad += len(surprises)
        print(
            f"\n  {len(surprises)} effect(s) the port implements and its gate refuses:\n"
            "    " + ", ".join(f.identifier for f in surprises) + "\n"
            "  A refused cell is not a wrong answer -- Python fills it -- so decide which\n"
            "  this is: add the id to the gate in rust/src/resolve.rs, or, if the name is\n"
            "  only in a refusal message, write the id and that reason into ACKNOWLEDGED,\n"
            "  or record it in KNOWN_UNGATED as a bill this project is choosing to pay."
        )
    spent = sorted((set(ACKNOWLEDGED) | KNOWN_UNGATED) - {f.identifier for f in ungated})
    if spent:
        bad += len(spent)
        print(
            f"\n  {len(spent)} id(s) recorded as decided that are no longer findings --\n"
            "  the gate passes them now, or the port stopped naming them. Strike them\n"
            "  off: " + ", ".join(spent)
        )
    if new:
        bad += len(new)
        print(
            f"\n  {len(new)} name(s) the gate passes that nothing implements, and that\n"
            "  are not in KNOWN_UNREFERENCED. This is the direction that returns a wrong\n"
            "  answer rather than a refused cell: nothing refuses it and nothing does it."
        )
        for finding in new:
            print(f"    {finding.identifier}: {finding.note}")
    if gone:
        bad += len(gone)
        print(
            f"\n  {len(gone)} id(s) in KNOWN_UNREFERENCED that the port now names, or that\n"
            "  the gate no longer passes. Strike them off, or the list stops describing\n"
            "  anything: " + ", ".join(gone)
        )
    if bad:
        print(f"\n  {bad} thing(s) to decide about what the port's gates pass.")
        return 1
    print("\n  both sets are the ones this file records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
