"""`tools/port_coverage.py --check`: the generated Rust lists are what the engine says.

`rust/src/inert.rs` says it must be regenerated when the engine learns a new effect, and
until 2026-09-23 nothing could tell whether it had been. Regenerating it gave a diff with
not one id in it, because its header had been reworded by hand after it was written, so
"regenerate and diff" was not a test (IKA-72). The tool now writes the file's own words and
`--check` compares without writing. Like every check here it is worth nothing until it has
been seen failing, so each shape it claims to catch is put in front of it:

    the committed file edited by hand             the 9/22 shape (words, a blank line at
                                                  the end), an id struck, an id moved
    the engine naming an id the file calls inert  the shape it exists for
    the engine no longer naming one it did        the same, the other way
    the dex losing a handler                      the same for modelled.rs (IKA-210)
    the port no longer naming a custom-code move  a report instead of a silent no-op

Everything is done to copies or to text handed in: a test that edited `rust/src` and died
half-way would leave the tree it checks dirty.
"""

from __future__ import annotations

import shutil

import pytest

from ._harness import load_tool

REGULATION = "gen9championsvgc2026regmc"

tool = load_tool("port_coverage")


def test_the_committed_files_are_what_the_tool_writes(capsys):
    """What CI runs, as a test too. It failed on the tree IKA-72 was filed against."""
    assert tool.main(["--check", "--regulation", REGULATION]) == 0
    out = capsys.readouterr().out
    assert "ok       rust/src/inert.rs" in out
    assert "ok       rust/src/modelled.rs" in out


@pytest.fixture
def copies(tmp_path, monkeypatch):
    """The committed files, copied, and `--check` pointed at the copies."""
    for name in ("inert.rs", "modelled.rs"):
        shutil.copyfile(tool.RUST_SRC / name, tmp_path / name)
    monkeypatch.setattr(tool, "RUST_SRC", tmp_path)
    return tmp_path


def test_the_copies_pass_before_anything_is_changed(copies):
    """The null control for the edits below: a failure there is the edit, not the copy."""
    assert tool.main(["--check", "--regulation", REGULATION]) == 0


def _move_moxie_to_the_items(text: str) -> str:
    arm = '            | "{}"\n'
    text = text.replace(arm.format("moxie"), "", 1)
    return text.replace(arm.format("metronome"), arm.format("metronome") + arm.format("moxie"), 1)


#: The committed `inert.rs` edited by hand, and what `--check` must say about it. The first
#: three are one line each, the edits IKA-72 names.
EDITS = {
    # The 9/22 shape: a sentence of the header reworded after generation.
    "a header sentence reworded": (
        lambda text: text.replace("it must be regenerated", "it should be regenerated", 1),
        "no id moved",
    ),
    # The other half of the 9/22 diff.
    "a blank line added at the end": (lambda text: text + "\n", "no id moved"),
    # An id struck by hand. The port would stop ignoring it, and refuse it instead.
    "an id struck out": (
        lambda text: text.replace('            | "moxie"\n', "", 1),
        "ability_is_inert: the tool would list, the file does not: moxie",
    ),
    # Why the summary is per function: pooled, this would read "no id moved".
    "an id moved to the other predicate": (
        _move_moxie_to_the_items,
        "item_is_inert: the file lists, the tool would not: moxie",
    ),
}


@pytest.mark.parametrize("edit", sorted(EDITS))
def test_a_hand_edit_is_caught(copies, capsys, edit):
    change, says = EDITS[edit]
    path = copies / "inert.rs"
    before = path.read_bytes()
    path.write_bytes(change(before.decode("utf-8")).encode("utf-8"))
    assert path.read_bytes() != before, "the edit changed nothing, so it tests nothing"

    assert tool.main(["--check", "--regulation", REGULATION]) == 1
    out = capsys.readouterr().out
    assert "DIFFERS  rust/src/inert.rs" in out
    assert "ok       rust/src/modelled.rs" in out
    assert says in out


def test_crlf_is_named_rather_than_shown_as_every_line(copies, capsys):
    """This tool wrote CRLF itself until IKA-114, and a diff of that is every line of the
    file, each looking the same as its pair."""
    path = copies / "inert.rs"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert tool.main(["--check", "--regulation", REGULATION]) == 1
    out = capsys.readouterr().out
    assert "only the line endings differ" in out
    assert "---" not in out


def test_the_engine_naming_an_id_the_file_calls_inert_is_caught(capsys):
    """The shape `--check` exists for: the port learns an effect, the files are not
    regenerated, and its gate goes on taking the id as one it ignores -- and the hit goes on
    being reported as a gap it no longer has (IKA-210: the scan is the port's own)."""
    committed = (tool.RUST_SRC / "inert.rs").read_text(encoding="utf-8")
    assert '| "moxie"' in committed, "the control needs an id the file calls inert today"
    engine = tool.port_text()
    assert tool.check(REGULATION, engine=engine) == 0
    capsys.readouterr()

    learned = engine + '\n\nfn moxie(mon: &Pokemon) -> bool {\n    mon.ability == "moxie"\n}\n'
    assert tool.check(REGULATION, engine=learned) == 2
    out = capsys.readouterr().out
    assert "DIFFERS  rust/src/inert.rs" in out
    assert "ability_is_inert: the file lists, the tool would not: moxie" in out
    # Moxie has a Showdown handler (`onSourceAfterFaint`), so naming it is what takes its
    # note away.
    assert "ability_is_modelled: the tool would list, the file does not: moxie" in out


def test_the_engine_dropping_an_id_it_named_is_caught_too(capsys):
    """The other direction: the port stops naming `intimidate`, and the files go on
    treating it as an effect the port acts on."""
    engine = tool.port_text()
    assert tool.mentioned(engine, "intimidate")
    forgot = engine.replace('"intimidate"', '"intimidat"')
    assert not tool.mentioned(forgot, "intimidate")

    # Both files: the gate would take it as inert, and its hits would be noted again.
    assert tool.check(REGULATION, engine=forgot) == 2
    out = capsys.readouterr().out
    assert "ability_is_inert: the tool would list, the file does not: intimidate" in out
    assert "ability_is_modelled: the file lists, the tool would not: intimidate" in out


def test_a_handler_in_the_dex_decides_modelled_rs(capsys, monkeypatch):
    """`modelled.rs` is the dex's handlers against the port's names (IKA-210). An ability the
    port never names is noted exactly when Showdown has a handler for it: take Aftermath's
    away in a copy of the dump and it needs no note. Without this, the only test of this
    half of the check would be the one that passes."""
    committed = (tool.RUST_SRC / "modelled.rs").read_text(encoding="utf-8")
    assert '"aftermath"' not in committed, "the control needs an ability noted today"
    real = tool.regulation_dump

    def without_the_handler(regulation: str) -> dict:
        data = real(regulation)
        for entry in data["abilities"]:
            if entry["id"] == "aftermath":
                assert entry["customHooks"], "the premise: Showdown has a handler for it"
                entry["customHooks"] = []
        return data

    monkeypatch.setattr(tool, "regulation_dump", without_the_handler)
    assert tool.check(REGULATION) == 1
    out = capsys.readouterr().out
    assert "ok       rust/src/inert.rs" in out
    assert "DIFFERS  rust/src/modelled.rs" in out
    assert "ability_is_modelled: the tool would list, the file does not: aftermath" in out


def test_the_files_generated_from_m_c_cover_m_b():
    """IKA-187: the recorded games are M-B, and the files are generated from M-C only.

    Whether an id is inert depends on the port's text alone, and modelled.rs on the dump's
    handlers and M-C's stones -- so the M-C files answer every M-B id exactly as M-B's own
    would, as long as M-B has no id M-C lacks. Today its abilities and moves
    are M-C's and its items a subset. An M-B-only id would be missing from both files:
    refused by the gate (a bill), or reported by the port and not by Python (a note).
    """
    mb = tool.regulation_dump("gen9championsvgc2026regmb")
    mc = tool.regulation_dump(REGULATION)
    for kind in ("abilities", "items", "moves"):
        extra = {e["id"] for e in mb[kind]} - {e["id"] for e in mc[kind]}
        assert not extra, f"M-B {kind} that M-C lacks: {sorted(extra)}"
    assert set(mb["megaMap"]) <= set(mc["megaMap"])


def test_a_custom_code_move_the_port_stops_naming_is_reported(capsys):
    """IKA-187's shape, where it lives since IKA-210. Moonlight, Synthesis and Morning Sun
    heal by `onHit` (the dump has no `heal` field), so a port that stopped naming them would
    heal nothing -- and the regenerated `modelled.rs` takes them out of the fully modelled,
    so the turn reports `status move: moonlight` instead of saying nothing."""
    engine = tool.port_text()
    forgot = engine
    for move in ("moonlight", "synthesis", "morningsun"):
        assert tool.mentioned(engine, move)
        forgot = forgot.replace(f'"{move}"', f'"{move[:-1]}"')
    assert tool.check(REGULATION, engine=forgot) >= 1
    out = capsys.readouterr().out
    says = "status_move_is_fully_modelled: the file lists, the tool would not: "
    assert says + "moonlight, morningsun, synthesis" in out


def test_a_move_that_is_its_fields_needs_no_name():
    """The null control: Dragon Dance has no custom code (`boosts` is the whole move) and
    the port never names it, and it is fully modelled all the same."""
    assert not tool.mentioned(tool.port_text(), "dragondance")
    assert '"dragondance"' in (tool.RUST_SRC / "modelled.rs").read_text(encoding="utf-8")
