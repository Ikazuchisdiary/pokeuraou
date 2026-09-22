"""Does `names_report.py --skeleton` keep what was typed into the override file? (IKA-115)

Asked on 2026-09-23. Runs the tool from before the fix and the tool in this tree on the same
inputs. Only the file it rewrites is moved to a temporary directory: the regulation and the
generated names are the real ones, so which kinds have gaps is what the tool itself sees.

    git show 9ecc11a:tools/names_report.py > <tmp>/before.py
    uv run python scratchpad/ika115_skeleton.py <tmp>/before.py

Answer that day (before = 9ecc11a, after = the fix; one long line wrapped):

    A no file              bytes identical True | kinds ['_note', 'species', 'items']
    B checked-in file      after == checked-in bytes True
                           before loses [('moves', 'recharge')] | kinds ['_note', 'species', 'items']
    C 7 hand values        after keeps all True
                           before drops [('abilities', 'zzhandability'), ('moves', 'recharge'),
                             ('moves', 'zzhandmove'), ('natures', 'adamant')]
                           stale empty placeholder dropped by after True
    D only `items`         before ['_note', 'species', 'items'] | after ['_note', 'items', 'species']
    E `_source`  before    file unchanged False | _source None | recharge None | error -
    E `_source`  after     file unchanged True | _source （手で書いた注） | recharge 反動で動けない | error ValueError: dictionary update sequence element #0 has length 1; 2 is required
    F filled species/item  species alcremie 0 -> 9 of 10 | items abomasite 0 -> 0 of 75

E is a file shape nothing writes today: the fixed tool stops before it writes, so nothing is
lost, where the old one dropped the key silently. F is a move within a kind, not a loss:
species coverage counts a hand-filled name as translated, items coverage does not.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pokeuraou.names as names_module  # noqa: E402


def load(path: Path, name: str):  # noqa: ANN201
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


before = load(Path(sys.argv[1]), "names_report_before")
after = load(ROOT / "tools" / "names_report.py", "names_report_after")


def run(tool, existing: bytes | None, directory: Path | None = None) -> tuple[bytes, str]:  # noqa: ANN001
    """The bytes `--skeleton` leaves in the file, and the error it stopped on, if any."""
    with tempfile.TemporaryDirectory() as tmp:
        where = directory or Path(tmp)
        if existing is not None:
            (where / "ja-extra.json").write_bytes(existing)
        tool.names_dir = lambda: where
        tool.find_cached_chaos = lambda *_args, **_kwargs: None
        sys.argv = ["names_report.py", "--locale", "ja", "--skeleton"]
        error = "-"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tool.main()
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        return (where / "ja-extra.json").read_bytes(), error


def filled(doc: dict) -> dict[tuple[str, str], str]:
    return {
        (kind, key): value
        for kind, table in doc.items()
        if kind != "_note" and isinstance(table, dict)
        for key, value in table.items()
        if value
    }


real_bytes = (ROOT / "configs" / "names" / "ja-extra.json").read_bytes()
real = json.loads(real_bytes)

# A. no file yet: the default output must not have changed.
a_before, _ = run(before, None)
a_after, _ = run(after, None)
print("A no file              bytes identical", a_before == a_after, "| kinds", list(json.loads(a_after)))

# B. the checked-in file.
b_before, _ = run(before, real_bytes)
b_after, _ = run(after, real_bytes)
lost = sorted(set(filled(real)) - set(filled(json.loads(b_before))))
print("B checked-in file      after == checked-in bytes", b_after == real_bytes)
print("                       before loses", lost, "| kinds", list(json.loads(b_before)))

# C. a hand value in every place one can sit.
synthetic = json.loads(real_bytes)
synthetic["moves"]["zzhandmove"] = "手2"  # a kind with no gap, a second name in it
synthetic["abilities"] = {"zzhandability": "手3"}  # a kind with no gap, new to the file
synthetic["items"]["abomasite"] = "手4"  # a kind with gaps, the id still missing
synthetic["items"]["zzhanditem"] = "手5"  # a kind with gaps, an id outside the regulation
synthetic["species"]["zzhandspecies"] = "手6"  # species, an id outside the regulation
synthetic["natures"] = {"adamant": "手7"}  # a kind load_names does not read
synthetic["moves"]["zzstale"] = ""  # an empty placeholder for an id that is not missing
c_in = json.dumps(synthetic, ensure_ascii=False, indent=2).encode("utf-8")
c_before, c_after = (json.loads(run(tool, c_in)[0]) for tool in (before, after))
want = filled(synthetic)
print(
    f"C {len(want)} hand values        after keeps all",
    all((c_after.get(kind) or {}).get(key) == value for (kind, key), value in want.items()),
)
print(
    "                       before drops",
    sorted(k for k, v in want.items() if (c_before.get(k[0]) or {}).get(k[1]) != v),
)
print("                       stale empty placeholder dropped by after", "zzstale" not in c_after["moves"])

# D. a file holding only `items`: the kind with gaps it lacks goes after it.
d_in = json.dumps({"items": {"abomasite": "（手）"}}, ensure_ascii=False).encode("utf-8")
print(
    "D only `items`         before", list(json.loads(run(before, d_in)[0])),
    "| after", list(json.loads(run(after, d_in)[0])),
)

# E. a top-level value that is not a table of names.
with_source = json.loads(real_bytes)
with_source["_source"] = "（手で書いた注）"
e_in = json.dumps(with_source, ensure_ascii=False, indent=2).encode("utf-8")
for label, tool in (("before", before), ("after", after)):
    out, error = run(tool, e_in)
    doc = json.loads(out)
    print(
        f"E `_source`  {label:6}    file unchanged", out == e_in, "| _source", doc.get("_source"),
        "| recharge", (doc.get("moves") or {}).get("recharge"), "| error", error,
    )

# F. a filled species id against a filled item, with load_names reading the same file the
# tool rewrites (as in real use). Species coverage counts an override as translated; items
# coverage reads Showdown's table only.
with tempfile.TemporaryDirectory() as tmp:
    where = Path(tmp)
    shutil.copy(ROOT / "configs" / "names" / "ja.json", where / "ja.json")
    marked = json.loads(real_bytes)
    marked["species"]["alcremie"] = "（手）"
    marked["items"]["abomasite"] = "（手）"
    names_module.names_dir = lambda: where
    names_module.load_names.cache_clear()
    out, _ = run(after, json.dumps(marked, ensure_ascii=False, indent=2).encode("utf-8"), where)
    doc = json.loads(out)
    print(
        "F filled species/item  species alcremie", list(marked["species"]).index("alcremie"), "->",
        list(doc["species"]).index("alcremie"), "of", len(doc["species"]),
        "| items abomasite", list(marked["items"]).index("abomasite"), "->",
        list(doc["items"]).index("abomasite"), "of", len(doc["items"]),
    )
