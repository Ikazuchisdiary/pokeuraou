"""What the Japanese name table covers, and what is left in English.

Coverage is measured against a regulation rather than against Showdown's whole text data:
Showdown has hundreds of untranslated Pokedex entries, nearly all Gigantamax and cosmetic
formes Champions never sees, so a global figure reports a problem that does not exist and
hides the one that does.

Untranslated items are ranked by how much of the metagame actually holds them, because
"75 items untranslated" and "the item on 27% of teams is untranslated" are very different
statements and only the second one is worth acting on.

    uv run python tools/names_report.py
    uv run python tools/names_report.py --skeleton   # write the override file to fill in
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.names import DEFAULT_LOCALE, available_locales, load_names, localiser, names_dir
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.regulation import load_regulation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regulation", default="gen9championsvgc2026regmb")
    ap.add_argument("--locale", default=DEFAULT_LOCALE)
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument(
        "--skeleton",
        action="store_true",
        help="write configs/names/<locale>-extra.json with the untranslated ids as empty "
        "keys, ready to fill in. An empty value falls through to English, so a half-filled "
        "file is safe.",
    )
    args = ap.parse_args()

    reg = load_regulation(args.regulation)
    names = load_names(args.locale)
    loc = localiser(reg, args.locale)
    assert loc is not None

    print(f"locale {names.locale}, from Showdown {names.showdown_commit[:12]}")
    print(f"available locales: {', '.join(available_locales()) or '(none)'}")
    print(f"\ncoverage over {reg.meta.format_id}")
    coverage = names.coverage(reg)
    for entry in coverage:
        print(f"  {entry}")
    if names.overrides:
        filled = {k: len(v) for k, v in names.overrides.items()}
        print(f"  hand-supplied via {args.locale}-extra.json: {filled}")

    # Rank the untranslated items by the share of teams that hold one.
    cached = find_cached_chaos(reg.meta.format_id)
    items = next(c for c in coverage if c.kind == "items")
    if cached is not None and items.missing:
        prior = load_chaos(cached, reg)
        weight: dict[str, float] = {}
        for species in prior.species.values():
            for item_id, share in species.items.items():
                weight[item_id] = weight.get(item_id, 0.0) + share * species.usage
        ranked = sorted(items.missing, key=lambda i: -weight.get(i, 0.0))
        held = sum(weight.get(i, 0.0) for i in items.missing)
        total = sum(weight.values()) or 1.0
        print(
            f"\nuntranslated items carry {held / total * 100:.1f}% of all item slots "
            f"({len(items.missing)} ids). Ranked by how often they are actually held:"
        )
        for item_id in ranked[: args.show]:
            english = reg.items[item_id].name if item_id in reg.items else item_id
            share = weight.get(item_id, 0.0)
            if share <= 0:
                break
            print(f"  {share * 100:5.2f}% of slots  {item_id:22} {english}")
        unheld = [i for i in ranked if weight.get(i, 0.0) <= 0]
        if unheld:
            print(f"  ...and {len(unheld)} nobody in this metagame holds")

    print("\nspot check")
    for species_id in ("charizard", "charizardmegay", "sinistcha", "sneasler", "kingambit"):
        if species_id in reg.species:
            print(f"  {species_id:18} {loc.species(species_id)}")

    if args.skeleton:
        path = names_dir() / f"{args.locale}-extra.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        out: dict[str, object] = {
            "_note": (
                "Showdown が訳していない項目を手で補うファイル。生成される "
                f"{args.locale}.json とは別なので、ダンプを再生成しても消えない。"
                "値が空文字のものは英語名にフォールバックするので、途中まで埋めた状態でも安全。"
                "根拠のない訳を入れないこと（tools/names_report.py が不足を数える）。"
            ),
        }
        for entry in coverage:
            if entry.kind not in ("species", "items", "moves", "abilities") or not entry.missing:
                continue
            previous = dict(existing.get(entry.kind) or {})
            out[entry.kind] = {
                key: previous.get(key, "") for key in entry.missing
            } | {k: v for k, v in previous.items() if v}
        # `newline`: a tracked file, and text mode would write it CRLF on Windows.
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"\n-> {path} ({sum(len(v) for k, v in out.items() if isinstance(v, dict))} ids)")


if __name__ == "__main__":
    main()
