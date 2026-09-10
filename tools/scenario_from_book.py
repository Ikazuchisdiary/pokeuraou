"""Builds the turn-1 scenario the selection book actually expects, ready for the analyser.

The book says what to bring against a given team sheet and what that opponent brings back.
That answer is only useful if you can then look at the resulting turn -- "so I brought
Charizard + Garchomp into Kingambit, now what?" -- and the analyser already answers exactly
that from a scenario file. This is the plumbing between them.

    uv run --group learn python tools/scenario_from_book.py --place 1 --out /c/tmp/t1.json
    uv run python -m pokeuraou.cli /c/tmp/t1.json --out /c/tmp/t1.txt

Two choices worth stating, because they decide what the analyser is being asked:

- **the opponent's SP is left hidden**, with only the nature written down. That is what an
  open team sheet actually shows, so the analyser runs its belief layer over the spread
  rather than being handed the answer. Passing the sampled spread instead would quietly
  turn the hardest part of the problem off.
- **the opponent's four are written in full**, which is *not* what a player knows at turn 1
  -- their other two are hidden until they appear. There is no belief layer over the bench
  yet, so this overstates what we know; the scenario records the fact in a note rather
  than pretending otherwise.

Selections default to each side's equilibrium from the book, so the position is the one the
solver's own advice leads to. Either can be overridden by name to ask a different question
("what if I lead Venusaur instead").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.priors import SampledSet
from pokeuraou.regulation import Regulation, to_id
from pokeuraou.selection_book import BookEntry, SelectionBook, key_for_team, selection_dir
from pokeuraou.standings import TournamentTeam, find_cached_standings, load_standings
from pokeuraou.teams import all_selections, load_roster


def our_member(reg: Regulation, entry: SampledSet, active: bool) -> dict:
    species = reg.species[entry.species]
    item = reg.items.get(entry.item or "")
    ability = reg.abilities.get(entry.ability)
    out = {
        "species": species.name,
        "ability": ability.name if ability else species.abilities[0],
        "item": item.name if item else "",
        "nature": entry.nature,
        "moves": [reg.moves[m].name for m in entry.moves],
        "sp": {k: v for k, v in entry.sp.items() if v},
    }
    if active:
        out["active"] = True
    return out


def their_member(
    reg: Regulation, team: TournamentTeam, index: int, fallback: SampledSet, active: bool
) -> dict:
    """Their sheet, with the investment left out on purpose.

    The ability comes from the sheet where the sheet showed one. Champions sheets show the
    pre-mega ability, so a member recorded with the mega forme's is taken from the book's
    sampled class instead -- and that is a *guess*, which is why it is reported.
    """
    member = team.members[index]
    ability_id = (
        fallback.ability
        if member.ability is None or member.ability_is_post_mega
        else member.ability
    )
    ability = reg.abilities.get(ability_id)
    species = reg.species[member.species]
    item = reg.items.get(member.item or "")
    out = {
        "species": species.name,
        "ability": ability.name if ability else species.abilities[0],
        "item": item.name if item else "",
        "nature": member.nature,
        "moves": [reg.moves[to_id(m)].name for m in member.moves],
    }
    if active:
        out["active"] = True
    return out


def resolve_selection(
    entry: BookEntry,
    selections: list[tuple[int, ...]],
    names: list[str],
    override: str | None,
    strategy: np.ndarray,
) -> tuple[int, ...]:
    if override is None:
        return selections[int(np.argmax(strategy))]
    wanted = [to_id(part) for part in override.replace("/", ",").replace("+", ",").split(",")]
    where = {to_id(n): i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in where]
    if missing:
        raise SystemExit(f"{missing} not on that six: {sorted(where)}")
    if len(wanted) != len(selections[0]):
        raise SystemExit(f"a selection is {len(selections[0])} names, got {len(wanted)}")
    front = tuple(sorted(where[w] for w in wanted[:2]))
    back = tuple(sorted(where[w] for w in wanted[2:]))
    if front + back not in selections:
        raise SystemExit(f"{front + back} is not a legal selection")
    return front + back


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--model", default="value-gen2")
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--place", type=int, default=None)
    ap.add_argument("--player", default=None)
    ap.add_argument(
        "--species",
        default=None,
        help="pick the highest-placed team whose sheet has this species, e.g. kingambit",
    )
    ap.add_argument("--ours", default=None, help="lead1+lead2/back1+back2 by species name")
    ap.add_argument("--theirs", default=None, help="same, on their six")
    ap.add_argument("--class-index", type=int, default=0, help="which spread class to use")
    ap.add_argument("--turn", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    book_path = args.book or (selection_dir() / f"{args.roster}-{args.model}.jsonl.gz")
    book = SelectionBook.read(book_path)
    book.require_roster(args.roster)
    standings = load_standings(find_cached_standings(), reg)
    covered = [t for t in standings.teams if key_for_team(t) in book.entries]

    if args.player:
        matching = [t for t in covered if args.player.lower() in t.player.lower()]
    elif args.species:
        wanted = to_id(args.species)
        matching = sorted(
            (t for t in covered if wanted in {to_id(s) for s in t.species}),
            key=lambda t: t.place,
        )
    else:
        place = args.place if args.place is not None else 1
        matching = [t for t in covered if t.place == place]
    if not matching:
        raise SystemExit("no team in the book matches that filter")
    team = matching[0]
    entry = book.entries[key_for_team(team)]
    selections = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    ours = resolve_selection(
        entry,
        selections,
        [s.species for s in roster.sets],
        args.ours,
        np.asarray(entry.our_strategy, dtype=np.float64),
    )
    class_index = max(0, min(args.class_index, len(entry.class_sets) - 1))
    theirs = resolve_selection(
        entry,
        selections,
        [s.species for s in entry.class_sets[class_index]],
        args.theirs,
        np.asarray(entry.their_strategies[class_index], dtype=np.float64),
    )

    scenario = {
        "regulation": reg.meta.format_id,
        "turn": args.turn,
        "note": (
            f"{team.player}（{team.place}位）に対する選出解のターン{args.turn}。"
            f"自陣は均衡頻度 {float(entry.our_strategy[selections.index(ours)]) * 100:.1f}% の選出、"
            f"相手は配分クラス {class_index + 1} の均衡選出（頻度 "
            f"{float(entry.their_strategies[class_index][selections.index(theirs)]) * 100:.1f}%）。"
            "均衡値 "
            f"{entry.value * 100:.1f}%。"
            "相手の配分は伏せてある（シートが見せないので信念層が扱う）。"
            "一方で相手の4匹は全部書いてあり、これは実戦より知りすぎている"
            "（裏2匹の信念層は未実装）"
        ),
        "sides": [
            {
                "id": "p1",
                "name": "自陣",
                "team": [
                    our_member(reg, roster.sets[i], active=rank < 2)
                    for rank, i in enumerate(ours)
                ],
            },
            {
                "id": "p2",
                "name": team.player,
                "team": [
                    their_member(
                        reg, team, j, entry.class_sets[class_index][j], active=rank < 2
                    )
                    for rank, j in enumerate(theirs)
                ],
            },
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(scenario["note"], file=sys.stderr)
    print(f"→ {args.out}", file=sys.stderr)
    print(
        "  次: uv run python -m pokeuraou.cli "
        f"{args.out} --out <読める出力先>",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
