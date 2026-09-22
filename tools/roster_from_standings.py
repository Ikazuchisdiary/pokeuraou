"""Writes a tournament six out as an own-side roster, so the rating can leave one team.

Every number this project has measured is conditional on one six. The anchor, the
generation ladder, the +141 Elo the book is worth -- all of it is "this agent, playing
rizabanadohido", and whether generation 11L beats generation 10 *in general* has never
been asked. It cannot be, because `configs/teams/` holds exactly one file.

The cheapest thing that settles whether that matters: take two models whose ordering is
known on our six, put them on a DIFFERENT six, and see whether the ordering survives. No
new training, no new generation, one match.

What this tool cannot do honestly: a standings entry has species, ability, item, nature
and moves, and NO investment -- the SP spread is exactly the hidden information the
belief layer exists to model. So the spread is sampled from the usage prior, with the
seed recorded, and the written file says so in `source`. It is a well-formed team, not
a real player's team, and a file that claimed otherwise would be the kind of record this
project has spent a day repairing.

    uv run python tools/roster_from_standings.py --place 1 --out-id worlds1
    uv run python tools/roster_from_standings.py --place 1 --seed 7 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.regulation import load_regulation, repo_root  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--place", type=int, required=True, help="the standings place to take")
    ap.add_argument("--out-id", default=None, help="file stem; default place<N>")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seeds the spread draw. Recorded in the file, because the spreads are the "
        "one part of this team that nobody played -- a different seed is a different "
        "team, and the file has to say which one it is.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reg = load_regulation(args.format)
    standings = load_standings(find_cached_standings(), reg)
    prior = load_chaos(find_cached_chaos(args.format), reg)
    wanted = [t for t in standings.pool("all") if t.place == args.place]
    if not wanted:
        places = sorted({t.place for t in standings.pool("all")})
        raise SystemExit(
            f"no team at place {args.place}; the pool holds "
            f"{len(places)} places from {places[0]} to {places[-1]}"
        )
    team = wanted[0]
    rng = np.random.default_rng([args.seed, args.place])
    sampled = sample_standings_team(rng, reg, prior, team)

    out_id = args.out_id or f"place{args.place}"
    document = {
        "id": out_id,
        "name": f"{team.player}（{team.place}位）",
        "side": "own",
        "regulation": args.format,
        "source": {
            "kind": "standings six, spreads SAMPLED from the usage prior",
            "place": team.place,
            "player": team.player,
            "spread_seed": args.seed,
            "note": (
                "The species, abilities, items, natures and moves are the recorded sheet. "
                "The SP spreads are NOT: a standings entry has no investment, which is "
                "the hidden information the belief layer models, so they are drawn from "
                "the usage prior under `spread_seed`. This is a well-formed team, not "
                "this player's team, and nothing measured on it may be attributed to them."
            ),
        },
        "team": [
            {
                "species": reg.species[entry.species].name,
                "ability": reg.abilities[entry.ability].name
                if entry.ability in reg.abilities
                else entry.ability,
                "item": (
                    reg.items[entry.item].name
                    if entry.item and entry.item in reg.items
                    else entry.item
                ),
                "nature": entry.nature,
                "moves": [
                    reg.moves[m].name if m in reg.moves else m for m in entry.moves
                ],
                "sp": {k: v for k, v in entry.sp.items() if v},
            }
            for entry in sampled
        ],
    }

    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.dry_run:
        print(text)
        return
    path = repo_root() / "configs" / "teams" / f"{out_id}.json"
    if path.exists():
        raise SystemExit(
            f"{path} exists. A roster file is referred to by name from recorded games, "
            "so overwriting one silently rewrites what those games were played with."
        )
    # `newline`: a tracked file, and text mode would write it CRLF on Windows.
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"-> {path}")
    print(f"   {team.player}, place {team.place}, spreads sampled at seed {args.seed}")
    for entry in sampled:
        total = sum(entry.sp.values())
        print(
            f"   {reg.species[entry.species].name:14s} {entry.nature:10s} "
            f"{entry.item or '-':18s} sp {total}"
        )


if __name__ == "__main__":
    main()
