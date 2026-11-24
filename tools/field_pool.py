"""IKA-448: a pool of tournament teams, to measure what the value net never learned.

The generation pool (`regmc-matchupweb`, 65 teams) is one player's prep list from 9/22. The
M-C field since then (ten events in `data/standings/2027-*.json.gz`) brings teams and sets the
pool never shows. This tool turns field teams into a pool file `match_queue.py --pool` and
`team_search.py value` read:

* The field's teams are grouped by their six species. A group's weight is how many of its
  teams reached day 2 (phase two), then how many entered; the top groups are the
  representatives. Each representative is the group's best-placed team, whole, so its six
  sets are one real team that passed the Item Clause.
* A team sheet shows no spread (SP). The spread is the IKA-407 rule (`rankedentry`): the
  pool's commonest spread of the species with the same nature, else of the species, else a
  neutral one from the base stats, labelled "neutral".
* A mega's sheet ability is recorded as the mega forme's (`ability_is_post_mega`); the
  species' commonest pre-mega ability in the field stands in for it, and the team says so.

Every team carries, under ``ika448``, its species overlap with the closest pool team and
the ids (species, moves, items, abilities) no pool team has: value-mc6 learned from the
pool's games alone, so those ids' embedding rows were never trained.

    field_pool.py --events baltimore frankfurt ... --reps 30 --out field.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.pool import find_pool, load_pool  # noqa: E402
from pokeuraou.rankedentry import neutral_spread  # noqa: E402
from pokeuraou.regulation import STAT_IDS, repo_root, to_id  # noqa: E402
from pokeuraou.standings import TournamentTeam, load_standings  # noqa: E402
from pokeuraou.teams import TeamError, roster_from_data  # noqa: E402

EVENTS = ("baltimore", "frankfurt", "brisbane", "recife", "louisville", "nice", "puebla",
          "gdansk", "buenos-aires", "laic")


def _write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes((json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    os.replace(tmp, path)


class Spreads:
    """The IKA-407 spread rule over the pool's sets."""

    def __init__(self, pool: Any, reg: Any) -> None:  # noqa: ANN401
        self.reg = reg
        self.have: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
        for roster in pool.teams:
            for one in roster.sets:
                self.have.setdefault(one.species, []).append(
                    (one.nature, tuple(int(one.sp.get(s, 0)) for s in STAT_IDS)))

    def pick(self, species: str, nature: str) -> tuple[dict[str, int], str]:
        have = self.have.get(species, [])
        for label, found in (("pool-nature", [s for n, s in have if n == nature]),
                             ("pool-species", [s for _n, s in have])):
            if found:
                top = sorted(Counter(found).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                return {k: v for k, v in zip(STAT_IDS, top, strict=True) if v}, label
        return {k: v for k, v in neutral_spread(self.reg, species).items() if v}, "neutral"


def pool_ids(pool: Any) -> dict[str, set[str]]:  # noqa: ANN401
    out: dict[str, set[str]] = {"species": set(), "moves": set(), "items": set(), "abilities": set()}
    for roster in pool.teams:
        for one in roster.sets:
            out["species"].add(one.species)
            out["moves"].update(one.moves)
            if one.item:
                out["items"].add(one.item)
            if one.ability:
                out["abilities"].add(one.ability)
    return out


def group_teams(events: list[tuple[str, list[TournamentTeam]]]) -> list[dict[str, Any]]:
    """Field teams grouped by their six species, heaviest first."""
    groups: dict[frozenset[str], dict[str, Any]] = {}
    for event, teams in events:
        for team in teams:
            key = frozenset(team.species)
            g = groups.setdefault(key, {"species": key, "day2": 0, "entries": 0, "teams": []})
            g["entries"] += 1
            g["day2"] += int(team.phase_two)
            g["teams"].append((event, team))
    out = list(groups.values())
    out.sort(key=lambda g: (-g["day2"], -g["entries"], sorted(g["species"])))
    return out


def best_team(group: dict[str, Any]) -> tuple[str, TournamentTeam]:
    """The group's best-placed team (day 2 first, then place, then event order)."""
    return min(group["teams"], key=lambda et: (not et[1].phase_two, et[1].place, et[0]))


def to_entry(event: str, team: TournamentTeam, reg: Any, spreads: Spreads,
             abilities: dict[str, Counter[str]], known: dict[str, set[str]],
             pool_sixes: list[frozenset[str]]) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    notes: list[str] = []
    for m in team.members:
        ability = m.ability
        if ability is None or m.ability_is_post_mega:
            seen = abilities.get(m.species)
            ability = (seen.most_common(1)[0][0] if seen
                       else to_id(reg.species[m.species].abilities[0]))
            notes.append(f"{m.species}: ability {ability} (sheet showed the mega's)")
        sp, source = spreads.pick(m.species, m.nature)
        if source != "pool-nature":
            notes.append(f"{m.species}: spread from {source}")
        member = {"species": reg.species[m.species].name, "ability": ability,
                  "nature": m.nature, "moves": list(m.moves), "sp": sp}
        if m.item:
            member["item"] = m.item
        members.append(member)
    six = frozenset(team.species)
    missing = {
        "species": sorted(s for s in team.species if s not in known["species"]),
        "moves": sorted({mv for m in team.members for mv in m.moves} - known["moves"]),
        "items": sorted({m.item for m in team.members if m.item} - known["items"]),
        "abilities": sorted({str(x["ability"]) for x in members} - known["abilities"]),
    }
    blob = json.dumps(members, sort_keys=True, ensure_ascii=False).encode("utf-8")
    tid = "f" + hashlib.sha256(blob).hexdigest()[:15]
    names = " ".join(reg.species[s].name for s in team.species)
    return {
        "id": tid, "name": f"{event} {team.place} {names}", "team": members,
        "ika448": {"event": event, "place": team.place, "day2": team.phase_two,
                   "overlap": max(len(six & p) for p in pool_sixes),
                   "unseen": missing, "notes": notes},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", nargs="+", default=list(EVENTS))
    ap.add_argument("--season", default="2027")
    ap.add_argument("--standings", default=None, help="the standings directory (default data/standings)")
    ap.add_argument("--pool", default="regmc-matchupweb")
    ap.add_argument("--reps", type=int, default=30, help="how many groups to keep")
    ap.add_argument("--unseen-reps", type=int, default=0,
                    help="then this many more groups, heaviest first, whose team has at least "
                         "--unseen-min ids no pool team has")
    ap.add_argument("--unseen-min", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    reg = pool.reg
    pool_data = json.loads(find_pool(args.pool).read_bytes().decode("utf-8"))
    spreads = Spreads(pool, reg)
    known = pool_ids(pool)
    pool_sixes = [frozenset(one.species for one in r.sets) for r in pool.teams]

    loaded: list[tuple[str, list[TournamentTeam]]] = []
    abilities: dict[str, Counter[str]] = {}
    for event in args.events:
        root = Path(args.standings) if args.standings else repo_root() / "data" / "standings"
        path = root / f"{args.season}-{event}.json.gz"
        st = load_standings(path, reg)
        loaded.append((event, st.teams))
        for team in st.teams:
            for m in team.members:
                if m.ability and not m.ability_is_post_mega:
                    abilities.setdefault(m.species, Counter())[m.ability] += 1
        print(f"{event}: {len(st.teams)} teams ({len(st.rejected)} rejected)")

    groups = group_teams(loaded)
    kept: list[dict[str, Any]] = []
    refused = 0
    unseen_kept = 0
    for g in groups:
        if len(kept) >= args.reps + args.unseen_reps:
            break
        event, team = best_team(g)
        entry = to_entry(event, team, reg, spreads, abilities, known, pool_sixes)
        if len(kept) >= args.reps:
            if sum(len(v) for v in entry["ika448"]["unseen"].values()) < args.unseen_min:
                continue
            entry["ika448"]["pick"] = "unseen"
            unseen_kept += 1
        else:
            entry["ika448"]["pick"] = "weight"
        try:
            roster_from_data({**entry, "regulation": pool_data["regulation"]},
                             default_id=entry["id"], reg=reg)
        except (TeamError, KeyError, ValueError) as error:
            refused += 1
            print(f"refused {entry['name']}: {error}")
            continue
        entry["ika448"].update(groupDay2=g["day2"], groupEntries=g["entries"])
        kept.append(entry)

    out = {k: v for k, v in pool_data.items() if k not in ("teams", "source", "counts", "excluded")}
    out["id"] = Path(args.out).stem
    out["name"] = f"IKA-448 field: top {len(kept)} six-species groups of {', '.join(args.events)}"
    out["notes"] = ("IKA-448: field teams grouped by six species, weighted by day-2 entries; "
                    "spreads by the IKA-407 rule; see each team's ika448 block")
    out["teams"] = kept
    _write(Path(args.out), out)
    print(f"{len(groups)} groups; kept {len(kept)}, refused {refused}")
    for e in kept:
        x = e["ika448"]
        unseen = sum(len(v) for v in x["unseen"].values())
        print(f"  {e['id']} day2 {x['groupDay2']:3d} entries {x['groupEntries']:3d} "
              f"overlap {x['overlap']} unseen {unseen:2d}  {e['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
