"""IKA-411 step 3: how often the forgotten Mega forme shaped a gen-3 belief.

    python impact.py <data dir> <out.json> <jobs>

For every game of the directory (hidden bench) and each side S whose six holds Floette-Eternal
or Meowstic-F: every recorded decision (move or replacement) where S's Pokemon of that forme
is a Mega on the board and S still has unseen slots. There the other side's belief about S
was built as `play_game` built it -- completions of S's six with seen_slots of the recorded
shownIdentities -- and is built again with that Mega's base_species set to its set species
(what the fix makes `_make_pokemon` write). A decision counts as affected when the two
counts differ. Also counted: whether the other side's recorded rankViews (the completion it
ranked its menu from) holds the forme itself (a world with a second Floette).
"""

import json
import sys
from collections import Counter
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

FORMES = {"floettemega": "floetteeternal", "meowsticfmega": "meowsticf"}
NEEDLES = [k.encode() for k in FORMES]


def init():  # noqa: ANN201
    global reg, completions, seen_slots, Position, pool, teams
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.hidden import completions, seen_slots
    from pokeuraou.pool import load_pool
    from pokeuraou.position import Position
    pool = load_pool("C:/Users/Ikazuchi/repos/pokeuraou/data/pool/regmc-matchupweb.json")
    teams = {t.id: t for t in pool.teams}
    reg = pool.reg
    register_mega_stones(reg)


def sheet_of(game, side):  # noqa: ANN001, ANN201
    named = game["pool"]
    assert named["sha256"] == pool.sha256
    team = teams[named["teams"][side]]
    return list(team.sets)


def scan(path: str) -> dict:
    c = Counter()
    examples = []
    with open(path, "rb") as fh:
        for n, line in enumerate(fh):
            c["games"] += 1
            for kind in ("move", "replacement", "selfswitch"):
                c[f"all_{kind}"] += line.count(f'"kind": "{kind}"'.encode())
            if not any(x in line for x in NEEDLES):
                # still count the games where a side brought the forme but never Mega Evolved
                if b"floetteeternal" in line or b"meowsticf" in line:
                    c["games_forme_on_a_sheet_no_mega"] += 1
                continue
            g = json.loads(line)
            if g.get("information") != "hidden-bench":
                c["games_open"] += 1
                continue
            c["games_with_forme_mega"] += 1
            game_hit = False
            for side in (0, 1):
                sheet = sheet_of(g, side)
                sheet_ids = {s.species for s in sheet}
                if not sheet_ids & set(FORMES.values()):
                    continue
                side_hit = False
                for k, d in enumerate(g["decisions"]):
                    pos = Position.from_json(d["position"])
                    mons = pos.sides[side].pokemon
                    megas = [m for m in mons if m.is_mega and m.species in FORMES]
                    if not megas:
                        continue
                    shown = d.get("shownIdentities")
                    if shown is None:
                        c["decisions_without_shown"] += 1
                        continue
                    seen = seen_slots(pos, side, shown[side])
                    if len(seen) == len(mons):
                        c[f"{d['kind']}_after_mega_bench_open"] += 1
                        continue
                    old = completions(reg, pos, side, sheet, seen=seen)
                    fixed_pos = deepcopy(pos)
                    for m in fixed_pos.sides[side].pokemon:
                        if m.is_mega and m.species in FORMES:
                            m.base_species = FORMES[m.species]
                    new = completions(reg, fixed_pos, side, sheet, seen=seen)
                    c[f"{d['kind']}_after_mega_bench_hidden"] += 1
                    if len(old) != len(new):
                        c[f"{d['kind']}_affected"] += 1
                        c[f"{d['kind']}_affected_{len(new)}to{len(old)}"] += 1
                        side_hit = True
                        rv = d.get("rankViews")
                        if d["kind"] == "move" and rv is not None and rv[1 - side] is not None:
                            if set(FORMES.values()) & set(rv[1 - side][1]):
                                c["move_ranked_from_impossible_world"] += 1
                        if len(examples) < 5:
                            examples.append({"file": Path(path).name, "line": n, "game": g.get("gameIndex"),
                                             "side": side, "decision": k, "turn": d["turn"], "kind": d["kind"],
                                             "old": len(old), "new": len(new)})
                if side_hit:
                    c["sides_affected"] += 1
                    game_hit = True
            if game_hit:
                c["games_affected"] += 1
    return {"counts": dict(c), "examples": examples}


def main() -> None:
    import os

    src, out, jobs = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    # heavy.py's elastic grant (--cores-min): use what was granted, not the ceiling.
    jobs = min(jobs, int(os.environ.get("HEAVY_CORES", jobs)))
    print("jobs", jobs, file=sys.stderr, flush=True)
    files = sorted(str(p) for p in src.glob("games-*.jsonl"))
    total = Counter()
    examples = []
    with Pool(jobs, initializer=init) as pool:
        for k, got in enumerate(pool.imap_unordered(scan, files)):
            total.update(got["counts"])
            examples += got["examples"]
            if k % 40 == 0:
                print(k, len(files), dict(total), file=sys.stderr, flush=True)
    out.write_bytes(json.dumps({"counts": dict(total), "examples": examples[:40], "files": len(files)},
                               indent=1).encode())
    print(json.dumps(dict(total), indent=1))


if __name__ == "__main__":
    main()
