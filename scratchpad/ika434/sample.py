"""IKA-434 step 1: held-out gen-4 games -> positions.jsonl (one line per recorded decision,
for the port's residual-features) and sample.npz (game, turn, kind, outcome and the
features Python reads off the record).

    python sample.py GAMES SEED

The held-out games are value-mc4bindaux's: Dataset.split_by_game(0.15, 0) on the merged
gen-0..4 game numbers (IKA-421 §1), gen-4's share of them. Every decision of each picked game
is kept, in the shard's row order (the picked games are the first GAMES held-out games in
the files' order); the decisions' count and the outcome are checked against
the gen-4 shard.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-aa55f8431dd4ced5d/src")
from pokeuraou.regulation import load_regulation  # noqa: E402

DATA = Path("C:/Users/Ikazuchi/repos/pokeuraou/data")
OUT = Path("C:/tmp/ika434")
N_GAMES, SEED = int(sys.argv[1]), int(sys.argv[2])
started = time.time()

merged = np.load(DATA / "selfplay-mc01234-encoded.npz")["game"]
games = np.unique(merged)
rng = np.random.default_rng(0)
rng.shuffle(games)
cut = int(len(games) * (1.0 - 0.15))
val_games = np.sort(games[cut:])
offset = 0
for gen in range(4):
    g = np.load(DATA / f"selfplay-mc{gen}-encoded.npz")["game"]
    offset += int(g.max()) + 1
shard = np.load(DATA / "selfplay-mc4-encoded.npz")
own_game = shard["game"]
count = int(own_game.max()) + 1
mine = val_games[(val_games >= offset) & (val_games < offset + count)] - offset
rows_val = int(np.isin(own_game, mine).sum())
print(f"gen-4 held-out games {len(mine):,}, rows {rows_val:,} (offset {offset:,})", flush=True)
# The first N_GAMES held-out games in the files' order: reading the 20 GB of records is the
# cost (about 12 MB/s under the generation run), so the sample stops early. The files are
# batches of the same generation run; SEED is kept for the record but unused here.
picked = np.sort(mine)[: min(N_GAMES, len(mine))]
picked_set = set(picked.tolist())
rows_per_game = np.bincount(own_game, minlength=count)
outcome_of = np.zeros(count)
outcome_of[own_game] = shard["outcome"]
meta = json.loads(str(shard["meta_json"]))
sources = [s[0] for s in meta["sources"]]

reg = load_regulation("gen9championsvgc2026regmc")
PROTECT = {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap",
           "burningbulwark", "obstruct", "wideguard", "quickguard"}
# Weather and terrain: which moves' types and abilities gain, which lose.
WEATHER_GAIN = {
    "raindance": ({"Water"}, {"swiftswim", "raindish", "hydration", "dryskin"}, {"Fire"}),
    "primordialsea": ({"Water"}, {"swiftswim", "raindish", "hydration", "dryskin"}, {"Fire"}),
    "sunnyday": ({"Fire"}, {"chlorophyll", "solarpower", "protosynthesis", "flowergift", "harvest"}, {"Water"}),
    "desolateland": ({"Fire"}, {"chlorophyll", "solarpower", "protosynthesis", "flowergift", "harvest"}, {"Water"}),
    "sandstorm": (set(), {"sandrush", "sandforce", "sandveil"}, set()),
    "snowscape": (set(), {"slushrush", "snowcloak", "icebody"}, set()),
}
TERRAIN_GAIN = {
    "electricterrain": ("Electric", {"surgesurfer", "quarkdrive"}),
    "grassyterrain": ("Grass", {"grasspelt"}),
    "psychicterrain": ("Psychic", set()),
    "mistyterrain": (None, set()),
}


def grounded(mon: dict) -> bool:
    return ("Flying" not in mon["types"] and mon["ability"] != "levitate"
            and mon.get("item") != "airballoon")


def move_types(mon: dict) -> set[str]:
    out = set()
    for slot in mon["moves"]:
        mv = reg.moves.get(slot["id"])
        if mv is not None and mv.category != "Status":
            out.add(mv.type)
    return out


PY_NAMES = ["moves_shown", "species_shown", "item_shown", "pp_zero", "pp_quarter",
            "protect_last", "fakeout_ready", "weather_gain", "terrain_gain",
            "weather_turns", "terrain_turns", "alive"]


def side_features(pos: dict, s: int, shown: list) -> list[float]:
    side = pos["sides"][s]
    mons = side["pokemon"]
    alive = [m for m in mons if not m["fainted"] and m["hp"] > 0]
    actives = [mons[i] for i in side["active"] if i is not None and not mons[i]["fainted"]]
    moves_shown = sum(sum(1 for slot in m["moves"] if slot.get("used")) for m in mons)
    item_shown = sum(1 for m in mons if m.get("isMega") or m.get("item") != m.get("baseItem"))
    pp_zero = sum(1 for m in alive for slot in m["moves"] if slot["pp"] == 0)
    pp_quarter = sum(1 for m in alive for slot in m["moves"] if slot["pp"] * 4 <= slot["maxpp"])
    protect_last = sum(1 for m in actives if m.get("lastMove") in PROTECT)
    fakeout = sum(1 for m in actives if m.get("activeMoveActions", 1) == 0
                  and any(slot["id"] == "fakeout" and slot["pp"] > 0 and not slot["disabled"]
                          for slot in m["moves"]))
    field = pos["field"]
    w = field.get("weather")
    wg = 0
    if w in WEATHER_GAIN:
        gain_types, gain_abil, lose_types = WEATHER_GAIN[w]
        for m in alive:
            t = move_types(m)
            wg += int(bool(t & gain_types) or m["ability"] in gain_abil
                      or (w == "sandstorm" and "Rock" in m["types"])
                      or (w == "snowscape" and "Ice" in m["types"]))
            wg -= int(bool(t & lose_types) and not (t & gain_types))
    tr = field.get("terrain")
    tg = 0
    if tr in TERRAIN_GAIN:
        kind, abil = TERRAIN_GAIN[tr]
        for m in alive:
            t = move_types(m)
            if tr == "mistyterrain":
                tg -= int(grounded(m) and "Dragon" in t)
            elif tr == "psychicterrain":
                tg += int(grounded(m) and "Psychic" in t)
            else:
                tg += int((grounded(m) and kind in t) or m["ability"] in abil)
    wt = float(field.get("weatherDuration") or 0) if w else 0.0
    tt = float(field.get("terrainDuration") or 0) if tr else 0.0
    return [moves_shown, len(shown[s]), item_shown, pp_zero, pp_quarter, protect_last, fakeout,
            wg, tg, wt * (wg > 0) - wt * (wg < 0), tt * (tg > 0) - tt * (tg < 0), len(alive)]


positions = (OUT / "positions.jsonl").open("wb")
game_l, turn_l, kind_l, out_l, feats, fainted_l = [], [], [], [], [], []
gid = 0
checked = 0
for name in sources:
    with (DATA / "selfplay-mc4" / name).open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            here = gid
            gid += 1
            if here not in picked_set:
                continue
            g = json.loads(line)
            if len(g["decisions"]) != rows_per_game[here] or float(g["outcome"]) != float(outcome_of[here]):
                raise SystemExit(f"{name}: game {here} does not match the shard")
            checked += 1
            for d in g["decisions"]:
                pos = d["position"]
                positions.write(json.dumps(pos, separators=(",", ":")).encode("utf-8") + b"\n")
                game_l.append(here)
                turn_l.append(d["turn"])
                kind_l.append(0 if d["kind"] == "move" else 1)
                out_l.append(float(g["outcome"]))
                shown = d.get("shownIdentities") or [[], []]
                feats.append([side_features(pos, s, shown) for s in (0, 1)])
    if checked == len(picked):
        break
positions.close()
assert checked == len(picked), (checked, len(picked))
np.savez(OUT / "sample.npz", game=np.array(game_l), turn=np.array(turn_l), kind=np.array(kind_l),
         outcome=np.array(out_l), py=np.array(feats, dtype=np.float32), py_names=np.array(PY_NAMES),
         picked=picked, heldout_games=len(mine), heldout_rows=rows_val)
print(f"{checked:,} games, {len(game_l):,} positions, {time.time() - started:.0f}s", flush=True)
