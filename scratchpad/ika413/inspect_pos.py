import json, sys
sys.path.insert(0, "C:/tmp/ika413/wt/src")
from pokeuraou.position import Position
rows=[json.loads(l) for l in open("impact30k-real.jsonl")]
H={"friendguard","cloudnine","airlock","fairyaura","darkaura","aurabreak"}
no=[r for r in rows if not (set(r["abilities"])&H)]
idx=int(sys.argv[1])
r=no[idx]
pos=Position.from_json(r["position"])
print(r["cells"], r["worst"])
print("weather", pos.field.weather, pos.field.terrain, [e.id for e in pos.field.pseudo_weather])
for si, side in enumerate(pos.sides):
    print("side", si, [c.id for c in side.side_conditions])
    for m in side.active_pokemon():
        print("  ", m.species, m.ability, m.item, m.hp, m.maxhp, m.moves and [x.id for x in m.moves], m.status, [v.id for v in m.volatiles], m.boosts)
print("bench side1:", [(m.species, m.ability, m.hp) for m in pos.sides[1].pokemon])
