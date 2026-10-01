"""IKA-411: what Showdown's position snapshot (position.ts) writes as baseSpecies, before and
after a Mega Evolution, for the two formes whose dex baseSpecies is not the set species.

Uses the main checkout's built sim-bridge (read only).
"""

from pathlib import Path

from pokeuraou.oracle import Oracle, TeamSet

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
JS = M / "packages/sim-bridge/dist/cli/oracle.js"
FORMAT = "gen9championsvgc2026regmc"
FILL = ["protect", "helpinghand", "tackle", "growl"]


def mon(species, ability, item=None):
    return TeamSet(species=species, ability=ability, nature="Serious", moves=FILL, item=item,
                   sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 10})


with Oracle(script=JS) as o:
    for lead, ability, stone in (("Floette-Eternal", "Flower Veil", "Floettite"),
                                 ("Meowstic-F", "Competitive", "Meowsticite"),
                                 ("Charizard", "Blaze", "Charizardite Y")):
        mine = [mon("Incineroar", "Blaze"), mon("Kangaskhan", "Inner Focus")]
        theirs = [mon(lead, ability, stone), mon("Garchomp", "Rough Skin"), mon("Milotic", "Marvel Scale")]
        b = o.create(FORMAT, mine, theirs)
        p = b.position["sides"][1]["pokemon"][0]
        print(f"{lead:16s} turn1: species={p['species']:16s} baseSpecies={p['baseSpecies']}")
        b.step(["team 12", "team 123"])
        p = b.position["sides"][1]["pokemon"][0]
        print(f"{lead:16s} turn1: species={p['species']:16s} baseSpecies={p['baseSpecies']}")
        b.step(["move 1, move 1", "move 1 mega, move 1"]); print("  errors", b.choice_errors)
        p = b.position["sides"][1]["pokemon"][0]
        print(f"{lead:16s} after: species={p['species']:16s} baseSpecies={p['baseSpecies']} isMega={p.get('isMega')}")
