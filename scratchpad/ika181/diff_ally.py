"""IKA-181: the port against Showdown on turns where a `normal` move names the ally.

`tools/diff_turn.py`'s whole-turn comparison, with:
- the teams: four of a pool team, led by a (user, ally) pair `team_ally_benefits` names
  (`--teams benefit`), or by any two members (`--teams all`);
- the menus: `actions` in mode ``all`` (every ally target Showdown accepts) or ``benefit``;
- the chooser: with probability BIAS an action that names the ally, else uniform.
`--control` hands Showdown the same choice with the ally target replaced by foe 1, so the
port and Showdown resolve different turns on exactly those slots: the comparison must
then report divergences (the control that it can fail).

    python diff_ally.py <pool.json> <battles> <seed> <mode> <bias> [--control] [--teams all]
"""
import random
import sys
from dataclasses import replace
from pathlib import Path

W = Path(__file__).resolve().parent
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))

import diff_turn as dt  # noqa: E402

from pokeuraou import actions  # noqa: E402
from pokeuraou.actions import MoveAction, SideAction, names_ally, side_actions  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402

pool_path, battles, seed, mode, bias = sys.argv[1:6]
battles, seed, bias = int(battles), int(seed), float(bias)
control = "--control" in sys.argv
any_lead = "--teams" in sys.argv and sys.argv[sys.argv.index("--teams") + 1] == "all"
actions.set_ally_targets(mode)
pool = load_pool(pool_path)
reg_pool = pool.reg

leads = []
for t, team in enumerate(pool.teams):
    if any_lead:
        leads += [(t, u, a) for u in range(6) for a in range(6) if u != a]
    else:
        leads += [(t, u, a) for u, a, _m in actions.team_ally_benefits(reg_pool, team.sets)]
print(f"leads to draw from: {len(leads)}", flush=True)

chaos = sorted((Path("C:/Users/Ikazuchi/repos/pokeuraou/data/priors/raw")).glob("*-1760.json*"))[-1]
dt.find_cached_chaos = lambda fmt, cutoff=1760: chaos


def sample_team(rng, reg, prior, **kw):  # noqa: ARG001
    t, u, a = leads[int(rng.integers(len(leads)))]
    rest = [i for i in range(6) if i not in (u, a)]
    rng.shuffle(rest)
    sets = pool.teams[t].sets
    return [sets[u], sets[a], sets[rest[0]], sets[rest[1]]]


dt.sample_team = sample_team
COUNT = {"picks": 0, "ally_picks": 0, "ally_offered": 0}


def pick_action(reg, pos, side_index, py_rng, wanted, b):  # noqa: ARG001
    options = side_actions(reg, pos, side_index)
    ally = [o for o in options if names_ally(reg, o)]
    COUNT["picks"] += 1
    COUNT["ally_offered"] += int(bool(ally))
    if ally and py_rng.random() < bias:
        COUNT["ally_picks"] += 1
        return py_rng.choice(ally)
    return py_rng.choice(options)


dt.pick_action = pick_action

if control:
    original = dt.showdown_choice

    def showdown_choice(pick: SideAction, request):
        slots = tuple(
            replace(s, target=1)
            if isinstance(s, MoveAction) and s.target is not None and s.target < 0
            and reg_pool.moves[s.move_id].target == "normal"
            else s
            for s in pick.slots
        )
        return original(SideAction(slots=slots), request)

    dt.showdown_choice = showdown_choice

#: Compared turns and divergent ones, split by whether either side named its ally.
SPLIT = {True: [0, 0, 0], False: [0, 0, 0]}  # compared, divergent, silent-divergent
EXAMPLES = []
original_compare = dt.compare_turn


def compare_turn(reg, before, chosen, handle, log, roll, report, py_rng, nodes):
    column = next(iter(report.ports.values()))
    was = len(column.records)
    was_silent = getattr(column, "silent", None)
    original_compare(reg, before, chosen, handle, log, roll, report, py_rng, nodes)
    ally = any(names_ally(reg, c) for c in chosen)
    got = len(column.records) - was
    SPLIT[ally][0] += 1
    SPLIT[ally][1] += int(got > 0)
    if got > 0:
        record = column.records[-1]
        SPLIT[ally][2] += int(not record["unmodelled"])
        EXAMPLES.append({**record, "allyTurn": ally})
    del was_silent


dt.compare_turn = compare_turn

random.seed(seed)
report = dt.run(battles, 8, seed, 10, quiet=False)
print("COUNT", COUNT, flush=True)
print("SPLIT (ally turn -> compared, divergent, silent)", SPLIT, flush=True)
import json  # noqa: E402

out = Path(sys.argv[-1]) if sys.argv[-1].endswith(".json") else None
if out is not None:
    out.write_bytes(json.dumps(EXAMPLES, indent=1, default=str).encode("utf-8"))
