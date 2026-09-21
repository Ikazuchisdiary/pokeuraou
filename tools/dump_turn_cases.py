"""Every `resolve_turn` call a generation run makes, with the branches Python returned.

This is the fixture the Rust resolver is held to. Generation spends 96.7% of its wall
clock inside `resolve_turn` (measured with counters, not cProfile -- cProfile misattributes
16% of the run to scipy's option validation, which a standalone benchmark puts at 4.5 ms a
solve), so this call is the whole target, and the only way to port it is to be able to
check it turn by turn.

Positions are stored once and referenced by index: a 24x24 node calls `resolve_turn` 576
times against the same root, so storing them per case would multiply the file by that.

The budget is *recorded*, not assumed. A generated game calls the resolver with two
different ones -- `Budget.matrix()` to fill the search's matrix and `Budget.exact()` to
advance the game -- and writing "matrix" for both made the exact calls look like matrix
calls that branched too much. Both of the two turns the Rust port appeared to get wrong
were this, and neither was a port bug.

    uv run python tools/dump_turn_cases.py --games 2 --cases 4000 --out rust/turns.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import resolve as resolve_mod  # noqa: E402
from pokeuraou.actions import MoveAction, PassAction, SwitchAction  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


def dump_action(action) -> dict:  # noqa: ANN001
    if isinstance(action, MoveAction):
        return {
            "kind": "move",
            "slot": action.slot,
            "moveIndex": action.move_index,
            "moveId": action.move_id,
            "target": action.target,
            "mega": action.mega,
        }
    if isinstance(action, SwitchAction):
        return {
            "kind": "switch",
            "slot": action.slot,
            "partyIndex": action.party_index,
            "species": action.species,
        }
    if isinstance(action, PassAction):
        return {"kind": "pass", "slot": action.slot}
    raise TypeError(f"unknown action {action!r}")


def dump_side_action(side_action) -> list[dict]:  # noqa: ANN001
    return [dump_action(a) for a in side_action.slots]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--cases", type=int, default=4000, help="keep at most this many")
    ap.add_argument(
        "--stride", type=int, default=7, help="keep one call in this many, to spread the sample"
    )
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--only-move",
        default=None,
        help="comma-separated move ids; keep only calls where one of them is a chosen "
        "action, and keep every such call. For holding the port to moves it has just "
        "learned, which a spread sample may contain none of.",
    )
    ap.add_argument(
        "--only-ability",
        default=None,
        help="comma-separated ability ids; keep only calls where one of them is on the "
        "field. Coverage is a property of the sample -- an ability no sampled team carries "
        "is an ability the differential says nothing about.",
    )
    ap.add_argument(
        "--value",
        default=None,
        help="play the games with a trained value function (data/models/*.pt). The games a "
        "generation run actually plays are these, and they reach different positions.",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="rust/turns.json")
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    if args.value:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        device = torch.device(args.device)
        encoder = Encoder(reg)
        net, _meta = load_model(Path(args.value), encoder)
        objective = BatchedValue(net.to(device), encoder, device=device).objective("win")
    else:
        objective = OBJECTIVES["hp-share"]

    wanted_moves = {m for m in (args.only_move or "").split(",") if m}
    wanted_abilities = {a for a in (args.only_ability or "").split(",") if a}

    positions: list[dict] = []
    position_index: dict[str, int] = {}
    cases: list[dict] = []
    seen_cases = 0
    real = resolve_mod.resolve_turn

    def intern(position) -> int:  # noqa: ANN001
        text = json.dumps(position.to_json(), sort_keys=True)
        hit = position_index.get(text)
        if hit is None:
            hit = len(positions)
            position_index[text] = hit
            positions.append(json.loads(text))
        return hit

    def recording(reg_, pos, actions, *, budget):  # noqa: ANN001
        nonlocal seen_cases
        result = real(reg_, pos, actions, budget=budget)
        seen_cases += 1
        wanted = True
        if wanted_moves:
            wanted = any(
                getattr(a, "move_id", None) in wanted_moves
                for side in actions
                for a in side.slots
            )
        if wanted and wanted_abilities:
            wanted = any(
                mon.ability in wanted_abilities
                for side in pos.sides
                for mon in side.pokemon
                if not mon.fainted
            )
        # Keep a spread across the run rather than the first N, which would all come from
        # turn 1 of game 1 and hide everything the later turns reach. A move filter keeps
        # every match instead: there are few of them and the point is to have any at all.
        stride = 1 if (wanted_moves or wanted_abilities) else max(1, args.stride)
        if wanted and len(cases) < args.cases and seen_cases % stride == 0:
            cases.append(
                {
                    "position": intern(pos),
                    "ours": dump_side_action(actions[0]),
                    "theirs": dump_side_action(actions[1]),
                    "budget": {
                        "damageRolls": budget.damage_rolls,
                        "enumerateCrit": budget.enumerate_crit,
                        "enumerateAccuracy": budget.enumerate_accuracy,
                        "enumerateStatusChecks": budget.enumerate_status_checks,
                        "enumerateSecondary": budget.enumerate_secondary,
                        "enumerateSpeedTies": budget.enumerate_speed_ties,
                        "pinnedPolicy": budget.pinned_policy,
                        "maxBranches": budget.max_branches,
                        "mergeDuplicates": budget.merge_duplicates,
                    },
                    "expect": {
                        "suspended": bool(result.suspended),
                        "exact": bool(result.exact),
                        "unmodelled": sorted(result.unmodelled),
                        "branches": [
                            {
                                "probability": float(branch.probability),
                                "position": intern(branch.position),
                            }
                            for branch in result.branches
                        ],
                    },
                }
            )
        return result

    resolve_mod.resolve_turn = recording
    import pokeuraou.search as search_mod
    import pokeuraou.selfplay as selfplay_mod

    search_mod.resolve_turn = recording
    selfplay_mod.resolve_turn = recording

    rng = np.random.default_rng(args.seed)
    for _ in range(args.games):
        team = pool[int(rng.integers(len(pool)))]
        foe_six = sample_standings_team(rng, reg, prior, team)
        own_pick = selections[int(rng.integers(len(selections)))]
        foe_pick = selections[int(rng.integers(len(selections)))]
        play_game(
            reg,
            rng,
            [roster.sets[i] for i in own_pick],
            [foe_six[j] for j in foe_pick],
            "turn-dump",
            objective=objective,
            search_limit=args.limit,
            max_turns=args.max_turns,
        )

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {"format_id": reg.meta.format_id, "positions": positions, "cases": cases},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    branches = sum(len(c["expect"]["branches"]) for c in cases)
    suspended = sum(1 for c in cases if c["expect"]["suspended"])
    print(f"{len(cases)} turns of {seen_cases} seen -> {out}")
    print(f"  distinct positions {len(positions)}")
    print(f"  branches           {branches} ({branches / max(len(cases), 1):.1f} per turn)")
    print(f"  suspended          {suspended}")


if __name__ == "__main__":
    main()
