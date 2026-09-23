"""IKA-120: on recorded games, how often does the belief move the self-switch choice?

Replays the `selfswitch` decisions of a hidden-bench pool through `_do_self_switch_node`
twice -- once as it shipped (the true pause, `hidden=None`) and once with the belief over
the opponent's unseen slots (`hidden=_HiddenBench(...)`) -- on the same pause and the same
leaf, and counts how often the chosen switch-in changes.

    PYTHONPATH=<tree>/src python scratchpad/ika120_selfswitch_fire_rate.py \\
        <pool dir> <model.pt> --positions 300

The pause is rebuilt, not read: the record holds the position at the pause but not its
continuation. The move decision the turn started from is resolved again with its recorded
choices (`Budget.exact()`, as `_advance_turn` does), and the suspension whose position
equals the recorded one is taken -- the way `tools/show_game.py` finds it. A turn that
paused twice is walked through its earlier pause with the recorded answer.

Controls, declared before the run:

* Reconstruction: the shipped path on the rebuilt pause has to give the recorded choice.
  Its agreement rate is printed; a disagreement is a replay that is not the game's (the
  record was played on CUDA at an older commit, this is CPU at this one).
* Null: the shipped path run twice on the same pause (`--null` replaces the belief run
  with a second shipped run). Every change count must be 0.
* Where the opponent has nothing unseen at the pause the belief path is the shipped path
  by construction; the change count there is printed separately and must be 0.

Where it departs from generation:

* Sheets: side 0 is the roster whose six matches `ownSix`; side 1 is the recorded four plus
  the two unbrought members of the standings team with the same six, items and moves from
  the standings and SP drawn from usage (seed 0), as in IKA-119's replay.
* Completion weights are uniform. Generation weighted by the selection book's bench prior
  when the book named the opponent; the book entry is not replayed here.
* Seen identities are accumulated over the recorded move and replacement positions up to
  the move decision the turn started from, exactly what `play_game` carries into
  `_advance_turn`; the leads are read off the first decision.
* The leaf is one model on CPU, one torch thread.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

STATE: dict = {}


def _init(model: str) -> None:
    import torch

    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.priors import find_cached_chaos, load_chaos
    from pokeuraou.regulation import load_regulation
    from pokeuraou.standings import find_cached_standings, load_standings
    from pokeuraou.teams import load_roster
    from pokeuraou.value import BatchedValue, load_model

    torch.set_num_threads(1)
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    encoder = Encoder(reg)
    net, _meta = load_model(model, encoder)
    STATE["reg"] = reg
    STATE["leaf"] = BatchedValue(net, encoder, device=torch.device("cpu"))
    STATE["rosters"] = [load_roster(name) for name in ("rizabanadohido", "place1")]
    STATE["prior"] = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    STATE["standings"] = load_standings(find_cached_standings("2026", "worlds"), reg)


def _set_from_json(entry: dict):  # noqa: ANN202
    from pokeuraou.priors import SampledSet

    return SampledSet(
        species=entry["species"], ability=entry["ability"], item=entry.get("item") or None,
        nature=entry["nature"], sp={k: int(v) for k, v in (entry.get("sp") or {}).items()},
        moves=list(entry["moves"]),
    )


def sheets_for(game: dict):  # noqa: ANN202
    from pokeuraou.regulation import to_id
    from pokeuraou.standings import sample_standings_team

    own = None
    for roster in STATE["rosters"]:
        if sorted(to_id(s.species) for s in roster.sets) == sorted(game["ownSix"]):
            own = list(roster.sets)
    if own is None:
        return None
    six = sorted(game["foeSix"])
    team = next(
        (t for t in STATE["standings"].teams if sorted(to_id(s) for s in t.species) == six),
        None,
    )
    if team is None:
        return None
    drawn = sample_standings_team(np.random.default_rng(0), STATE["reg"], STATE["prior"], team)
    brought = {entry["species"]: _set_from_json(entry) for entry in game["foeTeam"]}
    foe = [brought.get(to_id(s.species), s) for s in drawn]
    return own, foe


def _key(position) -> str:  # noqa: ANN001
    return json.dumps(position.to_json() if hasattr(position, "to_json") else position,
                      sort_keys=True)


def collect(pool: Path, games: int | None) -> tuple[list[dict], dict]:
    """Every selfswitch decision with what replaying it needs, plus the pool's counts."""
    from pokeuraou.hidden import seen_identities, seen_slots, shown_species
    from pokeuraou.position import Position

    cases: list[dict] = []
    counts = {"games": 0, "decisions": 0, "selfswitch": 0, "opponent_unseen": 0,
              "games_with_one": 0}
    for path in sorted(pool.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if games is not None and counts["games"] >= games:
                    break
                game = json.loads(line)
                if game.get("information") != "hidden-bench":
                    continue
                counts["games"] += 1
                decisions = game["decisions"]
                counts["decisions"] += len(decisions)
                seen: list[frozenset[str]] = [frozenset(), frozenset()]
                carried_at: dict[int, list[frozenset[str]]] = {}
                start = Position.from_json(decisions[0]["position"])
                # As `play_game` reads them (unused while the weights are uniform).
                leads = [
                    frozenset(shown_species(start, s, frozenset(
                        m.slot for m in start.sides[s].pokemon if m.active_index is not None
                    )))
                    if start.turn == 1 else None
                    for s in (0, 1)
                ]
                last_move = None
                had = False
                for index, d in enumerate(decisions):
                    pos = Position.from_json(d["position"])
                    if d["kind"] in ("move", "replacement"):
                        seen = [seen_identities(pos, s, seen[s]) for s in (0, 1)]
                    if d["kind"] == "move":
                        last_move = index
                        carried_at[index] = list(seen)
                        continue
                    if d["kind"] != "selfswitch":
                        continue
                    counts["selfswitch"] += 1
                    chooser = 0 if d["ownActions"] != ["pass"] else 1
                    other = 1 - chooser
                    anchor = carried_at.get(last_move) if last_move is not None else None
                    carried = seen_identities(pos, other, anchor[other] if anchor else seen[other])
                    unseen = len(pos.sides[other].pokemon) - len(seen_slots(pos, other, carried))
                    if unseen > 0:
                        counts["opponent_unseen"] += 1
                        had = True
                    if last_move is None:
                        continue
                    cases.append({
                        "file": path.name,
                        "gameIndex": game.get("gameIndex"),
                        "game": {k: game[k] for k in ("ownSix", "foeSix", "foeTeam")},
                        "move": decisions[last_move],
                        "between": [
                            decisions[k] for k in range(last_move + 1, index)
                            if decisions[k]["kind"] == "selfswitch"
                        ],
                        "decision": d,
                        "chooser": chooser,
                        "unseen": unseen,
                        "seen": [sorted(s) for s in anchor],
                        "leads": [sorted(x) if x is not None else None for x in leads],
                    })
                counts["games_with_one"] += int(had)
        if games is not None and counts["games"] >= games:
            break
    return cases, counts


def _pause_for(reg, case: dict):  # noqa: ANN001, ANN202
    from pokeuraou.actions import side_actions
    from pokeuraou.position import Position
    from pokeuraou.resolve import Budget, resolve_turn, resume_alternatives

    move = case["move"]
    pos = Position.from_json(move["position"])
    chosen = []
    for side, key in ((0, "ownChosen"), (1, "foeChosen")):
        legal = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        if move[key] not in legal:
            return None, "move not legal now"
        chosen.append(legal[move[key]])
    result = resolve_turn(reg, pos, chosen, budget=Budget.exact())
    for step in [*case["between"], case["decision"]]:
        wanted = _key(step["position"])
        pause = next((p for p in result.suspended if _key(p.position) == wanted), None)
        if pause is None:
            return None, "recorded pause not among the suspensions"
        if step is case["decision"]:
            return pause, None
        side, alternatives = resume_alternatives(reg, pause)
        answer = step["foeChosen" if side == 1 else "ownChosen"]
        result = next((r for a, r in alternatives if a.to_choice() == answer), None)
        if result is None:
            return None, "earlier pause's answer not among its options"
    return None, "unreachable"


def run_one(case: dict, null: bool) -> dict:
    from pokeuraou.selfplay import GameRecord, _do_self_switch_node, _HiddenBench

    reg, leaf = STATE["reg"], STATE["leaf"]
    sheets = sheets_for(case["game"])
    if sheets is None:
        return {"skipped": "sheet"}
    pause, why = _pause_for(reg, case)
    if pause is None:
        return {"skipped": why}
    hidden = _HiddenBench(
        sheets=sheets,
        seen=[frozenset(s) for s in case["seen"]],
        bench_prior=None,
        leads=[frozenset(x) if x is not None else None for x in case["leads"]],
    )
    key = "foeChosen" if case["chooser"] == 1 else "ownChosen"
    picks = []
    seconds = []
    for belief in (False, not null):
        record = GameRecord(own_team=[], foe_team=[], foe_archetype="replay")
        started = time.perf_counter()
        try:
            _do_self_switch_node(
                reg, pause, record, (leaf, leaf), None, hidden=hidden if belief else None
            )
        except ValueError as problem:
            return {"skipped": f"completions: {problem}"[:60]}
        seconds.append(time.perf_counter() - started)
        made = record.decisions[-1]
        picks.append({
            "chosen": made.own_chosen if case["chooser"] == 0 else made.foe_chosen,
            "value": made.search_value,
            "options": len(made.own_actions if case["chooser"] == 0 else made.foe_actions),
        })
    return {
        "unseen": case["unseen"],
        "recorded": case["decision"][key],
        "old": picks[0],
        "new": picks[1],
        "seconds": seconds,
        "turn": case["decision"]["turn"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--seed", type=int, default=120)
    ap.add_argument("--null", action="store_true",
                    help="both runs take the shipped path: every change count must be 0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    started = time.time()
    cases, counts = collect(args.pool, args.games)
    c = counts
    print(f"{c['games']} hidden-bench games, {c['decisions']} decisions; selfswitch "
          f"{c['selfswitch']} ({c['selfswitch'] / max(c['decisions'], 1):.1%}); the "
          f"opponent had an unseen slot at {c['opponent_unseen']} "
          f"({c['opponent_unseen'] / max(c['selfswitch'], 1):.1%}), in "
          f"{c['games_with_one']} games ({c['games_with_one'] / max(c['games'], 1):.1%})",
          flush=True)
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(cases), size=min(args.positions, len(cases)), replace=False))
    sample = [cases[i] for i in pick]
    _init(str(args.model))
    results = [run_one(case, args.null) for case in sample]
    done = [r for r in results if "skipped" not in r]
    skipped: dict[str, int] = {}
    for r in results:
        if "skipped" in r:
            skipped[r["skipped"]] = skipped.get(r["skipped"], 0) + 1
    print(f"sampled {len(sample)}, replayed {len(done)}, skipped {skipped}, "
          f"wall {time.time() - started:.0f}s" + ("  [NULL CONTROL]" if args.null else ""))
    agree = np.mean([r["old"]["chosen"] == r["recorded"] for r in done])
    print(f"reconstruction: shipped path gives the recorded choice in {agree:.1%}")
    for label, rows in (
        ("opponent had an unseen slot", [r for r in done if r["unseen"] > 0]),
        ("opponent had nothing unseen", [r for r in done if r["unseen"] == 0]),
        ("all", done),
    ):
        if not rows:
            continue
        changed = np.array([r["old"]["chosen"] != r["new"]["chosen"] for r in rows])
        dv = np.abs([r["new"]["value"] - r["old"]["value"] for r in rows])
        choice = np.array([r["old"]["options"] >= 2 for r in rows])
        print(f"  {label}: n={len(rows)} (>=2 options {choice.sum()}), switch-in changed "
              f"{changed.sum()} ({changed.mean():.1%}); |dvalue| mean {dv.mean():.4f} "
              f"max {dv.max():.4f}; seconds old {np.mean([r['seconds'][0] for r in rows]):.2f}"
              f" new {np.mean([r['seconds'][1] for r in rows]):.2f}")
    if args.out is not None:
        args.out.write_bytes(json.dumps({"counts": counts, "results": results}).encode())


if __name__ == "__main__":
    main()
