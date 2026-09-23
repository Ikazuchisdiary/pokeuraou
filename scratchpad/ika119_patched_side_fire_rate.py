"""IKA-119: on recorded positions, how often does fixing `_patched` move `belief_solve`?

Replays recorded move decisions of a hidden-bench pool through `belief_solve` twice -- once
with the `_patched` that shipped before IKA-119 (side vector copied from the true position,
bench `can_mega` copied from the completion's root), once with the fixed one -- on the same
menus, the same completions and the same leaf, and prints how far each side's value and
strategy moved, and how often the move actually played would change.

    PYTHONPATH=<tree>/src POKEURAOU_RUST_NODE=1 python scratchpad/ika119_patched_side_fire_rate.py \\
        <pool dir> <model.pt> --positions 200 --games 400 --workers 8

What it holds fixed and where it departs from generation (declared before the run):

* Menus are the recorded ones (`ownActions` / `foeActions`), matched to legal actions by
  choice string. The leaf ranking that built them reads `items[0].position` whole, not a
  patched encoding, so the fix cannot move a menu; holding them fixed isolates the matrix.
* Sheets: side 0 is the roster whose six matches `ownSix`; side 1 is the recorded four plus
  the two unbrought members of the standings team with the same six, their items and moves
  from the standings and their SP drawn from usage (seed 0) -- generation drew them with
  its own rng, which is not recorded. SP moves max HP, so it moves `team_hp_fraction`.
* Completion weights are uniform. Generation weighted by the selection book's bench prior
  when the book named the opponent; the book entry is not replayed here.
* Seen slots are tracked by species across the game's recorded decisions (the identity
  rule), not by carried slot numbers (`hidden.seen_slots`, IKA-117).
* The leaf is one model on CPU, one torch thread per worker.

"Played move changes" is the probability, under the generator's own inverse-CDF draw with
one shared uniform, that the index drawn from the new strategy differs from the one drawn
from the old: 1 - sum_i |[C_old(i-1), C_old(i)] n [C_new(i-1), C_new(i)]|.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

ARGS: argparse.Namespace | None = None
STATE: dict = {}


def _patched_before_ika119(reference, side, slots, item, reg, position):  # noqa: ANN001, ANN202
    """`beliefnode._patched` as it was before IKA-119 (commit be3b896 + IKA-121), verbatim."""
    from pokeuraou.beliefnode import _encoder_for
    from pokeuraou.encode import Encoded

    encoder = _encoder_for(reg)
    source = encoder.encode_positions([item.position])
    out = Encoded(
        species=reference.species.copy(),
        ability=reference.ability.copy(),
        item=reference.item.copy(),
        moves=reference.moves.copy(),
        mon=reference.mon.copy(),
        mask=reference.mask.copy(),
        side=reference.side,
        field=reference.field,
        unknown_volatiles=dict(reference.unknown_volatiles),
    )
    for slot in slots:
        out.species[:, side, slot] = source.species[0, side, slot]
        out.ability[:, side, slot] = source.ability[0, side, slot]
        out.item[:, side, slot] = source.item[0, side, slot]
        out.moves[:, side, slot] = source.moves[0, side, slot]
        out.mon[:, side, slot] = source.mon[0, side, slot]
        out.mask[:, side, slot] = source.mask[0, side, slot]
    return out


def _revealed(mon) -> bool:  # noqa: ANN001
    return bool(
        mon.active_index is not None or mon.fainted or mon.hp != mon.maxhp
        or mon.status is not None or any(mon.boosts.values()) or mon.volatiles
        or mon.is_mega
    )


def collect(pool: Path, games: int, want: int, seed: int) -> tuple[list[dict], dict]:
    from pokeuraou.position import Position

    candidates: list[dict] = []
    counts = {"games": 0, "moves": 0, "hidden_moves": 0}
    for path in sorted(pool.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if counts["games"] >= games:
                    break
                game = json.loads(line)
                if game.get("information") != "hidden-bench":
                    continue
                counts["games"] += 1
                seen: list[set[str]] = [set(), set()]
                for d in game["decisions"]:
                    pos = Position.from_json(d["position"])
                    for s in (0, 1):
                        seen[s] |= {m.species for m in pos.sides[s].pokemon if _revealed(m)}
                    if d["kind"] != "move":
                        continue
                    counts["moves"] += 1
                    shown = [
                        sorted(m.slot for m in pos.sides[s].pokemon if m.species in seen[s])
                        for s in (0, 1)
                    ]
                    if all(len(shown[s]) >= len(pos.sides[s].pokemon) for s in (0, 1)):
                        continue
                    counts["hidden_moves"] += 1
                    candidates.append({
                        "position": d["position"],
                        "own": d["ownActions"],
                        "foe": d["foeActions"],
                        "shown": shown,
                        "ownSix": game["ownSix"],
                        "foeSix": game["foeSix"],
                        "foeTeam": game["foeTeam"],
                        "ownTeam": game["ownTeam"],
                    })
        if counts["games"] >= games:
            break
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(candidates), size=min(want, len(candidates)), replace=False))
    return [candidates[i] for i in pick], counts


def _init(model: str, null: bool) -> None:
    import torch

    STATE["null"] = null

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


def sheets_for(case: dict):  # noqa: ANN202
    from pokeuraou.regulation import to_id
    from pokeuraou.standings import sample_standings_team

    own = None
    for roster in STATE["rosters"]:
        if sorted(to_id(s.species) for s in roster.sets) == sorted(case["ownSix"]):
            own = list(roster.sets)
    if own is None:
        return None
    six = sorted(case["foeSix"])
    team = next(
        (t for t in STATE["standings"].teams if sorted(to_id(s) for s in t.species) == six),
        None,
    )
    if team is None:
        return None
    drawn = sample_standings_team(np.random.default_rng(0), STATE["reg"], STATE["prior"], team)
    brought = {entry["species"]: _set_from_json(entry) for entry in case["foeTeam"]}
    foe = [brought.get(to_id(s.species), s) for s in drawn]
    return own, foe


def _crn_change(old: np.ndarray, new: np.ndarray) -> float:
    a = np.concatenate([[0.0], np.cumsum(old / old.sum())])
    b = np.concatenate([[0.0], np.cumsum(new / new.sum())])
    overlap = np.clip(np.minimum(a[1:], b[1:]) - np.maximum(a[:-1], b[:-1]), 0.0, None)
    return float(max(0.0, 1.0 - overlap.sum()))


def run_one(case: dict) -> dict | None:
    from pokeuraou import beliefnode
    from pokeuraou.actions import side_actions
    from pokeuraou.equilibrium import EquilibriumError
    from pokeuraou.hidden import completions
    from pokeuraou.position import Position
    from pokeuraou.resolve import Budget
    from pokeuraou.search import belief_solve

    reg, leaf = STATE["reg"], STATE["leaf"]
    sheets = sheets_for(case)
    if sheets is None:
        return {"skipped": "sheet"}
    pos = Position.from_json(case["position"])
    menus = []
    for s, recorded in ((0, case["own"]), (1, case["foe"])):
        legal = {a.to_choice(): a for a in side_actions(reg, pos, s)}
        menu = [legal.get(c) for c in recorded]
        if not menu or any(a is None for a in menu):
            return {"skipped": "menu"}
        menus.append(menu)
    try:
        spreads = {
            s: completions(reg, pos, s, sheets[s], seen=frozenset(case["shown"][s]))
            for s in (0, 1)
        }
    except ValueError:
        return {"skipped": "completions"}
    answers = {}
    fixed = beliefnode._patched
    calls = {"old": 0, "new": 0}

    def counted(label, patch):  # noqa: ANN001, ANN202
        def call(*a, **k):  # noqa: ANN002, ANN003, ANN202
            calls[label] += 1
            return patch(*a, **k)
        return call

    before = fixed if STATE.get("null") else _patched_before_ika119
    started = time.perf_counter()
    try:
        for label, patch in (("old", before), ("new", fixed)):
            beliefnode._patched = counted(label, patch)
            answers[label] = belief_solve(
                reg, pos, menus[0], menus[1], spreads, {0: leaf, 1: leaf},
                budget=Budget.matrix(),
            )
    except EquilibriumError:
        return {"skipped": "equilibrium"}
    finally:
        beliefnode._patched = fixed
    out = {
        "seconds": time.perf_counter() - started,
        "classes": [len(spreads[s]) for s in (0, 1)],
        "hidden": [len(pos.sides[s].pokemon) - len(case["shown"][s]) for s in (0, 1)],
        "turn": pos.turn,
        # Zero means the shared path never ran and old and new are the same code.
        "patch_calls": calls["new"],
    }
    for s in (0, 1):
        old, new = answers["old"][s], answers["new"][s]
        out[f"dv{s}"] = new.value - old.value
        out[f"tv{s}"] = 0.5 * float(np.abs(new.strategy - old.strategy).sum())
        out[f"argmax{s}"] = int(np.argmax(new.strategy) != np.argmax(old.strategy))
        out[f"crn{s}"] = _crn_change(old.strategy, new.strategy)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=119)
    ap.add_argument("--out", type=Path, default=None, help="per-position results as JSON")
    ap.add_argument(
        "--null", action="store_true",
        help="both solves use the fixed patch: the null control, every number must be 0",
    )
    args = ap.parse_args()
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    started = time.time()
    cases, counts = collect(args.pool, args.games, args.positions, args.seed)
    print(f"{counts['games']} games, {counts['moves']} move decisions, "
          f"{counts['hidden_moves']} with a hidden slot on either side "
          f"({counts['hidden_moves'] / max(counts['moves'], 1):.1%}); sampled {len(cases)}",
          flush=True)
    with Pool(args.workers, initializer=_init, initargs=(str(args.model), args.null)) as pool:
        results = pool.map(run_one, cases, chunksize=1)
    done = [r for r in results if r and "skipped" not in r]
    skipped: dict[str, int] = {}
    for r in results:
        if r and "skipped" in r:
            skipped[r["skipped"]] = skipped.get(r["skipped"], 0) + 1
    print(f"solved {len(done)}, skipped {skipped}, wall {time.time() - started:.0f}s, "
          f"node seconds (both solves) mean {np.mean([r['seconds'] for r in done]):.2f}; "
          f"positions where the shared path ran: "
          f"{sum(r['patch_calls'] > 0 for r in done)} of {len(done)}"
          + ("  [NULL CONTROL]" if args.null else ""))
    for s in (0, 1):
        dv = np.abs([r[f"dv{s}"] for r in done])
        tv = np.array([r[f"tv{s}"] for r in done])
        crn = np.array([r[f"crn{s}"] for r in done])
        am = np.array([r[f"argmax{s}"] for r in done])
        moved = (dv > 1e-9) | (tv > 1e-9)
        print(f"side {s}: value or strategy moved {moved.mean():.1%}  "
              f"|dv| mean {dv.mean():.4f} median {np.median(dv):.4f} "
              f"p90 {np.quantile(dv, 0.9):.4f} max {dv.max():.4f}  "
              f">0.01 {(dv > 0.01).mean():.1%}")
        print(f"        strategy TV mean {tv.mean():.4f} p90 {np.quantile(tv, 0.9):.4f} "
              f"max {tv.max():.4f}; argmax changes {am.mean():.1%}; "
              f"played move changes (shared draw) mean {crn.mean():.1%}, "
              f"in {(crn > 1e-9).mean():.1%} of positions P>0")
    either = np.array([max(r["crn0"], r["crn1"]) for r in done])
    print(f"either side's played move changes (max of the two), mean {either.mean():.1%}")
    if args.out is not None:
        payload = {"counts": counts, "results": results}
        args.out.write_bytes(json.dumps(payload).encode("utf-8"))


if __name__ == "__main__":
    main()
