"""IKA-143: does it matter WHICH completion the leaf ranking of `_menus` reads?

`_menus.views(side)` ranks side's candidates on the first-enumerated completion of the
opponent's unseen slots, ignoring the weights the bench prior gives them. On recorded
mid-game positions of a hidden-bench pool, build side's width-12 menu three ways:

  (a) first completion           -- what ships
  (b) heaviest completion        -- argmax weight, first on ties
  (c) weight-averaged ranking    -- `believed_ranking` over every completion

and count how many of the 12 actions differ from (a). Null control: (a) built twice.

    PYTHONPATH=<tree>/src python scratchpad/ika143_rank_completion.py \\
        <pool dir> <model.pt> <book.jsonl.gz> --positions 60

As generation passes it (`generate_selfplay`): the bench prior is `BenchPrior.of(entry,
side, species, epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE)` for the book
entry of the opponent's standings team, species in the order the book indexes (roster
order for side 0, `class_sets[0]` for side 1); weights come from `bench_weights` with
the carried seen identities (IKA-117) and the turn-1 lead pair (IKA-118), and an empty
result falls back to uniform exactly as `_bench_weights` does. Positions whose opponent
the book does not name are skipped (generation would rank them under a uniform belief,
where first vs heaviest is a tie by construction).

Where it departs from generation: the opponent's two unbrought sheet members are
`class_sets[0]`'s sets (generation used the drawn class's; the species are the same, so
the weights are too); the leaf is value-gen11L on CPU, one thread.

Cost: every `batched_payoff` call a ranking makes is counted as cells (ours x theirs).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

STATE: dict = {}
CELLS = [0]


def _init(model: str, book: str) -> None:
    import torch

    from pokeuraou import search as search_module
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.regulation import load_regulation
    from pokeuraou.selection_book import SelectionBook
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
    STATE["book"] = SelectionBook.read(Path(book))
    STATE["roster"] = load_roster(STATE["book"].roster or "rizabanadohido")
    STATE["standings"] = load_standings(find_cached_standings("2026", "worlds"), reg)

    real = search_module.batched_payoff

    def counted(reg_, pos, ours, theirs, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        CELLS[0] += len(ours) * len(theirs)
        return real(reg_, pos, ours, theirs, *args, **kwargs)

    search_module.batched_payoff = counted


def _set_from_json(entry: dict):  # noqa: ANN202
    from pokeuraou.priors import SampledSet

    return SampledSet(
        species=entry["species"], ability=entry["ability"], item=entry.get("item") or None,
        nature=entry["nature"], sp={k: int(v) for k, v in (entry.get("sp") or {}).items()},
        moves=list(entry["moves"]),
    )


def setup_for(game: dict):  # noqa: ANN202
    """(sheets, bench_prior) as generation built them for this game, or None."""
    from pokeuraou.regulation import to_id
    from pokeuraou.selection_book import (
        DEFAULT_EPSILON,
        DEFAULT_TEMPERATURE,
        BenchPrior,
    )

    roster = STATE["roster"]
    if sorted(to_id(s.species) for s in roster.sets) != sorted(game["ownSix"]):
        return None
    six = sorted(game["foeSix"])
    team = next(
        (t for t in STATE["standings"].teams if sorted(to_id(s) for s in t.species) == six),
        None,
    )
    if team is None:
        return None
    entry = STATE["book"].get(team)
    if entry is None:
        return None
    base = list(entry.class_sets[0])
    brought = {e["species"]: _set_from_json(e) for e in game["foeTeam"]}
    foe = [brought.get(to_id(s.species), s) for s in base]
    own = list(roster.sets)
    prior = (
        BenchPrior.of(entry, 0, [s.species for s in own],
                      epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE),
        BenchPrior.of(entry, 1, [s.species for s in base],
                      epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE),
    )
    return (own, foe), prior


def collect(pool: Path, games: int | None) -> tuple[list[dict], dict]:
    """Mid-game (turn >= 2) move decisions with the carried seen set and leads."""
    from pokeuraou.hidden import seen_identities, shown_species
    from pokeuraou.position import Position

    cases: list[dict] = []
    counts = {"games": 0, "move_mid": 0}
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
                start = Position.from_json(decisions[0]["position"])
                leads = [
                    sorted(shown_species(start, s, frozenset(
                        m.slot for m in start.sides[s].pokemon if m.active_index is not None
                    )))
                    if start.turn == 1 else None
                    for s in (0, 1)
                ]
                seen: list[frozenset[str]] = [frozenset(), frozenset()]
                for d in decisions:
                    if d["kind"] not in ("move", "replacement"):
                        continue
                    pos = Position.from_json(d["position"])
                    seen = [seen_identities(pos, s, seen[s]) for s in (0, 1)]
                    if d["kind"] != "move" or pos.turn < 2:
                        continue
                    counts["move_mid"] += 1
                    cases.append({
                        "file": path.name, "gameIndex": game.get("gameIndex"),
                        "game": {k: game[k] for k in ("ownSix", "foeSix", "foeTeam")},
                        "position": d["position"], "turn": pos.turn,
                        "seen": [sorted(s) for s in seen], "leads": leads,
                    })
        if games is not None and counts["games"] >= games:
            break
    return cases, counts


def spreads_for(case: dict):  # noqa: ANN202
    from pokeuraou.hidden import completions, seen_slots, shown_species
    from pokeuraou.position import Position

    got = setup_for(case["game"])
    if got is None:
        return None, "no book entry / sheet"
    sheets, prior = got
    pos = Position.from_json(case["position"])
    spreads = {}
    fallback = 0
    for side in (0, 1):
        shown = seen_slots(pos, side, frozenset(case["seen"][side]))
        leads = case["leads"][side]
        weights = prior[side].weights(
            shown_species(pos, side, shown), frozenset(leads) if leads else None
        )
        if not weights:
            weights = None
            fallback += 1
        spreads[side] = completions(reg_(), pos, side, sheets[side], seen=shown,
                                    weights=weights)
    return (pos, spreads, fallback), None


def reg_():  # noqa: ANN201
    return STATE["reg"]


def menu(pos, side, rank):  # noqa: ANN001, ANN201
    from pokeuraou.narrow import narrow

    return [a.to_choice() for a in narrow(reg_(), pos, side, limit=12, rank=rank).actions]


def run_side(pos, side: int, items) -> dict:  # noqa: ANN001
    from pokeuraou.resolve import Budget
    from pokeuraou.search import believed_ranking, leaf_ranking

    budget = Budget.matrix()
    leaf = STATE["leaf"]
    weights = [it.weight for it in items]
    heavy = int(np.argmax(weights))
    out: dict = {"n": len(items), "weights": weights, "heavy": heavy}
    menus = {}
    for label, parts in (
        ("a", [(items[0].position, 1.0)]),
        ("a2", [(items[0].position, 1.0)]),
        ("b", [(items[heavy].position, 1.0)]),
        ("c", [(it.position, it.weight) for it in items]),
    ):
        CELLS[0] = 0
        started = time.perf_counter()
        rank = believed_ranking(
            [(leaf_ranking(reg_(), at, side, leaf, budget=budget), w) for at, w in parts]
        )
        menus[label] = menu(pos, side, rank)
        out[f"cells_{label}"] = CELLS[0]
        out[f"sec_{label}"] = time.perf_counter() - started
    out["menus"] = menus
    base = set(menus["a"])
    for label in ("a2", "b", "c"):
        out[f"diff_{label}"] = len(base - set(menus[label]))
    out["diff_bc"] = len(set(menus["b"]) - set(menus["c"]))
    out["top_changed_b"] = menus["a"][:1] != menus["b"][:1]
    out["size"] = len(menus["a"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("book", type=Path)
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=143)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    started = time.time()
    cases, counts = collect(args.pool, args.games)
    print(f"{counts['games']} hidden-bench games, {counts['move_mid']} move decisions at "
          f"turn >= 2", flush=True)
    _init(str(args.model), str(args.book))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(cases))
    rows: list[dict] = []
    tried = {"no book entry / sheet": 0, "nothing uncertain": 0, "used": 0,
             "fallback_uniform_sides": 0}
    for index in order:
        if tried["used"] >= args.positions:
            break
        case = cases[int(index)]
        got, why = spreads_for(case)
        if got is None:
            tried[why] += 1
            continue
        pos, spreads, fallback = got
        sides = [s for s in (0, 1) if len(spreads[1 - s]) >= 2]
        if not sides:
            tried["nothing uncertain"] += 1
            continue
        tried["used"] += 1
        tried["fallback_uniform_sides"] += fallback
        for side in sides:
            row = run_side(pos, side, spreads[1 - side])
            row.update(turn=case["turn"], side=side, file=case["file"],
                       gameIndex=case["gameIndex"])
            rows.append(row)
        print(f"  {tried['used']:3d}  rows {len(rows)}  wall {time.time() - started:.0f}s",
              flush=True)
    print(f"positions: {tried}; ranked (position, side) rows {len(rows)}; "
          f"wall {time.time() - started:.0f}s")

    def summary(label: str, sel: list[dict]) -> None:
        if not sel:
            return
        print(f"-- {label}: rows {len(sel)}")
        for arm in ("a2", "b", "c"):
            d = np.array([r[f"diff_{arm}"] for r in sel])
            print(f"   (a) vs ({arm}): menu changed {int((d > 0).sum())}/{len(sel)} "
                  f"({(d > 0).mean():.1%}); actions swapped mean {d.mean():.2f} "
                  f"(of {np.mean([r['size'] for r in sel]):.1f}), max {d.max()}, "
                  f"histogram {np.bincount(d).tolist()}")
        d = np.array([r["diff_bc"] for r in sel])
        print(f"   (b) vs (c): changed {int((d > 0).sum())}/{len(sel)}, mean {d.mean():.2f}")
        for arm in ("a", "c"):
            cells = np.array([r[f"cells_{arm}"] for r in sel])
            sec = np.array([r[f"sec_{arm}"] for r in sel])
            print(f"   cost ({arm}): cells mean {cells.mean():.0f}, seconds mean "
                  f"{sec.mean():.3f}")
        ratio = np.array([r["cells_c"] / max(r["cells_a"], 1) for r in sel])
        print(f"   (c)/(a) cells ratio mean {ratio.mean():.2f}; completions mean "
              f"{np.mean([r['n'] for r in sel]):.2f}")

    heavy_first = [r for r in rows if r["heavy"] == 0]
    heavy_other = [r for r in rows if r["heavy"] != 0]
    uniform = [r for r in rows if max(r["weights"]) - min(r["weights"]) < 1e-12]
    print(f"heaviest is the first: {len(heavy_first)}/{len(rows)}; weights uniform: "
          f"{len(uniform)}/{len(rows)}; mean weight first {np.mean([r['weights'][0] for r in rows]):.3f}"
          f" heaviest {np.mean([max(r['weights']) for r in rows]):.3f}")
    summary("all rows", rows)
    summary("heaviest is not the first", heavy_other)
    if args.out is not None:
        args.out.write_bytes(json.dumps({"counts": counts, "tried": tried,
                                         "rows": rows}).encode())


if __name__ == "__main__":
    main()
