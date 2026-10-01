"""The typed-in position's controls that need `data/` (IKA-408): they cannot run on CI, so they are
a tool with its numbers recorded in records/IKA-408.md.

    uv run python tools/ranked_position_controls.py run --records data/matches-mc/<run> --positions 140 \\
        --out out.jsonl [--shard 0/4] [--max-steps 0]
    uv run python tools/ranked_position_controls.py summarize out.jsonl [out2.jsonl ...]

What it asks: a person typed a recorded position into the screen's form -- their own side exactly, the
opponent's HP as a per cent, only what was seen -- how far is the screen's read from the read of the
recorded position itself?

For each position of a recorded M-C game (a pool match: both sheets, the recorded position with the
true sets), the analysis mode's read (`analysis.Analyzer.run`, ``--max-steps`` fixed so the answer is
reproducible) is made from side 0 for:

======  =====================================================================================
ref     the record's own position (true sets, exact HP, every state a battle carries)
null    the form of that position, true sets, exact HP: what the form's coarser state costs
        (volatiles, PP, last-turn memory are not in a form)
a_low   as null, the opponent's HP only as the bottom / middle / top of the per cent's band
a_mid   (so: how much does a per cent cost, and does the middle do)
a_high
b       the opponent's sets *estimated* from its six species (`FieldPrior`, the field teams that
        bring exactly that six taken out first), nothing seen used
c       as b, with what was seen: the moves it used on earlier turns and a spent item, through
        the screen's own narrowing (`rankedposition.derive`)
x       as b with one active opponent's set swapped for the field's rarest set of the species
        (a wrong estimate: the comparison can fail)
======  =====================================================================================

The numbers: the total variation distance between the two mixtures (over the same actions, named
by move, target and switch-in, not by party number), the difference of the value (side 0's), and
whether the most likely action is the same. The positions are split by whether the opponent's bench
is hidden (some of its four not shown yet) or open (all four shown).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()

import numpy as np  # noqa: E402

from pokeuraou import analysis, humanplay  # noqa: E402
from pokeuraou import rankedposition as rp  # noqa: E402
from pokeuraou.actions import MoveAction, SwitchAction, side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.hidden import identity, seen_slots  # noqa: E402
from pokeuraou.hpdisplay import displayed_colour, displayed_percent, uses_floor_display  # noqa: E402
from pokeuraou.humanplay import parse_choice  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402
from pokeuraou.priors import SampledSet  # noqa: E402
from pokeuraou.rankedboard import make_game  # noqa: E402
from pokeuraou.rankedentry import FieldPrior, OpponentSet  # noqa: E402
from pokeuraou.regulation import to_id  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"
VARIANTS = ("null", "a_low", "a_mid", "a_high", "b", "c", "x", "n_pp", "n_mem", "n_all", "n_unseen", "x_true",
            "n_swap", "n_best")


# ----------------------------------------------------------------------------- a record as forms


class Unusable(Exception):
    """A position the form cannot carry (a state it has no field for)."""


def base_id(reg, species_id: str) -> str:  # noqa: ANN001
    return to_id(reg.species[species_id].base_species)


def moves_used(reg, game, k: int, foe_id: dict[str, str]) -> dict[str, list[str]]:  # noqa: ANN001
    """The moves the opponent's Pokemon used before decision ``k`` (what a person watching saw)."""
    out: dict[str, list[str]] = defaultdict(list)
    for d in range(k):
        point = game.points[d]
        choice = point.played[1]
        if not choice:
            continue
        pos = point.pos()
        try:
            action = parse_choice(choice, side_actions(reg, pos, 1))
        except (ValueError, KeyError, IndexError):
            continue
        for slot in action.slots:
            if isinstance(slot, MoveAction):
                party = pos.sides[1].active[slot.slot]
                if party is None:
                    continue
                sid = foe_id.get(identity(pos.sides[1].pokemon[party]))
                if sid and slot.move_id in reg.moves and slot.move_id not in out[sid]:
                    out[sid].append(slot.move_id)
    return out


def carried(mon, *, locks: bool = True) -> dict:  # noqa: ANN001
    """What the person sees of a Pokemon's volatile state that the form has a field for: how many
    turns in a row it used Protect, and the move a Choice item has it locked into (``locks``: a lock
    names a move the Pokemon used, so a variant that must not use what was seen leaves it out)."""
    out: dict = {}
    for effect in mon.volatiles:
        if effect.id == "stall":
            out["protect"] = min(2, max(1, round(math.log(max(effect.counter or 3, 3), 3))))
        elif effect.id == "choicelock" and effect.move and locks:
            out["locked"] = effect.move
    return out


def now_ability(mon, came_with: str | None) -> str | None:  # noqa: ANN001
    """The ability in effect when it is not the one the Pokemon's set has (Trace copied one). The
    Mega forme's own ability is the Mega Evolution's, not this."""
    if mon.is_mega or not came_with or to_id(mon.ability) == to_id(came_with):
        return None
    return mon.ability


def form_of(reg, game, d: int, mine_idx, foe_id, used, *, observed: bool) -> dict:  # noqa: ANN001
    """The form a person would type for decision ``d``'s position."""
    mine_roster, foe_roster = game.teams
    foe_ability = {s.species: s.ability for s in foe_roster.sets}
    point = game.points[d]
    pos = point.pos()
    floor = uses_floor_display(reg)
    side0, side1 = pos.sides
    shown1 = seen_slots(pos, 1, point.seen[1])
    mine, theirs = {}, {}
    for mon in side0.pokemon:
        idx = mine_idx[identity(mon)]
        mine[str(idx)] = {
            "hp": mon.hp, "status": None if mon.status == "fnt" else mon.status,
            "boosts": {k: v for k, v in mon.boosts.items() if v},
            "mega": bool(mon.is_mega), "itemGone": mon.item is None and mon.base_item is not None,
            **carried(mon), "abilityNow": now_ability(mon, mine_roster.sets[idx].ability)}
    for mon in side1.pokemon:
        if mon.slot not in shown1 and mon.active_index is None:
            continue
        sid = foe_id[identity(mon)]
        gone = mon.item is None and mon.base_item is not None
        entry = {
            "hpExact": mon.hp, "pct": max(1, displayed_percent(mon.hp, mon.maxhp, floor_rule=floor)),
            "colour": displayed_colour(mon.hp, mon.maxhp),
            "status": None if mon.status == "fnt" else mon.status,
            "boosts": {k: v for k, v in mon.boosts.items() if v}, "mega": bool(mon.is_mega),
            "fainted": bool(mon.fainted), "moves": list(used.get(sid, [])) if observed else [],
            **carried(mon, locks=observed),
            "abilityNow": now_ability(mon, foe_ability[sid]) if observed else None,
            "item": mon.base_item if (gone and observed) else None, "itemGone": gone and observed,
            "ability": None}
        theirs[sid] = entry
    weather = pos.field.weather
    sides = []
    for side in (side0, side1):
        conds: dict[str, int] = {}
        for effect in side.side_conditions:
            if effect.id in rp.TIMED:
                conds[effect.id] = max(1, int(effect.duration or 1))
            elif effect.id in rp.LAYERED:
                conds[effect.id] = int(effect.layers or 1)
        sides.append(conds)
    trick = next((int(e.duration or 1) for e in pos.field.pseudo_weather if e.id == "trickroom"), 0)
    form = {
        "turn": point.turn,
        # a fainted Pokemon left in an active slot (no one to replace it) is not on the field
        "mineActive": [None if a is None or side0.pokemon[a].fainted
                       else mine_idx[identity(side0.pokemon[a])] for a in side0.active],
        "theirActive": [None if a is None or side1.pokemon[a].fainted
                        else foe_id[identity(side1.pokemon[a])] for a in side1.active],
        "mine": mine, "theirs": theirs,
        "field": {"weather": weather, "weatherTurns": int(pos.field.weather_duration or 5),
                  "terrain": pos.field.terrain, "terrainTurns": int(pos.field.terrain_duration or 5),
                  "trickRoom": trick},
        "sides": sides, "events": {"order": [], "damage": []},
    }
    return form


def board_for(reg, game, k: int, *, observed: bool, namer=None):  # noqa: ANN001, ANN201
    """The board a person would have by decision ``k``: one form a decision up to it."""
    mine, foe = game.teams
    mine_idx = {base_id(reg, s.species): i for i, s in enumerate(mine.sets)}
    foe_id = {base_id(reg, s.species): s.species for s in foe.sets}
    lead_mine = sorted(game.leads[0] or ())
    lead_foe = sorted(game.leads[1] or ())
    if len(lead_mine) != 2 or len(lead_foe) != 2:
        raise Unusable("no leads in the record")
    pos = game.points[k].pos()
    brought = [mine_idx[i] for i in lead_mine]
    brought += [mine_idx[identity(m)] for m in pos.sides[0].pokemon if mine_idx[identity(m)] not in brought]
    board = rp.Board(reg, mine, brought, [s.species for s in foe.sets], [foe_id[i] for i in lead_foe], [])
    board.namer = namer
    try:
        board.check()
        used_at = {d: moves_used(reg, game, d, foe_id) for d in range(k + 1)}
        board.turns = [rp.normalize(reg, board, form_of(reg, game, d, mine_idx, foe_id, used_at[d],
                                                        observed=observed)) for d in range(k + 1)]
    except rp.RankedError as exc:
        raise Unusable(str(exc)) from exc
    return board


#: What a form cannot say of a Pokemon, in the order it is put back to see what the gap is made of.
MEMORY_FIELDS = ("last_move", "locked_move", "move_last_turn_failed", "times_attacked", "active_move_actions",
                 "status_duration", "status_counter", "trapped", "newly_switched", "ability_state")


def with_record_state(typed, record, groups: tuple[str, ...]):  # noqa: ANN001, ANN201
    """The typed position with some of what the record's position holds put back: ``pp`` (the move
    slots: PP, order), ``mem`` (what a Pokemon remembers of the turns before), ``vol`` (the volatiles and the
    side conditions the form has no field for). Pokemon are matched by identity; an unseen member
    of the opponent's four has no match and is left."""
    out = typed.copy()
    for side_t, side_r in zip(out.sides, record.sides, strict=True):
        by_id = {identity(m): m for m in side_r.pokemon}
        for mon in side_t.pokemon:
            ref = by_id.get(identity(mon))
            if ref is None:
                continue
            # the move slots as the battle had them: PP spent, used, and the order they stand in
            if "pp" in groups and sorted(m.id for m in ref.moves) == sorted(m.id for m in mon.moves):
                mon.moves = [m.copy() for m in ref.moves]
            if "mem" in groups:
                for name in MEMORY_FIELDS:
                    setattr(mon, name, getattr(ref, name))
            if "vol" in groups:
                mon.volatiles = [e.copy() for e in ref.volatiles]
                mon.unmodelled_volatiles = list(ref.unmodelled_volatiles)
        if "vol" in groups:
            side_t.side_conditions = [e.copy() for e in side_r.side_conditions]
            side_t.slot_conditions = [[e.copy() for e in g] for g in side_r.slot_conditions]
            side_t.mega_used = side_r.mega_used
    if "vol" in groups:
        out.field = record.field.copy()
    if "unseen" in groups:
        # the opponent's members nobody has seen: the typed position holds some of its six, the
        # record's the ones it really brought; their identity is the one thing a person cannot type
        for side_t, side_r in zip(out.sides, record.sides, strict=True):
            have = {identity(m) for m in side_t.pokemon}
            spare = [m for m in side_r.pokemon if identity(m) not in have]
            for mon in side_t.pokemon:
                if identity(mon) not in {identity(m) for m in side_r.pokemon} and spare:
                    new = spare.pop(0).copy()
                    new.slot = mon.slot
                    side_t.pokemon[mon.slot] = new
            side_t.mega_capable_slots = list(side_r.mega_capable_slots)
    return out


def with_unseen(reg, typed, chosen: list, seen_ids):  # noqa: ANN001, ANN201
    """The typed position with the opponent's unseen members replaced by ``chosen`` (sets of its sheet)."""
    from pokeuraou.selfplay import _make_pokemon

    out = typed.copy()
    side = out.sides[1]
    placed = [m for m in side.pokemon if m.active_index is None and identity(m) not in seen_ids]
    for mon, sset in zip(placed, chosen, strict=False):
        side.pokemon[mon.slot] = _make_pokemon(reg, mon.slot, sset, None)
    side.mega_capable_slots = [m.slot for m in side.pokemon if reg.mega_target(m.species, m.item) is not None]
    return out


def override_of(one: SampledSet) -> OpponentSet:
    return OpponentSet(replace(one, moves=list(one.moves), sp=dict(one.sp)), "override", 0, 0, (),
                       "override", 0)


def rarest(prior, species: str, six) -> OpponentSet:  # noqa: ANN001
    """The estimate with the field's rarest whole set of the species put in (a wrong one)."""
    base = prior.fill(species, six)
    assert base.belief is not None
    cands = base.belief.candidates()
    return prior._build(species, cands[-1], "candidate", belief=base.belief, whole_n=cands[-1].count,
                        alternatives=base.alternatives, notes=[])


# ----------------------------------------------------------------------------- the reads


def canon(reg, pos, side: int, choices, probs) -> dict[tuple, float]:  # noqa: ANN001
    """A mixture over actions named by what they do (move, target, switch-in), not by party number."""
    legal = side_actions(reg, pos, side)
    out: dict[tuple, float] = defaultdict(float)
    for choice, p in zip(choices, probs, strict=True):
        try:
            action = parse_choice(choice, legal)
        except (ValueError, KeyError, IndexError):
            continue
        key = tuple(
            ("m", s.move_id, s.target, s.mega) if isinstance(s, MoveAction)
            else ("s", s.species) if isinstance(s, SwitchAction) else ("p",)
            for s in action.slots)
        out[key] += float(p)
    return dict(out)


def tv(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def best(a: dict) -> tuple:
    return sorted(a.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def load_analyzer(args):  # noqa: ANN001, ANN201
    import ranked_entry

    reg = load_pool(args.data / "pool" / "regmc-matchupweb.json").reg
    register_mega_stones(reg)
    values = [args.data / "models" / m for m in ("value-mc3.pt", "value-mc3-s1.pt")]
    evaluate, encoder, device = humanplay.load_leaf(reg, values, "cpu", graphs=False)
    ph = ranked_entry._play_human()
    fill, _files = ph.install_menus(None, args.data / "models" / "q-mc3.pt", encoder, evaluate, None, device,
                                    lambda t: print(t, file=sys.stderr))
    settings = analysis.Settings(width=analysis.DEFAULT_WIDTH, oracle=ph._oracle_width("sall"),
                                 levels=humanplay.PLAY_MAX_LEVELS or analysis.MAX_LEVELS, rank_fill=fill)
    return reg, analysis.Analyzer(reg, evaluate, "value-mc3x2", settings=settings)


def choose_positions(reg, records: list[Path], count: int, seed: int, pool) -> list[tuple]:  # noqa: ANN001
    """``count`` positions, half with the opponent's bench hidden and half open, from records spread
    over the files; (record file, game number, decision number)."""
    rng = random.Random(seed)
    pools = {pool.id: pool}
    hidden: list[tuple] = []
    open_: list[tuple] = []
    for path in records:
        games = analysis.load_games(path, pools=pools, limit=24)
        rng.shuffle(games)
        for game in games[:6]:
            if game.teams is None or game.information != "hidden-bench" or not game.points:
                continue
            if len(game.leads[0] or ()) != 2 or len(game.leads[1] or ()) != 2:
                continue
            picks = rng.sample(range(len(game.points)), min(3, len(game.points)))
            for k in picks:
                pt = game.points[k]
                pos = pt.pos()
                n_seen = len(seen_slots(pos, 1, pt.seen[1]))
                (hidden if n_seen < len(pos.sides[1].pokemon) else open_).append((path, game.index, k))
    rng.shuffle(hidden)
    rng.shuffle(open_)
    half = count // 2
    return [(*t, "hidden") for t in hidden[:half]] + [(*t, "open") for t in open_[: count - half]]


def run(args) -> None:  # noqa: ANN001
    reg, analyzer = load_analyzer(args)
    pool = load_pool(args.data / "pool" / "regmc-matchupweb.json")
    prior_all = FieldPrior.load(reg, args.data)
    files = sorted(args.records.glob("games-worker*.jsonl"))
    chosen = choose_positions(reg, files, args.positions, args.seed, pool)
    print(f"{len(chosen)} positions: {sum(1 for c in chosen if c[3] == 'hidden')} hidden-bench, "
          f"{sum(1 for c in chosen if c[3] == 'open')} open", flush=True)
    shard, of = (int(x) for x in args.shard.split("/"))
    pools = {pool.id: pool}
    games_cache: dict = {}
    out = args.out.open("ab")
    done = 0
    for n, (path, gindex, k, kind) in enumerate(chosen):
        if n % of != shard or (args.only and str(n) not in args.only.split(",")):
            continue
        if path not in games_cache:
            games_cache[path] = {g.index: g for g in analysis.load_games(path, pools=pools, limit=24)}
        game = games_cache[path][gindex]
        started = time.perf_counter()
        row: dict = {"n": n, "file": path.name, "game": gindex, "decision": k, "kind": kind,
                     "turn": game.points[k].turn}
        try:
            mine, foe = game.teams
            six = [s.species for s in foe.sets]
            kept = [t for t in prior_all.standings.teams if frozenset(t.species) != frozenset(six)]
            prior = FieldPrior(reg, replace(prior_all.standings, teams=kept), None)
            if any(s not in prior.members for s in six):
                raise Unusable("a species the held-out field never shows")
            est, problems = prior.fill_team(six, six)
            est_base = {o.set.species: o for o in est}
            true_base = {s.species: override_of(s) for s in foe.sets}
            wrong_base = dict(est_base)
            wrong_true = dict(true_base)
            actives = [a for a in game.points[k].pos().sides[1].active if a is not None]
            first = game.points[k].pos().sides[1].pokemon[actives[0]]
            wrong_sid = next(s.species for s in foe.sets if base_id(reg, s.species) == identity(first))
            if len(prior.belief(wrong_sid, six).candidates()) > 1:
                wrong_base[wrong_sid] = wrong_true[wrong_sid] = rarest(prior, wrong_sid, six)
                row["xDiffers"] = True
            else:
                row["xDiffers"] = False
            plain, observed = board_for(reg, game, k, observed=False), board_for(reg, game, k, observed=True)
            point0 = game.points[k]
            pos0 = point0.pos()
            row["volatiles"] = any(m.volatiles for s in pos0.sides for m in s.pokemon)
            row["volatileIds"] = sorted({e.id for s in pos0.sides for m in s.pokemon for e in m.volatiles})
            specs = {
                "null": (observed, true_base, "exact"), "a_low": (observed, true_base, "low"),
                "a_mid": (observed, true_base, "mid"), "a_high": (observed, true_base, "high"),
                "b": (plain, est_base, "mid"), "c": (observed, est_base, "mid"),
                "x": (plain, wrong_base, "mid"), "x_true": (observed, wrong_true, "mid"),
            }
            reads = {}

            def read(g, p, ms=args.max_steps):  # noqa: ANN001, ANN202
                r = analyzer.run(g, p, 0, max_steps=ms)
                position = analysis.Position.from_json(p.position)
                return r, canon(reg, position, 0, r.ours, [float(x) for x in r.strategy])

            ref, ref_mix = read(game, point0)
            reads["ref"] = (ref, ref_mix)
            sets_by_variant = {}
            for name, (board, base, mode) in specs.items():
                derived = rp.derive(reg, board, prior, base, hp_mode=mode, observe_spread=False)
                g, p, _built = make_game(reg, mine, board, derived, len(board.turns) - 1)
                reads[name] = read(g, p)
                if name == "null":
                    # the same form with what it cannot say put back from the record, a group at a time
                    for label, groups in (("n_pp", ("pp",)), ("n_mem", ("pp", "mem")),
                                          ("n_all", ("pp", "mem", "vol")),
                                          ("n_unseen", ("pp", "mem", "vol", "unseen"))):
                        filled = with_record_state(analysis.Position.from_json(p.position), pos0, groups)
                        reads[label] = read(g, replace(p, position=filled.to_json()))
                if name == "null":
                    # an arbitrary other pair of the unseen candidates, and the analysis' own heaviest
                    unseen = [x for x in foe.sets if base_id(reg, x.species) not in p.seen[1]]
                    held = {identity(m) for m in analysis.Position.from_json(p.position).sides[1].pokemon}
                    others = [x for x in unseen if base_id(reg, x.species) not in held] or unseen
                    typed_pos = analysis.Position.from_json(p.position)
                    n_unseen_slots = sum(1 for m in typed_pos.sides[1].pokemon
                                         if m.active_index is None and identity(m) not in p.seen[1])
                    if n_unseen_slots:
                        swapped = with_unseen(reg, typed_pos, others[:n_unseen_slots], p.seen[1])
                        reads["n_swap"] = read(g, replace(p, position=swapped.to_json()))
                        spreads = analyzer.spreads(g, p, typed_pos, analyzer.settings, [])
                        heaviest = max(spreads[1], key=lambda c: c.weight)
                        wanted = [x for x in foe.sets if x.species in set(heaviest.species)]
                        best_pos = with_unseen(reg, typed_pos, wanted, p.seen[1])
                        reads["n_best"] = read(g, replace(p, position=best_pos.to_json()))
                    else:
                        reads["n_swap"] = reads["n_best"] = reads["null"]
                sets_by_variant[name] = {
                    s: (v.item, tuple(sorted(v.moves)), v.nature) for s, v in derived.sets.items()}
            if args.repeat and n < args.repeat:
                board, base, mode = specs["b"]
                derived = rp.derive(reg, board, prior, base, hp_mode=mode, observe_spread=False)
                g, p, _built = make_game(reg, mine, board, derived, len(board.turns) - 1)
                again = read(g, p)
                row["repeatSame"] = bool(
                    again[1] == reads["b"][1] and again[0].value0 == reads["b"][0].value0)
            row["cChanged"] = sets_by_variant["c"] != sets_by_variant["b"]
            row["refValue"] = ref.value0
            row["steps"] = ref.steps
            for name in VARIANTS:
                result, mix = reads[name]
                row[name] = {"tv": tv(ref_mix, mix), "dv": result.value0 - ref.value0, "value": result.value0,
                             "same": best(ref_mix) == best(mix)}
            row["seconds"] = time.perf_counter() - started
        except (Unusable, rp.RankedError, KeyError, ValueError) as exc:
            row["skipped"] = f"{type(exc).__name__}: {exc}"
        out.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        out.flush()
        done += 1
        note = row.get("skipped") or f"{row['seconds']:.1f} s"
        print(f"{n}: {kind} turn {row['turn']} {note}", flush=True)
    print(f"shard {shard}/{of}: {done} positions")


# ----------------------------------------------------------------------------- the typing


def flatten(form: dict, prefix: str = "") -> dict[str, object]:
    """A form as ``path -> value`` (the hidden measuring key and the turn number left out)."""
    out: dict[str, object] = {}
    for key, value in form.items():
        if key in ("hpExact", "turn", "events"):
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, path + "."))
        else:
            out[path] = tuple(value) if isinstance(value, list) else value
    return out


def inputs(args) -> None:  # noqa: ANN001
    """How much a person types: the fields that changed from one turn's form to the next (the
    form carries the rest over), over the recorded games' turns, against the fields in a form."""
    reg = load_pool(args.data / "pool" / "regmc-matchupweb.json").reg
    pool = load_pool(args.data / "pool" / "regmc-matchupweb.json")
    pools = {pool.id: pool}
    refused: dict[str, int] = defaultdict(int)
    changes: list[int] = []
    sizes: list[int] = []
    per_game: list[int] = []
    for path in sorted(args.records.glob("games-worker*.jsonl"))[: args.files]:
        for game in analysis.load_games(path, pools=pools, limit=args.games):
            if game.teams is None or len(game.points) < 2 or len(game.leads[0] or ()) != 2:
                continue
            try:
                board = board_for(reg, game, len(game.points) - 1, observed=True)
            except Unusable as why:
                refused[str(why)[:60]] += 1
                continue
            flats = [flatten(f) for f in board.turns]
            sizes.append(len(flats[-1]))
            moved = []
            for a, b in zip(flats, flats[1:], strict=False):
                keys = set(a) | set(b)
                moved.append(sum(1 for k in keys if a.get(k) != b.get(k)))
            changes += moved
            per_game.append(sum(moved))
    arr = np.asarray(changes)
    print("not usable:", dict(refused))
    print(f"{len(per_game)} games, {len(changes)} turn-to-turn steps; "
          f"fields in a form: mean {np.mean(sizes):.0f}")
    print(f"fields changed per turn: mean {arr.mean():.1f}, median {np.median(arr):.0f}, "
          f"90th {np.percentile(arr, 90):.0f}, max {arr.max()}")
    print(f"a game's whole input after turn 1: mean {np.mean(per_game):.0f} field changes "
          f"over {len(changes) / len(per_game):.1f} steps")


# ----------------------------------------------------------------------------- the numbers


def mean_se(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")


def summarize(files: list[Path]) -> None:
    rows = [json.loads(line) for f in files for line in f.read_bytes().splitlines() if line.strip()]
    ok = [r for r in rows if "skipped" not in r]
    print(f"rows {len(rows)}, read {len(ok)}, skipped {len(rows) - len(ok)}")
    for r in rows:
        if "skipped" in r:
            print("  skipped:", r["n"], r["skipped"][:100])
    names = {"x_true": "x_true (true sets, one active's set wrong)",
             "n_unseen": "n_unseen (n_all + the unseen members as they were)",
             "n_pp": "n_pp  (null + PP from the record)", "n_mem": "n_mem (n_pp + what it remembers)",
             "n_all": "n_all (n_mem + volatiles, side states)",
             "null": "null  (form, true sets, exact HP)", "a_low": "a_low (true sets, HP band bottom)",
             "a_mid": "a_mid (true sets, HP band middle)", "a_high": "a_high (true sets, HP band top)",
             "b": "b     (estimated sets)", "c": "c     (b + seen moves/items)",
             "x": "x     (b with one wrong set)"}
    for group in ("all", "hidden", "open"):
        sub = [r for r in ok if group == "all" or r["kind"] == group]
        if not sub:
            continue
        print(f"\n== {group}: {len(sub)} positions (deterministic repeats same: "
              f"{sum(1 for r in sub if r.get('repeatSame'))}/{sum(1 for r in sub if 'repeatSame' in r)})")
        print(f"{'variant':<38}{'TV':>8}{'SE':>8}{'|dV|':>8}{'SE':>8}{'best same':>11}")
        for v in VARIANTS:
            sel = [r for r in sub if (v not in ("x", "x_true") or r["xDiffers"])]
            t, ts = mean_se([r[v]["tv"] for r in sel])
            d, ds = mean_se([abs(r[v]["dv"]) for r in sel])
            same = sum(1 for r in sel if r[v]["same"])
            print(f"{names[v]:<38}{t:>8.4f}{ts:>8.4f}{d:>8.4f}{ds:>8.4f}{same:>7}/{len(sel):<3}")
        # paired differences
        for a, b, label in (("c", "b", "c - b (what seeing buys)"), ("x", "b", "x - b (a wrong set costs)"),
                            ("x_true", "a_mid", "x_true - a_mid (a wrong set among true ones)"),
                            ("a_low", "a_high", "a_low - a_high (the HP band's width)"),
                            ("a_mid", "null", "a_mid - null (a per cent vs exact)")):
            sel = [r for r in sub if (a not in ("x", "x_true") or r["xDiffers"])]
            dt, dts = mean_se([r[a]["tv"] - r[b]["tv"] for r in sel])
            dv, dvs = mean_se([abs(r[a]["dv"]) - abs(r[b]["dv"]) for r in sel])
            print(f"  {label:<42} dTV {dt:+.4f} ({dts:.4f})   d|dV| {dv:+.4f} ({dvs:.4f})   n {len(sel)}")
        changed = [r for r in sub if r["cChanged"]]
        if changed:
            dt, dts = mean_se([r["c"]["tv"] - r["b"]["tv"] for r in changed])
            print(f"  where seeing changed the estimate: {len(changed)} positions, "
                  f"c - b dTV {dt:+.4f} ({dts:.4f}); "
                  f"c closer in {sum(1 for r in changed if r['c']['tv'] < r['b']['tv'])}, "
                  f"farther in {sum(1 for r in changed if r['c']['tv'] > r['b']['tv'])}")
        vol = [r for r in sub if r["volatiles"]]
        if vol:
            t, ts = mean_se([r["null"]["tv"] for r in vol])
            print(f"  null on the {len(vol)} positions with a volatile on the field: TV {t:.4f} ({ts:.4f}); "
                  f"without: {mean_se([r['null']['tv'] for r in sub if not r['volatiles']])[0]:.4f}")
        print(f"  steps of the reference read: mean {np.mean([r['steps'] for r in sub]):.1f}; "
              f"{np.mean([r['seconds'] for r in sub]):.1f} s a position")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, default=ROOT / "data")
    r.add_argument("--records", type=Path, required=True, help="a directory of games-worker*.jsonl")
    r.add_argument("--positions", type=int, default=140)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-steps", type=int, default=0)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--only", default="", help="positions by number, comma separated (default: all)")
    r.add_argument("--repeat", type=int, default=0,
                   help="re-read the first N positions' b to check determinism")
    r.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("inputs")
    i.add_argument("--data", type=Path, default=ROOT / "data")
    i.add_argument("--records", type=Path, required=True)
    i.add_argument("--files", type=int, default=6)
    i.add_argument("--games", type=int, default=6)
    s = sub.add_parser("summarize")
    s.add_argument("files", type=Path, nargs="+")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    elif args.cmd == "inputs":
        inputs(args)
    else:
        summarize(args.files)


if __name__ == "__main__":
    main()
