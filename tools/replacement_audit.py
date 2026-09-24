"""IKA-223: how far the hidden-bench replacement matrix's off-by-one reached into the records.

From d959ab0 (9/17) until IKA-209 moved generation onto the port, the replacement node
under a hidden bench built its matrix in every completion of the opponent's unseen bench,
and Python's `resolve_replacements` passed `switch_to=party_index` (1-based) where every
other caller subtracted one. `_find_switch_target` looks by species first, so on the true
position nothing happened; in a completion that holds some other species in the slot the
switch named, it fell through to the index and brought in the *next* party member -- or
nobody, when that index was out of range, fainted or already out. The move that was then
*played* was resolved on the true position and was right; what was wrong is the matrix,
so the recorded `ownPolicy`, `foePolicy`, `searchValue` and `foeSearchValue` of those
replacement decisions, and the replacement each side drew from them.

Two modes.

``count`` reads every game of a directory (one worker per file) and counts, per
directory: games, decisions by kind, replacement decisions, and the replacement
decisions that could have been hit -- some side owes a replacement, one of its options
switches into a slot its opponent has not seen, and the sheet leaves more candidates
than unseen slots, so some completion of that bench lacks the species the option names.
Pure JSON; no port, no leaf.

``solve`` re-solves a sample of replacement decisions with today's port, twice: once as
the node is now (``fixed``) and once with every switch's party index moved up by one
(``shifted``), which is the recorded code's fall-through exactly -- the port's
`switch_target` is Python's (species, then index), so index + 1 reproduces
`switch_to=party_index`. Each is compared with the record, and with the other. The inputs
the recorded node had are rebuilt per era (`--carry`, `--onboard`, `--prior`): which
Pokemon each side had shown, which species counted as on the board, and the bench prior.
`--definition true` solves the node on the true position, which is what the node did
before d959ab0 (the null control on gen11h). One worker per decision (`--jobs`); the
summary does not depend on the number of workers (the `--jobs 1` null control).

    PYTHONPATH=<wt>/src python tools/replacement_audit.py count <dir> [<dir> ...] --jobs 8
    PYTHONPATH=<wt>/src python tools/replacement_audit.py solve <dir> --value <model.pt> \\
        --carry slots --onboard species --prior none --games 40 --jobs 1 --out rows.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("POKEURAOU_RUST_NODE", "1")

# ---------------------------------------------------------------------------
# JSON-level helpers (count mode reads the raw records; nothing is built)
# ---------------------------------------------------------------------------


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _shows_itself(mon: dict[str, Any]) -> bool:
    """`hidden._shows_itself` on a recorded Pokemon."""
    return bool(
        mon.get("activeIndex") is not None
        or mon.get("fainted")
        or mon.get("hp") != mon.get("maxhp")
        or mon.get("status") is not None
        or any((mon.get("boosts") or {}).values())
        or mon.get("volatiles")
        or mon.get("isMega")
    )


def _shown_slots_json(
    side: dict[str, Any], carry: str, carried: Any, recorded: list[str] | None
) -> tuple[frozenset[int], Any]:
    """This position's shown slots, and what to carry to the next decision.

    ``slots``: the code before IKA-117 carried party slots (`seen_slots(pos, i, shown)`).
    ``identities``: IKA-117's base-species identities. ``recorded``: the decision's own
    ``shownIdentities`` (written from IKA-127 on).
    """
    mons = side["pokemon"]
    if carry == "slots":
        shown = frozenset(carried) | {m["slot"] for m in mons if _shows_itself(m)}
        return frozenset(shown), shown
    if carry == "recorded" and recorded is not None:
        ids = frozenset(recorded)
    else:
        ids = frozenset(carried) | {to_id(m["baseSpecies"]) for m in mons if _shows_itself(m)}
    shown = frozenset(m["slot"] for m in mons if _shows_itself(m) or to_id(m["baseSpecies"]) in ids)
    return shown, ids


def _on_board_json(side: dict[str, Any], shown: frozenset[int], onboard: str) -> set[str]:
    out: set[str] = set()
    for mon in side["pokemon"]:
        if mon["slot"] not in shown:
            continue
        out.add(to_id(mon["species"]))
        if onboard == "shown":
            out.add(to_id(mon["baseSpecies"]))
    return out


def _switch_targets(choice: str) -> list[int]:
    """Party indices (1-based) a recorded replacement choice switches into."""
    return [int(part.split()[1]) for part in choice.split(",") if part.strip().startswith("switch")]


def reachable(
    position: dict[str, Any],
    shown: list[frozenset[int]],
    sixes: list[list[str]],
    options: list[list[str]],
    onboard: str,
) -> list[bool]:
    """Per side t: some option of t switches into a slot t has not shown, and some
    completion of t's bench lacks that species (more candidates than unseen slots) -- so
    side 1-t's matrix held a cell the off-by-one moved."""
    out = []
    for t in (0, 1):
        side = position["sides"][t]
        hidden = [m["slot"] for m in side["pokemon"] if m["slot"] not in shown[t]]
        if not hidden:
            out.append(False)
            continue
        board = _on_board_json(side, shown[t], onboard)
        candidates = [s for s in sixes[t] if to_id(s) not in board]
        into_unseen = any(
            index - 1 in hidden for choice in options[t] for index in _switch_targets(choice)
        )
        out.append(bool(into_unseen and len(candidates) > len(hidden)))
    return out


def count_file(args: tuple[str, str, str]) -> dict[str, Any]:
    path, carry, onboard = args
    c: Counter[str] = Counter()
    commits: Counter[str] = Counter()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                c["torn"] += 1
                continue
            finished = g.get("outcome") is not None
            c["games"] += 1
            c["games_finished"] += int(finished)
            commits[str((g.get("engine") or {}).get("commit"))[:9]] += 1
            hidden_game = g.get("information") == "hidden-bench"
            sixes = [g.get("ownSix") or [], g.get("foeSix") or []]
            carried: list[Any] = [frozenset(), frozenset()]
            for d in g["decisions"]:
                kind = d["kind"]
                c[f"dec_{kind}"] += 1
                c["decisions"] += 1
                if finished:
                    c["rows"] += 1
                    c[f"rows_{kind}"] += 1
                if kind not in ("move", "replacement"):
                    continue
                pos = d["position"]
                shown = []
                for t in (0, 1):
                    s, carried[t] = _shown_slots_json(
                        pos["sides"][t], carry, carried[t],
                        (d.get("shownIdentities") or [None, None])[t],
                    )
                    shown.append(s)
                if kind != "replacement" or not hidden_game:
                    continue
                c["rep_hidden"] += 1
                reach = reachable(pos, shown, sixes, [d["ownActions"], d["foeActions"]], onboard)
                # reach[t]: side (1 - t)'s matrix was moved.
                if any(reach):
                    c["rep_hit"] += 1
                    if finished:
                        c["rows_rep_hit"] += 1
                c["rep_hit_own_matrix"] += int(reach[1])
                c["rep_hit_foe_matrix"] += int(reach[0])
    return {"path": path, "counts": dict(c), "commits": dict(commits)}


# ---------------------------------------------------------------------------
# Solve mode
# ---------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _init_worker(value: str | None, format_id: str, threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.regulation import load_regulation

    reg = load_regulation(format_id)
    register_mega_stones(reg)
    _W["reg"] = reg
    _W["leaf"] = None
    if value:
        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        encoder = Encoder(reg)
        net, _meta = load_model(value, encoder)
        _W["leaf"] = BatchedValue(net.to(torch.device("cpu")), encoder, device=torch.device("cpu"))


def _completions(reg: Any, pos: Any, side: int, sheet: Any, seen: frozenset[int],
                 weights: dict | None, onboard: str) -> list[Any]:
    """`hidden.completions`, with the on-board rule of the era (`species` before 43867b8)."""
    from pokeuraou import hidden
    from pokeuraou.selfplay import _make_pokemon

    if onboard == "shown":
        return hidden.completions(reg, pos, side, sheet, seen=seen, weights=weights)
    s = pos.sides[side]
    hid = [m.slot for m in s.pokemon if m.slot not in seen]
    if not hid:
        return [hidden.Completion(pos, (), (), 1.0, exact=True)]
    board = {to_id(m.species) for m in s.pokemon if m.slot in seen}
    candidates = [e for e in sheet if to_id(e.species) not in board]
    if len(candidates) < len(hid):
        raise ValueError(f"{len(hid)} unseen slots, {len(candidates)} candidates")
    made = []
    for choice in combinations(range(len(candidates)), len(hid)):
        picked = [candidates[i] for i in choice]
        key = tuple(sorted(e.species for e in picked))
        made.append(hidden.Completion(
            hidden.substitute(reg, pos, side, hid, picked, _make_pokemon),
            tuple(e.species for e in picked), tuple(hid), float((weights or {}).get(key, 0.0)),
        ))
    total = sum(m.weight for m in made)
    if total <= 0.0:
        return [hidden.Completion(m.position, m.species, m.slots, 1.0 / len(made)) for m in made]
    return [hidden.Completion(m.position, m.species, m.slots, m.weight / total) for m in made]


def _shifted(option: Any) -> Any:
    from pokeuraou.actions import SideAction, SwitchAction

    return SideAction(slots=tuple(
        SwitchAction(slot=a.slot, party_index=a.party_index + 1, species=a.species)
        if isinstance(a, SwitchAction) else a
        for a in option.slots
    ))


def solve_decision(task: dict[str, Any]) -> dict[str, Any]:
    """Re-solves one recorded replacement decision, fixed and shifted."""
    from pokeuraou import port
    from pokeuraou.actions import SideAction, switch_actions_after_faint
    from pokeuraou.equilibrium import EquilibriumError, solve, solve_bayesian
    from pokeuraou.position import Position
    from pokeuraou.selfplay import _pass

    reg, leaf = _W["reg"], _W["leaf"]
    pos = Position.from_json(task["position"])
    out: dict[str, Any] = {k: task[k] for k in ("label", "file", "line", "decision")}
    owed = port.replacements_needed(reg, pos)
    options = []
    for side in range(2):
        must = list(owed[side])
        found = switch_actions_after_faint(reg, pos, side, must) if any(must) else []
        if not found:
            found = [SideAction(slots=tuple(_pass(s) for s in range(len(pos.sides[side].active))))]
        options.append(found)
    got = [[a.to_choice() for a in o] for o in options]
    if got != [task["ownActions"], task["foeActions"]]:
        out["status"] = "options differ"
        out["options"] = got
        return out

    def matrix(at: Any, shifted: bool) -> tuple[np.ndarray, int]:
        rows = [_shifted(a) for a in options[0]] if shifted else options[0]
        cols = [_shifted(b) for b in options[1]] if shifted else options[1]
        after = [port.resolve_replacements(reg, at, [a, b]).position for a in rows for b in cols]
        values = np.asarray(leaf(after), dtype=np.float64)
        keys = [json.dumps(p.to_json(), sort_keys=True) for p in after]
        return values.reshape(len(rows), len(cols)), keys

    def answer(shifted: bool) -> dict[str, Any]:
        if task["definition"] == "true":
            m, _k = matrix(pos, shifted)
            try:
                e = solve(m)
                return {"own": [float(x) for x in e.row_strategy],
                        "foe": [float(x) for x in e.col_strategy],
                        "value": float(e.value), "foeValue": None}
            except EquilibriumError:
                return {"error": "equilibrium"}
        answers = {}
        moved_weight = [0.0, 0.0]
        for side in (0, 1):
            items = task["_spreads"][1 - side]
            built = []
            for k, item in enumerate(items):
                m, keys = matrix(item.position, shifted)
                built.append(m)
                if shifted:
                    fixed_keys = task["_fixed_keys"][1 - side][k]
                    if keys != fixed_keys:
                        moved_weight[side] += item.weight
                else:
                    task["_fixed_keys"][1 - side].append(keys)
            weights = np.asarray([item.weight for item in items], dtype=np.float64)
            solved = solve_bayesian([m if side == 0 else -m.T for m in built], weights)
            answers[side] = ([float(x) for x in solved.row_strategy], float(solved.value))
        res = {"own": answers[0][0], "value": answers[0][1], "foe": answers[1][0],
               "foeValue": -answers[1][1]}
        if shifted:
            res["movedWeight"] = moved_weight
        return res

    try:
        if task["definition"] != "true":
            spreads = {}
            for side in (0, 1):
                prior = task["prior"][side] if task["prior"] else None
                weights = None
                if prior is not None:
                    from pokeuraou.hidden import shown_species

                    seen_species = shown_species(pos, side, task["shown"][side])
                    weights = (prior.weights(seen_species, task["leads"][side])
                               if task["leads"] is not None else prior.weights(seen_species)) or None
                spreads[side] = _completions(reg, pos, side, task["sheets"][side],
                                             task["shown"][side], weights, task["onboard"])
            task["_spreads"] = spreads
            task["_fixed_keys"] = {0: [], 1: []}
            out["completions"] = [len(spreads[0]), len(spreads[1])]
        fixed = answer(False)
        shifted = answer(True)
    except port.PortRefused as refused:
        out["status"] = f"refused: {refused}"
        return out
    except ValueError as problem:
        out["status"] = f"sheet: {problem}"
        return out
    out["status"] = "ok"
    out["fixed"] = fixed
    out["shifted"] = shifted
    out["record"] = {"own": task["ownPolicy"], "foe": task["foePolicy"],
                     "value": task["searchValue"], "foeValue": task.get("foeSearchValue")}
    return out


# --- rebuilding the node's inputs from the record ---------------------------


@dataclass
class Era:
    carry: str
    onboard: str
    prior: str
    leads: bool
    definition: str


def _set_key(entry: Any) -> str:
    from pokeuraou.selfplay import _set_json

    return json.dumps(_set_json(None, entry), sort_keys=True)


class SheetFinder:
    """The two sixes (and bench priors) a recorded game was played with."""

    def __init__(self, args: argparse.Namespace, reg: Any) -> None:
        self.args = args
        self.reg = reg
        self.pool = None
        self.solver = None
        self.roster = None
        self.index: dict[tuple[str, ...], list[tuple[Any, tuple[Any, ...]]]] = {}
        if args.pool:
            from pokeuraou.pool import load_pool

            self.pool = load_pool(args.pool, reg)
            self.team_index = {t.id: i for i, t in enumerate(self.pool.teams)}
        if args.roster:
            from pokeuraou.teams import load_roster

            self.roster = load_roster(args.roster)
        if args.book:
            from pokeuraou.selection_book import SelectionBook

            book = SelectionBook.read(Path(args.book))
            for entry in book.entries.values():
                for sets in entry.class_sets:
                    self.index.setdefault(tuple(s.species for s in sets), []).append((entry, sets))

    def for_game(self, g: dict[str, Any]) -> tuple[tuple[Any, Any], tuple[Any, Any] | None]:
        from pokeuraou.selection_book import BenchPrior

        eps, temp = self.args.epsilon, self.args.temperature
        if self.pool is not None:
            a, b = (self.team_index[t] for t in g["pool"]["teams"])
            six0, six1 = list(self.pool.teams[a].sets), list(self.pool.teams[b].sets)
            prior = None
            if self.args.prior == "pool":
                entry = self._pool_entry(a, b)
                prior = (BenchPrior.of(entry, 0, [s.species for s in six0], epsilon=eps, temperature=temp),
                         BenchPrior.of(entry, 1, [s.species for s in six1], epsilon=eps, temperature=temp))
            return (six0, six1), prior
        assert self.roster is not None
        own = list(self.roster.sets)
        if [s.species for s in own] != g["ownSix"]:
            raise ValueError("own six is not the roster")
        if g["foeSix"] == g["ownSix"] and g.get("foeArchetype") == "mirror":
            return (own, list(own)), None
        want = [json.dumps(t, sort_keys=True) for t in g["foeTeam"]]
        for entry, sets in self.index.get(tuple(g["foeSix"]), []):
            if [_set_key(sets[i]) for i in g["foePick"]] == want:
                prior = None
                if self.args.prior == "book":
                    prior = (
                        BenchPrior.of(entry, 0, [s.species for s in own], epsilon=eps, temperature=temp),
                        BenchPrior.of(entry, 1, [s.species for s in sets], epsilon=eps, temperature=temp),
                    )
                return (own, list(sets)), prior
        raise ValueError("foe six not in the book")

    def _pool_entry(self, a: int, b: int) -> Any:
        from pokeuraou.poolplay import SolvedSelections

        if self.solver is None:
            store = Path(self.args.selection_store)
            tag = json.loads(next(store.glob("*.json")).read_bytes())["tag"]
            self.solver = SolvedSelections(self.reg, self.pool.teams, evaluate=None, store=store, tag=tag)
        return self.solver.entry(a, b)


def _slots_from_json(
    pos: Any, side: int, carry: str, carried: Any, recorded: Any
) -> tuple[frozenset[int], Any]:
    from pokeuraou.hidden import _shows_itself, seen_identities, seen_slots

    if carry == "slots":
        shown = frozenset(carried) | {m.slot for m in pos.sides[side].pokemon if _shows_itself(m)}
        return frozenset(shown), shown
    if carry == "recorded" and recorded is not None:
        ids = frozenset(recorded)
    else:
        ids = seen_identities(pos, side, frozenset(carried))
    return seen_slots(pos, side, ids), ids


def tasks_for(args: argparse.Namespace, era: Era, directory: str, label: str) -> tuple[list[dict], dict]:
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.hidden import shown_species
    from pokeuraou.position import Position
    from pokeuraou.regulation import load_regulation

    files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    per_file = -(-args.games // len(files))
    finder = None
    tasks: list[dict] = []
    skipped: Counter[str] = Counter()
    format_id = None
    for path in files:
        taken = 0
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if taken >= per_file:
                    break
                if not line.strip():
                    continue
                g = json.loads(line)
                if g.get("outcome") is None:
                    continue
                if era.definition == "belief" and g.get("information") != "hidden-bench":
                    continue
                taken += 1
                if format_id is None:
                    format_id = g["decisions"][0]["position"]["format"]
                    reg = load_regulation(format_id)
                    register_mega_stones(reg)
                    finder = SheetFinder(args, reg)
                sheets = prior = None
                if era.definition == "belief":
                    try:
                        sheets, prior = finder.for_game(g)
                    except (ValueError, KeyError) as problem:
                        skipped[f"game: {problem}"] += 1
                        continue
                carried: list[Any] = [frozenset(), frozenset()]
                leads = None
                for number, d in enumerate(g["decisions"]):
                    if d["kind"] not in ("move", "replacement"):
                        continue
                    pos = Position.from_json(d["position"])
                    if leads is None and era.leads and pos.turn == 1:
                        leads = [frozenset(shown_species(pos, i, frozenset(
                            m.slot for m in pos.sides[i].pokemon if m.active_index is not None)))
                            for i in (0, 1)]
                    shown = []
                    for t in (0, 1):
                        s, carried[t] = _slots_from_json(
                            pos, t, era.carry, carried[t], (d.get("shownIdentities") or [None, None])[t])
                        shown.append(s)
                    if d["kind"] != "replacement":
                        continue
                    sixes = [g.get("ownSix") or [], g.get("foeSix") or []]
                    menus = [d["ownActions"], d["foeActions"]]
                    reach = reachable(d["position"], shown, sixes, menus, era.onboard)
                    if args.only_hit and not any(reach):
                        skipped["not reachable"] += 1
                        continue
                    tasks.append({
                        "label": label, "file": os.path.basename(path), "line": line_no,
                        "decision": number, "position": d["position"],
                        "ownActions": d["ownActions"], "foeActions": d["foeActions"],
                        "ownPolicy": d["ownPolicy"], "foePolicy": d["foePolicy"],
                        "searchValue": d["searchValue"], "foeSearchValue": d.get("foeSearchValue"),
                        "sheets": sheets, "prior": prior, "shown": shown,
                        "leads": leads if era.leads else None,
                        "onboard": era.onboard, "definition": era.definition,
                        "reach": reach,
                    })
    return tasks, {"skipped": dict(skipped), "format": format_id}


# --- comparison --------------------------------------------------------------


def tv(p: list[float], q: list[float]) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def modal(p: list[float]) -> int | None:
    a = np.asarray(p)
    order = np.argsort(-a, kind="stable")
    if len(a) > 1 and a[order[0]] - a[order[1]] < 1e-6:
        return None
    return int(order[0])


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {"tvOwn": tv(a["own"], b["own"]), "tvFoe": tv(a["foe"], b["foe"]),
           "dValue": abs(a["value"] - b["value"])}
    for side in ("own", "foe"):
        ma, mb = modal(a[side]), modal(b[side])
        out[f"modal_{side}"] = "tie" if ma is None or mb is None else ("changed" if ma != mb else "same")
    if a.get("foeValue") is not None and b.get("foeValue") is not None:
        out["dFoeValue"] = abs(a["foeValue"] - b["foeValue"])
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    out: dict[str, Any] = {"decisions": len(rows), "ok": len(ok),
                           "status": dict(Counter(r.get("status", "?").split(":")[0] for r in rows))}
    pairs = {"record~fixed": ("record", "fixed"), "record~shifted": ("record", "shifted"),
             "fixed~shifted": ("fixed", "shifted")}
    for name, (x, y) in pairs.items():
        comps = [compare(r[x], r[y]) for r in ok if "error" not in r[x] and "error" not in r[y]]
        if not comps:
            continue
        s: dict[str, Any] = {"n": len(comps)}
        for key in ("tvOwn", "tvFoe", "dValue", "dFoeValue"):
            vals = np.array([c[key] for c in comps if key in c], dtype=np.float64)
            if not len(vals):
                continue
            s[key] = {"mean": float(vals.mean()), "median": float(np.median(vals)),
                      "p90": float(np.quantile(vals, 0.9)), "max": float(vals.max()),
                      "exact": int((vals == 0).sum()), "gt1e-6": int((vals > 1e-6).sum()),
                      "gt0.01": int((vals > 0.01).sum()), "gt0.1": int((vals > 0.1).sum())}
        for side in ("own", "foe"):
            s[f"modal_{side}"] = dict(Counter(c[f"modal_{side}"] for c in comps))
        out[name] = s
    moved = [r["shifted"].get("movedWeight") for r in ok if r["shifted"].get("movedWeight")]
    if moved:
        arr = np.array(moved, dtype=np.float64)
        out["movedWeight"] = {"own_matrix_mean": float(arr[:, 0].mean()),
                              "foe_matrix_mean": float(arr[:, 1].mean()),
                              "any_moved": int((arr.max(axis=1) > 0).sum())}
    return out


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("count", "solve"))
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--carry", choices=("slots", "identities", "recorded"), default="recorded")
    ap.add_argument("--onboard", choices=("species", "shown"), default="shown")
    ap.add_argument("--prior", choices=("none", "book", "pool"), default="none")
    ap.add_argument("--leads", action="store_true", help="condition the bench prior on the leads (IKA-118)")
    ap.add_argument("--definition", choices=("belief", "true"), default="belief")
    ap.add_argument("--value", help="the leaf the records were generated with (.pt)")
    ap.add_argument("--book", help="selection book (M-B): finds the foe six and its prior")
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--pool", help="pool file (M-C)")
    ap.add_argument("--selection-store", help="the run's selection-solved directory (M-C)")
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--games", type=int, default=40, help="games per directory (solve)")
    ap.add_argument("--only-hit", action="store_true", help="solve only decisions `count` calls reachable")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", help="per-decision rows (solve)")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()

    import pokeuraou

    print(f"pokeuraou from {pokeuraou.__file__}", file=sys.stderr)
    started = time.perf_counter()
    if args.mode == "count":
        jobs = [(p, args.carry, args.onboard) for d in args.dirs
                for p in sorted(glob.glob(os.path.join(d, "*.jsonl")))]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(count_file, jobs))
        for d in args.dirs:
            total: Counter[str] = Counter()
            commits: Counter[str] = Counter()
            for r in results:
                if os.path.dirname(os.path.abspath(r["path"])) == os.path.abspath(d):
                    total.update(r["counts"])
                    commits.update(r["commits"])
            print(json.dumps({"dir": d, "carry": args.carry, "onboard": args.onboard,
                              "counts": dict(sorted(total.items())), "commits": dict(commits)},
                             ensure_ascii=False))
        print(f"wall {time.perf_counter() - started:.1f}s", file=sys.stderr)
        return

    era = Era(args.carry, args.onboard, args.prior, args.leads, args.definition)
    rows: list[dict[str, Any]] = []
    for d in args.dirs:
        label = args.label or os.path.basename(os.path.normpath(d))
        tasks, info = tasks_for(args, era, d, label)
        print(f"{label}: {len(tasks)} replacement decisions to solve, {info}", file=sys.stderr)
        if not tasks:
            continue
        with ProcessPoolExecutor(
            max_workers=args.jobs, initializer=_init_worker,
            initargs=(args.value, info["format"], args.torch_threads),
        ) as ex:
            rows.extend(ex.map(solve_decision, tasks, chunksize=4))
    if args.out:
        with open(args.out, "wb") as handle:
            for r in rows:
                handle.write((json.dumps(_clean(r), ensure_ascii=False) + "\n").encode("utf-8"))
    summary = summarise(rows)
    summary["era"] = vars(era)
    summary["dirs"] = args.dirs
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"wall {time.perf_counter() - started:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
