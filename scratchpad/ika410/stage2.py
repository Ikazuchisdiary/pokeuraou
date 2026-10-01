"""IKA-410 stage 2: replay chosen generation games and split each example's value change.

    python stage2.py <examples.jsonl> <games.jsonl> <shard> <shards> <out.jsonl>

For every chosen (game, seat, d0, d1) row: replays the game with the generation's settings
(gen-3: seed 40002, value-mc2 x1, width 12, hidden bench, q-nocover with q-mc2, the solved
selection store), checks the replayed record against the original one, and then splits the
seat's change of value from the move decision d0 to the next move decision d1, in the seat's
own win probability, into a telescoping chain:

  V0   recorded value at d0 = sum_k w_k v_k, v_k = x G_k y_k per completion k of the foe's bench
  A    sum_k w'_k v_k, w' = the d0 belief conditioned on what the foe showed by d1   a = A - V0
  VT   v_true, the true completion's value                                         a2 = VT - A
  B1   x G_t[:, j], the foe's actual action j                                      b_foe = B1 - VT
  B2   G_t[i, j], the seat's own actual action i                                   b_own = B2 - B1
  CE   the exact cell (Budget.exact, every branch and pause)                       c_budget = CE - B2
  C    leaf of the drawn branch (when the drawn outcome is a branch, not a pause)  c = C - CE
  D    the d1 search value in the true completion, x' G'_t y'_t                    d = D - C
  V1   recorded value at d1                                                        e = V1 - D

and d_node = D - leaf(d1's position) (the leaf at the d1 node itself against its own search).
The same d1 node is solved again from the captured spreads with _menus + belief_solve, with
value-mc2 (control: it must give the captured D) and with value-mc3 (D3, d_node3, d3).
"""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from pokeuraou import poolplay, qrank, rustnode, selfplay
from pokeuraou.actions import side_actions
from pokeuraou.beliefnode import belief_payoffs
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.hidden import Completion
from pokeuraou.pool import load_pool
from pokeuraou.position import Position
from pokeuraou.search import belief_solve
from pokeuraou.value import BatchedValue, load_model

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
TOL = 1e-4

examples = [json.loads(x) for x in Path(sys.argv[1]).read_bytes().splitlines() if x.strip()]
games_raw = {}
for ln in Path(sys.argv[2]).read_bytes().splitlines():
    if ln.strip():
        key, body = ln.split(b"\t", 1)
        k = json.loads(key)
        games_raw[(k["file"], k["line"])] = body
shard, shards = int(sys.argv[3]), int(sys.argv[4])
out_path = Path(sys.argv[5])
torch.set_num_threads(1)

pool = load_pool(str(M / "data/pool/regmc-matchupweb.json"))
reg = pool.reg
register_mega_stones(reg)
encoder = Encoder(reg)
net2, _ = load_model(M / "data/models/value-mc2.pt", encoder)
leaf2 = BatchedValue(net2.to("cpu"), encoder, device=torch.device("cpu"))
MC3 = not os.environ.get("IKA410_NO_MC3")
if MC3:
    net3, _ = load_model(M / "data/models/value-mc3.pt", encoder)
    leaf3 = BatchedValue(net3.to("cpu"), encoder, device=torch.device("cpu"))
qmodel = qrank.LocalQ(M / "data/models/q-mc2.pt", encoder, device="cpu")
qrank.install(qmodel, "")
q_record = qrank.record_fields({"": qmodel})
rustnode.hold_positions()
store = Path(f"C:/tmp/ika410/store-{shard}")

captured: list[dict] = []
advances: list[dict] = []
real_belief_solve = selfplay.belief_solve
real_advance_turn = selfplay._advance_turn


def spy_belief_solve(reg_, pos, ours, theirs, spreads, evaluators, **kw):
    answers = real_belief_solve(reg_, pos, ours, theirs, spreads, evaluators, **kw)
    captured.append({"turn": pos.turn, "position": pos, "spreads": spreads, "answers": answers,
                     "ours": list(ours), "theirs": list(theirs)})
    return answers


def spy_advance_turn(reg_, rng, pos, chosen, record, leaves, objective, *, hidden=None):
    entry = {"turn": pos.turn, "position": pos, "chosen": [a.to_choice() for a in chosen],
             "decisions": len(record.decisions)}
    advances.append(entry)
    got = real_advance_turn(reg_, rng, pos, chosen, record, leaves, objective, hidden=hidden)
    entry["landed"] = got
    return got


selfplay.belief_solve = spy_belief_solve
selfplay._advance_turn = spy_advance_turn

# The generation ranked the menus with the leaf on the GPU; on the CPU near-tied candidates can
# come out in another order, and the action is drawn by its index in the menu, so the same
# mixture then draws another action. Where a move node's menus are the recorded SETS, put them
# in the recorded ORDER (the record still has to agree on sets, mixtures and values).
recorded_menus: dict[tuple, tuple[list[str], list[str]]] = {}
reordered = [0, 0]  # nodes put back in the recorded order, nodes already in it
real_menus = selfplay._menus


def spy_menus(reg_, pos, *args, **kw):
    ours, theirs = real_menus(reg_, pos, *args, **kw)
    want = recorded_menus.get((pos.turn, json.dumps(pos.to_json(), sort_keys=True)))
    if want is None:
        return ours, theirs
    out = []
    for got, names in zip((ours, theirs), want):
        by = {a.to_choice(): a for a in got}
        if set(by) != set(names) or len(by) != len(got):
            return ours, theirs
        out.append([by[n] for n in names])
    same = [a.to_choice() for a in ours] == want[0] and [a.to_choice() for a in theirs] == want[1]
    reordered[1 if same else 0] += 1
    return out[0], out[1]


selfplay._menus = spy_menus


def seat_rec(d: dict, seat: int) -> float:
    v = float(d["searchValue"])
    if seat == 0:
        return v
    f = d.get("foeSearchValue")
    return 1.0 - float(v if f is None else f)


def policy_map(actions, policy) -> dict:
    return {a: float(p) for a, p in zip(actions, policy)}


def compare(orig: dict, rep: dict) -> tuple[int | None, float]:
    """First decision index that differs (None when all agree), and the largest value/policy gap."""
    od, rd = orig["decisions"], rep["decisions"]
    worst = 0.0
    for n in range(max(len(od), len(rd))):
        if n >= len(od) or n >= len(rd):
            return n, worst
        a, b = od[n], rd[n]
        if a["kind"] != b["kind"] or a["turn"] != b["turn"]:
            return n, worst
        if a.get("ownChosen") != b.get("ownChosen") or a.get("foeChosen") != b.get("foeChosen"):
            return n, worst
        for key in ("searchValue", "foeSearchValue"):
            if (a.get(key) is None) != (b.get(key) is None):
                return n, worst
            if a.get(key) is not None:
                gap = abs(float(a[key]) - float(b[key]))
                worst = max(worst, gap)
                if gap > TOL:
                    return n, worst
        for acts, pol in (("ownActions", "ownPolicy"), ("foeActions", "foePolicy")):
            if acts not in a:
                continue
            pa, pb = policy_map(a[acts], a[pol]), policy_map(b[acts], b[pol])
            if set(pa) != set(pb):
                return n, worst
            gap = max((abs(pa[k] - pb[k]) for k in pa), default=0.0)
            worst = max(worst, gap)
            if gap > TOL:
                return n, worst
    if orig.get("outcome") != rep.get("outcome"):
        return len(od), worst
    return None, worst


def seat_matrices(answer, seat: int) -> list[np.ndarray]:
    row, col, built, weights = answer.node_payoff
    return [np.asarray(m) if seat == 0 else 1.0 - np.asarray(m).T for m in built]


def seat_lists(answer, seat: int):
    row, col, _built, _w = answer.node_payoff
    rows, cols = (row, col) if seat == 0 else (col, row)
    return [a.to_choice() for a in rows], [a.to_choice() for a in cols]


def true_index(items, real: Position, foe: int) -> int:
    want = [p.species for p in real.sides[foe].pokemon]
    hits = [k for k, it in enumerate(items) if [p.species for p in it.position.sides[foe].pokemon] == want]
    if len(hits) != 1:
        raise RuntimeError(f"true completion: {len(hits)} matches")
    return hits[0]


def leaf_seat(leaf, positions: list[Position], seat: int) -> np.ndarray:
    v = np.asarray(leaf(positions), dtype=np.float64)
    return v if seat == 0 else 1.0 - v


def action(pos: Position, side: int, choice: str):
    for a in side_actions(reg, pos, side):
        if a.to_choice() == choice:
            return a
    raise ValueError(choice)


def resolve_d1(cap: dict, seat: int, leaf) -> dict:
    """The d1 move node solved again from its captured spreads, generation-style, with `leaf`."""
    pos, spreads = cap["position"], cap["spreads"]
    ours, theirs = real_menus(reg, pos, (12, 12), leaf, Budget.matrix(), True, None, spreads,
                                   rank_view="heaviest", used={}, rank_fill="q-nocover")
    got = belief_solve(reg, pos, ours, theirs, spreads, {0: leaf, 1: leaf},
                       budget=Budget.matrix(), sides=(seat,))
    ans = got[seat]
    items = spreads[1 - seat]
    t = true_index(items, pos, 1 - seat)
    G = seat_matrices(ans, seat)
    x = np.asarray(ans.strategy)
    values = [float(x @ G[k] @ np.asarray(ans.replies[k])) for k in range(len(G))]
    w = np.asarray([it.weight for it in items])
    return {"D": values[t], "V": float(w @ values / w.sum()),
            "same_menu_as_capture": ({a.to_choice() for a in ours} == {a.to_choice() for a in cap["ours"]}
                                     and {a.to_choice() for a in theirs} == {a.to_choice() for a in cap["theirs"]})}


def analyse(ex: dict, orig: dict, moves: list[int]) -> dict:
    seat, foe = ex["seat"], 1 - ex["seat"]
    k0, k1 = moves.index(ex["d0"]), moves.index(ex["d1"])
    c0, c1 = captured[k0], captured[k1]
    dec0, dec1 = orig["decisions"][ex["d0"]], orig["decisions"][ex["d1"]]
    out: dict = {}
    # T: per completion
    a0 = c0["answers"][seat]
    items0 = c0["spreads"][foe]
    G0 = seat_matrices(a0, seat)
    x0 = np.asarray(a0.strategy)
    v0 = np.array([float(x0 @ G0[k] @ np.asarray(a0.replies[k])) for k in range(len(G0))])
    w0 = np.asarray([it.weight for it in items0], dtype=np.float64)
    w0 = w0 / w0.sum()
    t0 = true_index(items0, c0["position"], foe)
    # d1
    a1 = c1["answers"][seat]
    items1 = c1["spreads"][foe]
    G1 = seat_matrices(a1, seat)
    x1 = np.asarray(a1.strategy)
    v1 = np.array([float(x1 @ G1[k] @ np.asarray(a1.replies[k])) for k in range(len(G1))])
    w1 = np.asarray([it.weight for it in items1], dtype=np.float64)
    w1 = w1 / w1.sum()
    t1 = true_index(items1, c1["position"], foe)
    # conditioning of the d0 belief on what was shown by d1
    newly = set(items0[t0].species) - set(items1[t1].species)
    consistent = np.array([newly <= set(it.species) for it in items0])
    wc = np.where(consistent, w0, 0.0)
    wc = wc / wc.sum()
    # check: the conditioned d0 belief mapped onto the d1 items equals the d1 belief
    mapped: dict[tuple, float] = {}
    for k, it in enumerate(items0):
        if consistent[k]:
            key = tuple(sorted(set(it.species) - newly))
            mapped[key] = mapped.get(key, 0.0) + wc[k]
    d1w = {tuple(sorted(it.species)): w1[k] for k, it in enumerate(items1)}
    out["belief_check"] = max(abs(mapped.get(key, 0.0) - d1w.get(key, 0.0)) for key in set(mapped) | set(d1w))
    rows, cols = seat_lists(a0, seat)
    adv = next(a for a in advances if a["turn"] == dec0["turn"] and a["decisions"] > ex["d0"])
    own_c, foe_c = adv["chosen"][seat], adv["chosen"][foe]
    i, j = rows.index(own_c), cols.index(foe_c)
    Gt = G0[t0]
    yt = np.asarray(a0.replies[t0])
    V0 = float(w0 @ v0)
    A = float(wc @ v0)
    VT = float(v0[t0])
    B1 = float(x0 @ Gt[:, j])
    B2 = float(Gt[i, j])
    pos0 = c0["position"]
    acts = [action(pos0, 0, adv["chosen"][0]), action(pos0, 1, adv["chosen"][1])]
    exact_spreads = {s: [Completion(pos0, (), (), 1.0, exact=True)] for s in (0, 1)}
    node = belief_payoffs(reg, pos0, [acts[0]], [acts[1]], leaf2, budget=Budget.exact(), spreads=exact_spreads)
    cell = float(np.asarray(node.matrices[0][0])[0, 0])
    CE = cell if seat == 0 else 1.0 - cell
    D = float(v1[t1])
    V1 = float(w1 @ v1)
    pos1 = c1["position"]
    L1 = float(leaf_seat(leaf2, [pos1], seat)[0])
    # the drawn outcome: a branch of the exact turn, or a pause
    from pokeuraou import port
    turn = port.turn(reg, pos0, acts, Budget.exact(), full=True, events=False)
    landed = adv["landed"]
    drawn = None
    if landed is not None:
        lj = landed.to_json()
        for n, b in enumerate(turn.outcomes):
            if b.position.to_json() == lj:
                drawn = n
                break
    total = sum(turn.branches) + sum(turn.suspended)
    out.update({
        "V0_rec": seat_rec(dec0, seat), "V1_rec": seat_rec(dec1, seat),
        "V0": V0, "A": A, "VT": VT, "B1": B1, "B2": B2, "CE": CE, "D": D, "V1": V1, "L1": L1,
        "x_own_chosen": float(x0[i]), "y_foe_chosen": float(yt[j]),
        "completions0": len(items0), "completions1": len(items1), "w_true0": float(w0[t0]),
        "newly_shown": sorted(newly), "branches": len(turn.outcomes),
        "pause_mass": float(sum(turn.suspended) / total) if total else 0.0,
        "drawn_found": drawn is not None,
        "between": [d["kind"] for d in orig["decisions"][ex["d0"] + 1: ex["d1"]]],
    })
    if drawn is not None:
        C = float(leaf_seat(leaf2, [turn.outcomes[drawn].position], seat)[0])
        out["C"] = C
        out["p_drawn"] = float(turn.branches[drawn] / total)
        if MC3:
            out["C3"] = float(leaf_seat(leaf3, [turn.outcomes[drawn].position], seat)[0])
    # value-mc2 again (control) and value-mc3 at d1
    r2 = resolve_d1(c1, seat, leaf2)
    r3 = resolve_d1(c1, seat, leaf3) if MC3 else None
    out.update({"D_resolve2": r2["D"], "V1_resolve2": r2["V"], "menu_same2": r2["same_menu_as_capture"]})
    if MC3:
        out.update({"D3": r3["D"], "V1_3": r3["V"], "L1_3": float(leaf_seat(leaf3, [pos1], seat)[0])})
    return out


def main() -> None:
    by_game: dict[tuple, list[dict]] = {}
    for ex in examples:
        by_game.setdefault((ex["file"], ex["line"]), []).append(ex)
    keys = sorted(by_game)[shard::shards]
    with out_path.open("ab") as f:
        for key in keys:
            orig = json.loads(games_raw[key])
            captured.clear()
            recorded_menus.clear()
            reordered[:] = [0, 0]
            for d in orig["decisions"]:
                if d["kind"] == "move":
                    recorded_menus[(d["turn"], json.dumps(d["position"], sort_keys=True))] = (
                        d["ownActions"], d["foeActions"])
            advances.clear()
            started = time.perf_counter()
            row_base = {"file": key[0], "line": key[1], "game": orig["gameIndex"]}
            try:
                with tempfile.TemporaryDirectory(dir="C:/tmp/ika410/tmp") as tmp:
                    rec_path = Path(tmp) / "g.jsonl"
                    poolplay.generate_pool(
                        reg, pool, games=1, hide_bench=True, seed=40002, out=rec_path,
                        selection=poolplay.SOLVED, evaluate=leaf2, memo=True, store=store,
                        leaf="value:value-mc2", search_limit=12, rank_by_leaf=True,
                        rank_fill="q-nocover", indices=[orig["gameIndex"]], q_record=q_record,
                    )
                    rep = json.loads(rec_path.read_bytes().splitlines()[0])
                first_diff, worst = compare(orig, rep)
                moves = [n for n, d in enumerate(orig["decisions"]) if d["kind"] == "move"]
                cap_full = len(captured) == len(moves) and all(
                    captured[q]["turn"] == orig["decisions"][n]["turn"] for q, n in enumerate(moves))
            except Exception:
                for ex in by_game[key]:
                    f.write(json.dumps({**row_base, "ex": ex, "error": traceback.format_exc()}).encode() + b"\n")
                f.flush()
                continue
            replay_s = time.perf_counter() - started
            for ex in by_game[key]:
                k1 = moves.index(ex["d1"])
                cap_ok = len(captured) > k1 and all(
                    captured[q]["turn"] == orig["decisions"][moves[q]]["turn"] for q in range(k1 + 1))
                row = {**row_base, "ex": ex, "first_diff": first_diff, "worst_gap": worst,
                       "captures_ok": cap_ok, "captures_full": cap_full, "replay_s": replay_s,
                       "reordered": list(reordered)}
                matched = (first_diff is None or first_diff > ex["d1"]) and cap_ok
                row["matched"] = matched
                if matched:
                    try:
                        row.update(analyse(ex, orig, moves))
                    except Exception:
                        row["error"] = traceback.format_exc()
                f.write(json.dumps(row).encode() + b"\n")
                f.flush()
            print(key, first_diff, f"{worst:.1e}", f"{time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)


main()
