"""Where a read's loss comes from, on saved position sets (IKA-394, E0: no new reading).

Every number is a win-rate loss in a reference game: the reference's value minus what the
answer's mixture guarantees there (`position_set._best_got`; a hidden position's game is
Bayesian: side 1 knows its bench). The references and the answers are the ones IKA-367,
369 and 376 saved; this tool only re-solves and re-scores them.

    python tools/error_decompose.py hidden --set SET_HIDDEN --out DIR
    python tools/error_decompose.py open --set SET_OPEN --sweep SWEEP_DIR --out DIR

* ``hidden`` (needs ``ref-r24`` with the completions' matrices, ``positions.json``):
  - the true bench: is the recorded position's bench among the completions
    (species and full sets)? -- said first, since the rest of the section leans on it;
  - (c1) fewer completions: the answer solved on a subset of the completions, scored in the
    game over all of them;
  - (c2) information: the value of seeing the bench (`sum_k w_k V_k - V`), the loss of the
    hidden answer in the true bench's game, and that loss as the belief puts more weight on
    the true bench (the weight 1 case is the open game's own answer: loss 0, a control);
  - depth 1 read and (if ``ref-deep``) the depth-2 answer scored in the deeper game.
* ``open`` (needs ``ref-r24``, ``ref-R4A@2``, ``ref-R4D5@3``, and a sweep of a ladder read
  with ``rungScores``):
  - (a) the ladder's rungs scored in the deepest reference, by depth of the rung (with the
    stored loss recomputed as a control);
  - (d) the depth-4 answer scored in references with wider roots / children / depth 5;
  - (e) substituting the deeper reference's values into the depth-2 matrix for a few cells
    only -- a ladder-shaped rectangle from the depth-2 answer, the cells whose depth-2
    value is nearest the game value, and the same number of cells at random -- solved and
    scored in the deeper reference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

from pokeuraou.equilibrium import solve, solve_bayesian


def _guarantee(x: np.ndarray, mats: list[np.ndarray], w: np.ndarray) -> float:
    return float(sum(wk * float((x @ m).min()) for wk, m in zip(w, mats, strict=True)))


def _value(mats: list[np.ndarray], w: np.ndarray) -> tuple[float, np.ndarray]:
    if len(mats) == 1:
        eq = solve(mats[0])
        return float(eq.value), eq.row_strategy
    eq = solve_bayesian(mats, w)
    return float(eq.value), eq.row_strategy


def _mean_se(v) -> tuple[float, float]:  # noqa: ANN001
    a = np.asarray(v, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def _row(label: str, v, extra: str = "") -> str:  # noqa: ANN001
    a = np.asarray(v, dtype=np.float64)
    m, se = _mean_se(a)
    return (f"  {label:<44} n={len(a):>3}  {m:.4f} (se {se:.4f})  median {np.median(a):.4f}  "
            f"max {a.max():.4f}  >0.005: {int((a > 0.005).sum()):>2}{extra}")


def _write(path: Path, payload) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=1) + "\n").encode("utf-8"))


def _turn_group(turn: int) -> str:
    return "turn2-3" if turn <= 3 else ("turn4-5" if turn <= 5 else "turn6+")


# ---------------------------------------------------------------------------- hidden


def _bench_key(pokemon: list[dict], slots: list[int]) -> str:
    return json.dumps([(q["species"], q["item"], q["nature"], sorted(q["sp"].items()),
                        [m["id"] for m in q["moves"]]) for q in pokemon if q["slot"] in slots],
                      sort_keys=True)


def hidden(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915 - one pass over the set
    set_dir = Path(args.set)
    data = json.loads((set_dir / "positions.json").read_bytes())
    rng = np.random.default_rng(args.seed)
    rows = []
    found_true = 0
    shuffled_true = 0
    for n, entry in enumerate(data["positions"]):
        path = set_dir / "ref-r24" / f"{n}.npz"
        if not path.exists():
            continue
        ref = np.load(path)
        d2 = np.asarray(ref["d2"], dtype=np.float64)
        d1 = np.asarray(ref["d1"], dtype=np.float64)
        w = np.asarray(ref["weights"], dtype=np.float64)
        w = w / w.sum()
        k_all = len(w)
        mats = [d2[k] for k in range(k_all)]
        # The true bench among the completions.
        completions = entry["spreads"]["1"]
        slots = completions[0]["slots"]
        key = _bench_key(entry["position"]["sides"][1]["pokemon"], slots)
        keys = [_bench_key(c["position"]["sides"][1]["pokemon"], slots) for c in completions]
        true = [i for i, c in enumerate(keys) if c == key]
        other = data["positions"][(n + 1) % len(data["positions"])]
        other_slots = other["spreads"]["1"][0]["slots"]
        other_key = _bench_key(other["position"]["sides"][1]["pokemon"], other_slots)
        if other_key in keys:
            shuffled_true += 1
        found_true += bool(true)
        t = true[0] if true else None
        v_full, x_full = _value(mats, w)
        v_worlds = [float(solve(m).value) for m in mats]
        evpi = float(np.dot(w, v_worlds) - v_full)
        rec = {"n": n, "turn": entry["turn"], "K": k_all, "trueIn": t is not None,
               "V": v_full, "evpi": evpi}
        # (c1) subsets of the completions.
        c1 = {}
        for m in range(1, k_all):
            subsets = list(itertools.combinations(range(k_all), m))
            if len(subsets) > args.max_subsets:
                pick = rng.choice(len(subsets), size=args.max_subsets, replace=False)
                subsets = [subsets[i] for i in pick]
            losses = []
            for s in subsets:
                ws = w[list(s)] / w[list(s)].sum()
                _v, xs = _value([mats[i] for i in s], ws)
                losses.append(v_full - _guarantee(xs, mats, w))
            c1[m] = float(np.mean(losses))
        rec["c1"] = c1
        # (c2) the true bench's game, with the belief sharpening toward it.
        if t is not None:
            sharpen = {}
            for q in (1.0 / k_all, 0.25, 0.5, 0.75, 0.9, 1.0):
                if q < 1.0 / k_all - 1e-12:
                    continue
                wq = np.full(k_all, (1.0 - q) / (k_all - 1)) if k_all > 1 else np.ones(1)
                wq[t] = q
                _vq, xq = _value(mats, wq) if q < 1.0 else _value([mats[t]], np.ones(1))
                sharpen[f"{q:.3f}"] = v_worlds[t] - float((xq @ mats[t]).min())
            rec["trueLoss"] = sharpen
            rec["trueLossPrior"] = v_worlds[t] - float((x_full @ mats[t]).min())
        # depth 1 read of the same menus, scored in the depth-2 game.
        _v1, x1 = _value([d1[k] for k in range(k_all)], w)
        rec["d1Loss"] = v_full - _guarantee(x1, mats, w)
        deep = set_dir / "ref-deep" / f"{n}.npz"
        if deep.exists():
            dd = np.load(deep)
            dm = [np.asarray(dd["d2"], dtype=np.float64)[k] for k in range(k_all)]
            vd, _xd = _value(dm, w)
            rec["deepLoss"] = vd - _guarantee(x_full, dm, w)
            rec["deepGap"] = vd - v_full
        rows.append(rec)
        print(f"hidden {n}: K={k_all} evpi {evpi:.4f} c1 {c1}", file=sys.stderr, flush=True)
    _write(Path(args.out) / "hidden.json", rows)
    lines = [f"HIDDEN set {set_dir}: {len(rows)} positions",
             f"  the true bench is among the completions (species, items, natures, sp, moves): "
             f"{found_true}/{len(rows)}; control (the next position's bench looked up here): "
             f"{shuffled_true}/{len(rows)}"]
    for label, sel in (("all", rows), ("K=3", [r for r in rows if r["K"] == 3]),
                       ("K=6", [r for r in rows if r["K"] == 6])):
        if not sel:
            continue
        lines.append(f"[{label}: {len(sel)} positions]")
        lines.append(_row("(c2) value of seeing the bench (evpi)", [r["evpi"] for r in sel]))
        tl = [r["trueLossPrior"] for r in sel if "trueLossPrior" in r]
        lines.append(_row("(c2) loss in the true bench's game (prior belief)", tl))
        for q in ("0.250", "0.500", "0.750", "0.900", "1.000"):
            v = [r["trueLoss"][q] for r in sel if "trueLoss" in r and q in r["trueLoss"]]
            if v:
                lines.append(_row(f"     belief weight {q} on the true bench", v))
        ks = sorted({m for r in sel for m in r["c1"]})
        for m in ks:
            v = [r["c1"][m] for r in sel if m in r["c1"]]
            lines.append(_row(f"(c1) answer from {m} of the completions", v))
        lines.append(_row("depth-1 read scored in the depth-2 game", [r["d1Loss"] for r in sel]))
        dl = [r["deepLoss"] for r in sel if "deepLoss" in r]
        if dl:
            lines.append(_row("depth-2 (r24) answer in the deep reference", dl))
    for grp in ("turn2-3", "turn4-5", "turn6+"):
        sel = [r for r in rows if _turn_group(r["turn"]) == grp]
        if sel:
            lines.append(f"[{grp}: {len(sel)} positions]")
            lines.append(_row("(c2) evpi", [r["evpi"] for r in sel]))
            lines.append(_row("(c1) answer from 1 completion", [r["c1"][1] for r in sel]))
    text = "\n".join(lines) + "\n"
    (Path(args.out) / "hidden.txt").write_bytes(text.encode("utf-8"))
    print(text)


# ---------------------------------------------------------------------------- open


def _load(set_dir: Path, name: str, n: int) -> np.ndarray | None:
    path = set_dir / f"ref-{name}" / f"{n}.npz"
    return np.asarray(np.load(path)["d2"], dtype=np.float64) if path.exists() else None


def _support(strategy: np.ndarray, k: int) -> list[int]:
    live = np.flatnonzero(np.asarray(strategy) > 1e-6)
    order = live[np.argsort(-np.asarray(strategy)[live], kind="stable")]
    return [int(i) for i in order[:k]]


def _rectangle(strategy: np.ndarray, ev: np.ndarray, count: int, *, larger: bool) -> list[int]:
    """The support (heaviest first), then the actions the opponent's answer does best against."""
    chosen = _support(strategy, count)
    order = np.argsort(-ev if larger else ev, kind="stable")
    for i in order:
        if len(chosen) >= count:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return chosen


def _score(x: np.ndarray, m: np.ndarray) -> float:
    return float(solve(m).value) - float((x @ m).min())


def open_set(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915 - one pass over the set
    set_dir = Path(args.set)
    positions = json.loads((set_dir / "positions.json").read_bytes())["positions"]
    sweep = Path(args.sweep)
    rng = np.random.default_rng(args.seed)
    out_rows = []
    lines = [f"OPEN set {set_dir}, ladder read {sweep}"]
    depth_rows: dict[str, list[float]] = {}
    per_pos = []
    for f in sorted(sweep.glob("*.json"), key=lambda p: int(p.stem)):
        n = int(f.stem)
        row = json.loads(f.read_bytes())
        deep = _load(set_dir, "R4D5@3", n)
        r24 = _load(set_dir, "r24", n)
        if deep is None or r24 is None:
            continue
        turn = positions[n].get("turn", 0)
        best = float(solve(deep).value)
        # (a) the rungs, recomputed; the stored loss as a control.
        drift = 0.0
        by_depth: dict[int, float] = {}
        first = None
        for g in row["rungScores"]:
            x = np.asarray(g["x"], dtype=np.float64)
            loss = best - float((x @ deep).min())
            if "d2-R4D5" in g:
                drift = max(drift, abs(loss - g["d2-R4D5"]["loss"]))
            m = re.match(r"d(\d+)", g["stage"])
            depth = int(m.group(1)) if m else 1
            if first is None:
                first = loss
            by_depth[depth] = loss
        depth_last = {d: by_depth[d] for d in sorted(by_depth)}
        per_pos.append({"n": n, "turn": turn, "byDepth": depth_last, "storedDrift": drift})
        for d, v in depth_last.items():
            depth_rows.setdefault(f"depth {d} (last rung of that depth)", []).append(v)
        # (d) the depth-4 answer (R4A@2's equilibrium) in wider references.
        parts = set_dir / "ref-R4A" / "parts" / f"{n}.json"
        wide = {}
        if parts.exists():
            # The answer the ladder gave at the end of stage 2 (the rectangle's game solved,
            # not the whole matrix's: a tie in the matrix would pick another optimum).
            x4 = np.asarray(json.loads(parts.read_bytes())["rungs"][-1]["x"], dtype=np.float64)
            for name in ("R4S@3", "R4W@3", "R4D5@3"):
                m = _load(set_dir, name, n)
                if m is not None:
                    wide[name] = _score(x4, m)
        # (e) the deep values in a few cells only.
        x0 = solve(r24).row_strategy
        eq0 = solve(r24)
        v0 = float(eq0.value)
        diff = np.abs(deep - r24) > 1e-9
        cells = np.argwhere(diff)
        rec_e = {"cells": int(diff.sum()), "size": list(r24.shape),
                 "none": _score(x0, deep)}
        rows_ev = eq0.row_ev
        cols_ev = eq0.col_ev
        for k in args.rect:
            rr = _rectangle(eq0.row_strategy, rows_ev, k, larger=True)
            cc = _rectangle(eq0.col_strategy, cols_ev, k, larger=False)
            mask = np.zeros_like(diff)
            mask[np.ix_(rr, cc)] = True
            mask &= diff
            count = int(mask.sum())
            if count == 0:
                rec_e[f"rect{k}"] = {"cells": 0, "loss": rec_e["none"], "near": rec_e["none"],
                                     "random": rec_e["none"]}
                continue
            near = np.zeros_like(diff)
            order = cells[np.argsort(np.abs(r24[cells[:, 0], cells[:, 1]] - v0), kind="stable")]
            near[order[:count, 0], order[:count, 1]] = True
            rnd = []
            for _ in range(args.draws):
                pick = cells[rng.choice(len(cells), size=min(count, len(cells)), replace=False)]
                mk = np.zeros_like(diff)
                mk[pick[:, 0], pick[:, 1]] = True
                rnd.append(_score(_answer(r24, deep, mk), deep))
            rec_e[f"rect{k}"] = {"cells": count,
                                 "loss": _score(_answer(r24, deep, mask), deep),
                                 "near": _score(_answer(r24, deep, near), deep),
                                 "random": float(np.mean(rnd))}
        rec_e["all"] = _score(_answer(r24, deep, diff), deep)
        out_rows.append({"n": n, "turn": turn, "wide": wide, "e": rec_e, "drift": drift,
                         "d1": first, "byDepth": depth_last})
        print(f"open {n}: e {json.dumps({k: v for k, v in rec_e.items() if k != 'size'})[:200]}",
              file=sys.stderr, flush=True)
    _write(Path(args.out) / "open.json", out_rows)
    lines.append(f"  {len(out_rows)} positions with r24 and R4D5@3; recomputed-vs-stored loss, "
                 f"largest difference {max(r['drift'] for r in out_rows):.2e} (control)")
    lines.append("(a) the ladder's answer by depth, scored in R4D5 (depth-5, 4x4 root reference)")
    lines.append(_row("depth-1 answer (before the first rung)", [r["d1"] for r in out_rows]))
    for key in sorted(depth_rows, key=lambda s: int(re.findall(r"\d+", s)[0])):
        lines.append(_row(key, depth_rows[key]))
    lines.append("(d) the depth-4 answer (R4A stage 2) scored in wider references")
    for name in ("R4S@3", "R4W@3", "R4D5@3"):
        v = [r["wide"][name] for r in out_rows if name in r["wide"]]
        if v:
            lines.append(_row(name, v))
    lines.append("(e) deep values of a few cells put in the depth-2 matrix (loss in R4D5)")
    lines.append(_row("no cell (the depth-2 answer)", [r["e"]["none"] for r in out_rows]))
    for k in args.rect:
        sel = [r for r in out_rows if f"rect{k}" in r["e"] and r["e"][f"rect{k}"]["cells"] > 0]
        if not sel:
            continue
        c = np.mean([r["e"][f"rect{k}"]["cells"] for r in sel])
        lines.append(f"  -- rectangle {k}x{k} from the depth-2 answer: {c:.1f} cells on average")
        lines.append(_row("     the rectangle's cells", [r["e"][f"rect{k}"]["loss"] for r in sel]))
        lines.append(_row("     same count, nearest the game value",
                          [r["e"][f"rect{k}"]["near"] for r in sel]))
        lines.append(_row("     same count, at random", [r["e"][f"rect{k}"]["random"] for r in sel]))
    lines.append(_row("all differing cells (control: must be 0)", [r["e"]["all"] for r in out_rows]))
    lines.append("  differing cells per position: "
                 + ", ".join(f"{r['n']}:{r['e']['cells']}/{r['e']['size'][0]}x{r['e']['size'][1]}"
                             for r in out_rows))
    lines.append("loss by turn (depth 2 answer -> depth 3+):")
    for r in sorted(out_rows, key=lambda r: r["turn"]):
        bd = r["byDepth"]
        lines.append(f"  pos {r['n']:>2} turn {r['turn']}: d1 {r['d1']:.4f}  "
                     + "  ".join(f"d{d} {v:.4f}" for d, v in bd.items()))
    text = "\n".join(lines) + "\n"
    (Path(args.out) / "open.txt").write_bytes(text.encode("utf-8"))
    print(text)


def _bayes(mats: list[np.ndarray], w: np.ndarray):  # noqa: ANN202 - the solved Bayesian game
    return solve_bayesian(mats, w)


def cross(args: argparse.Namespace) -> None:
    """E1 (IKA-394): the answer of one evaluation model's depth-2 game scored in another's.
    ``--model LABEL REFNAME``: SET/ref-REFNAME holds that model's game on the same rows and
    columns (`position_set.py reference --menus-from`). A's answer is A's own game's
    equilibrium (Bayesian on a hidden position); the loss is B's game value minus what A's
    mixture guarantees in B's game. A = B is the control (0)."""
    labels = [m[0] for m in args.model]
    table: dict[tuple[str, str], list[float]] = {}
    per: dict[int, dict] = {}
    for n in range(10**6):
        games = {}
        for label, ref in args.model:
            path = Path(args.set) / f"ref-{ref}" / f"{n}.npz"
            if path.exists():
                d = np.load(path)
                d2 = np.asarray(d["d2"], dtype=np.float64)
                w = np.asarray(d["weights"], dtype=np.float64) if "weights" in d.files else None
                games[label] = ([d2] if d2.ndim == 2 else [d2[k] for k in range(len(d2))],
                                np.ones(1) if w is None else w / w.sum())
        if n > 400 and not games:
            break
        if len(games) != len(labels):
            continue
        answers = {a: _value(*games[a])[1] for a in labels}
        values = {b: _value(*games[b])[0] for b in labels}
        row = {}
        for a in labels:
            for b in labels:
                loss = values[b] - _guarantee(answers[a], *games[b])
                table.setdefault((a, b), []).append(loss)
                row[f"{a}->{b}"] = loss
        per[n] = row
        off = {k: round(v, 4) for k, v in row.items() if k.split("->")[0] != k.split("->")[1]}
        print(f"cross {n}: {json.dumps(off)}", file=sys.stderr, flush=True)
    lines = [f"CROSS {args.set}: {len(per)} positions; loss of A's answer (row) in B's game (column)"]
    for a in labels:
        for b in labels:
            lines.append(_row(f"{a} -> {b}", table[a, b]))
    turns = json.loads((Path(args.set) / "positions.json").read_bytes())["positions"]
    for grp in ("turn2-3", "turn4-5", "turn6+"):
        ns = [n for n in per if _turn_group(turns[n].get("turn", 0)) == grp]
        if ns:
            lines.append(f"[{grp}: {len(ns)} positions] mean loss of A's answer in B's game")
            for a in labels:
                lines.append("  " + f"{a:<12}" + "  ".join(
                    f"{b}: {np.mean([per[n][f'{a}->{b}'] for n in ns]):.4f}" for b in labels))
    text = "\n".join(lines) + "\n"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"cross-{args.tag}.txt").write_bytes(text.encode("utf-8"))
    _write(Path(args.out) / f"cross-{args.tag}.json", per)
    print(text)


def hidden_deep(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915 - one pass
    """E2 (IKA-394): hidden positions with a ladder reference read to depth 3 (``ref-<NAME>`` and
    its stages ``ref-<NAME>@<i>``, from ``position_set.py ladder-ref``): the answers of depth 2
    (r24), of each stage, and of the older ``ref-deep`` scored in the last stage's game;
    and (e) a few cells' depth-3 values put into the depth-2 matrices."""
    set_dir = Path(args.set)
    rng = np.random.default_rng(args.seed)
    data = json.loads((set_dir / "positions.json").read_bytes())
    rows = []
    for n, entry in enumerate(data["positions"]):
        final_path = set_dir / f"ref-{args.name}" / f"{n}.npz"
        parts_path = set_dir / f"ref-{args.name}" / "parts" / f"{n}.json"
        if not final_path.exists() or not parts_path.exists():
            continue
        r24 = np.load(set_dir / "ref-r24" / f"{n}.npz")
        w = np.asarray(r24["weights"], dtype=np.float64)
        w = w / w.sum()
        k_all = len(w)
        shallow = [np.asarray(r24["d2"], dtype=np.float64)[k] for k in range(k_all)]
        final = [np.asarray(np.load(final_path)["d2"], dtype=np.float64)[k] for k in range(k_all)]
        best = _bayes(final, w)
        rec = {"n": n, "turn": entry["turn"], "K": k_all, "V": float(best.value)}
        rungs = json.loads(parts_path.read_bytes())["rungs"]
        answers = {"r24 (depth 2)": _bayes(shallow, w).row_strategy}
        for i, g in enumerate(rungs):
            answers[f"stage {i + 1}: {g['stage']}"] = np.asarray(g["x"], dtype=np.float64)
        deep_path = set_dir / "ref-deep" / f"{n}.npz"
        deep = None
        if deep_path.exists():
            deep = [np.asarray(np.load(deep_path)["d2"], dtype=np.float64)[k] for k in range(k_all)]
            answers["old deep (8x8 depth 3)"] = _bayes(deep, w).row_strategy
        rec["loss"] = {k: float(best.value) - _guarantee(x, final, w) for k, x in answers.items()}
        if deep is not None:
            vd = float(_bayes(deep, w).value)
            rec["lossInDeep"] = {k: vd - _guarantee(x, deep, w) for k, x in answers.items()}
        # (e) the depth-3 values of a few cells, in the depth-2 matrices.
        eq0 = _bayes(shallow, w)
        x0 = eq0.row_strategy
        diffs = [np.abs(final[k] - shallow[k]) > 1e-9 for k in range(k_all)]
        total = int(sum(d.sum() for d in diffs))
        row_ev = sum(w[k] * (shallow[k] @ eq0.col_strategies[k]) for k in range(k_all))
        values = [float(x0 @ shallow[k] @ eq0.col_strategies[k]) for k in range(k_all)]
        e = {"cells": total, "none": rec["loss"]["r24 (depth 2)"]}

        def solved_loss(masks: list[np.ndarray], shallow=shallow, final=final, w=w,
                        best=best) -> float:
            mixed = [np.where(m, f, s) for m, f, s in zip(masks, final, shallow, strict=True)]
            xm = _bayes(mixed, w).row_strategy
            return float(best.value) - _guarantee(xm, final, w)

        for k in args.rect:
            rr = _rectangle(x0, row_ev, k, larger=True)
            masks = []
            for j in range(k_all):
                col_ev = x0 @ shallow[j]
                cc = _rectangle(eq0.col_strategies[j], col_ev, k, larger=False)
                m = np.zeros_like(diffs[j])
                m[np.ix_(rr, cc)] = True
                masks.append(m & diffs[j])
            count = int(sum(m.sum() for m in masks))
            if count == 0:
                e[f"rect{k}"] = {"cells": 0, "loss": e["none"], "near": e["none"],
                                 "random": e["none"]}
                continue
            cells = [(j, i, c) for j in range(k_all) for i, c in np.argwhere(diffs[j])]
            order = sorted(cells, key=lambda t: abs(shallow[t[0]][t[1], t[2]] - values[t[0]]))
            near = [np.zeros_like(d) for d in diffs]
            for j, i, c in order[:count]:
                near[j][i, c] = True
            rnd = []
            for _ in range(args.draws):
                pick = rng.choice(len(cells), size=min(count, len(cells)), replace=False)
                mk = [np.zeros_like(d) for d in diffs]
                for t in pick:
                    j, i, c = cells[t]
                    mk[j][i, c] = True
                rnd.append(solved_loss(mk))
            e[f"rect{k}"] = {"cells": count, "loss": solved_loss(masks),
                             "near": solved_loss(near), "random": float(np.mean(rnd))}
        e["all"] = solved_loss(diffs)
        rec["e"] = e
        rows.append(rec)
        shown = {a: round(b, 4) for a, b in rec["loss"].items()}
        print(f"hidden-deep {n}: K={k_all} loss {json.dumps(shown)}", file=sys.stderr, flush=True)
    _write(Path(args.out) / f"hidden-deep-{args.name}.json", rows)
    lines = [f"HIDDEN DEEP {set_dir} ref-{args.name}: {len(rows)} positions "
             f"(K: {[r['K'] for r in rows]}, turns {[r['turn'] for r in rows]})",
             "loss of each answer in the last stage's Bayesian game (depth 3 of the ladder)"]
    keys = list(rows[0]["loss"])
    for key in keys:
        lines.append(_row(key, [r["loss"][key] for r in rows if key in r["loss"]]))
    lines.append("loss of each answer in the old deep reference (the reference the earlier records used)")
    sel = [r for r in rows if "lossInDeep" in r]
    for key in keys:
        v = [r["lossInDeep"][key] for r in sel if key in r["lossInDeep"]]
        if v:
            lines.append(_row(key, v))
    lines.append("(e) a few cells' depth-3 values in the depth-2 matrices (loss in the last stage's game)")
    lines.append(_row("no cell (the depth-2 answer)", [r["e"]["none"] for r in rows]))
    for k in args.rect:
        sel = [r for r in rows if f"rect{k}" in r["e"] and r["e"][f"rect{k}"]["cells"] > 0]
        if not sel:
            continue
        lines.append(f"  -- rectangle {k}x{k}: {np.mean([r['e'][f'rect{k}']['cells'] for r in sel]):.1f} "
                     f"cells (over all completions) on average of {len(sel)} positions")
        for label, key in (("the rectangle's cells", "loss"), ("same count, nearest the value", "near"),
                           ("same count, at random", "random")):
            lines.append(_row(f"     {label}", [r["e"][f"rect{k}"][key] for r in sel]))
    lines.append(_row("all differing cells (control: must be 0)", [r["e"]["all"] for r in rows]))
    lines.append("per position (n turn K cells | depth-2 loss | stage losses):")
    for r in rows:
        lines.append(f"  {r['n']} t{r['turn']} K{r['K']} cells {r['e']['cells']} | "
                     + " ".join(f"{v:.4f}" for v in r["loss"].values()))
    text = "\n".join(lines) + "\n"
    (Path(args.out) / f"hidden-deep-{args.name}.txt").write_bytes(text.encode("utf-8"))
    print(text)


def calib(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915 - one pass over the games
    """The evaluation model against the recorded results (IKA-394): for the move decisions of
    recorded games, the model's win probability of the position (side 0) and the recorded
    search value against the game's outcome (1 = side 0 won): Brier, log loss, a
    reliability table, by turn. The positions are the games' true positions."""
    from pokeuraou import humanplay
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.pool import load_pool

    reg = load_pool(args.pool).reg
    register_mega_stones(reg)
    games = []
    for f in sorted(Path(args.games).glob("games-worker*.jsonl"))[: args.files]:
        for line in f.read_bytes().splitlines():
            if line.strip():
                games.append(json.loads(line))
            if len(games) >= args.games_max:
                break
        if len(games) >= args.games_max:
            break
    rows = []
    for gi, g in enumerate(games):
        for d in g["decisions"]:
            if d["kind"] == "move" and d.get("searchValue") is not None:
                rows.append((gi, d["turn"], float(g["outcome"]), float(d["searchValue"]),
                             d["position"]))
    print(f"{len(games)} games, {len(rows)} move decisions", file=sys.stderr, flush=True)
    game = np.array([r[0] for r in rows])
    turn = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    search = np.array([r[3] for r in rows])
    preds = {"search (recorded)": search}
    for label, paths in args.model:
        leaf, _enc, _dev = humanplay.load_leaf(reg, [Path(p) for p in paths.split(",")],
                                               args.device, graphs=False)
        out = []
        for i in range(0, len(rows), 256):
            out.append(np.asarray(leaf([r[4] for r in rows[i:i + 256]]), dtype=np.float64))
        preds[label] = np.concatenate(out)
        print(f"model {label} done", file=sys.stderr, flush=True)
    eps = 1e-6

    def brier(p: np.ndarray) -> np.ndarray:
        return (p - y) ** 2

    def logloss(p: np.ndarray) -> np.ndarray:
        q = np.clip(p, eps, 1 - eps)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))

    def by_game(v: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        ids = np.unique(game[mask])
        means = np.array([v[mask & (game == i)].mean() for i in ids])
        return float(means.mean()), float(means.std(ddof=1) / np.sqrt(len(means)))

    lines = [f"CALIBRATION {args.games}: {len(games)} games, {len(rows)} move decisions; "
             f"outcome mean {y.mean():.3f}; correlation search/outcome "
             f"{np.corrcoef(search, y)[0, 1]:+.3f}"]
    everything = np.ones(len(rows), dtype=bool)
    groups = [("all turns", everything)] + [
        (f"turn {a}-{b}", (turn >= a) & (turn <= b)) for a, b in ((1, 2), (3, 4), (5, 6), (7, 99))]
    for name, mask in groups:
        lines.append(f"[{name}: {int(mask.sum())} decisions, {len(np.unique(game[mask]))} games]"
                     f"  constant-0.5 Brier {by_game(np.full(len(rows), 0.25), mask)[0]:.4f}")
        for label, p in preds.items():
            b, bse = by_game(brier(p), mask)
            ll, _ = by_game(logloss(p), mask)
            lines.append(f"  {label:<30} Brier {b:.4f} (se {bse:.4f})  log loss {ll:.4f}")
        base = "search (recorded)"
        for label, p in preds.items():
            if label != base:
                dd, dse = by_game(brier(p) - brier(preds[base]), mask)
                lines.append(f"  Brier {label} - {base}: {dd:+.4f} (se {dse:.4f}, games)")
    for label, p in preds.items():
        lines.append(f"reliability of {label} (bin of the prediction: mean prediction -> mean outcome, n)")
        edges = np.linspace(0, 1, 11)
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            mk = (p >= lo) & (p < hi + (1e-9 if hi == 1 else 0))
            if mk.sum() >= 20:
                cells.append(f"{p[mk].mean():.2f}->{y[mk].mean():.2f} ({int(mk.sum())})")
        lines.append("  " + "  ".join(cells))
    text = "\n".join(lines) + "\n"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "calib.txt").write_bytes(text.encode("utf-8"))
    print(text)


def _mix(shallow: np.ndarray, deep: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = shallow.copy()
    out[mask] = deep[mask]
    return out


def _answer(shallow: np.ndarray, deep: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return solve(_mix(shallow, deep, mask)).row_strategy


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hidden")
    h.add_argument("--set", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--seed", type=int, default=39400)
    h.add_argument("--max-subsets", type=int, default=20)
    o = sub.add_parser("open")
    o.add_argument("--set", type=Path, required=True)
    o.add_argument("--sweep", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--seed", type=int, default=39401)
    o.add_argument("--rect", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    o.add_argument("--draws", type=int, default=20)
    cr = sub.add_parser("cross")
    cr.add_argument("--set", type=Path, required=True)
    cr.add_argument("--out", type=Path, required=True)
    cr.add_argument("--tag", required=True)
    cr.add_argument("--model", nargs=2, action="append", required=True, metavar=("LABEL", "REF"))
    hd = sub.add_parser("hidden-deep")
    hd.add_argument("--set", type=Path, required=True)
    hd.add_argument("--name", required=True)
    hd.add_argument("--out", type=Path, required=True)
    hd.add_argument("--seed", type=int, default=39402)
    hd.add_argument("--rect", type=int, nargs="+", default=[2, 3, 4, 6])
    hd.add_argument("--draws", type=int, default=10)
    c = sub.add_parser("calib")
    c.add_argument("--games", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--pool", default="regmc-matchupweb")
    c.add_argument("--files", type=int, default=4)
    c.add_argument("--games-max", type=int, default=2000)
    c.add_argument("--device", default=None)
    c.add_argument("--model", nargs=2, action="append", default=[], metavar=("LABEL", "PATH,PATH"))
    args = ap.parse_args(argv)
    {"hidden": hidden, "open": open_set, "calib": calib,
     "hidden-deep": hidden_deep, "cross": cross}[args.cmd](args)


if __name__ == "__main__":
    main()
