"""Crit as a chance branch, by stage of the game (IKA-359, measure first).

`Budget.matrix()` collapses crits: a hit never crits. That was measured on 12 mid-game
positions (value 0.0005, policy TV 0.003, cost 2.68x; `tools/budget_effect.py`) and folded,
"a crit changes one damage number rather than whether the move happened". In an endgame a
crit decides whether a hit knocks out, which is the whole of the closing reading -- and
that was never measured.

This measures it, on move decisions from recorded games (the positions the search meets),
read open (the recorded position is the true one), over the recorded candidate sets, with
the learned leaf at depth 1 -- the matrix `search` builds -- under several budgets:

  base    `Budget.matrix()`, the search's budget
  wide    the same with the branch cap raised (so the cap alone is not credited to crit)
  crit    wide + crits enumerated
  rolls   wide + all sixteen damage rolls (no crit), to set crit against the other half
  ko16    base + the knock-out branch (`Budget.enumerate_knockouts`, candidate (a) in the
          port: a hit forks only where crit or roll decides a knock-out)
  ko      the same at the raised cap
  exact   `Budget.exact()`, the reference

and a cell-level stand-in for candidate (a), "branch on a crit only where it changes a
knock-out": each cell takes the ``crit`` value when some crit branch ends with a different
set of fainted Pokemon from every no-crit branch, and the ``wide`` value otherwise (an upper
bound on what a per-hit rule in the port could recover, at its cost of that share of cells).

Per budget against ``exact``: the value shift, the policy TV of each side, and the
exploitability of its equilibrium in the exact game (``loss``, the NashConv: what playing
the collapsed answer gives away when the game really has crits and rolls), and the cost
(wall seconds of the matrix fill, the port included, over ``base``'s; and branches a cell).

Stages, by standing Pokemon (field and bench) per side:

  end     neither side has more than two left
  late    five or fewer in all, not end
  mid     the rest, from turn 3

    python tools/crit_effect.py scan --out scan.jsonl --games-per-file 40 data/selfplay-mc1
    python tools/crit_effect.py read --scan scan.jsonl --per-stage 40 --jobs 6 \\
        --out read.jsonl --value data/models/value-mc0.pt data/models/value-mc0-s1.pt
    python tools/crit_effect.py analyse read.jsonl

Each `read` worker appends one line per position to ``<out>.part<N>`` as it finishes it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STAGES = ("end", "late", "mid")
BUDGETS = ("base", "wide", "crit", "rolls", "ko16", "ko", "exact")
#: The cap for every richer budget; the matrix budget's 16 would truncate what is measured.
CAP = 512


def budgets() -> dict[str, Any]:
    from pokeuraou.budget import Budget

    base = Budget.matrix()
    wide = replace(base, max_branches=CAP)
    return {
        "base": base,
        "wide": wide,
        "crit": replace(wide, enumerate_crit=True),
        "rolls": replace(wide, damage_rolls=16),
        "ko16": replace(base, enumerate_knockouts=True),
        "ko": replace(wide, enumerate_knockouts=True),
        "exact": replace(Budget.exact(), max_branches=CAP),
    }


# --------------------------------------------------------------------------------------
# scan


def standing(position: dict[str, Any]) -> list[int]:
    return [
        sum(1 for m in side.get("pokemon") or () if not m.get("fainted") and m.get("hp", 1) > 0)
        for side in position["sides"]
    ]


def stage_of(turn: int, alive: Sequence[int]) -> str | None:
    if min(alive) <= 0:
        return None
    if max(alive) <= 2:
        return "end"
    if sum(alive) <= 5:
        return "late"
    if turn >= 3:
        return "mid"
    return None


def _field_hp(position: dict[str, Any]) -> float:
    """Mean HP fraction of the Pokemon on the field, both sides."""
    fr = [
        m["hp"] / max(1, m["maxhp"])
        for side in position["sides"]
        for m in side.get("pokemon") or ()
        if not m.get("fainted") and m.get("activeIndex") is not None
    ]
    return float(np.mean(fr)) if fr else 0.0


def _scan_file(job: tuple[str, int]) -> list[dict[str, Any]]:
    path, games = job
    out = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no >= games:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, d in enumerate(record.get("decisions", ())):
                if d.get("kind") != "move":
                    continue
                pos = d["position"]
                alive = standing(pos)
                stage = stage_of(int(d.get("turn", 0)), alive)
                if stage is None:
                    continue
                out.append({
                    "file": path, "line": line_no, "decision": k, "stage": stage,
                    "turn": int(d.get("turn", 0)), "alive": alive,
                    "fieldHp": round(_field_hp(pos), 4),
                    "menu": [len(d.get("ownActions") or ()), len(d.get("foeActions") or ())],
                })
    return out


def run_scan(args: argparse.Namespace) -> None:
    files = sorted(
        str(p.resolve()) for d in args.dirs for p in Path(d).glob("games-worker*.jsonl")
    )
    with Pool(args.jobs) as pool:
        parts = pool.map(_scan_file, [(f, args.games_per_file) for f in files])
    rows = [r for part in parts for r in part]
    with open(args.out, "wb") as handle:
        for r in rows:
            handle.write((json.dumps(r) + "\n").encode("utf-8"))
    counts = {s: sum(1 for r in rows if r["stage"] == s) for s in STAGES}
    print(f"{len(rows):,} move decisions from {len(files)} files -> {args.out}  {counts}")


# --------------------------------------------------------------------------------------
# read


def _faints(position: Any) -> frozenset[tuple[int, str]]:
    return frozenset(
        (s, m.species) for s, side in enumerate(position.sides) for m in side.pokemon if m.fainted
    )


class Reader:
    def __init__(self, values: Sequence[str], light: bool = False) -> None:
        import torch

        #: Skip the stand-in for (a) and the dear branch counts: the mid-game's full
        #: outcomes of the crit budget are most of a position's time.
        self.light = light

        torch.set_num_threads(1)
        self.torch = torch
        self.values = values
        self.reg = None
        self.leaf = None
        self.budgets = budgets()

    def _setup(self, pos: Any) -> None:
        from pokeuraou.damage import register_mega_stones
        from pokeuraou.encode import Encoder
        from pokeuraou.regulation import load_regulation
        from pokeuraou.value import BatchedValue, load_ensemble

        self.reg = load_regulation(pos.format)
        register_mega_stones(self.reg)
        encoder = Encoder(self.reg)
        nets, _ = load_ensemble([Path(v) for v in self.values], encoder)
        self.leaf = BatchedValue(nets, encoder, device=self.torch.device("cpu"))

    def _signatures(
        self, pos: Any, row: list, col: list, budget: Any, *, full: bool = True
    ) -> tuple[list, list]:
        """Per cell: the set of faint signatures over its outcomes (with ``full``), and its
        branch count (outcomes and pauses)."""
        from pokeuraou import port

        answers = port.turns(
            self.reg, [(pos, [a, b]) for a in row for b in col], budget, full=full
        )
        sigs, counts = [], []
        for ans in answers:
            if isinstance(ans, Exception):
                raise ans
            counts.append(len(ans.branches) + len(ans.suspended))
            if full:
                outs = [o.position for o in ans.outcomes or ()] + [
                    p.position for p in ans.pauses or ()
                ]
                sigs.append({_faints(p) for p in outs})
        return sigs, counts

    def read(self, record: dict[str, Any], decision: int) -> dict[str, Any]:
        from pokeuraou import port
        from pokeuraou.actions import side_actions
        from pokeuraou.equilibrium import EquilibriumError, solve
        from pokeuraou.position import Position

        d = record["decisions"][decision]
        pos = Position.from_json(d["position"])
        if self.reg is None:
            self._setup(pos)
        menu = (d["ownActions"], d["foeActions"])
        sides = []
        for side in (0, 1):
            by = {a.to_choice(): a for a in side_actions(self.reg, pos, side)}
            missing = [c for c in menu[side] if c not in by]
            if missing:
                return {"skipped": f"side {side} menu has {missing}"}
            sides.append([by[c] for c in menu[side]])
        row, col = sides
        out: dict[str, Any] = {"menu": [len(row), len(col)]}
        # Untimed, so the first budget does not carry the process's first-call costs.
        port.batched_payoffs(self.reg, pos, row, col, [self.leaf], budget=self.budgets["base"])
        for name, budget in self.budgets.items():
            # Wall, not process CPU: the port is another process. One worker holds one port.
            wall = time.perf_counter()
            try:
                mats, _notes, exact = port.batched_payoffs(
                    self.reg, pos, row, col, [self.leaf], budget=budget
                )
            except port.PortRefused as err:
                return {"skipped": f"refused: {err}"[:200]}
            seconds = time.perf_counter() - wall
            payoff = mats[0]
            try:
                eq = solve(payoff)
            except EquilibriumError:
                return {"skipped": f"{name}: no equilibrium"}
            out[name] = {
                "payoff": np.round(payoff, 7).tolist(),
                "value": float(eq.value),
                "row": np.round(eq.row_strategy, 7).tolist(),
                "col": np.round(eq.col_strategy, 7).tolist(),
                "sec": round(seconds, 5),
                "inexact": int(np.size(exact) - int(np.sum(exact))),
            }
        _, base_n = self._signatures(pos, row, col, self.budgets["base"], full=False)
        _, ko16_n = self._signatures(pos, row, col, self.budgets["ko16"], full=False)
        if self.light:
            out["branches"] = {"base": base_n, "ko16": ko16_n}
            return out
        # Which cells a crit changes a knock-out in, and the branch counts behind the cost.
        wide_sigs, wide_n = self._signatures(pos, row, col, self.budgets["wide"])
        crit_sigs, crit_n = self._signatures(pos, row, col, self.budgets["crit"])
        _, exact_n = self._signatures(pos, row, col, self.budgets["exact"], full=False)
        out["critKo"] = [int(bool(c - w)) for w, c in zip(wide_sigs, crit_sigs, strict=True)]
        out["branches"] = {
            "base": base_n, "ko16": ko16_n, "wide": wide_n, "crit": crit_n, "exact": exact_n,
        }
        return out


_READER: Reader | None = None


def _read_job(job: tuple[dict[str, Any], list[str], str, bool]) -> dict[str, Any]:
    global _READER  # noqa: PLW0603 - one leaf per worker process
    row, values, part, light = job
    if _READER is None:
        _READER = Reader(values, light)
    with open(row["file"], encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            if n == row["line"]:
                record = json.loads(line)
                break
    started = time.perf_counter()
    got = _READER.read(record, row["decision"])
    result = {**row, **got, "wall": round(time.perf_counter() - started, 3)}
    with open(part, "ab") as handle:
        handle.write((json.dumps(result) + "\n").encode("utf-8"))
    return {"stage": row["stage"], "skipped": "skipped" in got}


def run_read(args: argparse.Namespace) -> None:
    rows = [json.loads(x) for x in Path(args.scan).read_bytes().splitlines() if x]
    if args.max_menu:
        rows = [r for r in rows if r["menu"][0] * r["menu"][1] <= args.max_menu]
    chosen = []
    rng = random.Random(args.seed)
    for stage in STAGES:
        pool = [r for r in rows if r["stage"] == stage]
        rng.shuffle(pool)  # before the stage filter, so a stage's sample does not depend on it
        if stage not in args.stages:
            continue
        chosen.extend(pool[: args.per_stage])
    rng.shuffle(chosen)
    jobs = [
        (r, args.value, f"{args.out}.part{i % args.jobs}", args.light) for i, r in enumerate(chosen)
    ]
    with Pool(args.jobs, maxtasksperchild=None) as pool:
        for done, _got in enumerate(pool.imap_unordered(_read_job, jobs), start=1):
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    parts = sorted(Path(args.out).parent.glob(Path(args.out).name + ".part*"))
    with open(args.out, "wb") as handle:
        for p in parts:
            handle.write(p.read_bytes())
    print(f"{len(jobs)} positions -> {args.out}")


# --------------------------------------------------------------------------------------
# analyse


def _loss(x: np.ndarray, y: np.ndarray, ref: np.ndarray, value: float) -> tuple[float, float]:
    """What each side's strategy gives away in the reference game (row, column)."""
    return value - float((x @ ref).min()), float((ref @ y).max()) - value


def _tv(a: Sequence[float], b: Sequence[float]) -> float:
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def measures(rec: dict[str, Any]) -> dict[str, dict[str, float]]:
    from pokeuraou.equilibrium import solve

    ref = np.asarray(rec["exact"]["payoff"])
    v_ref = rec["exact"]["value"]
    base_cpu = max(rec["base"]["sec"], 1e-6)
    rows: dict[str, dict[str, float]] = {}
    labelled = {name: rec[name] for name in BUDGETS}
    share = float("nan")
    if "critKo" in rec:
        # Candidate (a) at the cell level: the crit value where a crit changes a knock-out.
        mask = np.asarray(rec["critKo"], dtype=bool).reshape(ref.shape)
        crit = np.asarray(rec["crit"]["payoff"])
        a_payoff = np.where(mask, crit, np.asarray(rec["wide"]["payoff"]))
        eq = solve(a_payoff)
        share = float(mask.mean())
        labelled["critKO"] = {
            "payoff": a_payoff, "value": eq.value, "row": eq.row_strategy, "col": eq.col_strategy,
            # The port would pay crit's price on the flagged cells and wide's on the rest.
            "sec": share * rec["crit"]["sec"] + (1 - share) * rec["wide"]["sec"],
        }
    for name, got in labelled.items():
        x = np.asarray(got["row"])
        y = np.asarray(got["col"])
        lr, lc = _loss(x, y, ref, v_ref)
        rows[name] = {
            "dv": abs(got["value"] - v_ref),
            "tv": 0.5 * (_tv(x, rec["exact"]["row"]) + _tv(y, rec["exact"]["col"])),
            "loss": lr + lc,
            "cost": got["sec"] / base_cpu,
        }
    rows["_"] = {"critKoShare": share}
    return rows


def run_analyse(args: argparse.Namespace) -> None:
    recs = []
    for path in args.reads:
        recs.extend(json.loads(x) for x in Path(path).read_bytes().splitlines() if x)
    kept = [r for r in recs if "skipped" not in r]
    print(f"{len(recs)} read, {len(recs) - len(kept)} skipped")
    all_names = (*BUDGETS, "critKO")
    rng = np.random.default_rng(0)
    for stage in (*STAGES, "all"):
        group = [r for r in kept if stage in ("all", r["stage"])]
        if not group:
            continue
        m = [measures(r) for r in group]
        names = [n for n in all_names if all(n in x for x in m)]
        share = np.mean([x["_"]["critKoShare"] for x in m])
        with_ko = np.mean([x["_"]["critKoShare"] > 0 for x in m])
        cells = np.mean([r["menu"][0] * r["menu"][1] for r in group])
        keys = [k for k in group[0]["branches"] if all(k in r["branches"] for r in group)]
        br = {k: np.mean([np.mean(r["branches"][k]) for r in group]) for k in keys}
        print(
            f"\n{stage}: {len(group)} positions, {cells:.0f} cells, crit changes a KO in "
            f"{share:.1%} of cells ({with_ko:.0%} of positions have one); branches/cell "
            + " ".join(f"{k} {v:.1f}" for k, v in br.items())
        )
        print(f"  {'budget':>7}  {'|dv|':>7}  {'p90':>7}  {'TV':>6}  {'loss':>7}  "
              f"{'loss 95%':>17}  {'cost':>6}")
        for name in names:
            dv = np.array([x[name]["dv"] for x in m])
            tv = np.array([x[name]["tv"] for x in m])
            loss = np.array([x[name]["loss"] for x in m])
            cost = np.array([x[name]["cost"] for x in m])
            boots = [loss[rng.integers(0, len(loss), len(loss))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(
                f"  {name:>7}  {dv.mean():7.4f}  {np.percentile(dv, 90):7.4f}  {tv.mean():6.3f}  "
                f"{loss.mean():7.4f}  [{lo:7.4f},{hi:7.4f}]  {np.exp(np.log(cost).mean()):5.2f}x"
            )
        # The pair that decides: what crit (and its stand-in) recover of wide's loss.
        for name in [n for n in ("crit", "critKO", "rolls", "ko16", "ko") if n in names]:
            diff = np.array([x["wide"]["loss"] - x[name]["loss"] for x in m])
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"    wide - {name} loss {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--games-per-file", type=int, default=40)
    s.add_argument("--jobs", type=int, default=1)
    r = sub.add_parser("read")
    r.add_argument("--scan", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--per-stage", type=int, default=40)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--max-menu", type=int, default=0, help="skip menus with more cells")
    r.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    r.add_argument("--light", action="store_true", help="no (a) stand-in, fewer branch counts")
    r.add_argument("--value", nargs="+", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("reads", nargs="+")
    args = ap.parse_args(argv)
    {"scan": run_scan, "read": run_read, "analyse": run_analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
