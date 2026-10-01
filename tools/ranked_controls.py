"""The ranked entry's controls that need `data/` (IKA-407): they cannot run on CI, so they are
a tool with its numbers recorded in records/IKA-407.md.

    uv run python tools/ranked_controls.py paste               # the pool's pastes read as the pool
    uv run python tools/ranked_controls.py compare --pairs 40  # true opponent sets vs the estimate

``paste``: every cached paste of the pool (`data/pool/regmc-matchupweb/pastes`) goes through
`rankedentry.roster_from_paste`; each is compared with the pool's own roster (the importer's
supplements are applied to a few pastes, which a plain paste read cannot reproduce: those are
counted apart, not hidden).

``compare``: for pairs of pool teams, our side is team i pasted, the opponent is team j. The
selection is solved against (a) j's true sets and (b) the sets estimated from j's six species
(`FieldPrior`: each species alone with ``--tiers 0``, or read for the whole six, the field teams
that share the most of it first, with ``--tiers N`` = `TIER_MIN`; ``--holdout`` takes the field
teams that bring exactly j's six out first, since the pool's teams are mostly field entries).
The estimate's mixture is then scored in the TRUE game: the loss is the value of the true
game minus the worst case of the estimate's mixture against the true matrix. The uniform
mixture's loss is the scale. Leaf only (the leaf's one estimate of each cell, no deeper
reading), on the production leaf value-mc4 x2.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()

import numpy as np  # noqa: E402

from pokeuraou import humanplay  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402
from pokeuraou.rankedentry import TIER_MIN, FieldPrior, roster_from_paste  # noqa: E402
from pokeuraou.selection import SpreadClass, solve_selection  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


def sets_equal(a, b) -> bool:  # noqa: ANN001
    return (a.species, a.ability, a.item, a.nature, sorted(a.moves), a.sp) == (
        b.species, b.ability, b.item, b.nature, sorted(b.moves), b.sp)


def paste_control(root: Path) -> None:
    pool = load_pool(root / "pool" / "regmc-matchupweb.json")
    reg = pool.reg
    pastes = root / "pool" / "regmc-matchupweb" / "pastes"
    same = differ = missing = 0
    for roster in pool.teams:
        path = pastes / f"{roster.id}.txt"
        if not path.exists():
            missing += 1
            continue
        got = roster_from_paste(reg, path.read_text(encoding="utf-8"))
        pairs = [] if got.roster is None else zip(got.roster.sets, roster.sets, strict=True)
        if got.roster is not None and all(sets_equal(x, y) for x, y in pairs):
            same += 1
        else:
            differ += 1
            print("  differs:", roster.id, [p["reason"] for p in got.problems][:2])
    print(f"pool teams {len(pool.teams)}: paste read == pool roster {same}, differs {differ}, "
          f"no cached paste {missing}")


def compare(root: Path, pairs: int, values: list[Path], tiers: list[int], holdout: bool) -> None:
    """Each setting of the estimate is scored on the same pairs, in the same true games.

    ``tiers``: 0 reads each species alone (the marginal, what stage 1a first shipped); N > 0
    reads it for the opposing six with `TIER_MIN` = N. ``holdout``: the field teams that bring
    exactly the opposing six are taken out of the field before the estimate (the pool's teams
    may themselves be tournament entries, which would hand the estimate its answer).
    """
    pool = load_pool(root / "pool" / "regmc-matchupweb.json")
    reg = pool.reg
    register_mega_stones(reg)
    prior = FieldPrior.load(reg, root)
    evaluate, _enc, _dev = humanplay.load_leaf(reg, values, "cpu", graphs=False)
    n = len(pool.teams)

    def without_exact(theirs) -> FieldPrior:  # noqa: ANN001
        six = frozenset(s.species for s in theirs.sets)
        kept = [t for t in prior.standings.teams if frozenset(t.species) != six]
        return FieldPrior(reg, replace(prior.standings, teams=kept), pool)

    chosen = []
    for k in range(pairs * 3):
        i, j = k % n, (k * 7 + 3) % n
        if i == j:
            continue
        # a species the field never shows (with the holdout: nor without the exact-six teams)
        # has nothing to estimate from: refused, so not a pair
        field = without_exact(pool.teams[j]) if holdout else prior
        if any(s.species not in field.members for s in pool.teams[j].sets):
            continue
        _est, problems = prior.fill_team([s.species for s in pool.teams[j].sets], [])
        if problems:
            continue  # the marginal read has a clause clash it cannot move off: the same pairs for all
        chosen.append((i, j))
        if len(chosen) == pairs:
            break
    exact = sum(1 for _i, j in chosen
                if any(frozenset(t.species) == frozenset(s.species for s in pool.teams[j].sets)
                       for t in prior.standings.teams))
    print(f"pairs {len(chosen)}; opponents whose exact six is a field entry: {exact}", flush=True)
    truth = []
    for i, j in chosen:
        mine, theirs = pool.teams[i], pool.teams[j]
        true = solve_selection(reg, mine.sets, [SpreadClass(1.0, tuple(theirs.sets), "true")], evaluate)
        truth.append((mine, theirs, true, np.asarray(true.equilibrium.row_strategy), true.matrices[0]))
    uniform_loss = float(np.mean([t[2].value - float((np.full(90, 1 / 90) @ t[4]).min()) for t in truth]))
    print(f"loss of the uniform mixture in the true games (the scale): {uniform_loss:.4f}")
    head = ("setting", "same sets", "loss", "SE", "top same", "|dv|", "overlap>=2", "vs marginal (paired SE)")
    print(f"{head[0]:<28}{head[1]:>12}{head[2]:>9}{head[3]:>8}{head[4]:>10}{head[5]:>8}{head[6]:>12}"
          f"  {head[7]}")
    for held in ((False, True) if holdout else (False,)):
        base_losses: list[float] = []
        for tier in tiers:
            same = total = tops = clash = deep = 0
            losses, dvs = [], []
            for mine, theirs, true, x_true, a_true in truth:
                field = without_exact(theirs) if held else prior
                species = [s.species for s in theirs.sets]
                est, problems = field.fill_team(species, [] if tier == 0 else None,
                                                tier_min=tier or TIER_MIN)
                clash += bool(problems)
                deep += sum(1 for o in est if o.belief is not None and o.belief.overlap >= 2)
                est_sets = [o.set for o in est]
                total += 6
                same += sum(sets_equal_nosp(a, b) for a, b in zip(est_sets, theirs.sets, strict=True))
                guess = solve_selection(reg, mine.sets, [SpreadClass(1.0, tuple(est_sets), "estimate")],
                                        evaluate)
                x_est = np.asarray(guess.equilibrium.row_strategy)
                losses.append(true.value - float((x_est @ a_true).min()))
                dvs.append(abs(true.value - guess.value))
                tops += int(np.argmax(x_est) == np.argmax(x_true))
            label = ("marginal" if tier == 0 else f"six, tier_min {tier}") + (" +holdout" if held else "")
            se = np.std(losses, ddof=1) / np.sqrt(len(losses))
            if tier == 0:
                base_losses = losses
            diff = np.array(losses) - np.array(base_losses)
            versus = ("" if tier == 0 or not base_losses else
                      f"{diff.mean():+.4f} ({np.std(diff, ddof=1) / np.sqrt(len(diff)):.4f})")
            note = f"  (clauses unresolved in {clash} pairs)" if clash else ""
            print(f"{label:<28}{same:>5}/{total:<6}{np.mean(losses):>9.4f}{se:>8.4f}"
                  f"{tops:>6}/{len(truth):<3}{np.mean(dvs):>8.4f}{deep:>8}/{total:<3}  {versus}{note}",
                  flush=True)


def sets_equal_nosp(a, b) -> bool:  # noqa: ANN001
    return (a.species, a.ability, a.item, a.nature, sorted(a.moves)) == (
        b.species, b.ability, b.item, b.nature, sorted(b.moves))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("paste", "compare"))
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--value", type=Path, nargs="+", default=None)
    ap.add_argument("--tiers", type=int, nargs="+", default=[0, 3, 5, 10, 20],
                    help="0: each species alone; N: read for the opposing six with TIER_MIN = N")
    ap.add_argument("--holdout", action="store_true",
                    help="also score with the field teams that bring exactly the opposing six removed")
    args = ap.parse_args()
    if args.mode == "paste":
        paste_control(args.data_dir)
    else:
        compare(args.data_dir, args.pairs, args.value or [
            args.data_dir / "models" / "value-mc4.pt", args.data_dir / "models" / "value-mc4-s1.pt"],
            args.tiers, args.holdout)


if __name__ == "__main__":
    main()
