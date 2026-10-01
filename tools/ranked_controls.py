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
alone (`FieldPrior`). The estimate's mixture is then scored in the TRUE game: the loss is the
value of the true game minus the worst case of the estimate's mixture against the true
matrix. The uniform mixture's loss is the scale. Leaf only (the leaf's one estimate of each
cell, no deeper reading), on the production leaf value-mc3 x2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()

import numpy as np  # noqa: E402

from pokeuraou import humanplay  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402
from pokeuraou.rankedentry import FieldPrior, roster_from_paste  # noqa: E402
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


def compare(root: Path, pairs: int, values: list[Path]) -> None:
    pool = load_pool(root / "pool" / "regmc-matchupweb.json")
    reg = pool.reg
    register_mega_stones(reg)
    prior = FieldPrior.load(reg, root)
    evaluate, _enc, _dev = humanplay.load_leaf(reg, values, "cpu", graphs=False)
    n = len(pool.teams)
    rows = []
    member_same = member_total = 0
    for k in range(pairs * 3):
        if len(rows) == pairs:
            break
        i, j = k % n, (k * 7 + 3) % n
        if i == j:
            continue
        mine, theirs = pool.teams[i], pool.teams[j]
        if any(s.species not in prior.members for s in theirs.sets):
            continue  # a species the field never shows: nothing to estimate from (refused)
        est, problems = prior.fill_team([s.species for s in theirs.sets])
        if problems:
            continue
        est_sets = [o.set for o in est]
        member_total += 6
        member_same += sum(sets_equal_nosp(a, b) for a, b in zip(est_sets, theirs.sets, strict=True))
        true = solve_selection(reg, mine.sets, [SpreadClass(1.0, tuple(theirs.sets), "true")], evaluate)
        guess = solve_selection(reg, mine.sets, [SpreadClass(1.0, tuple(est_sets), "estimate")], evaluate)
        a_true = true.matrices[0]
        x_est = np.asarray(guess.equilibrium.row_strategy)
        x_true = np.asarray(true.equilibrium.row_strategy)
        uniform = np.full(len(x_est), 1.0 / len(x_est))
        def worst(x, matrix=a_true):  # noqa: ANN001, ANN202
            return float((x @ matrix).min())

        rows.append({
            "pair": (mine.id, theirs.id), "v_true": true.value, "v_est": guess.value,
            "loss_est": true.value - worst(x_est), "loss_uniform": true.value - worst(uniform),
            "loss_true": true.value - worst(x_true),
            "same_top": int(np.argmax(x_est) == np.argmax(x_true)),
            "l1": float(np.abs(x_est - x_true).sum()),
        })
        print(f"  {mine.id} vs {theirs.id}: value true {true.value:.3f} est {guess.value:.3f}; "
              f"loss est {rows[-1]['loss_est']:.4f} uniform {rows[-1]['loss_uniform']:.4f} "
              f"(true mixture {rows[-1]['loss_true']:.1e}); top same {rows[-1]['same_top']}", flush=True)
    m = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    sd = lambda key: float(np.std([r[key] for r in rows], ddof=1) / np.sqrt(len(rows)))  # noqa: E731
    print(f"\npairs {len(rows)}; estimated members equal to the true set in species, ability, item, "
          f"nature, moves: {member_same}/{member_total} (spread not compared)")
    print(f"our value: true game {m('v_true'):.4f}, estimated game {m('v_est'):.4f}, "
          f"|difference| {np.mean([abs(r['v_true'] - r['v_est']) for r in rows]):.4f}")
    print(f"loss in the true game: estimate's mixture {m('loss_est'):.4f} (SE {sd('loss_est'):.4f}); "
          f"uniform {m('loss_uniform'):.4f} (SE {sd('loss_uniform'):.4f}); true mixture {m('loss_true'):.1e}")
    print(f"top selection same as the true game's: {sum(r['same_top'] for r in rows)}/{len(rows)}; "
          f"mean L1 between mixtures {m('l1'):.3f}")


def sets_equal_nosp(a, b) -> bool:  # noqa: ANN001
    return (a.species, a.ability, a.item, a.nature, sorted(a.moves)) == (
        b.species, b.ability, b.item, b.nature, sorted(b.moves))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("paste", "compare"))
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--value", type=Path, nargs="+", default=None)
    args = ap.parse_args()
    if args.mode == "paste":
        paste_control(args.data_dir)
    else:
        compare(args.data_dir, args.pairs, args.value or [
            args.data_dir / "models" / "value-mc3.pt", args.data_dir / "models" / "value-mc3-s1.pt"])


if __name__ == "__main__":
    main()
