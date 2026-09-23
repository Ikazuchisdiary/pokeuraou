"""What the lossless branch merge folds, and the proof that it folded nothing else.

Two questions, and they are not the same one.

**Is it lossless?** The merge claims that the branches it folds are the same state reached
twice, so the distribution over successor positions is unchanged. That is checkable: run
the same call twice, once with `merge_duplicates` on and once off, and compare the two
distributions position by position. The key is `to_json`, deliberately -- the resolver's
own comparison is what is on trial, so the check may not use it.

The comparison only means something where *both* runs were exact. When the branch cap
binds, the unmerged run drops its least likely branches and renormalises what is left, so
the two distributions differ because one of them is wrong -- which is the point, and is
counted separately rather than averaged in.

**What does it save?** Branches folded, and the wall clock either way, per call, paired.
Reported per budget, because the two the engine uses are not alike: `Budget.matrix()` pins
one damage roll and caps at 16 branches, so what it can fold is Speed ties that end the
same way and secondaries with nothing left to apply; `Budget.exact()` keeps all 16 rolls,
where a guaranteed knock-out is 16 branches of one position.

Measured on the positions the search meets, not on turn 1 and not on a synthetic case:
the games are played here, through `play_game`, with the agent generation ships. Without
`--value` the player is the hp-share heuristic and the positions are a different player's
-- the run says so in its header.

    uv run python tools/branch_dedup.py --games 2 --value data/models/value-gen11L.pt --hide-bench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.benchflags import add_bench_flags, require_bench  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.resolve import Budget, TurnResult  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


def budget_label(budget: Budget) -> str:
    """The name the engine knows this budget by, from its own fields.

    Read off the budget rather than from the call site: generation resolves with two of
    them and a tool that labels by where the call came from cannot tell them apart.
    """
    if budget.fixed_roll is not None and budget.enumerate_speed_ties:
        return "matrix"
    if budget.fixed_roll is not None:
        return "deterministic"
    if budget.damage_rolls >= 16:
        return "exact"
    return f"rolls={budget.damage_rolls}"


def distribution(result: TurnResult) -> dict[str, float]:
    """Successor position -> probability, by a key the resolver had no hand in.

    A suspended outcome is kept apart from a finished one with the same position: they are
    different things -- one owes a replacement choice -- and folding them together here
    would hide a difference rather than show it.
    """
    out: dict[str, float] = {}
    for branch in result.branches:
        key = json.dumps(branch.position.to_json(), sort_keys=True)
        out[key] = out.get(key, 0.0) + float(branch.probability)
    for pause in result.suspended:
        key = "paused " + json.dumps(pause.position.to_json(), sort_keys=True)
        out[key] = out.get(key, 0.0) + float(pause.probability)
    return out


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


class Tally:
    """One budget's counters."""

    def __init__(self) -> None:
        self.calls = 0
        self.checked = 0
        self.branches_merged = 0
        self.branches_plain = 0
        self.folded = 0
        self.exact_merged = 0
        self.exact_plain = 0
        self.both_exact = 0
        self.worst_tv = 0.0
        self.worst_case: str = ""
        self.rescued = 0
        self.lost_exact = 0
        self.seconds_merged = 0.0
        self.seconds_plain = 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument(
        "--stride",
        type=int,
        default=17,
        help="check one matrix-budget call in this many against the unmerged tree. Every "
        "call is counted; only the checked ones are resolved twice. Budgets other than "
        "the matrix one are checked in full -- there is one per turn against thousands, "
        "so a stride over all calls together would sample them a handful of times and "
        "report a mean of the handful.",
    )
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--value", default=None, help="data/models/*.pt; the agent that ships")
    ap.add_argument("--device", default="cpu")
    add_bench_flags(
        ap,
        hidden_help="hand the search the sheets rather than the opponent's four, as "
        "generation does. One of this or --open-bench is required (IKA-123).",
    )
    args = ap.parse_args()
    require_bench(args)

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    leaf = None
    if args.value:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        device = torch.device(args.device)
        encoder = Encoder(reg)
        net, _meta = load_model(Path(args.value), encoder)
        leaf = BatchedValue(net.to(device), encoder, device=device)
        objective = leaf.objective("win")
    else:
        objective = OBJECTIVES["hp-share"]

    print(f"  roster {args.roster}  games {args.games}  width {args.limit}  seed {args.seed}")
    print(f"  leaf {args.value or 'hp-share'}  rank_by_leaf {bool(leaf)}  hide_bench {args.hide_bench}")
    if not leaf:
        print("  !! no --value: the hp-share heuristic is playing, so these are a "
              "different player's positions than generation reaches")

    # The Python resolver is the subject, so the bridge is off: with it on the port
    # answers the node and this hook never fires.
    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    import pokeuraou.resolve as resolve_mod
    import pokeuraou.search as search_mod
    import pokeuraou.selfplay as selfplay_mod

    real = resolve_mod.resolve_turn
    tallies: dict[str, Tally] = {}
    seen = [0]

    def measuring(reg_, pos, actions, *, budget):  # noqa: ANN001
        seen[0] += 1
        label = budget_label(budget)
        tally = tallies.setdefault(label, Tally())
        tally.calls += 1

        merged_budget = replace(budget, merge_duplicates=True)
        started = time.perf_counter()
        merged = real(reg_, pos, actions, budget=merged_budget)
        merged_seconds = time.perf_counter() - started
        stride = max(1, args.stride) if label == "matrix" else 1
        if tally.calls % stride:
            return merged

        # Alternate which one runs first. The second of a pair reads a warm cache and a
        # warm allocator, and a fixed order would put all of that on one side.
        plain_budget = replace(budget, merge_duplicates=False)
        if tally.checked % 2:
            started = time.perf_counter()
            plain = real(reg_, pos, actions, budget=plain_budget)
            plain_seconds = time.perf_counter() - started
        else:
            started = time.perf_counter()
            plain = real(reg_, pos, actions, budget=plain_budget)
            plain_seconds = time.perf_counter() - started
            started = time.perf_counter()
            merged = real(reg_, pos, actions, budget=merged_budget)
            merged_seconds = time.perf_counter() - started

        tally.checked += 1
        tally.seconds_merged += merged_seconds
        tally.seconds_plain += plain_seconds
        tally.branches_merged += len(merged.branches) + len(merged.suspended)
        tally.branches_plain += len(plain.branches) + len(plain.suspended)
        tally.folded += merged.merged
        tally.exact_merged += int(merged.exact)
        tally.exact_plain += int(plain.exact)
        if merged.exact and not plain.exact:
            tally.rescued += 1
        if plain.exact and not merged.exact:
            tally.lost_exact += 1
        if merged.exact and plain.exact:
            tally.both_exact += 1
            tv = total_variation(distribution(merged), distribution(plain))
            if tv > tally.worst_tv:
                tally.worst_tv = tv
                tally.worst_case = " / ".join(side.to_choice() for side in actions)
        return merged

    resolve_mod.resolve_turn = measuring
    search_mod.resolve_turn = measuring
    selfplay_mod.resolve_turn = measuring

    rng = np.random.default_rng(args.seed)
    try:
        for _ in range(args.games):
            team = pool[int(rng.integers(len(pool)))]
            foe_six = sample_standings_team(rng, reg, prior, team)
            own_pick = selections[int(rng.integers(len(selections)))]
            foe_pick = selections[int(rng.integers(len(selections)))]
            play_game(
                reg,
                rng,
                [roster.sets[i] for i in own_pick],
                [foe_six[j] for j in foe_pick],
                "branch-dedup",
                objective=objective,
                search_limit=args.limit,
                max_turns=args.max_turns,
                evaluate=leaf,
                rank_by_leaf=bool(leaf),
                sheets=(list(roster.sets), list(foe_six)) if args.hide_bench else None,
                open_information=not args.hide_bench,
            )
    finally:
        resolve_mod.resolve_turn = real
        search_mod.resolve_turn = real
        selfplay_mod.resolve_turn = real
        os.environ[rustnode.ENV_ENABLE] = "1"
        rustnode.reset()

    print(f"\n  {seen[0]} calls to resolve_turn\n")
    header = (
        f"  {'budget':>13}  {'calls':>7}  {'checked':>7}  {'branches':>16}  {'folded':>7}  "
        f"{'exact':>13}  {'ms/call':>14}"
    )
    print(header)
    for label, t in sorted(tallies.items()):
        if not t.checked:
            print(f"  {label:>13}  {t.calls:>7}  {0:>7}")
            continue
        plain_mean = t.branches_plain / t.checked
        merged_mean = t.branches_merged / t.checked
        share = 1.0 - (t.branches_merged / t.branches_plain) if t.branches_plain else 0.0
        ms_plain = 1000 * t.seconds_plain / t.checked
        ms_merged = 1000 * t.seconds_merged / t.checked
        print(
            f"  {label:>13}  {t.calls:>7}  {t.checked:>7}  "
            f"{plain_mean:>6.2f} -> {merged_mean:<6.2f}  {share:>6.1%}  "
            f"{t.exact_plain / t.checked:>5.1%} -> {t.exact_merged / t.checked:<5.1%}  "
            f"{ms_plain:>6.3f} -> {ms_merged:<6.3f}"
        )

    print("\n  losslessness (only the calls where both runs were exact):")
    for label, t in sorted(tallies.items()):
        if not t.both_exact:
            continue
        verdict = "lossless" if t.worst_tv < 1e-9 else "DIFFERENT DISTRIBUTION"
        print(
            f"  {label:>13}  {t.both_exact:>5} call(s)  worst total variation "
            f"{t.worst_tv:.3e}  {verdict}"
        )
        if t.worst_tv >= 1e-9:
            print(f"                worst on {t.worst_case}")
    for label, t in sorted(tallies.items()):
        if t.rescued:
            print(
                f"  {label:>13}  {t.rescued} call(s) the cap made inexact without the "
                "merge and exact with it"
            )
        if t.lost_exact:
            print(
                f"  {label:>13}  !! {t.lost_exact} call(s) went the other way: exact "
                "unmerged, inexact merged. The merge must not be able to do this."
            )


if __name__ == "__main__":
    main()
