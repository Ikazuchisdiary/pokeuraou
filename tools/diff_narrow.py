"""Does the menu come out the same when the port scores it?

`narrow` decides which choices reach the matrix at all, so a difference here is not a
difference in a number -- it is a different set of candidates, which is a different game.
That makes this the strictest of the differentials: the scores have to be *bit* equal, not
close, because the order they produce is what survives.

    POKEURAOU_RUST_NODE=1 uv run python tools/diff_narrow.py --games 3

It reports how many pools were scored identically, the worst difference in a score, and
whether the kept list or its order ever differed -- which is the thing that would matter.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


def collect_positions(reg, roster, prior, pool, selections, args) -> list:  # noqa: ANN001
    """Positions a real game visits, with the bridge off so the hook sees them all."""
    import pokeuraou.narrow as narrow_mod
    import pokeuraou.search as search_mod
    import pokeuraou.selfplay as selfplay_mod

    was = os.environ.get(rustnode.ENV_ENABLE, "")
    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()

    real = narrow_mod.narrow
    seen: list = []

    def recording(reg_, pos, side, **kwargs):  # noqa: ANN001
        if len(seen) < args.pools:
            seen.append((pos.copy(), side, dict(kwargs)))
        return real(reg_, pos, side, **kwargs)

    narrow_mod.narrow = recording
    search_mod.narrow = recording
    selfplay_mod.narrow = recording
    rng = np.random.default_rng(args.seed)
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
            "diff-narrow",
            objective=OBJECTIVES["hp-share"],
            search_limit=args.limit,
            max_turns=args.max_turns,
        )
        if len(seen) >= args.pools:
            break
    narrow_mod.narrow = real
    search_mod.narrow = real
    selfplay_mod.narrow = real
    os.environ[rustnode.ENV_ENABLE] = was or "1"
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--pools", type=int, default=200, help="stop after this many narrow calls")
    ap.add_argument("--roster", default="rizabanadohido")
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    positions = collect_positions(reg, roster, prior, pool, selections, args)
    build = rustnode.require_current_binary()
    print(f"binary {build['sha256']} built {build['built']}")

    checked = candidates = identical = 0
    worst = 0.0
    kept_differed = order_differed = detail_differed = 0
    python_seconds = rust_seconds = 0.0

    # One sweep each way, not a toggle per pool: `reset()` closes the port's process, so
    # flipping per pool would time a process start rather than a score.
    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    started = time.perf_counter()
    mine_all = [narrow(reg, pos, side, **kwargs) for pos, side, kwargs in positions]
    python_seconds = time.perf_counter() - started

    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    narrow(reg, *positions[0][:2], **positions[0][2])  # warm the process, not the clock
    started = time.perf_counter()
    theirs_all = [narrow(reg, pos, side, **kwargs) for pos, side, kwargs in positions]
    rust_seconds = time.perf_counter() - started

    for here, there in zip(mine_all, theirs_all, strict=True):
        checked += 1
        candidates += len(here.kept)
        if [c.action.to_choice() for c in here.kept] != [
            c.action.to_choice() for c in there.kept
        ]:
            if sorted(c.action.to_choice() for c in here.kept) == sorted(
                c.action.to_choice() for c in there.kept
            ):
                order_differed += 1
            else:
                kept_differed += 1
                print(f"  pool {checked}: a different set of candidates survived")
        for mine, theirs in zip(here.kept, there.kept, strict=False):
            if mine.score == theirs.score:
                identical += 1
            worst = max(worst, abs(mine.score - theirs.score))
            if mine.detail != theirs.detail:
                detail_differed += 1
                if detail_differed == 1:
                    print(f"  pool {checked}: detail {mine.detail} against {theirs.detail}")

    rustnode.reset()
    print(f"\n{checked} pools, {candidates} kept candidates")
    print(
        f"  bit-identical scores {identical}/{candidates} "
        f"({identical / max(candidates, 1) * 100:.2f}%)"
    )
    print(f"  worst score difference {worst:.3e}")
    print(f"  pools whose kept set differed  {kept_differed}")
    print(f"  pools whose order differed     {order_differed}")
    print(f"  candidates whose detail differed {detail_differed}")
    print(f"  python {python_seconds:.2f} s   rust {rust_seconds:.2f} s")
    if rust_seconds > 0:
        print(f"  narrow is {python_seconds / rust_seconds:.1f}x faster")


if __name__ == "__main__":
    main()
