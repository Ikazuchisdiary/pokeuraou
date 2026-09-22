"""How many cells does the equilibrium actually need?

A node at width 48 resolves 2,304 cells, and the resolver is 57% of a generation run. But
an equilibrium does not need every cell: to show that an unplayed action is not a better
reply, you need its payoff against the *opponent's mixed strategy*, and that strategy is
usually supported on a handful of columns.

Double oracle is the standard way to exploit that. Keep restricted row and column sets,
solve the small game, then look for a better reply for each side over all of its actions --
which costs one cell per (action, supported column). Add whatever beats the current
strategy and repeat. When neither side can improve, the restricted solution is an
equilibrium of the whole game, and that is a proof rather than an approximation.

This does not implement it. It replays it against matrices that have already been filled,
counts the distinct cells it would have needed, and checks whether the equilibrium it lands
on is the one the full matrix gives -- because the games this project generates are sampled
from that equilibrium, and two equilibria of equal value are still two different games.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--limit", type=int, default=48)
ap.add_argument("--nodes", type=int, default=8)
ap.add_argument("--seed", type=int, default=404)
ap.add_argument("--value", default=None, help="a file in data/models; omit for hp-share")
ap.add_argument("--device", default="cuda")
_args = ap.parse_args()
limit, wanted, seed, model = _args.limit, _args.nodes, _args.seed, _args.value


def double_oracle(payoff: np.ndarray, tolerance: float = 1e-12):
    """Replays double oracle against a matrix that is already filled.

    Returns the equilibrium it reaches, the set of cells it needed, and how many rounds.
    """
    rows, cols = payoff.shape
    used: set[tuple[int, int]] = set()
    # Start from one action each. Which one does not matter for correctness; the first is
    # the highest-scoring candidate, which is what `narrow` puts there.
    restricted_rows, restricted_cols = [0], [0]
    rounds = 0
    while True:
        rounds += 1
        for i in restricted_rows:
            for j in restricted_cols:
                used.add((i, j))
        sub = payoff[np.ix_(restricted_rows, restricted_cols)]
        equilibrium = solve(sub)
        x = np.asarray(equilibrium.row_strategy)
        y = np.asarray(equilibrium.col_strategy)

        # The row player's best reply over every row, against y. Only the columns y puts
        # weight on are needed, which is the whole point.
        live_cols = [restricted_cols[k] for k in range(len(y)) if y[k] > tolerance]
        weights = np.asarray([y[k] for k in range(len(y)) if y[k] > tolerance])
        for i in range(rows):
            for j in live_cols:
                used.add((i, j))
        against_y = payoff[:, live_cols] @ weights
        best_row = int(np.argmax(against_y))

        live_rows = [restricted_rows[k] for k in range(len(x)) if x[k] > tolerance]
        row_weights = np.asarray([x[k] for k in range(len(x)) if x[k] > tolerance])
        for i in live_rows:
            for j in range(cols):
                used.add((i, j))
        against_x = row_weights @ payoff[live_rows, :]
        best_col = int(np.argmin(against_x))

        grew = False
        if against_y[best_row] > equilibrium.value + 1e-9 and best_row not in restricted_rows:
            restricted_rows.append(best_row)
            grew = True
        if against_x[best_col] < equilibrium.value - 1e-9 and best_col not in restricted_cols:
            restricted_cols.append(best_col)
            grew = True
        if not grew or rounds > 60:
            full_x = np.zeros(rows)
            full_y = np.zeros(cols)
            for k, i in enumerate(restricted_rows):
                full_x[i] = x[k]
            for k, j in enumerate(restricted_cols):
                full_y[j] = y[k]
            return equilibrium.value, full_x, full_y, used, rounds


roster = load_roster("rizabanadohido")
reg = roster.reg
register_mega_stones(reg)
prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
standings = load_standings(find_cached_standings(), reg)
pool = standings.pool("all")
selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

if model:
    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.value import BatchedValue, load_model

    encoder = Encoder(reg)
    net, _meta = load_model(ROOT / "data" / "models" / model, encoder)
    leaf = BatchedValue(net.to(torch.device(_args.device)), encoder, device=torch.device(_args.device))
    scorer = leaf
    playing = leaf.objective("win")
else:
    scorer = OBJECTIVES["hp-share"].batch
    playing = OBJECTIVES["hp-share"]

# Collected with the bridge off: with it on the port answers and this hook never fires.
os.environ[rustnode.ENV_ENABLE] = "0"
rustnode.reset()
import pokeuraou.resolve as resolve_mod  # noqa: E402
import pokeuraou.search as search_mod  # noqa: E402

real = resolve_mod.resolve_turn
seen: list = []
calls = [0]


keys: set = set()


def recording(reg_, pos, actions, *, budget):
    # Every cell of a node resolves the same position, so taking every Nth call takes the
    # same node over and over. Distinct positions or nothing.
    calls[0] += 1
    if calls[0] % 89 == 0 and len(seen) < wanted:
        key = str(pos.to_json())
        if key not in keys:
            keys.add(key)
            seen.append(pos.copy())
    return real(reg_, pos, actions, budget=budget)


resolve_mod.resolve_turn = recording
search_mod.resolve_turn = recording
rng = np.random.default_rng(seed)
for _ in range(3):
    team = pool[int(rng.integers(len(pool)))]
    foe_six = sample_standings_team(rng, reg, prior, team)
    own = selections[int(rng.integers(len(selections)))]
    foe = selections[int(rng.integers(len(selections)))]
    play_game(
        reg, rng,
        [roster.sets[i] for i in own],
        [foe_six[j] for j in foe],
        "oracle",
        objective=playing,
        search_limit=limit,
        max_turns=40,
    )
    if len(seen) >= wanted:
        break
resolve_mod.resolve_turn = real
search_mod.resolve_turn = real
os.environ[rustnode.ENV_ENABLE] = "1"
rustnode.reset()
print(f"{len(seen)} nodes")

print(f"{'node':>10}  {'cells':>7}  {'needed':>7}  {'share':>6}  {'rounds':>6}  "
      f"{'supp':>7}  {'value gap':>10}  {'same play':>9}")
total_cells = total_used = 0
for pos in seen:
    row = narrow(reg, pos, 0, limit=limit).actions
    col = narrow(reg, pos, 1, limit=limit).actions
    if len(row) < 2 or len(col) < 2:
        continue
    payoffs, _notes, _exact = batched_payoffs(
        reg, pos, row, col, [scorer], budget=Budget.matrix()
    )
    matrix = np.asarray(payoffs[0])
    whole = solve(matrix)
    value, x, y, used, rounds = double_oracle(matrix)

    cells = matrix.size
    total_cells += cells
    total_used += len(used)
    support = f"{int((x > 1e-12).sum())}x{int((y > 1e-12).sum())}"
    moved = max(
        float(np.abs(x - np.asarray(whole.row_strategy)).max()),
        float(np.abs(y - np.asarray(whole.col_strategy)).max()),
    )
    print(
        f"{matrix.shape[0]:>4}x{matrix.shape[1]:<5} {cells:>7}  {len(used):>7}  "
        f"{len(used) / cells:>5.0%}  {rounds:>6}  {support:>7}  "
        f"{abs(value - whole.value):>10.2e}  {'yes' if moved < 1e-9 else f'no ({moved:.2f})':>9}"
    )

if total_cells:
    print(f"\n{total_used}/{total_cells} cells = {total_used / total_cells:.0%} of the matrix")
rustnode.reset()
