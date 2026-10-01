"""Capture the Bayesian games the generation-form search solves (IKA-196 stage 0).

Runs tools/pool_match.py in this process with both arms lp (value-mc4x2 on the CPU) and
records every `equilibrium.solve_bayesian` input: move nodes (belief_solve) and
replacements alike. Writes C:/tmp/ika196/captured.pkl.
"""

import pickle
import sys
from pathlib import Path

WT = Path("C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-acde882fb885a4aa0")
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "tools"))

import numpy as np  # noqa: E402

from pokeuraou import equilibrium  # noqa: E402

captured = []
real = equilibrium.solve_bayesian


def spy(matrices, weights, eps=1e-9):  # noqa: ANN001, ANN201
    captured.append(([np.array(m, dtype=np.float64) for m in matrices],
                     np.array(weights, dtype=np.float64)))
    return real(matrices, weights, eps)


equilibrium.solve_bayesian = spy

import pool_match  # noqa: E402

games = sys.argv[1] if len(sys.argv) > 1 else "10"
value = [str(M / "data/models/value-mc4.pt"), str(M / "data/models/value-mc4-s1.pt")]
pool_match.main([
    "--pool", str(M / "data/pool/regmc-matchupweb.json"), "--hide-bench",
    "--value", *value, "--baseline", *value,
    "--q-model", str(M / "data/models/q-mc4.pt"),
    "--limit", "12", "--rank-leaf", "--baseline-rank-leaf",
    "--selection-store", "C:/tmp/ika196/store-mc4x2",
    "--baseline-selection-store", "C:/tmp/ika196/store-mc4x2-b",
    "--games", games, "--seed", "19690", "--device", "cpu",
    "--games-out", "C:/tmp/ika196/capture/games.jsonl",
])
with open("C:/tmp/ika196/captured.pkl", "wb") as fh:
    pickle.dump(captured, fh)
print("captured", len(captured), "games", file=sys.stderr)
