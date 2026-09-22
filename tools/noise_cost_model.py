"""What does +-11 points of cell error cost in win rate? A model, not a measurement.

Two inputs are measured (G27/G28): matchup values are spread with SD 17.2 points, and
the leaf's estimate of a cell carries SD 11.3 points of error. Everything else here is
assumed, so the output is a model's answer and is labelled as one. The question it
settles is whether 11.3 points of cell error means 11.3 points of lost win rate -- it
does not, because solving a 90x90 game averages over the errors, and the size of what
survives is the thing worth knowing.

Two structures, because the honest worry about an i.i.d. matrix is that real selections
share Pokemon and so their cells are correlated:

  iid        every cell drawn independently
  latent     each selection gets a latent vector; the cell is their interaction, so
             selections that share Pokemon move together

Three opponents, because who we are wrong against changes the answer:

  truth      they solve the true matrix. The worst case for us
  noisy      they solve their own independent estimate. What book-against-book is
  uniform    they pick at random. What book-against-uniform measured at +18.9
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.equilibrium import solve  # noqa: E402

N = 90
SD_TRUE = 0.172  # measured: spread of matchup values
SD_NOISE = 0.113  # measured: the leaf's error on a cell
TRIALS = 200
rng = np.random.default_rng(20260919)


def iid_matrix() -> np.ndarray:
    return 0.5 + SD_TRUE * rng.standard_normal((N, N))


def latent_matrix() -> np.ndarray:
    """Cells that share a selection move together, at the same total spread."""
    k = 4
    u = rng.standard_normal((N, k))
    v = rng.standard_normal((N, k))
    raw = u @ v.T
    return 0.5 + SD_TRUE * raw / raw.std()


def value_of(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ a @ y)


print(f"  {TRIALS} trials, {N}x{N}, true spread {SD_TRUE:.3f}, cell error {SD_NOISE:.3f}")
for name, make in (("iid", iid_matrix), ("latent", latent_matrix)):
    rows = {k: [] for k in ("perfect", "noisy-vs-truth", "noisy-vs-noisy", "uniform")}
    gains = {k: [] for k in ("perfect", "noisy")}
    for _ in range(TRIALS):
        a = make()
        ahat = a + SD_NOISE * rng.standard_normal((N, N))
        bhat = a + SD_NOISE * rng.standard_normal((N, N))  # their independent estimate

        true_eq = solve(a)
        x_true = np.asarray(true_eq.row_strategy)
        y_true = np.asarray(true_eq.col_strategy)
        x_hat = np.asarray(solve(ahat).row_strategy)
        # They minimise, so their strategy is the column side of their own solve.
        y_hat = np.asarray(solve(bhat).col_strategy)
        unif = np.full(N, 1.0 / N)

        rows["perfect"].append(value_of(a, x_true, y_true))
        rows["noisy-vs-truth"].append(value_of(a, x_hat, y_true))
        rows["noisy-vs-noisy"].append(value_of(a, x_hat, y_hat))
        rows["uniform"].append(value_of(a, unif, y_true))
        # What the advice is worth against a uniform opponent -- the +18.9 measurement.
        gains["perfect"].append(
            value_of(a, x_true, unif) - value_of(a, unif, unif)
        )
        gains["noisy"].append(value_of(a, x_hat, unif) - value_of(a, unif, unif))

    print(f"\n  {name}")
    base = float(np.mean(rows["perfect"]))
    for key in ("perfect", "noisy-vs-truth", "noisy-vs-noisy", "uniform"):
        mean = float(np.mean(rows[key]))
        print(f"    {key:<16} {mean:6.1%}   noise costs {mean - base:+5.1%}")
    gp, gn = float(np.mean(gains["perfect"])), float(np.mean(gains["noisy"]))
    print(
        f"    advice against a uniform opponent: perfect +{gp:.1%}, "
        f"with the measured error +{gn:.1%}  ({gn / gp:.0%} of it survives)"
    )
