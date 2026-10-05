"""The same picked games as sample.py (first N held-out gen-4 games), written to picked.npy so
predict.py can start before sample.py ends. analyze.py checks the two agree row by row."""
import sys
from pathlib import Path

import numpy as np

DATA = Path("C:/Users/Ikazuchi/repos/pokeuraou/data")
N = int(sys.argv[1])
merged = np.load(DATA / "selfplay-mc01234-encoded.npz")["game"]
games = np.unique(merged)
np.random.default_rng(0).shuffle(games)
val = np.sort(games[int(len(games) * 0.85):])
offset = sum(int(np.load(DATA / f"selfplay-mc{g}-encoded.npz")["game"].max()) + 1 for g in range(4))
count = int(np.load(DATA / "selfplay-mc4-encoded.npz")["game"].max()) + 1
mine = val[(val >= offset) & (val < offset + count)] - offset
np.save("C:/tmp/ika434/picked.npy", np.sort(mine)[:N])
print(len(mine), N)
