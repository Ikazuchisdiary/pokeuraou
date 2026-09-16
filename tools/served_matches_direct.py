"""Does a match through the inference server play the same games as one without?

This is the acceptance condition for moving the leaves out of the workers, and it is not
"about as strong". A difference of 1.9e-06 in a leaf moves an equilibrium, an equilibrium
moves a mixed strategy, and a draw from a different mixture is a different game. So the
test is exact: same seed, same settings, and every recorded decision -- the menu, the
mixture, and the action drawn from it -- must match action for action.

Weaker checks would pass while being wrong. A win rate would need thousands of games to
notice a small bias, the equilibrium *value* is a float that can agree while the strategy
differs, and the outcome is one bit per game. The decision sequence is the whole strategy,
recorded, and it is free to compare because generation already writes it down.

The server is started and stopped here, so the run is self-contained.

    uv run --group learn python tools/served_matches_direct.py --games 6 --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _decisions(path: Path) -> list[tuple]:
    """Every decision of every game, flattened, in the order they were played.

    Includes both sides' menus and both sides' mixtures, not only what was drawn: two
    agents can draw the same action from different mixtures for a while, and that is a
    disagreement that has not surfaced yet rather than an agreement.
    """
    out: list[tuple] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for turn in record.get("decisions", ()):
            out.append((
                turn.get("turn"),
                turn.get("kind"),
                tuple(turn.get("ownActions") or ()),
                tuple(turn.get("foeActions") or ()),
                tuple(turn.get("ownPolicy") or ()),
                tuple(turn.get("foePolicy") or ()),
                turn.get("ownChosen"),
                turn.get("foeChosen"),
            ))
    return out


def _run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        done = subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=handle,  # noqa: S603
                              check=False)
    if done.returncode != 0:
        raise SystemExit(f"{command[1]} exited {done.returncode}; see {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=6, help="per seat, so twice this")
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--value", nargs="+",
                    default=["data/models/value-all.pt", "data/models/value-all-s1.pt"])
    ap.add_argument("--baseline", nargs="+", default=["data/models/value-gen8.pt"])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "matches" / "_served_check")
    args = ap.parse_args()

    out: Path = args.out
    for name in ("direct", "served"):
        (out / name).mkdir(parents=True, exist_ok=True)
        for stale in (out / name).glob("*.jsonl"):
            stale.unlink()

    shared = [
        "--seed", str(args.seed), "--games", str(args.games), "--device", args.device,
        "--torch-threads", "1", "--limit", str(args.limit),
        "--rank-leaf", "--baseline-rank-leaf",
    ]

    print(f"direct: both leaves in the worker ({args.device})", flush=True)
    started = time.perf_counter()
    _run(
        [sys.executable, str(ROOT / "tools" / "generation_match.py"),
         "--value", *args.value, "--baseline", *args.baseline,
         "--games-out", str(out / "direct" / "games.jsonl"), *shared],
        out / "direct.log",
    )
    direct_seconds = time.perf_counter() - started

    print("served: the leaves in one process, the worker holding none", flush=True)
    server_log = (out / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(  # noqa: S603
        [sys.executable, str(ROOT / "tools" / "inference_server.py"),
         "--device", args.device, "--arm", "value", *args.value,
         "--arm", "baseline", *args.baseline],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=server_log, text=True,
    )
    try:
        assert server.stdout is not None
        address = server.stdout.readline().strip()
        if not address:
            raise SystemExit(f"the server named no address; see {out / 'server.log'}")
        print(f"  server at {address}", flush=True)
        started = time.perf_counter()
        _run(
            [sys.executable, str(ROOT / "tools" / "generation_match.py"),
             "--inference", address, "--inference-arm", "value",
             "--baseline-inference-arm", "baseline",
             "--games-out", str(out / "served" / "games.jsonl"), *shared],
            out / "served.log",
        )
        served_seconds = time.perf_counter() - started
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    left = _decisions(out / "direct" / "games.jsonl")
    right = _decisions(out / "served" / "games.jsonl")
    print(f"\n  direct {len(left):,} decisions in {direct_seconds:.1f}s")
    print(f"  served {len(right):,} decisions in {served_seconds:.1f}s "
          f"({served_seconds / max(direct_seconds, 1e-9):.2f}x)")

    if not left:
        raise SystemExit("the direct run recorded nothing -- there is nothing to compare")
    if len(left) != len(right):
        raise SystemExit(
            f"FAIL: {len(left)} decisions against {len(right)}. The two runs diverged and "
            f"then played different-length games."
        )
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            fields = ["turn", "kind", "ownActions", "foeActions", "ownPolicy",
                      "foePolicy", "ownChosen", "foeChosen"]
            differing = [f for f, x, y in zip(fields, a, b, strict=True) if x != y]
            raise SystemExit(
                f"FAIL at decision {index} (turn {a[0]}): {', '.join(differing)} differ.\n"
                f"  direct {a}\n  served {b}"
            )
    print(f"\n  PASS: {len(left):,} decisions identical, menu, mixture and draw")


if __name__ == "__main__":
    main()
