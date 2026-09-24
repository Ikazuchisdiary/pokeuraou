"""How much of the menu a cheaper leaf-ranking fill changes, where generation meets it (IKA-268).

A cheaper fill (`search.parse_rank_fill`: fewer replies, or `Budget.fast`) changes which
actions reach the matrix, and a game played with it goes elsewhere after its first changed
decision -- so two generation runs cannot be compared decision by decision. This plays M-C
generation (`poolplay.generate_pool`, the shipping call) with the fill it ships, and at
every move decision also builds the menus each other fill would have built from the same
inputs, then throws them away. The game is the shipping game; only the side menus are
extra.

Per move decision and side it counts whether the other fill's menu is the same set, the
same order, how many of the played menu's actions it keeps, and how much of the played
equilibrium's mass sits on actions it would have dropped -- the part of the change the
game could feel.

    uv run --group learn python tools/rank_fill_menus.py --pool regmc-matchupweb \\
        --value data/models/value-mc0.pt data/models/value-mc0-s1.pt \\
        --store <solved selections for that leaf> --games 240 --jobs 4 \\
        --fills refs2,refs1,refs2-fast

Controls: a fill equal to the played one agrees at every decision (the null), and the
games written with and without the extra menus are the same games (`--games-out`, compared
by the caller). `--jobs 1` and `--jobs N` give the same table: games are seeded by index.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.search import DEFAULT_RANK_FILL, parse_rank_fill  # noqa: E402


def leaf_name(files: list[str]) -> str:
    stem = re.sub(r"-s\d+$", "", Path(files[0]).stem)
    return stem if len(files) == 1 else f"{stem}x{len(files)}"


def compare(played: list[str], other: list[str], mass: list[float] | None) -> dict[str, Any]:
    """One side's played menu against another fill's, with the played strategy if known."""
    kept = set(other)
    row = {
        "same_set": set(played) == kept,
        "same_order": played == other,
        "kept": sum(a in kept for a in played),
        "size": len(played),
    }
    if mass is not None:
        row["dropped_mass"] = float(sum(m for a, m in zip(played, mass, strict=True)
                                        if a not in kept))
    return row


def run_worker(args: argparse.Namespace) -> None:
    import torch

    from pokeuraou import poolplay, selfplay
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.pool import load_pool
    from pokeuraou.value import BatchedValue, load_ensemble

    torch.set_num_threads(1)
    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    encoder = Encoder(reg)
    nets, _ = load_ensemble(args.value, encoder)
    device = torch.device(args.device)
    evaluate = BatchedValue([n.to(device) for n in nets], encoder, device=device)
    label = f"value:{leaf_name(args.value)}"

    shadows = [f for f in args.fills.split(",") if f] if args.fills else []
    # game index -> the menus of each move decision, played and per shadow fill.
    menus: dict[int, list[dict[str, Any]]] = {}
    current: list[int | None] = [None]
    original_menus = selfplay._menus
    original_play = poolplay.play_game

    def menus_with_shadows(*a: Any, **kw: Any) -> Any:  # noqa: ANN401
        got = original_menus(*a, **kw)
        ranked, policy = a[5], a[6] if len(a) > 6 else kw.get("policy")
        if current[0] is None or not ranked or policy is not None:
            return got
        entry = {"played": [[x.to_choice() for x in side] for side in got], "shadows": {}}
        for fill in shadows:
            alt = original_menus(*a, **{**kw, "rank_fill": fill, "used": {}})
            entry["shadows"][fill] = [[x.to_choice() for x in side] for side in alt]
        menus.setdefault(current[0], []).append(entry)
        return got

    def play_counted(*a: Any, **kw: Any) -> Any:  # noqa: ANN401
        if kw.get("rank_fill", DEFAULT_RANK_FILL) != args.played:
            raise SystemExit(f"generation passed rank_fill {kw.get('rank_fill')!r}")
        return original_play(*a, **kw)

    selfplay._menus = menus_with_shadows
    poolplay.play_game = play_counted
    indices = [args.first_game + i for i in range(args.games) if i % args.jobs == args.worker]

    def scheduled() -> Any:  # noqa: ANN401
        for index in indices:
            current[0] = index
            yield index
        current[0] = None

    out = Path(args.games_out) if args.games_out else Path(args.rows).with_suffix(".games.jsonl")
    out.unlink(missing_ok=True)
    poolplay.generate_pool(
        reg, pool, games=len(indices), hide_bench=True, seed=args.seed, out=out,
        evaluate=evaluate, store=Path(args.store), leaf=label,
        search_limit=args.limit, rank_by_leaf=True, rank_fill=args.played,
        indices=scheduled(),
    )
    rows = []
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        game = json.loads(line)
        index = game["gameIndex"]
        moves = [d for d in game["decisions"] if d["kind"] == "move"]
        seen = menus.get(index, [])
        if len(seen) != len(moves):
            raise SystemExit(f"game {index}: {len(seen)} menus for {len(moves)} move decisions")
        for number, (decision, entry) in enumerate(zip(moves, seen, strict=True)):
            played = entry["played"]
            if decision["ownActions"] != played[0] or decision["foeActions"] != played[1]:
                raise SystemExit(f"game {index} decision {number}: menus out of step")
            masses = (decision["ownPolicy"], decision["foePolicy"])
            for fill, alt in entry["shadows"].items():
                for side in (0, 1):
                    rows.append({
                        "game": index, "decision": number, "turn": decision["turn"],
                        "side": side, "fill": fill,
                        **compare(played[side], alt[side], masses[side]),
                    })
    Path(args.rows).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )


def summarise(rows: list[dict[str, Any]], fills: list[str]) -> list[str]:
    lines = [
        "  fill          decisions  sides   same set (decision)  same set (side)  "
        "same order (side)  kept/size   dropped eq. mass (mean, share > 0)",
    ]
    for fill in fills:
        mine = [r for r in rows if r["fill"] == fill]
        if not mine:
            continue
        by_decision: dict[tuple[int, int], bool] = {}
        for r in mine:
            key = (r["game"], r["decision"])
            by_decision[key] = by_decision.get(key, True) and r["same_set"]
        n = len(mine)
        mass = [r["dropped_mass"] for r in mine]
        lines.append(
            f"  {fill:<12} {len(by_decision):>10} {n:>6}   "
            f"{sum(by_decision.values()) / len(by_decision) * 100:>17.1f}%  "
            f"{sum(r['same_set'] for r in mine) / n * 100:>14.1f}%  "
            f"{sum(r['same_order'] for r in mine) / n * 100:>16.1f}%  "
            f"{sum(r['kept'] for r in mine) / sum(r['size'] for r in mine) * 100:>8.1f}%   "
            f"{sum(mass) / n:.4f}, {sum(m > 1e-9 for m in mass) / n * 100:.1f}%"
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="regmc-matchupweb")
    ap.add_argument("--value", nargs="+", required=True)
    ap.add_argument("--store", required=True,
                    help="the solved selections for this leaf (read; a missing pair is solved)")
    ap.add_argument("--seed", type=int, default=7701)
    ap.add_argument("--first-game", type=int, default=0)
    ap.add_argument("--games", type=int, default=240)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--played", default=DEFAULT_RANK_FILL, help="the fill the games play")
    ap.add_argument("--fills", default="refs2,refs1,refs2-fast",
                    help="comma-separated fills whose menus are built beside the played one")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True, help="a directory for rows and games")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rows", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--games-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    for fill in [args.played, *[f for f in args.fills.split(",") if f]]:
        parse_rank_fill(fill)
    if args.worker is not None:
        run_worker(args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    workers = []
    for k in range(args.jobs):
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", str(k),
            "--pool", args.pool, "--value", *args.value, "--store", args.store,
            "--seed", str(args.seed), "--first-game", str(args.first_game),
            "--games", str(args.games), "--limit", str(args.limit), "--played", args.played,
            "--fills", args.fills, "--jobs", str(args.jobs), "--device", args.device,
            "--out", str(args.out),
            "--rows", str(args.out / f"rows{k}.jsonl"),
            "--games-out", str(args.out / f"games{k}.jsonl"),
        ]
        log = (args.out / f"worker{k}.log").open("w", encoding="utf-8")
        workers.append((subprocess.Popen(command, stdout=log, stderr=log), log))  # noqa: S603
    failed = 0
    for process, log in workers:
        failed += process.wait() != 0
        log.close()
    if failed:
        raise SystemExit(f"{failed} worker(s) failed; see {args.out}/worker*.log")
    rows = [
        json.loads(line)
        for k in range(args.jobs)
        for line in (args.out / f"rows{k}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda r: (r["fill"], r["game"], r["decision"], r["side"]))
    fills = [f for f in args.fills.split(",") if f]
    report = [
        f"rank_fill_menus: {args.games} games from {args.first_game}, seed {args.seed}, "
        f"width {args.limit}, leaf {leaf_name(args.value)}, played {args.played}, "
        f"{args.jobs} job(s)",
        *summarise(rows, fills),
    ]
    print("\n".join(report))
    (args.out / "summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8",
                                          newline="\n")
    (args.out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                         encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
