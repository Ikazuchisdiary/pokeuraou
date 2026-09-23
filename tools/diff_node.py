"""Does a whole node come out the same through the Rust port as without it?

The turn differential compares one `resolve_turn` at a time. This compares what the caller
actually asks for -- a filled matrix -- and it asks for it the way production does, through
`batched_payoffs`, with the bridge on and then off. Cells the port refuses are filled in
Python by the bridge itself, so what is compared is the finished matrix.

    uv run python tools/diff_node.py --games 2 --nodes 20
    uv run python tools/diff_node.py --games 2 --nodes 8 --value data/models/value-gen234.pt

With `--value` the leaves cross as the encoder's arrays rather than the payoff crossing as
a number, because a learned leaf's input *is* the leaves. The forward pass stays in torch
either way.

Exactness: the resolver is bit-identical, so a cell differs only where a weighted mean is
summed in a different order -- numpy's dot product against a sequential loop. The tool
reports how many cells are bit-identical, the worst difference, and what it does to the
equilibrium, which is the number anyone actually reads.

Holding the port to one held item, which is what taking an id into its gate asks for:

    uv run python tools/diff_node.py --games-dir data/selfplay --holding quickclaw
    uv run python tools/diff_node.py --games-dir data/selfplay-gen11L --give focusband

`--games-dir` takes the positions a search already met in recorded games instead of playing
new ones, and `--holding` keeps those with a holder on the field. An item no recorded team
carries can only be handed out, which is `--give`. Either way agreement on a cell the item
never touched is no evidence (IKA-58), so each cell is also resolved in Python with the
item taken off again: the cells whose answer that moves are the ones the item *fired* in,
and those are held to the port branch by branch -- every weight, every note, every
branch's position -- because a wrong 20% can still average to the right cell.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs, resolve_turn  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


def collect_positions(reg, roster, prior, pool, selections, args) -> list:  # noqa: ANN001
    """Positions a real game visits, collected with the bridge off.

    With it on the search never calls Python's resolver, so the hook that collects them
    would see almost nothing. Every cell of a node resolves against the same position, so
    one call in a few hundred is taken rather than all of them.
    """
    import pokeuraou.resolve as resolve_mod
    import pokeuraou.search as search_mod

    was = os.environ.get(rustnode.ENV_ENABLE, "")
    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()

    real = resolve_mod.resolve_turn
    seen: list = []
    calls = [0]

    def recording(reg_, pos, actions, *, budget):  # noqa: ANN001
        calls[0] += 1
        if calls[0] % 137 == 0 and len(seen) < args.nodes * 4:
            seen.append(pos.copy())
        return real(reg_, pos, actions, budget=budget)

    resolve_mod.resolve_turn = recording
    search_mod.resolve_turn = recording
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
            "diff-node",
            objective=OBJECTIVES["hp-share"],
            search_limit=args.limit,
            max_turns=args.max_turns,
        )
    resolve_mod.resolve_turn = real
    search_mod.resolve_turn = real
    os.environ[rustnode.ENV_ENABLE] = was or "1"

    distinct: list = []
    keys: set[str] = set()
    for pos in seen:
        key = str(pos.to_json())
        if key in keys:
            continue
        keys.add(key)
        distinct.append(pos)
        if len(distinct) >= args.nodes:
            break
    return distinct


def on_field(pos: Position, items: frozenset[str]) -> bool:
    """Whether a Pokemon on the field, still standing, holds one of these."""
    return any(
        mon is not None and not mon.fainted and mon.item in items
        for side in pos.sides
        for mon in side.active_pokemon()
    )


def recorded_positions(reg, args, holding: frozenset[str]) -> tuple[list[Position], int]:  # noqa: ANN001
    """Roots a search already filled a matrix at, read from recorded games.

    No game is played, so nothing here is generation. With `holding` every game is read,
    and a game is parsed only if its text names one of the items: a pool can carry an item
    once in a thousand games, and a sample of the first few hundred would find none of it.
    Returns the positions and how many were passed over for being another regulation's.
    """
    found: list[Position] = []
    keys: set[str] = set()
    other_format = 0
    enough = None if holding else 40 * args.nodes
    for directory in args.games_dir:
        for path in sorted(Path(directory).glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if holding and not any(item in line for item in holding):
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for decision in record.get("decisions", ()):
                        if decision.get("kind") != "move":
                            continue
                        if int(decision.get("turn", 0)) < args.min_turn:
                            continue
                        raw = decision["position"]
                        if raw.get("format") != reg.meta.format_id:
                            other_format += 1
                            continue
                        key = json.dumps(raw, sort_keys=True)
                        if key in keys:
                            continue
                        pos = Position.from_json(raw)
                        if holding and not on_field(pos, holding):
                            continue
                        keys.add(key)
                        found.append(pos)
                    if enough is not None and len(found) >= enough:
                        break
            if enough is not None and len(found) >= enough:
                break
    random.Random(args.seed).shuffle(found)
    return found[: args.nodes], other_format


def give(reg, pos: Position, item: str) -> int:  # noqa: ANN001
    """Hands `item` to every Pokemon on the field, and says how many took it.

    Not to one holding its mega stone: the stone is what the mega action is read from, so
    taking it away changes the menu rather than the item.
    """
    given = 0
    for side in pos.sides:
        for mon in side.active_pokemon():
            if mon is None or mon.fainted or (mon.item or "") in reg.mega_map:
                continue
            mon.item = mon.base_item = item
            given += 1
    return given


def without(pos: Position, items: frozenset[str]) -> tuple[Position, list[tuple]]:
    """The position with these items taken off every Pokemon holding one -- the control.

    Also returns who lost what, by side and base species rather than party slot: a slot is
    renumbered by a switch, and the branches this is compared through are after one.
    """
    bare = pos.copy()
    taken: list[tuple] = []
    for side_index, side in enumerate(bare.sides):
        for mon in side.pokemon:
            if mon.item in items:
                taken.append((side_index, mon.base_species, mon.item, mon.base_item))
                mon.item = mon.base_item = None
    return bare, taken


def outcome(result, taken: list[tuple] | None = None) -> tuple:  # noqa: ANN001
    """A turn's answer in a form two turns can be compared by.

    Each distinct state with its total weight, the paused ones apart, and the notes. With
    `taken`, the items the control took off are put back on each state first, so what is
    left to differ is only what the item *did*.
    """

    def key(pos: Position) -> str:
        if taken:
            pos = pos.copy()
            for side_index, species, item, base_item in taken:
                for mon in pos.sides[side_index].pokemon:
                    if mon.base_species == species and mon.item is None:
                        mon.item, mon.base_item = item, base_item
        return json.dumps(pos.to_json(), sort_keys=True)

    finished: dict[str, float] = {}
    for branch in result.branches:
        state = key(branch.position)
        finished[state] = finished.get(state, 0.0) + branch.probability
    paused: dict[str, float] = {}
    for pause in result.suspended:
        state = key(pause.position)
        paused[state] = paused.get(state, 0.0) + pause.probability
    return finished, paused, tuple(sorted(result.unmodelled))


def differ(a: tuple, b: tuple) -> bool:
    for mine, theirs in zip(a[:2], b[:2], strict=True):
        if set(mine) != set(theirs):
            return True
        if any(abs(mine[state] - theirs[state]) > 1e-12 for state in mine):
            return True
    return a[2] != b[2]


def branch_differences(node, reg, pos, a, b, here, budget) -> list[str]:  # noqa: ANN001
    """One cell through the port against Python's own result for it, branch by branch.

    The equality `pokeuraou-damage turns` holds a fixture to -- weights, suspended weights,
    notes, and each branch's position -- asked of the warm process instead of a file.
    """
    there = node.resolve(pos, [a, b], budget)
    if there is None:
        return ["the port refused the turn"]
    wrong: list[str] = []
    mine = [branch.probability for branch in here.branches]
    if len(mine) != len(there.branches) or any(
        abs(x - y) > 1e-12 for x, y in zip(mine, there.branches, strict=False)
    ):
        wrong.append(f"branch weights: python {mine}, rust {there.branches}")
    paused = [pause.probability for pause in here.suspended]
    if len(paused) != len(there.suspended) or any(
        abs(x - y) > 1e-12 for x, y in zip(paused, there.suspended, strict=False)
    ):
        wrong.append(f"suspended weights: python {paused}, rust {there.suspended}")
    if sorted(here.unmodelled) != sorted(there.unmodelled):
        wrong.append(
            f"notes: python {sorted(here.unmodelled)}, rust {sorted(there.unmodelled)}"
        )
    if wrong:
        return wrong
    for index, branch in enumerate(here.branches):
        chosen = node.resolve(pos, [a, b], budget, select=index)
        if chosen is None or chosen.position is None:
            return [f"branch {index}: the port gave no position"]
        if chosen.position.to_json() != branch.position.to_json():
            wrong.append(f"branch {index} position differs")
    return wrong


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--nodes", type=int, default=20, help="stop after this many nodes")
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--value",
        default=None,
        help="a trained value function (data/models/*.pt). The leaves then cross as the "
        "encoder's arrays instead of the payoff crossing as a number.",
    )
    ap.add_argument(
        "--games-dir",
        action="append",
        type=Path,
        default=None,
        help="take the positions from these recorded games (*.jsonl) instead of playing "
        "new ones. Repeatable.",
    )
    ap.add_argument(
        "--min-turn", type=int, default=1, help="with --games-dir, skip earlier decisions"
    )
    ap.add_argument(
        "--holding",
        default=None,
        help="comma-separated item ids: keep positions with a holder on the field, and "
        "resolve each cell again with the item taken off, to find where it fired",
    )
    ap.add_argument(
        "--give",
        default=None,
        help="an item id to hand every Pokemon on the field first -- for an item no "
        "recorded team carries. Implies --holding of the same item.",
    )
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    holding = frozenset(i for i in (args.holding or "").split(",") if i)
    if args.give:
        holding |= {args.give}

    if args.value:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        torch.set_num_threads(1)
        encoder = Encoder(reg)
        net, _meta = load_model(Path(args.value), encoder)
        leaf = BatchedValue(net.to(torch.device("cpu")), encoder, device=torch.device("cpu"))
        # The analyser's own pair: the learned objective and a parameter-free one beside
        # it as a cross-check. Both scored from the same leaves, one here and one there.
        names = ["the learned leaf", "hp-share"]
        evaluators = [leaf, OBJECTIVES["hp-share"].batch]
    else:
        names = ["hp-share", "faints"]
        evaluators = [OBJECTIVES[name].batch for name in names]

    if args.games_dir:
        positions, other_format = recorded_positions(reg, args, holding - {args.give})
        print(
            f"{len(positions)} recorded positions from "
            f"{', '.join(str(d) for d in args.games_dir)}"
            + (f" ({other_format} of another regulation passed over)" if other_format else "")
        )
    else:
        prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
        standings = load_standings(find_cached_standings(), reg)
        pool = standings.pool("all")
        selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
        positions = collect_positions(reg, roster, prior, pool, selections, args)
    if args.give:
        given = sum(give(reg, pos, args.give) for pos in positions)
        print(f"{args.give} handed to {given} Pokemon on the field")
    if holding:
        positions = [pos for pos in positions if on_field(pos, holding)]
    build = rustnode.require_current_binary()
    print(f"binary {build['sha256']} built {build['built']}")

    # A refused cell is filled by Python, and for a learned leaf that means its leaves are
    # scored in a forward pass of their own rather than with the rest of the node. float32
    # matrix arithmetic is not shape-independent, so that alone can move a cell -- which is
    # why the count is reported next to the worst difference rather than left implicit.
    # By reason, because "refused" is not a piece of work anyone can pick up.
    refused: Counter = Counter()
    for name in ("fill", "fill_encoded"):
        original = getattr(rustnode.RustNode, name)

        def counting(self, *args, _original=original, **kwargs):  # noqa: ANN001
            filled = _original(self, *args, **kwargs)
            refused.update(why for _i, _j, why in filled.refused)
            return filled

        # On the class, so the count survives the `reset()` between the two runs.
        setattr(rustnode.RustNode, name, counting)

    budget = Budget.matrix()
    checked = cells = identical = differing = 0
    worst = worst_value = worst_strategy = 0.0
    rust_seconds = python_seconds = 0.0
    notes_differ = 0
    # Where the held item fired, and what the port did there.
    fired = fired_wrong = fired_wrong_anyway = 0
    fired_worst = 0.0
    fired_notes: Counter = Counter()
    shown = 0

    for pos in positions:
        row = narrow(reg, pos, 0, limit=args.limit).actions
        col = narrow(reg, pos, 1, limit=args.limit).actions
        if not row or not col:
            continue

        os.environ[rustnode.ENV_ENABLE] = "0"
        rustnode.reset()
        started = time.perf_counter()
        expected, notes, python_exact = batched_payoffs(
            reg, pos, row, col, evaluators, budget=budget
        )
        python_seconds += time.perf_counter() - started

        os.environ[rustnode.ENV_ENABLE] = "1"
        rustnode.reset()
        started = time.perf_counter()
        got, rust_notes, rust_exact = batched_payoffs(
            reg, pos, row, col, evaluators, budget=budget
        )
        rust_seconds += time.perf_counter() - started
        if set(notes) != set(rust_notes):
            notes_differ += 1
            print(
                f"  the notes differ: python only {sorted(set(notes) - set(rust_notes))}, "
                f"rust only {sorted(set(rust_notes) - set(notes))}"
            )

        if holding:
            # The control, cell by cell and in Python: the same turn with the item off.
            bare, taken = without(pos, holding)
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    control = resolve_turn(reg, bare, [a, b], budget=budget)
                    if not differ(outcome(here), outcome(control, taken)):
                        continue
                    fired += 1
                    fired_notes.update(set(here.unmodelled) - set(control.unmodelled))
                    for index in range(len(evaluators)):
                        fired_worst = max(
                            fired_worst, abs(float(got[index][i, j] - expected[index][i, j]))
                        )
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    if wrong:
                        fired_wrong += 1
                        # And whether it is the item's at all: a cell the engines disagree
                        # on with the item taken off is a disagreement this item only led
                        # the control to.
                        if node is not None and branch_differences(
                            node, reg, bare, a, b, control, budget
                        ):
                            fired_wrong_anyway += 1
                        if shown < 5:
                            shown += 1
                            print(f"  cell {(i, j)} where the item fired: {wrong[0][:200]}")

        checked += 1
        cells += len(row) * len(col)
        for index, name in enumerate(names):
            gap = np.abs(np.asarray(got[index]) - np.asarray(expected[index]))
            worst = max(worst, float(gap.max()))
            identical += int((gap == 0).sum())
            differing += int((gap > 0).sum())
            if gap.max() > 1e-9:
                where = np.unravel_index(int(np.argmax(gap)), gap.shape)
                print(
                    f"  {name} differs by {gap.max():.3e} at cell {where}: "
                    f"python {expected[index][where]!r} rust {got[index][where]!r}"
                )
            python_eq = solve(np.asarray(expected[index]))
            rust_eq = solve(np.asarray(got[index]))
            worst_value = max(worst_value, abs(python_eq.value - rust_eq.value))
            worst_strategy = max(
                worst_strategy,
                float(np.abs(python_eq.row_strategy - rust_eq.row_strategy).max()),
                float(np.abs(python_eq.col_strategy - rust_eq.col_strategy).max()),
            )
        if not np.array_equal(np.asarray(rust_exact), np.asarray(python_exact)):
            print("  the exact mask differs")

    rustnode.reset()
    scored = identical + differing
    leaf_kind = "a learned leaf and hp-share" if args.value else "hp-share and faints"
    print(f"\n{checked} nodes, {cells} cells, scored by {leaf_kind}")
    print(
        f"  bit-identical {identical}/{scored} scored cells "
        f"({identical / max(scored, 1) * 100:.2f}%)"
    )
    # Both runs fill through the counting wrapper, but only the bridged one reaches the
    # port, so this is the bridged run's refusals and nothing is counted twice.
    print(f"  cells the port refused {sum(refused.values())}")
    for why, times in refused.most_common(8):
        print(f"    {times:>7}  {why}")
    print(f"  worst cell difference {worst:.3e}")
    print(f"  equilibrium value moved at most {worst_value:.3e}")
    print(f"  equilibrium frequency moved at most {worst_strategy:.3e}")
    print(f"  nodes whose notes differ {notes_differ}")
    if holding:
        print(f"\n  where {', '.join(sorted(holding))} fired -- the cells the control moves")
        print(f"    {fired} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {fired_wrong}")
        print(f"      of which differ the same way with the item taken off  {fired_wrong_anyway}")
        print(f"    worst cell difference there  {fired_worst:.3e}")
        for note, times in fired_notes.most_common(4):
            print(f"    note only the item's turn carries, {times} cells: {note}")
    print(f"  python {python_seconds:.2f} s   rust {rust_seconds:.2f} s")
    if rust_seconds > 0:
        print(f"  end to end {python_seconds / rust_seconds:.1f}x")


if __name__ == "__main__":
    main()
