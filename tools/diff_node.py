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

Holding the port to a move, which is what taking a refusal out of the port asks for:

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --using feint

`--using` keeps the positions with a Pokemon on the field that knows the move. Every cell
where an action uses it is held to the port branch by branch -- the port refused all of
them before -- and the cells where the move's effect *fired* are counted apart: for a
`breaksProtect` move (IKA-61), the same turn resolved in Python
with `_break_protection` taken out. A Feint into a foe that did not Protect agrees
without the break ever running, and that is not evidence the break is right (IKA-58).

`--using` also takes a [2, 5] multi-hit move (IKA-160):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --using bulletseed,rockblast

The control is the hit count as it was before IKA-160 -- 1/3, 1/3, 1/6, 1/6 and Skill Link
not read -- so the cells it moves are the ones where the count's distribution mattered.

Holding the port to a terrain no recorded game has (IKA-156):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --terrain psychicterrain

`--terrain` lays Psychic Terrain on each position, keeps those with a priority move (or a
priority-raising ability) on the field, and holds every cell that may use one branch by
branch. The cells where the terrain's per-target stop *fired* are the ones Python moves
with `_stopped_by_psychic_terrain` taken out, and the run fails if there are none.

Holding the port to Salt Cure, which no recorded team uses (IKA-159):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --salt-cure

`--salt-cure` puts the volatile on every Pokemon on the field and holds every cell branch
by branch. The cells where the champions mod's 1/16 and 1/8 *fired* are the ones Python
moves when the base game's 1/8 and 1/4 are put back, and the run fails if there are none.

The budget, and the second invocation that goes with it (IKA-146):

    uv run python tools/diff_node.py --scenario examples/scenario-turn5.json --budget fast --limit 0

`Budget.matrix()` pins the roll, so its `narrowed()` returns itself and a run under it never
takes the path that divides the branch budget among live branches -- nor sees what budget
a turn paused for a mid-turn switch is resumed on. IKA-140's line (the port writing the
narrowed budget into the turn) was on that path from cd23aab until IKA-62 counted leaves,
and this tool, run only under `matrix`, could not see it. `--budget fast` and `--budget
exact` take that path.
The invocation above is the one that does: scenario-turn5's whole menu, 88x52 cells, with
Incineroar's Parting Shot pausing turns after rolls have already branched. Under fast it
takes about ten seconds, nearly all of it Python; the IKA-140 build fails it on 12 cells
(worst 5.3e-4 in hp-share, 0.036 in faints). `--scenario` builds the node the way
`tests/test_rust_node.py` does -- the belief layer's modal spreads -- and `--limit 0` keeps
every legal action rather than narrowing. The whole menu is the point: narrowed to 8 or
12 a side, none of the 12 cells the old line moves is on it, and `--budget exact --limit
12` (30 seconds of Python) passes the IKA-140 build. A whole menu under `exact` is IKA-147's
gigabytes; `tests/test_rust_node.py` holds one such cell under `Budget()` instead.

The run exits 1 when a cell differs by more than `--tolerance`, or when the exact mask
differs on any cell, so a failure is a status and not only a line to be read. Until IKA-151
the mask was let off under `fast` and `exact`: Python marks a turn inexact for "damage
rolls stratified" and the port had no such reduction, so it called 3,599 of this node's
4,576 cells exact that Python does not.
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
from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.position import Effect, Position  # noqa: E402
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
            open_information=True,
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


def knows_on_field(pos: Position, moves: frozenset[str]) -> bool:
    """Whether a Pokemon on the field, still standing, knows one of these moves."""
    return any(
        mon is not None and not mon.fainted and any(slot.id in moves for slot in mon.moves)
        for side in pos.sides
        for mon in side.active_pokemon()
    )


def uses(action, moves: frozenset[str]) -> bool:  # noqa: ANN001
    """Whether a side's action has one of its slots use one of these moves."""
    return any(getattr(slot, "move_id", None) in moves for slot in action.slots)


class unbroken:  # noqa: N801 - read as a phrase at the call site
    """Python with `_break_protection` taken out: the control for a `breaksProtect` move.

    Everything else about the move stays -- it still passes the Protect it no longer
    breaks -- so what differs is only what the break did.
    """

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._break_protection
        resolve_mod._break_protection = lambda *_args: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._break_protection = self.real


class old_hit_counts:  # noqa: N801 - read as a phrase at the call site
    """Python with the hit count as it was before IKA-160: the control for a multi-hit move.

    A [2, 5] move is 1/3, 1/3, 1/6, 1/6 -- the older `sample([2, 2, 3, 3, 4, 5])` -- and the
    user's Skill Link is not read. Everything else about the move stays, so what differs is
    only what the count's distribution did.
    """

    OLD_2_5 = [(2, 1 / 3), (3, 1 / 3), (4, 1 / 6), (5, 1 / 6)]

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = real = resolve_mod.multihit_counts
        now_2_5 = list(resolve_mod.MULTIHIT_2_5)

        def before(move, budget, ability=None):  # noqa: ANN001, ANN202, ARG001
            counts = real(move, budget)
            return list(self.OLD_2_5) if counts == now_2_5 else counts

        resolve_mod.multihit_counts = before

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod.multihit_counts = self.real


class unchanged:  # noqa: N801 - read as a phrase at the call site
    """The control for `--using`: each kind of move named has its effect taken out."""

    def __init__(self, reg, moves: frozenset[str]) -> None:  # noqa: ANN001
        self.parts = []
        if any(reg.moves[m].raw.get("breaksProtect") for m in moves):
            self.parts.append(unbroken())
        if any(isinstance(reg.moves[m].raw.get("multihit"), list) for m in moves):
            self.parts.append(old_hit_counts())

    def __enter__(self) -> None:
        for part in self.parts:
            part.__enter__()

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        for part in reversed(self.parts):
            part.__exit__(*exc)


class unstopped:  # noqa: N801 - read as a phrase at the call site
    """Python with Psychic Terrain's per-target stop taken out: the control for --terrain.

    The terrain stays -- its Psychic boost and the Speed-order effects with it -- so what
    differs is only what the stop did (IKA-156).
    """

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._stopped_by_psychic_terrain
        resolve_mod._stopped_by_psychic_terrain = lambda *_args: False

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._stopped_by_psychic_terrain = self.real


class base_game_salt_cure:  # noqa: N801 - read as a phrase at the call site
    """Python with the base game's Salt Cure, 1/8 and 1/4: the control for --salt-cure.

    The volatile stays and its residual still runs, so what differs is only the champions
    mod's fraction (IKA-159) -- where 1/16 and 1/8 come out the same, nothing moves.
    """

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (resolve_mod.SALT_CURE_DAMAGE, resolve_mod.SALT_CURE_DAMAGE_WEAK)
        resolve_mod.SALT_CURE_DAMAGE, resolve_mod.SALT_CURE_DAMAGE_WEAK = (1, 8), (1, 4)

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod.SALT_CURE_DAMAGE, resolve_mod.SALT_CURE_DAMAGE_WEAK = self.real


def salt(pos: Position) -> int:
    """Puts Salt Cure on every Pokemon on the field, and says how many took it."""
    salted = 0
    for side in pos.sides:
        for mon in side.active_pokemon():
            if mon is None or mon.fainted or mon.has_volatile("saltcure"):
                continue
            mon.volatiles.append(Effect(id="saltcure"))
            salted += 1
    return salted


#: Abilities that raise a move's priority, and the moves they raise (`speed.move_priority`).
PRIORITY_ABILITIES = {"prankster", "galewings", "triage"}


def priority_moves(reg) -> frozenset[str]:  # noqa: ANN001
    """Moves with a priority of their own that reach another Pokemon."""
    return frozenset(
        move.id
        for move in reg.moves.values()
        if move.priority > 0
        and move.target not in ("self", "allySide", "allyTeam", "adjacentAlly", "all")
    )


def uses_priority(reg, pos: Position, side: int, action, own: frozenset[str]) -> bool:  # noqa: ANN001
    """Whether a slot uses a move that may go at positive priority: one of `own`, or any
    move of a Pokemon whose ability can raise it -- `move_priority` decides which."""
    for slot in action.slots:
        move_id = getattr(slot, "move_id", None)
        if move_id is None:
            continue
        if move_id in own:
            return True
        index = pos.sides[side].active[slot.slot]
        mon = pos.sides[side].pokemon[index] if index is not None else None
        if mon is not None and mon.ability in PRIORITY_ABILITIES:
            return True
    return False


def priority_on_field(pos: Position, own: frozenset[str]) -> bool:
    return knows_on_field(pos, own) or any(
        mon is not None and not mon.fainted and mon.ability in PRIORITY_ABILITIES
        for side in pos.sides
        for mon in side.active_pokemon()
    )


def recorded_positions(  # noqa: ANN001
    reg, args, holding: frozenset[str], using: frozenset[str] = frozenset()
) -> tuple[list[Position], int]:
    """Roots a search already filled a matrix at, read from recorded games.

    No game is played, so nothing here is generation. With `holding` every game is read,
    and a game is parsed only if its text names one of the items: a pool can carry an item
    once in a thousand games, and a sample of the first few hundred would find none of it.
    Returns the positions and how many were passed over for being another regulation's.
    """
    found: list[Position] = []
    keys: set[str] = set()
    other_format = 0
    wanted = holding | using
    enough = None if wanted else 40 * args.nodes
    for directory in args.games_dir:
        for path in sorted(Path(directory).glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if wanted and not any(name in line for name in wanted):
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
                        if using and not knows_on_field(pos, using):
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


BUDGETS = {"matrix": Budget.matrix, "fast": Budget.fast, "exact": Budget.exact}


def scenario_position(path: Path):  # noqa: ANN201
    """A scenario file's node, with the belief layer's modal spreads filled in.

    The node `tests/test_rust_node.py` holds the port to, built the same way: fabricated
    spreads leave HP and maximum HP inconsistent, and a position that could not occur is
    not a thing to hold two implementations to.
    """
    from pokeuraou.cli import _modal, build_beliefs
    from pokeuraou.setup import load_scenario, with_spreads

    scenario = load_scenario(path)
    beliefs = build_beliefs(scenario)
    pos = with_spreads(scenario, {key: _modal(b) for key, b in beliefs.items()})
    return scenario.reg, pos


def menu(reg, pos: Position, side: int, limit: int) -> list:  # noqa: ANN001
    """The actions a side is compared on: narrowed to `limit`, or every legal one at 0."""
    if limit == 0:
        return list(side_actions(reg, pos, side))
    return narrow(reg, pos, side, limit=limit).actions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument(
        "--limit",
        type=int,
        default=24,
        help="actions per side (the search's narrowing). 0 keeps every legal action, "
        "which only a --scenario node can afford.",
    )
    ap.add_argument(
        "--budget",
        choices=sorted(BUDGETS),
        default="matrix",
        help="the budget each cell is resolved under. matrix pins the roll and never "
        "narrows; fast and exact narrow, which is where a resumed turn's budget is read.",
    )
    ap.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="compare this scenario file's node (modal spreads) instead of positions "
        "from games",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="exit 1 if a cell differs by more than this. A learned leaf's float32 "
        "batches differ by ~5e-8; a named objective by a few ulps.",
    )
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
    ap.add_argument(
        "--using",
        default=None,
        help="comma-separated breaksProtect or [2, 5] multi-hit move ids: keep positions "
        "where a Pokemon on the field knows one, hold every cell that uses one to the port "
        "branch by branch, and count where the break or the hit count fired",
    )
    ap.add_argument(
        "--terrain",
        choices=["psychicterrain"],
        default=None,
        help="lay this terrain on every position first -- no recorded game has one -- keep "
        "those with a priority move or a priority-raising ability on the field, hold every "
        "cell that may use one to the port branch by branch, and count where the "
        "terrain's per-target stop fired",
    )
    ap.add_argument(
        "--salt-cure",
        action="store_true",
        help="put Salt Cure on every Pokemon on the field first -- no recorded team has "
        "it -- hold every cell to the port branch by branch, and count where the champions "
        "mod's fraction fired: the cells the base game's 1/8 and 1/4 move",
    )
    args = ap.parse_args()

    if args.limit == 0 and not args.scenario:
        ap.error("--limit 0 (the whole menu) is for a --scenario node")
    roster = load_roster(args.roster)
    reg = roster.reg
    scenario_pos = None
    if args.scenario:
        reg, scenario_pos = scenario_position(args.scenario)
    register_mega_stones(reg)
    holding = frozenset(i for i in (args.holding or "").split(",") if i)
    if args.give:
        holding |= {args.give}
    using = frozenset(m for m in (args.using or "").split(",") if m)
    for move_id in sorted(using):
        move = reg.moves.get(move_id)
        if move is None or not (
            move.raw.get("breaksProtect") or isinstance(move.raw.get("multihit"), list)
        ):
            ap.error(
                f"--using takes breaksProtect or ranged multi-hit moves; {move_id} is neither"
            )

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

    if scenario_pos is not None:
        positions = [scenario_pos]
        print(f"the node of {args.scenario}")
    elif args.games_dir:
        positions, other_format = recorded_positions(
            reg, args, holding - {args.give}, using
        )
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
    if using:
        positions = [pos for pos in positions if knows_on_field(pos, using)]
    quick = priority_moves(reg) if args.terrain else frozenset()
    if args.terrain:
        for pos in positions:
            pos.field.terrain = args.terrain
            pos.field.terrain_duration = 5
        positions = [pos for pos in positions if priority_on_field(pos, quick)]
        print(f"{args.terrain} laid on {len(positions)} positions with a priority move on the field")
    if args.salt_cure:
        salted = sum(salt(pos) for pos in positions)
        print(f"Salt Cure put on {salted} Pokemon on the field")
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

    budget = BUDGETS[args.budget]()
    print(f"budget {args.budget}: {budget}")
    checked = cells = identical = differing = 0
    worst = worst_value = worst_strategy = 0.0
    rust_seconds = python_seconds = 0.0
    notes_differ = masks_differ = mask_cells = mask_rust_only = 0
    # Where the held item fired, and what the port did there.
    fired = fired_wrong = fired_wrong_anyway = 0
    fired_worst = 0.0
    fired_notes: Counter = Counter()
    shown = 0
    # Where the move was used, where its break fired, and what the port did there.
    used = used_wrong = broke = broke_wrong = broke_paused = 0
    broke_worst = 0.0
    # Where a priority move may go under the terrain, and where the terrain stopped it.
    quick_used = quick_wrong = stopped = stopped_wrong = 0
    stopped_worst = 0.0
    # Under Salt Cure, every cell; and where the mod's fraction moved the answer.
    cured = cured_wrong = cured_refused = cured_fired = cured_fired_wrong = 0
    cured_worst = 0.0

    for pos in positions:
        row = menu(reg, pos, 0, args.limit)
        col = menu(reg, pos, 1, args.limit)
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

        if using:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    if not (uses(a, using) or uses(b, using)):
                        continue
                    used += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unchanged(reg, using):
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    fired_here = differ(outcome(here), outcome(control))
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    used_wrong += bool(wrong)
                    if fired_here:
                        broke += 1
                        broke_wrong += bool(wrong)
                        # `branch_differences` compares a paused turn's weight but not its
                        # position -- the port hands back only a finished branch's -- so a
                        # break that shows only in a paused state is not held here.
                        broke_paused += bool(here.suspended)
                        for index in range(len(evaluators)):
                            broke_worst = max(
                                broke_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} using {sorted(using)}: {wrong[0][:200]}")

        if args.terrain:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    if not (
                        uses_priority(reg, pos, 0, a, quick) or uses_priority(reg, pos, 1, b, quick)
                    ):
                        continue
                    quick_used += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unstopped():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    quick_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        stopped += 1
                        stopped_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            stopped_worst = max(
                                stopped_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} under {args.terrain}: {wrong[0][:200]}")

        if args.salt_cure:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    cured += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with base_game_salt_cure():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    # A cell the port refuses (a gate that is not Salt Cure's -- Flower
                    # Trick's willCrit, say) is filled in Python and is counted apart.
                    refused_here = wrong == ["the port refused the turn"]
                    cured_refused += refused_here
                    if refused_here:
                        wrong = []
                    cured_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        cured_fired += 1
                        cured_fired_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            cured_worst = max(
                                cured_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} under Salt Cure: {wrong[0][:200]}")

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
        mask_gap = np.asarray(rust_exact) != np.asarray(python_exact)
        if mask_gap.any():
            masks_differ += 1
            mask_cells += int(mask_gap.sum())
            mask_rust_only += int((mask_gap & np.asarray(rust_exact)).sum())
            print(f"  the exact mask differs on {int(mask_gap.sum())} cells")

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
    print(
        f"  cells whose exact flag differs {mask_cells} "
        f"(exact in the port only: {mask_rust_only})"
    )
    if holding:
        print(f"\n  where {', '.join(sorted(holding))} fired -- the cells the control moves")
        print(f"    {fired} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {fired_wrong}")
        print(f"      of which differ the same way with the item taken off  {fired_wrong_anyway}")
        print(f"    worst cell difference there  {fired_worst:.3e}")
        for note, times in fired_notes.most_common(4):
            print(f"    note only the item's turn carries, {times} cells: {note}")
    if using:
        print(f"\n  where {', '.join(sorted(using))} was used -- every one held branch by branch")
        print(f"    {used} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {used_wrong}")
        print("  where the effect fired -- the cells the control (`unchanged`) moves")
        print(f"    {broke} of {used} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {broke_wrong}")
        print(f"    cells with a paused branch, whose position is not compared  {broke_paused}")
        print(f"    worst cell difference there  {broke_worst:.3e}")
    if args.terrain:
        print(f"\n  under {args.terrain}, cells that may use a priority move -- held branch by branch")
        print(f"    {quick_used} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {quick_wrong}")
        print("  where the terrain stopped one -- the cells `unstopped` moves")
        print(f"    {stopped} of {quick_used} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {stopped_wrong}")
        print(f"    worst cell difference there  {stopped_worst:.3e}")
    if args.salt_cure:
        print("\n  under Salt Cure, every cell -- held branch by branch")
        print(f"    {cured} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {cured_wrong}")
        print(f"    cells the port refused, filled in Python and not held  {cured_refused}")
        print("  where the mod's fraction fired -- the cells the base game's 1/8 and 1/4 move")
        print(f"    {cured_fired} of {cured} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {cured_fired_wrong}")
        print(f"    worst cell difference there  {cured_worst:.3e}")
    print(f"  python {python_seconds:.2f} s   rust {rust_seconds:.2f} s")
    if rust_seconds > 0:
        print(f"  end to end {python_seconds / rust_seconds:.1f}x")
    failed = []
    if worst > args.tolerance:
        failed.append(f"worst cell difference {worst:.3e} > {args.tolerance:.0e}")
    # Under every budget. Until IKA-151 the port had no "damage rolls stratified"
    # reduction and a stratifying budget was let off; it now marks the same turns inexact.
    if masks_differ:
        failed.append(f"the exact mask differs on {mask_cells} cells of {masks_differ} nodes")
    if used_wrong:
        failed.append(f"{used_wrong} cells using {', '.join(sorted(using))} differ by branch")
    if using and not broke:
        failed.append("the effect fired in no cell, so agreeing here says nothing")
    if quick_wrong:
        failed.append(f"{quick_wrong} cells under {args.terrain} differ by branch")
    if args.terrain and not stopped:
        failed.append(f"{args.terrain} stopped nothing in any cell, so agreeing here says nothing")
    if cured_wrong:
        failed.append(f"{cured_wrong} cells under Salt Cure differ by branch")
    if args.salt_cure and not cured_fired:
        failed.append("the mod's Salt Cure fraction moved no cell, so agreeing here says nothing")
    if failed:
        print(f"\nFAIL ({args.budget}): " + "; ".join(failed))
        sys.exit(1)
    print(f"\nOK ({args.budget})")


if __name__ == "__main__":
    main()
