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
`--using doubleshock` (or `burnup`) takes the other kind of move it knows, one that spends
its user's type (IKA-162); its control is the turn with `TYPE_SPENDING_MOVES` emptied.

`--using` also takes a drain move (IKA-161) and a status move made only of the effects
`JUDGED_STATUS_EFFECTS` scores (IKA-171):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --using drainpunch,matchagotcha
    uv run python tools/diff_node.py --games-dir data/ika73/w12 --using recover,toxic,sunnyday

The control for a drain move is the move with its `drain` taken off; for a judged status
move, `_apply_status_move` without the judgement, so the cells it moves are the ones where
"it did nothing" set `move_failed`.

`--using` also takes a [2, 5] multi-hit move (IKA-160):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --using bulletseed,rockblast

The control is the hit count as it was before IKA-160 -- 1/3, 1/3, 1/6, 1/6 and Skill Link
not read -- so the cells it moves are the ones where the count's distribution mattered.

`--using perishsong` holds the port to Perish Song (IKA-172), which it listed as fully
modelled without ever giving anyone the counter. The control is the turn with
`_perish_song` taken out, so the cells it moves are the ones where somebody was given one.

`--using helpinghand` holds the port to Helping Hand's failure on a partner that has
already moved (IKA-184). The control is the turn with `_helping_hand_fails` answering no,
so the cells it moves are the ones where a help failed.

`--using taunt` holds the port to Taunt's `onBeforeMove` and its `duration++` (IKA-188): a
status move chosen the turn a faster Taunt lands is stopped, and a Taunt on a Pokemon that
has already moved is 4 long. The control is the turn with `_taunt_stops` answering no and
`_taunt_lasts_longer` doing nothing, so the cells it moves are the ones where either fired.

`--using stoneaxe` (or `ceaselessedge`) takes a move that lays a hazard from its own
`onAfterHit` (IKA-173); its control is the turn with `AFTER_HIT_HAZARDS` emptied, the
hit laying nothing as before.

Holding the port to Toxic Debris, whose Toxic Spikes a partner's hit and a knock-out now
lay (IKA-173):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --toxic-debris

`--toxic-debris` keeps the recorded positions with Toxic Debris on the field and holds
every cell branch by branch. The cells where the new rule *fired* are the ones Python
moves when `_toxic_debris` is put back to a foe's hit on a Glimmora that survives, and the
run fails if there are none.

`--using` takes a rampage move too (IKA-174), and `--rampage` holds the rest of it:

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --using outrage,petaldance
    uv run python tools/diff_node.py --games-dir data/ika73/w12 --rampage

The control is `lockedmove` as it was before IKA-174, bare and never ended (`unraged`). A
recorded game carries only that bare marker, so `--using` reaches the rampage's first turn;
`--rampage` puts a second turn's lock -- the move, one turn left, no length -- on each
knower first, which is where the length branches, the lock ends and the confusion lands.

Holding the port to a `randomNormal` move's foe (IKA-178):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --random-target
    uv run python tools/diff_node.py --games-dir data/ika73/w12 --random-target --rampage

`--random-target` keeps the recorded positions with a Pokemon on the field that knows a
`randomNormal` move (Outrage and its kin, Uproar, Struggle) and holds every cell that uses
one branch by branch. The cells where the draw *fired* are the ones Python moves with the
foe undrawn (`undrawn`): the first foe standing and no redirection, as before. With
`--rampage` the knowers are mid-rampage first, so the locked turn draws too.

Holding the port to a frozen Pokemon (IKA-171):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --frozen

`--frozen` keeps the recorded positions with a frozen Pokemon on the field and holds every
cell branch by branch. The cells where a thaw *fired* -- a `defrost` move used frozen, or a
Fire or `thawsTarget` move into the frozen one -- are the ones Python moves with
`_defrosts` and `_thaw_on_hit` taken out, and the run fails if there are none.

Holding the port to a Choice lock (IKA-179):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --choice-locked

`--choice-locked` keeps the recorded positions with a Choice-locked Pokemon on the field
that has to Struggle, whose lock Showdown would have dropped, or beside Knock Off, Trick or
Switcheroo, and holds every cell branch by branch. The cells where the lock *fired* -- one on
`struggle` or on a Choice item knocked off dropped, a Trick dropping both -- are the ones
Python moves with the old rule put back (`unlocked`), and the run fails if there are none.

Holding the port to a terrain no recorded game has (IKA-156):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --terrain psychicterrain

`--terrain` lays Psychic Terrain on each position, keeps those with a priority move (or a
priority-raising ability) on the field, and holds every cell that may use one branch by
branch. The cells where the terrain's per-target stop *fired* are the ones Python moves
with `_stopped_by_psychic_terrain` taken out, and the run fails if there are none.

Holding the port to the priority-blocking abilities (IKA-158):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --priority-block

`--priority-block` keeps the recorded positions with Armor Tail, Queenly Majesty or
Dazzling on the field and holds every cell that may use a priority move branch by branch.
The cells where the stop *fired* are the ones Python moves with `_priority_blocked_by`
taken out, and the run fails if there are none.

Holding the port to Salt Cure, which no recorded team uses (IKA-159):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --salt-cure

`--salt-cure` puts the volatile on every Pokemon on the field and holds every cell branch
by branch. The cells where the champions mod's 1/16 and 1/8 *fired* are the ones Python
moves when the base game's 1/8 and 1/4 are put back, and the run fails if there are none.

Holding the port to the hazards, which no recorded team carries (IKA-165):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --hazards

`--hazards` teaches every Pokemon on the field a foeSide hazard in its last move slot --
Stealth Rock, Spikes, Toxic Spikes and Sticky Web in turn -- and holds every cell that uses
one branch by branch. The cells where the placement *fired* are the ones Python moves when
the condition is put back on the user's side, and the run fails if there are none.

Holding the port to a charge's stored target (IKA-176):

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --charging

`--charging` keeps the recorded positions with a Pokemon on the field that knows a charging
move (the dump's `charge` flag), and puts every other one mid-charge -- `twoturnmove` with
its move and a stored target, the second foe where it stands, so the stored target is not
the one an untargeted choice falls back on. On the others every cell that uses a charging
move is held branch by branch, on the mid-charge ones every cell. The cells where the
target *fired* are the ones Python moves with the target neither stored nor read
(`untargeted`), and the run fails if there are none.

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --hammer

`--hammer` teaches Gigaton Hammer, which no recorded team carries, to every Pokemon on the
field and holds every cell that uses it. On every other position the taught Pokemon's last
move is the hammer, so the menu -- Python's, the port builds none -- drops it; those
slots are counted against the menu with the `cantusetwice` rule taken out.

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --confusion-guard

`--confusion-guard` (IKA-189) teaches Confuse Ray to every Pokemon on the field, confuses
the first on each side with Showdown's own `time` 3, and puts Own Tempo, Misty Terrain or
Safeguard on the rest by position in turn -- no recorded team carries any of the three.
Every cell is held branch by branch; the cells `unguarded` (the refusals only for the
fatigue, the self-hit at the highest roll) moves are counted, and each part alone.

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --substitute

`--substitute` (IKA-180) teaches Substitute to every Pokemon on the field and puts a doll in
front of the first on each side -- a whole one (`floor(maxhp / 4)`) or a worn one (an
eighth) by position in turn. Every cell is held branch by branch; the cells `unsubbed` (the
doll a mark and Substitute free, as before IKA-180) moves are counted, and each part alone:
the use (the HP, the refusals) and the doll (the hits, the status moves, Intimidate).

    uv run python tools/diff_node.py --games-dir data/ika73/w12 --eject

`--eject` (IKA-191) hands out, by position in turn, Eject Button, Emergency Exit, Wimp Out
or Red Card to every Pokemon on the field -- the recorded M-B games have none of them.
Every cell is held branch by branch; the cells `uneject` (none of the four doing anything)
moves are counted by what was handed out. A Red Card that fires is refused by the port, as
a forceSwitch move is, so those cells count as refused and are filled in Python.

    uv run python tools/diff_node.py --roster <an M-C roster> \
        --games-dir C:/tmp/ika77/out/g600 --trace-sync

`--trace-sync` (IKA-203) keeps the recorded positions with a Trace holder on the bench (it
traces as it comes in, by a switch or a mid-turn replacement) or a Synchronize holder on
the field, and holds every cell branch by branch. The cells `untraced` (Trace copying
nothing) and `unsynced` (Synchronize passing nothing back) move are counted apart, and the
run fails if neither moved any.

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
from pokeuraou.position import Effect, MoveSlot, Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.resolve import (  # noqa: E402
    AFTER_HIT_HAZARDS,
    JUDGED_STATUS_EFFECTS,
    PRIORITY_BLOCKING_ABILITIES,
    TYPE_SPENDING_MOVES,
    UNJUDGED_STATUS_EFFECTS,
    Budget,
    batched_payoffs,
    resolve_turn,
)
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


class unspent:  # noqa: N801 - read as a phrase at the call site
    """Python with `TYPE_SPENDING_MOVES` emptied: the control for Double Shock and Burn Up.

    The move is then an ordinary attack -- it neither fails for want of the type nor spends
    it -- so what differs is only what the rule did (IKA-162). A switch out and a faint still
    put the species' types back; with nothing spent, that writes what was already there.
    """

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod.TYPE_SPENDING_MOVES
        resolve_mod.TYPE_SPENDING_MOVES = {}

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod.TYPE_SPENDING_MOVES = self.real


def rampage_moves(reg) -> frozenset[str]:  # noqa: ANN001
    """Outrage and the rest: the moves whose `self` puts `lockedmove` on their user."""
    return frozenset(
        m.id for m in reg.moves.values()
        if (m.raw.get("self") or {}).get("volatileStatus") == "lockedmove"
    )


class unraged:  # noqa: N801 - read as a phrase at the call site
    """Python with the rampage as it was before IKA-174: the control for a rampage move.

    `lockedmove` goes on bare -- no move, no duration, no length -- and nothing ends it,
    rolls it, confuses or spares the PP; a lock already on the position just counts down
    and falls off. What differs is only what the rampage did.
    """

    NAMES = (
        "_start_rampage", "_roll_rampage", "_rampage_after_move", "_rampage_runs_out",
        "_rampage_residual", "_rampage_move",
    )

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = {name: getattr(resolve_mod, name) for name in self.NAMES}

        def bare(turn, side, slot):  # noqa: ANN001, ANN202
            mon = turn.mon_at(side, slot)
            if mon is not None and not mon.fainted and not mon.has_volatile("lockedmove"):
                mon.volatiles.append(Effect(id="lockedmove"))

        resolve_mod._start_rampage = bare
        resolve_mod._roll_rampage = lambda *_args: None
        resolve_mod._rampage_after_move = lambda *_args: None
        resolve_mod._rampage_runs_out = lambda *_args: None
        resolve_mod._rampage_residual = lambda *_args: None
        resolve_mod._rampage_move = lambda *_args: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        for name, real in self.real.items():
            setattr(resolve_mod, name, real)


def enrage(pos: Position, moves: frozenset[str]) -> int:
    """Puts a rampage on its second turn on every Pokemon on the field that knows one --
    its move, one turn left, no length yet: what a generated game carries -- and says how
    many took it."""
    raged = 0
    for side in pos.sides:
        for mon in side.active_pokemon():
            if mon is None or mon.fainted or mon.has_volatile("lockedmove"):
                continue
            known = next((slot.id for slot in mon.moves if slot.id in moves), None)
            if known is None:
                continue
            # A Choice item locked into another move could not have started the rampage.
            choice = mon.volatile("choicelock")
            if choice is not None and choice.move not in (None, known):
                continue
            mon.volatiles.append(Effect(id="lockedmove", duration=1, move=known))
            raged += 1
    return raged


class unsung:  # noqa: N801 - read as a phrase at the call site
    """Python with `_perish_song` taken out: the control for Perish Song (IKA-172).

    The move is still used -- its PP spent, its `lastMove` written -- and gives nobody a
    counter, which is what the port did before it learned the move.
    """

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._perish_song
        resolve_mod._perish_song = lambda *_args: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._perish_song = self.real


class unhelped:  # noqa: N801 - read as a phrase at the call site
    """Python with `_helping_hand_fails` answering no: the control for Helping Hand
    (IKA-184). A help on a partner that has already moved lands, as before."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._helping_hand_fails
        resolve_mod._helping_hand_fails = lambda *_args: False

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._helping_hand_fails = self.real


class untaunted:  # noqa: N801 - read as a phrase at the call site
    """Python's Taunt as it was before IKA-188: the control for `--using taunt`. It still
    lands and shapes the next menu, but stops nothing this turn and is 3 long for everyone."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (resolve_mod._taunt_stops, resolve_mod._taunt_lasts_longer)
        resolve_mod._taunt_stops = lambda *_args: False
        resolve_mod._taunt_lasts_longer = lambda *_args: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._taunt_stops, resolve_mod._taunt_lasts_longer = self.real


class unlaid:  # noqa: N801 - read as a phrase at the call site
    """Python with Stone Axe's and Ceaseless Edge's `onAfterHit` taken out, as before
    IKA-173: the control for `--using stoneaxe`. The hit still lands."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod.AFTER_HIT_HAZARDS
        resolve_mod.AFTER_HIT_HAZARDS = {}

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod.AFTER_HIT_HAZARDS = self.real


def _old_toxic_debris(turn, move, target, attacker_side) -> None:  # noqa: ANN001
    """Toxic Debris as it was before IKA-173: a foe's physical hit on a Glimmora that
    survived, and nothing else."""
    defender = turn.mon_at(*target)
    if defender is None or defender.fainted:
        return
    if move.category == "Physical" and target[0] != attacker_side:
        existing = turn.pos.sides[attacker_side].side_condition("toxicspikes")
        if existing is None or (existing.layers or 1) < 2:
            turn.add_side_condition(attacker_side, "toxicspikes")


class old_toxic_debris:  # noqa: N801 - read as a phrase at the call site
    """Python with Toxic Debris put back as it was: the control for --toxic-debris."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._toxic_debris
        resolve_mod._toxic_debris = _old_toxic_debris

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._toxic_debris = self.real


class undrained:  # noqa: N801 - read as a phrase at the call site
    """Python with these moves' `drain` taken off: the control for a drain move (IKA-161).

    The move still hits for the same damage, so what differs is only what the heal did.
    """

    def __init__(self, reg, moves: frozenset[str]) -> None:  # noqa: ANN001
        self.moves = [reg.moves[m] for m in sorted(moves) if reg.moves[m].raw.get("drain")]

    def __enter__(self) -> None:
        self.kept = [move.raw.pop("drain") for move in self.moves]

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        for move, drain in zip(self.moves, self.kept, strict=True):
            move.raw["drain"] = drain


class unjudged:  # noqa: N801 - read as a phrase at the call site
    """Python's status moves without the "did nothing" judgement: the control for a judged
    status move. The move's effects are applied the same, so what differs is only whether
    `move_failed` is set (IKA-171)."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._apply_status_move_and_judge
        resolve_mod._apply_status_move_and_judge = resolve_mod._apply_status_move

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._apply_status_move_and_judge = self.real


def judged(move) -> bool:  # noqa: ANN001
    """A status move `_apply_status_move_and_judge` judges."""
    raw = move.raw
    return (
        move.category == "Status"
        and any(raw.get(key) for key in JUDGED_STATUS_EFFECTS)
        and not move.has_custom_code
        and not any(raw.get(key) for key in UNJUDGED_STATUS_EFFECTS)
    )


class unthawed:  # noqa: N801 - read as a phrase at the call site
    """Python without the thaws: the control for --frozen. A `defrost` move is rolled for
    like any other, and a Fire or `thawsTarget` hit leaves the target frozen (IKA-171)."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (resolve_mod._defrosts, resolve_mod._thaw_on_hit)
        resolve_mod._defrosts = lambda *_args: False
        resolve_mod._thaw_on_hit = lambda *_args: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._defrosts, resolve_mod._thaw_on_hit = self.real


class unlocked:  # noqa: N801 - read as a phrase at the call site
    """Python's Choice lock as it was before IKA-179: the control for --choice-locked. The
    first move is kept, but a lock naming no move slot or no Choice item is never dropped
    -- not at the end of the turn, not by Trick, not by the next move."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (
            resolve_mod._live_choice_lock,
            resolve_mod._choice_lock_ends,
            resolve_mod._swap_ends_choice_locks,
        )
        resolve_mod._live_choice_lock = lambda mon: mon.volatile("choicelock")
        resolve_mod._choice_lock_ends = lambda *_args: None
        resolve_mod._swap_ends_choice_locks = lambda *_args: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        (
            resolve_mod._live_choice_lock,
            resolve_mod._choice_lock_ends,
            resolve_mod._swap_ends_choice_locks,
        ) = self.real


#: Moves that take a Choice item away from a locked holder or swap it.
ITEM_TAKING_MOVES = frozenset({"knockoff", "trick", "switcheroo"})


def choice_locked_on_field(reg, pos: Position) -> bool:  # noqa: ANN001
    """A Choice-locked Pokemon on the field whose lock IKA-179's rules can move: it has to
    Struggle, its lock is one Showdown would have dropped, or a Pokemon on the field knows a
    move that takes the item away. An ordinary lock that holds is the same in every rule."""
    from pokeuraou.actions import is_struggling
    from pokeuraou.resolve import _choice_lock_is_stale

    field = [mon for side in pos.sides for mon in side.active_pokemon() if mon and not mon.fainted]
    takers = any(m.id in ITEM_TAKING_MOVES for mon in field for m in mon.moves)
    return any(
        mon.has_volatile("choicelock")
        and (takers or is_struggling(mon, reg) or _choice_lock_is_stale(reg, mon))
        for mon in field
    )


def frozen_on_field(pos: Position) -> bool:
    return any(
        mon is not None and not mon.fainted and mon.status == "frz"
        for side in pos.sides
        for mon in side.active_pokemon()
    )


class unconfused:  # noqa: N801 - read as a phrase at the call site
    """Python with confusion as it was before IKA-177: the control for --confused.

    Nothing ends it or counts its tries, the self-hit is 1/3, and only the rampage's
    fatigue meets a Persim or Lum Berry. Paralysis still comes after the self-hit, so a
    cell whose only change is that order is not counted as fired.
    """

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (
            resolve_mod._roll_confusion, resolve_mod._confusion_try,
            resolve_mod._start_confusion, resolve_mod._confused_by_fatigue,
            resolve_mod.CONFUSION_SELF_HIT_CHANCE,
        )
        real_fatigue = resolve_mod._confused_by_fatigue

        def start(turn, side, slot) -> None:  # noqa: ANN001
            mon = turn.mon_at(side, slot)
            if mon is not None and not mon.fainted and not mon.has_volatile("confusion"):
                mon.volatiles.append(Effect(id="confusion"))

        def fatigue(turn, side, slot) -> None:  # noqa: ANN001
            real_fatigue(turn, side, slot)
            mon = turn.mon_at(side, slot)
            if (
                mon is not None and mon.has_volatile("confusion")
                and mon.item in ("persimberry", "lumberry") and not turn.berries_blocked(side)
            ):
                turn.consume_item(side, slot, reason=mon.item)
                mon.volatiles = [v for v in mon.volatiles if v.id != "confusion"]

        resolve_mod._roll_confusion = lambda *_args: None
        resolve_mod._confusion_try = lambda turn, side, slot: bool(
            turn.mon_at(side, slot) is not None and turn.mon_at(side, slot).has_volatile("confusion")
        )
        resolve_mod._start_confusion = start
        resolve_mod._confused_by_fatigue = fatigue
        resolve_mod.CONFUSION_SELF_HIT_CHANCE = 1.0 / 3.0

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        (
            resolve_mod._roll_confusion, resolve_mod._confusion_try,
            resolve_mod._start_confusion, resolve_mod._confused_by_fatigue,
            resolve_mod.CONFUSION_SELF_HIT_CHANCE,
        ) = self.real


#: The confusions --confused puts on, by position in turn: tries our resolver counted
#: (1 to 3, and Axe Kick's), Showdown's own `time` (1 cures at the next try), and the bare
#: one a record made before IKA-177 carries (read as fresh).
CONFUSIONS = (
    {"tries": 1}, {"tries": 2}, {"time": 1}, {"tries": 3}, {"tries": 2, "min": 3},
    {"time": 3}, {},
)


def confuse(pos: Position, index: int) -> int:
    """Confuses the first Pokemon on the field on each side -- the one a record already
    had included -- with one of `CONFUSIONS`, and says how many took it."""
    put = 0
    for side_index, side in enumerate(pos.sides):
        mon = next((m for m in side.active_pokemon() if m is not None and not m.fainted), None)
        if mon is None:
            continue
        extra = dict(CONFUSIONS[(index * 2 + side_index) % len(CONFUSIONS)])
        mon.volatiles = [v for v in mon.volatiles if v.id != "confusion"]
        mon.volatiles.append(Effect(id="confusion", extra=extra))
        put += 1
    return put


def confused_on_field(pos: Position) -> bool:
    return any(
        mon is not None and not mon.fainted and mon.has_volatile("confusion")
        for side in pos.sides
        for mon in side.active_pokemon()
    )


#: What --confusion-guard puts on, by position in turn (IKA-189).
CONFUSION_GUARDS = ("owntempo", "mistyterrain", "safeguard")


def guard_confusion(reg, pos: Position, index: int) -> str:  # noqa: ANN001
    """For --confusion-guard: Confuse Ray in the last free move slot of every Pokemon on
    the field (as `teach_hazards` picks it), the first Pokemon on each side confused with
    Showdown's own `time` 3 -- so the self-hit's roll is the only chance in it -- and one of
    `CONFUSION_GUARDS`, by position in turn: Own Tempo on every other Pokemon on the field,
    Misty Terrain, or Safeguard on both sides. Returns the guard."""
    guard = CONFUSION_GUARDS[index % len(CONFUSION_GUARDS)]
    pp = reg.moves["confuseray"].pp
    for side in pos.sides:
        first = True
        for mon in side.active_pokemon():
            if mon is None or mon.fainted:
                continue
            if mon.moves and not any(s.id == "confuseray" for s in mon.moves):
                held = {v.move for v in mon.volatiles if v.move} | {mon.last_move}
                free = [i for i, s in enumerate(mon.moves) if s.id not in held]
                if free:
                    mon.moves[free[-1]] = MoveSlot(id="confuseray", pp=pp, maxpp=pp)
            if first:
                mon.volatiles = [v for v in mon.volatiles if v.id != "confusion"]
                mon.volatiles.append(Effect(id="confusion", extra={"time": 3}))
                first = False
            elif guard == "owntempo":
                # A confused Own Tempo Pokemon is a state no game reaches (`onUpdate`).
                mon.ability = "owntempo"
                mon.volatiles = [v for v in mon.volatiles if v.id != "confusion"]
        if guard == "safeguard" and side.side_condition("safeguard") is None:
            side.side_conditions.append(Effect(id="safeguard", duration=5))
    if guard == "mistyterrain":
        pos.field.terrain = "mistyterrain"
        pos.field.terrain_duration = 5
    return guard


class unguarded:  # noqa: N801 - read as a phrase at the call site
    """Python with confusion refused and dealt as before IKA-189: the control for
    --confusion-guard. Only the rampage's fatigue (its own source) meets Own Tempo and
    Misty Terrain, and the self-hit is the highest roll whatever the budget's. `part`
    takes out only the refusals or only the roll, so each can be counted on its own."""

    def __init__(self, part: str = "both") -> None:
        assert part in ("both", "refusals", "roll")
        self.part = part

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (resolve_mod._confusion_refused, resolve_mod._confusion_damage)
        refused, damage = self.real

        def old_refused(turn, side, slot, source):  # noqa: ANN001, ANN202
            return refused(turn, side, slot, source) if source == (side, slot) else None

        if self.part != "roll":
            resolve_mod._confusion_refused = old_refused
        if self.part != "refusals":
            resolve_mod._confusion_damage = lambda turn, side, slot, roll=0: damage(
                turn, side, slot, 0
            )

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._confusion_refused, resolve_mod._confusion_damage = self.real


def substitute_on_field(reg, pos: Position, index: int) -> int:  # noqa: ANN001
    """For --substitute: Substitute in the last free move slot of every Pokemon on the field
    (as `guard_confusion` picks it), and a doll in front of the first on each side -- a
    whole one, or by position in turn a worn one at an eighth of the HP. Returns how many
    dolls were put up."""
    pp = reg.moves["substitute"].pp
    dolls = 0
    for side in pos.sides:
        first = True
        for mon in side.active_pokemon():
            if mon is None or mon.fainted:
                continue
            if mon.moves and not any(s.id == "substitute" for s in mon.moves):
                held = {v.move for v in mon.volatiles if v.move} | {mon.last_move}
                free = [i for i, s in enumerate(mon.moves) if s.id not in held]
                if free:
                    mon.moves[free[-1]] = MoveSlot(id="substitute", pp=pp, maxpp=pp)
            if first:
                mon.volatiles = [v for v in mon.volatiles if v.id != "substitute"]
                hp = mon.maxhp // 4 if index % 2 == 0 else max(1, mon.maxhp // 8)
                mon.volatiles.append(Effect(id="substitute", extra={"hp": hp}))
                dolls += 1
                first = False
    return dolls


class unsubbed:  # noqa: N801 - read as a phrase at the call site
    """Python with Substitute as before IKA-180: the control for --substitute. The doll is a
    mark that nothing meets, and Substitute puts it up for free and says it is not
    modelled. `part` takes out only the use or only the doll, so each is counted alone."""

    def __init__(self, part: str = "both") -> None:
        assert part in ("both", "use", "doll")
        self.part = part

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (
            resolve_mod._use_substitute,
            resolve_mod._hits_substitute,
            resolve_mod._intimidate_meets_substitute,
        )

        def free(reg, turn, action) -> None:  # noqa: ANN001
            del reg
            turn.add_volatile(action.side, action.slot, "substitute")
            turn.unmodelled.add("status move: substitute")

        if self.part != "doll":
            resolve_mod._use_substitute = free
        if self.part != "use":
            resolve_mod._hits_substitute = lambda turn, action, move, target: False
            resolve_mod._intimidate_meets_substitute = lambda turn, target: False

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        (
            resolve_mod._use_substitute,
            resolve_mod._hits_substitute,
            resolve_mod._intimidate_meets_substitute,
        ) = self.real


#: What --eject hands out, by position in turn (IKA-191): (kind, id).
EJECT_KINDS = (
    ("item", "ejectbutton"),
    ("ability", "emergencyexit"),
    ("ability", "wimpout"),
    ("item", "redcard"),
)


def hand_out_eject(reg, pos: Position, index: int) -> str:  # noqa: ANN001
    """For --eject: one of `EJECT_KINDS`, by position in turn, on every Pokemon on the
    field -- an item not over a mega stone (as `give`), an ability on anyone. Returns it."""
    kind, eid = EJECT_KINDS[index % len(EJECT_KINDS)]
    for side in pos.sides:
        for mon in side.active_pokemon():
            if mon is None or mon.fainted:
                continue
            if kind == "ability":
                mon.ability = eid
            elif (mon.item or "") not in reg.mega_map:
                mon.item = mon.base_item = eid
    return eid


class uneject:  # noqa: N801 - read as a phrase at the call site
    """Python as before IKA-191, the control for --eject: nothing after the hits switches
    anyone out, and the residual phase is only the residual phase."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (
            resolve_mod._after_move_secondary_switches,
            resolve_mod._user_exits_if_crossed,
            resolve_mod._residuals_then_emergency_exit,
        )
        resolve_mod._after_move_secondary_switches = lambda turn, action, move: None
        resolve_mod._user_exits_if_crossed = lambda turn, action, move: None
        resolve_mod._residuals_then_emergency_exit = resolve_mod._residuals

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        (
            resolve_mod._after_move_secondary_switches,
            resolve_mod._user_exits_if_crossed,
            resolve_mod._residuals_then_emergency_exit,
        ) = self.real


def trace_sync_on_board(pos: Position) -> bool:
    """For --trace-sync: a Trace holder on the bench, or a Synchronize holder on the field
    (IKA-203). A Trace already on the field traced as it came in, or found nothing."""
    for side in pos.sides:
        for mon in side.pokemon:
            if mon.fainted:
                continue
            if mon.ability == "trace" and not mon.is_active:
                return True
            if mon.ability == "synchronize" and mon.is_active:
                return True
    return False


class untraced:  # noqa: N801 - read as a phrase at the call site
    """Python as before IKA-203 for Trace, the first control for --trace-sync."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._trace
        resolve_mod._trace = lambda turn, side, slot: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._trace = self.real


class unsynced:  # noqa: N801 - read as a phrase at the call site
    """Python as before IKA-203 for Synchronize, the second control for --trace-sync."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._synchronize
        resolve_mod._synchronize = lambda turn, holder, source, status: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._synchronize = self.real


class unchanged:  # noqa: N801 - read as a phrase at the call site
    """The control for `--using`: each kind of move named has its effect taken out."""

    def __init__(self, reg, moves: frozenset[str]) -> None:  # noqa: ANN001
        self.parts = []
        if any(reg.moves[m].raw.get("breaksProtect") for m in moves):
            self.parts.append(unbroken())
        if any(isinstance(reg.moves[m].raw.get("multihit"), list) for m in moves):
            self.parts.append(old_hit_counts())
        if any(m in TYPE_SPENDING_MOVES for m in moves):
            self.parts.append(unspent())
        if any(m in AFTER_HIT_HAZARDS for m in moves):
            self.parts.append(unlaid())
        if moves & rampage_moves(reg):
            self.parts.append(unraged())
        if "perishsong" in moves:
            self.parts.append(unsung())
        if "helpinghand" in moves:
            self.parts.append(unhelped())
        if "taunt" in moves:
            self.parts.append(untaunted())
        if any(reg.moves[m].raw.get("drain") for m in moves):
            self.parts.append(undrained(reg, moves))
        if any(judged(reg.moves[m]) for m in moves):
            self.parts.append(unjudged())

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


class unblocked:  # noqa: N801 - read as a phrase at the call site
    """Python with the priority-blocking abilities' stop taken out: the control for
    --priority-block. The move still starts and spends its PP (IKA-158)."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._priority_blocked_by
        resolve_mod._priority_blocked_by = lambda *_args: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._priority_blocked_by = self.real


def ability_on_field(pos: Position, abilities: frozenset[str]) -> bool:
    """Whether a Pokemon on the field, still standing, has one of these abilities."""
    return any(
        mon is not None and not mon.fainted and mon.ability in abilities
        for side in pos.sides
        for mon in side.active_pokemon()
    )


#: The foeSide moves with a `sideCondition` in both dumps, in the order `teach_hazards` deals.
HAZARD_MOVES = ("stealthrock", "spikes", "toxicspikes", "stickyweb")


class hazards_on_the_users_side:  # noqa: N801 - read as a phrase at the call site
    """Python with every side condition laid on the user's side, as before IKA-165: the
    control for --hazards. The move still runs and the condition still goes down, so what
    differs is only whose side it lands on."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = resolve_mod._side_condition_side
        resolve_mod._side_condition_side = lambda action, _move: action.side

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._side_condition_side = self.real


def teach_hazards(reg, pos: Position, dealt: list[int]) -> int:  # noqa: ANN001
    """Puts a hazard in the last move slot of every Pokemon on the field that knows none,
    dealing them in turn, and says how many were taught.

    Not over a move the position still points at -- a Choice lock's, an Encore's, a
    Disable's, the last one used: a lock on a move the Pokemon no longer knows is a state
    no game reaches, and the two engines part on it (the port rewrites the lock, Python
    keeps it) for reasons that are not the hazard's.
    """
    taught = 0
    for side in pos.sides:
        for mon in side.active_pokemon():
            if mon is None or mon.fainted or not mon.moves:
                continue
            if any(slot.id in HAZARD_MOVES for slot in mon.moves):
                continue
            held = {v.move for v in mon.volatiles if v.move} | {mon.last_move}
            free = [index for index, slot in enumerate(mon.moves) if slot.id not in held]
            if not free:
                continue
            move_id = HAZARD_MOVES[dealt[0] % len(HAZARD_MOVES)]
            dealt[0] += 1
            pp = reg.moves[move_id].pp
            mon.moves[free[-1]] = MoveSlot(id=move_id, pp=pp, maxpp=pp)
            taught += 1
    return taught


def random_target_moves(reg) -> frozenset[str]:  # noqa: ANN001
    """The moves Showdown aims at a foe drawn at random: the `randomNormal` target."""
    return frozenset(m.id for m in reg.moves.values() if m.target == "randomNormal")


class undrawn:  # noqa: N801 - read as a phrase at the call site
    """Python with a `randomNormal` move's foe as it was before IKA-178: the control for
    --random-target. The first foe standing, never drawn and never redirected."""

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod

        self.real = (resolve_mod._draw_random_target, resolve_mod._redirection_target)
        real_redirect = self.real[1]

        def first(turn, action, move, budget):  # noqa: ANN001, ANN202, ARG001
            from dataclasses import replace

            if move.target == "randomNormal":
                action = replace(action, target=None)
            return [(1.0, turn, action)]

        def unredirected(turn, action, move, chosen):  # noqa: ANN001, ANN202
            return None if move.target == "randomNormal" else real_redirect(turn, action, move, chosen)

        resolve_mod._draw_random_target = first
        resolve_mod._redirection_target = unredirected

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod

        resolve_mod._draw_random_target, resolve_mod._redirection_target = self.real


def charge_moves(reg) -> frozenset[str]:  # noqa: ANN001
    """The moves that spend a turn charging: the dump's `charge` flag."""
    return frozenset(m.id for m in reg.moves.values() if "charge" in m.flags)


class untargeted:  # noqa: N801 - read as a phrase at the call site
    """Python with a charge's target neither stored nor read: the control for --charging.

    The charge still happens and the second turn still fires the move, at whatever the
    choice names -- none, now, so the first live foe -- which is IKA-169's resolver.
    """

    def __enter__(self) -> None:
        import pokeuraou.resolve as resolve_mod
        import pokeuraou.speed as speed_mod

        self.real = (resolve_mod._store_charge_target, speed_mod.charge_target)
        resolve_mod._store_charge_target = lambda *_args: None
        speed_mod.charge_target = lambda *_args: None

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.resolve as resolve_mod
        import pokeuraou.speed as speed_mod

        resolve_mod._store_charge_target, speed_mod.charge_target = self.real


def charge_mid_turn(reg, pos: Position, moves: frozenset[str]) -> int:  # noqa: ANN001
    """Puts every Pokemon on the field that knows a charging move mid-charge, and says how
    many: `twoturnmove` with the move, one turn left and a stored target -- the second foe
    if it stands, else the first. A recorded marker without them gets them."""
    from pokeuraou.actions import locked_move

    put = 0
    for side_index, side in enumerate(pos.sides):
        foe = pos.sides[1 - side_index]
        standing = [
            i + 1 for i, p in enumerate(foe.active) if p is not None and not foe.pokemon[p].fainted
        ]
        if not standing:
            continue
        for mon in side.active_pokemon():
            if mon is None or mon.fainted:
                continue
            if mon.has_volatile("mustrecharge") or mon.has_volatile("lockedmove"):
                continue
            move_id = locked_move(reg, mon)
            if move_id not in moves:
                move_id = next((s.id for s in mon.moves if s.id in moves), None)
            if move_id is None:
                continue
            mon.volatiles = [v for v in mon.volatiles if v.id != "twoturnmove"]
            mon.volatiles.append(
                Effect(id="twoturnmove", duration=1, move=move_id, extra={"targetLoc": standing[-1]})
            )
            mon.last_move = move_id
            put += 1
    return put


def teach_hammer(reg, pos: Position, remember: bool) -> list[tuple[int, int]]:  # noqa: ANN001
    """Puts Gigaton Hammer in the last free move slot of every Pokemon on the field (as
    `teach_hazards` picks it), and with `remember` makes it their last move. Returns the
    (side, slot) of each Pokemon taught."""
    taught = []
    for side_index, side in enumerate(pos.sides):
        for slot, party in enumerate(side.active):
            mon = side.pokemon[party] if party is not None else None
            if mon is None or mon.fainted or not mon.moves:
                continue
            if any(s.id == "gigatonhammer" for s in mon.moves):
                continue
            held = {v.move for v in mon.volatiles if v.move} | {mon.last_move}
            free = [index for index, s in enumerate(mon.moves) if s.id not in held]
            if not free:
                continue
            pp = reg.moves["gigatonhammer"].pp
            mon.moves[free[-1]] = MoveSlot(id="gigatonhammer", pp=pp, maxpp=pp)
            if remember:
                mon.last_move = "gigatonhammer"
            taught.append((side_index, slot))
    return taught


class hammer_twice:  # noqa: N801 - read as a phrase at the call site
    """The menu with the `cantusetwice` rule taken out: the control for --hammer."""

    def __enter__(self) -> None:
        import pokeuraou.actions as actions_mod

        self.real = actions_mod._disabled_after_itself
        actions_mod._disabled_after_itself = lambda *_args: False

    def __exit__(self, *_exc) -> None:  # noqa: ANN002
        import pokeuraou.actions as actions_mod

        actions_mod._disabled_after_itself = self.real


def sides_agree(node, pos: Position, a, b, here, budget) -> bool:  # noqa: ANN001
    """Whether every branch the port gives lays the same side conditions as Python's."""
    for index, branch in enumerate(here.branches):
        chosen = node.resolve(pos, [a, b], budget, select=index)
        if chosen is None or chosen.position is None:
            return False
        for mine, theirs in zip(branch.position.sides, chosen.position.sides, strict=True):
            if [c.to_json() for c in mine.side_conditions] != [
                c.to_json() for c in theirs.side_conditions
            ]:
                return False
    return True


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
    reg,
    args,
    holding: frozenset[str],
    using: frozenset[str] = frozenset(),
    abilities: frozenset[str] = frozenset(),
    frozen: bool = False,
    locked: bool = False,
    texts: frozenset[str] = frozenset(),
    keep=None,  # noqa: ANN001 - a predicate on the position (--trace-sync)
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
    wanted = holding | using | abilities | ({'"frz"'} if frozen else set())
    wanted |= {'"choicelock"'} if locked else set()
    wanted |= texts
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
                        if abilities and not ability_on_field(pos, abilities):
                            continue
                        if frozen and not frozen_on_field(pos):
                            continue
                        if locked and not choice_locked_on_field(reg, pos):
                            continue
                        if keep is not None and not keep(pos):
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
        help="comma-separated breaksProtect, [2, 5] multi-hit, type-spending, "
        "hazard-laying (Stone Axe), drain or judged status move ids, or taunt: keep positions where "
        "a Pokemon on the field knows one, hold every cell that uses one to the port branch "
        "by branch, and count where the effect fired",
    )
    ap.add_argument(
        "--frozen",
        action="store_true",
        help="keep positions with a frozen Pokemon on the field, hold every cell to the port "
        "branch by branch, and count where a thaw fired: the cells `unthawed` moves",
    )
    ap.add_argument(
        "--choice-locked",
        action="store_true",
        help="keep positions with a Choice-locked Pokemon on the field that Struggles, "
        "carries a stale lock or faces an item-taking move, hold every cell to the port "
        "branch by branch, and count where IKA-179's lock fired: the cells `unlocked` moves",
    )
    ap.add_argument(
        "--confused",
        action="store_true",
        help="confuse the first Pokemon on the field on each side first (a length our "
        "resolver counted, Showdown's own, or a record's bare one), hold every cell to the "
        "port branch by branch, and count where IKA-177's rule fired: the cells "
        "`unconfused` moves",
    )
    ap.add_argument(
        "--confusion-guard",
        action="store_true",
        help="teach Confuse Ray to every Pokemon on the field, confuse the first on each "
        "side (Showdown's `time` 3) and put Own Tempo, Misty Terrain or Safeguard on the "
        "rest by position in turn, hold every cell to the port branch by branch, and count "
        "where IKA-189's refusals or self-hit roll fired: the cells `unguarded` moves",
    )
    ap.add_argument(
        "--substitute",
        action="store_true",
        help="teach Substitute to every Pokemon on the field, put a doll (whole, or worn to "
        "an eighth by position in turn) in front of the first on each side, hold every cell "
        "to the port branch by branch, and count where IKA-180's Substitute fired: the cells "
        "`unsubbed` moves",
    )
    ap.add_argument(
        "--eject",
        action="store_true",
        help="hand Eject Button, Emergency Exit, Wimp Out or Red Card to every Pokemon on "
        "the field, by position in turn, hold every cell to the port branch by branch, and "
        "count where IKA-191's switches fired: the cells `uneject` moves",
    )
    ap.add_argument(
        "--trace-sync",
        action="store_true",
        help="keep positions with a Trace holder on the bench or a Synchronize holder on "
        "the field, hold every cell to the port branch by branch, and count where IKA-203's "
        "Trace or Synchronize fired: the cells `untraced` or `unsynced` moves",
    )
    ap.add_argument(
        "--toxic-debris",
        action="store_true",
        help="keep positions with Toxic Debris on the field, hold every cell to the port "
        "branch by branch, and count where IKA-173's rule fired: the cells the old one moves",
    )
    ap.add_argument(
        "--rampage",
        action="store_true",
        help="--using every rampage move (Outrage, Petal Dance, Raging Fury, Thrash), with "
        "a rampage on its second turn put on each knower on the field first -- recorded "
        "games carry only the bare marker from before IKA-174 -- so the length's branch, "
        "the end and the confusion are held to the port",
    )
    ap.add_argument(
        "--random-target",
        action="store_true",
        help="keep positions with a randomNormal move (Outrage, Uproar, Struggle...) on the "
        "field, hold every cell that uses one to the port branch by branch, and count where "
        "the draw of the foe fired: the cells `undrawn` moves (IKA-178)",
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
    ap.add_argument(
        "--priority-block",
        action="store_true",
        help="keep positions with Armor Tail, Queenly Majesty or Dazzling on the field, "
        "hold every cell that may use a priority move to the port branch by branch, and "
        "count where the ability's stop fired",
    )
    ap.add_argument(
        "--hazards",
        action="store_true",
        help="teach every Pokemon on the field a foeSide hazard first -- no recorded team "
        "has one -- hold every cell that uses one to the port branch by branch, and count "
        "where the placement fired: the cells the user's side would move",
    )
    ap.add_argument(
        "--charging",
        action="store_true",
        help="keep positions with a charging move on the field, put every other one "
        "mid-charge with a stored target, hold the cells that use one (every cell mid-charge) "
        "to the port branch by branch, and count where the stored target fired",
    )
    ap.add_argument(
        "--hammer",
        action="store_true",
        help="teach Gigaton Hammer to every Pokemon on the field first -- no recorded team "
        "has it -- hold every cell that uses it to the port branch by branch, and on every "
        "other position make it the last move and count the slots whose menu drops it",
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
    if args.rampage:
        using |= rampage_moves(reg)
    blockers = PRIORITY_BLOCKING_ABILITIES if args.priority_block else frozenset()
    debris = frozenset({"toxicdebris"}) if args.toxic_debris else frozenset()
    for move_id in sorted(using):
        move = reg.moves.get(move_id)
        if move is None or not (
            move.raw.get("breaksProtect")
            or isinstance(move.raw.get("multihit"), list)
            or move_id in TYPE_SPENDING_MOVES
            or move_id in AFTER_HIT_HAZARDS
            or move_id in rampage_moves(reg)
            or move_id == "perishsong"
            or move_id == "helpinghand"
            or move_id == "taunt"
            or move.raw.get("drain")
            or judged(move)
        ):
            ap.error(
                f"--using takes breaksProtect, ranged multi-hit, type-spending "
                f"({sorted(TYPE_SPENDING_MOVES)}), hazard-laying "
                f"({sorted(AFTER_HIT_HAZARDS)}), rampage "
                f"({sorted(rampage_moves(reg))}), drain or judged status moves, "
                f"perishsong, helpinghand or taunt; {move_id} is none of them"
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
        charging = charge_moves(reg) if args.charging else frozenset()
        randomers = random_target_moves(reg) if args.random_target else frozenset()
        positions, other_format = recorded_positions(
            reg,
            args,
            holding - {args.give},
            using | charging | randomers,
            blockers | debris,
            args.frozen,
            locked=args.choice_locked,
            **(
                {"texts": frozenset({'"trace"', '"synchronize"'}), "keep": trace_sync_on_board}
                if args.trace_sync
                else {}
            ),
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
    randomers = random_target_moves(reg) if args.random_target else frozenset()
    if randomers:
        positions = [pos for pos in positions if knows_on_field(pos, randomers)]
    if args.rampage:
        raged = sum(enrage(pos, rampage_moves(reg)) for pos in positions)
        print(f"a rampage on its second turn put on {raged} Pokemon on the field")
    if blockers:
        positions = [pos for pos in positions if ability_on_field(pos, blockers)]
    if debris:
        positions = [pos for pos in positions if ability_on_field(pos, debris)]
    if args.frozen:
        positions = [pos for pos in positions if frozen_on_field(pos)]
    if args.trace_sync:
        positions = [pos for pos in positions if trace_sync_on_board(pos)]
    if args.choice_locked:
        positions = [pos for pos in positions if choice_locked_on_field(reg, pos)]
    quick =priority_moves(reg) if args.terrain or blockers else frozenset()
    if args.terrain:
        for pos in positions:
            pos.field.terrain = args.terrain
            pos.field.terrain_duration = 5
        positions = [pos for pos in positions if priority_on_field(pos, quick)]
        print(f"{args.terrain} laid on {len(positions)} positions with a priority move on the field")
    if args.confused:
        dazed = sum(confuse(pos, index) for index, pos in enumerate(positions))
        print(f"a confusion put on {dazed} Pokemon on the field")
    if args.substitute:
        dolls = sum(substitute_on_field(reg, pos, index) for index, pos in enumerate(positions))
        print(f"Substitute taught and {dolls} dolls put up on the field")
    if args.confusion_guard:
        guards = Counter(guard_confusion(reg, pos, index) for index, pos in enumerate(positions))
        print(f"Confuse Ray taught and a confusion guard put on: {dict(guards)}")
    ejected_by: dict[int, str] = {}
    if args.eject:
        for index, pos in enumerate(positions):
            ejected_by[id(pos)] = hand_out_eject(reg, pos, index)
        print(f"handed out on the field, by position: {dict(Counter(ejected_by.values()))}")
    if args.salt_cure:
        salted = sum(salt(pos) for pos in positions)
        print(f"Salt Cure put on {salted} Pokemon on the field")
    if args.hazards:
        dealt = [0]
        taught = sum(teach_hazards(reg, pos, dealt) for pos in positions)
        print(f"a hazard taught to {taught} Pokemon on the field")
    charging = charge_moves(reg) if args.charging else frozenset()
    mid_charge: set[int] = set()
    if args.charging:
        positions = [pos for pos in positions if knows_on_field(pos, charging)]
        put = 0
        for index, pos in enumerate(positions):
            if index % 2:
                count = charge_mid_turn(reg, pos, charging)
                if count:
                    mid_charge.add(id(pos))
                    put += count
        print(
            f"{len(positions)} positions with a charging move on the field; "
            f"{put} Pokemon put mid-charge on {len(mid_charge)} of them"
        )
    remembered: dict[int, list[tuple[int, int]]] = {}
    if args.hammer:
        taught_hammer = 0
        for index, pos in enumerate(positions):
            taught_here = teach_hammer(reg, pos, remember=bool(index % 2))
            taught_hammer += len(taught_here)
            if index % 2 and taught_here:
                remembered[id(pos)] = taught_here
        print(
            f"Gigaton Hammer taught to {taught_hammer} Pokemon on the field; the last move "
            f"on {len(remembered)} positions"
        )
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
    # Where a priority move may go beside a blocking ability, and where the ability stopped it.
    block_used = block_wrong = blocked = blocked_wrong = 0
    blocked_worst = 0.0
    # Under Salt Cure, every cell; and where the mod's fraction moved the answer.
    cured = cured_wrong = cured_refused = cured_fired = cured_fired_wrong = 0
    cured_worst = 0.0
    # Beside a confused Pokemon, every cell; and where IKA-177's rule moved the answer.
    dazed_cells = dazed_wrong = dazed_refused = dazed_fired = dazed_fired_wrong = 0
    dazed_worst = 0.0
    # Under a confusion guard, every cell; and where IKA-189's rule moved the answer.
    guard_cells = guard_wrong = guard_refused = guard_fired = guard_fired_wrong = 0
    eject_cells = eject_wrong = eject_refused = eject_fired = eject_fired_wrong = 0
    eject_fired_by: Counter = Counter()
    eject_refused_by: Counter = Counter()
    eject_worst = 0.0
    # Beside Trace or Synchronize, every cell; and where IKA-203's rules moved the answer.
    ts_cells = ts_wrong = ts_refused = ts_fired = ts_fired_wrong = 0
    ts_traced = ts_synced = 0
    ts_worst = 0.0
    guard_by_refusal = guard_by_roll = 0
    guard_worst = 0.0
    # Beside a doll, every cell; and where IKA-180's Substitute moved the answer.
    doll_cells = doll_wrong = doll_refused = doll_fired = doll_fired_wrong = 0
    doll_by_use = doll_by_doll = 0
    doll_worst = 0.0
    # Beside a frozen Pokemon, every cell; and where a thaw moved the answer.
    icy = icy_wrong = icy_refused = thawed = thawed_wrong = 0
    thawed_worst = 0.0
    # Beside a Choice lock, every cell; and where IKA-179's lock moved the answer.
    lock_cells = lock_wrong = lock_refused = lock_fired = lock_fired_wrong = 0
    lock_worst = 0.0
    # Where a hazard was used, and where laying it on the foe's side moved the answer.
    laid = laid_wrong = laid_refused = laid_fired = laid_fired_wrong = 0
    laid_fired_refused = laid_fired_paused = laid_wrong_elsewhere = 0
    laid_worst = 0.0
    hazard_moves = frozenset(HAZARD_MOVES)
    # Beside Toxic Debris, every cell; and where IKA-173's rule moved the answer.
    debris_cells = debris_wrong = debris_refused = debris_fired = debris_fired_wrong = 0
    debris_worst = 0.0
    # Where a charging move was used or fired, and where its stored target moved the answer.
    charged = charged_wrong = aimed = aimed_wrong = aimed_paused = 0
    aimed_worst = 0.0
    # Where a randomNormal move was used, and where drawing its foe moved the answer.
    drew = drew_wrong = drew_fired = drew_fired_wrong = drew_paused = 0
    drew_worst = 0.0
    # Where the hammer was used, and the remembered slots whose menu dropped it.
    hammered = hammered_wrong = dropped = dropped_control = 0

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

        if args.charging:
            node = rustnode.node_for(reg)
            every = id(pos) in mid_charge
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    if not (every or uses(a, charging) or uses(b, charging)):
                        continue
                    charged += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with untargeted():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    charged_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        aimed += 1
                        aimed_wrong += bool(wrong)
                        aimed_paused += bool(here.suspended)
                        for index in range(len(evaluators)):
                            aimed_worst = max(
                                aimed_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} with a charge: {wrong[0][:200]}")

        if randomers:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    if not (uses(a, randomers) or uses(b, randomers)):
                        continue
                    drew += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with undrawn():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    drew_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        drew_fired += 1
                        drew_fired_wrong += bool(wrong)
                        drew_paused += bool(here.suspended)
                        for index in range(len(evaluators)):
                            drew_worst = max(
                                drew_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} using a randomNormal move: {wrong[0][:200]}")

        if args.hammer:
            node = rustnode.node_for(reg)
            for side_index, slot in remembered.get(id(pos), ()):
                offered = {
                    getattr(act.slots[slot], "move_id", None)
                    for act in side_actions(reg, pos, side_index)
                }
                with hammer_twice():
                    control_offered = {
                        getattr(act.slots[slot], "move_id", None)
                        for act in side_actions(reg, pos, side_index)
                    }
                dropped += "gigatonhammer" not in offered
                dropped_control += "gigatonhammer" in control_offered
            hammer = frozenset({"gigatonhammer"})
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    if not (uses(a, hammer) or uses(b, hammer)):
                        continue
                    hammered += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    hammered_wrong += bool(wrong)
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} using the hammer: {wrong[0][:200]}")

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

        if blockers:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    if not (
                        uses_priority(reg, pos, 0, a, quick) or uses_priority(reg, pos, 1, b, quick)
                    ):
                        continue
                    block_used += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unblocked():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    block_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        blocked += 1
                        blocked_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            blocked_worst = max(
                                blocked_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} beside a blocking ability: {wrong[0][:200]}")

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

        if args.hazards:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    if not (uses(a, hazard_moves) or uses(b, hazard_moves)):
                        continue
                    laid += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with hazards_on_the_users_side():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    # A cell the port refuses for a gate that is not the hazard's is filled
                    # in Python and is counted apart.
                    refused_here = wrong == ["the port refused the turn"]
                    laid_refused += refused_here
                    if refused_here:
                        wrong = []
                    laid_wrong += bool(wrong)
                    if wrong and node is not None:
                        # Whether the two engines at least lay the same side conditions: a
                        # cell they part on elsewhere is a disagreement the hazard only
                        # led the tool to (IKA-165 met the port's missing Perish Song so).
                        laid_wrong_elsewhere += sides_agree(node, pos, a, b, here, budget)
                    mine, theirs = outcome(here), outcome(control)
                    if differ(mine, theirs):
                        laid_fired += 1
                        laid_fired_wrong += bool(wrong)
                        laid_fired_refused += refused_here
                        # `branch_differences` holds a paused turn's weight, not its
                        # position, so a placement that shows only there is not held.
                        laid_fired_paused += not differ(
                            (mine[0], {}, mine[2]), (theirs[0], {}, theirs[2])
                        )
                        for index in range(len(evaluators)):
                            laid_worst = max(
                                laid_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} using a hazard: {wrong[0][:200]}")

        if debris:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    debris_cells += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with old_toxic_debris():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    refused_here = wrong == ["the port refused the turn"]
                    debris_refused += refused_here
                    if refused_here:
                        wrong = []
                    debris_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        debris_fired += 1
                        debris_fired_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            debris_worst = max(
                                debris_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} beside Toxic Debris: {wrong[0][:200]}")

        if args.confused and confused_on_field(pos):
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    dazed_cells += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unconfused():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    refused_here = wrong == ["the port refused the turn"]
                    dazed_refused += refused_here
                    if refused_here:
                        wrong = []
                    dazed_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        dazed_fired += 1
                        dazed_fired_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            dazed_worst = max(
                                dazed_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} beside a confused Pokemon: {wrong[0][:200]}")

        if args.confusion_guard:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    guard_cells += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unguarded():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    refused_here = wrong == ["the port refused the turn"]
                    guard_refused += refused_here
                    if refused_here:
                        wrong = []
                    guard_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        with unguarded("refusals"):
                            unrefused = resolve_turn(reg, pos, [a, b], budget=budget)
                        with unguarded("roll"):
                            unrolled = resolve_turn(reg, pos, [a, b], budget=budget)
                        guard_by_refusal += differ(outcome(here), outcome(unrefused))
                        guard_by_roll += differ(outcome(here), outcome(unrolled))
                        guard_fired += 1
                        guard_fired_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            guard_worst = max(
                                guard_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} under a confusion guard: {wrong[0][:200]}")

        if args.substitute:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    doll_cells += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unsubbed():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    refused_here = wrong == ["the port refused the turn"]
                    doll_refused += refused_here
                    if refused_here:
                        wrong = []
                    doll_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        with unsubbed("use"):
                            unused = resolve_turn(reg, pos, [a, b], budget=budget)
                        with unsubbed("doll"):
                            undolled = resolve_turn(reg, pos, [a, b], budget=budget)
                        doll_by_use += differ(outcome(here), outcome(unused))
                        doll_by_doll += differ(outcome(here), outcome(undolled))
                        doll_fired += 1
                        doll_fired_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            doll_worst = max(
                                doll_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} beside a doll: {wrong[0][:200]}")

        if args.eject:
            node = rustnode.node_for(reg)
            handed = ejected_by.get(id(pos), "?")
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    eject_cells += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with uneject():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    refused_here = wrong == ["the port refused the turn"]
                    eject_refused += refused_here
                    eject_refused_by[handed] += refused_here
                    if refused_here:
                        wrong = []
                    eject_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        eject_fired += 1
                        eject_fired_by[handed] += 1
                        eject_fired_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            eject_worst = max(
                                eject_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} with {handed} on the field: {wrong[0][:200]}")

        if args.trace_sync:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    ts_cells += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with untraced():
                        no_trace = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unsynced():
                        no_sync = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    refused_here = wrong == ["the port refused the turn"]
                    ts_refused += refused_here
                    if refused_here:
                        wrong = []
                    ts_wrong += bool(wrong)
                    traced = differ(outcome(here), outcome(no_trace))
                    synced = differ(outcome(here), outcome(no_sync))
                    ts_traced += traced
                    ts_synced += synced
                    if traced or synced:
                        ts_fired += 1
                        ts_fired_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            ts_worst = max(
                                ts_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} beside Trace or Synchronize: {wrong[0][:200]}")

        if args.frozen:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    icy += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unthawed():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    refused_here = wrong == ["the port refused the turn"]
                    icy_refused += refused_here
                    if refused_here:
                        wrong = []
                    icy_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        thawed += 1
                        thawed_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            thawed_worst = max(
                                thawed_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} beside a frozen Pokemon: {wrong[0][:200]}")

        if args.choice_locked:
            node = rustnode.node_for(reg)
            for i, a in enumerate(row):
                for j, b in enumerate(col):
                    lock_cells += 1
                    here = resolve_turn(reg, pos, [a, b], budget=budget)
                    with unlocked():
                        control = resolve_turn(reg, pos, [a, b], budget=budget)
                    wrong = (
                        ["no warm process"]
                        if node is None
                        else branch_differences(node, reg, pos, a, b, here, budget)
                    )
                    refused_here = wrong == ["the port refused the turn"]
                    lock_refused += refused_here
                    if refused_here:
                        wrong = []
                    lock_wrong += bool(wrong)
                    if differ(outcome(here), outcome(control)):
                        lock_fired += 1
                        lock_fired_wrong += bool(wrong)
                        for index in range(len(evaluators)):
                            lock_worst = max(
                                lock_worst,
                                abs(float(got[index][i, j] - expected[index][i, j])),
                            )
                    if wrong and shown < 5:
                        shown += 1
                        print(f"  cell {(i, j)} beside a Choice lock: {wrong[0][:200]}")

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
    if args.confused:
        print("\n  beside a confused Pokemon, every cell -- held branch by branch")
        print(f"    {dazed_cells} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {dazed_wrong}")
        print(f"    cells the port refused, filled in Python and not held  {dazed_refused}")
        print("  where the length, the odds or the berry fired -- the cells `unconfused` moves")
        print(f"    {dazed_fired} of {dazed_cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {dazed_fired_wrong}")
        print(f"    worst cell difference there  {dazed_worst:.3e}")
    if args.substitute:
        print("\n  beside a doll, every cell -- held branch by branch")
        print(f"    {doll_cells} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {doll_wrong}")
        print(f"    cells the port refused, filled in Python and not held  {doll_refused}")
        print("  where Substitute fired -- the cells `unsubbed` moves")
        print(f"    {doll_fired} of {doll_cells} cells")
        print(f"      the use alone moves  {doll_by_use}; the doll alone  {doll_by_doll}")
        print(f"    cells whose branches, weights, notes or positions differ  {doll_fired_wrong}")
        print(f"    worst cell difference there  {doll_worst:.3e}")
    if args.confusion_guard:
        print("\n  under a confusion guard, every cell -- held branch by branch")
        print(f"    {guard_cells} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {guard_wrong}")
        print(f"    cells the port refused, filled in Python and not held  {guard_refused}")
        print("  where a refusal or the self-hit's roll fired -- the cells `unguarded` moves")
        print(f"    {guard_fired} of {guard_cells} cells")
        print(f"      a refusal alone moves  {guard_by_refusal}; the roll alone  {guard_by_roll}")
        print(f"    cells whose branches, weights, notes or positions differ  {guard_fired_wrong}")
        print(f"    worst cell difference there  {guard_worst:.3e}")
    if args.eject:
        print("\n  with an Eject Button, Emergency Exit, Wimp Out or Red Card on the field")
        print(f"    {eject_cells} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {eject_wrong}")
        print(
            f"    cells the port refused, filled in Python and not held  {eject_refused}  "
            f"{dict(eject_refused_by)}"
        )
        print("  where a switch fired -- the cells `uneject` moves")
        print(f"    {eject_fired} of {eject_cells} cells  {dict(eject_fired_by)}")
        print(f"    cells whose branches, weights, notes or positions differ  {eject_fired_wrong}")
        print(f"    worst cell difference there  {eject_worst:.3e}")
    if args.trace_sync:
        print("\n  with a Trace holder on the bench or a Synchronize holder on the field")
        print(f"    {ts_cells} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {ts_wrong}")
        print(f"    cells the port refused, filled in Python and not held  {ts_refused}")
        print("  where Trace or Synchronize fired -- the cells `untraced` or `unsynced` moves")
        print(f"    {ts_fired} of {ts_cells} cells (Trace {ts_traced}, Synchronize {ts_synced})")
        print(f"    cells whose branches, weights, notes or positions differ  {ts_fired_wrong}")
        print(f"    worst cell difference there  {ts_worst:.3e}")
    if args.frozen:
        print("\n  beside a frozen Pokemon, every cell -- held branch by branch")
        print(f"    {icy} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {icy_wrong}")
        print(f"    cells the port refused, filled in Python and not held  {icy_refused}")
        print("  where a thaw fired -- the cells `unthawed` moves")
        print(f"    {thawed} of {icy} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {thawed_wrong}")
        print(f"    worst cell difference there  {thawed_worst:.3e}")
    if args.choice_locked:
        print("\n  beside a Choice lock, every cell -- held branch by branch")
        print(f"    {lock_cells} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {lock_wrong}")
        print(f"    cells the port refused, filled in Python and not held  {lock_refused}")
        print("  where IKA-179's lock fired -- the cells `unlocked` moves")
        print(f"    {lock_fired} of {lock_cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {lock_fired_wrong}")
        print(f"    worst cell difference there  {lock_worst:.3e}")
    if blockers:
        print("\n  beside a priority-blocking ability, cells that may use a priority move")
        print(f"    {block_used} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {block_wrong}")
        print("  where the ability stopped one -- the cells `unblocked` moves")
        print(f"    {blocked} of {block_used} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {blocked_wrong}")
        print(f"    worst cell difference there  {blocked_worst:.3e}")
    if args.hazards:
        print("\n  where a hazard was used -- every one held branch by branch")
        print(f"    {laid} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {laid_wrong}")
        print(f"      of which the side conditions agree, branch by branch  {laid_wrong_elsewhere}")
        print(f"    cells the port refused, filled in Python and not held  {laid_refused}")
        print("  where the placement fired -- the cells the user's side would move")
        print(f"    {laid_fired} of {laid} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {laid_fired_wrong}")
        print(f"    cells the port refused, not held  {laid_fired_refused}")
        print(f"    cells that differ only in a paused branch, not compared  {laid_fired_paused}")
        print(f"    worst cell difference there  {laid_worst:.3e}")
    if debris:
        print("\n  beside Toxic Debris, every cell -- held branch by branch")
        print(f"    {debris_cells} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {debris_wrong}")
        print(f"    cells the port refused, filled in Python and not held  {debris_refused}")
        print("  where IKA-173's rule fired -- the cells the old Toxic Debris moves")
        print(f"    {debris_fired} of {debris_cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {debris_fired_wrong}")
        print(f"    worst cell difference there  {debris_worst:.3e}")
    if args.charging:
        print("\n  with a charge -- every cell mid-charge, and the ones using a charging move")
        print(f"    {charged} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {charged_wrong}")
        print("  where the stored target fired -- the cells `untargeted` moves")
        print(f"    {aimed} of {charged} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {aimed_wrong}")
        print(f"    cells with a paused branch, whose position is not compared  {aimed_paused}")
        print(f"    worst cell difference there  {aimed_worst:.3e}")
    if randomers:
        print("\n  where a randomNormal move was used -- every one held branch by branch")
        print(f"    {drew} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {drew_wrong}")
        print("  where the draw of the foe fired -- the cells `undrawn` moves")
        print(f"    {drew_fired} of {drew} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {drew_fired_wrong}")
        print(f"    cells with a paused branch, whose position is not compared  {drew_paused}")
        print(f"    worst cell difference there  {drew_worst:.3e}")
    if args.hammer:
        print("\n  where Gigaton Hammer was used -- every one held branch by branch")
        print(f"    {hammered} of {cells} cells")
        print(f"    cells whose branches, weights, notes or positions differ  {hammered_wrong}")
        print("  slots whose last move was the hammer (the menu is Python's)")
        print(f"    {sum(len(v) for v in remembered.values())} slots, the menu drops it on {dropped}")
        print(f"    offered there with `cantusetwice` taken out  {dropped_control}")
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
    if block_wrong:
        failed.append(f"{block_wrong} cells beside a blocking ability differ by branch")
    if dazed_wrong:
        failed.append(f"{dazed_wrong} cells beside a confused Pokemon differ by branch")
    if args.confused and not dazed_fired:
        failed.append("IKA-177's confusion moved no cell, so agreeing here says nothing")
    if doll_wrong:
        failed.append(f"{doll_wrong} cells beside a doll differ by branch")
    if args.substitute and not doll_fired:
        failed.append("IKA-180's Substitute moved no cell, so agreeing here says nothing")
    if guard_wrong:
        failed.append(f"{guard_wrong} cells under a confusion guard differ by branch")
    if args.confusion_guard and not guard_fired:
        failed.append("IKA-189's confusion guard moved no cell, so agreeing here says nothing")
    if eject_wrong:
        failed.append(f"{eject_wrong} cells with an ejecting holder differ by branch")
    if args.eject and not eject_fired:
        failed.append("IKA-191's switches moved no cell, so agreeing here says nothing")
    if ts_wrong:
        failed.append(f"{ts_wrong} cells beside Trace or Synchronize differ by branch")
    if args.trace_sync and not ts_fired:
        failed.append("IKA-203's Trace and Synchronize moved no cell, so agreeing here says nothing")
    if icy_wrong:
        failed.append(f"{icy_wrong} cells beside a frozen Pokemon differ by branch")
    if args.frozen and not thawed:
        failed.append("no thaw fired in any cell, so agreeing here says nothing")
    if lock_wrong:
        failed.append(f"{lock_wrong} cells beside a Choice lock differ by branch")
    if args.choice_locked and not lock_fired:
        failed.append("IKA-179's lock moved no cell, so agreeing here says nothing")
    if blockers and not blocked:
        failed.append("no blocking ability stopped anything in any cell, so agreeing here says nothing")
    if laid_wrong:
        failed.append(f"{laid_wrong} cells using a hazard differ by branch")
    if charged_wrong:
        failed.append(f"{charged_wrong} cells with a charge differ by branch")
    if args.charging and not aimed:
        failed.append("the stored target moved no cell, so agreeing here says nothing")
    if drew_wrong:
        failed.append(f"{drew_wrong} cells using a randomNormal move differ by branch")
    if randomers and not drew_fired:
        failed.append("drawing the foe moved no cell, so agreeing here says nothing")
    if hammered_wrong:
        failed.append(f"{hammered_wrong} cells using the hammer differ by branch")
    if args.hammer and not (hammered and dropped):
        failed.append("the hammer was used in no cell or dropped from no menu")
    if args.hazards and not laid_fired:
        failed.append("laying a hazard on the foe's side moved no cell, so agreeing here says nothing")
    if debris_wrong:
        failed.append(f"{debris_wrong} cells beside Toxic Debris differ by branch")
    if debris and not debris_fired:
        failed.append("the new Toxic Debris rule moved no cell, so agreeing here says nothing")
    if failed:
        print(f"\nFAIL ({args.budget}): " + "; ".join(failed))
        sys.exit(1)
    print(f"\nOK ({args.budget})")


if __name__ == "__main__":
    main()
