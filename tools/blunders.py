"""Moves that fail by rule, counted from recorded games with no model involved.

The project's whole scale is internal: `hp-share` to the top agent is 139.9 Elo, and every
point of it was measured against another version of itself. Nothing has been checked
against a standard a person would recognise, and the record has one instance of what that
costs -- a 21.8-point miss found by a human reading one game after seven hours of automated
measurement found nothing.

This is the cheapest human standard available: moves that cannot work. Not moves that are
unwise, which needs an opinion about the position, but moves the rules refuse.

**Every row is reported against its OPPORTUNITIES, not just as a count.** A zero means
nothing on its own: `actions._usable_move_slots` filters choice-time legality -- locks,
Taunt, Torment, Throat Chop, Disable, Encore, Choice, PP -- and a move that is merely
doomed stays on the menu, exactly as Showdown leaves it there. But that has to be shown
rather than assumed, so each check counts how often the losing choice was actually
available and how often it was taken. A rate needs a denominator; a count without one
cannot tell "never blundered" from "never could".

    fake out late     Fake Out fails unless the user has not acted since switching in.
                      Showdown gates it on `activeMoveActions`, which the recorded position
                      carries, so the opportunity and the choice are both exact.
    switch read       a status move fired at a field where every active already has a
                      major status. **Not a blunder.** A move targets a slot, so if the
                      opponent switches it lands on whoever arrives -- this is a bet on a
                      switch, and the rate is reported because it is worth knowing, not
                      because it is wrong.
    stall repeat      Protect and its family lose two thirds of their success rate on
                      each consecutive use. Read off the engine's own `stall` volatile,
                      which is what divides the rate, and against the family the
                      REGULATION marks `stallingMove` -- a hand-written list had Wide Guard
                      and Quick Guard in it, which do not share the counter, and our own
                      Toxapex runs Wide Guard. Not a certain failure either way, and
                      sometimes right, so it is a rate to look at and not a blunder count.
    no pp             a move chosen with zero PP, which should be unreachable and is here
                      as a check on this harness rather than on the agent.

Reported by turn band, because the recorded picture is that the value function is weakest
early -- turn-1 AUC 0.70 against 0.96 by turn 10 -- and the open question is whether the
play is weak in the same place.

    uv run python tools/blunders.py
    uv run python tools/blunders.py data/selfplay-gen11L --games 4000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

#: Moves that apply a major status. Not a blunder list -- see the docstring.
STATUS_MOVES = frozenset(
    {"toxic", "willowisp", "thunderwave", "spore", "sleeppowder", "poisonpowder",
     "glare", "hypnosis", "yawn", "darkvoid", "stunspore"}
)
MAJOR = frozenset({"psn", "tox", "brn", "par", "slp", "frz"})

MOVE_RE = re.compile(r"move (\d+)")


def stalling_moves(format_id: str) -> frozenset[str]:
    """The Protect family, from the regulation dump rather than from memory."""
    path = (
        Path(__file__).resolve().parents[1] / "configs" / "regulations" / f"{format_id}.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(m["id"] for m in data.get("moves", []) if m.get("stallingMove"))


def stall_counter(mon: dict) -> int:
    """The engine's consecutive-Protect counter, 0 when it is not running.

    Showdown's `stall` condition is what halves and thirds the success rate, and it
    is removed as soon as a turn goes by without a Protect -- so this is the rule,
    where `lastMove` was a guess at it.
    """
    for v in mon.get("volatiles", []):
        key = v.get("id") if isinstance(v, dict) else str(v)
        if key == "stall":
            counter = v.get("counter") if isinstance(v, dict) else None
            return int(counter) if isinstance(counter, int) else 1
    return 0


def struggling(mon: dict) -> bool:
    """True when nothing on the moveset can be selected, so `move N` means Struggle.

    Showdown offers Struggle as a single fake move slot, so the choice string still says
    `move 1` -- and reading that against the recorded moveset finds slot 1, which may well
    be the move that ran out of PP. The first draft of this tool reported exactly that as
    "a move chosen with no PP", once, and it was a Choice-locked Garchomp out of Earthquake
    with Struggle as its only option. The harness check caught the harness.
    """
    locked = None
    for v in mon.get("volatiles", []):
        if (v.get("id") if isinstance(v, dict) else str(v)) == "choicelock":
            locked = v.get("move") if isinstance(v, dict) else None
    for move in mon.get("moves", []):
        if int(move.get("pp", 0)) <= 0 or move.get("disabled"):
            continue
        if locked and str(move.get("id")) != str(locked):
            continue
        return False
    return True


BANDS = ((1, 1), (2, 3), (4, 6), (7, 10), (11, 99))
LABELS = [f"{lo}-{hi}" if lo != hi else str(lo) for lo, hi in BANDS]


def band_of(turn: int) -> str:
    for lo, hi in BANDS:
        if lo <= turn <= hi:
            return f"{lo}-{hi}" if lo != hi else str(lo)
    return LABELS[-1]


def actives(side: dict) -> list[dict]:
    """The Pokemon on the floor for one side, in the order the choice string addresses."""
    out = [m for m in side.get("pokemon", []) if m.get("activeIndex") is not None]
    return sorted(out, key=lambda m: m.get("activeIndex", 0))


def slot_moves(chosen: str, mons: list[dict]) -> list[tuple[dict, dict]]:
    """(the Pokemon, the move slot it was told to use) for each part of a choice string."""
    out: list[tuple[dict, dict]] = []
    for index, part in enumerate(str(chosen).split(",")):
        if index >= len(mons):
            break
        match = MOVE_RE.search(part.strip())
        if match is None:
            continue
        # A struggling Pokemon is offered one fake slot, so `move 1` is Struggle and not
        # slot 1 of the recorded moveset. Reading it as the latter is how this tool first
        # reported a zero-PP choice that never happened.
        if struggling(mons[index]):
            continue
        slot = int(match.group(1)) - 1
        moves = mons[index].get("moves", [])
        if 0 <= slot < len(moves):
            out.append((mons[index], moves[slot]))
    return out


def offered(actions: list[str], mons: list[dict], move_ids: frozenset[str]) -> bool:
    """Was any of `move_ids` selectable for any active, in the menu the search was given?"""
    for action in actions:
        for mon, move in slot_moves(action, mons):
            if str(move.get("id", "")) in move_ids and mon is not None:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pools", nargs="*", default=None)
    ap.add_argument("--games", type=int, default=0, help="stop after this many games (0 = all)")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    args = ap.parse_args()

    pools = args.pools or sorted(d for d in glob.glob("data/selfplay-*") if os.path.isdir(d))
    stall = stalling_moves(args.format)
    keys = ("decisions", "fakeout_chance", "fakeout_late", "status_chance", "status_bad",
            "stall_chance", "stall_repeat", "no_pp")
    c: dict[str, Counter] = {k: Counter() for k in keys}

    games = 0
    for pool in pools:
        for path in sorted(glob.glob(os.path.join(pool, "*.jsonl"))):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if args.games and games >= args.games:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    games += 1
                    for decision in record.get("decisions", []):
                        if decision.get("kind") != "move":
                            continue
                        band = band_of(int(decision.get("turn", 0)))
                        c["decisions"][band] += 1
                        sides = (decision.get("position") or {}).get("sides") or []
                        if not sides:
                            continue
                        mons = actives(sides[0])
                        foes = actives(sides[1]) if len(sides) > 1 else []
                        menu = list(decision.get("ownActions") or [])

                        # Opportunities: was a doomed choice on the menu at all?
                        late = [m for m in mons if int(m.get("activeMoveActions", 0)) > 0]
                        if late and offered(menu, mons, frozenset({"fakeout"})):
                            c["fakeout_chance"][band] += 1
                        statused = foes and all(
                            str(f.get("status") or "") in MAJOR for f in foes
                        )
                        if statused and offered(menu, mons, STATUS_MOVES):
                            c["status_chance"][band] += 1
                        if any(stall_counter(m) for m in mons) and offered(
                            menu, mons, stall
                        ):
                            c["stall_chance"][band] += 1

                        # And what was taken.
                        for mon, move in slot_moves(decision.get("ownChosen", ""), mons):
                            move_id = str(move.get("id", ""))
                            if int(move.get("pp", 1)) <= 0:
                                c["no_pp"][band] += 1
                            if move_id == "fakeout" and int(mon.get("activeMoveActions", 0)) > 0:
                                c["fakeout_late"][band] += 1
                            if move_id in STATUS_MOVES and statused:
                                c["status_bad"][band] += 1
                            if move_id in stall and stall_counter(mon):
                                c["stall_repeat"][band] += 1
            if args.games and games >= args.games:
                break

    total_decisions = sum(c["decisions"].values())
    print(f"  {games:,} games, {total_decisions:,} move decisions"
          f" from {len(pools)} pool(s)\n")
    print(f"  {'':<26}" + "".join(f"{lab:>9}" for lab in LABELS) + f"{'all':>10}")
    print(f"  {'move decisions':<26}"
          + "".join(f"{c['decisions'][lab]:>9,}" for lab in LABELS)
          + f"{total_decisions:>10,}")
    print()
    for taken, chance, label in (
        ("fakeout_late", "fakeout_chance", "fake out too late"),
        ("status_bad", "status_chance", "status move, all foes statused"),
        ("stall_repeat", "stall_chance", "protect back to back"),
    ):
        got, opp = sum(c[taken].values()), sum(c[chance].values())
        print(f"  {label:<26}" + "".join(f"{c[taken][lab]:>9,}" for lab in LABELS)
              + f"{got:>10,}")
        print(f"  {'  when it was on the menu':<26}"
              + "".join(f"{c[chance][lab]:>9,}" for lab in LABELS) + f"{opp:>10,}")
        rate = f"{got / opp:.2%}" if opp else "no opportunity"
        print(f"  {'  taken':<26}{'':>{9 * len(LABELS)}}{rate:>10}\n")
    no_pp = sum(c["no_pp"].values())
    print(f"  {'move with no PP':<26}" + "".join(f"{c['no_pp'][lab]:>9,}" for lab in LABELS)
          + f"{no_pp:>10,}   (harness check; must be 0)")

    print("\n  A zero with a zero denominator says the choice was never offered, not that\n"
          "  the agent declined it. Read the two lines together or not at all.")


if __name__ == "__main__":
    main()
