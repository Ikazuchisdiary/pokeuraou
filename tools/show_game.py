"""Renders one recorded game as a readable log, in Japanese.

The self-play records carry positions, candidate actions, both sides' equilibrium mixtures
and the search's value, but nothing human-readable -- so a game that looks wrong cannot be
looked at. This prints the turn-by-turn view, and it prints the *mixture* rather than a
single chosen move, because a mixture is what the search actually produces and reducing it
to "the best move" would be the one output this project is not allowed to have.

Choice strings are turned back into names using the position they were offered in: "move 3 1"
means the third move slot of the first active Pokemon, aimed at the opponent's first slot,
and the position says which move that is.

Written to a file rather than to the terminal, because the Windows console encoding mangles
Japanese and the point of this tool is to be read.

    uv run python tools/show_game.py --game 0 --out game.txt
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import RECHARGE, STRUGGLE, is_struggling
from pokeuraou.hidden import seen_slots
from pokeuraou.names import Localiser, load_names
from pokeuraou.position import Position
from pokeuraou.regulation import Regulation, load_regulation, to_id
from pokeuraou.resolve import RESIDUAL_PHASE
from pokeuraou.teams import all_selections


def hp_bar(current: int, maximum: int, width: int = 10) -> str:
    if maximum <= 0:
        return " " * width
    filled = round(current / maximum * width)
    return "#" * filled + "." * (width - filled)


def occupant(loc: Localiser, side: dict, slot: int) -> str:
    """The name of whoever is standing in one active slot."""
    party = side["active"][slot] if 0 <= slot < len(side["active"]) else None
    if party is None:
        return f"枠{slot + 1}"
    mon = side["pokemon"][party]
    return loc.species(mon["species"])


def _effect_text(loc: Localiser, effect: dict) -> str:
    """One field effect: its name, then how much of it is left.

    Duration counts the residual phases still to come, this turn's included, so `残1` is
    the last turn -- which is also the turn a sandstorm deals no damage, because Showdown
    decrements before the handler runs. Layers are Spikes and Toxic Spikes, which stack
    instead of expiring, so they never carry a duration and it is not printed as zero.
    """
    out = named_id(loc, effect.get("id", ""))
    layers = effect.get("layers")
    if layers and layers > 1:
        out += f" {layers}層"
    duration = effect.get("duration")
    if duration is not None:
        out += f" 残{duration}"
    return out


def field_line(loc: Localiser, position: dict) -> str:
    """Weather, terrain, room and screens, with what is left of each.

    Printed only when something is up. A line reading "場: なし" on the great majority of
    turns would push the part a reader is looking for further down the screen every time,
    to say something they can already see from its absence.

    Everything is read from the position rather than from the event log, so this is the
    state the search was given -- which is the point of showing it next to the mixture.
    """
    field = position.get("field") or {}
    parts: list[str] = []
    for key, duration_key in (("weather", "weatherDuration"), ("terrain", "terrainDuration")):
        value = field.get(key)
        if value:
            parts.append(_effect_text(loc, {"id": value, "duration": field.get(duration_key)}))
    parts.extend(_effect_text(loc, e) for e in field.get("pseudoWeather") or [])

    for index, side in enumerate(position.get("sides") or []):
        label = "自" if index == 0 else "敵"
        own: list[str] = [_effect_text(loc, e) for e in side.get("sideConditions") or []]
        # Slot conditions are Wish and Healing Wish: they belong to one slot of one side,
        # so they are named with the slot rather than folded into the side's list.
        for slot, group in enumerate(side.get("slotConditions") or []):
            own.extend(f"{slot + 1}番:{_effect_text(loc, e)}" for e in group)
        if own:
            parts.append(f"{label} " + "・".join(own))
    return "  ｜  ".join(parts)


def name_action(
    reg: Regulation,
    loc: Localiser,
    sides: list[dict],
    actor: int,
    choice: str,
    parsed: Position | None = None,
) -> str:
    """Turns one side's choice string into names, using the position it was offered in.

    Targets are named rather than numbered. Showdown's target numbers are positional --
    1 and 2 are the foes, -1 and -2 the allies -- and "→ 相手1" makes a reader count
    slots to find out who is being hit, in a format whose whole point is to be read.
    """
    side = sides[actor]
    foes = sides[1 - actor]
    parts = []
    for slot, piece in enumerate(p.strip() for p in choice.split(",")):
        tokens = piece.split()
        if not tokens:
            continue
        party = side["active"][slot] if slot < len(side["active"]) else None
        mon = side["pokemon"][party] if party is not None else None
        who = loc.species(mon["species"]) if mon else f"枠{slot + 1}"
        if tokens[0] == "pass":
            parts.append(f"{who}: 行動なし")
        elif tokens[0] == "switch":
            index = int(tokens[1]) - 1
            incoming = side["pokemon"][index] if 0 <= index < len(side["pokemon"]) else None
            to = loc.species(incoming["species"]) if incoming else f"#{tokens[1]}"
            parts.append(f"{who}: 交代 → {to}")
        elif tokens[0] == "move":
            index = int(tokens[1]) - 1
            # `move 1` means the one fake move Showdown offers, not the first move slot,
            # when the Pokemon owes a recharge. Reading the slot instead named the log's
            # recharge turn after whatever happens to sit in slot 1 -- usually the Hyper
            # Beam that caused it, which reads as the move being used twice.
            recharging = any(
                v.get("id") == "mustrecharge" for v in (mon or {}).get("volatiles") or []
            )
            # Same shape, different cause: with every move out of PP, Showdown offers
            # Struggle and offers it as `move 1`. Reading slot 1 named a 75-turn stall's
            # closing turns after Infestation and Muddy Water, the two moves that had run
            # dry and were the reason Struggle was on offer at all -- so the log showed
            # both sides using moves they could not use, and the result made no sense.
            #
            # "Every move out of PP" is not the rule, though, and reading it that way put
            # a Choice Scarf Garchomp's Struggle down as Earthquake: the lock had left one
            # move selectable and that one was empty, so Struggle was on offer with three
            # full move slots beside it. The rule is `_usable_move_slots`, the same
            # function the search enumerates with, so the log and the legal set cannot
            # disagree about what was on the menu.
            struggling = (
                parsed is not None
                and party is not None
                and is_struggling(parsed.sides[actor].pokemon[party], reg)
            )
            move_id = (
                RECHARGE
                if recharging
                else STRUGGLE
                if struggling
                else (
                    mon["moves"][index]["id"]
                    if mon and 0 <= index < len(mon["moves"])
                    else f"?{tokens[1]}"
                )
            )
            text = f"{who}: {loc.move(move_id)}"
            # The recharge takes no target. Pre-fix records carry one anyway, because the
            # generator offered the real move's targets; printing it would teach a reader
            # a rule that does not exist.
            if not recharging and not struggling and len(tokens) > 2 and tokens[2] not in ("mega",):
                target = int(tokens[2])
                # Positive numbers are the foes' slots, negative the allies'.
                whose = foes if target > 0 else side
                # Which side, not just which species: both teams can bring Incineroar,
                # and "→ ガオガエン" then says nothing about who is being hit.
                marker = "敵" if target > 0 else "味方"
                text += f" → {marker}{occupant(loc, whose, abs(target) - 1)}"
            if tokens[-1] == "mega":
                text += " + メガ"
            parts.append(text)
        else:
            parts.append(f"{who}: {piece}")
    return " ｜ ".join(parts)


SLOT_LABELS = ("自1", "自2", "敵1", "敵2")
#: The resolver writes these after "->" for a status, where a forme change writes a species.
STATUSES = frozenset({"brn", "psn", "tox", "par", "slp", "frz", "fnt"})


def named_id(loc: Localiser, ident: str) -> str:
    """An id as a name, whichever kind of thing it is. Unknown ids come back unchanged.

    The resolver's events name moves, abilities, items, species and weather without
    saying which, so each is tried in turn. Returning the id untouched is the right
    failure: a log with `stealthrock` in it is readable, one with a wrong translation is
    worse than untranslated.
    """
    for lookup in (loc.move, loc.ability, loc.item, loc.species):
        named = lookup(ident)
        if named and named != ident:
            return named
    return ident


#: Reasons the resolver names that are not the id of any game object -- mechanics rather
#: than moves, items or abilities, so `named_id` cannot find them. Everything here was
#: taken from a tally over real games (25 games put `tox` at 115 occurrences and
#: `partiallytrapped` at 56), not from reading the resolver for things it might write:
#: a table built from the source grows entries nobody ever sees, and misses the ones that
#: matter.
MECHANICS = {
    "flinch": "ひるみ",
    "recoil": "反動",
    "partiallytrapped": "バインド",
    "fainted": "瀕死",
    "pinch berry": "HP減少",
    "berry": "きのみ",
    "counter": "カウント",
    "no target": "対象なし",
    "crit": "急所",
}

#: Phrases that need the words moved, not just replaced. Japanese puts the cause before
#: the verb, so `blocked by Baneful Bunker` cannot be done with a word-for-word table.
REORDERINGS = (
    (re.compile(r"blocked by (\S+)"), r"\1に防がれた"),
    (re.compile(r"absorbed by (\S+)"), r"\1に吸収された"),
    (re.compile(r"immune（([^）]*)）"), r"無効（\1）"),
    # Protect's success is gated on how many times in a row it has been used, and the
    # resolver writes the odds as "1 in 4". As a bare fragment it reads like a count.
    (re.compile(r"（(\d+) in (\d+)）"), r"（確率 \1/\2）"),
)

#: Whole-line replacements, longest first. A shorter phrase that is a substring of a
#: longer one would otherwise consume it.
PHRASES = (
    ("did not happen", "不発"),
    ("had no effect", "効果がなかった"),
    ("must switch out", "交代が必要"),
    # The id `recharge` translates on its own, so only the verb in front of it is left.
    ("must recharge", "反動で動けない"),
    ("stat drops undone", "能力低下を戻した"),
    ("cannot use sound moves", "音技が使えない"),
    ("must use", "この技しか出せない:"),
    ("retargeted to", "対象変更:"),
    ("redirected to", "対象そらし:"),
    ("did nothing, so nobody switched", "不発のため交代なし"),
    ("nothing left to act", "行動できる味方がいない"),
    ("woke up", "目を覚ました"),
    ("is charging", "溜め"),
    ("blocked the drop", "低下を防いだ"),
    ("absorbed", "吸収"),
    ("protected", "防いだ"),
    ("fainted", "瀕死"),
    ("missed", "外れ"),
    ("thawed", "解凍"),
    ("started", "開始"),
    ("ended", "終了"),
    ("failed", "失敗"),
    ("immune", "無効"),
    ("lost", "消費:"),
    ("side", "サイド"),
    ("set", "展開:"),
)


def translate_reason(loc: Localiser, text: str) -> str:
    """What was inside a pair of brackets: a mechanic, a game object, or a status.

    Tried in that order because the sets overlap in the wrong direction -- `psn` is a
    status and not an item, but a lookup that guesses will find *something* for almost
    any string, and a confidently wrong name is worse than an English one.
    """
    if text in MECHANICS:
        return MECHANICS[text]
    named = named_id(loc, text)
    if named != text:
        return named
    status = loc.status(text) if text in STATUSES else ""
    return status or text


def translate_event(loc: Localiser, line: str, occupants: dict[str, str]) -> str:
    """One resolver event, with the slot codes and ids turned into names.

    `occupants` is updated as the line is read, because a switch inside the turn changes
    who "p1a" refers to -- naming the Pokemon that started the turn in that slot would
    attribute the rest of the turn's damage to the wrong one.
    """
    words = line.split()
    if not words:
        return line
    code = words[0]
    rest = words[1:]
    if code[:2] in ("p1", "p2") and len(code) >= 3 and code[2] in "ab":
        index = (0 if code[1] == "1" else 2) + (0 if code[2] == "a" else 1)
        # "<-" is a switch, "->" a forme change; both replace who stands in the slot, so
        # both have to move the occupant on or the rest of the turn is misattributed.
        if rest[:1] in (["<-"], ["->"]) and len(rest) > 1:
            was = occupants.get(code, "")
            # "-> " is overloaded: a forme change names a species, a status change names
            # a status. Reading the second as the first renamed Garchomp to "tox" and
            # then attributed the rest of the turn to a Pokemon that does not exist.
            if rest[1] in STATUSES:
                caused = f"（{named_id(loc, rest[2].strip('()'))}）" if len(rest) > 2 else ""
                return (
                    f"{SLOT_LABELS[index]} {was} → {loc.status(rest[1])}{caused}"
                ).strip()
            occupants[code] = named_id(loc, rest[1])
            arrow = "←" if rest[0] == "<-" else "→"
            how = "交代で登場" if rest[0] == "<-" else "メガシンカ"
            return f"{SLOT_LABELS[index]} {was} {arrow} {occupants[code]}（{how}）".strip()
        who = occupants.get(code, "")
        head = f"{SLOT_LABELS[index]} {who}".strip()
        # `atk -1 -> -1` is "dropped by one, now at -1", not "from -1 to -1" -- the
        # resolver writes the delta first and the resulting stage second. Printed raw it
        # reads as a change that did not happen.
        if (
            len(rest) >= 4
            and rest[1][:1] in "+-"
            and rest[2] == "->"
            and rest[1][1:].isdigit()
        ):
            reason = rest[4].strip("()") if len(rest) > 4 else ""
            caused = f"（{named_id(loc, reason)}）" if reason else ""
            return (
                f"{head} {loc.stat(rest[0], short=True)} {rest[1]}段階"
                f" → 現在 {rest[3]}{caused}"
            )
    elif code in ("p1", "p2"):
        head = "自" if code == "p1" else "敵"
    else:
        head = code

    body = " ".join(rest)
    # A parenthetical is one reason, not a sequence of words, and it has to be translated
    # as one. Translating it token by token turned `(pinch berry)` into `(pinch きのみ`:
    # the closing bracket was attached to `berry`, and stripping it to look the word up
    # threw it away. So the group is matched whole, and only then looked up.
    body = re.sub(r"\(([^)]*)\)", lambda m: f"（{translate_reason(loc, m.group(1))}）", body)
    # English phrases before bare ids, not after. `p2b must recharge` is one line whose
    # last word is also a move id, so translating tokens first turned it into "must
    # 反動で動けない" and the phrase "must recharge" no longer matched anything. Longest
    # first, so "did not happen" is not found as "happen" once "did" and "not" are gone.
    for english, japanese in PHRASES:
        body = body.replace(english, japanese)
    # Ids also appear bare -- `set sunnyday`, `side +wideguard` -- where there is no group
    # to match, so those are still done token by token.
    pieces = []
    for word in body.split():
        if word.startswith("（"):
            pieces.append(word)
            continue
        sign = word[0] if word[:1] in "+-" else ""
        core = word.lstrip("+-")
        named = named_id(loc, core)
        pieces.append(f"{sign}{named}" if named != core else word)
    body = " ".join(pieces)
    for pattern, replacement in REORDERINGS:
        body = pattern.sub(replacement, body)
    return f"{head} {body}".strip().replace(" （", "（")


def _occupants(loc: Localiser, pos: object) -> dict[str, str]:
    """slot code -> who stands there at the start of the turn."""
    out: dict[str, str] = {}
    for side_index, side in enumerate(pos.sides):
        for slot, party in enumerate(side.active):
            if party is None:
                continue
            out[f"p{side_index + 1}{'ab'[slot]}"] = loc.species(
                side.pokemon[party].species
            )
    return out


def translate_act(
    loc: Localiser, label: str, occupants: dict[str, str], pos: object
) -> str:
    """One action's header: who acted, and what they did.

    `label` is `QueuedAction.label` -- `"p2a Flare Blitz"`, `"p1b switch->2"`,
    `"p1a mega"` -- or `RESIDUAL_PHASE` for the end of the turn, which is nobody's
    action. The move arrives as its English display name because that is what the
    resolver's own event lines carry, so it is turned back into an id the way Showdown
    would and looked up from there.

    `occupants` is read, not written: the caller keeps it current as the turn's lines go
    by, so an action that happens after a switch is attributed to whoever is standing in
    the slot by then rather than to whoever started the turn there.
    """
    if label == RESIDUAL_PHASE:
        return "ターン終了"
    words = label.split()
    code = words[0]
    if not (code[:2] in ("p1", "p2") and len(code) >= 3 and code[2] in "ab"):
        return label
    index = (0 if code[1] == "1" else 2) + (0 if code[2] == "a" else 1)
    who = f"{SLOT_LABELS[index]} {occupants.get(code, '')}".strip()
    rest = " ".join(words[1:])
    if rest.startswith("switch->"):
        # The label carries the party index; the position says which Pokemon that is.
        party = rest[len("switch->"):]
        side = pos.sides[0 if code[1] == "1" else 1]
        incoming = (
            loc.species(side.pokemon[int(party)].species)
            if party.isdigit() and int(party) < len(side.pokemon)
            else f"#{party}"
        )
        return f"{who}: 交代 → {incoming}"
    if rest == "mega":
        return f"{who}: メガシンカ"
    named = loc.move(to_id(rest))
    return f"{who}: {named if named else rest}"


def group_events(
    loc: Localiser, events: list[str], acts: list[tuple[int, str]], pos: object
) -> list[tuple[str | None, list[str]]]:
    """The flat trace, cut into one group per action, each headed by whose action it was.

    The cut points come from the resolver (`Branch.acts`) rather than from reading the
    lines, because the lines do not carry enough to recover them: a damage line names the
    move that caused it, not who used it, and in a mirror both sides use the same moves.

    A group with no header holds lines that precede the first action, which should not
    happen -- but printing them unattributed is the right failure, because a log that
    silently drops events is worse than one with an odd-looking first group.
    """
    occupants = _occupants(loc, pos)
    bounds = [start for start, _label in acts] + [len(events)]
    groups: list[tuple[str | None, list[str]]] = []
    if acts and acts[0][0] > 0:
        groups.append(
            (None, [translate_event(loc, line, occupants) for line in events[: acts[0][0]]])
        )
    elif not acts:
        return [(None, [translate_event(loc, line, occupants) for line in events])]
    for k, (start, label) in enumerate(acts):
        header = translate_act(loc, label, occupants, pos)
        words = label.split()
        move_id = (
            to_id(" ".join(words[1:]))
            if len(words) > 1 and not words[1].startswith(("switch->", "mega"))
            else ""
        )
        cause = f"({move_id})"
        lines = []
        for line in events[start : bounds[k + 1]]:
            # Two kinds of redundancy, both of which the header now covers.
            #
            # A line that repeats the action's own label -- "p1a Earth Power did not
            # happen (fainted)" under the Earth Power header -- keeps its slot code and
            # loses the move name.
            if line.startswith(label + " "):
                line = words[0] + line[len(label) :]
            # And a consequence attributed to the action's own move -- "-159
            # (flareblitz)" under Flare Blitz -- loses the attribution. Only when it is
            # the *same* move: Life Orb recoil and Leftovers name something else, and
            # that is the part a reader needs.
            elif move_id and line.endswith(cause):
                line = line[: -len(cause)].rstrip()
            lines.append(translate_event(loc, line, occupants))
        if lines:
            groups.append((header, lines))
    return groups


def _winner_side(pos: object) -> float | None:
    """A finished position's result in the same units as a record's `outcome`.

    1.0 when side 0 won, 0.0 when side 1 did, 0.5 for a mutual wipe-out. `None` when the
    battle has not ended, so a caller can tell "not decided" from "decided as a draw".
    """
    if not pos.ended:
        return None
    if pos.winner is None:
        return 0.5
    return 1.0 if pos.winner == pos.sides[0].id else 0.0


def nearest(candidates: list, recorded: dict):  # noqa: ANN001, ANN201
    """The candidate closest to a recorded position, and how far off it is.

    HP alone does not identify an outcome. A secondary effect that lands -- a burn, a
    Special Attack drop, Ancient Power raising five stats -- costs no HP on the turn it
    fires, so the outcome where it fired and the one where it did not are the same distance
    from the record. So the status, the boosts and the volatiles are part of the key. They
    are in the recorded position too, which is what makes this a matching problem rather
    than a missing-information one.

    Returns `(candidate, gap)`; a gap of `(0, 0)` means it *is* the recorded one. Used for
    both halves of a turn -- the branches it ends in and, on an interrupted turn, the
    suspensions it passes through -- because a `Branch` and a `SuspendedTurn` both carry
    the position the record can be held against.
    """

    def _recorded(mon: dict) -> tuple:
        return (
            mon.get("status"),
            tuple(sorted((mon.get("boosts") or {}).items())),
            tuple(sorted(v.get("id", "") for v in mon.get("volatiles") or [])),
        )

    def _live(mon) -> tuple:  # noqa: ANN001
        return (
            mon.status,
            tuple(sorted((mon.boosts or {}).items())),
            tuple(sorted(v.id for v in mon.volatiles)),
        )

    target = [mon["hp"] for side in recorded["sides"] for mon in side["pokemon"]]
    target_state = [_recorded(mon) for side in recorded["sides"] for mon in side["pokemon"]]

    def distance(candidate) -> tuple[int, int]:  # noqa: ANN001
        mons = [mon for side in candidate.position.sides for mon in side.pokemon]
        hp = sum(abs(mon.hp - want) for mon, want in zip(mons, target, strict=False))
        # Ordered after HP, not added to it: HP is the thing the record pins exactly, and a
        # state mismatch must never outrank a damage roll that is simply wrong.
        state = sum(
            _live(mon) != want for mon, want in zip(mons, target_state, strict=False)
        )
        return (hp, state)

    best = min(candidates, key=distance)
    return best, distance(best)


class _PortTurn:
    """The port's answer read as the `TurnResult` this module reads: `branches` and
    `suspended`, each with a `probability`, a `position`, `events` and `acts` (IKA-215)."""

    def __init__(self, port: object) -> None:
        self.branches = list(port.outcomes or [])
        self.suspended = list(port.pauses or [])


def turn_events(
    reg: Regulation,
    loc: Localiser,
    decision: dict,
    rest: list[dict],
    outcome: float | None = None,
    node: object | None = None,
) -> list[tuple[str | None, list[str]]]:
    """What actually happened, by re-resolving the turn that was played.

    The records keep the position and both chosen actions but not the resolver's event
    log, so it is recomputed here rather than guessed at from a diff of two positions: a
    diff says Charizard lost 137 HP, the log says which move did it and in what order.

    Which branch was played is recovered by matching the resulting HP against the next
    recorded position. Without a next position -- the last turn of the game -- the most
    likely branch is shown and said to be that.

    ``rest`` is every decision after this one, not just the next: an interrupted turn eats
    one of them per interrupt, and what the end of the turn has to be matched against is
    whatever is left.

    ``node`` is a `rustnode.RustNode`: the turn and its resumptions are then the port's,
    trace included (IKA-215), and everything else here reads them as it reads Python's.
    """
    from pokeuraou.narrow import narrow
    from pokeuraou.position import Position
    from pokeuraou.resolve import Budget, resolve_turn, resume_alternatives

    own = decision.get("ownChosen")
    foe = decision.get("foeChosen")
    if decision.get("kind") != "move" or not own or not foe:
        return []
    try:
        pos = Position.from_json(decision["position"])
    except Exception:  # noqa: BLE001 - a log must not die on one unreadable record
        return []
    # Every legal pair, not the menu the search would build. `narrow` ranks and truncates,
    # and the ranking is a *setting* -- generation ranks by the leaf, this has no model and
    # ranks by damage -- so rebuilding the menu here asks whether a different ranking would
    # have offered the same move, which is not the question. A recorded choice is legal by
    # construction; it just need not survive someone else's top 48. Doubles tops out near
    # 54 legal pairs, so this keeps them all.
    ALL_LEGAL = 512
    lookup = []
    for side, wanted in ((0, own), (1, foe)):
        actions = {a.to_choice(): a for a in narrow(reg, pos, side, limit=ALL_LEGAL).actions}
        if wanted not in actions:
            # Say so rather than print nothing. With the whole legal set in hand this now
            # means what it says: a rule genuinely changed since the record was written --
            # every pre-fix Hyper Beam recharge turn is one -- and a turn with an empty
            # "what happened" block reads as a turn where nothing happened.
            label = "自" if side == 0 else "敵"
            return [
                (
                    None,
                    [
                        f"⚠ 再現できません: 記録の{label}の手 {wanted!r} は、"
                        "現在の合法手生成では選べません（記録より後に直った規則があります）"
                    ],
                )
            ]
        lookup.append(actions[wanted])

    if node is None:
        result = resolve_turn(reg, pos, lookup, budget=Budget.exact())
    else:
        answered = node.turn(pos, lookup, Budget.exact(), full=True, events=True)
        if answered is None:
            return [(None, ["⚠ port が このターンを断りました"])]
        result = _PortTurn(answered)
    prefix: list[str] = []
    resumed = False
    notes: list[str] = []
    ahead = list(rest)
    # A self-switching move -- Parting Shot, U-turn, Volt Switch -- stops the turn for a
    # replacement choice, and that choice is the *next* recorded decision. A turn can do it
    # more than once, so this is a loop: two Parting Shots in one turn record two
    # `selfswitch` nodes, and resuming only the first left the second half of the turn
    # rendered from a position the game never stood in.
    while result.suspended:
        answered_by = ahead[0] if ahead else None
        # *Which* suspension. `Budget.exact()` enumerates the damage rolls, so an attack
        # that lands before the interrupt splits the pause into up to a thousand of them,
        # and index 0 is the one where every roll came out maximum. Taking it was the whole
        # of F1: the record showed Stomping Tantrum for 74 and the log printed 126, and
        # 126 is 74's roll block read off the top of the list -- not, as it looked, the two
        # halves of the project ordering a resumed turn differently. Measured on 200 games
        # of `selfplay-gen11L`: 105 interrupted turns, the recorded pause present among the
        # suspensions every time, and index 0 the wrong one in 66 of them.
        if answered_by is not None and answered_by.get("kind") == "selfswitch":
            paused, gap = nearest(list(result.suspended), answered_by["position"])
        else:
            paused, gap = max(result.suspended, key=lambda s: s.probability), (0, 0)
        prefix = list(paused.events)
        # Which side owes the replacement decides which half of the next record answers
        # it. Reading `ownChosen` either way silently lost every turn where the *opponent*
        # was the one switching out -- half of them.
        if node is None:
            side, alternatives = resume_alternatives(reg, paused)
        else:
            resumed_by_port = node.resume_alternatives(paused, events=True)
            if resumed_by_port is None:
                return [(None, ["⚠ port が 再開を断りました"])]
            side = resumed_by_port[0]
            alternatives = [(action, _PortTurn(r)) for action, r in resumed_by_port[1]]
        key = "foeChosen" if side == 1 else "ownChosen"
        answer = (answered_by or {}).get(key)
        picked = next(
            (r for action, r in alternatives if action.to_choice() == answer), None
        )
        if picked is None or not (picked.branches or picked.suspended):
            return [
                *group_events(loc, list(prefix), list(paused.acts), pos),
                (None, [f"（中断までの表示。記録の交代手 {answer!r} が再開手と一致しない）"]),
            ]
        if gap != (0, 0):
            prefix_note = (
                f"⚠ 中断時の局面が記録と一致しません（HP差 {gap[0]}、状態差 {gap[1]}）。"
                "以下は最も近い枝です"
            )
            notes.append(prefix_note)
        result = picked
        # The resumed result carries the *whole* turn, the part before the interrupt
        # included, so prepending what was collected before it prints everything twice.
        prefix = []
        # The position it lands in is not the one that decision recorded -- that one is
        # mid-turn, taken at the interrupt -- but the decision *after* it is the end of
        # this turn, and that is a match target. Treating "two nodes later" as "nothing to
        # match against" is what made this branch the likeliest rather than the played
        # one, and a likeliest branch is a different damage roll: one game showed Toxapex
        # fainting to Rough Skin on a turn it ended at 66 HP.
        ahead = ahead[1:]
        resumed = True
    following = ahead[0] if ahead else None
    if not result.branches:
        return []
    branch = max(result.branches, key=lambda b: b.probability)
    note = ""
    if following is not None:
        branch, gap = nearest(result.branches, following["position"])
        if gap != (0, 0):
            note = (
                "⚠ 以下は記録と一致しない枝です。実際に起きたことではありません"
                "（乱数の再現に失敗）"
            )
    elif resumed and len(result.branches) > 1:
        note = "以下は中断ターンを再開した最尤の枝です（照合先がない）"
    else:
        # The last turn has no next position to match against -- but the *outcome* is
        # recorded, and that is a match key too. Without it the log showed the likeliest
        # branch, which on a turn decided by a 1-in-3 Protect is the branch where nothing
        # happens: the game visibly stopped one turn short of ending, with the side that
        # lost still standing. The record said 11 turns and a loss; the log ended at 10
        # and looked like a bug in the engine.
        decided = [
            b
            for b in result.branches
            if b.position.ended and _winner_side(b.position) == outcome
        ]
        if outcome is not None and decided:
            branch = max(decided, key=lambda b: b.probability)
            note = (
                "以下は最終ターンです。記録の勝敗に一致する枝のうち最尤のものを表示"
                "（局面の照合先はない）"
            )
        elif outcome is not None and any(b.position.ended for b in result.branches):
            note = (
                "⚠ 最終ターン: 決着する枝はありますが、どれも記録の勝敗と一致しません"
            )
        elif len(result.branches) > 1:
            note = "以下は最終ターンの最尤の枝です（照合先がない）"

    groups = group_events(loc, [*prefix, *branch.events], list(branch.acts), pos)
    if note:
        notes.append(note)
    if notes:
        # In front of the events, not after them. A caveat about a list belongs before the
        # list: printed underneath, it was read past, and a poison tick from a branch that
        # was never played got reported as a missing-damage bug. The reader was right to
        # trust the lines -- the log was the thing that was wrong.
        groups.insert(0, (None, notes))
    return groups


def hidden_line(loc: Localiser, record: dict, seen: list[set[str]]) -> str:
    """What each side could not see, at the decision `seen` has been carried up to.

    The board printed above is the *record*, which is the truth: it names the opponent's
    whole four from turn one. The search was not shown that, and without a line saying so
    a reader compares a mixture against information the mixture was never allowed. So this
    prints, per side, the sheet members that side had not yet met and how many fours they
    made -- C(4,2) = 6 at the opening, 3 once one more has come in, and nothing once the
    four is out.

    `seen` is accumulated by the caller across decisions, because a Pokemon that came in
    and went back out is still known and the position alone stops saying so. What counts
    as shown comes from `hidden.seen_slots`, the rule the search itself used -- a display
    that decided for itself would drift from the numbers it is annotating, and the drift
    would read as the search having been more or less certain than it was.
    """
    sixes = (record.get("ownSix") or [], record.get("foeSix") or [])
    parts: list[str] = []
    for index, label in ((0, "自"), (1, "敵")):
        sheet = [str(name) for name in sixes[index]]
        if len(sheet) < 6:
            continue
        unseen = [name for name in sheet if name not in seen[index]]
        # They brought four, so the slots still to come are four minus what they have
        # shown. The rest of the sheet they did not bring, and that is the whole of the
        # uncertainty: which of the unmet ones are in those slots.
        slots = 4 - len([name for name in sheet if name in seen[index]])
        if slots <= 0:
            parts.append(f"{label}の裏: なし（4体とも出た）")
            continue
        if len(unseen) < slots:
            # The sheet cannot explain the board, which `hidden.completions` refuses
            # outright. Said plainly rather than rounded down to "nothing hidden", which
            # is the one reading that would look normal.
            parts.append(f"{label}の裏: 不整合（{slots}枠に対し未見 {len(unseen)}体）")
            continue
        names = "・".join(loc.species(name) for name in unseen)
        parts.append(
            f"{label}の裏: {slots}枠を {names} から → {comb(len(unseen), slots)}通り"
        )
    return "  ｜  ".join(parts)


def render(
    reg: Regulation, loc: Localiser, record: dict, top: int, node: object | None = None
) -> str:
    out = io.StringIO()
    own = "・".join(loc.species(s["species"]) for s in record["ownTeam"])
    foe = "・".join(loc.species(s["species"]) for s in record["foeTeam"])
    outcome = record.get("outcome")
    verdict = "勝ち" if outcome == 1.0 else "負け" if outcome == 0.0 else "打ち切り"
    out.write(f"自陣: {own}\n相手: {foe}  [{record.get('foeArchetype', '?')}]\n")
    if record.get("information") == "hidden-bench":
        out.write(
            "不完全情報: どちらの探索も相手の未公開の控えを見ていない。"
            "下の盤面は記録された真実で、混合はそれを見ずに解かれたもの。"
            + "\n"
        )
    # A match game's `foeArchetype` is the seat label the tool played under -- something
    # like "value-gen234.pt = side 0" -- which on the 相手 line reads as if that model
    # were the opponent, when it is whichever side the label says. The provenance block
    # is the authority and names both sides, so print it whenever it is there.
    source = record.get("provenance")
    if source:
        leaves = source.get("leaves") or []
        limits = source.get("limits") or []
        if len(leaves) == 2:
            out.write(
                f"対戦: {source.get('kind', '?')}  自(side 0) の葉 {leaves[0]}"
                f" ／ 敵(side 1) の葉 {leaves[1]}"
            )
            if len(limits) == 2:
                out.write(f"  探索幅 {limits[0]}/{limits[1]}")
            if source.get("note"):
                out.write(f"  {source['note']}")
            out.write("\n")
    # The selection, when the record carries it. Ordered, so the first two are the leads --
    # a different decision from the two behind them, and the reason there are 90 selections
    # a side rather than 15.
    if record.get("selectionSource") == "book":
        value = record.get("selectionValue")
        shown = f"均衡値 {value * 100:.1f}%、" if value is not None else ""
        out.write(
            f"選出は選出解から（{shown}"
            "実際に引いたのは均衡＋探索の混合なので下の頻度とずれることがある）\n"
        )
    for label, six_key, pick_key, policy_key in (
        ("自", "ownSix", "ownPick", "ownSelectionPolicy"),
        ("相手", "foeSix", "foePick", "foeSelectionPolicy"),
    ):
        six = record.get(six_key) or []
        pick = record.get(pick_key) or []
        if not six or not pick:
            continue
        names = [loc.species(entry) for entry in six]
        brought = [names[i] for i in pick]
        left = [n for i, n in enumerate(names) if i not in pick]
        out.write(
            f"{label}の選出: 先発 {' + '.join(brought[:2])}"
            f" ／ 裏 {' + '.join(brought[2:])}"
            f"   （不選出: {'・'.join(left)}）\n"
        )
        # The mixture it was drawn from, when the game came from a selection book. One
        # selection with a frequency is a recommendation; the whole mixture is the answer,
        # which is the shape this tool is supposed to print everywhere else too.
        policy = record.get(policy_key) or []
        if not policy:
            continue
        selections = all_selections(len(six), len(pick))
        # What was actually drawn, and how likely that was. Generation draws from the
        # equilibrium *softened* by exploration, so a pure equilibrium still misses a
        # quarter of the time -- and a line reading "100.0%" above a different selection
        # is a contradiction to anyone who has not memorised that. Saying the drawn
        # probability here puts the explanation where the confusion is.
        mixture = record.get(policy_key.replace("Policy", "Mixture")) or []
        drawn = next(
            (i for i, s in enumerate(selections) if list(s) == list(pick)), None
        )
        if drawn is not None and policy[drawn] <= 1e-6:
            odds = f"{mixture[drawn] * 100:.2f}%" if drawn < len(mixture) else "?"
            out.write(
                f"   ※ 実際に引いたのは均衡の外（探索）: この選出の確率 {odds}\n"
            )
        order = sorted(range(len(policy)), key=lambda i: -policy[i])
        for rank, index in enumerate(order):
            if rank >= top or policy[index] <= 1e-6:
                break
            selection = selections[index]
            chosen = "☆" if list(selection) == list(pick) else "　"
            out.write(
                f"   {chosen}{policy[index] * 100:5.1f}%  "
                f"{' + '.join(names[i] for i in selection[:2])}"
                f" ／ {' + '.join(names[i] for i in selection[2:])}\n"
            )
    out.write(
        f"結果: {verdict}   {record.get('turns', '?')} ターン   "
        f"探索 {record.get('searchLimit', '?')}x / 葉 {record.get('searchObjective', '?')}\n"
    )

    decisions = record["decisions"]
    # Carried across decisions: what each side has shown by now.
    seen_species: list[set[str]] = [set(), set()]
    for position_in_game, decision in enumerate(decisions):
        sides = decision["position"]["sides"]
        # Parsed once: the legality rule wants Pokemon objects, and rebuilding one per
        # named action would parse the same position a few dozen times a turn.
        parsed = Position.from_json(decision["position"])
        out.write(f"\n─── ターン {decision['turn']}  ({decision['kind']})\n")
        conditions = field_line(loc, decision["position"])
        if conditions:
            out.write(f"  場: {conditions}\n")
        for index, side in enumerate(sides):
            label = "自" if index == 0 else "敵"
            live = []
            for party in side["active"]:
                if party is None:
                    live.append("（空）")
                    continue
                mon = side["pokemon"][party]
                status = f" {mon['status']}" if mon.get("status") else ""
                boosts = (
                    " " + ",".join(f"{k}{v:+d}" for k, v in (mon.get("boosts") or {}).items())
                    if mon.get("boosts")
                    else ""
                )
                live.append(
                    f"{loc.species(mon['species'])} "
                    f"[{hp_bar(mon['hp'], mon['maxhp'])}] "
                    f"{mon['hp']}/{mon['maxhp']}{status}{boosts}"
                )
            bench = [
                loc.species(m["species"])
                for i, m in enumerate(side["pokemon"])
                if i not in [p for p in side["active"] if p is not None] and not m["fainted"]
            ]
            out.write(f"  {label}: {'  '.join(live)}")
            if bench:
                out.write(f"   控え: {'・'.join(bench)}")
            out.write("\n")

        if record.get("information") == "hidden-bench":
            # `seen_slots` is the rule the search used, so the annotation and the mixture
            # it annotates agree by construction. Deciding here that "active or fainted"
            # meant shown would have under-counted every Pokemon that had been damaged,
            # statused or Mega Evolved and put back.
            for index, side in enumerate(sides):
                for slot in seen_slots(parsed, index):
                    seen_species[index].add(str(side["pokemon"][slot]["species"]))
            line = hidden_line(loc, record, seen_species)
            if line:
                out.write(f"  {line}" + "\n")
        out.write(f"  探索値（自分の勝率）: {decision['searchValue']:.3f}\n")
        for index, (actions, policy) in enumerate(
            (
                (decision["ownActions"], decision["ownPolicy"]),
                (decision["foeActions"], decision["foePolicy"]),
            )
        ):
            label = "自" if index == 0 else "敵"
            ranked = sorted(
                zip(actions, policy, strict=True), key=lambda pair: -pair[1]
            )
            shown = [(a, w) for a, w in ranked if w > 0.001][:top]
            if not shown:
                continue
            # Which action was actually drawn, when the record carries it. The mixture is
            # what the search computed; this is one draw from it, and marking it is the
            # difference between reading a distribution and reading a game.
            played = decision.get("ownChosen" if index == 0 else "foeChosen")
            out.write(f"  {label}の均衡混合:\n")
            for choice, weight in shown:
                # The star sits in front of the frequency so the eye finds the played row
                # while scanning the column of numbers, not at the end of a line whose
                # length depends on how many Pokemon are named in it.
                mark = "☆" if played is not None and choice == played else "　"
                out.write(
                    f"   {mark}{weight * 100:>5.1f}%  "
                    f"{name_action(reg, loc, sides, index, choice, parsed)}\n"
                )
            rest = sum(w for _a, w in ranked[len(shown):])
            if rest > 0.001:
                out.write(f"    {rest * 100:>5.1f}%  （残り {len(ranked) - len(shown)} 手）\n")
            # A draw from outside the shown rows would otherwise vanish from the log.
            if played is not None and all(choice != played for choice, _w in shown):
                out.write(
                    f"   ☆  ---  {name_action(reg, loc, sides, index, played, parsed)}"
                    "（表示範囲の外から引かれた手）\n"
                )

        # What the chosen pair actually did, recomputed from the position: the records
        # keep the mixtures and the draw but not the resolver's log, and a mixture
        # without its consequence is only half of what a reader needs.
        happened = turn_events(
            reg,
            loc,
            decision,
            decisions[position_in_game + 1 :],
            record.get("outcome"),
            node,
        )
        if happened:
            out.write("  起きたこと:\n")
            for header, lines in happened:
                # No header means the lines belong to no single action -- a note about
                # which branch was shown, or a trace that began before the first action.
                if header:
                    out.write(f"    {header}\n")
                indent = "      " if header else "    "
                for line in lines:
                    out.write(f"{indent}・{line}\n")
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/selfplay-gen2"))
    ap.add_argument("--game", type=int, default=0, help="index within the first file")
    ap.add_argument("--file", type=Path, default=None, help="a specific jsonl file")
    ap.add_argument("--top", type=int, default=5, help="how many actions of the mixture")
    ap.add_argument("--out", type=Path, default=Path("game.txt"))
    ap.add_argument("--locale", default="ja")
    ap.add_argument(
        "--engine", choices=("python", "port"), default="python",
        help="who re-resolves the turns for \"起きたこと\" (port: the Rust node, IKA-215)",
    )
    args = ap.parse_args()

    path = args.file or sorted(args.dir.glob("*.jsonl"))[0]
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == args.game:
                record = json.loads(line)
                break
        else:
            raise SystemExit(f"{path} has no game at index {args.game}")

    reg = load_regulation(record["decisions"][0]["position"]["format"])
    loc = Localiser(reg, load_names(args.locale))
    node = None
    if args.engine == "port":
        from pokeuraou import rustnode
        from pokeuraou.damage import register_mega_stones

        register_mega_stones(reg)
        node = rustnode.RustNode(reg)
    text = render(reg, loc, record, args.top, node)
    args.out.write_text(text, encoding="utf-8")
    print(f"{path.name} game {args.game} -> {args.out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
