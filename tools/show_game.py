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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.names import Localiser, load_names
from pokeuraou.regulation import Regulation, load_regulation
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


def name_action(
    reg: Regulation, loc: Localiser, sides: list[dict], actor: int, choice: str
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
            move_id = (
                mon["moves"][index]["id"]
                if mon and 0 <= index < len(mon["moves"])
                else f"?{tokens[1]}"
            )
            text = f"{who}: {loc.move(move_id)}"
            if len(tokens) > 2 and tokens[2] not in ("mega",):
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
    elif code in ("p1", "p2"):
        head = "自" if code == "p1" else "敵"
    else:
        head = code

    # Ids appear bare (`set sunnyday`, `side +wideguard`) or in parentheses (the move or
    # item that caused the line). Translating token by token covers both without needing
    # to know the grammar of every event the resolver writes.
    pieces = []
    for word in rest:
        stripped = word.strip("()")
        prefix = word[: len(word) - len(word.lstrip("(+-"))]
        sign = prefix if prefix in ("+", "-") else ""
        core = stripped.lstrip("+-")
        named = named_id(loc, core)
        if named != core:
            pieces.append(
                f"（{named}）" if word.startswith("(") else f"{sign}{named}"
            )
        else:
            pieces.append(word)
    body = " ".join(pieces)
    for english, japanese in (
        ("fainted", "瀕死"),
        ("missed", "外れ"),
        ("side", "サイド"),
        ("set", "展開:"),
        ("protected", "で防いだ"),
        ("did not happen", "不発"),
    ):
        body = body.replace(english, japanese)
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


def turn_events(
    reg: Regulation, loc: Localiser, decision: dict, following: dict | None
) -> list[str]:
    """What actually happened, by re-resolving the turn that was played.

    The records keep the position and both chosen actions but not the resolver's event
    log, so it is recomputed here rather than guessed at from a diff of two positions: a
    diff says Charizard lost 137 HP, the log says which move did it and in what order.

    Which branch was played is recovered by matching the resulting HP against the next
    recorded position. Without a next position -- the last turn of the game -- the most
    likely branch is shown and said to be that.
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
    limit = len(decision["ownActions"])
    lookup = []
    for side, wanted in ((0, own), (1, foe)):
        actions = {a.to_choice(): a for a in narrow(reg, pos, side, limit=max(limit, 24)).actions}
        if wanted not in actions:
            return []
        lookup.append(actions[wanted])

    result = resolve_turn(reg, pos, lookup, budget=Budget.exact())
    prefix: list[str] = []
    if result.suspended:
        # A self-switching move -- Parting Shot, U-turn, Volt Switch -- stops the turn for
        # a replacement choice, and that choice is the *next* recorded decision. Bailing
        # out here left the turn with no explanation at all while the position visibly
        # changed, which is the one thing a log must not do.
        paused = result.suspended[0]
        prefix = list(paused.events)
        answer = (following or {}).get("ownChosen") or (following or {}).get("foeChosen")
        side, alternatives = resume_alternatives(reg, paused)
        picked = next(
            (r for action, r in alternatives if action.to_choice() == answer), None
        )
        if picked is None or not picked.branches:
            return [
                *[translate_event(loc, line, _occupants(loc, pos)) for line in prefix],
                "（すてゼリフ等でターンが中断。以降は次のノード）",
            ]
        result = picked
        # The resumed result carries the *whole* turn, the part before the interrupt
        # included, so prepending what was collected before it prints everything twice.
        prefix = []
        # And the position it lands in is not the one the next decision recorded -- that
        # is two nodes later -- so there is nothing to match a branch against.
        following = None
    if not result.branches:
        return []
    branch = max(result.branches, key=lambda b: b.probability)
    note = ""
    if following is not None:
        target = [
            mon["hp"]
            for side in following["position"]["sides"]
            for mon in side["pokemon"]
        ]

        def distance(candidate) -> int:  # noqa: ANN001
            got = [
                mon.hp for side in candidate.position.sides for mon in side.pokemon
            ]
            return sum(abs(a - b) for a, b in zip(got, target, strict=False))

        branch = min(result.branches, key=distance)
        if distance(branch) != 0:
            note = "（記録と完全には一致しない枝: 乱数の再現に失敗している）"
    elif len(result.branches) > 1:
        note = "（最終ターンなので最尤の枝）"

    occupants = _occupants(loc, pos)
    lines = [
        translate_event(loc, line, occupants) for line in (*prefix, *branch.events)
    ]
    if note:
        lines.append(note)
    return lines


def render(reg: Regulation, loc: Localiser, record: dict, top: int) -> str:
    out = io.StringIO()
    own = "・".join(loc.species(s["species"]) for s in record["ownTeam"])
    foe = "・".join(loc.species(s["species"]) for s in record["foeTeam"])
    outcome = record.get("outcome")
    verdict = "勝ち" if outcome == 1.0 else "負け" if outcome == 0.0 else "打ち切り"
    out.write(f"自陣: {own}\n相手: {foe}  [{record.get('foeArchetype', '?')}]\n")
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
    for position_in_game, decision in enumerate(decisions):
        sides = decision["position"]["sides"]
        out.write(f"\n─── ターン {decision['turn']}  ({decision['kind']})\n")
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
                    f"{name_action(reg, loc, sides, index, choice)}\n"
                )
            rest = sum(w for _a, w in ranked[len(shown):])
            if rest > 0.001:
                out.write(f"    {rest * 100:>5.1f}%  （残り {len(ranked) - len(shown)} 手）\n")
            # A draw from outside the shown rows would otherwise vanish from the log.
            if played is not None and all(choice != played for choice, _w in shown):
                out.write(
                    f"   ☆  ---  {name_action(reg, loc, sides, index, played)}"
                    "（表示範囲の外から引かれた手）\n"
                )

        # What the chosen pair actually did, recomputed from the position: the records
        # keep the mixtures and the draw but not the resolver's log, and a mixture
        # without its consequence is only half of what a reader needs.
        following = (
            decisions[position_in_game + 1]
            if position_in_game + 1 < len(decisions)
            else None
        )
        happened = turn_events(reg, loc, decision, following)
        if happened:
            out.write("  起きたこと:\n")
            for line in happened:
                out.write(f"    ・{line}\n")
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/selfplay-gen2"))
    ap.add_argument("--game", type=int, default=0, help="index within the first file")
    ap.add_argument("--file", type=Path, default=None, help="a specific jsonl file")
    ap.add_argument("--top", type=int, default=5, help="how many actions of the mixture")
    ap.add_argument("--out", type=Path, default=Path("game.txt"))
    ap.add_argument("--locale", default="ja")
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
    text = render(reg, loc, record, args.top)
    args.out.write_text(text, encoding="utf-8")
    print(f"{path.name} game {args.game} -> {args.out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
