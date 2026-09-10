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


def hp_bar(current: int, maximum: int, width: int = 10) -> str:
    if maximum <= 0:
        return " " * width
    filled = round(current / maximum * width)
    return "#" * filled + "." * (width - filled)


def name_action(
    reg: Regulation, loc: Localiser, side: dict, choice: str
) -> str:
    """Turns one side's choice string into names, using the position it was offered in."""
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
                text += f" → {'相手' if target > 0 else '味方'}{abs(target)}"
            if tokens[-1] == "mega":
                text += " + メガ"
            parts.append(text)
        else:
            parts.append(f"{who}: {piece}")
    return " ｜ ".join(parts)


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
    for label, six_key, pick_key in (
        ("自", "ownSix", "ownPick"),
        ("相手", "foeSix", "foePick"),
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
    out.write(
        f"結果: {verdict}   {record.get('turns', '?')} ターン   "
        f"探索 {record.get('searchLimit', '?')}x / 葉 {record.get('searchObjective', '?')}\n"
    )

    for decision in record["decisions"]:
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
                mark = " ←選択" if played is not None and choice == played else ""
                out.write(
                    f"    {weight * 100:>5.1f}%  "
                    f"{name_action(reg, loc, sides[index], choice)}{mark}\n"
                )
            rest = sum(w for _a, w in ranked[len(shown):])
            if rest > 0.001:
                out.write(f"    {rest * 100:>5.1f}%  （残り {len(ranked) - len(shown)} 手）\n")
            # A draw from outside the shown rows would otherwise vanish from the log.
            if played is not None and all(choice != played for choice, _w in shown):
                out.write(f"    ←選択  {name_action(reg, loc, sides[index], played)}\n")
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
