# ruff: noqa: E501  -- the page's CSS and HTML are strings, written one rule / one element to a line.
"""One game of a time match as a single static HTML page, in Japanese (`show_game.py --transcript`).

The input is a line of ``transcripts-worker<k>.jsonl`` (`time_match --transcript`,
`timematch.transcript_of`): per decision the position it was made in, both sides' choices, the
read's value, and for a move decision the port's account of the outcome the game drew. The
page is one long column -- nothing opens or closes -- with, from the top: the result, the win
rate turn by turn, the two teams, one card a turn (the field, the two choices, what happened,
the two reads), and a table of the win rates.

**Whose value.** Every win rate on the page is *side 0's*: the game's value (`value0`) is in
side 0's units. A read of side 1 is shown as it sees it (1 - value0) and is said so in its own
words. Side 0 is orange and above, side 1 is blue and below, everywhere.

**What is not made up.** The events are the port's trace of the outcome drawn, turned into
Japanese by `show_game`'s translators; a turn that paused for a switch in the middle has the
trace up to the pause, and the page says so. A read a seat did not write (a replacement, a
switch in the middle of a turn) is a dash. Names come from `configs/names/ja.json`
(`pokeuraou.names`); a name it has not got stays English, as the localiser does.

**A generation game** (`selfplay_line`, `show_game.py --file ... --html`) is turned into the same
line from its record: what happened is the port's re-resolution of each turn (`show_game.turn_trace`,
the branch matched to the next recorded position, as the text is), the two reads are the record's
two seats' own values (``searchValue``, ``foeSearchValue``) and mixtures. The record has no ladder
stage, clock, per-move values or the model each seat held of the other's mixture, no condition
name, seed or pair: the page says "生成の局" / "—" for them (``generated`` on the line and the
reads) and does not show a table it has no rows for.
"""

from __future__ import annotations

import base64
import contextlib
import html
import re
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import show_game

from pokeuraou import narrow, timematch
from pokeuraou.actions import side_actions
from pokeuraou.hpdisplay import displayed_percent, uses_floor_display
from pokeuraou.humanplay import sprite_id
from pokeuraou.names import Localiser
from pokeuraou.position import Position
from pokeuraou.regulation import Regulation, to_id

SEAT = ("席 0", "席 1")
#: Events a card lists before it says how many more there were.
RESIDUAL_MAX = 6
#: Two reads of a turn this far apart (points) are shown as split; a move of the win rate this
#: far (points) between turns is marked on the chart.
SPLIT_PT = 8
MARK_PT = 5
#: A move is one the mixture plays (in its support) from this probability up.
SUPPORT_MIN = timematch.SUPPORT_MIN
SPRITES = "https://play.pokemonshowdown.com/sprites/gen5/{}.png"
END_REASONS = {
    "wipeout": "全滅で決着",
    "draw": "両者全滅で引き分け",
    "turn-cap": "ターン上限で打ち切り",
    "unresolved": "解決できないターンがあり打ち切り",
    "adjudicated": "勝率が決まり切ったと見て判定で打ち切り",
}
MINUS = "−"


# ----------------------------------------------------------------------------- small things


def signed(value: float, unit: str = "") -> str:
    """+3 / −22 (a true minus sign, so a column of them lines up)."""
    text = f"{abs(value):.0f}"
    return f"{'+' if value >= 0 else MINUS}{text}{unit}"


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def name_html(name: str) -> str:
    """A species name, its forme (what the localiser leaves in English in brackets) in small
    type on its own line, so a long one does not get cut."""
    base, sep, forme = name.partition(" (")
    if not sep:
        return esc(name)
    return f'{esc(base)}<small class="fm">({esc(forme.rstrip(")"))})</small>'


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


@contextlib.contextmanager
def seat_labels() -> Iterator[None]:
    """`show_game`'s translators name the sides 自/敵; here they are 席 0 and 席 1."""
    saved = (show_game.SLOT_LABELS, show_game.SIDE_LABELS)
    show_game.SLOT_LABELS = ("席 0", "席 0", "席 1", "席 1")
    show_game.SIDE_LABELS = SEAT
    try:
        yield
    finally:
        show_game.SLOT_LABELS, show_game.SIDE_LABELS = saved


class Sprites:
    """Where an icon comes from: Showdown's gen-5 sprite of the species, the base species when
    the forme has none, and a letter when neither loads (the page's ``onerror``). With
    ``embed`` the images are fetched now and written into the page as data URIs."""

    def __init__(self, reg: Regulation, embed: bool = False, open_mix: bool = False) -> None:
        self.reg = reg
        self.embed = embed
        #: Whether the page is written with every turn's mixtures open (for a screenshot).
        self.open_mix = open_mix
        self.cache: dict[str, str | None] = {}
        self.missing: list[str] = []

    def ids(self, species: str) -> tuple[str, str]:
        full = sprite_id(self.reg, species)
        return full, full.split("-")[0]

    def _fetch(self, sid: str) -> str | None:
        if sid not in self.cache:
            data = None
            try:
                with urllib.request.urlopen(SPRITES.format(sid), timeout=15) as got:  # noqa: S310
                    data = got.read()
            except OSError:
                data = None
            self.cache[sid] = (
                None if not data else "data:image/png;base64," + base64.b64encode(data).decode("ascii")
            )
        return self.cache[sid]

    def img(self, species: str, name: str, cls: str = "ic") -> str:
        full, base = self.ids(species)
        alt = SPRITES.format(base)
        src = SPRITES.format(full)
        if self.embed:
            got = self._fetch(full)
            if got is None and base != full:
                got = self._fetch(base)
            if got is None:
                self.missing.append(species)
                return f'<span class="ph {cls}" title="{esc(name)}">{esc(name[:1])}</span>'
            src, alt = got, ""
        return (
            f'<img class="{cls}" src="{esc(src)}" data-alt="{esc(alt)}" alt="{esc(name[:1])}" '
            f'title="{esc(name)}" loading="lazy" onerror="var t=this;if(t.dataset.alt&&!t.dataset.tried)'
            "{t.dataset.tried=1;t.src=t.dataset.alt}else{var s=document.createElement('span');"
            "s.className='ph '+t.className;s.title=t.title;s.textContent=t.alt;t.replaceWith(s)}\">"
        )


# ----------------------------------------------------------------------------- reads


def stage_depth(stage: str | None) -> int | None:
    """The depth a ladder stage reads (``d2r12b3k16`` is depth 2), or None."""
    got = re.match(r"^d(\d+)", stage or "")
    return int(got.group(1)) if got else None


def read_of(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """One seat's read of a move decision from its clock row: the value (side 0's units), the
    stage that gave the answer, the depth, the seconds."""
    if row is None or "value0" not in row:
        return None
    stage = None
    #: A read with no stage finished (the ladder's first did not fit the budget) or no deepening
    #: is the depth-1 answer: the root's matrix solved (a row with no ``value0`` has no read at all).
    depth: int | None = None if row.get("generated") else 1
    ladder = row.get("ladder")
    if ladder and ladder.get("rungs"):
        stage = ladder["rungs"][-1].get("stage")
        depth = stage_depth(stage)
    elif isinstance(row.get("deepened"), dict):
        depth = row["deepened"].get("depth", 1)
    return {
        "value0": float(row["value0"]),
        "stage": stage,
        "depth": depth,
        "condition": row.get("condition"),
        "seconds": row.get("seconds"),
        "budget": row.get("budget"),
    }


# ----------------------------------------------------------------------------- the board


def field_texts(loc: Localiser, position: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """The weather, terrain and rooms (with what is left of each), then each side's screens."""
    field = position.get("field") or {}
    middle: list[str] = []
    for key, duration_key in (("weather", "weatherDuration"), ("terrain", "terrainDuration")):
        if field.get(key):
            middle.append(
                show_game._effect_text(loc, {"id": field[key], "duration": field.get(duration_key)})
            )
    middle.extend(show_game._effect_text(loc, e) for e in field.get("pseudoWeather") or [])
    sides = []
    for side in position.get("sides") or []:
        own = [show_game._effect_text(loc, e) for e in side.get("sideConditions") or []]
        for slot, group in enumerate(side.get("slotConditions") or []):
            own.extend(f"{slot + 1}体目:{show_game._effect_text(loc, e)}" for e in group)
        sides.append(own)
    return middle, sides


def mon_view(reg: Regulation, loc: Localiser, mon: Any, after: Any | None) -> dict[str, Any]:  # noqa: ANN401
    floor = uses_floor_display(reg)
    pct = displayed_percent(mon.hp, mon.maxhp, floor_rule=floor)
    delta = None
    left = False
    if after is not None:
        if after.fainted and not mon.fainted:
            delta = -pct
        else:
            delta = displayed_percent(after.hp, after.maxhp, floor_rule=floor) - pct
        left = after.active_index is None and not after.fainted
    chips = []
    if mon.status and mon.status != "fnt":
        chips.append(loc.status(mon.status))
    chips += [
        f"{loc.stat(k, short=True)}{'+' if v > 0 else MINUS}{abs(v)}" for k, v in mon.boosts.items() if v
    ]
    now = None
    if after is not None and after.status and after.status != mon.status and after.status != "fnt":
        now = loc.status(after.status)
    return {
        "species": mon.species,
        "name": loc.species(mon.species),
        "pct": pct,
        "hp": [mon.hp, mon.maxhp],
        "fainted": bool(mon.fainted),
        "chips": chips,
        "delta": delta,
        "newStatus": now,
        "left": left,
        "faintedNow": bool(after is not None and after.fainted and not mon.fainted),
    }


def board(
    reg: Regulation,
    loc: Localiser,
    pos: Position,
    after: Position | None,
    shown: list[list[str]] | None,
) -> list[dict[str, Any]]:
    """Per side: the two active slots (None: empty) and the bench (fainted ones marked)."""
    from pokeuraou.hidden import identity

    out = []
    for index, side in enumerate(pos.sides):
        old = {identity(m): m for m in after.sides[index].pokemon} if after is not None else {}
        active = []
        for mon in side.active_pokemon():
            active.append(None if mon is None else mon_view(reg, loc, mon, old.get(identity(mon))))
        bench = [
            {
                "species": mon.species,
                "name": loc.species(mon.species),
                "fainted": bool(mon.fainted),
                "pct": displayed_percent(mon.hp, mon.maxhp, floor_rule=uses_floor_display(reg)),
            }
            for mon in side.pokemon
            if mon.active_index is None
        ]
        out.append({"active": active, "bench": bench})
    del shown  # what each side had seen is not this page's subject: a record shows the whole game
    return out


# ----------------------------------------------------------------------------- the choices


#: What a field move that is sure to fail is called (`narrow.dead_field_move`, IKA-395): the
#: candidate sets are unchanged, the page only says so.
DEAD_TEXT = {
    narrow.DEAD: "失敗が決まっている",
    narrow.DEAD_BUT_DODGES_SUCKER_PUNCH: "失敗するがふいうちは外す",
    narrow.DEAD_BUT_FEEDS_STOMPING_TANTRUM: "失敗するがじだんだを強める",
    narrow.DEAD_BUT_CHANGEABLE: "先に場が消えれば成功",
    narrow.DEAD_IF_FIRST_ACTS: "味方が先に張れば失敗",
}
_ACTIONS: dict[tuple[int, int], dict[str, Any]] = {}


def dead_labels(reg: Regulation, position: dict[str, Any], side: int, choice: str) -> list[str | None]:
    """Per slot of a choice: the words for a field move that fails, or None. The legal actions of
    the position are listed once (the choice string finds its `SideAction`)."""
    key = (id(position), side)
    if key not in _ACTIONS:
        parsed = Position.from_json(position)
        _ACTIONS[key] = {
            "pos": parsed,
            "by": {a.to_choice(): a for a in side_actions(reg, parsed, side)},
        }
    got = _ACTIONS[key]
    action = got["by"].get(choice)
    if action is None:
        return []
    try:
        verdicts = narrow.dead_field_moves(reg, got["pos"], side, action)
    except Exception:  # noqa: BLE001 - a note must not stop a page
        return []
    return [DEAD_TEXT.get(v) if v else None for v in verdicts]


def hands(
    reg: Regulation, loc: Localiser, position: dict[str, Any], choices: list[str | None]
) -> list[list[dict[str, Any] | None]]:
    """Each side's choice as the words for each active slot (None: nothing to do there), from the
    position it was made in, and for a field move that fails, why (`dead_labels`)."""
    parsed = Position.from_json(position)
    out: list[list[dict[str, Any] | None]] = []
    for side, choice in enumerate(choices):
        if not choice:
            out.append([])
            continue
        text = show_game.name_action(reg, loc, position["sides"], side, choice, parsed)
        dead = dead_labels(reg, position, side, choice)
        row: list[dict[str, Any] | None] = []
        for slot, part in enumerate(text.split(" ｜ ")):
            _who, _, what = part.partition(": ")
            what = what.replace("→ 敵", "→ 相手の").replace("→ 味方", "→ 味方の")
            row.append(
                None
                if what == "行動なし"
                else {"text": what, "dead": dead[slot] if slot < len(dead) else None}
            )
        out.append(row)
    return out


# ----------------------------------------------------------------------------- what happened

_HP_LINE = re.compile(r"^(p[12][ab]) ([+-])(\d+) \((.+)\)$")
_STATUS_LINE = re.compile(r"^(p[12][ab]) -> (brn|psn|tox|par|slp|frz)(?: \((.+)\))?$")
_BOOST_LINE = re.compile(r"^(p[12][ab]) (\w+) ([+-]\d+) -> ([+-]?\d+)(?: \((.+)\))?$")
_SWITCH_IN = re.compile(r"^(p[12][ab]) <- (\S+)$")
_FORME = re.compile(r"^(p[12][ab]) -> (\S+)$")
_BLOCKED = re.compile(r"^(p[12][ab]) (.+) blocked by (\S+)$")
_DID_NOT = re.compile(r"^(p[12][ab]) (.+) (did not happen|failed) \((.+)\)$")
_MISSED = re.compile(r"^(p[12][ab]) (.+) missed$")
_CHARGING = re.compile(r"^(p[12][ab]) (.+) is charging$")
_PROTECTED = re.compile(r"^(p[12][ab]) (.+) protected \((\d+) in (\d+)\)$")
_FAINTED = re.compile(r"^(p[12][ab]) fainted$")
_SIDE = re.compile(r"^(p[12]) side ([+-])(\S+)$")
_LOST = re.compile(r"^(p[12][ab]) lost (\S+)(?: \((.+)\))?$")
_TRACED = re.compile(r"^(p[12][ab]) traced (\S+)$")
_FIELD = re.compile(r"^(terrain|weather) -> (\S+)$")
_ENDED = re.compile(r"^(\S+) ended$")
_REDIRECT = re.compile(r"^(p[12][ab]) (.+) redirected to (p[12][ab])$")
_SET = re.compile(r"^(p[12][ab]) set (\S+)$")
_LEFT_TO = re.compile(r"nothing left to (\S+)")
#: Two English words in a row in what a translator gave back: an event it could not put in Japanese.
_ENGLISH = re.compile(r"[A-Za-z]{3,} [A-Za-z]{3,}")


def narrate(
    reg: Regulation, loc: Localiser, events: list[str], acts: list[list[Any]], pos: Position
) -> dict[str, Any]:
    """The port's trace of a turn, read into rows: ``acts`` (one per action, in the order they
    were taken: who, what, whom it hit and for how many percent of the Pokemon's maximum HP, and
    the small marks -- a critical hit is not in the trace, a block is), ``residual`` (the end of
    the turn, one item each) and ``pre`` (what came before the first action). Nothing that the
    trace does not say is added; a line no rule reads is a mark named "その他", the words on hover."""
    species_at: dict[str, str] = {}
    maxhp: dict[str, int] = {}
    for si, side in enumerate(pos.sides):
        for slot, party in enumerate(side.active):
            if party is not None:
                code = f"p{si + 1}{'ab'[slot]}"
                species_at[code] = side.pokemon[party].species
                maxhp[code] = side.pokemon[party].maxhp
    by_species = {(si, to_id(m.species)): m.maxhp for si, side in enumerate(pos.sides) for m in side.pokemon}

    def seat_of(code: str) -> int | None:
        return {"1": 0, "2": 1}.get(code[1]) if code[:1] == "p" and len(code) > 1 else None

    def who(code: str) -> dict[str, Any]:
        sp = species_at.get(code, "")
        return {"code": code, "side": seat_of(code), "species": sp, "name": loc.species(sp) if sp else code}

    def why(reason: str | None) -> str:
        return show_game.translate_reason(loc, reason) if reason else ""

    def percent_of(code: str, amount: int) -> float | None:
        return amount / maxhp[code] * 100 if maxhp.get(code) else None

    def jp(text: str) -> str:
        """``nothing left to counter``: the one English sentence the trace has, in Japanese."""
        return _LEFT_TO.sub(lambda m: f"{show_game.named_id(loc, m.group(1))}する対象がなかった", text)

    def read(act: dict[str, Any], line: str, cause: str, *, actor: str | None) -> None:
        """One trace line into ``act`` (its target rows are in ``act['targets']`` by code)."""
        targets: dict[str, dict[str, Any]] = act["_targets"]

        def target(code: str) -> dict[str, Any]:
            if code not in targets:
                targets[code] = {**who(code), "dmg": None, "tags": [], "faint": False}
            return targets[code]

        def mark(code: str | None, text: str, raw: str | None = None) -> None:
            holder = act if code is None or code == actor else target(code)
            # A mark about the user of the move (its recoil, its own drop) is the user's, not
            # the move's: shown apart, after the targets.
            holder["tags"].append({"text": text, "raw": raw, "self": code is not None and code == actor})

        if found := _HP_LINE.match(line):
            code, sign, amount, reason = found.groups()
            share = percent_of(code, int(amount))
            if share is None:
                mark(code, f"HP {MINUS if sign == '-' else '+'}{amount}（{why(reason)}）")
            elif sign == "-" and to_id(reason) == cause and code != actor:
                entry = target(code)
                entry["dmg"] = (entry["dmg"] or 0) + share
            else:
                note = f"（{why(reason)}）" if to_id(reason) != cause else ""
                mark(code, f"HP {MINUS if sign == '-' else '+'}{max(share, 0.5):.0f}%{note}")
            return
        if found := _FAINTED.match(line):
            target(found.group(1))["faint"] = True
            return
        if found := _BLOCKED.match(line):
            mark(None, f"防がれた（{why(found.group(3))}）")
            return
        if found := _PROTECTED.match(line):
            if found.group(3) != found.group(4):
                mark(None, f"まもる 成功（連続 {found.group(3)}/{found.group(4)}）")
            return
        if found := _DID_NOT.match(line):
            word = "不発" if found.group(3) == "did not happen" else "失敗"
            # A reason is said only where a translator knows it (a status, a mechanic); the
            # trace's own sentences ("nothing left to counter") and chances are not put in
            # words that might say something else: the mark is just the failure.
            reason = why(found.group(4))
            known = not (
                _ENGLISH.search(reason)
                or re.match(r"\d+ in \d+$", found.group(4))
                or _LEFT_TO.search(found.group(4))
            )
            mark(None, f"{word}（{reason}）" if known else word)
            return
        if _MISSED.match(line):
            mark(None, "外れ")
            return
        if _CHARGING.match(line):
            mark(None, "溜め")
            return
        if found := _REDIRECT.match(line):
            mark(None, f"対象変更 → {who(found.group(3))['name']}")
            return
        if found := _STATUS_LINE.match(line):
            mark(found.group(1), loc.status(found.group(2)))
            return
        if found := _BOOST_LINE.match(line):
            code, stat, amount, _now, reason = found.groups()
            steps = int(amount)
            mark(
                code,
                f"{loc.stat(stat)} {MINUS if steps < 0 else '+'}{abs(steps)}"
                + (f"（{why(reason)}）" if reason and to_id(reason) != cause else ""),
            )
            return
        if found := _SIDE.match(line):
            _side, sign, cond = found.groups()
            mark(None, f"{'+' if sign == '+' else MINUS}{show_game.named_id(loc, cond)}")
            return
        if found := _FIELD.match(line):
            mark(None, f"{show_game.named_id(loc, found.group(2))}に")
            return
        if found := _ENDED.match(line):
            mark(None, f"{show_game.named_id(loc, found.group(1))} 終了")
            return
        if _SET.match(line):
            return
        if found := _TRACED.match(line):
            mark(found.group(1), f"トレース → {loc.ability(found.group(2))}")
            return
        if found := _LOST.match(line):
            mark(found.group(1), f"{show_game.named_id(loc, found.group(2))}を消費")
            return
        told = jp(show_game.translate_event(loc, line, {c: loc.species(s) for c, s in species_at.items()}))
        if _ENGLISH.search(told):
            mark(None, "その他", raw=line)
        else:
            mark(None, told)

    def enter(act: dict[str, Any], line: str) -> bool:
        """A switch-in or a forme change, which change who stands in the slot."""
        if found := _SWITCH_IN.match(line):
            code, sid = found.groups()
            si = seat_of(code)
            maxhp[code] = by_species.get((si, to_id(sid)), maxhp.get(code, 0))
            species_at[code] = sid
            act["into"] = who(code)
            return True
        if (found := _FORME.match(line)) and not _STATUS_LINE.match(line):
            code, sid = found.groups()
            species_at[code] = sid
            act["into"] = who(code)
            return True
        return False

    out: dict[str, Any] = {"acts": [], "residual": [], "pre": []}
    bounds = [start for start, _ in acts] + [len(events)]

    def loose(lines: list[str]) -> list[dict[str, Any]]:
        """Lines of no action: one item each (the end of a turn, a start), who they are about."""
        items = []
        for line in lines:
            holder = {"tags": [], "_targets": {}}
            read(holder, line, "", actor=None)
            for entry in holder["_targets"].values():
                bits = [f"HP {MINUS}{entry['dmg']:.0f}%"] if entry["dmg"] else []
                bits += ["ひんし"] if entry["faint"] else []
                bits += [t["text"] for t in entry["tags"]]
                items.append({**{k: entry[k] for k in ("side", "species", "name")}, "text": " ".join(bits)})
            for tag in holder["tags"]:
                found = re.match(r"^(p[12][ab]) ", line)
                code = found.group(1) if found else None
                subject = who(code) if code else {"side": None, "species": "", "name": ""}
                if (side_line := _SIDE.match(line)) is not None:
                    subject = {"side": seat_of(side_line.group(1)), "species": "", "name": ""}
                items.append(
                    {
                        **{k: subject[k] for k in ("side", "species", "name")},
                        "text": tag["text"],
                        "raw": tag["raw"],
                    }
                )
        return items

    if acts and acts[0][0] > 0:
        first = events[: acts[0][0]]
        out["pre"] = loose(first)
        for line in first:
            enter({}, line)
    number = 0
    for k, (start, label) in enumerate(acts or [(0, show_game.RESIDUAL_PHASE)]):
        label = str(label)
        lines = events[start : bounds[k + 1] if acts else len(events)]
        if label == show_game.RESIDUAL_PHASE:
            out["residual"] = loose(lines)
            for line in lines:
                enter({}, line)
            continue
        words = label.split()
        code = words[0]
        rest = " ".join(words[1:])
        number += 1
        act: dict[str, Any] = {
            "no": number,
            "actor": who(code),
            "side": seat_of(code),
            "kind": "move",
            "move": None,
            "targets": [],
            "tags": [],
            "_targets": {},
        }
        cause = ""
        if rest.startswith("switch->"):
            act["kind"] = "switch"
        elif rest == "mega":
            act["kind"] = "mega"
        else:
            cause = to_id(rest)
            act["move"] = loc.move(cause) or rest
        for line in lines:
            if not enter(act, line):
                read(act, line, cause, actor=code)
        act["targets"] = list(act.pop("_targets").values())
        out["acts"].append(act)
    return out


# ----------------------------------------------------------------------------- the mixtures


def pair_of(
    reg: Regulation, loc: Localiser, position: dict[str, Any], side: int, choice: str
) -> list[dict[str, str]]:
    """A choice as its Pokemon's parts (species, name, the move with the target it names)."""
    parsed = Position.from_json(position)
    text = show_game.name_action(reg, loc, position["sides"], side, choice, parsed)
    slots = position["sides"][side]["active"]
    dead = dead_labels(reg, position, side, choice)
    out = []
    for slot, part in enumerate(text.split(" ｜ ")):
        who, _, what = part.partition(": ")
        if what == "行動なし":
            continue
        what = what.replace("→ 敵", "→ 相手の").replace("→ 味方", "→ 味方の")  # as `hands`
        party = slots[slot] if slot < len(slots) else None
        species = position["sides"][side]["pokemon"][party]["species"] if party is not None else ""
        out.append(
            {
                "species": species,
                "name": loc.species(species) if species else who,
                "text": what,
                "dead": dead[slot] if slot < len(dead) else None,
            }
        )
    return out


def support_of(vector: list[float] | None, shown: list[list[Any]]) -> dict[str, Any] | None:
    """How many moves a mixture plays (probability at least `SUPPORT_MIN`), the ones among them
    the table does not list (count, sum), and the candidates it gives about none (count, sum).
    None for a record from before the whole vector was kept."""
    if vector is None:
        return None
    played = [v for v in vector if v >= SUPPORT_MIN]
    listed = sum(p for _c, p in shown if p >= SUPPORT_MIN)
    listed_n = sum(1 for _c, p in shown if p >= SUPPORT_MIN)
    return {
        "n": len(played),
        "menu": len(vector),
        "rest": [len(played) - listed_n, max(0.0, sum(played) - listed)],
        "small": [len(vector) - len(played), max(0.0, 1.0 - sum(played))],
    }


def place_text(m: dict[str, Any]) -> str:
    """Where the drawn move stands: among the moves the mixture plays, and out of them when its
    probability is below the threshold; the menu's size beside, small."""
    sup = m.get("sup")
    p = m["chosenP"] or 0.0
    if sup is None:
        return f"（{m['chosenRank']} 位）" if m["chosenRank"] else ""
    if p < SUPPORT_MIN:
        return f"（均衡の外・確率 {p * 100:.1f}%）"
    return f"（{sup['n']} 通り中 {m['chosenRank']} 位）"


def mixtures(
    reg: Regulation, loc: Localiser, decision: dict[str, Any], reads: list[dict[str, Any] | None]
) -> list[dict[str, Any] | None]:
    """Each seat's read for the page: the moves it plays (its support) and where the drawn one
    stands, the other side's it modelled, and how the drawn move does against the other side's
    moves -- the worst of all its columns, the best among those the other side plays, the one it
    really played. Names are given here once so the page only lays them out."""
    position = decision["position"]
    saw = decision.get("reads") or {}
    chosen = [decision.get("ownChosen"), decision.get("foeChosen")]
    out: list[dict[str, Any] | None] = []
    for side in (0, 1):
        got = saw.get(str(side))
        if got is None:
            out.append(None)
            continue
        other = 1 - side

        def pair(choice: str, actor: int) -> list[dict[str, str]]:
            return pair_of(reg, loc, position, actor, choice)

        value0 = reads[side]["value0"] if reads[side] else None
        own_win = None if value0 is None else (value0 if side == 0 else 1 - value0)
        supp = got.get("supp")
        if supp is None:  # a record from before the support was kept: the rows it has (>= 1%)
            supp = got["rows"]
        osupp = got.get("oppSupp")
        if osupp is None:
            osupp = got["opp"]
        actual = chosen[other]
        cols, vs = got.get("cols"), got.get("vs")
        on_menu = bool(cols) and actual in cols
        m = {
            "side": side,
            "own": own_win,
            "menu": got["menu"],
            "chosen": pair(got["chosen"], side),
            "chosenP": got["chosenP"],
            "chosenRank": got["chosenRank"],
            "chosenEv": got.get("chosenEv"),
            "rows": [{"pair": pair(c, side), "p": p, "picked": c == got["chosen"]} for c, p in supp],
            "sup": support_of(got.get("p"), got["rows"]),
            "osup": support_of(got.get("q"), got["opp"]),
            "opp": [{"pair": pair(c, other), "p": p, "actual": c == actual} for c, p in osupp],
            "eq": got.get("eq"),
            "guarantee": got.get("guarantee"),
            "generated": bool(got.get("generated")),
            # A generation read solved again and checked against the record (`reread`), or the
            # reason a re-solve was not shown.
            "reread": bool(got.get("reread")),
            "rereadNote": got.get("rereadNote"),
        }
        for key, field in (("worst", "hardChosen"), ("best", "bestIn")):
            found = got.get(field)
            m[key] = (
                None
                if found is None
                else {"pair": pair(found[0], other), "ev": found[1], "actual": found[0] == actual}
            )
        m["actualEv"] = (
            {"pair": pair(actual, other), "ev": vs[cols.index(actual)]} if on_menu and vs else None
        )
        m["actualOffMenu"] = bool(cols) and not on_menu
        out.append(m)
    return out


# ----------------------------------------------------------------------------- the model


def build(reg: Regulation, loc: Localiser, line: dict[str, Any]) -> dict[str, Any]:
    """Everything the page shows, as plain values."""
    tr = line["transcript"]
    decisions = tr["decisions"]
    reads: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in line["moves"]:
        got = read_of(row)
        if got is not None:
            reads[row["decision"]][row["side"]] = got
    positions = [Position.from_json(d["position"]) for d in decisions]
    final = Position.from_json(tr["finalPosition"]) if tr.get("finalPosition") else None
    move_ids = [i for i, d in enumerate(decisions) if d["kind"] == "move"]
    turns: list[dict[str, Any]] = []
    previous = None
    with seat_labels():
        for n, i in enumerate(move_ids):
            d = decisions[i]
            nxt = move_ids[n + 1] if n + 1 < len(move_ids) else None
            after = positions[nxt] if nxt is not None else final
            between = [decisions[j] for j in range(i + 1, nxt if nxt is not None else len(decisions))]
            values = [r["value0"] for r in reads[i].values()]
            value = sum(values) / len(values) if values else None
            events = d.get("events")
            entries: dict[str, Any] = {"acts": [], "residual": [], "pre": [], "arrive": []}
            note = None
            if events:
                entries = narrate(reg, loc, events["lines"], events["acts"], positions[i])
                entries["arrive"] = []
                if events.get("paused"):
                    note = (
                        "途中で交代が入ったため、出来事は交代までの分です（その後は場の変化を見てください）"
                    )
                elif not events.get("matched", True):
                    note = "port の再現が実際の局面と一致しませんでした"
                if events.get("note"):
                    note = f"{note} {events['note']}" if note else events["note"]
            for extra in between:
                sides = extra["position"]["sides"]
                for s in (0, 1):
                    choice = extra["ownChosen" if s == 0 else "foeChosen"]
                    if extra["kind"] not in ("replacement", "selfswitch") or not choice:
                        continue
                    for part in choice.split(", "):
                        if part.startswith("switch"):
                            mon = sides[s]["pokemon"][int(part.split()[1]) - 1]
                            entries["arrive"].append(
                                {
                                    "side": s,
                                    "species": mon["species"],
                                    "name": loc.species(mon["species"]),
                                    "after": "ひんし" if extra["kind"] == "replacement" else "途中の交代",
                                }
                            )
            turn = {
                "turn": d["turn"],
                "decision": i,
                "board": board(reg, loc, positions[i], after, d.get("shown")),
                "field": field_texts(loc, d["position"]),
                "hands": hands(reg, loc, d["position"], [d["ownChosen"], d["foeChosen"]]),
                "events": entries,
                "note": note,
                "reads": [reads[i].get(0), reads[i].get(1)],
                "mix": mixtures(reg, loc, d, [reads[i].get(0), reads[i].get(1)]),
                "value": value,
                "delta": None if value is None or previous is None else (value - previous) * 100,
            }
            previous = value if value is not None else previous
            turns.append(turn)
    return {
        "turns": turns,
        "final": board(reg, loc, final, None, None) if final else None,
        "finalField": field_texts(loc, tr["finalPosition"]) if tr.get("finalPosition") else ([], [[], []]),
        "outcome": line.get("outcome"),
        "endReason": line.get("endReason"),
        "turnCount": line.get("turns"),
        "adjudicated": line.get("adjudicated"),
        "conditions": line.get("conditions"),
        "seed": line.get("seed"),
        "pair": line.get("pair"),
        "game": line.get("game"),
        "teams": line.get("teams"),
        "teamNames": tr.get("teamNames"),
        "sheets": [
            {"six": tr["ownSix"], "pick": tr["ownPick"]},
            {"six": tr["foeSix"], "pick": tr["foePick"]},
        ],
    }


# ----------------------------------------------------------------------------- the page

CSS = """
:root{--bg:#f2f4f8;--surface:#fff;--surface-2:#eceff5;--line:#d9dfe9;--ink:#172033;--dim:#5b6780;--faint:#8a94a8;
--track:#e2e6ee;--s0:#c55f1b;--s0-ink:#9c4710;--s1:#2462c8;--s1-ink:#1b4d9f;--good:#1f9a55;--warn:#c29206;--bad:#cf4338;
--shadow:0 1px 2px rgba(23,32,51,.06),0 4px 16px rgba(23,32,51,.05);--radius:14px;
--font:"Zen Kaku Gothic New","Hiragino Sans","Yu Gothic UI","Meiryo",system-ui,sans-serif;
--num:"IBM Plex Mono",ui-monospace,"Cascadia Mono",Consolas,monospace}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#0f131a;--surface:#161b24;--surface-2:#1d2330;--line:#283040;
--ink:#e5e9f1;--dim:#9aa4b8;--faint:#6d788d;--track:#242b39;--s0:#f08d45;--s0-ink:#f6a86f;--s1:#5d9df5;--s1-ink:#8ab8f8;
--good:#45c27d;--warn:#e2bd3a;--bad:#ee695e;--shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.25)}}
*{box-sizing:border-box;min-width:0}html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--font);-webkit-font-smoothing:antialiased}
main{max-width:960px;margin:0 auto;padding:16px 14px 48px}
.n{font-family:var(--num);font-variant-numeric:tabular-nums}
.dim{color:var(--dim)}small{font-size:12px;color:var(--faint)}
.c0{color:var(--s0-ink)}.c1{color:var(--s1-ink)}
h2{font-size:15px;margin:28px 0 8px;color:var(--dim);font-weight:600;letter-spacing:.02em}
.banner{border-radius:var(--radius);padding:18px 20px;background:var(--surface);box-shadow:var(--shadow);border-left:8px solid var(--faint)}
.banner.w0{border-left-color:var(--s0)}.banner.w1{border-left-color:var(--s1)}
.banner h1{margin:0 0 6px;font-size:28px;line-height:1.25}
.banner p{margin:3px 0;color:var(--dim);font-size:13px}
.banner .vs{color:var(--ink);font-size:15px;font-weight:600}
.seat{display:inline-block;font-size:12px;font-weight:700;padding:1px 8px;border-radius:999px;color:#fff;white-space:nowrap}
.seat.s0{background:var(--s0)}.seat.s1{background:var(--s1)}
.chart{background:var(--surface);border-radius:var(--radius);box-shadow:var(--shadow);padding:10px 8px 4px}
.chart svg{display:block;width:100%;height:auto}.chart .mob{display:none}
.chart text{font-size:11px;fill:var(--dim);font-family:var(--num)}
.chart .lab{font-size:12px;font-weight:700}
.chart a text{fill:var(--ink)}
.chart .ax{stroke:var(--line);stroke-width:1}.chart .mid{stroke:var(--faint);stroke-dasharray:4 4;stroke-width:1}
.chart .ln{fill:none;stroke:var(--ink);stroke-width:2.2;stroke-linejoin:round}
.chart .a0{fill:var(--s0);opacity:.22}.chart .a1{fill:var(--s1);opacity:.22}
.chart .dot{fill:var(--surface);stroke:var(--ink);stroke-width:2}
.chart .end0{fill:var(--s0)}.chart .end1{fill:var(--s1)}.chart .endh{fill:var(--faint)}
.teams{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.team{background:var(--surface);border-radius:var(--radius);box-shadow:var(--shadow);padding:12px 14px;border-top:5px solid var(--faint)}
.team.s0{border-top-color:var(--s0)}.team.s1{border-top-color:var(--s1)}
.team h3{margin:0 0 8px;font-size:16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.six{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}
.sx{text-align:center;font-size:11px;line-height:1.25}
.sx>span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sx>span.sn{white-space:normal;min-height:2.5em}.fm{display:block;font-size:9.5px;line-height:1.2}
.nm .fm{display:inline;margin-left:3px}
.sx .ic{width:100%;max-width:64px;height:auto;aspect-ratio:1}
.sx.out{opacity:.4}.sx .tag{display:block;font-size:10px;font-weight:700}
.sx.pick .ic,.sx.pick .ph{outline:2px solid var(--faint);outline-offset:-2px;border-radius:8px}
.team.s0 .sx.pick .ic,.team.s0 .sx.pick .ph{outline-color:var(--s0)}.team.s1 .sx.pick .ic,.team.s1 .sx.pick .ph{outline-color:var(--s1)}
.ic{image-rendering:pixelated;width:52px;height:52px;object-fit:contain}
.ph{display:inline-flex;align-items:center;justify-content:center;background:var(--surface-2);color:var(--dim);font-weight:700;border-radius:8px;width:52px;height:52px}
.ic.sm,.ph.sm{width:32px;height:32px;font-size:14px}
.turn{margin:14px 0;background:var(--surface);border-radius:var(--radius);box-shadow:var(--shadow);padding:12px 14px;scroll-margin-top:12px}
.turn-h{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.tn{font-size:13px;color:var(--dim)}.tn b{font-size:30px;font-family:var(--num);color:var(--ink);line-height:1}
.wv{font-size:14px}.wv b{font-family:var(--num);font-size:16px}
.dl{font-family:var(--num);font-weight:700;font-size:14px;margin-left:.6em}
.dl .up{color:var(--s0-ink)}.dl .down{color:var(--s1-ink)}
.reads{padding-bottom:6px;margin-bottom:6px;border-bottom:1px solid var(--line)}
.legend{font-size:12px;color:var(--dim);margin:0 0 8px}
.mv{font-size:12px;margin-top:3px;color:var(--ink);line-height:1.35;white-space:normal}
.ic.faint{filter:grayscale(1);opacity:.5}
.wb{position:relative;width:100%;max-width:260px;height:26px;margin:2px 0 8px}
.wb .bar{position:absolute;left:0;right:0;top:10px;display:flex;height:6px;border-radius:3px;overflow:hidden;background:var(--track)}
.wb .bar i{display:block;height:100%}.wb .b0{background:var(--s0)}.wb .b1{background:var(--s1)}
.wb .gap{position:absolute;top:8px;height:10px;background:var(--warn);opacity:.55;border-radius:2px}
.wb .mk{position:absolute;font-size:10px;line-height:10px;transform:translateX(-50%)}
.wb .m0{top:-1px;color:var(--s0-ink)}.wb .m1{top:17px;color:var(--s1-ink)}
.wb .split{position:absolute;left:calc(100% + 8px);top:5px;font-size:11px;color:var(--warn);white-space:nowrap;font-weight:700}
.body{display:grid;grid-template-columns:1.15fr 1fr;gap:14px}
.row{display:flex;align-items:stretch;gap:8px;padding:6px 0 6px 8px;border-left:4px solid var(--faint)}
.row.s0{border-left-color:var(--s0)}.row.s1{border-left-color:var(--s1)}
.row .seat{align-self:flex-start;margin-top:2px}
.mons{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.mon{display:flex;gap:6px;align-items:center}
.mon.faint{filter:grayscale(1);opacity:.55}
.mon .mi{flex:1}.nm{font-weight:700;font-size:13px;line-height:1.25;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hp{height:7px;border-radius:4px;background:var(--track);overflow:hidden;margin:3px 0 1px}
.hp i{display:block;height:100%;background:var(--good)}.hp i.mid{background:var(--warn)}.hp i.low{background:var(--bad)}
.hpt{font-size:11px;color:var(--dim)}.hpt .dd{font-weight:700;color:var(--bad)}.hpt .du{font-weight:700;color:var(--good)}
.chips{display:flex;gap:3px;flex-wrap:wrap}.chip{font-size:10px;padding:0 5px;border-radius:5px;background:var(--surface-2);color:var(--dim);border:1px solid var(--line)}
.chip.new{background:var(--warn);color:#000;border-color:var(--warn)}
.bench{display:flex;align-items:center;gap:3px;margin-top:3px;flex-wrap:wrap}
.ph.q{border:1px dashed var(--faint);background:transparent}
.mid{text-align:center;font-size:12px;color:var(--dim);padding:2px 0}
.mid span{display:inline-block;background:var(--surface-2);border-radius:999px;padding:0 10px;margin:0 3px}
.hand{padding:3px 0 3px 8px;border-left:4px solid var(--faint);margin-top:6px}
.hand.s0{border-left-color:var(--s0)}.hand.s1{border-left-color:var(--s1)}
.hand .it{display:flex;align-items:center;gap:6px;font-size:13px;margin-top:2px}
.ev{list-style:none;margin:0;padding:0}.ev li{font-size:13px;padding:2px 0 2px 8px;border-left:3px solid var(--line);margin-bottom:3px}
.ev li.s0{border-left-color:var(--s0)}.ev li.s1{border-left-color:var(--s1)}
.ev li.end{color:var(--dim)}
.ev .more{font-size:12px;color:var(--faint);border-left-color:transparent}
.ev .fr{white-space:normal}
.evh{font-size:12px;color:var(--dim);margin:12px 0 4px;font-weight:600}
.field{margin-top:6px}
.acts{list-style:none;margin:0;padding:0}
.act{display:flex;gap:8px;align-items:flex-start;padding:5px 0 5px 8px;border-left:4px solid var(--faint);margin-bottom:4px;background:var(--surface-2);border-radius:0 8px 8px 0}
.act.s0{border-left-color:var(--s0)}.act.s1{border-left-color:var(--s1)}
.act .no{color:var(--faint);font-size:12px;min-width:1.4em;text-align:right;padding-top:5px}
.act .ln{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;font-size:14px;flex:1}
.who{display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
.wn{font-weight:700}.wn .fm{display:inline;margin-left:2px}
.ic.xs,.ph.xs{width:28px;height:28px;font-size:12px}
.ic.f0,.ph.f0{outline:2px solid var(--s0);outline-offset:-2px;border-radius:6px}
.ic.f1,.ph.f1{outline:2px solid var(--s1);outline-offset:-2px;border-radius:6px}
.mv{font-weight:800}.ar{color:var(--faint)}
.tg{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;flex-wrap:nowrap}
.tg.mine{margin-left:auto}
.mix{margin:6px 0 4px;border-radius:10px;background:var(--surface-2)}
.mix>summary{cursor:pointer;list-style:none;padding:8px 12px;position:relative}
.mix>summary::-webkit-details-marker{display:none}
.mix>summary::after{content:"▸";position:absolute;right:12px;top:8px;color:var(--dim)}
.mix[open]>summary::after{content:"▾"}
.mh{display:block;font-size:12px;font-weight:700;color:var(--dim);margin-bottom:4px}.mh small{font-weight:400;margin-left:8px}
.mix>summary{padding-right:32px}
.msl{font-size:12.5px;padding:2px 0 2px 8px;margin-bottom:4px}
.msl .l1,.msl .l2{display:block}.msl .l2{padding-left:2.6em;margin-top:1px}
.msl .seat{margin-right:6px}
.rng{position:relative;height:8px;margin:2px 8px 8px 8px;max-width:360px;background:var(--track);border-radius:4px}
.rng .m50{position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:var(--faint)}
.rng .band{position:absolute;top:0;height:8px;border-radius:4px;background:var(--faint)}
.rng .band.s0{background:var(--s0)}.rng .band.s1{background:var(--s1)}
.rng .pt{position:absolute;top:-3px;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:var(--surface);border:3px solid var(--ink)}
.tag.dead{color:var(--warn);border-color:var(--warn);font-size:11px;white-space:normal}
.eqt{margin-top:4px}.eqt .tag{font-size:12px;white-space:normal;border-radius:10px}.eqt .tag.off{color:var(--faint)}
.tag.lb{margin-right:6px;font-size:11px;white-space:nowrap}
.gq{color:var(--good);font-size:11px}
.bc.wr{position:relative;overflow:visible;background:var(--track);height:6px;margin:1px 0}
.bc.wr .m50{position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:var(--faint)}
.bc.wr .dt{position:absolute;top:-4px;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:var(--faint);border:2px solid var(--surface)}
.bc.wr .dt.s0{background:var(--s0)}.bc.wr .dt.s1{background:var(--s1)}
.nc .gz{color:var(--faint);font-size:11px}
.msl b.rare{color:var(--faint)}.msl b.often{font-weight:800}
.mbody{padding:4px 12px 12px;display:grid;grid-template-columns:1fr;gap:12px}
.mseat{border-left:4px solid var(--faint);padding-left:10px}.mseat.s0{border-left-color:var(--s0)}.mseat.s1{border-left-color:var(--s1)}
.mseat h4{margin:4px 0 6px;font-size:14px}.mseat h5{margin:10px 0 4px;font-size:13px}.mseat h5 small{margin-left:6px}
.mseat p{margin:2px 0;font-size:12.5px}
.mt{width:100%}
.mt .r{display:grid;grid-template-columns:minmax(0,320px) minmax(0,380px) 64px;gap:4px 14px;align-items:center;padding:4px 8px}
.mt .r:nth-child(odd){background:color-mix(in srgb,var(--surface) 55%,transparent)}
.mt .r.pick{outline:2px solid var(--faint);outline-offset:-2px;border-radius:6px}
.mseat.s0 .mt .r.pick{outline-color:var(--s0)}.mseat.s1 .mt .r.pick{outline-color:var(--s1)}
.pc .pr{display:flex;align-items:flex-start;gap:6px;font-size:12.5px;line-height:1.5}
.pt{min-width:0}.pt .wn{margin-right:2px}
.bc{height:8px;border-radius:4px;background:var(--track);overflow:hidden}.bb{display:block;height:8px;border-radius:4px;background:var(--faint)}
.bb.s0{background:var(--s0)}.bb.s1{background:var(--s1)}
.nc{text-align:right;white-space:nowrap}.nc .gp{color:var(--bad);font-size:11px}
.tag.pk{margin:2px 0 0 30px;font-size:11px}
.ic.xs.f0,.ic.xs.f1{width:24px;height:24px;flex:none}
@media (max-width:720px){.mt .r{grid-template-columns:minmax(0,1fr) auto}.bc{grid-column:1/-1;order:3}.pt .pm{display:block}
.mh small{display:block;margin-left:0}.msl .l2{padding-left:0}}
.dmg{font-family:var(--num);font-weight:700;color:var(--bad)}
.tag{display:inline-block;font-size:12px;line-height:1.5;padding:0 8px;border-radius:999px;border:1px solid var(--line);background:var(--surface);color:var(--dim);white-space:nowrap}
.resid{margin:8px 0 0;padding:8px 10px;border-radius:10px;background:var(--surface-2)}
.resid .rh{margin:0 0 4px;font-size:12px;font-weight:700;color:var(--dim)}
.resid ul{list-style:none;margin:0;padding:0}
.resid li{display:flex;align-items:center;gap:8px;font-size:13px;padding:2px 0 2px 8px;border-left:3px solid var(--line)}
.resid li.s0{border-left-color:var(--s0)}.resid li.s1{border-left-color:var(--s1)}
.resid li.more{color:var(--faint);font-size:12px}
.resid.arr{background:transparent;border:1px dashed var(--line)}
.note{font-size:12px;color:var(--warn);margin:4px 0}
.reads{margin-top:10px;font-size:12.5px;color:var(--dim)}.reads p{margin:3px 0}
.reads .seat{margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}
th,td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;font-size:12px}
.end{margin:14px 0;padding:12px 14px;border-radius:var(--radius);background:var(--surface);box-shadow:var(--shadow)}
@media (max-width:720px){.teams{grid-template-columns:1fr}.body{grid-template-columns:1fr}
.six{grid-template-columns:repeat(3,minmax(0,1fr));gap:8px 4px}.banner h1{font-size:24px}.ic,.ph{width:40px;height:40px}
.ic.sm,.ph.sm{width:32px;height:32px}.mons{grid-template-columns:1fr}.mons .mon+.mon{margin-top:10px;padding-top:8px;border-top:1px solid var(--line)}
.sx{font-size:10px}.sx .ic,.sx .ph{width:48px;height:48px;max-width:none;aspect-ratio:auto}
.chart .desk{display:none}.chart .mob{display:block}.chart .mob text{font-size:12px}.chart .mob .lab{font-size:13px}
.wb .split{left:0;top:26px}.wb.hs{margin-bottom:26px}}
.mnote{font-size:12px;margin:0}
.dl-legend{list-style:none;margin:0 0 6px;padding:0;font-size:13px}.dl-legend li{margin:4px 0}
"""


def _hp_class(pct: int) -> str:
    return "" if pct > 50 else "mid" if pct > 20 else "low"


def _mon_html(mon: dict[str, Any] | None, sprites: Sprites, hand: dict[str, Any] | None = None) -> str:
    if mon is None:
        return '<div class="mon"><span class="dim">（空き）</span></div>'
    delta = ""
    if mon["faintedNow"]:
        delta = ' <span class="dd">ひんし</span>'
    elif mon["delta"]:
        delta = f' <span class="{"dd" if mon["delta"] < 0 else "du"}">{signed(mon["delta"], "%")}</span>'
    chips = "".join(f'<span class="chip">{esc(c)}</span>' for c in mon["chips"])
    if mon["newStatus"]:
        chips += f'<span class="chip new">{esc(mon["newStatus"])}</span>'
    if mon["left"]:
        chips += '<span class="chip">交代</span>'
    state = "faint" if mon["fainted"] else ""
    hp = (
        ""
        if mon["fainted"]
        else f'<div class="hp"><i class="{_hp_class(mon["pct"])}" style="width:{mon["pct"]}%"></i></div>'
    )
    text = "" if mon["fainted"] else f'{mon["pct"]}% <small class="n">{mon["hp"][0]}/{mon["hp"][1]}</small>'
    name = name_html(mon["name"]) + (" ひんし" if mon["fainted"] else "")
    move = ""
    if hand:
        mark = (
            f' <span class="tag dead" title="{esc(dead_meaning(hand["dead"]))}">{esc(hand["dead"])}</span>'
            if hand.get("dead")
            else ""
        )
        move = f'<div class="mv">{esc(hand["text"])}{mark}</div>'
    return (
        f'<div class="mon {state}">{sprites.img(mon["species"], mon["name"])}'
        f'<div class="mi"><div class="nm">{name}</div>{hp}'
        f'<div class="hpt n">{text}{delta}</div><div class="chips">{chips}</div>{move}</div></div>'
    )


def _side_html(
    side: int,
    view: dict[str, Any],
    sprites: Sprites,
    hands: list[dict[str, Any] | None] | None = None,
    tag: str = "",
) -> str:
    bench = ""
    if view["bench"]:
        icons = "".join(
            sprites.img(
                b["species"],
                f"{b['name']} {b['pct']}%" if not b["fainted"] else f"{b['name']} ひんし",
                "ic sm faint" if b["fainted"] else "ic sm",
            )
            for b in view["bench"]
        )
        bench = f'<div class="bench"><small>裏</small>{icons}</div>'
    hands = hands or []
    mons = "".join(
        _mon_html(m, sprites, hands[slot] if slot < len(hands) else None)
        for slot, m in enumerate(view["active"])
    )
    return (
        f'<div class="row s{side}"><span class="seat s{side}">{SEAT[side]}</span>'
        f'<div style="flex:1"><div class="mons">{mons}</div>{bench}{tag}</div></div>'
    )


def _reads_html(turn: dict[str, Any]) -> str:
    out = []
    for side in (0, 1):
        r = turn["reads"][side]
        who = f'<span class="seat s{side}">{SEAT[side]}</span> の読み'
        cond = ""
        if r is None:
            out.append(f"<p>{who} —</p>")
            continue
        depth = f"深さ {r['depth']}・" if r["depth"] is not None else ""
        stage = ""
        title = (
            f' title="読みの段: {esc(r["stage"])}"'
            if r["stage"]
            else ' title="どの段も終わらず、根の行列の答え（深さ 1）"'
            if r["depth"] == 1
            else ""
        )
        from0 = r["value0"]
        same = percent(1 - from0) == percent(from0)
        own = "" if side == 0 or same else f"（席 1 自身の見立ては {percent(1 - from0)}）"
        out.append(
            f'<p{title}>{who}{cond} {depth}席 0 から見て <b class="n c{side}">{percent(from0)}</b>{own}{stage}</p>'
        )
    return f'<div class="reads">{"".join(out)}</div>'


def _bar_html(turn: dict[str, Any]) -> str:
    value = turn["value"]
    if value is None:
        return ""
    marks, split = "", ""
    r0, r1 = turn["reads"]
    if r0 and r1:
        a, b = r0["value0"], r1["value0"]
        marks = (
            f'<span class="mk m0" style="left:{a * 100:.1f}%" title="席 0 の読み（席 0 から見て {percent(a)}）">▼</span>'
            f'<span class="mk m1" style="left:{b * 100:.1f}%" title="席 1 の読み（席 0 から見て {percent(b)}）">▲</span>'
        )
        if abs(a - b) * 100 >= SPLIT_PT:
            lo, hi = sorted((a, b))
            marks += f'<span class="gap" style="left:{lo * 100:.1f}%;width:{(hi - lo) * 100:.1f}%"></span>'
            split = f'<span class="split">読みが割れた（{abs(a - b) * 100:.0f}pt）</span>'
    return (
        f'<div class="wb{" hs" if split else ""}"><div class="bar"><i class="b0" style="width:{value * 100:.1f}%"></i>'
        f'<i class="b1" style="width:{(1 - value) * 100:.1f}%"></i></div>{marks}{split}</div>'
    )


def _who_html(side: int | None, species: str, name: str, sprites: Sprites, *, faint: bool = False) -> str:
    """The icon (framed in the seat's colour) and the name of one Pokemon in an action's line."""
    frame = f"f{side}" if side is not None else ""
    icon = sprites.img(species, name, f"ic xs {frame}{' faint' if faint else ''}") if species else ""
    return f'<span class="who">{icon}<span class="wn">{name_html(name)}</span></span>'


def _tag_html(tag: dict[str, Any]) -> str:
    hint = f' title="{esc(tag["raw"])}"' if tag.get("raw") else ""
    return f'<span class="tag"{hint}>{esc(tag["text"])}</span>'


def _act_html(act: dict[str, Any], sprites: Sprites) -> str:
    side = act["side"]
    actor = act["actor"]
    head = _who_html(side, actor["species"], actor["name"], sprites)
    if act["kind"] == "switch":
        into = act.get("into") or {"species": "", "name": "?"}
        body = f'{head}<span class="ar">→</span>{_who_html(side, into["species"], into["name"], sprites)}<span class="dim"> に交代</span>'
    elif act["kind"] == "mega":
        into = act.get("into") or {"species": actor["species"], "name": actor["name"]}
        body = f'{head}<span class="dim"> が</span><b class="mv">メガシンカ</b><span class="ar">→</span>{_who_html(side, into["species"], into["name"], sprites)}'
    else:
        body = f'{head}<b class="mv">{esc(act["move"])}</b>'
    targets = []
    for t in act["targets"]:
        bits = _who_html(t["side"], t["species"], t["name"], sprites, faint=t["faint"])
        if t["dmg"]:
            bits += f'<span class="dmg">{MINUS}{t["dmg"]:.0f}%</span>'
        if t["faint"]:
            bits += '<span class="tag">ひんし</span>'
        bits += "".join(_tag_html(x) for x in t["tags"])
        arrow = '<span class="ar">→</span>' if not targets else ""
        targets.append(f'<span class="tg">{arrow}{bits}</span>')
    body += "".join(targets)
    body += "".join(_tag_html(x) for x in act["tags"] if not x.get("self"))
    mine = [x for x in act["tags"] if x.get("self")]
    if mine:
        # The user's own marks, set apart with its icon.
        icon = sprites.img(actor["species"], actor["name"], f"ic xs f{side}") if actor["species"] else ""
        body += f'<span class="tg mine">{icon}<span class="tag">自分</span>{"".join(_tag_html(x) for x in mine)}</span>'
    return f'<li class="act s{side}"><span class="no n">{act["no"]}</span><div class="ln">{body}</div></li>'


def _items_html(items: list[dict[str, Any]], sprites: Sprites, cap: int | None = None) -> str:
    rows = []
    for item in items[:cap]:
        who = (
            _who_html(item["side"], item["species"], item["name"], sprites)
            if item["species"]
            else (
                f'<span class="seat s{item["side"]}">{SEAT[item["side"]]}</span>'
                if item["side"] is not None
                else ""
            )
        )
        hint = f' title="{esc(item["raw"])}"' if item.get("raw") else ""
        rows.append(
            f'<li class="it s{item["side"]}">{who}<span class="tx"{hint}>{esc(item["text"])}</span></li>'
        )
    if cap is not None and len(items) > cap:
        rows.append(f'<li class="more">ほか {len(items) - cap} 件</li>')
    return "".join(rows)


def _events_html(turn: dict[str, Any], sprites: Sprites) -> str:
    ev = turn["events"]
    if not (ev["acts"] or ev["residual"] or ev["pre"] or ev["arrive"]):
        return '<div class="evs"><p class="evh">起きたこと</p><p class="dim">出来事の記録なし</p></div>'
    parts = ['<div class="evs"><p class="evh">起きたこと</p>']
    if ev["pre"]:
        parts.append(
            f'<div class="resid"><p class="rh">ターン開始時</p><ul>{_items_html(ev["pre"], sprites, RESIDUAL_MAX)}</ul></div>'
        )
    if ev["acts"]:
        parts.append(f'<ol class="acts">{"".join(_act_html(a, sprites) for a in ev["acts"])}</ol>')
    if ev["residual"]:
        parts.append(
            f'<div class="resid"><p class="rh">ターン終了</p><ul>{_items_html(ev["residual"], sprites, RESIDUAL_MAX)}</ul></div>'
        )
    if ev["arrive"]:
        rows = "".join(
            f'<li class="it s{a["side"]}">{_who_html(a["side"], a["species"], a["name"], sprites)}'
            f'<span class="tx">が場に出た（{a["after"]}の後）</span></li>'
            for a in ev["arrive"]
        )
        parts.append(f'<div class="resid arr"><p class="rh">交代で場に出た</p><ul>{rows}</ul></div>')
    parts.append("</div>")
    return "".join(parts)


def _pair_html(pair: list[dict[str, str]], side: int, sprites: Sprites) -> str:
    """A move of one seat as its Pokemon's lines: icon, name, the move."""
    lines = "".join(
        f'<div class="pr">{sprites.img(p["species"], p["name"], f"ic xs f{side}") if p["species"] else ""}'
        f'<span class="pt"><span class="wn">{name_html(p["name"])}</span> <span class="pm">{esc(p["text"])}</span>'
        + (
            f' <span class="tag dead" title="{esc(dead_meaning(p["dead"]))}">{esc(p["dead"])}</span>'
            if p.get("dead")
            else ""
        )
        + "</span></div>"
        for p in pair
    )
    return lines or '<div class="pr dim">行動なし</div>'


def _pct(p: float) -> str:
    return "<1%" if p < 0.005 else f"{p * 100:.0f}%"


def _table_html(rows: list[dict[str, Any]], side: int, sprites: Sprites, *, kind: str) -> str:
    """One table of a mixture: the pair of moves, a bar (a probability) and its number, and a mark
    on the row that was drawn or that the other side really played."""
    body = []
    for r in rows:
        mark = ""
        if r.get("picked"):
            mark = '<span class="tag pk">選んだ手</span>'
        elif r.get("actual"):
            mark = '<span class="tag pk">相手が実際に選んだ手</span>'
        share = r["p"]
        body.append(
            f'<div class="r{" pick" if r.get("picked") else ""}"><div class="pc">{_pair_html(r["pair"], side, sprites)}{mark}</div>'
            f'<div class="bc"><span class="bb s{side}" style="width:{max(0.0, min(1.0, share)) * 100:.0f}%"></span></div>'
            f'<div class="nc"><span class="n">{_pct(share)}</span></div></div>'
        )
    return f'<div class="mt">{"".join(body)}</div>'


def _reply_table(items: list[dict[str, Any]], side: int, sprites: Sprites) -> str:
    """The drawn move against some of the other side's moves: a label, the pair, the win rate as a
    point on a 0-100 scale (a line at 50%, not a bar: a bar is a share), and the difference from
    the read's value."""
    body = []
    for it in items:
        gap = it.get("gap")
        if gap is None:
            diff = ""
        elif abs(gap) * 100 < 1:
            diff = ' <span class="gz n">±0</span>'
        else:
            cls = "gp" if gap < 0 else "gq"
            diff = f' <span class="{cls} n">{MINUS if gap < 0 else "+"}{abs(gap) * 100:.0f}pt</span>'
        mark = (
            '<span class="tag pk">実際の手</span>'
            if it.get("actual") and it["label"] != "実際に選んだ手"
            else ""
        )
        body.append(
            f'<div class="r"><div class="pc"><span class="tag lb">{esc(it["label"])}</span>'
            f"{_pair_html(it['pair'], 1 - side, sprites)}{mark}</div>"
            f'<div class="bc wr"><span class="m50"></span>'
            f'<span class="dt s{side}" style="left:{max(0.0, min(1.0, it["ev"])) * 100:.1f}%"></span></div>'
            f'<div class="nc"><span class="n">{it["ev"] * 100:.0f}%</span>{diff}</div></div>'
        )
    return f'<div class="mt">{"".join(body)}</div>'


def _rest_html(sup: dict[str, Any] | None) -> str:
    """Under a table: the candidates that were given about no probability (said apart, so that
    they are not read as moves played)."""
    if sup is None or sup["small"][0] <= 0:
        return ""
    return (
        f'<p class="dim rest">ほかの候補 {sup["small"][0]} 通りは確率 {SUPPORT_MIN * 100:g}% 未満'
        f"（合計 {_pct(sup['small'][1])}）で、均衡では打たない。</p>"
    )


def _played_n(m: dict[str, Any]) -> int:
    return m["sup"]["n"] if m.get("sup") else len(m["rows"])


def _eq_tag(m: dict[str, Any] | None) -> str:
    """The small tag under a seat's Pokemon: how likely the equilibrium is to play the drawn move."""
    if m is None:
        return ""
    p = m["chosenP"] or 0.0
    where = esc(place_text(m))
    if p < SUPPORT_MIN:
        return f'<div class="eqt"><span class="tag off">均衡外の手{where}</span></div>'
    return f'<div class="eqt"><span class="tag">均衡で打つ確率 <b class="n">{_pct(p)}</b>{where}</span></div>'


def _mix_seat(m: dict[str, Any] | None, side: int, sprites: Sprites) -> str:
    again = (
        "<small>　解き直した読み（記録の混合と一致を確かめた）</small>" if m is not None and m.get("reread") else ""
    )
    head = f'<h4><span class="seat s{side}">{SEAT[side]}</span> の読み{again}</h4>'
    if m is None:
        return f'<div class="mseat s{side}">{head}<p class="dim">この席の読みはありません</p></div>'
    view = f"（{SEAT[side]} から見た勝率）"
    rows = list(m["rows"])
    picked = (
        ""
        if any(r["picked"] for r in rows)
        else (
            f'<p class="dim">選んだ手は均衡の外（確率 {(m["chosenP"] or 0.0) * 100:.1f}%）:'
            + " / ".join(f" {esc(p['name'])} {esc(p['text'])}" for p in m["chosen"])
            + "</p>"
        )
    )
    osup = m.get("osup")
    n_opp = f"　相手の均衡で打つ {osup['n']} 通り" if osup else ""
    base = m["eq"] if m.get("eq") is not None else m["own"]
    items = []
    for key, label in (("worst", "一番辛い"), ("best", "一番有利")):
        got = m.get(key)
        if got is not None:
            items.append(
                {
                    "label": label,
                    "pair": got["pair"],
                    "ev": got["ev"],
                    "actual": got["actual"],
                    "gap": None if base is None else got["ev"] - base,
                }
            )
    real = m.get("actualEv")
    if real is not None and not any(i["actual"] for i in items):
        items.append(
            {
                "label": "実際に選んだ手",
                "pair": real["pair"],
                "ev": real["ev"],
                "actual": True,
                "gap": None if base is None else real["ev"] - base,
            }
        )
    if items:
        reply = _reply_table(items, side, sprites)
    else:
        reply = '<p class="dim">この読みには手ごとの値がありません（深さ 1 の答え）</p>'
    if m.get("actualOffMenu"):
        reply += '<p class="dim">相手が実際に選んだ手は、この読みの候補にありませんでした。</p>'
    if m.get("generated"):
        missing = (
            f"この手番は解き直しが記録と一致しなかったので、解き直した読みは出していません（{esc(m['rereadNote'])}）。"
            if m.get("rereadNote")
            else "生成の局の記録には、手ごとの値と、この席が相手に見ていた混合がありません。"
        )
        return (
            f'<div class="mseat s{side}">{head}'
            f"<h5>均衡で打っていた手<small>　均衡で打つ {_played_n(m)} 通り</small></h5>"
            f"{picked}{_table_html(rows, side, sprites, kind='mix')}{_rest_html(m.get('sup'))}"
            f'<p class="dim">{missing}</p></div>'
        )
    return (
        f'<div class="mseat s{side}">{head}'
        f"<h5>均衡で打っていた手<small>　均衡で打つ {_played_n(m)} 通り</small></h5>"
        f"{picked}{_table_html(rows, side, sprites, kind='mix')}{_rest_html(m.get('sup'))}"
        f"<h5>選んだ手に対する相手の手<small>{view}</small></h5>{reply}"
        f"<h5>相手の読み<small>{n_opp}</small></h5>"
        f"{_table_html(list(m['opp']), 1 - side, sprites, kind='opp')}{_rest_html(osup)}</div>"
    )


def _mix_line(m: dict[str, Any] | None, side: int) -> str:
    """The closed form: a seat's line of numbers and a scale of the drawn move's range."""
    label = f'<span class="seat s{side}">{SEAT[side]}</span>'
    if m is None:
        return f'<div class="msl"><div class="l1">{label}<span class="dim">この席の読みはありません</span></div></div>'
    line = f'{label}<span class="dim">均衡で打つ</span> <b class="n">{_played_n(m)}</b> <span class="dim">通り</span>'
    scale = ""
    worst, best = m.get("worst"), m.get("best")
    if worst and best:
        lo, hi = sorted((worst["ev"], best["ev"]))
        spot = m.get("chosenEv") if m.get("chosenEv") is not None else m["own"]
        dot = (
            f'<span class="pt s{side}" style="left:{max(0.0, min(1.0, spot)) * 100:.1f}%"></span>'
            if spot is not None
            else ""
        )
        line += (
            f' <span class="dim">・相手の均衡の中で 最悪</span> <b class="n">{worst["ev"] * 100:.0f}%</b>'
            f' <span class="dim">／ 最善</span> <b class="n">{best["ev"] * 100:.0f}%</b>'
            f' <span class="dim">（{SEAT[side]} から見て）</span>'
        )
        scale = (
            f'<div class="rng"><span class="m50"></span>'
            f'<span class="band s{side}" style="left:{max(0.0, lo) * 100:.1f}%;width:{max(0.01, min(1.0, hi) - max(0.0, lo)) * 100:.1f}%"></span>'
            f"{dot}</div>"
        )
    elif m.get("generated"):
        line += ' <span class="dim">・手ごとの値は記録にありません</span>'
    else:
        line += ' <span class="dim">・手ごとの値はありません（深さ 1 の答え）</span>'
    return f'<div class="msl"><div class="l1">{line}</div>{scale}</div>'


def _mix_html(t: dict[str, Any], sprites: Sprites) -> str:
    mix = t["mix"]
    if not any(mix):
        return ""
    lines = "".join(_mix_line(mix[s], s) for s in (0, 1))
    seats = "".join(_mix_seat(mix[s], s, sprites) for s in (0, 1))
    # The words of "worst" and "best" only where a table of replies stands under them.
    has_values = any(m and (m.get("worst") or m.get("best")) for m in mix)
    note = (
        '<p class="dim mnote">辛い＝相手の全ての手の中で、選んだ手に一番辛い手。有利＝相手が均衡の中で打つ手のうち、'
        "選んだ手に一番有利な手。値は相手が裏によらず同じ手を打つときの期待値で、深く読んでいない手は浅い読みのまま。</p>"
        if has_values
        else ""
    )
    return (
        f'<details class="mix"{" open" if sprites.open_mix else ""}><summary>'
        '<span class="mh">この手番の読み <small>両席の均衡と、相手の読みの中の値。真の強さではありません</small></span>'
        f"{lines}</summary>"
        f'<div class="mbody">{note}{seats}</div></details>'
    )


def _turn_html(t: dict[str, Any], model: dict[str, Any], sprites: Sprites) -> str:
    value = t["value"]
    head = "勝率なし" if value is None else f'席 0 から <b class="n">{percent(value)}</b>'
    delta = ""
    if t["delta"] is not None and round(t["delta"]) != 0:
        rounded = round(t["delta"])
        cls = "up" if rounded > 0 else "down"
        arrow = "▲" if rounded > 0 else "▼"
        delta = (
            f'<span class="dl" title="前のターンとの差（席 0 から見て。▲は席 0 有利、▼は席 1 有利）">'
            f'<span class="{cls}">{arrow}</span>{signed(t["delta"], "pt")}</span>'
        )
    middle, screens = t["field"]
    field_mid = (
        f'<div class="mid">{"".join(f"<span>{esc(x)}</span>" for x in middle)}</div>' if middle else ""
    )
    kept = "".join(
        f'<div class="mid"><span class="seat s{s}">{SEAT[s]}</span> '
        + "".join(f"<span>{esc(x)}</span>" for x in screens[s])
        + "</div>"
        for s in (0, 1)
        if s < len(screens) and screens[s]
    )
    sides = (
        _side_html(0, t["board"][0], sprites, t["hands"][0], _eq_tag(t["mix"][0]))
        + field_mid
        + _side_html(1, t["board"][1], sprites, t["hands"][1], _eq_tag(t["mix"][1]))
        + kept
    )
    note = f'<p class="note">{esc(t["note"])}</p>' if t["note"] else ""
    return (
        f'<section class="turn" id="t{t["turn"]}">'
        f'<div class="turn-h"><div class="tn">T<b>{t["turn"]}</b></div><div class="wv">{head}{delta}</div></div>'
        f"{_bar_html(t)}"
        f"{_reads_html(t)}"
        f"{_mix_html(t, sprites)}"
        f'<div class="field">{sides}</div>'
        f"{_events_html(t, sprites)}{note}"
        "</section>"
    )


def _axis(values: list[float]) -> tuple[float, float]:
    """The chart's vertical range (0-1): the values' with 10 points around, on multiples of 5,
    always with 50% in it; all of 0-100 when that is nearly what it comes to."""
    lo = min([*values, 0.5]) * 100
    hi = max([*values, 0.5]) * 100
    lo = max(0, int((lo - 10) // 5 * 5))
    hi = min(100, int(-((-(hi + 10)) // 5) * 5))
    if hi - lo >= 85:
        lo, hi = 0, 100
    return lo / 100, hi / 100


def _chart_svg(model: dict[str, Any], turns: list[dict[str, Any]], mobile: bool) -> str:
    """The win-rate line once: 960x230 for a wide screen, 360x280 (text at reading size, fewer
    marks) for a narrow one; the page shows the one that fits (`.desk`, `.mob`)."""
    width, height, left, right, top, bottom = (
        (360, 280, 40, 14, 26, 26) if mobile else (960, 230, 44, 20, 24, 26)
    )
    uid = "m" if mobile else "d"
    outcome = model["outcome"]
    decided = outcome in (0.0, 1.0)
    n = len(turns)
    last = n if decided else max(n - 1, 1)
    lo, hi = _axis([t["value"] for t in turns])

    def x(k: int) -> float:
        return left + (width - left - right) * (k / last)

    def y(v: float) -> float:
        return top + (height - top - bottom) * (1 - (v - lo) / (hi - lo))

    points = [(x(k), y(t["value"])) for k, t in enumerate(turns)]
    if decided:
        points.append((x(n), y(hi if outcome == 1.0 else lo)))
    mid = y(0.5)
    line = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    area = f"{points[0][0]:.1f},{mid:.1f} {line} {points[-1][0]:.1f},{mid:.1f}"
    grid = "".join(
        f'<line class="ax" x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
        f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{round(v * 100)}%</text>'
        for v in {lo, hi}
    )
    every = -(-n // 8) if mobile else (1 if n <= 20 else 2)
    ticks = "".join(
        f'<text x="{x(k):.1f}" y="{height - 8}" text-anchor="middle">{t["turn"]}</text>'
        for k, t in enumerate(turns)
        if k % every == 0 or k == n - 1
    )
    if decided:
        ticks += f'<text x="{x(n):.1f}" y="{height - 8}" text-anchor="end">結果</text>'
    big = sorted(
        (k for k, t in enumerate(turns) if t["delta"] is not None and abs(t["delta"]) >= MARK_PT),
        key=lambda k: -abs(turns[k]["delta"]),
    )[: 3 if mobile else 5]
    marks = ""
    for k in sorted(big):
        t = turns[k]
        px, py = points[k]
        # A rising line goes up past its point to the right, so the note goes under it; a
        # falling one, over it.
        ty = py + 20 if t["delta"] > 0 else py - 12
        anchor = "start" if px < width - (100 if mobile else 130) else "end"
        marks += (
            f'<a href="#t{t["turn"]}"><circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="5"/>'
            f'<text x="{px:.1f}" y="{ty:.1f}" text-anchor="{anchor}" class="lab">T{t["turn"]} {signed(t["delta"], "pt")}</text></a>'
        )
    cls = "end0" if decided and outcome == 1.0 else "end1" if decided else "endh"
    endx, endy = points[-1]
    return (
        f'<svg class="{"mob" if mobile else "desk"}" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="席 0 から見た勝率の推移">'
        f'<defs><clipPath id="up{uid}"><rect x="0" y="0" width="{width}" height="{mid:.1f}"/></clipPath>'
        f'<clipPath id="dn{uid}"><rect x="0" y="{mid:.1f}" width="{width}" height="{height}"/></clipPath></defs>'
        f'{grid}<line class="mid" x1="{left}" x2="{width - right}" y1="{mid:.1f}" y2="{mid:.1f}"/>'
        f'<text x="{left - 6}" y="{mid + 4:.1f}" text-anchor="end">50%</text>'
        f'<polygon class="a0" clip-path="url(#up{uid})" points="{area}"/>'
        f'<polygon class="a1" clip-path="url(#dn{uid})" points="{area}"/>'
        f'<polyline class="ln" points="{line}"/>{marks}'
        f'<circle class="{cls}" cx="{endx:.1f}" cy="{endy:.1f}" r="{6 if mobile else 7}"/>'
        f'<text x="{left + 4}" y="{top - 9}" class="lab" style="fill:var(--s0-ink)">↑ 席 0 優勢</text>'
        f'<text x="{left + 4}" y="{height - bottom - 6}" class="lab" style="fill:var(--s1-ink)">↓ 席 1 優勢</text>'
        f"{ticks}</svg>"
    )


def _chart(model: dict[str, Any]) -> str:
    turns = [t for t in model["turns"] if t["value"] is not None]
    if not turns:
        return ""
    lo, hi = _axis([t["value"] for t in turns])
    decided = model["outcome"] in (0.0, 1.0)
    zoom = "" if (lo, hi) == (0, 1) else f"縦軸は {round(lo * 100)}〜{round(hi * 100)}% に拡大。"
    end_note = "右端の丸は結果（勝った席の色）。" if decided else "右端の丸は打ち切り時点の勝率（灰色）。"
    return (
        f'<div class="chart">{_chart_svg(model, turns, False)}{_chart_svg(model, turns, True)}'
        f'<p style="margin:0 6px 6px"><small>横はターン、縦は席 0 から見た勝率（1 ターンの 2 つの読みの平均）。{zoom}'
        f"50% より上が席 0 優勢（橙）、下が席 1 優勢（青）。{end_note}"
        f"ターンへの印は前のターンと {MARK_PT}pt 以上動いたところ。</small></p></div>"
    )


def _teams(model: dict[str, Any], names: list[list[str]], sprites: Sprites) -> str:
    out = []
    for s in (0, 1):
        sheet = model["sheets"][s]
        cells = []
        for index, name in enumerate(names[s]):
            order = sheet["pick"].index(index) if index in sheet["pick"] else None
            tag = "選出外" if order is None else ("先発" if order < 2 else "裏")
            cls = "out" if order is None else "pick"
            cells.append(
                f'<div class="sx {cls}">{sprites.img(sheet["six"][index], name)}<span class="sn">{name_html(name)}</span>'
                f'<span class="tag c{s}">{tag}</span></div>'
            )
        cond = model["conditions"][s] if model["conditions"] else ""
        team = (model.get("teamNames") or (model["teams"] or ["", ""]))[s]
        out.append(
            f'<div class="team s{s}"><h3><span class="seat s{s}">{SEAT[s]}</span>'
            f'<span>{esc(team)}</span><small>条件 {esc(cond)}</small></h3><div class="six">{"".join(cells)}</div></div>'
        )
    return f'<div class="teams">{"".join(out)}</div>'


def _table(model: dict[str, Any]) -> str:
    rows = "".join(
        f'<tr><td><a href="#t{t["turn"]}">T{t["turn"]}</a></td>'
        f'<td class="n">{"—" if t["value"] is None else percent(t["value"])}</td>'
        f'<td class="n">{"—" if t["delta"] is None else signed(t["delta"], "pt")}</td></tr>'
        for t in model["turns"]
    )
    return f"<table><tr><th>ターン</th><th>席 0 から見た勝率</th><th>前のターンとの差</th></tr>{rows}</table>"


#: What each mark on a field move that fails means (the words the marks carry, one line each).
DEAD_MEANING = {
    narrow.DEAD: "すでに張ってある場を張り直す技で、失敗する（失敗を変える事情は見つからない）。",
    narrow.DEAD_BUT_DODGES_SUCKER_PUNCH: "失敗するが、相手のふいうちは変化技を選んだ相手に当たらないので、外す意味が残る。",
    narrow.DEAD_BUT_FEEDS_STOMPING_TANTRUM: "失敗するが、味方のじだんだ・テンパーフレア・メトロノームの威力を上げる。",
    narrow.DEAD_BUT_CHANGEABLE: "場を消す・変える技（かわらわり等）を相手が先に出せば、張り直しが成功する。",
    narrow.DEAD_IF_FIRST_ACTS: "同じターンの 2 体目の手。1 体目が止まらなければ失敗する（保険の組）。",
}


def dead_meaning(label: str) -> str:
    """The meaning of a mark's words (its hover text), from `DEAD_MEANING`."""
    for key, words in DEAD_TEXT.items():
        if words == label:
            return f"{label}: {DEAD_MEANING[key]}"
    return label


def _dead_legend(turns_html: str) -> str:
    """The legend of the marks, when the page has any (each mark's words with its meaning)."""
    used = [k for k, words in DEAD_TEXT.items() if f">{esc(words)}<" in turns_html]
    if not used:
        return ""
    items = "".join(
        f'<li><span class="tag dead">{esc(DEAD_TEXT[k])}</span> {esc(DEAD_MEANING[k])}</li>' for k in used
    )
    return (
        '<h2>失敗の札の意味</h2><ul class="dl-legend">'
        f'{items}</ul><p class="legend">候補には残してある（読みは変えていない）。札は局面から見た注記。</p>'
    )


# ----------------------------------------------------------------------------- a generation game


def _read_of_record(decision: dict[str, Any], side: int, chosen: str | None) -> dict[str, Any] | None:
    """One seat's read of a generation decision as `mixtures` reads it: the seat's own mixture over
    its menu (the record's ``ownActions``/``ownPolicy`` for seat 0, ``foeActions``/``foePolicy`` for
    seat 1). The record has no per-move values and not the model the seat held of the other's
    mixture, so ``opp`` is empty and ``cols`` absent (``generated`` says so)."""
    names = decision["ownActions" if side == 0 else "foeActions"]
    policy = decision["ownPolicy" if side == 0 else "foePolicy"]
    if not names or len(names) != len(policy):
        return None
    p = [float(v) for v in policy]
    order = sorted(range(len(p)), key=lambda k: -p[k])
    index = names.index(chosen) if chosen in names else None
    return {
        "menu": [len(names), len(decision["foeActions" if side == 0 else "ownActions"])],
        "rows": [[names[k], round(p[k], 4)] for k in order if p[k] >= 0.01][:8],
        "chosen": chosen,
        "chosenP": None if index is None else round(p[index], 4),
        "chosenRank": None if index is None else order.index(index) + 1,
        "opp": [],
        "oppSupp": [],
        "supp": [[names[k], round(p[k], 5)] for k in order if p[k] >= SUPPORT_MIN],
        "p": [round(v, 5) for v in p],
        "q": None,
        "generated": True,
    }


def selfplay_line(  # noqa: PLR0912 - one pass over the record's decisions
    reg: Regulation, loc: Localiser, record: dict[str, Any], node: Any,  # noqa: ANN401
    rereads: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """A generation game (a line of ``games-*.jsonl``) as the line `render_html` reads.

    What happened is `show_game.turn_trace` -- the port's re-resolution of each move decision, the
    branch matched to the next recorded position -- in the shape `timematch._note_turn` writes.
    Seat 0's read is ``searchValue`` and its mixture; seat 1's is ``foeSearchValue`` (side 0's
    units) and its mixture where the record has one (a hidden-bench game: each seat solved its own
    game). A record without ``foeSearchValue`` from an open game solved one game, so both seats
    read the same value; from a hidden-bench game, seat 1 then has no read (a dash). Nothing else
    is made up: no stage, clock, condition, seed or pair.

    ``rereads`` (`tools/reread.py`'s `reread_game`, by decision index): where a decision's
    re-solve matched the record, each seat's read is the re-solve's -- its mixture, the other
    side's mixture it modelled, the values of its moves -- and the page says it was solved again;
    where it did not, the record's mixture stays and the reason is shown."""
    decisions_in = record["decisions"]
    decisions: list[dict[str, Any]] = []
    moves: list[dict[str, Any]] = []
    hidden = record.get("information") == "hidden-bench"
    for i, d in enumerate(decisions_in):
        events = None
        reads: dict[str, Any] = {}
        if d["kind"] == "move":
            traced = show_game.turn_trace(reg, d, decisions_in[i + 1 :], record.get("outcome"), node)
            if isinstance(traced, show_game._Trace):
                cut = [traced.cut] if traced.cut else []
                events = {
                    "lines": list(traced.lines),
                    "acts": [[s, lbl] for s, lbl in traced.acts],
                    "paused": traced.paused or bool(traced.cut),
                    "matched": True,
                    "note": " ".join([*traced.notes, *cut]) or None,
                }
            elif traced:
                events = {
                    "lines": [], "acts": [], "paused": False, "matched": True,
                    "note": " ".join(text for _h, lines in traced for text in lines),
                }
            chosen = (d.get("ownChosen"), d.get("foeChosen"))
            foe_value = d.get("foeSearchValue")
            values = (d.get("searchValue"), foe_value if foe_value is not None else (None if hidden else d.get("searchValue")))
            again = None if rereads is None else rereads.get(i)
            for side in (0, 1):
                if values[side] is None:
                    continue
                if again is not None and again.matched:
                    got = again.reads[side]
                else:
                    got = _read_of_record(d, side, chosen[side])
                    if got is not None and rereads is not None:
                        got["rereadNote"] = "解き直していない" if again is None else again.reason
                if got is not None:
                    reads[str(side)] = got
                moves.append({"decision": i, "side": side, "value0": values[side], "generated": True})
        decisions.append({
            "kind": d["kind"], "turn": d["turn"], "position": d["position"],
            "ownChosen": d.get("ownChosen"), "foeChosen": d.get("foeChosen"),
            "value": d.get("searchValue"), "shown": None, "events": events, "reads": reads,
        })
    pool = record.get("pool") or {}
    names = pool.get("names")
    source = str(pool.get("id") or record.get("foeArchetype") or "?")
    if record.get("gameIndex") is not None:
        source += f"・局 {record['gameIndex']}"
    return {
        "generated": source,
        "transcript": {
            "ownSix": record["ownSix"], "foeSix": record["foeSix"],
            "ownPick": record["ownPick"], "foePick": record["foePick"],
            "ownTeam": record["ownTeam"], "foeTeam": record["foeTeam"],
            "decisions": decisions, "finalPosition": record.get("finalPosition"),
            "teamNames": names if names and len(names) == 2 else None,
        },
        "moves": moves,
        "outcome": record.get("outcome"), "endReason": record.get("endReason"),
        "turns": record.get("turns"), "conditions": ["—", "—"], "teams": None,
        "seed": None, "pair": None, "game": None, "adjudicated": None,
        "reread": None if rereads is None else {
            "matched": sum(1 for got in rereads.values() if got.matched),
            "total": sum(1 for d in decisions_in if d["kind"] == "move"),
        },
    }


def render_html(
    reg: Regulation, loc: Localiser, line: dict[str, Any], *, embed: bool = False, open_mix: bool = False
) -> str:
    """The page for one transcript line."""
    model = build(reg, loc, line)
    sprites = Sprites(reg, embed, open_mix)
    tr = line["transcript"]
    names = [[loc.species(sp) for sp in tr["ownSix"]], [loc.species(sp) for sp in tr["foeSix"]]]
    outcome = model["outcome"]
    reason = model["endReason"]
    #: The turns played: the cards' (the game line's ``turns`` is the number of the turn the
    #: game stood at when it stopped, one more than the last played when it did not end there).
    played = model["turns"][-1]["turn"] if model["turns"] else 0
    if outcome == 1.0:
        head, cls = "席 0 の勝ち", "w0"
    elif outcome == 0.0:
        head, cls = "席 1 の勝ち", "w1"
    else:
        head, cls = f"打ち切り（{played} ターン）", ""
    if outcome in (0.0, 1.0) and reason == "adjudicated":
        head += "（判定）"
    generated = bool(line.get("generated"))
    conditions = model["conditions"] or ["", ""]
    team = model.get("teamNames") or model["teams"] or ["", ""]
    vs = (
        f'<span class="seat s0">席 0</span> {esc(team[0])}（条件 {esc(conditions[0])}） 対 '
        f'<span class="seat s1">席 1</span> {esc(team[1])}（条件 {esc(conditions[1])}）'
    )
    if generated:
        extra = (
            f"生成の 1 局（{line['generated']}）・{played} ターン ・ {END_REASONS.get(reason, reason or '—')}"
            "・読みの段・時計・席の条件は記録にない（—）"
        )
        again = line.get("reread")
        if again is not None:
            extra += (
                f"・各手番の読みは、記録の局面と設定で解き直したもの（解き直した読み（記録の混合と一致を確かめた）: "
                f"{again['matched']} / {again['total']} 手番。一致しない手番は記録の混合だけを出す）"
            )
    else:
        extra = (
            f"対戦評価の 1 局（第 {model['pair']} 組の {int(model['game']) + 1} 局目）・乱数の種 {model['seed']} ・ "
            f"{played} ターン ・ {END_REASONS.get(reason, reason)}"
        )
    if model["turnCount"] is not None and model["turnCount"] != played:
        # The game line's `turns` is the turn number the game stood at when it stopped: one more
        # than the turns played, when the last of them ended it.
        extra += f"（局の記録の turns は終了時のターン番号 {model['turnCount']}）"
    if model["adjudicated"]:
        got = model["adjudicated"]
        extra += f"（T{got['turn']} の勝率 {percent(got['value'])}、席 0 から見て）"
    turns = "".join(_turn_html(t, model, sprites) for t in model["turns"])
    last = ""
    if reason != "wipeout" and model["turns"]:
        tail = model["turns"][-1]
        v = "" if tail["value"] is None else f"、その時点の勝率は席 0 から見て {percent(tail['value'])}"
        last = f'<div class="end"><b>ここで打ち切り</b>（{esc(END_REASONS.get(reason, reason))}）{v}。</div>'
    final = ""
    if model["final"]:
        sides = "".join(_side_html(s, model["final"][s], sprites) for s in (0, 1))
        title = (
            "終局の場"
            if outcome in (0.0, 1.0) and reason != "adjudicated"
            else (f"打ち切り時点の場（T{played + 1} の開始時）")
        )
        final = f'<h2>{title}</h2><div class="turn">{sides}</div>'
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(head)} — {"生成の 1 局" if generated else "対戦評価の 1 局"}</title><style>{CSS}</style></head><body><main>"
        f'<div class="banner {cls}"><h1>{esc(head)}</h1><p class="vs">{vs}</p><p>{esc(extra)}</p></div>'
        f"<h2>席 0 から見た勝率の推移</h2>{_chart(model)}"
        f"<h2>チーム（6 体・選出 4 体）</h2>{_teams(model, names, sprites)}"
        f'<h2>ターンごと</h2><p class="legend">各ターンの帯の ▼（橙）は席 0 の読み、▲（青）は席 1 の読み。'
        f"どちらも席 0 から見た勝率で、{SPLIT_PT}pt 以上離れたターンは間を塗って「読みが割れた」と書く。"
        f"見出しの ▲ は席 0 有利へ、▼ は席 1 有利への動き。"
        f"「均衡で打つ手」は、混合の確率が {SUPPORT_MIN * 100:g}% 以上の手（サポート）。"
        f"「候補」は行列に載せた手の数で、そのほとんどは確率がほぼ 0 の手。</p>{turns}{last}{final}"
        f"<h2>勝率の表</h2>{_table(model)}"
        f"{_dead_legend(turns)}"
        "</main></body></html>\n"
    )
