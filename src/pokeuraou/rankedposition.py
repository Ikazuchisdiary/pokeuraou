"""Reading a position typed in by hand: ranked stage 1b (IKA-408).

A ranked match shows the person the board and nothing else of the opponent: species, HP as a
per cent, what has happened. After the match they type the board in, turn by turn, and this
module turns it into what the analysis mode reads (`analysis.Analyzer.run`): a `Position`, the
two sheets (the person's team, the opponent's six as estimated), what each side has shown.

**The form is the state.** One turn is one plain dict (`Board.turns`): who is out, each seen
Pokemon's HP (the person's exactly, the opponent's as a per cent), status, boosts, Mega, item
gone, what the opponent has been seen to use (moves, item, ability), weather, terrain, screens,
hazards, and what happened on the way in (who moved first, damage taken). The next turn is a
copy with the turn number raised and the events cleared, so the person types only what changed.
Nothing here is tracked by the port: what the form says is believed (that is stage 4).

**The opponent's set is the system's job** (user, 10/1). The whole-set estimate is stage 1a's
(`rankedentry.FieldPrior`); every observation then narrows it:

* What was seen to be used -- moves, the item shown or spent, the ability that fired -- removes
  the field's sets that cannot have produced it (`rankedentry.SetBelief.observe`). Never what
  was *not* seen. If nothing in the field's sets is left, the seen things are added to the
  estimate and the card says so.
* Who moved first and the exact damage the person's Pokemon took narrow the opponent's *spread*,
  not its set: `observe.update` over a grid of the one to three stats the observation reads
  (Speed for an order, Attack or Special Attack for a hit), the other stats as estimated. The
  spread is then moved to the nearest value still possible and the total brought back to the
  limit. An observation no spread explains is reported and not applied.

The opponent's HP per cent is one HP, the middle of the values the game's own display rule
allows (`hpdisplay.band`); `hp_mode` takes the bottom or the top for measuring what that costs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .belief import SpreadBelief
from .hpdisplay import band, uses_floor_display
from .observe import DamageTaken, MovedFirst, update
from .position import Effect, Field, Position, Side, validate_position
from .priors import SampledSet
from .rankedentry import FieldPrior, Observation, OpponentSet, RankedError
from .regulation import BOOST_IDS, STAT_IDS, Regulation, to_id
from .selfplay import _make_pokemon
from .stats import nature_multipliers, stats_from_sp
from .teams import Roster

STATUSES = ("brn", "par", "slp", "frz", "psn", "tox")
WEATHERS = ("sunnyday", "raindance", "sandstorm", "snowscape")
TERRAINS = ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain")
#: Side conditions with turns left, and the turns a fresh one has.
TIMED = {"reflect": 5, "lightscreen": 5, "auroraveil": 5, "tailwind": 4}
#: Hazards with layers, and the most layers.
LAYERED = {"stealthrock": 1, "spikes": 3, "toxicspikes": 2, "stickyweb": 1}
HP_MODES = ("low", "mid", "high")
#: The stats an observation can narrow, by what reads them.
SPREAD_MOVES = {"bodypress": "def"}


# ----------------------------------------------------------------------------- the board


@dataclass
class Board:
    """The typed-in match: the person's team and four, the opponent's six and who led, and one
    form a turn."""

    reg: Regulation
    mine: Roster
    #: Roster indices of the person's four, the two leads first.
    brought: list[int]
    opp_six: list[str]
    opp_leads: list[str]
    #: Species of the opponent's four that were shown at the start (the leads, and anyone else
    #: the selection screen or the match has shown).
    opp_seen: list[str] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)

    def check(self) -> None:
        reg = self.reg
        n = reg.meta.team_size
        four = reg.meta.picked_team_size
        if len(self.brought) != four or len(set(self.brought)) != four:
            raise RankedError(f"自分の選出は {four} 体を重ならずに選んでください")
        if any(not 0 <= i < len(self.mine.sets) for i in self.brought):
            raise RankedError("自分の選出が構築の外にあります")
        if len(self.opp_six) != n:
            raise RankedError(f"相手は {n} 種族です")
        active = reg.meta.active_per_side
        if len(self.opp_leads) != active or len(set(self.opp_leads)) != active:
            raise RankedError(f"相手の先発は {active} 体を重ならずに選んでください")
        for sid in (*self.opp_leads, *self.opp_seen):
            if sid not in self.opp_six:
                raise RankedError("相手の先発・見えた体は、相手の 6 種族の中から選んでください")
        extra = {*self.opp_leads, *self.opp_seen}
        if len(extra) > four:
            raise RankedError(f"相手の見えた体が {len(extra)} 体です。選出は {four} 体までです")

    def seen_species(self, upto: int | None = None) -> list[str]:
        """The opponent's species shown by turn ``upto`` (all by default): the leads, the ones
        the person listed, and everyone who stood on the field in any turn so far."""
        out = [*self.opp_leads]
        out += [s for s in self.opp_seen if s not in out]
        for form in self.turns[: None if upto is None else upto + 1]:
            for sid in (*form["theirActive"], *form["theirs"]):
                if sid and sid not in out:
                    out.append(sid)
        return out

    def mine_seen(self, upto: int | None = None) -> list[int]:
        """Roster indices of the person's Pokemon the opponent has seen by turn ``upto``: the
        leads, and everyone who stood on the field or fell in any turn so far."""
        out = list(self.brought[:2])
        for form in self.turns[: None if upto is None else upto + 1]:
            for idx in form["mineActive"]:
                if idx is not None and idx not in out:
                    out.append(idx)
            for key, mon in form["mine"].items():
                if mon["hp"] == 0 and int(key) not in out:
                    out.append(int(key))
        return out


def max_hp(reg: Regulation, sset: SampledSet) -> int:
    stats = stats_from_sp(
        reg, np.array(reg.species[sset.species].base_stats),
        np.array([[sset.sp.get(s, 0) for s in STAT_IDS]]),
        nature_multipliers(reg, [sset.nature]), level=reg.meta.level)
    return int(stats[0, 0])


def _blank_mine(reg: Regulation, sset: SampledSet) -> dict[str, Any]:
    return {"hp": max_hp(reg, sset), "status": None, "boosts": {}, "mega": False, "itemGone": False}


def _blank_theirs() -> dict[str, Any]:
    return {"pct": 100, "colour": None, "status": None, "boosts": {}, "mega": False, "fainted": False,
            "moves": [], "item": None, "itemGone": False, "ability": None}


def blank_form(reg: Regulation, board: Board) -> dict[str, Any]:
    """Turn 1 before the leads' switch-ins: everyone full, nothing changed."""
    return {
        "turn": 1,
        "mineActive": list(board.brought[: reg.meta.active_per_side]),
        "theirActive": list(board.opp_leads),
        "mine": {str(i): _blank_mine(reg, board.mine.sets[i]) for i in board.brought},
        "theirs": {sid: _blank_theirs() for sid in board.seen_species(0)},
        "field": {"weather": None, "weatherTurns": 5, "terrain": None, "terrainTurns": 5, "trickRoom": 0},
        "sides": [{}, {}],
        "events": {"order": [], "damage": []},
    }


def next_form(form: dict[str, Any]) -> dict[str, Any]:
    """The next turn: a copy with the turn number raised and the events cleared. Timed things
    (weather, screens, Trick Room) lose a turn; the person corrects what the game shows."""
    out = json.loads(json.dumps(form))
    out["turn"] = int(form["turn"]) + 1
    out["events"] = {"order": [], "damage": []}
    f = out["field"]
    if f["weather"]:
        f["weatherTurns"] = max(1, int(f["weatherTurns"]) - 1)
    if f["terrain"]:
        f["terrainTurns"] = max(1, int(f["terrainTurns"]) - 1)
    if f["trickRoom"]:
        f["trickRoom"] = max(1, int(f["trickRoom"]) - 1)
    for side in out["sides"]:
        for cid in TIMED:
            if cid in side:
                side[cid] = max(1, int(side[cid]) - 1)
    return out


def normalize(reg: Regulation, board: Board, form: dict[str, Any]) -> dict[str, Any]:
    """The form with every field present and of the right kind, or a `RankedError` naming what
    is wrong. Unknown keys are dropped."""
    active_n = reg.meta.active_per_side

    def pick(value: Any, allowed: Sequence[str] | None, what: str) -> str | None:  # noqa: ANN401
        if value in (None, ""):
            return None
        value = to_id(str(value))
        if allowed is not None and value not in allowed:
            raise RankedError(f"{what} {value!r} を知りません")
        return value

    def integer(value: Any, low: int, high: int, what: str) -> int:  # noqa: ANN401
        try:
            n = int(value)
        except (TypeError, ValueError):
            raise RankedError(f"{what} は整数で入れてください") from None
        if not low <= n <= high:
            raise RankedError(f"{what} は {low}〜{high} です（{n}）")
        return n

    def boosts(raw: Any) -> dict[str, int]:  # noqa: ANN401
        out = {}
        for k, v in dict(raw or {}).items():
            if k not in BOOST_IDS:
                raise RankedError(f"能力変化 {k!r} を知りません")
            n = integer(v, -6, 6, "能力変化")
            if n:
                out[k] = n
        return out

    mine_in = dict(form.get("mine", {}))
    mine: dict[str, Any] = {}
    for i in board.brought:
        raw = dict(mine_in.get(str(i)) or _blank_mine(reg, board.mine.sets[i]))
        name = reg.species[board.mine.sets[i].species].name
        maxhp = _blank_mine(reg, board.mine.sets[i])["hp"]
        mine[str(i)] = {
            "hp": integer(raw.get("hp", maxhp), 0, maxhp, f"{name} の HP"),
            "status": pick(raw.get("status"), STATUSES, "状態異常"),
            "boosts": boosts(raw.get("boosts")),
            "mega": bool(raw.get("mega")),
            "itemGone": bool(raw.get("itemGone")),
        }
    active_mine = [None if x in (None, "") else int(x) for x in list(form.get("mineActive", []))[:active_n]]
    active_mine += [None] * (active_n - len(active_mine))
    for idx in active_mine:
        if idx is not None and idx not in board.brought:
            raise RankedError("出ている自分の体が、選んだ 4 体の中にありません")
        if idx is not None and mine[str(idx)]["hp"] == 0:
            name = reg.species[board.mine.sets[idx].species].name
            raise RankedError(f"{name} は倒れているので場に出せません")
    if len([x for x in active_mine if x is not None]) != len({x for x in active_mine if x is not None}):
        raise RankedError("同じ自分の体が 2 回出ています")

    theirs_in = dict(form.get("theirs", {}))
    active_theirs = [None if x in (None, "") else str(x)
                     for x in list(form.get("theirActive", []))[:active_n]]
    active_theirs += [None] * (active_n - len(active_theirs))
    seen = list(dict.fromkeys([*board.seen_species(len(board.turns) - 1 if board.turns else 0),
                               *[s for s in active_theirs if s], *theirs_in]))
    theirs: dict[str, Any] = {}
    for sid in seen:
        if sid not in board.opp_six:
            raise RankedError("相手の体は、相手の 6 種族の中から選んでください")
        raw = dict(theirs_in.get(sid) or _blank_theirs())
        name = reg.species[sid].name
        colour = raw.get("colour") or None
        if colour not in (None, "r", "y", "g"):
            raise RankedError("HP の色は r・y・g のどれかです")
        moves = [pick(m, reg.moves, "技") for m in (raw.get("moves") or [])]
        moves = list(dict.fromkeys(m for m in moves if m))
        if len(moves) > reg.meta.max_move_count:
            raise RankedError(f"{name} の見えた技が {reg.meta.max_move_count} 個を超えています")
        item = pick(raw.get("item"), reg.items, "持ち物")
        item_gone = bool(raw.get("itemGone"))
        if item_gone and item is None:
            raise RankedError(f"{name} の持ち物が使われた・落とされたなら、その持ち物の名前を入れてください")
        theirs[sid] = {
            **({"hpExact": int(raw["hpExact"])} if raw.get("hpExact") is not None else {}),
            "pct": integer(raw.get("pct", 100), 1, 100, f"{name} の HP（%）"),
            "colour": colour,
            "status": pick(raw.get("status"), STATUSES, "状態異常"),
            "boosts": boosts(raw.get("boosts")),
            "mega": bool(raw.get("mega")),
            "fainted": bool(raw.get("fainted")),
            "moves": moves,
            "item": item,
            "itemGone": item_gone,
            "ability": pick(raw.get("ability"), reg.abilities, "特性"),
        }
    for sid in active_theirs:
        if sid is not None and sid not in theirs:
            raise RankedError("出ている相手の体が見えた体の中にありません")
        if sid is not None and theirs[sid]["fainted"]:
            raise RankedError(f"{reg.species[sid].name} は倒れているので場に出せません")
    if len([x for x in active_theirs if x]) != len({x for x in active_theirs if x}):
        raise RankedError("同じ相手の体が 2 回出ています")

    fraw = dict(form.get("field") or {})
    weather = pick(fraw.get("weather"), WEATHERS, "天気")
    terrain = pick(fraw.get("terrain"), TERRAINS, "フィールド")
    field_out = {
        "weather": weather, "weatherTurns": integer(fraw.get("weatherTurns", 5), 1, 8, "天気の残りターン"),
        "terrain": terrain,
        "terrainTurns": integer(fraw.get("terrainTurns", 5), 1, 8, "フィールドの残りターン"),
        "trickRoom": integer(fraw.get("trickRoom", 0) or 0, 0, 5, "トリックルームの残りターン"),
    }
    sides = []
    for raw in list(form.get("sides") or [{}, {}])[:2]:
        side: dict[str, int] = {}
        for cid, value in dict(raw or {}).items():
            if cid in TIMED:
                if value:
                    side[cid] = integer(value, 1, 8, f"{cid} の残りターン")
            elif cid in LAYERED:
                if value:
                    side[cid] = integer(value, 1, LAYERED[cid], f"{cid} の枚数")
            else:
                raise RankedError(f"場の状態 {cid!r} を知りません")
        sides.append(side)
    sides += [{}] * (2 - len(sides))

    events = dict(form.get("events") or {})
    order = []
    for raw in events.get("order") or []:
        first, second = str(raw.get("first", "")), str(raw.get("second", ""))
        for ref in (first, second):
            if ref[:2] not in ("m:", "t:"):
                raise RankedError("先に動いた体の指定を読めません")
        if first == second:
            raise RankedError("先に動いた体と後に動いた体が同じです")
        order.append({"first": first, "second": second,
                      "firstMove": pick(raw.get("firstMove"), reg.moves, "技"),
                      "secondMove": pick(raw.get("secondMove"), reg.moves, "技")})
    damage = []
    for raw in events.get("damage") or []:
        attacker = str(raw.get("attacker", ""))
        if attacker not in board.opp_six:
            raise RankedError("ダメージを与えた相手の体を選んでください")
        move = pick(raw.get("move"), reg.moves, "技")
        if move is None:
            raise RankedError("ダメージを与えた技を選んでください")
        target = int(raw.get("target"))
        if target not in board.brought:
            raise RankedError("ダメージを受けた自分の体が選んだ 4 体の中にありません")
        damage.append({"attacker": attacker, "move": move, "target": target,
                       "amount": integer(raw.get("amount"), 1, 999, "受けたダメージ"),
                       "crit": bool(raw.get("crit"))})
    return {
        "turn": integer(form.get("turn", 1), 1, 99, "ターン"),
        "mineActive": active_mine, "theirActive": active_theirs,
        "mine": mine, "theirs": theirs, "field": field_out, "sides": sides,
        "events": {"order": order, "damage": damage},
    }


# ----------------------------------------------------------------------------- position


@dataclass
class Built:
    """A form as the analysis reads it."""

    position: Position
    #: Roster index -> the party slot, and species -> the party slot, on the built position.
    mine_slots: dict[int, int]
    opp_slots: dict[str, int]
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def hp_for(reg: Regulation, pct: int, colour: str | None, maxhp: int, mode: str = "mid") -> int:
    """One HP for a displayed per cent: the middle of the values the game's display rule allows
    (``low``/``high``: the ends)."""
    b = band(pct, maxhp, colour=colour, floor_rule=uses_floor_display(reg))
    if mode == "low":
        return b.low
    if mode == "high":
        return b.high
    return b.low + (b.width - 1) // 2


def _apply_mega(reg: Regulation, mon: Any, notes: list[str]) -> None:  # noqa: ANN401
    target = reg.mega_target(mon.species, mon.item)
    if target is None:
        raise RankedError(f"{reg.species[mon.species].name} は持ち物でメガシンカできません")
    found = reg.species[target]
    mon.species = target
    mon.types = tuple(found.types)
    mon.ability = to_id(found.abilities[0])
    mon.is_mega = True
    base = reg.species.get(to_id(mon.base_species))
    notes.append(f"{base.name if base is not None else mon.base_species}"
                 " はメガシンカ後として読みます")


def build_position(
    reg: Regulation, board: Board, form: dict[str, Any], sets: dict[str, SampledSet],
    *, hp_mode: str = "mid",
) -> Built:
    """The form as a `Position`: actives first on each side (so ``active`` is ``[0, 1]``), then
    the rest of the four. The opponent's four are the ones seen, then unseen ones from its six
    (they are completed again by the analysis, which weighs who the opponent would bring)."""
    four = reg.meta.picked_team_size
    notes: list[str] = []
    active_n = reg.meta.active_per_side

    # The person's side.
    order = [i for i in form["mineActive"] if i is not None]
    order += [i for i in board.brought if i not in order]
    mine_slots = {idx: slot for slot, idx in enumerate(order)}
    mons = []
    for idx, slot in mine_slots.items():
        sset = board.mine.sets[idx]
        state = form["mine"][str(idx)]
        active = form["mineActive"].index(idx) if idx in form["mineActive"] else None
        mon = _make_pokemon(reg, slot, sset, active)
        _fill(reg, mon, state, mon.maxhp, notes)
        mon.hp = int(state["hp"])
        mon.fainted = mon.hp == 0
        mons.append(mon)
    mine_side = _side("p1", "検討する側", form, 0, mons, reg)

    # The opponent's side.
    seen = list(dict.fromkeys([*[s for s in form["theirActive"] if s],
                               *[s for s in form["theirs"]]]))
    unseen = [s for s in board.opp_six if s not in seen]
    species = (seen + unseen)[:four]
    opp_slots = {sid: slot for slot, sid in enumerate(species)}
    mons = []
    for sid, slot in opp_slots.items():
        active = form["theirActive"].index(sid) if sid in form["theirActive"] else None
        mon = _make_pokemon(reg, slot, sets[sid], active)
        state = form["theirs"].get(sid)
        if state is not None:
            _fill(reg, mon, state, mon.maxhp, notes)
            if state["fainted"]:
                mon.hp = 0
                mon.fainted = True
            elif hp_mode == "exact" and state.get("hpExact") is not None:
                mon.hp = int(state["hpExact"])    # measuring only: the screen never has this
            else:
                mon.hp = hp_for(reg, state["pct"], state["colour"], mon.maxhp,
                                "mid" if hp_mode == "exact" else hp_mode)
            if state["itemGone"]:
                mon.item = None
        mons.append(mon)
    opp_side = _side("p2", "相手", form, 1, mons, reg)

    f = form["field"]
    pseudo = [Effect(id="trickroom", duration=int(f["trickRoom"]))] if f["trickRoom"] else []
    position = Position(
        format=reg.meta.format_id, sides=[mine_side, opp_side], turn=int(form["turn"]),
        field=Field(weather=f["weather"], weather_duration=f["weatherTurns"] if f["weather"] else None,
                    terrain=f["terrain"], terrain_duration=f["terrainTurns"] if f["terrain"] else None,
                    pseudo_weather=pseudo),
    )
    problems = validate_position(position, active_n)
    return Built(position, mine_slots, opp_slots, problems, notes)


def _fill(reg: Regulation, mon: Any, state: dict[str, Any], maxhp: int, notes: list[str]) -> None:  # noqa: ANN401
    mon.status = state["status"]
    mon.boosts = {k: int(v) for k, v in state["boosts"].items() if v}
    if state.get("itemGone"):
        mon.item = None
    if state["mega"]:
        item = mon.item or mon.base_item
        held = mon.item
        mon.item = item
        _apply_mega(reg, mon, notes)
        mon.item = held if held else item
    del maxhp


def _side(
    side_id: str, name: str, form: dict[str, Any], index: int, mons: list[Any], reg: Regulation,
) -> Side:
    active_n = reg.meta.active_per_side
    actives = []
    for slot in range(active_n):
        actives.append(next((m.slot for m in mons if m.active_index == slot), None))
    conds: list[Effect] = []
    for cid, value in form["sides"][index].items():
        if cid in TIMED:
            conds.append(Effect(id=cid, duration=int(value)))
        else:
            conds.append(Effect(id=cid, layers=int(value)))
    return Side(
        id=side_id, name=name, active=actives, pokemon=mons, side_conditions=conds,
        slot_conditions=[[] for _ in range(active_n)],
        mega_used=any(m.is_mega for m in mons),
        mega_capable_slots=[m.slot for m in mons if reg.mega_target(m.species, m.item) is not None],
    )


# ----------------------------------------------------------------------------- the opponent's set


@dataclass
class Refined:
    """One opposing Pokemon's estimate after what was seen of it."""

    one: OpponentSet
    notes: list[str] = field(default_factory=list)
    #: Field members behind the estimate before and after the observations (None: not from the
    #: field's sets, so nothing was counted).
    members: tuple[int, int] | None = None
    #: Nothing in the field's sets matches what was seen; the seen things were added to the set.
    unmatched: bool = False


def observation_of(state: dict[str, Any]) -> Observation:
    return Observation(moves=frozenset(state["moves"]), item=state["item"], ability=state["ability"])


def seen_union(board: Board, upto: int | None = None) -> dict[str, Observation]:
    """What each opposing Pokemon has been seen to use, over every turn: a thing seen once is
    seen (the person may have typed it into any turn)."""
    out: dict[str, Observation] = {}
    for form in board.turns[: None if upto is None else upto + 1]:
        for sid, state in form["theirs"].items():
            now = observation_of(state)
            out[sid] = out[sid].merged(now) if sid in out else now
    return out


def _force(reg: Regulation, one: SampledSet, seen: Observation) -> tuple[SampledSet, list[str]]:
    """The set with what was seen put in: the seen moves (the commonest others fill the rest),
    the seen item and ability."""
    notes: list[str] = []
    moves = [m for m in one.moves if m in seen.moves]
    moves += [m for m in seen.moves if m not in moves]
    moves += [m for m in one.moves if m not in moves]
    moves = moves[: reg.meta.max_move_count]
    ability, item = one.ability, one.item
    if seen.ability and to_id(ability or "") != seen.ability:
        ability = seen.ability
        notes.append(f"特性は見えた {reg.abilities[seen.ability].name} にしました")
    if seen.item and item != seen.item:
        item = seen.item
        notes.append(f"持ち物は見えた {reg.items[seen.item].name} にしました")
    if set(moves) != set(one.moves):
        notes.append("技に見えた技を入れました")
    return replace(one, moves=moves, ability=ability, item=item), notes


def refine_opponent(prior: FieldPrior, base: OpponentSet, seen: Observation) -> Refined:
    """The estimate for one opposing Pokemon given what has been seen of it (never what has
    not): the field's sets that could have produced it, the commonest of them; the person's own
    choice or typed set is kept when it is consistent with it."""
    reg = prior.reg
    if not (seen.moves or seen.item or seen.ability):
        return Refined(base)
    notes: list[str] = []
    members = None
    unmatched = False
    one = base
    if base.belief is not None and base.kind not in ("override", "manual"):
        narrowed = base.belief.observe(seen)
        members = (len(base.belief.members), len(narrowed.members))
        if not narrowed.members:
            unmatched = True
            notes.append("見えたものに合う型が大会データに無いので、見えたものを入れた型で読みます")
        elif not (base.kind == "candidate" and _consistent(base.set, seen)):
            one = prior.estimate(narrowed)
            if tuple(one.set.moves) != tuple(base.set.moves) or one.set.item != base.set.item:
                notes.append(f"見えたものから、大会の型 {members[0]} 体のうち {members[1]} 体に絞りました")
    chosen, more = _force(reg, one.set, seen)
    notes += more
    if chosen is not one.set:
        one = replace(one, set=chosen)
    return Refined(one, notes, members, unmatched)


def _consistent(one: SampledSet, seen: Observation) -> bool:
    if not seen.moves <= set(one.moves):
        return False
    if seen.item is not None and one.item != seen.item:
        return False
    return seen.ability is None or to_id(one.ability or "") == seen.ability


# ----------------------------------------------------------------------------- the spread


@dataclass
class SpreadNote:
    """What an observation did to one stat of one opposing Pokemon's spread."""

    species: str
    stat: str
    #: SP values still possible after everything seen, as (lowest, highest), and the stat value
    #: they give; the SP before and after the estimate was moved.
    low: int
    high: int
    stat_low: int
    stat_high: int
    before: int
    after: int
    because: list[str] = field(default_factory=list)


def _grid(
    reg: Regulation, one: SampledSet, axes: Sequence[str], allowed: dict[str, list[int]],
) -> SpreadBelief:
    cap = reg.meta.sp_per_stat_max
    ranges = [allowed.get(a) or list(range(cap + 1)) for a in axes]
    grid = np.array(list(itertools.product(*ranges)), dtype=np.int64).reshape(-1, len(axes))
    base = np.array([one.sp.get(s, 0) for s in STAT_IDS], dtype=np.int64)
    spreads = np.tile(base, (grid.shape[0], 1))
    for j, stat in enumerate(axes):
        spreads[:, STAT_IDS.index(stat)] = grid[:, j]
    stats = stats_from_sp(
        reg, np.array(reg.species[one.species].base_stats, dtype=np.int64), spreads,
        nature_multipliers(reg, [one.nature] * spreads.shape[0]), level=reg.meta.level)
    return SpreadBelief(
        species_id=one.species, nature=one.nature, spreads=spreads, stats=stats,
        weights=np.full(spreads.shape[0], 1.0 / spreads.shape[0]), provenance="a grid over what was seen")


def _axis_of_hit(reg: Regulation, move_id: str) -> str | None:
    found = reg.moves[move_id]
    if move_id in SPREAD_MOVES:
        return SPREAD_MOVES[move_id]
    if found.category == "Physical":
        return "atk"
    if found.category == "Special":
        return "spa"
    return None


def _slot(pos: Position, side: int, party: int) -> int | None:
    """The active slot (0, 1) of the Pokemon at party index ``party``, or None when it is not out."""
    for i, found in enumerate(pos.sides[side].active):
        if found is not None and found == party:
            return i
    return None


def _spread_hit(reg: Regulation, pos: Position, defender_slot: int, move_id: str) -> bool:
    """Whether a move hits more than one: the dex's target says it can, and two stand there."""
    kind = reg.moves[move_id].target
    if kind == "allAdjacentFoes":
        return sum(1 for p in pos.sides[0].active if p is not None) > 1
    return kind == "allAdjacent" and sum(
        1 for s in pos.sides for p in s.active if p is not None) > 2


def _rebalance(reg: Regulation, sp: dict[str, int], keep: set[str]) -> dict[str, int]:
    """The spread with its total brought back to the limit by taking points from the stats the
    observation did not read: HP first, then the defences, then the rest."""
    out = dict(sp)
    over = sum(out.values()) - reg.meta.sp_limit
    for stat in ("hp", "def", "spd", "atk", "spa", "spe"):
        if over <= 0:
            break
        if stat in keep:
            continue
        take = min(out[stat], over)
        out[stat] -= take
        over -= take
    for stat in ("spe", "atk", "spa", "def", "spd", "hp"):
        if over <= 0:
            break
        take = min(out[stat], over)
        out[stat] -= take
        over -= take
    return out


def narrow_spreads(
    reg: Regulation, board: Board, sets: dict[str, SampledSet],
    built_for: Callable[[int], Built],
) -> tuple[dict[str, SampledSet], list[SpreadNote], list[str]]:
    """Each opposing Pokemon's spread moved to what the events say: who moved first (Speed) and
    the damage the person's Pokemon took (Attack or Special Attack).

    The events of turn ``t`` happened in the position of turn ``t - 1`` (``built_for(t - 1)``).
    Returns the sets with the new spreads, a note per stat that was narrowed, and the lines the
    screen shows for what could not be used."""
    allowed: dict[str, dict[str, list[int]]] = {}
    because: dict[tuple[str, str], list[str]] = {}
    lines: list[str] = []
    for t in range(1, len(board.turns)):
        events = board.turns[t]["events"]
        if not (events["order"] or events["damage"]):
            continue
        built = built_for(t - 1)
        pos = built.position
        # What the observations of this turn say, as (observation, species, axis, label).
        found: list[tuple[Any, str, str, str]] = []
        for raw in events["order"]:
            kinds = {raw["first"][:2], raw["second"][:2]}
            if kinds == {"m:", "t:"}:
                mine_ref = raw["first"] if raw["first"][:2] == "m:" else raw["second"]
                their_ref = raw["first"] if raw["first"][:2] == "t:" else raw["second"]
                sid = their_ref[2:]
                if sid not in built.opp_slots or int(mine_ref[2:]) not in built.mine_slots:
                    lines.append(f"ターン {t}: 先に動いた順の体が前のターンに出ていないので使えません")
                    continue
                their_slot = _slot(pos, 1, built.opp_slots[sid])
                mine_slot = _slot(pos, 0, built.mine_slots[int(mine_ref[2:])])
                if their_slot is None or mine_slot is None:
                    lines.append(f"ターン {t}: 先に動いた順の体が前のターンに出ていないので使えません")
                    continue
                mine_is_first = raw["first"] == mine_ref
                first = (0, mine_slot) if mine_is_first else (1, their_slot)
                second = (1, their_slot) if mine_is_first else (0, mine_slot)
                obs = MovedFirst(first, second, first_move=raw["firstMove"], second_move=raw["secondMove"])
                label = "先に動いた" if not mine_is_first else "後に動いた"
                found.append((obs, sid, "spe", f"ターン {t}: {reg.species[sid].name} が自分の体より{label}"))
            else:
                lines.append(f"ターン {t}: 相手どうし・自分どうしの順は素早さの手がかりにしません")
        for raw in events["damage"]:
            sid, move = raw["attacker"], raw["move"]
            idx = raw["target"]
            axis = _axis_of_hit(reg, move)
            if sid not in built.opp_slots or idx not in built.mine_slots:
                lines.append(f"ターン {t}: ダメージの体が前のターンに出ていないので使えません")
                continue
            their_slot = _slot(pos, 1, built.opp_slots[sid])
            mine_slot = _slot(pos, 0, built.mine_slots[idx])
            if their_slot is None or mine_slot is None:
                lines.append(f"ターン {t}: ダメージの体が前のターンに出ていないので使えません")
                continue
            if axis is None:
                lines.append(f"ターン {t}: {reg.moves[move].name} は配分の手がかりになりません")
                continue
            capped = board.turns[t]["mine"][str(idx)]["hp"] == 0
            obs = DamageTaken(
                (1, their_slot), (0, mine_slot), move, int(raw["amount"]), crit=bool(raw["crit"]),
                spread=_spread_hit(reg, pos, mine_slot, move), capped=capped)
            found.append((obs, sid, axis, f"ターン {t}: {reg.species[sid].name} の {reg.moves[move].name} で "
                                          f"{raw['amount']} ダメージ"))
        # One grid per Pokemon over the stats its observations read.
        for sid in dict.fromkeys(f[1] for f in found):
            mine_obs = [f for f in found if f[1] == sid]
            axes = list(dict.fromkeys(f[2] for f in mine_obs))
            one = sets[sid]
            belief = _grid(reg, one, axes, allowed.get(sid, {}))
            slot = _slot(pos, 1, built.opp_slots[sid])
            # A fresh grid per observation so one that nothing explains does not spoil the rest.
            for obs, _sid, axis, label in mine_obs:
                key = (1, slot)
                after, report = update(reg, pos, {key: belief}, [obs])
                if report.skipped:
                    lines.append(f"{label}: 推定した型のどの配分でも起こりません。"
                                 "入力か、型（持ち物・特性）が違うかもしれません")
                    continue
                if report.uninformative or not report.entries:
                    lines.append(f"{label}: 配分の手がかりになりませんでした")
                    continue
                belief = after[key]
                col = STAT_IDS.index(axis)
                values = sorted({int(v) for v in belief.spreads[belief.weights > 0, col]})
                allowed.setdefault(sid, {})
                # The grid's other axes are still free; this axis's possible values narrow.
                every = range(reg.meta.sp_per_stat_max + 1)
                allowed[sid][axis] = [v for v in (allowed[sid].get(axis) or every) if v in values]
                because.setdefault((sid, axis), []).append(label)
    out = dict(sets)
    notes: list[SpreadNote] = []
    for sid, per in allowed.items():
        # A stat the events left free (every value possible) was not narrowed: nothing to say.
        per = {stat: v for stat, v in per.items() if len(v) <= reg.meta.sp_per_stat_max}
        if not per:
            continue
        one = out[sid]
        sp = dict(one.sp)
        keep = set(per)
        before = dict(sp)
        for stat, values in per.items():
            if not values:
                continue
            now = sp[stat]
            sp[stat] = min(values, key=lambda v: (abs(v - now), v))
        sp = _rebalance(reg, sp, keep)
        for stat, values in per.items():
            base = np.array(reg.species[sid].base_stats, dtype=np.int64)
            lo_hi = []
            for v in (min(values), max(values)):
                trial = dict(sp)
                trial[stat] = v
                lo_hi.append(int(stats_from_sp(
                    reg, base, np.array([[trial[s] for s in STAT_IDS]]),
                    nature_multipliers(reg, [one.nature]), level=reg.meta.level)[0, STAT_IDS.index(stat)]))
            notes.append(SpreadNote(sid, stat, min(values), max(values), lo_hi[0], lo_hi[1],
                                    before[stat], sp[stat], because.get((sid, stat), [])))
        out[sid] = replace(one, sp=sp)
    return out, notes, lines


# ----------------------------------------------------------------------------- deriving


@dataclass
class Derived:
    """Everything the screen and the read need from the board as it stands."""

    sets: dict[str, SampledSet]
    refined: dict[str, Refined]
    spread_notes: list[SpreadNote]
    lines: list[str]
    #: Per turn, the built position (the person's problems show on the turn they are in).
    built: list[Built]

    def roster_id(self) -> str:
        text = json.dumps(
            {s: [x.ability, x.item, x.nature, x.sp, x.moves] for s, x in sorted(self.sets.items())},
            sort_keys=True)
        return "ranked-opponent-" + hashlib.sha1(text.encode()).hexdigest()[:10]


def derive(
    reg: Regulation, board: Board, prior: FieldPrior, base: dict[str, OpponentSet], *,
    hp_mode: str = "mid", observe_spread: bool = True,
    choices: dict[str, tuple[Any, ...]] | None = None,
) -> Derived:
    """The opponent's sets after every observation of every turn, and each turn's position.

    ``choices`` are the person's picks among the remaining candidates, ``species -> (item, nature,
    moves)``: kept while that set is still one of the commonest the observations leave."""
    seen = seen_union(board)
    refined = {sid: refine_opponent(prior, one, seen.get(sid, Observation()))
               for sid, one in base.items()}
    for sid, key in (choices or {}).items():
        ref = refined.get(sid)
        if ref is None or ref.one.belief is None or ref.unmatched:
            continue
        for index, alt in enumerate(ref.one.alternatives):
            if (alt.item, alt.nature, alt.moves) == tuple(key):
                refined[sid] = Refined(prior.choose(ref.one, index), ref.notes, ref.members, False)
                break
    sets = {sid: r.one.set for sid, r in refined.items()}
    built_cache: dict[int, Built] = {}

    def built_for(t: int) -> Built:
        if t not in built_cache:
            built_cache[t] = build_position(reg, board, board.turns[t], sets, hp_mode=hp_mode)
        return built_cache[t]

    spread_notes: list[SpreadNote] = []
    lines: list[str] = []
    if observe_spread and len(board.turns) > 1:
        sets, spread_notes, lines = narrow_spreads(reg, board, sets, built_for)
        built_cache.clear()
    built = [built_for(t) for t in range(len(board.turns))]
    return Derived(sets, refined, spread_notes, lines, built)


def opponent_roster_of(reg: Regulation, board: Board, derived: Derived) -> Roster:
    """The opponent's six as the analysis reads them: the six species in order, each as
    estimated (and narrowed)."""
    sets = [derived.sets[sid] for sid in board.opp_six]
    return Roster(
        id=derived.roster_id(), name="相手（推定）", reg=reg,
        sets=[replace(s, moves=list(s.moves)) for s in sets],
        shown_stats=[None] * len(sets), source={"kind": "estimated", "event": "ranked-1b"},
    )


def initial_form(reg: Regulation, board: Board, sets: dict[str, SampledSet]) -> dict[str, Any]:
    """Turn 1 as the game shows it when the first turn starts: the leads' switch-in abilities
    have run (weather set, Intimidate taken), so those are put in for the person to correct."""
    from . import port

    form = normalize(reg, board, blank_form(reg, board))
    built = build_position(reg, board, form, sets)
    led = port.apply_lead_abilities(reg, built.position).position
    form["field"]["weather"] = led.field.weather
    form["field"]["weatherTurns"] = int(led.field.weather_duration or 5)
    form["field"]["terrain"] = led.field.terrain
    form["field"]["terrainTurns"] = int(led.field.terrain_duration or 5)
    for idx, slot in built.mine_slots.items():
        form["mine"][str(idx)]["boosts"] = {k: v for k, v in led.sides[0].pokemon[slot].boosts.items() if v}
    for sid, slot in built.opp_slots.items():
        if sid in form["theirs"]:
            form["theirs"][sid]["boosts"] = {k: v for k, v in led.sides[1].pokemon[slot].boosts.items() if v}
    return form
