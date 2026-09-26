"""One game against a person, on a clock (IKA-330, stage 1).

A person plays one side at the terminal and the agent plays the other. The game is
advanced on the port the way self-play advances it -- the turn's outcome weights drawn
once from the game's generator, a mid-turn switch answered where the port pauses, the
replacement phase after the turn -- and the person is shown only what their side can see:
the open team sheets, their own side exactly, the opponent's active Pokemon with HP as
the game displays it (`hpdisplay`), and of the opponent's bench only the Pokemon it has
shown. Which four the opponent brought stays hidden until they come out.

**The agent** is the one that ships in M-C, with a clock instead of a fixed width:

- its leaf (the M-C ensemble when it is there, hp-share otherwise, said at the start),
- its menus ranked by the leaf (``rank_by_leaf``), filled by ``q-nocover`` when a Q is
  available (IKA-274 stage 2; the user's decision of 9/26 for boards and human play) and
  by the default fill otherwise, with a printed note,
- its selection solved on the spot from the two sheets (`selection.solve_selection`, as
  `poolplay` does), drawn from the pure equilibrium, and its belief about the person's
  bench taken from the same solve,
- the hidden bench, as everything M-C plays.

**The clock** (`plan_move`, the rule IKA-322 is to replace). Per move decision the budget
is ``seconds``. The records say where time is worth spending: at large budgets width beats
depth (IKA-293: deepening arms lose about -25 to width 40 at depth 1), deepening the nodes
where a bench is still hidden brought a deepened width-12 agent level with d1@40
(IKA-294, the ``h`` labels), and width 48 against 24 is +3.4 (IKA-12). So the rule is
width first, depth with what is left:

1. The widest menu in `WIDTHS` whose predicted depth-1 node fits in `WIDTH_SHARE` of the
   budget. The prediction is `NODE_TIME` (a fixed cost plus a price per resolved cell --
   rows x columns x completions of the person's bench), measured here (IKA-330) per core
   count; an unmeasured count uses the nearest measured one below it, which predicts
   slow and so errs narrow.
2. The rest of the budget deepens the root best first, hidden nodes included (the ``m``
   reading with ``h``: `deepen.deepen_belief` / `deepen.deepen_root`).

**The deadline.** Two clocks, ``wall`` (the default) and ``count``:

- ``wall`` stops the deepening at the budget by the wall clock: the budget is handed to
  the deepening as milliseconds and `WallCost` reads the elapsed time where a `deepen.Cost`
  would read the counted work. The deepening is anytime -- the root is re-solved after
  every step -- so the answer at the deadline is the last one solved, and the overrun is
  the last step. If the menus and the depth-1 node already used the budget, no step is
  taken. A game on this clock depends on the machine's speed and is not replayed by seed.
- ``count`` spends the rest of the budget as `deepen.cells_for_seconds` counts it, with
  the measured prices (`deepen.COSTS`, 1 core only today): the same seed and the same
  person play the same game, byte for byte, on any machine.

Both clocks use the same width rule, which reads counts only, so the width of every move
is a function of the position. What the prediction cannot bound is the depth-1 node
itself; its time is written per move beside the budget (`clock` below).

**Records.** One line per game, in the shape `pool_match` writes (`GameRecord.to_json`,
`provenance`, the pool and the picks): each decision's two menus, the agent's mixture and
the agent's model of the person's (the reply its solve holds, averaged over completions by
the belief's weights), what each side played, and per move decision its ``plan`` (budget,
width, deepening budget and what the deepening did). The person's inputs are written as
the lines a script would give (``human.inputs``), so a game with a person replays from its
own record. The wall clock (``searchSeconds``, and per decision the seconds against the
budget) goes to a second file beside it (``<out>.clock.jsonl``), so the game file is the
game alone and two replays compare as bytes.

**Watching** (IKA-332). A `listener` -- ``listener(kind, payload)``, Python objects, nothing
serialised -- hears the game as it goes: ``sheets`` and ``select`` before the selection,
``board`` (the board as the person sees it) before every decision, ``think`` when the agent
starts a move, ``step`` with a `progress.Snapshot` of its answer as it forms (the first and
the last of each move always, the ones between at most every ``interval_ms``), ``answer``
when it has one (its mixture and value -- not the action it drew, which the person sees in
``turn`` after they have chosen), ``prompt`` when the person is asked, ``turn`` and ``end``.
The snapshots only read the search's tree, so a game on the count clock is the same game,
byte for byte, with a listener and without (`liveview` is the listener that shows it).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from . import port
from .actions import PassAction, SideAction, side_actions, switch_actions_after_faint, target_names
from .budget import Budget
from .deepen import COSTS, cells_for_seconds
from .equilibrium import EquilibriumError, solve_bayesian
from .hidden import (
    DEFAULT_BENCH_DROP,
    completions,
    identity,
    seen_identities,
    seen_slots,
    shown_species,
)
from .hpdisplay import displayed_percent, uses_floor_display
from .narrow import drop_dead_actions
from .payoff import HP_SHARE, Objective
from .position import Position
from .priors import SampledSet
from .progress import SLOT_SEPARATOR, Reader, Recorder, action_label
from .regulation import Regulation
from .search import belief_solve, search
from .selection_book import BenchPrior, BookEntry
from .selfplay import (
    MAX_TURNS,
    Decision,
    GameRecord,
    LeafEvaluator,
    _believed,
    _bench_weights,
    _close_record,
    _do_self_switch_node,
    _HiddenBench,
    _menus,
    _sample_index,
    _set_json,
    _shown_record,
    position_from_sets,
)
from .teams import Roster, pick_four_indices

# ----------------------------------------------------------------------------- the rule

#: The menu widths the rule chooses from, narrowest first. 48 is the human form IKA-32
#: timed; 64 is about every legal action of a doubles side (48 x 44 on average, IKA-293).
WIDTHS = (8, 12, 16, 24, 32, 48, 64)

#: The share of the budget the depth-1 node may be predicted to take. The rest deepens.
WIDTH_SHARE = 0.5

#: Below this many milliseconds left for it, a move does not deepen at all: one step of
#: the deepening is a fill of child matrices, a few to a few tens of milliseconds.
MIN_DEEPEN_MS = 20.0


@dataclass(frozen=True, slots=True)
class NodeTime:
    """What a depth-1 move node costs the agent, in wall milliseconds: its menus (the
    ranking included) and the fixed work of the node, plus a price per resolved cell."""

    fixed_ms: float
    cell_ms: float

    def ms(self, cells: int) -> float:
        return self.fixed_ms + self.cell_ms * cells


#: Measured (IKA-330 §3), by (machine form, port threads): least squares of each move
#: node's own time (menus through the depth-1 solve, no deepening: `--width-only`) on its
#: cells, M-C ensemble leaf on the local GPU, Q ranking.
NODE_TIME: dict[tuple[str, int], NodeTime] = {
    ("local", 1): NodeTime(fixed_ms=39.3, cell_ms=0.04176),
    ("local", 8): NodeTime(fixed_ms=13.9, cell_ms=0.02851),
}


def node_time(cores: int, form: str = "local") -> NodeTime:
    """`NODE_TIME` at ``cores``, or at the nearest measured count below it (slower: the
    rule then errs narrow). Below every measured count, the smallest."""
    measured = sorted(c for f, c in NODE_TIME if f == form)
    if not measured:
        raise ValueError(f"no node times for form {form!r}; measured: {sorted(NODE_TIME)}")
    below = [c for c in measured if c <= cores]
    return NODE_TIME[form, below[-1] if below else measured[0]]


@dataclass(frozen=True, slots=True)
class MovePlan:
    """How one move decision spends its budget."""

    budget_ms: float
    width: int
    #: Cells of the depth-1 node at that width: rows x columns x completions.
    cells: int
    predicted_ms: float
    #: What is left for the deepening, in milliseconds (0: no deepening).
    deepen_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "budgetMs": round(self.budget_ms, 3),
            "width": self.width,
            "nodeCells": self.cells,
            "predictedMs": round(self.predicted_ms, 3),
            "deepenMs": round(self.deepen_ms, 3),
        }


def plan_move(
    seconds: float,
    cores: int,
    rows: int,
    cols: int,
    classes: int,
    *,
    form: str = "local",
    width_only: bool = False,
) -> MovePlan:
    """Width first, depth with the rest (the module's docstring; IKA-322 replaces this).

    ``rows`` / ``cols`` are the agent's and the person's legal actions here, ``classes``
    the completions of the person's bench the agent believes (1 when nothing is hidden).
    The widest of `WIDTHS` whose node is predicted within `WIDTH_SHARE` of the budget, or
    the narrowest when none is; what the prediction leaves goes to the deepening, unless
    it is under `MIN_DEEPEN_MS` or ``width_only``.
    """
    budget_ms = max(0.0, seconds * 1000.0)
    price = node_time(cores, form)

    def cells_at(width: int) -> int:
        return min(width, rows) * min(width, cols) * max(classes, 1)

    width = WIDTHS[0]
    for candidate in WIDTHS:
        if price.ms(cells_at(candidate)) <= WIDTH_SHARE * budget_ms:
            width = candidate
        # Past every legal action on both sides a wider menu is the same menu.
        if candidate >= rows and candidate >= cols:
            break
    predicted = price.ms(cells_at(width))
    left = budget_ms - predicted
    deepen_ms = 0.0 if width_only or left < MIN_DEEPEN_MS else left
    return MovePlan(budget_ms, width, cells_at(width), predicted, deepen_ms)


class WallCost:
    """A `deepen.Cost` that reads the wall clock instead of the counted work.

    `deepen._Meter.spent` is ``cost.ms(fills, refines, cells, probed, qs) / cost.cell``;
    with ``cell`` 1 it is the milliseconds since ``start``, so a deepening given the budget
    in milliseconds stops at the deadline, one step late at most. The counts are ignored:
    the clock already holds whatever they cost (IKA-322 added ``probed`` and ``qs``).
    """

    cell = 1.0

    def __init__(self, start: float) -> None:
        self.start = start

    def ms(  # noqa: ARG002
        self, fills: int, refines: int, cells: int, probed: int = 0, qs: int = 0
    ) -> float:
        return (time.perf_counter() - self.start) * 1000.0


CLOCKS = ("wall", "count")

# ----------------------------------------------------------------------------- the person


class Person:
    """Whoever plays the other side. Two questions: which four (ordered, the first two
    lead), and which of the legal actions. Each answer is also returned as the line a
    script would give, which the record keeps (`ScriptPerson` reads it back)."""

    kind = "person"

    def select(self, six: Sequence[SampledSet], size: int, text: str) -> tuple[int, ...]:
        raise NotImplementedError

    def choose(self, kind: str, legal: Sequence[SideAction], text: str) -> SideAction:
        raise NotImplementedError


def selection_line(pick: Sequence[int]) -> str:
    """A selection as a script line: the party numbers, 1-based, leads first."""
    return " ".join(str(i + 1) for i in pick)


def parse_selection(line: str, team_size: int, size: int) -> tuple[int, ...]:
    """``"3 1 5 6"`` -> (2, 0, 4, 5). Stops on anything but ``size`` distinct numbers."""
    parts = line.replace(",", " ").split()
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"a selection is {size} party numbers, not {line!r}") from None
    if len(numbers) != size or len(set(numbers)) != size:
        raise ValueError(f"a selection is {size} different party numbers, not {line!r}")
    if any(n < 1 or n > team_size for n in numbers):
        raise ValueError(f"party numbers run from 1 to {team_size}: {line!r}")
    return tuple(n - 1 for n in numbers)


def parse_choice(line: str, legal: Sequence[SideAction]) -> SideAction:
    """A choice line: a number (1-based, into ``legal``) or the choice string itself
    (``"move 1 1, switch 3"``, as `SideAction.to_choice` writes it)."""
    text = line.strip()
    if text.isdigit():
        index = int(text) - 1
        if not 0 <= index < len(legal):
            raise ValueError(f"{text} is not one of 1..{len(legal)}")
        return legal[index]
    wanted = ", ".join(" ".join(part.split()) for part in text.split(","))
    for action in legal:
        if action.to_choice() == wanted:
            return action
    raise ValueError(
        f"{text!r} is not legal here; legal: {[a.to_choice() for a in legal]}"
    )


class ScriptPerson(Person):
    """Answers from lines, in order: a selection line, then one choice line per question.
    Blank lines and lines starting with ``#`` are skipped. Running out stops the game."""

    kind = "script"

    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        self.at = 0

    @classmethod
    def from_file(cls, path: Path) -> ScriptPerson:
        return cls(path.read_text(encoding="utf-8").splitlines())

    @classmethod
    def from_record(cls, path: Path, line: int = 0) -> ScriptPerson:
        """The inputs of the ``line``-th game of a record file: that person, again."""
        rows = [r for r in path.read_text(encoding="utf-8").splitlines() if r.strip()]
        return cls(json.loads(rows[line])["human"]["inputs"])

    def _next(self) -> str:
        if self.at >= len(self.lines):
            raise ValueError(f"the script ran out after {self.at} answers")
        self.at += 1
        return self.lines[self.at - 1]

    def select(self, six: Sequence[SampledSet], size: int, text: str) -> tuple[int, ...]:  # noqa: ARG002
        return parse_selection(self._next(), len(six), size)

    def choose(self, kind: str, legal: Sequence[SideAction], text: str) -> SideAction:  # noqa: ARG002
        return parse_choice(self._next(), legal)


class PolicyPerson(Person):
    """A stand-in for a person: ``first`` takes the first four and the first legal action,
    ``random`` draws from its own generator (so the game's own stream is untouched)."""

    def __init__(self, policy: str, seed: int = 0) -> None:
        if policy not in ("first", "random"):
            raise ValueError(f"a policy person is first or random, not {policy!r}")
        self.kind = policy
        self.rng = np.random.default_rng([seed, 330])

    def select(self, six: Sequence[SampledSet], size: int, text: str) -> tuple[int, ...]:  # noqa: ARG002
        if self.kind == "first":
            return tuple(range(size))
        return tuple(int(i) for i in self.rng.permutation(len(six))[:size])

    def choose(self, kind: str, legal: Sequence[SideAction], text: str) -> SideAction:  # noqa: ARG002
        if self.kind == "first":
            return legal[0]
        return legal[int(self.rng.integers(len(legal)))]


class TerminalPerson(Person):
    """A person at the terminal. A move is asked slot by slot; at any prompt a whole choice
    string (or ``#<n>`` for the n-th legal action) is taken too."""

    kind = "terminal"

    def __init__(
        self,
        reg: Regulation,
        loc: Any = None,  # noqa: ANN401 - names.Localiser or None
        *,
        read: Callable[[str], str] | None = None,
        out: TextIO | None = None,
    ) -> None:
        self.reg = reg
        self.loc = loc
        self.out = out or sys.stdout
        self._read = read

    def _ask(self, prompt: str) -> str:
        if self._read is not None:
            self.out.write(prompt)
            return self._read(prompt)
        return input(prompt)

    def select(self, six: Sequence[SampledSet], size: int, text: str) -> tuple[int, ...]:
        self.out.write(text + "\n")
        while True:
            line = self._ask(f"選出 {size} 体を番号で（先の 2 体が先発。例: 1 2 3 4）> ")
            try:
                return parse_selection(line, len(six), size)
            except ValueError as problem:
                self.out.write(f"  {problem}\n")

    def choose(self, kind: str, legal: Sequence[SideAction], text: str) -> SideAction:
        self.out.write(text + "\n")
        if len(legal) == 1:
            self.out.write(
                f"  選べるのは 1 つだけ: {legal[0].describe(self.reg, self.loc, self._targets)}\n"
            )
            return legal[0]
        remaining = list(legal)
        slots = len(legal[0].slots)
        for slot in range(slots):
            options: list[Any] = []
            for action in remaining:
                if all(action.slots[slot].to_choice() != o.to_choice() for o in options):
                    options.append(action.slots[slot])
            if len(options) == 1:
                continue
            for n, option in enumerate(options, 1):
                self.out.write(f"  {n:>2}. {option.describe(self.reg, self.loc, self._targets)}\n")
            while True:
                line = self._ask(f"{slot + 1} 体目の行動を番号で（行動の文字列・#n も可）> ").strip()
                if line.startswith("#") or not line.isdigit():
                    try:
                        return parse_choice(line.lstrip("#"), legal)
                    except ValueError as problem:
                        self.out.write(f"  {problem}\n")
                        continue
                index = int(line) - 1
                if 0 <= index < len(options):
                    picked = options[index].to_choice()
                    remaining = [a for a in remaining if a.slots[slot].to_choice() == picked]
                    break
                self.out.write(f"  1〜{len(options)} で\n")
        return remaining[0]

    #: Who each target number names, set by the game before each question.
    _targets: Any = None


# ----------------------------------------------------------------------------- the view


def _name(loc: Any, kind: str, value: str | None) -> str:  # noqa: ANN401
    if loc is None:
        return str(value)
    return getattr(loc, kind)(value)


def render_sheet(reg: Regulation, six: Sequence[SampledSet], loc: Any = None) -> str:  # noqa: ANN401
    """A team sheet: what M-C shows of both sides before the selection (spreads too)."""
    lines = []
    for n, s in enumerate(six, 1):
        moves = " / ".join(_name(loc, "move", m) for m in s.moves)
        sp = "-".join(str(s.sp.get(k, 0)) for k in ("hp", "atk", "def", "spa", "spd", "spe"))
        lines.append(
            f"  {n}. {_name(loc, 'species', s.species)} @ {_name(loc, 'item', s.item)}"
            f" / {_name(loc, 'ability', s.ability)} / {_name(loc, 'nature', s.nature)}"
            f" / SP {sp}\n       {moves}"
        )
    del reg
    return "\n".join(lines)


def _boosts(boosts: dict[str, int]) -> str:
    shown = [f"{k}{v:+d}" for k, v in boosts.items() if v]
    return f" [{' '.join(shown)}]" if shown else ""


def render_position(
    reg: Regulation,
    pos: Position,
    me: int,
    foe_seen: frozenset[str],
    loc: Any = None,  # noqa: ANN401
) -> str:
    """The board as side ``me`` sees it: its own side exactly, the other side's active
    Pokemon and the ones it has shown (`foe_seen`, identities) with HP as displayed, the
    rest of its four only as a count."""
    floor = uses_floor_display(reg)
    foe = 1 - me
    out = [f"=== ターン {pos.turn} ==="]
    field_bits = []
    if pos.field.weather:
        field_bits.append(f"天候 {pos.field.weather}")
    if pos.field.terrain:
        field_bits.append(f"フィールド {pos.field.terrain}")
    field_bits += [e.id for e in pos.field.pseudo_weather]
    if field_bits:
        out.append("  場: " + " / ".join(field_bits))

    def mon_line(mon: Any, exact: bool) -> str:  # noqa: ANN401
        name = _name(loc, "species", mon.species)
        if mon.fainted:
            return f"{name} ひんし"
        if exact:
            hp = f"HP {mon.hp}/{mon.maxhp}"
        else:
            hp = f"HP {displayed_percent(mon.hp, mon.maxhp, floor_rule=floor)}%"
        status = f" {_name(loc, 'status', mon.status)}" if mon.status else ""
        item = f" @ {_name(loc, 'item', mon.item)}" if mon.item else ""
        vol = [e.id for e in mon.volatiles]
        return f"{name} {hp}{status}{_boosts(mon.boosts)}{item}" + (
            f" ({', '.join(vol)})" if vol else ""
        )

    side = pos.sides[foe]
    conds = [e.id for e in side.side_conditions]
    out.append("相手" + (f"（{', '.join(conds)}）" if conds else "") + ":")
    for mon in side.active_pokemon():
        if mon is not None:
            out.append(f"  [場] {mon_line(mon, exact=False)}")
    hidden = 0
    for mon in side.pokemon:
        if mon.active_index is not None:
            continue
        if identity(mon) in foe_seen:
            out.append(f"  [控] {mon_line(mon, exact=False)}")
        else:
            hidden += 1
    if hidden:
        out.append(f"  [控] まだ見ていない {hidden} 体")
    side = pos.sides[me]
    conds = [e.id for e in side.side_conditions]
    out.append("自分" + (f"（{', '.join(conds)}）" if conds else "") + ":")
    for mon in side.active_pokemon():
        if mon is not None:
            moves = ", ".join(f"{_name(loc, 'move', m.id)} {m.pp}/{m.maxpp}" for m in mon.moves)
            out.append(f"  [場] {mon_line(mon, exact=True)}\n        {moves}")
    for mon in side.pokemon:
        if mon.active_index is None:
            out.append(f"  [控] {mon_line(mon, exact=True)}")
    return "\n".join(out)


def sprite_id(reg: Regulation, species: str) -> str:
    """Showdown's sprite id: the base species' id, and ``-`` and the forme's id if there is
    one (``urshifu-rapidstrike``, ``charizard-megay``)."""
    from .regulation import to_id

    found = reg.species.get(to_id(species))
    if found is None:
        return to_id(species)
    base = to_id(found.base_species)
    return f"{base}-{to_id(found.forme)}" if found.forme else base


def species_types(reg: Regulation, species: str, current: Sequence[str] = ()) -> list[str]:
    """The types as lower-case ids: ``current`` when a battle changed them, else the dex's."""
    from .regulation import to_id

    if current:
        return [to_id(t) for t in current]
    found = reg.species.get(to_id(species))
    return [to_id(t) for t in found.types] if found is not None else []


def sheet_view(
    six: Sequence[SampledSet], loc: Any = None, reg: Regulation | None = None  # noqa: ANN401
) -> list[dict[str, Any]]:
    """A team sheet as plain values (what `render_sheet` prints), for a screen."""
    return [
        {
            "species": _name(loc, "species", s.species),
            **(
                {"id": sprite_id(reg, s.species), "types": species_types(reg, s.species)}
                if reg is not None else {}
            ),
            "item": _name(loc, "item", s.item),
            "ability": _name(loc, "ability", s.ability),
            "nature": _name(loc, "nature", s.nature),
            "sp": [int(s.sp.get(k, 0)) for k in ("hp", "atk", "def", "spa", "spd", "spe")],
            "moves": [_name(loc, "move", m) for m in s.moves],
        }
        for s in six
    ]


def _effect_name(loc: Any, effect: str) -> str:  # noqa: ANN401
    """A weather, terrain or condition by the move that makes it, where Showdown's text
    has one (its id is the move's), else its id."""
    if loc is None:
        return effect
    got = loc.move(effect)
    return got if got else effect


def board_view(
    reg: Regulation,
    pos: Position,
    viewer: int,
    foe_seen: frozenset[str],
    loc: Any = None,  # noqa: ANN401
    *,
    names: tuple[str, str] = ("側0", "側1"),
    seconds: float | None = None,
) -> dict[str, Any]:
    """`render_position` as plain values: the board as side ``viewer`` sees it -- its own
    side exactly, the other side's active Pokemon and the ones it has shown with HP as the
    game displays it, the rest only counted."""
    floor = uses_floor_display(reg)

    def mon_view(mon: Any, exact: bool) -> dict[str, Any]:  # noqa: ANN401
        view: dict[str, Any] = {
            "species": _name(loc, "species", mon.species),
            "id": sprite_id(reg, mon.species),
            "types": species_types(reg, mon.species, mon.types),
            "fainted": bool(mon.fainted),
            "percent": displayed_percent(mon.hp, mon.maxhp, floor_rule=floor),
            "status": _name(loc, "status", mon.status) if mon.status else None,
            "statusId": mon.status or None,
            "boosts": {k: v for k, v in mon.boosts.items() if v},
            "item": _name(loc, "item", mon.item) if mon.item else None,
            "volatiles": [e.id for e in mon.volatiles],
        }
        if exact:
            view["hp"] = [mon.hp, mon.maxhp]
            view["moves"] = [[_name(loc, "move", m.id), m.pp, m.maxpp] for m in mon.moves]
        return view

    sides = []
    for index in (0, 1):
        side = pos.sides[index]
        exact = index == viewer
        active = [
            None if mon is None else mon_view(mon, exact) for mon in side.active_pokemon()
        ]
        bench = []
        hidden = 0
        for mon in side.pokemon:
            if mon.active_index is not None:
                continue
            if exact or identity(mon) in foe_seen:
                bench.append(mon_view(mon, exact))
            else:
                hidden += 1
        sides.append({
            "name": names[index],
            "conditions": [_effect_name(loc, e.id) for e in side.side_conditions],
            "active": active,
            "bench": bench,
            "hidden": hidden,
        })
    field_bits = []
    if pos.field.weather:
        field_bits.append(_effect_name(loc, pos.field.weather))
    if pos.field.terrain:
        field_bits.append(_effect_name(loc, pos.field.terrain))
    field_bits += [_effect_name(loc, e.id) for e in pos.field.pseudo_weather]
    return {
        "turn": pos.turn,
        "viewer": viewer,
        "field": field_bits,
        "sides": sides,
        "ended": bool(pos.ended),
        "seconds": seconds,
    }


def turn_changes(
    reg: Regulation,
    before: Position,
    after: Position,
    viewer: int,
    loc: Any = None,  # noqa: ANN401
    *,
    names: tuple[str, str] = ("側0", "側1"),
) -> list[dict[str, Any]]:
    """What a turn did, as side ``viewer`` could see it: per Pokemon whose displayed HP,
    faint, status or place changed, the HP as displayed before and after (the other side's
    as the game shows it, a percentage), whether it fainted, a new status, and whether it
    came in. The other side's Pokemon it has not seen stay out unless they came in."""
    floor = uses_floor_display(reg)
    out = []
    for side in (0, 1):
        old = {identity(m): m for m in before.sides[side].pokemon}
        for mon in after.sides[side].pokemon:
            prev = old.get(identity(mon))
            if prev is None:
                continue
            came = mon.active_index is not None and prev.active_index is None
            if side != viewer and prev.active_index is None and not came:
                continue
            was = displayed_percent(prev.hp, prev.maxhp, floor_rule=floor)
            now = displayed_percent(mon.hp, mon.maxhp, floor_rule=floor)
            fainted = mon.fainted and not prev.fainted
            status = (
                mon.status
                if mon.status and mon.status != prev.status and not mon.fainted
                and mon.status != "fnt"
                else None
            )
            if was == now and not fainted and not status and not came:
                continue
            out.append({
                "side": side,
                "name": names[side],
                "species": _name(loc, "species", mon.species),
                "id": sprite_id(reg, mon.species),
                "from": was,
                "to": now,
                **({"hp": [prev.hp, mon.hp, mon.maxhp]} if side == viewer else {}),
                "fainted": fainted,
                "status": _name(loc, "status", status) if status else None,
                "statusId": status,
                "entered": came,
            })
    return out


# ----------------------------------------------------------------------------- the agent


@dataclass
class Agent:
    """The side the program plays: leaf, menus, belief and clock."""

    reg: Regulation
    evaluate: LeafEvaluator | None
    name: str
    seconds: float
    cores: int = 1
    clock: str = "wall"
    form: str = "local"
    rank_fill: str = "q-nocover"
    rank_by_leaf: bool = True
    bench_drop: str = DEFAULT_BENCH_DROP
    objective: Objective = HP_SHARE
    #: No deepening: the width rule alone (a baseline, and how `NODE_TIME` is measured).
    width_only: bool = False
    #: The deepening's depth guard (IKA-307, a label's ``g<L>``), or None: `MAX_LEVELS`.
    #: Given, each move's record says why the deepening and its lines stopped.
    max_levels: int | None = None
    #: The children's menus by the Q's k best (IKA-307, ``c<k>``), or None: `narrow`'s.
    child_q: int | None = None

    def __post_init__(self) -> None:
        if self.clock not in CLOCKS:
            raise ValueError(f"clock is one of {CLOCKS}, not {self.clock!r}")
        if self.clock == "count" and not self.width_only and (self.form, self.cores) not in COSTS:
            raise ValueError(
                f"the count clock spends the budget at measured prices, and there are none "
                f"for {self.form!r} on {self.cores} core(s) (deepen.COSTS: {sorted(COSTS)}); "
                "use the wall clock there"
            )

    @property
    def leaf(self) -> LeafEvaluator:
        return self.evaluate if self.evaluate is not None else self.objective.batch


def legal_count(reg: Regulation, pos: Position, side: int) -> int:
    """The actions `narrow` ranks for that side: every legal one less the dead ones (the
    same list as `qhead.legal_pool`)."""
    return len(drop_dead_actions(reg, pos, side, side_actions(reg, pos, side)))


def _not_asked(positions: list[Position]) -> np.ndarray:  # pragma: no cover - never called
    raise AssertionError("the person's side is never solved")


_HEADINGS = {
    "move": "行動を選ぶ",
    "replacement": "交代先を選ぶ",
    "selfswitch": "交代先を選ぶ（技の効果）",
}


def _averaged(replies: Sequence[np.ndarray], weights: Sequence[float]) -> list[float]:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    return [float(x) for x in sum(wk * np.asarray(r) for wk, r in zip(w, replies, strict=True))]


class HumanGame:
    """One game: the agent at side ``agent_side``, the person at the other."""

    def __init__(
        self,
        agent: Agent,
        person: Person,
        *,
        agent_side: int,
        sheets: tuple[Sequence[SampledSet], Sequence[SampledSet]],
        picks: tuple[tuple[int, ...], tuple[int, ...]],
        bench_prior: tuple[BenchPrior | None, BenchPrior | None] | None,
        rng: np.random.Generator,
        max_turns: int = MAX_TURNS,
        loc: Any = None,  # noqa: ANN401
        out: TextIO | None = None,
        listener: Callable[[str, Any], None] | None = None,
        interval_ms: float = 100.0,
    ) -> None:
        self.agent = agent
        self.reg = agent.reg
        self.person = person
        self.me = agent_side
        self.you = 1 - agent_side
        self.sheets = sheets
        self.picks = picks
        self.bench_prior = bench_prior
        self.rng = rng
        self.max_turns = max_turns
        self.loc = loc
        self.out = out
        self.inputs: list[str] = []
        self.clock: list[dict[str, Any]] = []
        self.extras: dict[int, dict[str, Any]] = {}
        self.leaves = (agent.evaluate, agent.evaluate)
        self.listener = listener
        self.interval_ms = interval_ms
        #: How the two sides are named on the screen, by side index.
        self.side_names = tuple("AI" if s == agent_side else "あなた" for s in (0, 1))

    # -- output
    def say(self, text: str) -> None:
        if self.out is not None:
            self.out.write(text + "\n")
            self.out.flush()

    def emit(self, kind: str, payload: Any) -> None:  # noqa: ANN401
        if self.listener is not None:
            self.listener(kind, payload)

    def _ask(self, kind: str, legal: Sequence[SideAction], pos: Position, seen: Any) -> SideAction:  # noqa: ANN401
        text = render_position(self.reg, pos, self.you, seen[self.me], self.loc)
        if isinstance(self.person, TerminalPerson):
            self.person._targets = target_names(pos, self.you)
        if self.listener is not None:
            targets = target_names(pos, self.you)
            self.emit("prompt", {
                "kind": kind,
                "heading": _HEADINGS.get(kind, kind),
                "turn": pos.turn,
                "choices": [a.to_choice() for a in legal],
                # Slot k of every action is the person's k-th active Pokemon (None: empty).
                "actives": [
                    None if mon is None else {
                        "species": _name(self.loc, "species", mon.species),
                        "id": sprite_id(self.reg, mon.species),
                    }
                    for mon in pos.sides[self.you].active_pokemon()
                ],
                "slots": [
                    [
                        (s.to_choice(), s.describe(self.reg, self.loc, targets))
                        for s in a.slots
                    ]
                    for a in legal
                ],
            })
        got = self.person.choose(kind, legal, f"{text}\n-- {_HEADINGS.get(kind, kind)}")
        self.inputs.append(got.to_choice())
        return got

    # -- the game
    def play(self) -> GameRecord:
        reg = self.reg
        four = [
            [self.sheets[side][i] for i in self.picks[side]] for side in (0, 1)
        ]
        record = GameRecord(
            own_team=[_set_json(reg, s) for s in four[0]],
            foe_team=[_set_json(reg, s) for s in four[1]],
            foe_archetype="human",
        )
        record.own_six = [s.species for s in self.sheets[0]]
        record.foe_six = [s.species for s in self.sheets[1]]
        record.own_pick = list(self.picks[0])
        record.foe_pick = list(self.picks[1])
        record.information = "hidden-bench"
        record.ranking = "leaf" if self.agent.rank_by_leaf else "damage"
        record.rank_fill = [self.agent.rank_fill, self.agent.rank_fill]
        record.bench_drop = [self.agent.bench_drop, self.agent.bench_drop]
        self.record = record
        pos = position_from_sets(reg, four[0], four[1], rng=self.rng)
        seen: list[frozenset[str]] = [frozenset(), frozenset()]
        leads: list[frozenset[str] | None] = [
            frozenset(
                shown_species(
                    pos, i,
                    frozenset(m.slot for m in pos.sides[i].pokemon if m.active_index is not None),
                )
            )
            for i in (0, 1)
        ]
        record.leads = [
            [
                identity(m)
                for m in sorted(
                    (m for m in pos.sides[i].pokemon if m.active_index is not None),
                    key=lambda m: m.active_index,
                )
            ]
            for i in (0, 1)
        ]
        for _step in range(self.max_turns * 2):
            if pos.ended:
                break
            seen = [seen_identities(pos, i, seen[i]) for i in (0, 1)]
            shown = [seen_slots(pos, i, seen[i]) for i in (0, 1)]
            recorded_shown = _shown_record(pos, seen)
            if self.listener is not None:
                self.emit("board", board_view(
                    reg, pos, self.you, seen[self.me], self.loc, names=self.side_names,
                    seconds=self.agent.seconds,
                ))
            owed = port.replacements_needed(reg, pos)
            if any(owed[0]) or any(owed[1]):
                pos = self._replacement(pos, owed, seen, shown, leads, recorded_shown)
                continue
            try:
                spreads = {
                    side: completions(
                        reg, pos, side, self.sheets[side], seen=shown[side],
                        weights=_bench_weights(
                            self.bench_prior, side, pos, shown[side], record, leads[side]
                        ),
                    )
                    for side in (0, 1)
                }
            except ValueError as problem:
                record.unmodelled.append(f"hidden bench: {problem}")
                break
            spreads = _believed(spreads, (self.agent.bench_drop, self.agent.bench_drop))
            moved = self._agent_move(pos, spreads, recorded_shown)
            if moved is None:
                break
            agent_action, decision, extra = moved
            legal = side_actions(reg, pos, self.you)
            human_action = self._ask("move", legal, pos, seen)
            chosen = [agent_action, human_action] if self.me == 0 else [human_action, agent_action]
            human_menu = decision.foe_actions if self.you == 1 else decision.own_actions
            extra["humanOffMenu"] = human_action.to_choice() not in human_menu
            if self.you == 0:
                decision.own_chosen = human_action.to_choice()
            else:
                decision.foe_chosen = human_action.to_choice()
            record.decisions.append(decision)
            self.extras[len(record.decisions) - 1] = extra
            self.say(
                "-- 相手の行動: "
                + agent_action.describe(reg, self.loc, target_names(pos, self.me))
            )
            advanced = self._advance_turn(
                pos, chosen, _HiddenBench(self.sheets, list(seen), self.bench_prior, list(leads))
            )
            if self.listener is not None:
                agent_label = action_label(reg, agent_action, pos, self.me, self.loc)
                person_label = action_label(reg, human_action, pos, self.you, self.loc)
                self.emit("turn", {
                    "turn": pos.turn,
                    "decision": len(record.decisions) - 1,
                    "agent": agent_label,
                    "person": person_label,
                    "agentSlots": agent_label.split(SLOT_SEPARATOR),
                    "personSlots": person_label.split(SLOT_SEPARATOR),
                    "offMenu": extra["humanOffMenu"],
                    "changes": [] if advanced is None else turn_changes(
                        reg, pos, advanced, self.you, self.loc, names=self.side_names
                    ),
                })
            if advanced is None:
                record.end_reason = "unresolved"
                break
            pos = advanced
            record.turns = pos.turn
        if pos.ended and pos.winner is not None:
            record.outcome = 1.0 if pos.winner == pos.sides[0].id else 0.0
        _close_record(record, pos)
        self.final = pos
        if self.listener is not None:
            self.emit("board", board_view(
                reg, pos, self.you, seen[self.me], self.loc, names=self.side_names,
                seconds=self.agent.seconds,
            ))
            self.emit("end", {
                "outcome": record.outcome,
                "reason": record.end_reason,
                "turns": record.turns,
                "personSide": self.you,
            })
        return record

    def _agent_move(
        self, pos: Position, spreads: dict[int, list], recorded_shown: list[list[str]]
    ) -> tuple[SideAction, Decision, dict[str, Any]] | None:
        reg, agent, me, you = self.reg, self.agent, self.me, self.you
        started = time.perf_counter()
        exact = all(len(items) == 1 and items[0].exact for items in spreads.values())
        classes = len(spreads[you])
        plan = plan_move(
            agent.seconds, agent.cores,
            legal_count(reg, pos, me), legal_count(reg, pos, you), classes,
            form=agent.form, width_only=agent.width_only,
        )
        budget = Budget.matrix()
        ours, theirs = _menus(
            reg, pos, (plan.width, plan.width), agent.leaf, budget, agent.rank_by_leaf,
            None, spreads, rank_fill=agent.rank_fill,
        )
        if not ours or not theirs:
            return None
        menu_seconds = time.perf_counter() - started
        progress = None
        if self.listener is not None:
            self.emit("think", {
                "decision": len(self.record.decisions), "turn": pos.turn, "plan": plan,
                "clock": agent.clock, "seconds": agent.seconds, "menuMs": menu_seconds * 1000.0,
                "classCount": classes, "exact": exact,
            })
            progress = Recorder(
                Reader(reg, pos, me, loc=self.loc, names=self.side_names),
                lambda snapshot: self.emit("step", snapshot),
                decision=len(self.record.decisions), turn=pos.turn, started=started,
                interval_ms=self.interval_ms,
            )
        watch = {} if progress is None else {"progress": progress}
        cells = 0
        cost: Any = None
        if plan.deepen_ms > 0:
            if agent.clock == "wall":
                cost = WallCost(started)
                cells = int(round(plan.budget_ms))
            else:
                cost = COSTS[agent.form, agent.cores]
                cells = cells_for_seconds(plan.deepen_ms / 1000.0, agent.cores, form=agent.form)
        deepened = None
        try:
            if exact:
                got = search(
                    reg, pos, ours, theirs, agent.leaf, budget=budget,
                    **(
                        {"deepen": cells, "deepen_cost": cost, "levels": agent.max_levels,
                         "child_q": agent.child_q}
                        if cells else {}
                    ),
                    **watch,
                )
                strategy = np.asarray(
                    got.equilibrium.row_strategy if me == 0 else got.equilibrium.col_strategy,
                    dtype=np.float64,
                )
                model = [
                    float(x)
                    for x in (got.equilibrium.col_strategy if me == 0 else got.equilibrium.row_strategy)
                ]
                ours, theirs = got.ours, got.theirs
                value = float(got.equilibrium.value)
                deepened = got.deepened
                unmodelled = set(got.unmodelled)
            else:
                answers = belief_solve(
                    reg, pos, ours, theirs, spreads,
                    {me: agent.leaf, you: _not_asked}, budget=budget, sides=(me,),
                    deepen=(
                        {me: {"cells": cells, "reading": "mixed", "swap": False,
                              "outside": None, "cost": cost, "levels": agent.max_levels,
                              "child_q": agent.child_q}}
                        if cells else None
                    ),
                    **watch,
                )
                got = answers[me]
                strategy = np.asarray(got.strategy, dtype=np.float64)
                model = _averaged(got.replies, [item.weight for item in spreads[you]])
                ours, theirs = got.ours, got.theirs
                # Side 1 solved the negated transpose: its value back in side 0's units.
                value = float(got.value) if me == 0 else -float(got.value)
                deepened = got.deepened
                unmodelled = set(got.unmodelled)
        except EquilibriumError:
            return None
        mine = ours if me == 0 else theirs
        index = _sample_index(self.rng, strategy)
        took = time.perf_counter() - started
        if self.listener is not None:
            self.emit("answer", {
                "decision": len(self.record.decisions), "turn": pos.turn,
                "actions": list(mine), "strategy": strategy, "value0": value,
                "seconds": took, "deepened": deepened,
                "steps": None if progress is None else progress.calls,
                "sent": None if progress is None else progress.sent,
            })
        self.record.unmodelled.extend(unmodelled)
        self.record.search_seconds[me] += took
        report = None if deepened is None else deepened.to_json()
        extra = {
            "plan": {
                **plan.to_json(),
                "clock": agent.clock,
                "deepenBudget": cells,
                "exact": exact,
                "classes": classes,
            }
        }
        self.clock.append({
            "decision": len(self.record.decisions),
            "turn": pos.turn,
            "kind": "move",
            "budget": agent.seconds,
            "seconds": round(took, 4),
            "ratio": round(took / agent.seconds, 4) if agent.seconds > 0 else None,
            "menuSeconds": round(menu_seconds, 4),
            "width": plan.width,
            "rows": len(mine),
            "cols": len(theirs if me == 0 else ours),
            "classes": classes,
            "nodeCells": len(ours) * len(theirs) * max(classes, 1),
            "predictedMs": round(plan.predicted_ms, 3),
            "deepenBudget": cells,
            **({"deepened": report} if report is not None else {}),
        })
        own_side = (mine, strategy.tolist()) if me == 0 else ((ours), model)
        foe_side = ((theirs), model) if me == 0 else (mine, strategy.tolist())
        decision = Decision(
            turn=pos.turn,
            kind="move",
            position=pos.to_json(),
            own_actions=[a.to_choice() for a in own_side[0]],
            own_policy=[float(x) for x in own_side[1]],
            foe_actions=[a.to_choice() for a in foe_side[0]],
            foe_policy=[float(x) for x in foe_side[1]],
            search_value=value,
            own_chosen=mine[index].to_choice() if me == 0 else None,
            foe_chosen=mine[index].to_choice() if me == 1 else None,
            shown=recorded_shown,
            deepened=(
                [report, None] if me == 0 else [None, report]
            ) if report is not None else None,
        )
        return mine[index], decision, extra

    def _replacement(
        self,
        pos: Position,
        owed: tuple[tuple[bool, ...], ...],
        seen: list[frozenset[str]],
        shown: list[frozenset[int]],
        leads: list[frozenset[str] | None],
        recorded_shown: list[list[str]],
    ) -> Position:
        """The replacement phase: the agent's side solved as its Bayesian game over the
        person's bench (as `selfplay._do_replacement_node` solves each side), the person
        asked."""
        reg, me, you, record = self.reg, self.me, self.you, self.record
        options: list[list[SideAction]] = []
        for side_index in range(2):
            must = list(owed[side_index])
            found = switch_actions_after_faint(reg, pos, side_index, must) if any(must) else []
            if not found:
                found = [
                    SideAction(
                        slots=tuple(
                            PassAction(slot=s) for s in range(len(pos.sides[side_index].active))
                        )
                    )
                ]
            options.append(found)
        started = time.perf_counter()
        leaf = self.agent.evaluate

        def matrix(at: Position) -> np.ndarray:
            resolved = [
                [port.resolve_replacements(reg, at, [a, b]).position for b in options[1]]
                for a in options[0]
            ]
            if leaf is None:
                return np.array([[HP_SHARE(p) for p in row] for row in resolved], dtype=np.float64)
            flat = [p for row in resolved for p in row]
            return np.asarray(leaf(flat), dtype=np.float64).reshape(len(options[0]), len(options[1]))

        mine = options[me]
        policy = [1.0]
        model = [1.0 / len(options[you])] * len(options[you])
        value = 0.5
        if len(mine) > 1 or len(options[you]) > 1:
            try:
                items = completions(
                    reg, pos, you, self.sheets[you], seen=shown[you],
                    weights=_bench_weights(
                        self.bench_prior, you, pos, shown[you], record, leads[you]
                    ),
                )
            except ValueError as problem:
                record.unmodelled.append(f"hidden bench at a replacement: {problem}")
                items = []
            if items:
                built = [matrix(item.position) for item in items]
                weights = np.asarray([item.weight for item in items], dtype=np.float64)
                try:
                    solved = solve_bayesian([m if me == 0 else -m.T for m in built], weights)
                    policy = [float(x) for x in solved.row_strategy]
                    model = _averaged(solved.col_strategies, weights)
                    value = float(solved.value) if me == 0 else -float(solved.value)
                except EquilibriumError:
                    policy = [1.0 / len(mine)] * len(mine)
            else:
                policy = [1.0 / len(mine)] * len(mine)
        agent_action = mine[_sample_index(self.rng, np.array(policy))]
        took = time.perf_counter() - started
        self.clock.append({
            "decision": len(record.decisions), "turn": pos.turn, "kind": "replacement",
            "seconds": round(took, 4), "options": len(mine),
        })
        if len(options[you]) > 1:
            human_action = self._ask("replacement", options[you], pos, seen)
        else:
            human_action = options[you][0]
        chosen = [agent_action, human_action] if me == 0 else [human_action, agent_action]
        pairs = [(options[0], policy if me == 0 else model), (options[1], model if me == 0 else policy)]
        record.decisions.append(
            Decision(
                turn=pos.turn,
                kind="replacement",
                position=pos.to_json(),
                own_actions=[a.to_choice() for a in pairs[0][0]],
                own_policy=[float(x) for x in pairs[0][1]],
                foe_actions=[a.to_choice() for a in pairs[1][0]],
                foe_policy=[float(x) for x in pairs[1][1]],
                search_value=value,
                own_chosen=chosen[0].to_choice(),
                foe_chosen=chosen[1].to_choice(),
                shown=recorded_shown,
            )
        )
        outcome = port.resolve_replacements(reg, pos, chosen, rng=self.rng)
        record.unmodelled.extend(outcome.unmodelled)
        return outcome.position

    def _advance_turn(
        self, pos: Position, chosen: list[SideAction], hidden: _HiddenBench
    ) -> Position | None:
        """`selfplay._advance_turn`, with the person answering their own mid-turn switch."""
        reg, record = self.reg, self.record
        weights = port.weights(reg, pos, chosen, Budget.exact())
        record.unmodelled.extend(weights.unmodelled)
        counts = np.array(weights.branches + weights.suspended, dtype=np.float64)
        if not counts.size or float(counts.sum()) <= 0:
            return None
        index = _sample_index(self.rng, counts)
        if index < len(weights.branches):
            return port.branch(reg, pos, chosen, Budget.exact(), index)
        paused = port.turn(reg, pos, chosen, Budget.exact(), select=index).pause
        if paused is None:
            raise port.PortRefused(f"the port gave no pause at index {index} of the turn")
        result = None
        pause = paused
        for attempt in range(5):
            if attempt:
                assert result is not None and result.outcomes is not None
                assert result.pauses is not None
                w = np.array(result.branches + result.suspended, dtype=np.float64)
                if not w.size or float(w.sum()) <= 0:
                    return None
                k = _sample_index(self.rng, w)
                if k < len(result.branches):
                    return result.outcomes[k].position
                pause = result.pauses[k - len(result.branches)]
            result = self._self_switch(pause, hidden)
            if result is None:
                return None
        record.unmodelled.append("more than five mid-turn replacements in one turn")
        return None

    def _self_switch(self, pause: Any, hidden: _HiddenBench) -> Any:  # noqa: ANN401
        reg, record = self.reg, self.record
        probe = port.alternatives_encoded(reg, pause, want=[])
        chooser = probe.chooser
        if chooser is None or not probe.options:
            return None
        if chooser == self.me:
            started = time.perf_counter()
            got = _do_self_switch_node(
                reg, pause, record, self.leaves, self.agent.objective, hidden=hidden
            )
            self.clock.append({
                "decision": len(record.decisions) - 1, "turn": pause.position.turn,
                "kind": "selfswitch", "seconds": round(time.perf_counter() - started, 4),
            })
            return got
        options = list(probe.options)
        seen = [seen_identities(pause.position, i, hidden.seen[i]) for i in (0, 1)]
        choice = (
            self._ask("selfswitch", options, pause.position, seen) if len(options) > 1 else options[0]
        )
        other = 1 - chooser
        passes = SideAction(
            slots=tuple(PassAction(slot=i) for i in range(len(pause.position.sides[other].active)))
        )
        names = [o.to_choice() for o in options]
        policy = [1.0 if n == choice.to_choice() else 0.0 for n in names]
        waiting = ["pass"]
        record.decisions.append(
            Decision(
                turn=pause.position.turn,
                kind="selfswitch",
                position=pause.position.to_json(),
                own_actions=names if chooser == 0 else waiting,
                own_policy=policy if chooser == 0 else [1.0],
                foe_actions=waiting if chooser == 0 else names,
                foe_policy=[1.0] if chooser == 0 else policy,
                # The person's choice has no search behind it.
                search_value=None,  # type: ignore[arg-type]
                own_chosen=choice.to_choice() if chooser == 0 else "pass",
                foe_chosen="pass" if chooser == 0 else choice.to_choice(),
                shown=_shown_record(pause.position, seen),
            )
        )
        self.extras[len(record.decisions) - 1] = {"humanPolicy": "chosen"}
        return port.resume(reg, pause, [choice, passes] if chooser == 0 else [passes, choice])


# ----------------------------------------------------------------------------- selection


def solve_entry(
    reg: Regulation, teams: tuple[Roster, Roster], evaluate: LeafEvaluator, model: str
) -> BookEntry:
    """The selection game of the two sheets, with side 0's team as the row player (as
    `poolplay.SolvedSelections` solves a pair)."""
    from .selection import SpreadClass, book_entry, solve_selection

    row, col = teams
    analysis = solve_selection(
        reg, row.sets, [SpreadClass(weight=1.0, sets=tuple(col.sets), label="sheet")], evaluate
    )
    return book_entry(analysis, key=f"{row.id}|{col.id}", player=col.name, model=model)


def agent_pick(
    entry: BookEntry | None, side: int, six: Sequence[SampledSet], size: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """The agent's ordered four: from its side of the solve (the pure equilibrium, as a
    board plays), or four of six uniformly without a leaf."""
    if entry is None:
        return tuple(pick_four_indices(rng, len(six), size=size))
    mixture = (
        entry.our_mixture(epsilon=0.0, temperature=1.0)
        if side == 0
        else entry.their_mixture(0, epsilon=0.0, temperature=1.0)
    )
    cumulative = np.cumsum(np.asarray(mixture, dtype=np.float64))
    cumulative /= cumulative[-1]
    index = min(int(np.searchsorted(cumulative, rng.random(), side="right")), len(cumulative) - 1)
    return tuple(entry.selections[index])


#: The belief about the person's four: the solve's mixture for that side, explored as
#: generation explores (epsilon 0.25), so a four off the equilibrium's support is still a
#: world the belief holds rather than one it cannot explain.
BELIEF_EPSILON = 0.25


def play(
    agent: Agent,
    person: Person,
    teams: tuple[Roster, Roster],
    *,
    agent_side: int,
    seed: int,
    game_index: int = 0,
    max_turns: int = MAX_TURNS,
    loc: Any = None,  # noqa: ANN401
    out: TextIO | None = None,
    belief_epsilon: float = BELIEF_EPSILON,
    listener: Callable[[str, Any], None] | None = None,
    interval_ms: float = 100.0,
) -> tuple[dict[str, Any], dict[str, Any], HumanGame]:
    """Plays one game. ``teams`` are side 0's and side 1's sheets. Returns the game's
    record line, its clock line, and the game object (for a caller that wants the board).
    ``listener`` hears the game as it goes (the module's docstring, IKA-332).
    """
    reg = agent.reg
    if agent_side not in (0, 1):
        raise ValueError(f"agent_side is 0 or 1, not {agent_side}")
    you = 1 - agent_side
    size = reg.meta.picked_team_size
    six = (list(teams[0].sets), list(teams[1].sets))
    species = ([s.species for s in six[0]], [s.species for s in six[1]])
    started = time.perf_counter()
    entry = (
        solve_entry(reg, teams, agent.evaluate, agent.name) if agent.evaluate is not None else None
    )
    selection_seconds = time.perf_counter() - started
    mine = agent_pick(
        entry, agent_side, six[agent_side], size,
        np.random.default_rng([seed, game_index, 1, agent_side]),
    )
    if out is not None:
        out.write(f"\n相手のチーム（サイド {agent_side}）:\n{render_sheet(reg, six[agent_side], loc)}\n")
        out.write(f"\n自分のチーム（サイド {you}）:\n")
    if listener is not None:
        listener("sheets", {
            "agentSide": agent_side, "personSide": you, "seed": seed, "gameIndex": game_index,
            "agent": agent.name, "seconds": agent.seconds, "clock": agent.clock,
            "cores": agent.cores,
            "teams": [sheet_view(six[s], loc, reg) for s in (0, 1)],
            "statNames": {
                k: (loc.stat(k, short=True) if loc is not None else k)
                for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
            },
            "statusNames": {
                k: (loc.status(k) if loc is not None else k)
                for k in ("brn", "par", "psn", "tox", "slp", "frz")
            },
            "names": ["AI" if s == agent_side else "あなた" for s in (0, 1)],
        })
        listener("select", {"size": size, "team": len(six[you])})
    yours = person.select(six[you], size, render_sheet(reg, six[you], loc))
    inputs = [selection_line(yours)]
    picks = (mine, yours) if agent_side == 0 else (yours, mine)
    priors = None
    if entry is not None:
        priors = tuple(
            BenchPrior.of(entry, side, species[side], epsilon=belief_epsilon, temperature=1.0)
            for side in (0, 1)
        )
    game = HumanGame(
        agent, person, agent_side=agent_side, sheets=six, picks=picks, bench_prior=priors,
        rng=np.random.default_rng([seed, game_index, 2]), max_turns=max_turns, loc=loc, out=out,
        listener=listener, interval_ms=interval_ms,
    )
    game.inputs = inputs
    record = game.play()
    record.selection_source = "solved" if entry is not None else "uniform"
    if entry is not None:
        record.selection_value = float(entry.value)
    widths = [row["width"] for row in game.clock if row["kind"] == "move"] or [0]
    payload = record.to_json(objective=agent.name, search_limit=(max(widths), max(widths)))
    search_seconds = payload.pop("searchSeconds")
    for index, extra in game.extras.items():
        payload["decisions"][index].update(extra)
    payload["provenance"] = {
        "kind": "human-play",
        "seat": f"agent = side {agent_side}",
        "leaves": [agent.name if s == agent_side else "person" for s in (0, 1)],
        "information": ["hidden-bench", "hidden-bench"],
        "rankFills": [agent.rank_fill if s == agent_side else "person" for s in (0, 1)],
        "benchDrops": [agent.bench_drop if s == agent_side else "person" for s in (0, 1)],
        "books": ["solved" if entry is not None and s == agent_side else
                  "person" if s == you else "uniform" for s in (0, 1)],
    }
    payload["gameIndex"] = game_index
    payload["seed"] = seed
    payload["teams"] = [teams[0].id, teams[1].id]
    payload["picks"] = [list(picks[0]), list(picks[1])]
    payload["human"] = {
        "side": you,
        # Who answered is the clock file's (`person`): the same inputs are the same game,
        # whether a person typed them or a script read them back.
        "inputs": game.inputs,
        "beliefEpsilon": belief_epsilon,
    }
    payload["clock"] = {
        "mode": agent.clock,
        "secondsPerMove": agent.seconds,
        "cores": agent.cores,
        "form": agent.form,
        "widthOnly": agent.width_only,
        "rule": {
            "widths": list(WIDTHS),
            "widthShare": WIDTH_SHARE,
            "minDeepenMs": MIN_DEEPEN_MS,
            "nodeTime": {"fixedMs": node_time(agent.cores, agent.form).fixed_ms,
                         "cellMs": node_time(agent.cores, agent.form).cell_ms},
        },
    }
    moves = [row for row in game.clock if row["kind"] == "move"]
    clock = {
        "gameIndex": game_index,
        "seed": seed,
        "person": person.kind,
        "mode": agent.clock,
        "cores": agent.cores,
        "secondsPerMove": agent.seconds,
        "searchSeconds": search_seconds,
        "selectionSeconds": round(selection_seconds, 4),
        "moves": len(moves),
        "meanRatio": round(float(np.mean([m["ratio"] for m in moves])), 4) if moves else None,
        "decisions": game.clock,
    }
    return payload, clock, game


def write_line(path: Path, payload: dict[str, Any]) -> None:
    """Appends one JSON line, LF, UTF-8 (bytes, so Windows writes no CR)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def clock_path(out: Path) -> Path:
    """The clock file beside a games file: ``games.jsonl`` -> ``games.clock.jsonl``."""
    return out.with_name(out.stem + ".clock" + (out.suffix or ".jsonl"))


__all__ = [
    "BELIEF_EPSILON",
    "CLOCKS",
    "MIN_DEEPEN_MS",
    "NODE_TIME",
    "WIDTHS",
    "WIDTH_SHARE",
    "Agent",
    "HumanGame",
    "MovePlan",
    "NodeTime",
    "Person",
    "PolicyPerson",
    "ScriptPerson",
    "TerminalPerson",
    "WallCost",
    "agent_pick",
    "board_view",
    "sprite_id",
    "species_types",
    "turn_changes",
    "clock_path",
    "node_time",
    "parse_choice",
    "parse_selection",
    "plan_move",
    "play",
    "render_position",
    "render_sheet",
    "selection_line",
    "sheet_view",
    "solve_entry",
    "write_line",
]
