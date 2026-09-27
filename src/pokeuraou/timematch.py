"""Two clocks against each other: the human-play agent on both seats (IKA-333).

`tools/pool_match.py` and `tools/match_queue.py` play agents of a fixed width and depth.
What a person meets is another agent: `humanplay`'s, which spends a number of seconds a
move (width first, the rest to the deepening, `humanplay.plan_move`). This module plays
that agent against itself under two *conditions* -- seconds a move, threads, the
deepening's guard, the swap oracle -- so a match can say what ten times the time is worth.

**The game** is `humanplay.play`'s, with the person's seat taken by a second agent
(`TimedGame`):

- *moves*: each seat reads the move from its own side with `HumanGame._agent_move` -- its
  own condition, its own belief about the other side's bench (the same public information
  a person's opponent has), its own threads. The two seats read **one after the other in
  one process**, so the seat reading has the process's cores to itself and the other
  waits: in a real game against a person the agent's cores are its own while it thinks.
  Reading both at once would not be fair on the wall clock: the seat with less time would
  share the cache, the memory bus and the card with the other for its whole read, and the
  seat with more time for only the start of its own;
- *replacements*: both seats solve their Bayesian game over the other side's bench
  (`HumanGame._replacement_mixture`, what the agent plays there), outside the clock;
- *mid-turn switches*: both seats choose as the agent does (`selfplay._do_self_switch_node`
  over the other side's bench), outside the clock;
- *selection*: the one solve of the two sheets (`humanplay.solve_entry`), each side drawing
  its ordered four from its own side's pure equilibrium, outside the clock. Both
  conditions get the same selection, whatever their seconds.

**A pair** is two games on one pair of teams from the M-C pool (`pool.draw_pair`) with the
conditions swapped between the sides, the same seed and the same selection draws, so the
teams, the leads and the first chance draws cancel between them; a pair scores 0, 1/2 or
1 for the tested condition (`sprt`). Which seat is `HumanGame`'s own (it reads first)
alternates by pair.

**Threads.** A condition's ``threads`` are applied before each of its reads
(`spread_threads`): the port's cell threads, the deepening's cells expanded ahead and a
big game's two LPs at once, as `humanplay.use_threads` spreads them. The worker processes
that expand ahead are started once, for the larger of the two conditions.

**Ponder** (IKA-344) is not a condition here: the user decided on 9/27 not to use it (a
game whose two clocks differ is rare in practice), so both seats play with it off, as
`tools/play_human.py` does by default.

**Records.** One line per game (`game_line`): the conditions by side, the outcome, and
per move decision the clock row `HumanGame` writes (seconds against the budget, width,
cells of the depth-1 node, and what the deepening did: steps, depth, cells, the oracle's
probes, and with a guard why it stopped), tagged with the seat and its condition.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import deepen, equilibrium, humanplay, rustnode
from .deepen import ALL_ACTIONS, MAX_LEVELS
from .selfplay import _do_self_switch_node, _sample_index
from .sprt import elo_of

# ----------------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class Condition:
    """One side of the comparison: how the agent spends its move."""

    name: str
    seconds: float
    #: Threads a move spreads over (`humanplay.default_threads`: IKA-343's default, 4).
    threads: int = humanplay.default_threads()
    #: The cores the width rule prices a move at: None is ``threads`` on the wall clock
    #: and 1 on the count clock (`tools/play_human.py`'s `resolve_cores`).
    cores: int | None = None
    clock: str = "wall"
    #: The root's swap oracle (`humanplay.PLAY_ORACLE`, ``sall``), or None.
    oracle: int | None = humanplay.PLAY_ORACLE
    #: The deepening's guard (`humanplay.PLAY_MAX_LEVELS`): None is `deepen.MAX_LEVELS`,
    #: unrecorded; a number is that guard, and each move's record says why it stopped.
    max_levels: int | None = humanplay.PLAY_MAX_LEVELS
    width_only: bool = False
    #: A fixed menu width in place of the width rule (``width=W``; with ``width_only`` it
    #: is a board's ``d1@W``), or None: the rule.
    width: int | None = None
    child_q: int | None = None
    #: Knock-outs forked in the move's turns (`humanplay.Agent.knockouts`, IKA-362).
    knockouts: bool = False
    #: The deepening's children's width and kept branches (`humanplay.Agent`, IKA-362).
    sub_limit: int | None = None
    sub_branches: int | None = None
    #: The restricted reading of the deepened root (`humanplay.Agent.restricted`, IKA-362).
    restricted: bool | str = False
    #: A fixed depth-2 read (`humanplay.Agent.depth`, IKA-362's D), its rectangle's side
    #: and passes.
    depth: int = 1
    refine: int | None = None
    passes: int | None = None
    #: The width rule, then a fixed depth-2 read as the budget allows (IKA-362).
    depth2_auto: bool = False
    #: depth2_auto's root at every legal action in the opening (`humanplay.Agent.root_all`,
    #: IKA-366).
    root_all: bool = False
    #: The width rule, then -- instead of the deepening -- the stages of a ladder read
    #: coarse to fine within the rest of the budget (`humanplay.Agent.ladder`, IKA-367): a
    #: name in `ladder.LADDERS` or stages joined by ``+``. None: off.
    ladder: str | None = None

    @property
    def price_cores(self) -> int:
        if self.cores is not None:
            return self.cores
        return self.threads if self.clock == "wall" else 1

    def oracle_label(self) -> str:
        if self.oracle is None:
            return "none"
        return "sall" if self.oracle >= ALL_ACTIONS else f"s{self.oracle}"

    def describe(self) -> str:
        """One line, every setting spelled out (the echo a run prints and writes)."""
        return (
            f"{self.name}: {self.seconds:g} s a move, {self.clock} clock, {self.threads} "
            f"thread(s) priced at {self.price_cores} core(s), oracle {self.oracle_label()}, "
            f"guard {self.max_levels if self.max_levels is not None else MAX_LEVELS}"
            f"{'' if self.max_levels is not None else ' (unrecorded)'}"
            + (" , width only" if self.width_only else "")
            + (f", width fixed at {self.width}" if self.width is not None else "")
            + (", knock-outs forked" if self.knockouts else "")
            + (f", children {self.sub_limit} wide" if self.sub_limit is not None else "")
            + (f", {self.sub_branches} branches kept" if self.sub_branches is not None else "")
            + ((", open roots read restricted" if self.restricted == "open"
                else ", root read restricted") if self.restricted else "")
            + (f", depth {self.depth} fixed" if self.depth > 1 else "")
            + (f", rectangle {self.refine}" if self.refine is not None else "")
            + (f", {self.passes} passes" if self.passes is not None else "")
            + (", depth 2 as the budget allows" if self.depth2_auto else "")
            + (", the opening's root at every legal action" if self.root_all else "")
            + (f", ladder {self.ladder}" if self.ladder is not None else "")
            + (f", child Q {self.child_q}" if self.child_q is not None else "")
        )

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["oracle"] = self.oracle_label()
        out["priceCores"] = self.price_cores
        return out


#: The keys a condition is written with, and what each one parses.
CONDITION_KEYS = ("seconds", "threads", "cores", "clock", "oracle", "levels", "width_only",
                  "width", "child_q", "knockouts", "sub_limit", "sub_branches", "restricted",
                  "depth", "refine", "passes", "depth2_auto", "root_all", "ladder")


def _oracle(spec: str) -> int | None:
    if spec == "none":
        return None
    if spec == "sall":
        return ALL_ACTIONS
    if spec.startswith("s") and spec[1:].isdigit() and int(spec[1:]) > 0:
        return int(spec[1:])
    raise ValueError(f"oracle is s<W>, sall or none, not {spec!r}")


def _flag(spec: str) -> bool:
    if spec in ("1", "on", "true", "yes"):
        return True
    if spec in ("0", "off", "false", "no"):
        return False
    raise ValueError(f"a switch is on or off, not {spec!r}")


def parse_condition(spec: str) -> Condition:
    """``name:key=value,key=value``. Every key left out is the human-play default
    (`Condition`'s, which reads `humanplay`'s), but for the threads on the count clock: 1,
    since a count-clock game is the same game at any number of threads (IKA-343) and one
    thread a game lets many games share the machine. ``levels=0`` is `deepen.MAX_LEVELS`
    unrecorded, as ``play_human --max-levels 0``.

        long:seconds=10
        g16:seconds=10,levels=16
        old:seconds=1,threads=1,oracle=none
        c10:seconds=10,clock=count
    """
    name, sep, rest = spec.partition(":")
    if not sep or not name or not rest:
        raise ValueError(f"a condition is name:key=value,..., not {spec!r}")
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"a condition's name is letters, digits, - and _, not {name!r}")
    got: dict[str, Any] = {}
    for item in rest.split(","):
        key, eq, value = item.partition("=")
        key = key.strip().replace("-", "_")
        if not eq or key not in CONDITION_KEYS:
            raise ValueError(f"unknown condition key in {item!r}; keys: {', '.join(CONDITION_KEYS)}")
        if key in got:
            raise ValueError(f"{key} given twice in {spec!r}")
        value = value.strip()
        if key == "seconds":
            got["seconds"] = float(value)
        elif key in ("threads", "cores"):
            got[key] = int(value)
        elif key == "clock":
            if value not in humanplay.CLOCKS:
                raise ValueError(f"clock is one of {humanplay.CLOCKS}, not {value!r}")
            got["clock"] = value
        elif key == "oracle":
            got["oracle"] = _oracle(value)
        elif key == "levels":
            got["max_levels"] = int(value) or None
        elif key == "child_q":
            got["child_q"] = int(value)
        elif key in ("width", "sub_limit", "sub_branches", "depth", "refine", "passes"):
            got[key] = int(value)
        elif key == "restricted":
            got[key] = "open" if value == "open" else _flag(value)
        elif key == "ladder":
            from .ladder import parse_ladder

            parse_ladder(value)  # refuses a stage it cannot read, before a game starts
            got[key] = value
        else:
            got[key] = _flag(value)
    if "seconds" not in got:
        raise ValueError(f"a condition names its seconds a move: {spec!r}")
    if got.get("clock") == "count" and "threads" not in got:
        got["threads"] = 1
    condition = Condition(name=name, **got)
    if condition.seconds <= 0 or condition.threads < 1:
        raise ValueError(f"seconds must be > 0 and threads >= 1: {spec!r}")
    return condition


# ----------------------------------------------------------------------------- threads


#: The port's cell threads for every read, whatever the condition's threads (None: the
#: condition's). A node-time run's games do not depend on them (IKA-343), so a run fills
#: the machine's cores without more processes on the card (IKA-362, `--port-threads`).
PORT_THREADS: int | None = None


def spread_threads(reg: Any, threads: int) -> None:  # noqa: ANN401
    """`humanplay.use_threads` without starting or stopping the worker processes: the
    port's cell threads, the cells expanded ahead and the two LPs at once for ``threads``,
    with the worker processes already started for the process."""
    port = PORT_THREADS or threads
    if (rustnode.port_threads() or 1) != port:
        rustnode.set_port_threads(port)
    remote = deepen.workers(reg) > 0
    deepen.set_ahead(
        0 if threads == 1 else threads,
        helpers=humanplay.AHEAD_HELPERS if threads >= 4 else 1,
        port_threads=threads, deeper=humanplay.AHEAD_DEEPER if remote else 0,
    )
    equilibrium.set_lp_pair(0 if threads == 1 else humanplay.LP_PAIR_CELLS)


# ----------------------------------------------------------------------------- the game


class SeatPerson(humanplay.Person):
    """The person's seat as an agent's: the selection from its own side of the solve.
    Everything else is answered by `TimedGame`."""

    kind = "agent"

    def __init__(self, entry: Any, side: int, rng: np.random.Generator) -> None:  # noqa: ANN401
        self.entry = entry
        self.side = side
        self.rng = rng

    def select(self, six, size, text):  # noqa: ANN001, ANN201, ARG002
        return humanplay.agent_pick(self.entry, self.side, six, size, self.rng)

    def choose(self, kind, legal, text):  # noqa: ANN001, ANN201, ARG002
        raise AssertionError("TimedGame answers the person's seat itself")


class TimedGame(humanplay.HumanGame):
    """`HumanGame` with an agent in the person's seat (the module's docstring)."""

    def __init__(
        self,
        *args: Any,  # noqa: ANN401
        seats: tuple[humanplay.Agent, humanplay.Agent],
        conditions: tuple[Condition, Condition],
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        self.seats = seats
        self.conditions = conditions
        #: The person's seat's move this turn, read beside the agent's.
        self._answer: Any = None
        self._owed: Any = None
        self._context: Any = None
        #: Reads that gave no answer (no menu, an LP failure), by side: the person's seat
        #: then plays its first legal action, the agent's seat ends the game.
        self.fallbacks = [0, 0]

    def order(self) -> tuple[int, int]:
        """Which side reads first: `me` (the reads are independent; the order only has to
        be fixed)."""
        return self.me, 1 - self.me

    def _read(self, side: int, pos: Any, spreads: Any, shown: Any) -> Any:  # noqa: ANN401
        saved = (self.agent, self.me, self.you)
        condition = self.conditions[side]
        self.agent, self.me, self.you = self.seats[side], side, 1 - side
        spread_threads(self.reg, condition.threads)
        start = len(self.clock)
        try:
            got = humanplay.HumanGame._agent_move(self, pos, spreads, shown)
        finally:
            self.agent, self.me, self.you = saved
        for row in self.clock[start:]:
            row["side"] = side
            row["condition"] = condition.name
        if got is None:
            self.fallbacks[side] += 1
        return got

    def _agent_move(self, pos, spreads, recorded_shown, **_ponder):  # noqa: ANN001, ANN003, ANN202
        got: dict[int, Any] = {}
        for side in self.order():
            got[side] = self._read(side, pos, spreads, recorded_shown)
        other = got[self.you]
        self._answer = None if other is None else other[0]
        return got[self.me]

    def _replacement(self, pos, owed, seen, shown, leads, recorded_shown):  # noqa: ANN001, ANN202
        self._owed = owed
        self._context = (shown, leads)
        return super()._replacement(pos, owed, seen, shown, leads, recorded_shown)

    def _ask(self, kind, legal, pos, seen):  # noqa: ANN001, ANN202, ARG002
        if kind == "move":
            choice = self._answer if self._answer is not None else legal[0]
            names = [a.to_choice() for a in legal]
            if choice.to_choice() not in names:
                self.fallbacks[self.you] += 1
                choice = legal[0]
            else:
                choice = legal[names.index(choice.to_choice())]
        elif kind == "replacement":
            started = time.perf_counter()
            options = humanplay.replacement_options(self.reg, pos, self._owed)
            shown, leads = self._context
            policy, _model, _value = self._replacement_mixture(pos, options, self.you, shown, leads)
            choice = options[self.you][_sample_index(self.rng, np.array(policy))]
            self.clock.append({
                "decision": len(self.record.decisions), "turn": pos.turn, "kind": "replacement",
                "seconds": round(time.perf_counter() - started, 4), "options": len(legal),
                "side": self.you,
            })
        else:
            raise AssertionError(f"TimedGame answers {kind!r} itself")
        self.inputs.append(choice.to_choice())
        return choice

    def _self_switch(self, pause, hidden):  # noqa: ANN001, ANN202
        """Either side's mid-turn switch, as the agent chooses its own."""
        from . import port

        probe = port.alternatives_encoded(self.reg, pause, want=[])
        chooser = probe.chooser
        if chooser is None or not probe.options:
            return None
        started = time.perf_counter()
        got = _do_self_switch_node(
            self.reg, pause, self.record, self.leaves, self.agent.objective, hidden=hidden
        )
        self.clock.append({
            "decision": len(self.record.decisions) - 1, "turn": pause.position.turn,
            "kind": "selfswitch", "seconds": round(time.perf_counter() - started, 4),
            "side": chooser,
        })
        return got


# ----------------------------------------------------------------------------- a pair


@dataclass
class Match:
    """What every game of a run shares."""

    reg: Any  # noqa: ANN401
    evaluate: Any  # noqa: ANN401
    leaf_name: str
    rank_fill: str
    bench_drop: str
    tested: Condition
    other: Condition
    seed: int
    max_turns: int
    halt: Any = None  # noqa: ANN401
    #: Menus ranked by the leaf (the human-play agent's); False only for cheap tests.
    rank_by_leaf: bool = True
    loc: Any = None  # noqa: ANN401
    #: Extra fields written into every game line (the pool, the Q).
    stamp: dict[str, Any] = field(default_factory=dict)

    def agent(self, condition: Condition) -> humanplay.Agent:
        return humanplay.Agent(
            reg=self.reg, evaluate=self.evaluate, name=self.leaf_name,
            seconds=condition.seconds, cores=condition.price_cores, clock=condition.clock,
            rank_fill=self.rank_fill, rank_by_leaf=self.rank_by_leaf, bench_drop=self.bench_drop,
            width_only=condition.width_only, width=condition.width,
            knockouts=condition.knockouts, sub_limit=condition.sub_limit,
            sub_branches=condition.sub_branches, restricted=condition.restricted,
            depth=condition.depth, refine=condition.refine, passes=condition.passes,
            depth2_auto=condition.depth2_auto, root_all=condition.root_all,
            ladder=condition.ladder, max_levels=condition.max_levels,
            child_q=condition.child_q, oracle=condition.oracle, halt=self.halt,
            # Off, as a person's game plays by default (the module's docstring).
            ponder=False, ponder_seconds=humanplay.PLAY_PONDER_SECONDS,
        )


def play_pair(match: Match, pair: int, teams: tuple[Any, Any]) -> list[dict[str, Any]]:  # noqa: ANN401
    """The two games of ``pair`` on ``teams`` (side 0's, side 1's): the tested condition on
    side 0, then on side 1. Returns their lines (`game_line`)."""
    reg = match.reg
    started = time.perf_counter()
    entry = (
        humanplay.solve_entry(reg, teams, match.evaluate, match.leaf_name)
        if match.evaluate is not None else None
    )
    selection_seconds = time.perf_counter() - started
    agent_side = pair % 2
    lines = []
    for game in (0, 1):
        tested_side = game
        conditions = (
            (match.tested, match.other) if tested_side == 0 else (match.other, match.tested)
        )
        seats = (match.agent(conditions[0]), match.agent(conditions[1]))
        you = 1 - agent_side
        person = SeatPerson(entry, you, np.random.default_rng([match.seed, pair, 1, you]))

        def make_game(*args: Any, seats=seats, conditions=conditions, **kwargs: Any) -> TimedGame:  # noqa: ANN401
            return TimedGame(*args, seats=seats, conditions=conditions, **kwargs)

        began = time.perf_counter()
        payload, clock, played = humanplay.play(
            seats[agent_side], person, teams, agent_side=agent_side, seed=match.seed,
            game_index=pair, max_turns=match.max_turns, loc=match.loc, entry=entry,
            make_game=make_game,
        )
        lines.append(game_line(
            match, pair, game, teams, conditions, tested_side, payload, clock, played,
            seconds=time.perf_counter() - began, selection_seconds=selection_seconds,
        ))
        del played
    return lines


def game_line(
    match: Match, pair: int, game: int, teams: tuple[Any, Any],
    conditions: tuple[Condition, Condition], tested_side: int, payload: dict[str, Any],
    clock: dict[str, Any], played: TimedGame, *, seconds: float, selection_seconds: float,
) -> dict[str, Any]:
    outcome = payload["outcome"]
    score = None if outcome is None else (outcome if tested_side == 0 else 1.0 - outcome)
    rows = clock["decisions"]
    return {
        "pair": pair,
        "game": game,
        "seed": match.seed,
        "teams": [teams[0].id, teams[1].id],
        "picks": payload["picks"],
        "conditions": [conditions[0].name, conditions[1].name],
        "tested": match.tested.name,
        "testedSide": tested_side,
        "agentSide": played.me,
        "outcome": outcome,
        "testedScore": score,
        "turns": payload["turns"],
        "endReason": payload["endReason"],
        "seconds": round(seconds, 3),
        "selectionSeconds": round(selection_seconds, 3),
        "fallbacks": list(played.fallbacks),
        "memoryStops": sum(1 for r in rows if r.get("memoryStop")),
        "unmodelled": len(payload.get("unmodelled") or []),
        "moves": [r for r in rows if r["kind"] == "move"],
        "others": [r for r in rows if r["kind"] != "move"],
        **match.stamp,
    }


# ----------------------------------------------------------------------------- reading a run


def pair_scores(lines: Sequence[dict[str, Any]]) -> tuple[dict[int, float], dict[int, str]]:
    """The tested condition's score of each complete pair (both games in), and the pairs
    left out of it with why: a game with no winner (the turn cap) is nobody's win, and a
    pair holding one is not scored rather than scored as a loss for either side."""
    games: dict[int, dict[int, dict[str, Any]]] = {}
    for line in lines:
        games.setdefault(int(line["pair"]), {})[int(line["game"])] = line
    scores: dict[int, float] = {}
    left: dict[int, str] = {}
    for pair, both in games.items():
        if len(both) < 2:
            continue
        got = [both[g]["testedScore"] for g in (0, 1)]
        if any(s is None for s in got):
            left[pair] = "no winner in " + ", ".join(
                f"game {g} ({both[g]['endReason']})" for g in (0, 1) if got[g] is None)
            continue
        scores[pair] = (float(got[0] > 0.5) + float(got[1] > 0.5)) / 2.0
    return scores, left


def prefix(scores: dict[int, float], left: dict[int, str], start: int = 0) -> list[float]:
    """The scores of the pairs from ``start`` on, up to the first one not yet finished --
    the order they were handed out, so a pair that takes longer is not left behind."""
    out = []
    pair = start
    while pair in scores or pair in left:
        if pair in scores:
            out.append(scores[pair])
        pair += 1
    return out


def elo_interval(scores: Sequence[float], z: float = 1.96) -> dict[str, Any]:
    """The tested condition's Elo from its pair scores, with a normal interval on the pair
    mean (the pair is the independent unit, `sprt`)."""
    n = len(scores)
    if not n:
        return {"pairs": 0}
    x = np.asarray(scores, dtype=np.float64)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean - z * se, mean + z * se
    return {
        "pairs": n,
        "score": round(mean, 4),
        "elo": round(elo_of(mean), 1),
        "low": round(elo_of(min(max(lo, 1e-9), 1 - 1e-9)), 1) if n > 1 else None,
        "high": round(elo_of(min(max(hi, 1e-9), 1 - 1e-9)), 1) if n > 1 else None,
        "counts": [int((x == v).sum()) for v in (0.0, 0.5, 1.0)],
    }


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": len(x),
        "mean": round(float(x.mean()), 4),
        "p10": round(float(np.quantile(x, 0.1)), 4),
        "p50": round(float(np.quantile(x, 0.5)), 4),
        "p90": round(float(np.quantile(x, 0.9)), 4),
        "max": round(float(x.max()), 4),
    }


def move_profile(lines: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    """What a condition's moves were: seconds against the budget, width, the depth-1
    node's cells, and the deepening's steps, depth and cells."""
    rows = [r for line in lines for r in line["moves"] if r.get("condition") == name]
    deep = [r.get("deepened") or {} for r in rows]
    widths: dict[int, int] = {}
    for r in rows:
        widths[r["width"]] = widths.get(r["width"], 0) + 1
    depths: dict[int, int] = {}
    for d in deep:
        depths[d.get("depth", 1)] = depths.get(d.get("depth", 1), 0) + 1
    stops: dict[str, int] = {}
    for d in deep:
        if "stop" in d:
            stops[d["stop"]] = stops.get(d["stop"], 0) + 1
    return {
        "moves": len(rows),
        "seconds": _quantiles([r["seconds"] for r in rows]),
        "ratio": _quantiles([r["ratio"] for r in rows if r.get("ratio") is not None]),
        "menuSeconds": _quantiles([r["menuSeconds"] for r in rows]),
        "width": dict(sorted(widths.items())),
        "nodeCells": _quantiles([r["nodeCells"] for r in rows]),
        "steps": _quantiles([d.get("expanded", 0) for d in deep]),
        "depth": dict(sorted(depths.items())),
        "deepCells": _quantiles([d.get("cells", 0) for d in deep]),
        "probed": _quantiles([d.get("probed", 0) for d in deep]),
        "guarded": sum(d.get("guarded", 0) for d in deep),
        "stops": stops,
        "memoryStops": sum(1 for r in rows if r.get("memoryStop")),
    }


__all__ = [
    "CONDITION_KEYS",
    "Condition",
    "Match",
    "SeatPerson",
    "TimedGame",
    "elo_interval",
    "game_line",
    "move_profile",
    "pair_scores",
    "parse_condition",
    "play_pair",
    "prefix",
    "spread_threads",
]
