"""The typed-in position of the ranked screen: its session and its read (IKA-408).

`BoardApp` is the second half of `rankedweb.RankedApp`'s session. The first half (IKA-407) has
the person's team and the opponent's six estimated sets; this one takes the four they brought
and who led, then one form a turn (`rankedposition`), derives the opponent's sets from what has
been seen, and reads a turn with the analysis mode's reading (`analysis.Analyzer.run`) for a
fixed time -- the person's game's own, 30 to 45 seconds a move.

It owns one never-exiting thread for the reads (a thread that ran HiGHS spins on exit; see
`rankedweb`), and answers in plain values for the page.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from . import rankedposition as rp
from .analysis import Game, Point, Result
from .hidden import identity
from .notes_ja import notes_ja
from .rankedentry import OpponentSet, RankedError
from .regulation import to_id

#: ``(game, point, seconds, report) -> result``: one read of a typed-in position.
Reader = Callable[[Game, Point, float, Callable[[dict[str, Any]], None]], Result]

NO_READER = "この画面では局面を読めません（読みの設定がありません）"

#: The analysis mode's notes in words a player reads.
_DEEP = re.compile(r"選択的延長のセルで、確率の高い分岐 (\d+) つだけを深さ 2 で読んだ")


def plain_notes(notes: list[str]) -> list[str]:
    out: list[str] = []
    for note in notes:
        found = _DEEP.fullmatch(note)
        text = (f"一部の局面は、確率の高い {found.group(1)} 通りの手だけを先まで読みました"
                if found else note)
        if text not in out:
            out.append(text)
    return out


def make_game(
    reg: Any, mine: Any, board: rp.Board, derived: rp.Derived, index: int,  # noqa: ANN401
) -> tuple[Game, Point, rp.Built]:
    """The analysis mode's input for turn ``index`` of a typed-in match: the position, the two
    sheets (the person's, and the opponent's six as estimated), who has been seen, who led."""
    built = derived.built[index]
    pos = built.position
    form = board.turns[index]
    mine_seen = frozenset(
        identity(pos.sides[0].pokemon[built.mine_slots[i]]) for i in board.mine_seen(index))
    opp_seen = frozenset(
        identity(pos.sides[1].pokemon[built.opp_slots[s]]) for s in board.seen_species(index))
    leads = (frozenset(to_id(mine.sets[i].species) for i in board.brought[:2]),
             # `identity` of the built bodies: the sheet's own species id (IKA-411), not the dex's
             # base species ("Floette" for Floette-Eternal).
             frozenset(identity(pos.sides[1].pokemon[built.opp_slots[s]]) for s in board.opp_leads))
    point = Point(decision=0, turn=int(form["turn"]), position=pos.to_json(), seen=(mine_seen, opp_seen))
    game = Game(label=f"ランクマの検討 ターン {form['turn']}", points=[point],
                teams=(mine, rp.opponent_roster_of(reg, board, derived)),
                leads=leads, side=0, information="hidden-bench")
    return game, point, built


class BoardApp:
    """One session's typed-in match."""

    def __init__(self, app: Any, reader: Reader | None, seconds: float = 40.0) -> None:  # noqa: ANN401
        self.app = app
        self.reader = reader
        self.seconds = seconds
        self.board: rp.Board | None = None
        self.derived: rp.Derived | None = None
        self.hp_mode = "mid"
        #: The person's picks among the candidates left, ``species -> (item, nature, moves)``.
        self.choices: dict[str, tuple[Any, ...]] = {}
        self.read_job: dict[str, Any] = {"state": "idle"}
        self._reads: queue.Queue[tuple[Game, Point, int, rp.Built]] = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    # -- the opponent as the first half has it
    def _base(self) -> dict[str, OpponentSet]:
        return {o.set.species: o for o in self.app.opponent if o is not None}

    def _check_ready(self) -> None:
        app = self.app
        if app.mine is None:
            raise RankedError("自分の構築を読み込んでください")
        why = app.solvable()
        if why is not None:
            raise RankedError(why)

    # -- the steps
    def start(self, brought: list[int], leads: list[str], seen: list[str]) -> dict[str, Any]:
        """The four, and who led and was seen: turn 1 as the game shows it."""
        app = self.app
        with app.lock:
            self._check_ready()
            assert app.mine is not None
            opp_six = [o.set.species for o in app.opponent if o is not None]
            board = rp.Board(app.reg, app.mine, [int(i) for i in brought], opp_six,
                             [str(s) for s in leads], [str(s) for s in seen])
            board.namer = app._name
            board.check()
            sets = {sid: o.set for sid, o in self._base().items()}
            board.turns = [rp.initial_form(app.reg, board, sets)]
            self.board = board
            self.choices = {}
            self._derive()
            self.read_job = {"state": "idle"}
            return self.state()

    def reset(self) -> dict[str, Any]:
        with self.app.lock:
            self.board = None
            self.derived = None
            self.read_job = {"state": "idle"}
            return self.state()

    def save(self, index: int, form: dict[str, Any]) -> dict[str, Any]:
        """A turn's form as the person has it now (the whole form: the page sends what it shows)."""
        with self.app.lock:
            board = self._need_board()
            if not 0 <= index < len(board.turns):
                raise RankedError("そのターンはありません")
            board.turns[index] = rp.normalize(self.app.reg, board, form)
            self._derive()
            self._stale()
            return self.state()

    def choose(self, species: str, alternative: int) -> dict[str, Any]:
        """One of the candidates left for an opposing Pokemon, as its set."""
        with self.app.lock:
            self._need_board()
            assert self.derived is not None
            ref = self.derived.refined.get(species)
            if ref is None or not 0 <= alternative < len(ref.one.alternatives):
                raise RankedError("その候補はありません")
            alt = ref.one.alternatives[alternative]
            self.choices[species] = (alt.item, alt.nature, alt.moves)
            self._derive()
            self._stale()
            return self.state()

    def next_turn(self) -> dict[str, Any]:
        with self.app.lock:
            board = self._need_board()
            if len(board.turns) >= 60:
                raise RankedError("ターンが多すぎます")
            board.turns.append(rp.normalize(self.app.reg, board, rp.next_form(board.turns[-1])))
            self._derive()
            self._stale()
            return self.state()

    def drop_last(self) -> dict[str, Any]:
        with self.app.lock:
            board = self._need_board()
            if len(board.turns) > 1:
                board.turns.pop()
            self._derive()
            self._stale()
            return self.state()

    def _need_board(self) -> rp.Board:
        if self.board is None:
            raise RankedError("局面の入力がまだ始まっていません")
        return self.board

    def _stale(self) -> None:
        if self.read_job.get("state") in ("done", "error"):
            self.read_job = {"state": "idle"}

    def _derive(self) -> None:
        board = self._need_board()
        self.derived = rp.derive(self.app.reg, board, self.app.prior, self._base(), hp_mode=self.hp_mode,
                                 choices=self.choices)

    # -- the read
    def game_and_point(self, index: int) -> tuple[Game, Point, rp.Built]:
        """The analysis mode's input for turn ``index``."""
        board, derived = self._need_board(), self.derived
        assert derived is not None and self.app.mine is not None
        return make_game(self.app.reg, self.app.mine, board, derived, index)

    def solvable(self, index: int) -> str | None:
        if self.reader is None:
            return NO_READER
        if self.board is None or self.derived is None:
            return "局面の入力がまだ始まっていません"
        if not 0 <= index < len(self.board.turns):
            return "そのターンはありません"
        built = self.derived.built[index]
        if built.problems:
            return "局面を直してください: " + built.problems[0]
        form = self.board.turns[index]
        if not [x for x in form["mineActive"] if x is not None]:
            return "場に出ている自分の体がありません"
        if not [x for x in form["theirActive"] if x is not None]:
            return "場に出ている相手の体がありません"
        return None

    def start_read(self, index: int) -> dict[str, Any]:
        with self.app.lock:
            why = self.solvable(index)
            if why is not None:
                raise RankedError(why)
            if self.read_job.get("state") == "running":
                raise RankedError("読んでいる最中です")
            game, point, built = self.game_and_point(index)
            self.read_job = {"state": "running", "startedAt": time.perf_counter(),
                             "seconds": self.seconds, "turn": index}
            self._reads.put((game, point, index, built))
        return self.job()

    def job(self) -> dict[str, Any]:
        with self.app.lock:
            job = dict(self.read_job)
        if job.get("state") == "running":
            job["elapsed"] = time.perf_counter() - job["startedAt"]
        job.pop("startedAt", None)
        return job

    def _work(self) -> None:
        while True:
            game, point, index, built = self._reads.get()

            def report(info: dict[str, Any]) -> None:
                with self.app.lock:
                    self.read_job.update(info)

            try:
                assert self.reader is not None
                result = self.reader(game, point, self.seconds, report)
                summary = self.summarize(built, result)
                with self.app.lock:
                    seconds = time.perf_counter() - self.read_job["startedAt"]
                    self.read_job = {"state": "done", "turn": index, "result": summary, "elapsed": seconds}
            except Exception as exc:  # noqa: BLE001 - the page shows it; the worker must live on
                with self.app.lock:
                    self.read_job = {"state": "error", "turn": index, "error": f"{type(exc).__name__}: {exc}"}

    def summarize(self, built: rp.Built, result: Result) -> dict[str, Any]:
        """A read as the page shows it: our mixture over whole actions and per Pokemon, the
        mixture we model for the opponent, our win probability."""
        from .actions import side_actions
        from .humanplay import parse_choice
        from .progress import action_label

        reg, loc = self.app.reg, self.app.loc
        pos = built.position

        def rows(side: int, choices: list[str], probs: list[float]) -> dict[str, Any]:
            legal = side_actions(reg, pos, side)
            joint: list[dict[str, Any]] = []
            per_slot: dict[int, dict[str, float]] = {}
            for choice, p in zip(choices, probs, strict=True):
                try:
                    action = parse_choice(choice, legal)
                except (ValueError, KeyError, IndexError):
                    continue
                label = action_label(reg, action, pos, side, loc)
                joint.append({"text": str(label), "p": float(p),
                              "slots": [{"slot": part.slot, "text": part.text} for part in label.rich]})
                for part in label.rich:
                    per_slot.setdefault(part.slot, {}).setdefault(part.text, 0.0)
                    per_slot[part.slot][part.text] += float(p)
            joint.sort(key=lambda r: -r["p"])
            slots = []
            for slot in sorted(per_slot):
                party = pos.sides[side].active[slot] if slot < len(pos.sides[side].active) else None
                name = "" if party is None else self.app._name(
                    "species", pos.sides[side].pokemon[party].species)
                slots.append({"slot": slot, "name": name,
                              "actions": sorted(({"text": t, "p": p} for t, p in per_slot[slot].items()),
                                                key=lambda r: -r["p"])[:8]})
            return {"joint": joint[:10], "slots": slots}

        return {
            "value": float(result.value),
            "ours": rows(0, list(result.ours), [float(x) for x in result.strategy]),
            "theirs": rows(1, list(result.theirs), [float(x) for x in result.model]),
            "seconds": float(result.seconds), "steps": int(result.steps), "stop": result.stop,
            "classes": int(result.classes), "exact": bool(result.exact),
            "notes": plain_notes(notes_ja(loc, result.notes)) + (
                ["メモリが少なくなったので、決めた時間より早く読みを止めました"]
                if result.stop == "memory" else []),
            "turn": int(result.turn),
        }

    # -- views
    def state(self) -> dict[str, Any]:
        app = self.app
        board, derived = self.board, self.derived
        if board is None or derived is None:
            return {"started": False, "read": self.job()}
        reg = app.reg
        opp: dict[str, Any] = {}
        for sid in board.opp_six:
            ref = derived.refined[sid]
            narrowed = any(n.species == sid for n in derived.spread_notes)
            one: OpponentSet = replace(
                ref.one, set=derived.sets[sid], sp_source="observed" if narrowed else ref.one.sp_source)
            view = app.view_set(derived.sets[sid], one)
            view["refine"] = {"notes": ref.notes, "members": ref.members, "unmatched": ref.unmatched}
            # What the field's sets are before anything was seen: the tier the estimate reads.
            base = self._base()[sid].belief
            members = list(base.members) if base is not None else app.prior.members.get(sid, [])
            counts: dict[str, int] = {}
            for m in members:
                for mv in m.moves:
                    counts[mv] = counts.get(mv, 0) + 1
            view["fieldMoves"] = [{"id": mv, "name": app._name("move", mv), "count": n}
                                  for mv, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:24]
            items: dict[str, int] = {}
            for m in members:
                if m.item:
                    items[m.item] = items.get(m.item, 0) + 1
            view["fieldItems"] = [{"id": i, "name": app._name("item", i), "count": n}
                                  for i, n in sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))][:12]
            opp[sid] = view
        turns_view = []
        for t, form in enumerate(board.turns):
            built = derived.built[t]
            pos = built.position
            hp: dict[str, Any] = {}
            for sid, slot in built.opp_slots.items():
                if sid in form["theirs"]:
                    mon = pos.sides[1].pokemon[slot]
                    state = form["theirs"][sid]
                    b = rp.band(state["pct"], mon.maxhp, colour=state["colour"],
                                floor_rule=rp.uses_floor_display(reg))
                    hp[sid] = {"hp": mon.hp, "max": mon.maxhp, "low": b.low, "high": b.high}
            turns_view.append({"form": form, "problems": built.problems, "notes": built.notes,
                               "oppHp": hp, "readable": self.solvable(t) is None})
        return {
            "started": True,
            "brought": board.brought, "oppLeads": board.opp_leads, "oppSeen": board.opp_seen,
            "oppSix": [{"id": s, "name": app._name("species", s)} for s in board.opp_six],
            "mine": [{"index": i, "view": app.view_set(s)} for i, s in enumerate(board.mine.sets)],
            "mineMaxHp": {str(i): rp.max_hp(reg, board.mine.sets[i]) for i in board.brought},
            "turns": turns_view,
            "opp": opp,
            "spread": [{"species": app._name("species", n.species), "id": n.species, "stat": n.stat,
                        "low": n.low, "high": n.high, "statLow": n.stat_low, "statHigh": n.stat_high,
                        "before": n.before, "after": n.after, "because": n.because}
                       for n in derived.spread_notes],
            "lines": derived.lines,
            "read": self.job(),
            "seconds": self.seconds,
            "reader": self.reader is not None,
            "options": self.options(),
        }

    def options(self) -> dict[str, Any]:
        reg, app = self.app.reg, self.app
        return {
            "weather": [{"id": w, "name": app._name("move", w) or w} for w in rp.WEATHERS],
            "terrain": [{"id": w, "name": app._name("move", w) or w} for w in rp.TERRAINS],
            "status": [{"id": s, "name": app._name("status", s)} for s in rp.STATUSES],
            "timed": [{"id": c, "name": app._name("move", c) or c, "turns": n} for c, n in rp.TIMED.items()],
            "layered": [{"id": c, "name": app._name("move", c) or c, "max": n}
                        for c, n in rp.LAYERED.items()],
            "abilities": sorted(({"id": a, "name": app._name("ability", a)} for a in reg.abilities),
                                key=lambda x: x["name"]),
        }
