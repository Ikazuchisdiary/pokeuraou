"""The ranked-entry screen's logic and its small web server (IKA-407).

`RankedApp` holds one person's session -- their pasted team, the six estimated opponent sets,
the running selection read -- and answers in plain values; `RankedServer` puts it behind a few
JSON routes and the page (`web/ranked.html`, `ranked.js`, `ranked.css`, which take their look
from the game page's `live.css`). What the estimate is and why is `pokeuraou.rankedentry`'s.

The read runs on one thread that never exits: a thread that ran HiGHS spins on exit
(records: highs-thread-exit-hangs-thread-start), so the jobs go through one long-lived worker.
"""

from __future__ import annotations

import html
import json
import queue
import re
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .humanplay import species_types, sprite_id
from .rankedboard import BoardApp, Reader
from .rankedentry import (
    WHOLE_MIN,
    FieldPrior,
    LearnedIds,
    OpponentSet,
    RankedError,
    UnknownSpecies,
    apply_override,
    blank_set,
    opponent_roster,
    roster_from_paste,
    selection_summary,
)
from .regulation import STAT_IDS, Regulation, to_id
from .selection_book import BookEntry
from .teams import Roster

WEB = Path(__file__).resolve().parent / "web"
SPRITE_URL = "https://play.pokemonshowdown.com/sprites/gen5/{id}.png"
FILES = {
    "/": ("ranked.html", "text/html; charset=utf-8"),
    "/ranked.html": ("ranked.html", "text/html; charset=utf-8"),
    "/ranked.css": ("ranked.css", "text/css; charset=utf-8"),
    "/ranked.js": ("ranked.js", "text/javascript; charset=utf-8"),
    "/position": ("ranked-position.html", "text/html; charset=utf-8"),
    "/ranked-position.html": ("ranked-position.html", "text/html; charset=utf-8"),
    "/ranked-position.css": ("ranked-position.css", "text/css; charset=utf-8"),
    "/ranked-position.js": ("ranked-position.js", "text/javascript; charset=utf-8"),
    "/live.css": ("live.css", "text/css; charset=utf-8"),
}

#: ``(mine, opponent, report) -> (entry, model)``: the selection read. ``report`` is told
#: ``{"seconds": budget}`` when the read starts.
Solver = Callable[[Roster, Roster, Callable[[dict[str, Any]], None]], tuple[BookEntry, str]]


def _kind_label(one: OpponentSet) -> str:
    if one.kind == "whole":
        return f"この編成でよく使われる型（{one.species_n} 体中 {one.whole_n} 体）"
    if one.kind == "part":
        top = one.alternatives[0].count if one.alternatives else 0
        return f"部分ごとの最頻（{one.species_n} 体の中で、最頻の型でも {top} 体）"
    if one.kind == "candidate":
        return f"候補の型（{one.whole_n} 体）"
    if one.kind == "override":
        return "上書き"
    return "手入力"


def _thin_reasons(one: OpponentSet) -> list[str]:
    """Why this estimate stands on little (the screen sets such a card apart): few members
    behind the set, a species the field does not show, a spread made up from base stats."""
    out: list[str] = []
    if one.kind == "part":
        top = one.alternatives[0].count if one.alternatives else 0
        out.append(f"大会でもこの種族の型はばらばらで、最頻の型でも {top} 体だけです")
    elif one.kind == "candidate" and one.whole_n < WHOLE_MIN:
        out.append(f"選んだ型は大会で {one.whole_n} 体だけです")
    elif one.kind == "manual":
        out.append("大会に出ていない種族なので、型は手で入れる必要があります")
    if one.sp_provisional:
        out.append("配分は仮です（基礎能力から作った中立の配分）")
    return out


def _basis_label(one: OpponentSet) -> str:
    """What teams the sets were read from: how many of the opposing six they share."""
    belief = one.belief
    if belief is None:
        return ""
    if belief.overlap <= 1:
        return f"この種族を使った構築すべて {belief.teams_in_tier()} 件（編成では絞れませんでした）"
    parts = "・".join(f"{k} 種族一致 {n} 件" for k, n in belief.tiers if k >= belief.overlap)
    return (f"相手の 6 種族のうち {belief.overlap} 種族以上が同じ構築 {belief.teams_in_tier()} 件"
            f"（{parts}）")


def _sp_label(one: OpponentSet) -> str:
    if one.sp_source == "pool-nature":
        return f"プールの同じ種族・同じ性格の最頻（{one.sp_n} 件）"
    if one.sp_source == "pool-species":
        return f"プールの同じ種族の最頻（{one.sp_n} 件、性格は違う）"
    if one.sp_source == "neutral":
        return "基礎能力から作った中立の配分"
    if one.sp_source == "observed":
        return "見えた動き（先に動いた・受けたダメージ）から絞った配分"
    return "上書き"


class RankedApp:
    """One session of the screen."""

    def __init__(self, reg: Regulation, prior: FieldPrior, learned: LearnedIds | None, loc: Any,  # noqa: ANN401
                 solver: Solver, *, seconds: float | None = None, model: str = "",
                 reader: Reader | None = None, read_seconds: float = 40.0) -> None:
        self.reg = reg
        self.prior = prior
        self.learned = learned
        self.loc = loc
        self.solver = solver
        self.seconds = seconds
        self.model = model
        self.mine: Roster | None = None
        self.opponent: list[OpponentSet | None] = []
        self.team_problems: list[str] = []
        self.lock = threading.RLock()
        self._job: dict[str, Any] = {"state": "idle"}
        self._jobs: queue.Queue[tuple[Roster, Roster, list[OpponentSet]]] = queue.Queue()
        self._names = self._species_names()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()
        #: The typed-in position (IKA-408), with its own read thread.
        self.board = BoardApp(self, reader, read_seconds)

    # -- names
    def _species_names(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for found in self.reg.species.values():
            if not found.team_legal or found.is_mega:
                continue
            out[to_id(found.name)] = found.id
            out[to_id(found.id)] = found.id
            if self.loc is not None:
                out[to_id(self.loc.species(found.id))] = found.id
        return out

    def species_options(self) -> list[dict[str, str]]:
        seen: dict[str, dict[str, str]] = {}
        for found in self.reg.species.values():
            if found.team_legal and not found.is_mega:
                seen[found.id] = {
                    "id": found.id,
                    "name": self.loc.species(found.id) if self.loc is not None else found.name,
                    "en": found.name,
                    "inField": found.id in self.prior.members,
                }
        return sorted(seen.values(), key=lambda s: (not s["inField"], s["en"]))

    def resolve_species(self, text: str) -> str:
        key = to_id(text)
        found = self._names.get(key)
        if found is None:
            raise RankedError(f"種族 {text!r} がわかりません（この規則で使える種族の名前を入れてください）")
        return found

    # -- views
    def _name(self, kind: str, value: str | None) -> str:
        if self.loc is None:
            return str(value) if value else ""
        return str(getattr(self.loc, kind)(value))

    def _pair(self, kind: str, value: str | None) -> dict[str, str] | None:
        if not value:
            return None
        return {"id": value, "name": self._name(kind, value)}

    def view_set(self, one: Any, extra: OpponentSet | None = None) -> dict[str, Any]:  # noqa: ANN401
        """A set (a `SampledSet`, with its estimate's record when ``extra`` is given) as the
        screen shows it."""
        reg = self.reg
        found = reg.species[one.species]
        unlearned = [] if self.learned is None else [
            {"kind": kind, "id": ident,
             "name": self._name({"species": "species", "move": "move", "item": "item",
                                 "ability": "ability"}[kind], ident)}
            for kind, ident in self.learned.missing(one)
        ]
        out: dict[str, Any] = {
            "species": {"id": one.species, "name": self._name("species", one.species),
                        "sprite": sprite_id(reg, one.species),
                        "types": species_types(reg, one.species)},
            "ability": self._pair("ability", one.ability),
            "abilityChoices": [{"id": to_id(a), "name": self._name("ability", to_id(a))}
                               for a in found.abilities],
            "item": self._pair("item", one.item),
            "nature": {"id": one.nature, "name": self._name("nature", one.nature)},
            "moves": [self._pair("move", m) for m in one.moves],
            "sp": [int(one.sp.get(s, 0)) for s in STAT_IDS],
            "spTotal": int(sum(one.sp.get(s, 0) for s in STAT_IDS)),
            "unlearned": unlearned,
        }
        if extra is not None:
            out.update({
                "kind": extra.kind, "kindLabel": _kind_label(extra),
                "basisLabel": _basis_label(extra),
                "speciesN": extra.species_n, "wholeN": extra.whole_n,
                "alternatives": [
                    {"index": i, "count": c.count,
                     "label": " / ".join([self._name("item", c.item) if c.item else "持ち物なし",
                                          self._name("nature", c.nature)]),
                     "moves": [self._name("move", m) for m in c.moves]}
                    for i, c in enumerate(extra.alternatives)
                ],
                "chosen": next((i for i, c in enumerate(extra.alternatives)
                                if extra.kind in ("whole", "candidate")
                                and tuple(sorted(one.moves)) == c.moves
                                and one.item == c.item and one.nature == c.nature), -1),
                "spSource": extra.sp_source, "spLabel": _sp_label(extra), "spN": extra.sp_n,
                "spProvisional": extra.sp_provisional,
                "thin": _thin_reasons(extra),
                "notes": list(extra.notes), "overridden": list(extra.overridden),
            })
        return out

    def meta(self) -> dict[str, Any]:
        reg = self.reg
        return {
            "regulation": reg.meta.format_name,
            "event": self.prior.source_line(),
            "spreadNote": self.prior.spread_line(),
            "estNote": self.prior.short_line(),
            "eventName": self.prior.standings.event,
            "eventTeams": len(self.prior.standings.teams),
            "species": self.species_options(),
            "items": sorted(({"id": i, "name": self._name("item", i)} for i in reg.items),
                            key=lambda x: x["name"]),
            "moves": sorted(({"id": m, "name": self._name("move", m)} for m in reg.moves),
                            key=lambda x: x["name"]),
            "natures": [{"id": n.name, "name": self._name("nature", n.name)} for n in reg.real_game_natures],
            "stats": list(STAT_IDS),
            "statNames": [self._name("stat", s) if self.loc is None else self.loc.stat(s, short=True)
                          for s in STAT_IDS],
            "spLimit": reg.meta.sp_limit, "spMax": reg.meta.sp_per_stat_max,
            "seconds": self.seconds,
            "readSeconds": self.board.seconds,
            "model": self.model,
            "learned": None if self.learned is None else list(self.learned.sources),
        }

    def state(self) -> dict[str, Any]:
        return {
            "mine": None if self.mine is None else [self.view_set(s) for s in self.mine.sets],
            "opponent": [None if o is None else self.view_set(o.set, o) for o in self.opponent],
            "teamProblems": list(self.team_problems),
            "job": self.job(),
        }

    # -- steps
    def set_mine(self, text: str) -> dict[str, Any]:
        result = roster_from_paste(self.reg, text)
        problems = []
        for p in result.problems:
            where = "" if p["member"] is None else f"{p['member'] + 1} 体目（{p['writtenAs']}）: "
            problems.append(f"{where}{p['field']}: {p['reason']}")
        with self.lock:
            self.mine = result.roster
            self._stale()
        return {"ok": result.roster is not None, "problems": problems, "warnings": result.warnings,
                "team": None if result.roster is None else [self.view_set(s) for s in result.roster.sets]}

    def set_opponent(self, names: list[str]) -> dict[str, Any]:
        ids: list[str] = []
        errors: list[str] = []
        for text in names:
            try:
                ids.append(self.resolve_species(text))
            except RankedError as exc:
                errors.append(str(exc))
        if errors:
            raise RankedError(" / ".join(errors))
        present = [i for i in ids if i in self.prior.members]
        filled = self.prior.fill_team(present, ids)[0] if present else []
        by_species = {o.set.species: o for o in filled}
        sets: list[OpponentSet | None] = [
            by_species[sid] if sid in by_species else blank_set(self.reg, sid) for sid in ids
        ]
        with self.lock:
            self.opponent = sets
            self._stale()
            self._recheck()
        return self.state()

    def _stale(self) -> None:
        """A finished read no longer belongs to what is entered (a running one is left to end)."""
        if self._job.get("state") in ("done", "error"):
            self._job = {"state": "idle"}

    def _slot(self, index: int) -> OpponentSet:
        if not 0 <= index < len(self.opponent) or self.opponent[index] is None:
            raise RankedError("その枠に相手の型がありません")
        one = self.opponent[index]
        assert one is not None
        return one

    def override(self, index: int, patch: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.opponent[index] = apply_override(self.reg, self._slot(index), patch)
            self._stale()
            self._recheck()
        return self.state()

    def choose(self, index: int, alternative: int) -> dict[str, Any]:
        with self.lock:
            self.opponent[index] = self.prior.choose(self._slot(index), alternative)
            self._stale()
            self._recheck()
        return self.state()

    def _recheck(self) -> None:
        """What is wrong with the six as they stand: size, clauses, sets still to be entered."""
        reg = self.reg
        problems: list[str] = []
        have = [o for o in self.opponent if o is not None]
        if len(have) != reg.meta.team_size:
            problems.append(f"{len(have)} 種族です。{reg.meta.team_size} 種族を入れてください")
        manual = [self._name("species", o.set.species) for o in have
                  if o.kind == "manual" and not o.set.moves]
        if manual:
            problems.append("大会データに無い種族は技・持ち物などを手で入れてください: " + ", ".join(manual))
        sets = [o.set for o in have]
        bases = [reg.species[s.species].base_species for s in sets]
        doubled = sorted({b for b in bases if bases.count(b) > 1})
        if doubled:
            problems.append(f"同じ種族が 2 体います: {', '.join(doubled)}")
        items = [s.item for s in sets if s.item]
        twice = sorted({i for i in items if items.count(i) > 1})
        if reg.meta.item_clause is not None and twice:
            problems.append("持ち物が重複しています: " + ", ".join(self._name("item", i) for i in twice))
        self.team_problems = problems

    # -- the read
    def solvable(self) -> str | None:
        """Why the selection cannot be read yet, or None."""
        if self.mine is None:
            return "自分の構築を読み込んでください"
        if len(self.opponent) != self.reg.meta.team_size or any(o is None for o in self.opponent):
            return f"相手の {self.reg.meta.team_size} 種族を入れてください"
        empty = [self._name("species", o.set.species) for o in self.opponent if o and not o.set.moves]
        if empty:
            return "技が入っていません: " + ", ".join(empty)
        clash = [p for p in self.team_problems if "重複" in p or "同じ種族" in p]
        if clash:
            return clash[0]
        return None

    def start(self) -> dict[str, Any]:
        with self.lock:
            why = self.solvable()
            if why is not None:
                raise RankedError(why)
            if self._job.get("state") == "running":
                raise RankedError("読んでいる最中です")
            assert self.mine is not None
            sets = [o for o in self.opponent if o is not None]
            roster = opponent_roster(self.reg, [o.set for o in sets])
            self._job = {"state": "running", "startedAt": time.perf_counter(), "seconds": self.seconds}
            self._jobs.put((self.mine, roster, sets))
        return self.job()

    def job(self) -> dict[str, Any]:
        with self.lock:
            job = dict(self._job)
        if job.get("state") == "running":
            job["elapsed"] = time.perf_counter() - job["startedAt"]
        job.pop("startedAt", None)
        return job

    def _work(self) -> None:
        while True:
            mine, roster, sets = self._jobs.get()

            def report(info: dict[str, Any]) -> None:
                with self.lock:
                    self._job.update(info)

            try:
                entry, model = self.solver(mine, roster, report)
                summary = selection_summary(entry, self.reg, mine.sets, roster.sets, self.loc)
                summary["model"] = model
                summary["estimated"] = [
                    {"species": self._name("species", o.set.species), "kind": _kind_label(o),
                     "provisional": o.sp_provisional, "overridden": list(o.overridden),
                     "unlearned": self.view_set(o.set, o)["unlearned"]}
                    for o in sets
                ]
                summary["mineUnlearned"] = [
                    {"species": self._name("species", s.species), "unlearned": self.view_set(s)["unlearned"]}
                    for s in mine.sets
                ]
                with self.lock:
                    seconds = time.perf_counter() - self._job["startedAt"]
                    self._job = {"state": "done", "result": summary, "elapsed": seconds}
            except Exception as exc:  # noqa: BLE001 - the page shows it; the worker must live on
                with self.lock:
                    self._job = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}


class RankedServer:
    """The page and its JSON routes on ``host:port`` (the game page's server is separate)."""

    def __init__(self, app: RankedApp, host: str = "127.0.0.1", port: int = 0,
                 *, sprite_url: str | None = None, web: Path = WEB) -> None:
        self.app = app
        self.web = web
        self.sprite_url = SPRITE_URL if sprite_url is None else sprite_url
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: ANN401
                del args

            def _json(self, code: int, payload: Any) -> None:  # noqa: ANN401
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/api/meta":
                    self._json(200, app.meta())
                elif path == "/api/state":
                    self._json(200, app.state())
                elif path == "/api/job":
                    self._json(200, app.job())
                elif path == "/api/board":
                    with app.lock:
                        self._json(200, app.board.state())
                elif path == "/api/board/job":
                    self._json(200, app.board.job())
                elif path in FILES:
                    name, kind = FILES[path]
                    body = (server.web / name).read_bytes()
                    if name.endswith(".html"):
                        body = re.sub(
                            rb'<meta name="sprite-url" content="[^"]*">',
                            lambda _m: b'<meta name="sprite-url" content="'
                            + html.escape(server.sprite_url).encode("utf-8") + b'">', body)
                    self.send_response(200)
                    self.send_header("Content-Type", kind)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                    path = self.path.split("?", 1)[0]
                    if path == "/api/mine":
                        self._json(200, app.set_mine(str(data.get("text", ""))))
                    elif path == "/api/opponent":
                        self._json(200, app.set_opponent([str(x) for x in data.get("species", [])]))
                    elif path == "/api/override":
                        self._json(200, app.override(int(data["index"]), dict(data["patch"])))
                    elif path == "/api/choose":
                        self._json(200, app.choose(int(data["index"]), int(data["alternative"])))
                    elif path == "/api/solve":
                        self._json(200, app.start())
                    elif path == "/api/board/start":
                        self._json(200, app.board.start(
                            [int(x) for x in data["brought"]], [str(x) for x in data["leads"]],
                            [str(x) for x in data.get("seen", [])]))
                    elif path == "/api/board/save":
                        self._json(200, app.board.save(int(data["index"]), dict(data["form"])))
                    elif path == "/api/board/next":
                        self._json(200, app.board.next_turn())
                    elif path == "/api/board/drop":
                        self._json(200, app.board.drop_last())
                    elif path == "/api/board/reset":
                        self._json(200, app.board.reset())
                    elif path == "/api/board/read":
                        self._json(200, app.board.start_read(int(data["index"])))
                    else:
                        self.send_error(404)
                except (RankedError, UnknownSpecies) as exc:
                    self._json(400, {"error": str(exc)})
                except (KeyError, ValueError, TypeError) as exc:
                    self._json(400, {"error": f"入力を読めません: {exc}"})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> RankedServer:
        self.thread.start()
        return self

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
