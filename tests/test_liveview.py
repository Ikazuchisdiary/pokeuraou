"""IKA-332: the agent's answer while it forms, and the page that shows it.

What these hold:

- **watching changes nothing**: a game on the count clock with a listener is the same game,
  byte for byte, as without one; a deepened `search` with a progress callback returns the
  same arrays as without (and the callback was called: the positive control);
- **the last step is the answer**: each move's last snapshot is ``done`` and its mixture and
  value are the recorded ones, bit for bit;
- **the principal variation** is read off the tree: deepened pairs open their chance
  branches (weights summing to one), others are the leaf's value;
- **the wire** gives back what was packed (the mixture as f64, so exactly);
- **the server** serves the page and streams the game to a scripted person at the socket,
  whose answers make the same game as the same answers read from a script.

The games are test_humanplay's: the M-B roster in two variants, hp-share scored in the port.
"""

from __future__ import annotations

import copy
import json
import threading
import urllib.request

import pytest

from pokeuraou import humanplay, liveview
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.humanplay import Agent, NodeTime, PolicyPerson, ScriptPerson
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.pool import load_pool
from pokeuraou.progress import Reader, Recorder, Snapshot
from pokeuraou.regulation import repo_root
from pokeuraou.search import search
from pokeuraou.selfplay import position_from_sets

PRICES = {("local", 1): NodeTime(fixed_ms=10.0, cell_ms=0.05)}


@pytest.fixture(autouse=True)
def _prices(monkeypatch):  # noqa: ANN001, ANN202
    monkeypatch.setattr(humanplay, "NODE_TIME", PRICES)


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    base = json.loads(
        (repo_root() / "configs" / "teams" / "rizabanadohido.json").read_text(encoding="utf-8")
    )["team"]
    rotated = copy.deepcopy(base[2:] + base[:2])
    data = {
        "id": "test-pool", "name": "test", "regulation": "gen9championsvgc2026regmb",
        "character": "a test pool",
        "validatedAgainst": {"formatId": "gen9championsvgc2026regmb"},
        "teams": [{"id": f"t{i}", "name": f"team {i}", "team": t}
                  for i, t in enumerate([base, rotated])],
    }
    path = tmp_path_factory.mktemp("pool") / "p.json"
    path.write_bytes(json.dumps(data).encode("utf-8"))
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _agent(pool) -> Agent:  # noqa: ANN001
    return Agent(reg=pool.reg, evaluate=None, name="hp-share", seconds=0.4, cores=1,
                 clock="count", rank_by_leaf=False, rank_fill="refs2")


def _play(pool, person, *, seed: int, listener=None, turns: int = 3, interval_ms: float = 0.0):  # noqa: ANN001, ANN202
    return humanplay.play(
        _agent(pool), person, (pool.teams[0], pool.teams[1]), agent_side=1, seed=seed,
        max_turns=turns, listener=listener, interval_ms=interval_ms,
    )


class Heard:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def __call__(self, kind: str, payload: object) -> None:
        self.events.append((kind, payload))

    def of(self, kind: str) -> list:
        return [p for k, p in self.events if k == kind]


# ------------------------------------------------------------------------ the engine


def test_a_listener_changes_no_game_and_each_move_ends_on_its_answer(pool) -> None:  # noqa: ANN001
    plain, _, _ = _play(pool, PolicyPerson("random", 7), seed=7)
    heard = Heard()
    watched, _, _ = _play(pool, PolicyPerson("random", 7), seed=7, listener=heard)
    assert json.dumps(watched, ensure_ascii=False) == json.dumps(plain, ensure_ascii=False)

    steps = heard.of("step")
    answers = heard.of("answer")
    moves = [d for d in watched["decisions"] if d["kind"] == "move"]
    assert answers and len(answers) == len(moves)
    # The positive control: the callback saw deepening steps, on a Bayesian root.
    assert any(s.kind == "refine" for s in steps)
    assert any(s.classes for s in steps)
    for answer in answers:
        mine = [s for s in steps if s.decision == answer["decision"]]
        assert mine[0].kind == "start" and mine[-1].kind == "done"
        last = mine[-1]
        assert last.our_p.tolist() == list(answer["strategy"])
        assert last.value0 == answer["value0"]
        assert [a.to_choice() for a in last.ours] == [a.to_choice() for a in answer["actions"]]
        assert answer["steps"] >= len(mine)
        record = watched["decisions"][answer["decision"]]
        assert record["foePolicy"] == last.our_p.tolist()  # the agent is side 1
        assert record["searchValue"] == last.value0
        # Values from the agent's side; the classes' weights are the belief.
        assert last.value == pytest.approx(1.0 - last.value0)
        if last.classes:
            assert sum(c.weight for c in last.classes) == pytest.approx(1.0)
    # Events in order: the board before the agent thinks, the prompt after its answer.
    kinds = [k for k, _ in heard.events]
    assert kinds[:2] == ["sheets", "select"] and kinds[-1] == "end"
    first_think = kinds.index("think")
    assert "board" in kinds[:first_think]
    assert kinds.index("prompt") > kinds.index("answer")


def test_the_principal_variation_reads_the_tree(pool) -> None:  # noqa: ANN001
    heard = Heard()
    _play(pool, PolicyPerson("random", 7), seed=7, listener=heard)
    done = [s for s in heard.of("step") if s.kind == "done"]
    deep = [pair for s in done for pair in s.pv if pair.read == "deep"]
    assert deep, "no deepened pair in the principal variation"
    for s in done:
        assert s.pv and sum(p.p for p in s.pv) <= 1.0 + 1e-9
        assert [p.p for p in s.pv] == sorted((p.p for p in s.pv), reverse=True)
    opened = 0
    for pair in deep:
        assert pair.branches
        assert sum(b.weight for b in pair.branches) == pytest.approx(1.0)
        for branch in pair.branches:
            assert branch.what
            if branch.node is not None:
                opened += 1
                assert 0.0 <= branch.node.value <= 1.0
    assert opened, "no branch opened a node"
    leafs = [pair for s in done for pair in s.pv if pair.read == "leaf"]
    assert all(not pair.branches for pair in leafs)


def test_a_deepened_search_is_the_same_with_a_callback(pool) -> None:  # noqa: ANN001
    reg = pool.reg
    pos = position_from_sets(reg, list(pool.teams[0].sets[:4]), list(pool.teams[1].sets[:4]))
    rows = narrow(reg, pos, 0, limit=6).actions
    # Reversed, so the two sides' strategies do not sit at the same indices.
    cols = narrow(reg, pos, 1, limit=6).actions[::-1]
    plain = search(reg, pos, rows, cols, HP_SHARE.batch, budget=Budget.matrix(), deepen=300)
    seen: list[Snapshot] = []
    reader = Reader(reg, pos, 0)
    recorder = Recorder(reader, seen.append, decision=0, turn=pos.turn, started=0.0,
                        interval_ms=0.0, clock=lambda: 0.0)
    tree: dict = {}
    other: list[Snapshot] = []
    side1 = Reader(reg, pos, 1)

    def watch(step) -> None:  # noqa: ANN001
        recorder(step)
        if step.kind == "done":
            other.append(side1.read(step))
            tree.update({cell: [w for w, _ in kids] for cell, kids in step.root.children.items()})

    watched = search(reg, pos, rows, cols, HP_SHARE.batch, budget=Budget.matrix(), deepen=300,
                     progress=watch)
    for name in ("row_strategy", "col_strategy", "row_ev_loss"):
        assert getattr(watched.equilibrium, name).tobytes() == getattr(plain.equilibrium, name).tobytes()
    assert watched.payoff.tobytes() == plain.payoff.tobytes()
    assert watched.deepened == plain.deepened
    assert recorder.calls == plain.deepened.expanded + plain.deepened.refused + 2
    assert seen[-1].kind == "done" and seen[-1].our_p.tolist() == plain.equilibrium.row_strategy.tolist()
    assert seen[-1].cells == plain.deepened.cells and seen[-1].depth == plain.deepened.depth
    # Side 1 reads the column strategy (the two differ here, so a swap would show).
    x, y = plain.equilibrium.row_strategy, plain.equilibrium.col_strategy
    assert x.tolist() != y.tolist()
    assert other[-1].our_p.tolist() == y.tolist() and other[-1].their_p.tolist() == x.tolist()
    assert other[-1].value == 1.0 - seen[-1].value0
    # The principal variation's branches are the tree's, weight for weight.
    deep = [pair for pair in seen[-1].pv if pair.read == "deep"]
    assert deep and any(len(tree[pair.cell]) > 1 for pair in deep)
    for pair in seen[-1].pv:
        assert (pair.read == "deep") == (pair.cell in tree)
        if pair.read == "deep":
            assert [b.weight for b in pair.branches] == tree[pair.cell]
    # Not deepened: the depth-1 answer, reported as start and done.
    seen.clear()
    depth1 = search(reg, pos, rows, cols, HP_SHARE.batch, budget=Budget.matrix(),
                    progress=Recorder(Reader(reg, pos, 1), seen.append, decision=0, turn=1,
                                      started=0.0, interval_ms=0.0, clock=lambda: 0.0))
    assert [s.kind for s in seen] == ["start", "done"]
    # Side 1's view: the column strategy is its mixture, the value its own.
    assert seen[-1].our_p.tolist() == depth1.equilibrium.col_strategy.tolist()
    assert seen[-1].value == 1.0 - float(depth1.equilibrium.value)
    assert all(p.read == "leaf" for p in seen[-1].pv)
    with pytest.raises(ValueError, match="progress"):
        search(reg, pos, rows, cols, HP_SHARE.batch, budget=Budget.matrix(), depth=2,
               progress=lambda step: None)


def test_the_recorder_sends_the_first_and_the_last_and_throttles_between(pool) -> None:  # noqa: ANN001
    reg = pool.reg
    pos = position_from_sets(reg, list(pool.teams[0].sets[:4]), list(pool.teams[1].sets[:4]))
    rows = narrow(reg, pos, 0, limit=6).actions
    cols = narrow(reg, pos, 1, limit=6).actions
    seen: list[Snapshot] = []
    recorder = Recorder(Reader(reg, pos, 0), seen.append, decision=0, turn=1, started=0.0,
                        interval_ms=1e9, clock=lambda: 1.0)
    search(reg, pos, rows, cols, HP_SHARE.batch, budget=Budget.matrix(), deepen=200,
           progress=recorder)
    assert recorder.calls > 2
    assert [s.kind for s in seen] == ["start", "done"] and recorder.sent == 2


# ------------------------------------------------------------------------ the wire


def test_the_wire_gives_back_what_it_packed(pool) -> None:  # noqa: ANN001
    heard = Heard()
    _play(pool, PolicyPerson("random", 7), seed=7, listener=heard)
    wire, decoder = liveview.Wire(), liveview.Decoder()
    got = []
    for kind, payload in heard.events:
        frames = wire.step(payload) if kind == "step" else wire.event(kind, payload)
        for frame in frames:
            back = decoder.feed(frame)
            if back is not None:
                got.append((kind, payload, back))
    steps = [(p, b) for k, p, b in got if k == "step"]
    assert steps
    for snap, back in steps:
        assert back["kind"] == snap.kind and back["cells"] == snap.cells
        assert back["ourP"] == snap.our_p.tolist()  # f64: exact
        assert back["ours"] == snap.our_labels and back["theirs"] == snap.their_labels
        assert back["value0"] == snap.value0
        assert len(back["classes"]) == len(snap.classes)
        assert [p["read"] for p in back["pv"]] == [p.read for p in snap.pv]
        assert [len(p["branches"]) for p in back["pv"]] == [len(p.branches) for p in snap.pv]
    events = [b for k, _, b in got if k != "step"]
    assert events[0]["type"] == "sheets" and events[-1]["type"] == "end"
    prompt = next(b for b in events if b["type"] == "prompt")
    assert prompt["choices"] and len(prompt["slots"]) == len(prompt["choices"])
    # Each string went once.
    assert len(decoder.strings) == len(set(decoder.strings))


def test_the_record_file_reads_back(tmp_path) -> None:  # noqa: ANN001
    clock = iter([0.0, 0.5, 1.25])
    sink = liveview.FileSink(tmp_path / "live.bin", clock=lambda: next(clock))
    sink(b"\x02{}")
    sink(b"\x02{\"a\": 1}")
    sink.close()
    got = list(liveview.read_record(tmp_path / "live.bin"))
    assert got == [(0.5, b"\x02{}"), (1.25, b"\x02{\"a\": 1}")]


# ------------------------------------------------------------------------ the server


def test_the_server_serves_the_page_and_a_scripted_person_plays_through_it(pool) -> None:  # noqa: ANN001
    server = liveview.LiveServer("127.0.0.1", 0).start()
    try:
        host, port = server.address
        page = urllib.request.urlopen(server.url, timeout=10).read().decode("utf-8")  # noqa: S310
        assert "対局と読み" in page and "live-data.js" in page
        data = urllib.request.urlopen(server.url + "live-data.js", timeout=10).read()  # noqa: S310
        assert b"WebSocket" in data and b"decodeStep" in data

        result: dict = {}

        def run() -> None:
            result["game"] = _play(
                pool, liveview.WebPerson(server, timeout=120), seed=9,
                listener=server.listener, turns=2, interval_ms=0.0,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        sock = liveview.ws_connect(host, port)
        decoder = liveview.Decoder()
        seen: list[dict] = []
        while True:
            opcode, payload = liveview.ws_read(sock)
            assert opcode == 2
            got = decoder.feed(payload)
            if got is None:
                continue
            seen.append(got)
            if got["type"] == "select":
                # A line that does not parse is said back, then the right one is taken.
                liveview.ws_send_text(sock, json.dumps({"line": "1 1 2 3"}))
                liveview.ws_send_text(sock, json.dumps({"line": "1 2 3 4"}))
            elif got["type"] == "prompt":
                liveview.ws_send_text(sock, json.dumps({"line": got["choices"][0]}))
            elif got["type"] == "end":
                break
        sock.close()
        thread.join(timeout=120)
        record, _, _ = result["game"]
    finally:
        server.close()
    types = [s["type"] for s in seen]
    assert "error" in types and "step" in types and "board" in types
    done = [s for s in seen if s["type"] == "step" and s["kind"] == "done"]
    answers = [s for s in seen if s["type"] == "answer"]
    assert len(done) == len(answers) > 0
    for step, answer in zip(done, answers, strict=True):
        assert step["ourP"] == answer["strategy"]
    # The same answers from a script are the same game.
    again, _, _ = _play(pool, ScriptPerson(record["human"]["inputs"]), seed=9, turns=2)
    assert json.dumps(again, ensure_ascii=False) == json.dumps(record, ensure_ascii=False)
    assert record["human"]["inputs"][0] == "1 2 3 4"


def test_a_page_that_connects_late_is_sent_the_game_so_far(pool) -> None:  # noqa: ANN001
    server = liveview.LiveServer("127.0.0.1", 0).start()
    try:
        _play(pool, PolicyPerson("first"), seed=4, listener=server.listener, turns=2)
        host, port = server.address
        sock = liveview.ws_connect(host, port)
        decoder = liveview.Decoder()
        seen = []
        sock.settimeout(5)
        while True:
            opcode, payload = liveview.ws_read(sock)
            got = decoder.feed(payload)
            if got is not None:
                seen.append(got)
                if got["type"] == "end":
                    break
        sock.close()
    finally:
        server.close()
    steps = [s for s in seen if s["type"] == "step"]
    # Of each finished move only its last step is kept.
    assert steps and all(s["kind"] == "done" for s in steps)
    assert len(steps) == len([s for s in seen if s["type"] == "answer"])


# ------------------------------------------------------------------------ the page's data


def test_the_events_carry_what_the_page_draws(pool) -> None:  # noqa: ANN001
    heard = Heard()
    _play(pool, PolicyPerson("random", 7), seed=7, listener=heard)
    sheets = heard.of("sheets")[0]
    for team in sheets["teams"]:
        for mon in team:
            assert mon["id"] and mon["id"] == mon["id"].lower() and mon["types"]
    assert sheets["statNames"]["atk"] and sheets["statusNames"]["par"]
    board = heard.of("board")[0]
    for side in board["sides"]:
        for mon in side["active"]:
            assert mon["id"] and mon["types"] and "statusId" in mon
    think = heard.of("think")[0]
    assert "classCount" in think and "classes" not in think
    prompt = heard.of("prompt")[0]
    assert len(prompt["actives"]) == len(prompt["slots"][0])
    turns = heard.of("turn")
    assert turns and any(t["changes"] for t in turns)
    for t in turns:
        assert t["agent"].split(" ／ ") == t["agentSlots"]
        for c in t["changes"]:
            assert c["from"] != c["to"] or c["fainted"] or c["entered"] or c["status"]
    # Labels are one part per slot, in slot order.
    step = heard.of("step")[-1]
    for label, action in zip(step.our_labels, step.ours, strict=True):
        assert len(label.split(" ／ ")) == len(action.slots)


def test_sprite_ids_carry_the_forme_and_the_server_writes_the_page_its_source(pool) -> None:  # noqa: ANN001
    reg = pool.reg
    assert humanplay.sprite_id(reg, "incineroar") == "incineroar"
    if "urshifurapidstrike" in reg.species:
        assert humanplay.sprite_id(reg, "urshifurapidstrike") == "urshifu-rapidstrike"
    formes = [s for s in reg.species.values() if s.forme]
    assert formes, "no forme in the dex; the check would be vacuous"
    for species in formes:
        got = humanplay.sprite_id(reg, species.id)
        assert got.count("-") >= 1 and " " not in got and got == got.lower()
    for url, want in ((None, liveview.SPRITE_URL), ("", ""), ("/sprites/{id}.png", "/sprites/{id}.png")):
        server = liveview.LiveServer("127.0.0.1", 0, sprite_url=url).start()
        try:
            page = urllib.request.urlopen(server.url, timeout=10).read().decode("utf-8")  # noqa: S310
        finally:
            server.close()
        assert f'<meta name="sprite-url" content="{want}">' in page
