"""What one decision sends the port twice is written and asked once, and nothing else moves
(IKA-264).

Three repeats of M-C generation's exchange with the port, each answered from the first:

* a position sent again in the same decision is not written again (`hold_positions`): it
  goes to the port once, as a `hold` line, and every request names it by number (IKA-302;
  until then its JSON text was kept and spliced into each request);
* a `score` request made again in the same decision is answered from the first;
* the port's `resolve` of a turn followed by the same turn with `select` (the caller's
  `weights` then `branch`) resolves the turn once.

And `repo_root` / `available()` stop asking the file system on every narrow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pokeuraou import port, regulation, rustnode, timing
from pokeuraou.actions import side_actions
from pokeuraou.budget import Budget
from pokeuraou.cli import _modal, build_beliefs
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.setup import load_scenario, with_spreads

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn5.json"


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.hold_positions(False)
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


def _node():  # noqa: ANN202
    scenario = load_scenario(EXAMPLE)
    reg = scenario.reg
    register_mega_stones(reg)
    beliefs = build_beliefs(scenario)
    pos = with_spreads(scenario, {key: _modal(b) for key, b in beliefs.items()})
    return reg, pos


def _lines(node: rustnode.RustNode) -> list[bytes]:
    """Every line this node writes to its child, from here on."""
    sent: list[bytes] = []
    real = node._process.stdin

    class Tap:
        def write(self, data: bytes) -> int:
            sent.append(bytes(data))
            return real.write(data)

        def flush(self) -> None:
            real.flush()

    node._process.stdin = Tap()
    return sent


def _named(request: dict, key: int) -> bytes:
    """`request`'s line with its position named by number, as a held one is sent."""
    text = json.dumps({**request, "position": "\x00"}, ensure_ascii=False)
    return text.replace('"\\u0000"', f'{{"held": {key}}}', 1).encode("utf-8")


def test_a_held_position_is_named_by_number(bridged: None) -> None:
    """The line with a held position is the line `json.dumps(request)` wrote with the
    position's number where the position was (positive: the entry is held and reused;
    control: switched off, the dict goes as before)."""
    reg, pos = _node()
    request = {"kind": "needed", "position": pos.to_json(), "side": 1}
    before = json.dumps(request, ensure_ascii=False).encode("utf-8")

    rustnode.hold_positions()
    held = rustnode._position(pos)
    assert isinstance(held, rustnode._Stored)
    assert rustnode._position(pos) is held, "the same object was written twice"
    assert held.json_text() == json.dumps(pos.to_json(), ensure_ascii=False)
    line = rustnode._payload({"kind": "needed", "position": held, "side": 1})
    assert line == _named(request, held.key)
    # Another object with the same content shares the number (the line is the same).
    assert rustnode._position(pos.copy()).key == held.key
    edited = pos.copy()
    edited.turn += 1
    assert rustnode._position(edited).key != held.key, "control: other content, other number"

    timing.decided("move")  # a decision ends: the memo goes with it
    assert rustnode._position(pos) is not held

    rustnode.hold_positions(False)
    assert isinstance(rustnode._position(pos), dict)
    assert rustnode._payload(request) == before


def test_the_switch_is_what_keeps_an_edited_position_honest(bridged: None) -> None:
    """Held, an object edited in place is sent as it was first written -- which is why the
    memo is off unless a worker that edits nothing switches it on (control: off, the edit
    is sent)."""
    _reg, pos = _node()
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    rustnode.hold_positions()
    first = rustnode._position(pos).json_text()
    mon.hp = max(1, mon.hp - 1)
    assert rustnode._position(pos).json_text() == first
    rustnode.hold_positions(False)
    assert rustnode._position(pos) != json.loads(first)


def test_a_repeated_score_is_answered_from_the_first(bridged: None) -> None:
    """The same `score` request twice in a decision crosses once and answers the same; the
    next decision asks again (control), and so does a node that is not holding."""
    reg, pos = _node()
    node = rustnode.node_for(reg)
    assert node is not None
    sent = _lines(node)

    rustnode.hold_positions()
    first = narrow(reg, pos, 1, limit=6)
    second = narrow(reg, pos, 1, limit=6)
    scores = [line for line in sent if b'"kind": "score"' in line]
    assert len(scores) == 1
    assert [(c.action.to_choice(), c.score, c.detail) for c in first.kept] == [
        (c.action.to_choice(), c.score, c.detail) for c in second.kept
    ]
    timing.decided("move")
    narrow(reg, pos, 1, limit=6)
    assert len([line for line in sent if b'"kind": "score"' in line]) == 2

    rustnode.hold_positions(False)
    narrow(reg, pos, 1, limit=6)
    narrow(reg, pos, 1, limit=6)
    assert len([line for line in sent if b'"kind": "score"' in line]) == 4


def test_a_score_request_is_the_line_json_dumps_wrote(bridged: None) -> None:
    """IKA-321: the candidate list written from each slot action's text, with the position
    held or not, is byte for byte the line `json.dumps` of the dicts wrote."""
    reg, pos = _node()
    for side in (0, 1):
        pool = side_actions(reg, pos, side)
        assert len(pool) > 1
        plain = {
            "kind": "score",
            "position": pos.to_json(),
            "side": side,
            "candidates": [[rustnode.dump_action(a) for a in c.slots] for c in pool],
        }
        before = json.dumps(plain, ensure_ascii=False).encode("utf-8")
        texted = {**plain, "candidates": rustnode._candidates(pool)}
        assert rustnode._payload(texted) == before
        # Twice: the second is written from the kept texts.
        assert rustnode._payload({**plain, "candidates": rustnode._candidates(pool)}) == before
        rustnode.hold_positions()
        stored = rustnode._position(pos)
        held = {**texted, "position": stored}
        assert isinstance(stored, rustnode._Stored)
        assert rustnode._payload(held) == _named(plain, stored.key)
        rustnode.hold_positions(False)


def test_the_detail_written_when_read_is_the_detail_written_before(bridged: None) -> None:
    """IKA-321: a candidate's detail lines, written when first read, are the lines the port's
    parts gave when `narrow` wrote them itself; ranked, the leaf's line comes first."""
    reg, pos = _node()
    node = rustnode.node_for(reg)
    assert node is not None
    for side in (0, 1):
        got = narrow(reg, pos, side, limit=6)
        scored = node.score(pos, side, [c.action for c in got.kept])
        assert scored is not None
        damage = {}
        for c, (_total, parts) in zip(got.kept, scored, strict=True):
            want = tuple(
                f"{c.action.slots[slot].describe(reg)} -> "
                f"{'foe' if is_foe else 'ally'}{target + 1} {signed:+.3f}{'' if exact else '?'}"
                for slot, target, is_foe, signed, exact in parts
            )
            assert c.detail == want
            damage[c.action.to_choice()] = want
        assert any(damage.values()), "control: some candidate has a line"

        ranked = narrow(
            reg, pos, side, limit=6, rank=lambda pool, _s: [0.5 - 0.01 * i for i in range(len(pool))]
        )
        for c in ranked.kept:
            assert c.detail[0] == f"leaf {c.score:+.4f}"
            if c.action.to_choice() in damage:
                assert c.detail[1:] == damage[c.action.to_choice()]


def test_the_branch_after_the_weights_is_the_branch_a_fresh_port_gives(bridged: None) -> None:
    """`weights` then `branch` (the port keeps the turn between them) against `branch` asked
    of a fresh process with nothing kept, and with another turn asked in between."""
    reg, pos = _node()
    row = narrow(reg, pos, 0, limit=4).actions
    col = narrow(reg, pos, 1, limit=4).actions
    chosen, other = [row[0], col[0]], [row[1], col[1]]
    budget = Budget.exact()

    weights = port.weights(reg, pos, chosen, budget)
    assert len(weights.branches) > 1, "want a turn with more than one branch"
    last = len(weights.branches) - 1
    kept = [port.branch(reg, pos, chosen, budget, k).to_json() for k in (0, last)]

    rustnode.reset()
    fresh = [port.branch(reg, pos, chosen, budget, k).to_json() for k in (0, last)]
    assert kept == fresh
    # And with another turn in between, which replaces the kept one.
    port.weights(reg, pos, other, budget)
    assert port.branch(reg, pos, chosen, budget, last).to_json() == fresh[1]
    assert kept[0] != kept[1], "control: two branches of the turn are different positions"


def test_a_position_crosses_once_a_decision(bridged: None) -> None:
    """IKA-302: held, a position goes to the port once -- a `hold` line with its JSON,
    ahead of the first request that names it -- and each request after names it by
    number; a decision's end sends `forget` ahead of the next request, and the position
    goes again. The answers are the ones the position itself gets (control: not held)."""
    reg, pos = _node()
    node = rustnode.node_for(reg)
    assert node is not None
    plain = [node.replacements_needed(pos), narrow(reg, pos, 0, limit=6).actions]
    sent = _lines(node)
    rustnode.hold_positions()
    held = [node.replacements_needed(pos), narrow(reg, pos, 0, limit=6).actions]
    assert held == plain
    whole = json.dumps(pos.to_json(), ensure_ascii=False).encode("utf-8")
    lines = b"".join(sent).splitlines()
    holds = [line for line in lines if line.startswith(b'{"kind": "hold"')]
    assert len(holds) == 1 and holds[0].endswith(b'"position": ' + whole + b"}")
    key = rustnode._position(pos).key
    asked = [line for line in lines if not line.startswith(b'{"kind": "hold"')]
    assert len(asked) == 2
    assert all(f'{{"held": {key}}}'.encode() in line and whole not in line for line in asked)

    sent.clear()
    timing.decided("move")
    assert node.replacements_needed(pos) == plain[0]
    lines = b"".join(sent).splitlines()
    assert lines[0] == b'{"kind": "forget"}'
    assert lines[1].startswith(b'{"kind": "hold"') and lines[1].endswith(whole + b"}")


def test_a_turns_branches_are_named_by_the_port(bridged: None) -> None:
    """IKA-302: held, a turn's branches come back with the port's (odd) numbers, and a
    branch sent back is named by its number and not written (no `hold` line) -- with the
    answers the branch itself gets. Equal branches share a number."""
    reg, pos = _node()
    row = narrow(reg, pos, 0, limit=3).actions
    col = narrow(reg, pos, 1, limit=3).actions
    asks = [(pos, [a, b]) for a in row for b in col]
    plain = port.turns(reg, asks, Budget.matrix())
    branches = [o.position for t in plain for o in t.outcomes]
    want = [narrow(reg, b, 1, limit=4).actions for b in branches if not b.ended]

    node = rustnode.node_for(reg)
    sent = _lines(node)
    rustnode.hold_positions()
    turns = port.turns(reg, asks, Budget.matrix())
    held = [o.position for t in turns for o in t.outcomes]
    assert [b.to_json() for b in held] == [b.to_json() for b in branches]
    keys = [rustnode._position(b).key for b in held]
    assert all(k % 2 == 1 for k in keys), "the port numbered every branch"
    texts = [json.dumps(b.to_json(), sort_keys=True) for b in held]
    assert len(set(keys)) == len(set(texts)), "equal branches share a number, others not"
    sent.clear()
    got = [narrow(reg, b, 1, limit=4).actions for b in held if not b.ended]
    assert got == want
    lines = b"".join(sent).splitlines()
    assert lines, "the scores were asked"
    assert not [line for line in lines if line.startswith(b'{"kind": "hold"')]


def test_the_repository_root_is_found_once() -> None:
    regulation.repo_root.cache_clear()
    first = regulation.repo_root()
    assert regulation.repo_root() is first
    assert regulation.repo_root.cache_info().hits >= 1
    assert (first / "src" / "pokeuraou" / "regulation.py").exists()


def test_a_binary_once_found_is_not_looked_for_again(tmp_path: Path) -> None:
    missing = tmp_path / "nothing.exe"
    assert not rustnode._binary_found(missing)
    missing.write_bytes(b"")
    assert rustnode._binary_found(missing)
    missing.unlink()
    assert rustnode._binary_found(missing), "found once, kept"
    assert not rustnode._binary_found(tmp_path / "other.exe")
