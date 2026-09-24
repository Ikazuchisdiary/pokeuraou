"""What one decision sends the port twice is written and asked once, and nothing else moves
(IKA-264).

Three repeats of M-C generation's exchange with the port, each answered from the first:

* a position sent again in the same decision is not written again (`hold_positions`): its
  JSON text is kept and spliced into the request, byte for byte the line it was;
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


def test_a_held_position_is_spliced_in_as_the_same_bytes(bridged: None) -> None:
    """The line with a held position is the line `json.dumps(request)` wrote (positive:
    the text is held and reused; control: switched off, the dict goes as before)."""
    reg, pos = _node()
    request = {"kind": "needed", "position": pos.to_json(), "side": 1}
    before = json.dumps(request, ensure_ascii=False).encode("utf-8")

    rustnode.hold_positions()
    held = rustnode._position(pos)
    assert isinstance(held, rustnode._Held)
    assert rustnode._position(pos) is held, "the same object was written twice"
    assert rustnode._payload({"kind": "needed", "position": held, "side": 1}) == before
    # Another object with the same content is its own entry (keyed on the object).
    assert rustnode._position(pos.copy()) is not held

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
    first = rustnode._position(pos).text
    mon.hp = max(1, mon.hp - 1)
    assert rustnode._position(pos).text == first
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
