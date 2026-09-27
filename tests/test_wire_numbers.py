"""The port's crossings that stopped being JSON in IKA-350 answer what the JSON ones did.

Four of them, each held to the road it replaced:

* `qfeatures` answers float32 rows behind a small header: the same bits as the JSON answer
  read into float32 here;
* a replacement node's matrix asks `replacementsEncoded` and gets the encoder's rows of
  the phases' positions: the rows this side's encoder gives the positions `replacements`
  answers pair by pair;
* a resumed turn is asked for its weights and then for the drawn outcome alone
  (`ResumedTurn.pick`): the outcome the full answer holds at that index;
* slot actions cross as numbers, each action's JSON once for the life of the process (an
  `acts` line), and a completion (`hidden.substitute`) is held as its base with the
  Pokemon it changed: the answers are the ones the whole JSON gets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import port, qhead, rustnode
from pokeuraou.actions import PassAction, SideAction, side_actions, switch_actions_after_faint
from pokeuraou.budget import Budget
from pokeuraou.cli import _modal, build_beliefs
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import position_from_sets
from pokeuraou.setup import load_scenario, with_spreads
from pokeuraou.teams import load_roster

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn5.json"
ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


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


def _scenario():  # noqa: ANN202
    scenario = load_scenario(EXAMPLE)
    reg = scenario.reg
    register_mega_stones(reg)
    beliefs = build_beliefs(scenario)
    return reg, with_spreads(scenario, {key: _modal(b) for key, b in beliefs.items()})


def _opening():  # noqa: ANN202
    """Turn 1 of a mirror with the first four brought, slots 0 and 1 leading (`test_hidden`)."""
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    return reg, position_from_sets(reg, sheet[:4], sheet[:4]), sheet


def _lines(node: rustnode.RustNode) -> list[bytes]:
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


def test_the_features_are_the_json_answers_bits(bridged: None) -> None:
    """Binary rows against the JSON answer's, bit for bit, both pools; and the binary road
    is the one taken (positive: a crossing counted, no `features` JSON in the answer)."""
    reg, pos = _scenario()
    pools = (side_actions(reg, pos, 0), side_actions(reg, pos, 1))
    assert all(len(pool) > 1 for pool in pools)
    old = qhead.port_features(reg, pos, pools, json_answer=True)
    new = qhead.port_features(reg, pos, pools)
    assert old is not None and new is not None
    for a, b in zip(old, new, strict=True):
        assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert np.any(new[0] != 0.0), "control: the rows carry numbers"
    assert not np.array_equal(new[0][0], new[0][-1]), "control: two candidates differ"


def _owing():  # noqa: ANN202
    """The opening with one lead fainted on each side, and each side's replacements."""
    reg, pos, _sheet = _opening()
    for side, slot in ((0, 0), (1, 1)):
        mon = pos.sides[side].pokemon[pos.sides[side].active[slot]]
        mon.hp = 0
        mon.fainted = True
    owed = port.replacements_needed(reg, pos)
    options = [switch_actions_after_faint(reg, pos, side, list(owed[side])) for side in (0, 1)]
    assert all(len(found) > 1 for found in options), options
    return reg, pos, options


def test_a_replacement_matrix_is_the_encoders_rows_of_its_phases(bridged: None) -> None:
    """`replacementsEncoded` of every pair against this side's encoder of the positions the
    `replacements` command answers pair by pair: the same arrays, row for row, and the same
    ended rows (control: two pairs' rows differ)."""
    reg, pos, options = _owing()
    pairs = [(a, b) for a in options[0] for b in options[1]]
    filled = port.replacements_encoded(reg, pos, pairs)
    phases = [port.resolve_replacements(reg, pos, [a, b]).position for a, b in pairs]
    here = Encoder(reg).encode_positions(phases)
    for name in ARRAYS:
        got, want = getattr(filled.encoded, name), getattr(here, name)
        assert got.shape == want.shape, name
        assert np.array_equal(got, want), name
    assert np.array_equal(filled.encoded.decided, here.decided, equal_nan=True)
    assert not np.array_equal(filled.encoded.mon[0], filled.encoded.mon[-1]), "control"


def test_the_replacement_node_scores_the_same_rows(bridged: None) -> None:
    """The node's own matrix: a leaf with an encoded form over the port's rows, against the
    same leaf over the positions (`evaluate(flat)`, the road it replaced)."""
    reg, pos, options = _owing()
    encoder = Encoder(reg)

    class Leaf:
        def __call__(self, positions):  # noqa: ANN001, ANN204
            return self.from_encoded(encoder.encode_positions(list(positions)))

        def from_encoded(self, encoded):  # noqa: ANN001, ANN201
            # Every feature of every row weighted differently, so a row that moved moves this.
            flat = encoded.mon.reshape(len(encoded.mon), -1).astype(np.float64)
            return flat @ np.linspace(-1.0, 1.0, flat.shape[1])

    leaf = Leaf()
    plan = port.encoded_leaf_plan([leaf])
    assert plan is not None and plan[0][1] is not None
    filled = port.replacements_encoded(reg, pos, [(a, b) for a in options[0] for b in options[1]])
    by_rows = np.asarray(plan[0][1](filled.encoded)).reshape(len(options[0]), len(options[1]))
    by_positions = np.asarray(
        leaf([port.resolve_replacements(reg, pos, [a, b]).position for a in options[0] for b in options[1]])
    ).reshape(len(options[0]), len(options[1]))
    assert np.array_equal(by_rows, by_positions)
    assert len(set(by_rows.ravel().tolist())) > 1, "control: the choices are worth different amounts"


def _paused():  # noqa: ANN202
    """A turn of the scenario that pauses for a Parting Shot replacement (`test_rust_node`)."""
    from .test_rust_node import _node, _pausing_cells, _with_parting_shot

    reg, pos, row, col = _node()
    pos.sides[0].active_pokemon()[1].boosts["spe"] = 6
    row = _with_parting_shot(reg, pos, row)
    cells = _pausing_cells(reg, pos, row, col, Budget.matrix())
    assert cells
    i, j = cells[0]
    turn = port.turn(reg, pos, [row[i], col[j]], Budget.matrix(), full=True)
    assert turn.pauses
    return reg, turn.pauses[0]


def test_a_resumed_turns_drawn_outcome_is_the_full_answers(bridged: None) -> None:
    """`resumed` then `pick(k)` against `resume`'s full answer at k, for every branch index
    (and a pause, if the resumed turn pauses again)."""
    reg, pause = _paused()
    chooser, alternatives = port.resume_alternatives(reg, pause)
    assert alternatives
    option = alternatives[0][0]
    other = 1 - chooser
    passes = SideAction(
        slots=tuple(PassAction(slot=i) for i in range(len(pause.position.sides[other].active)))
    )
    choices = [option, passes] if chooser == 0 else [passes, option]
    full = port.resume(reg, pause, choices)
    lazy = port.resumed(reg, pause, choices)
    assert lazy.branches == full.branches and lazy.suspended == full.suspended
    assert lazy.exact == full.exact and lazy.unmodelled == full.unmodelled
    assert len(full.branches) > 1, "want a resumed turn with more than one branch"
    for k in range(len(full.branches) + len(full.suspended)):
        want = full.pick(k)
        got = port.resumed(reg, pause, choices).pick(k)
        if k < len(full.branches):
            assert got.to_json() == want.to_json()
        else:
            assert got.raw == want.raw
    assert full.pick(0).to_json() != full.pick(len(full.branches) - 1).to_json(), "control"
    # Read whole, it is the full answer.
    assert [o.position.to_json() for o in lazy.outcomes] == [o.position.to_json() for o in full.outcomes]


def test_actions_cross_as_numbers_defined_once(bridged: None) -> None:
    """A `score` names its candidates' slot actions by number; each action's JSON goes once,
    in an `acts` line ahead of the first request that names it, and a second request sends
    none. The answers are the ones the actions' JSON gets (the `json_answer` road of
    `qfeatures` and a `score` written with dicts). A fresh process is given them again."""
    reg, pos = _scenario()
    node = rustnode.node_for(reg)
    assert node is not None
    pool = side_actions(reg, pos, 0)
    sent = _lines(node)
    first = node.score(pos, 0, pool)
    lines = b"".join(sent).splitlines()
    acts = [line for line in lines if line.startswith(b'{"kind": "acts"')]
    assert len(acts) == 1
    defined = json.loads(acts[0])["acts"]
    slot_actions = {a for c in pool for a in c.slots}
    assert len(defined) == len({json.dumps(rustnode.dump_action(a)) for a in slot_actions})
    for number, action in defined:
        assert json.dumps(action, ensure_ascii=False) == rustnode._ACTION_BY_NUMBER[number]
    score_line = next(line for line in lines if b'"kind": "score"' in line)
    assert b'"moveId"' not in score_line, "the candidates are numbers"
    sent.clear()
    assert node.score(pos, 0, pool) == first
    assert not [line for line in b"".join(sent).splitlines() if line.startswith(b'{"kind": "acts"')]
    rustnode.reset()
    again = rustnode.node_for(reg)
    sent = _lines(again)
    assert again.score(pos, 0, pool) == first
    assert [line for line in b"".join(sent).splitlines() if line.startswith(b'{"kind": "acts"')]


def test_the_scores_by_number_are_the_scores_by_json(bridged: None) -> None:
    """Every candidate of both sides scored by number and by the actions' JSON."""
    reg, pos = _scenario()
    node = rustnode.node_for(reg)
    assert node is not None
    for side in (0, 1):
        pool = side_actions(reg, pos, side)
        by_json = node._exchange(
            {
                "kind": "score",
                "position": pos.to_json(),
                "side": side,
                "candidates": [[rustnode.dump_action(a) for a in c.slots] for c in pool],
            }
        )
        by_number = node.score(pos, side, pool)
        assert by_number is not None
        assert [s for s, _d in by_number] == [float(s) for s in by_json["scores"]]
        assert [[tuple(p) for p in d] for _s, d in by_number] == [
            [(int(a), int(b), bool(c), float(d), bool(e)) for a, b, c, d, e in parts]
            for parts in by_json["detail"]
        ]


def test_a_completion_is_held_as_its_base_and_its_pokemon(bridged: None) -> None:
    """Held, a completion goes to the port as its base's number and the Pokemon it changed
    (no whole JSON), and every answer about it is the one its whole JSON gets: its
    candidates' scores and a fill's encoded leaves (control: two completions answer
    differently, so the Pokemon went across)."""
    reg, pos, sheet = _opening()
    made = completions(reg, pos, 1, sheet)
    assert len(made) == 6
    node = rustnode.node_for(reg)
    assert node is not None
    row = narrow(reg, pos, 0, limit=3).actions
    col = narrow(reg, pos, 1, limit=3).actions

    def answers(position):  # noqa: ANN001, ANN202
        encoded = node.fill_encoded(position, row, col, Budget.matrix(), [], None).encoded
        return (
            node.score(position, 1, side_actions(reg, position, 1)),
            [getattr(encoded, name).tobytes() for name in ARRAYS],
        )

    whole = [answers(c.position) for c in made]
    assert whole[0] != whole[1], "control: two completions answer differently"
    # The completions again, made while holding (each is noted as its base's).
    rustnode.hold_positions()
    made = completions(reg, pos, 1, sheet)
    sent = _lines(node)
    held = [answers(c.position) for c in made]
    assert held == whole
    lines = b"".join(sent).splitlines()
    holds = [json.loads(line) for line in lines if line.startswith(b'{"kind": "hold"')]
    deltas = [h for h in holds if "base" in h]
    assert len(deltas) == 6 and all("position" not in h for h in deltas)
    assert len([h for h in holds if "position" in h]) == 1, "the base went whole, once"
    assert all(len(h["pokemon"]) == 2 for h in deltas), "the two unseen slots"
