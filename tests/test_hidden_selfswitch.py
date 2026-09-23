"""The self-switch node's shared path against its definition (IKA-150).

Under a hidden bench `_do_self_switch_node` scores every option in every completion of the
opponent's unseen slots (IKA-120). By definition each completion's pause is rebuilt and
every option resumed in it from scratch; the shared path resolves an option once, on the
true pause, when its rest of the turn never reaches those slots, and swaps the
completion's Pokemon into the leaves. These hold the two equal, leaf for leaf, and show
that the checks deciding "never reaches" are what keeps them equal.
"""

from __future__ import annotations

import json

import numpy as np

from tests.test_hidden import UTURN_TURN, _action, _sheet, _uturn_start, setup  # noqa: F401

_ENCODED_ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def _slot_leaf(reg):  # noqa: ANN001, ANN202
    """A deterministic leaf reading every element of every encoded array, and its inputs.

    Like `tests/test_beliefnode.py`'s `_SlotLeaf`: a bench row that is the true Pokemon
    where it should be the completion's moves the value. It also keeps each batch it was
    handed as JSON, so the definition and the shared path can be compared leaf by leaf.
    """
    from pokeuraou.encode import Encoder

    encoder = Encoder(reg)
    batches: list[list[str]] = []
    weights: dict[tuple[str, int], np.ndarray] = {}

    def leaf(positions):  # noqa: ANN001, ANN202
        batches.append([json.dumps(p.to_json(), sort_keys=True) for p in positions])
        encoded = encoder.encode_positions(positions)
        n = len(encoded)
        raw = np.zeros(n, dtype=np.float64)
        for name in _ENCODED_ARRAYS:
            flat = getattr(encoded, name).reshape(n, -1).astype(np.float64)
            key = (name, flat.shape[1])
            if key not in weights:
                weights[key] = (
                    np.random.default_rng([150, len(name), flat.shape[1]]).normal(
                        size=flat.shape[1]
                    )
                    * 0.01
                )
            raw += (flat * weights[key]).sum(axis=1)
        return 1.0 / (1.0 + np.exp(-raw))

    return leaf, batches


def _self_switch_pause(reg, sheet, *, both: bool):  # noqa: ANN001, ANN202
    """Turn 1's pause after our U-turn; with `both`, side 1 U-turns later in the turn."""
    from pokeuraou.position import MoveSlot
    from pokeuraou.resolve import Budget, resolve_turn

    start = _uturn_start(reg, sheet[:4], sheet[:4])
    turn = dict(UTURN_TURN)
    if both:
        start.sides[1].pokemon[0].moves[0] = MoveSlot(id="uturn", pp=20, maxpp=20)
        turn[1] = "move 1 1, move 4"
    chosen = [_action(reg, start, side, turn[side]) for side in (0, 1)]
    truth = resolve_turn(reg, start, chosen, budget=Budget.deterministic(8))
    assert len(truth.suspended) == 1, "the U-turn has to pause the turn once"
    return truth.suspended[0]


def _node_both_ways(reg, sheet, pause):  # noqa: ANN001, ANN202
    """The node on `pause` by definition and by the shared path, with what each saw."""
    from pokeuraou.selfplay import GameRecord, _do_self_switch_node, _HiddenBench

    hidden = _HiddenBench(
        sheets=(sheet, sheet),
        seen=[frozenset(), frozenset()],
        bench_prior=None,
        leads=[None, None],
    )
    out = {}
    for definition in (True, False):
        leaf, batches = _slot_leaf(reg)
        record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
        _do_self_switch_node(
            reg, pause, record, (leaf, leaf), None, hidden=hidden, definition=definition
        )
        made = record.decisions[-1]
        out[definition] = {
            "leaves": batches,
            "value": made.search_value,
            "chosen": (made.own_chosen, made.foe_chosen),
            "unmodelled": list(record.unmodelled),
        }
    return out[True], out[False]


def _share_everything(reg, pause, other, alternatives, slots):  # noqa: ANN001, ANN202, ARG001
    """The shared path with its checks removed: every option's true plan, in every world."""
    from pokeuraou.resolve import turn_leaves

    return [turn_leaves(reg, resumed) for _option, resumed in alternatives]


def test_the_shared_self_switch_is_the_definition(setup) -> None:  # noqa: ANN001, F811
    """An option that never reaches their unseen slots is resolved once, and nothing moves.

    The shared path must hand the leaf exactly the positions resolving every completion
    from scratch hands it -- JSON for JSON, in the same order -- so the value is the same
    to the bit. And it must actually be the shared path: both options are vouched for
    here, or the comparison would be the definition against itself.
    """
    from pokeuraou.resolve import resume_alternatives
    from pokeuraou.selfplay import _shared_self_switch_plans

    reg, roster = setup
    sheet = _sheet(roster)
    pause = _self_switch_pause(reg, sheet, both=False)
    chooser, alternatives = resume_alternatives(reg, pause)
    shared = _shared_self_switch_plans(reg, pause, 1 - chooser, alternatives, (2, 3))
    assert [plan is not None for plan in shared] == [True, True]

    definition, fast = _node_both_ways(reg, sheet, pause)
    assert len(definition["leaves"]) == 1 and definition["leaves"][0]
    assert fast == definition
    # Six completions of side 1's back two, and no two of them hand the leaf the same leaf.
    assert len(set(definition["leaves"][0])) == len(definition["leaves"][0])


def test_an_option_that_reaches_their_bench_is_resolved_per_completion(
    setup, monkeypatch  # noqa: ANN001, F811
) -> None:
    """The positive control: sharing an option whose rest of the turn reaches the bench.

    With side 1 U-turning later in the same turn, their replacement is chosen from their
    bench -- the unseen slots included -- inside every option. The shared path has to
    refuse both options and still equal the definition; with its checks removed
    (`_share_everything`) it hands the leaf the true back two and the equality fails. The
    same removal on the plain U-turn changes nothing, so it is this option's reach that
    the checks catch.
    """
    from pokeuraou import selfplay
    from pokeuraou.resolve import resume_alternatives

    reg, roster = setup
    sheet = _sheet(roster)
    pause = _self_switch_pause(reg, sheet, both=True)
    chooser, alternatives = resume_alternatives(reg, pause)
    shared = selfplay._shared_self_switch_plans(
        reg, pause, 1 - chooser, alternatives, (2, 3)
    )
    assert [plan is not None for plan in shared] == [False, False]
    definition, fast = _node_both_ways(reg, sheet, pause)
    assert fast == definition

    monkeypatch.setattr(selfplay, "_shared_self_switch_plans", _share_everything)
    definition, broken = _node_both_ways(reg, sheet, pause)
    assert broken["leaves"] != definition["leaves"]
    assert broken["value"] != definition["value"]

    plain = _self_switch_pause(reg, sheet, both=False)
    definition, still = _node_both_ways(reg, sheet, plain)
    assert still == definition
