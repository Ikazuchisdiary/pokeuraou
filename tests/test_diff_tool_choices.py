"""Two small holes in the differential tools (IKA-324, IKA-327).

* `speed.action_overriding_effects` looked for `afteryou` in lines that say
  "move: After You", so After You was never found: diff_order counted the re-sorted turn
  as a divergence and diff_turn / diverge_report compared a turn they mean to skip.
* `diff_turn.showdown_choice` kept `move 1` for a Struggle the request does not list.
  Imprison's disable is hidden from a side's last Pokemon, so an Encore into an imprisoned
  Protect is asked with the four moves shown and Protect enabled; `move 1` named Draco
  Meteor and was refused with "needs a target" (diff_replacement, seed 4 battle 291).

The request below is Showdown's own, read off that refused step.
"""

from __future__ import annotations

from pokeuraou.actions import MoveAction, SideAction
from pokeuraou.speed import action_overriding_effects

from ._harness import load_tool

diff_turn = load_tool("diff_turn")


def test_after_you_and_dancer_are_found_by_their_display_names() -> None:
    # The protocol's own lines (Showdown `data/moves.ts`, `sim/battle-actions.ts`).
    assert action_overriding_effects(["|-activate|p2b: Toxapex|move: After You"]) == {
        "afteryou"
    }
    assert action_overriding_effects(
        ["|-activate|p2b: Toxapex|move: After You"], only_unmodelled=True
    ) == {"afteryou"}
    assert action_overriding_effects(["|-activate|p1a: Oricorio|ability: Dancer"]) == {
        "dancer"
    }
    assert action_overriding_effects(["|-activate|p1a: X|move: Quash"]) == {"quash"}
    assert action_overriding_effects(["|-singleturn|p1a: X|move: Instruct|[of] p1b: Y"]) == {
        "instruct"
    }
    # Encore starts without "move:"; its end reorders nothing.
    assert action_overriding_effects(["|-start|p2a: Y|Encore"]) == {"encore"}
    assert action_overriding_effects(["|-end|p2a: Y|Encore"]) == set()


def test_round_counts_only_when_a_second_round_is_moved_up() -> None:
    one = ["|move|p1a: X|Round|p2a: Y"]
    two = [*one, "|move|p2b: Z|Round|p1a: X"]
    assert action_overriding_effects(one) == set()
    assert action_overriding_effects(two) == {"round"}
    assert action_overriding_effects(two, only_unmodelled=True) == {"round"}


_DRAGALGE = {
    "moves": [
        {"move": "Draco Meteor", "id": "dracometeor", "pp": 7, "maxpp": 8,
         "target": "normal", "disabled": True},
        {"move": "Protect", "id": "protect", "pp": 5, "maxpp": 8,
         "target": "self", "disabled": False},
        {"move": "Toxic Spikes", "id": "toxicspikes", "pp": 19, "maxpp": 20,
         "target": "foeSide", "disabled": True},
        {"move": "Sludge Bomb", "id": "sludgebomb", "pp": 12, "maxpp": 12,
         "target": "normal", "disabled": True},
    ],
    "maybeDisabled": True,
    "maybeLocked": True,
}
_TOXAPEX = {
    "moves": [
        {"move": "Wide Guard", "id": "wideguard", "target": "allySide", "disabled": False},
        {"move": "Infestation", "id": "infestation", "target": "normal", "disabled": False},
        {"move": "Toxic", "id": "toxic", "target": "normal", "disabled": False},
        {"move": "Poison Jab", "id": "poisonjab", "target": "normal", "disabled": False},
    ]
}


def test_struggle_behind_a_hidden_imprison_takes_the_move_shown_enabled() -> None:
    pick = SideAction(
        slots=(
            MoveAction(slot=0, move_index=3, move_id="toxic", target=2),
            MoveAction(slot=1, move_index=1, move_id="struggle", target=None),
        )
    )
    request = {"active": [_TOXAPEX, _DRAGALGE]}
    assert diff_turn.showdown_choice(pick, request) == "move 3 2, move 2"


def test_struggle_the_request_lists_keeps_its_number() -> None:
    pick = SideAction(slots=(MoveAction(slot=0, move_index=1, move_id="struggle", target=None),))
    request = {
        "active": [
            {"moves": [{"move": "Struggle", "id": "struggle", "target": "randomNormal",
                        "disabled": False}]}
        ]
    }
    assert diff_turn.showdown_choice(pick, request) == "move 1"
