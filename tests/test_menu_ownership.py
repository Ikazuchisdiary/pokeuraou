"""A menu belongs to the agent that built it, in both seats and both information sets.

`_menus`' own docstring says so -- "handing one agent's menu to the other would make a
ranking comparison measure nothing" -- and three paths handed it over anyway:

  * `same_menu` compared the ranking FLAG and not the leaf, so two arms both passing
    --rank-leaf with different models counted as one menu. About 25,000 recorded games.
  * under `sheets` the rebuild was unreachable, so side 1 always played side 0's menu and
    `ranked[1]`/`policies[1]` were dead arguments the provenance recorded anyway.
  * the replacement node took both strategies off `matrix(pos, leaves[0])`.

The tests of the first two went with `play_game` (IKA-463); the Rust board worker builds each
side's menu from its own arm (`tests/test_rust_board_worker.py`). The third is left.
"""

from __future__ import annotations

import inspect


def test_the_replacement_node_solves_each_side_with_its_own_leaf() -> None:

    from pokeuraou import replacenode

    body = inspect.getsource(replacenode.solve)
    # Each side's game is built from its own leaf (the port's `replacementNode` request takes
    # the server each side's leaf is, IKA-460).
    assert "served_leaf(leaves[side]" in body, (
        "the column player's replacement is still chosen by the row player's evaluator"
    )
