"""IKA-253: a learned leaf scores a finished battle as 1/0 (0.5 for a draw), as hp-share does.

The net is trained on decision points only, so an ended position is outside what it saw:
before the fix `value-all` put 0.54 on a certain loss and 0.93 on a certain win. `payoff`'s
objectives (Python `_decided`, Rust `objective::decided`) had the substitution; the learned
leaf's roads did not. Every road is checked here on the named shape -- a cell of IKA-253's
end (Garchomp at 1 HP against Kingambit at 1 HP) that the turn decides, scored by a learned
leaf -- and on the cell that does not end, which must keep the net's own number:

- positions to `BatchedValue.__call__` (`port._scored_here`, `_subgame_value`, the root);
- the port's encoded leaves to `BatchedValue.from_encoded` (`port._encoded`);
- the same two through the inference server (`RemoteValue`);
- the pieces the hidden-bench node stacks its batch from (`beliefnode._stacked`, `_subset`);
- the search at depth 2 with one Sucker Punch left, which is exactly 1/2 once the ends are
  1/0 whatever the net says (every leaf of the second turn is decided).

An untrained net (seed 7) stands in for a trained one: what is tested is that its number
never reaches a decided leaf, and an untrained net's number is nowhere near 0 or 1.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pokeuraou import port  # noqa: E402
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import _decided  # noqa: E402
from pokeuraou.search import search  # noqa: E402
from pokeuraou.setup import parse_scenario  # noqa: E402
from pokeuraou.value import BatchedValue, ValueConfig, build  # noqa: E402

REG = "gen9championsvgc2026regmc"


def _filler(species: str, ability: str, nature: str, moves: list[str]) -> dict:
    return {"species": species, "ability": ability, "nature": nature, "moves": moves,
            "sp": {"hp": 1}, "hp": 0}


def _chomp_gambit(sucker_pp: int = 8):  # noqa: ANN202
    """IKA-253's end: both at 1 HP, Garchomp (SD/EQ, faster) against Kingambit (KC/SP)."""
    chomp = {"species": "Garchomp", "ability": "roughskin", "nature": "Jolly",
             "moves": ["Swords Dance", "Earthquake"], "sp": {"atk": 32, "spe": 32},
             "active": True, "hp": 1}
    gambit = {"species": "Kingambit", "ability": "defiant", "nature": "Adamant",
              "moves": ["Kowtow Cleave", "Sucker Punch"], "sp": {"hp": 32, "atk": 32},
              "active": True, "hp": 1}
    sides = []
    for lead in (chomp, gambit):
        dead = _filler("Incineroar", "intimidate", "Impish", ["Fake Out"])
        dead["active"] = True
        sides.append([lead, dead, _filler("Sinistcha", "hospitality", "Bold", ["Protect"]),
                      _filler("Farigiraf", "armortail", "Calm", ["Protect"])])
    data = {"regulation": REG, "turn": 1,
            "sides": [{"id": "p1", "team": sides[0]},
                      {"id": "p2", "spreadsKnown": True, "team": sides[1]}],
            "field": {}}
    sc = parse_scenario(data)
    register_mega_stones(sc.reg)
    pos = sc.position
    pp = {"swordsdance": 20, "earthquake": 12, "kowtowcleave": 12, "suckerpunch": sucker_pp}
    for side in pos.sides:
        for m in side.pokemon[0].moves:
            m.pp = m.maxpp = pp[m.id]
    return sc.reg, pos


@pytest.fixture(scope="module")
def game():
    reg, pos = _chomp_gambit(8)
    encoder = Encoder(reg)
    torch.manual_seed(7)
    net = build(encoder, ValueConfig()).eval()
    leaf = BatchedValue(net, encoder, device=torch.device("cpu"))
    row = narrow(reg, pos, 0, limit=8).actions
    col = narrow(reg, pos, 1, limit=8).actions
    return reg, pos, encoder, net, leaf, row, col


def _raw(leaf: BatchedValue, positions: list) -> np.ndarray:
    """The net's own number, with nothing substituted: what a leaf that is not over gets."""
    encoded = leaf.encoder.encode_positions(positions)
    batch = {
        name: torch.from_numpy(getattr(encoded, name))
        for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    }
    with torch.no_grad():
        return torch.sigmoid(leaf._mean_logit(batch)).double().numpy()


def _cells(reg, pos, row, col):  # noqa: ANN001, ANN202
    """Each cell's value if the turn decides it outright (every branch ended), else None."""
    out = {}
    for i, r in enumerate(row):
        for j, c in enumerate(col):
            turn = port.turn(reg, pos, [r, c], Budget.matrix(), full=True)
            leaves = [b.position for b in turn.outcomes or []]
            out[i, j] = (
                _decided(leaves[0])
                if leaves and all(p.ended for p in leaves)
                and len({_decided(p) for p in leaves}) == 1
                else None
            )
    return out


def test_the_scenario_has_won_lost_and_open_cells(game) -> None:
    """The shape exists: the turn decides some cells each way and leaves one open."""
    reg, pos, _encoder, _net, _leaf, row, col = game
    values = list(_cells(reg, pos, row, col).values())
    assert 1.0 in values and 0.0 in values and None in values


def test_an_ended_position_is_one_or_zero_to_the_leaf(game) -> None:
    reg, pos, _encoder, _net, leaf, row, col = game
    ended, open_ = [], []
    for r in row:
        for c in col:
            for b in port.turn(reg, pos, [r, c], Budget.matrix(), full=True).outcomes:
                (ended if b.position.ended else open_).append(b.position)
    assert ended and open_
    want = np.array([_decided(p) for p in ended])
    raw = _raw(leaf, ended)
    # Positive control: the net alone is nowhere near the answer.
    assert np.abs(raw - want).min() > 0.05
    assert leaf(ended).tolist() == want.tolist()
    # The JSON form reads the same flag.
    assert leaf([p.to_json() for p in ended]).tolist() == want.tolist()
    # A draw is a half.
    draw = ended[0].copy()
    draw.winner = None
    assert leaf([draw]).tolist() == [0.5]
    # And a position still in play keeps the net's number to the last bit.
    assert np.array_equal(leaf(open_), _raw(leaf, open_))


@pytest.mark.parametrize("road", ["encoded", "scored_here"])
def test_a_decided_cell_is_one_or_zero_in_the_node(game, road: str) -> None:
    """The port's encoded leaves (`_encoded`) and the Python road (`_scored_here`)."""
    reg, pos, _encoder, _net, leaf, row, col = game
    budget = Budget.matrix()
    if road == "encoded":
        payoffs, _notes, _exact = port.batched_payoffs(reg, pos, row, col, [leaf], budget=budget)
    else:
        payoffs, _notes, _exact = port._scored_here(reg, pos, row, col, [leaf], budget, None)
    payoff = payoffs[0]
    for (i, j), want in _cells(reg, pos, row, col).items():
        if want is not None:
            assert payoff[i, j] == want, (road, i, j, payoff[i, j], want)
        else:
            assert 0.0 < payoff[i, j] < 1.0


def test_the_ports_encoded_leaves_carry_the_end(game) -> None:
    reg, pos, _encoder, _net, leaf, row, col = game
    filled = port.ask(
        reg, lambda node: node.fill_encoded(pos, list(row), list(col), Budget.matrix(), [], None)
    )
    decided = filled.encoded.decided
    assert decided.shape == (len(filled.encoded),)
    assert np.isnan(decided).any() and (decided == 1.0).any() and (decided == 0.0).any()
    values = leaf.from_encoded(filled.encoded)
    known = ~np.isnan(decided)
    assert values[known].tolist() == decided[known].tolist()


def test_the_served_leaf_substitutes_too(game) -> None:
    from pokeuraou.inference import RemoteValue, serve, served_model

    reg, pos, encoder, net, leaf, row, col = game
    budget = Budget.matrix()
    local, _n, _e = port.batched_payoffs(reg, pos, row, col, [leaf], budget=budget)
    server, address = serve({"value": served_model(BatchedValue(net, encoder, device=torch.device("cpu")))})
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=8 << 20) as remote:
            got, _n, _e = port.batched_payoffs(reg, pos, row, col, [remote], budget=budget)
            here, _n, _e = port._scored_here(reg, pos, row, col, [remote], budget, None)
    finally:
        server.shutdown()
    assert np.array_equal(got[0], local[0])
    assert np.array_equal(here[0], local[0])
    for (i, j), want in _cells(reg, pos, row, col).items():
        if want is not None:
            assert got[0][i, j] == want and here[0][i, j] == want


def test_the_hidden_node_keeps_the_end_through_its_stacking(game) -> None:
    """`beliefnode` builds one batch from row subsets and stacked parts; the flag survives."""
    from pokeuraou.beliefnode import _stacked, _subset

    reg, pos, _encoder, _net, _leaf, row, col = game
    filled = port.ask(
        reg, lambda node: node.fill_encoded(pos, list(row), list(col), Budget.matrix(), [], None)
    )
    reference = filled.encoded
    rows = list(range(len(reference)))[::-1]
    sub = _subset(reference, rows)
    assert np.array_equal(sub.decided, reference.decided[rows], equal_nan=True)
    stacked, starts = _stacked([(len(reference), lambda: reference), (len(sub), lambda: sub)], reference)
    assert np.array_equal(
        stacked.decided, np.concatenate([reference.decided, sub.decided]), equal_nan=True
    )
    assert starts == [0, len(reference)]


def test_one_sucker_punch_left_is_exactly_a_half_at_depth_two() -> None:
    """IKA-253's control: every leaf of the second turn is decided, so the net drops out."""
    reg, pos = _chomp_gambit(1)
    encoder = Encoder(reg)
    torch.manual_seed(7)
    leaf = BatchedValue(build(encoder, ValueConfig()).eval(), encoder, device=torch.device("cpu"))
    row = narrow(reg, pos, 0, limit=48).actions
    col = narrow(reg, pos, 1, limit=48).actions
    result = search(reg, pos, row, col, leaf, budget=Budget.matrix(), depth=2)
    assert result.equilibrium.value == pytest.approx(0.5, abs=1e-9)


def test_the_old_rule_leaves_the_nets_number_and_counts_the_same(game) -> None:
    """`EncodingRules(net_scores_ends=True)` is master's leaf, for a match measuring the fix."""
    from pokeuraou.encode import EncodingRules

    reg, pos, encoder, net, leaf, row, col = game
    old = BatchedValue(
        net, Encoder(reg, vocab=encoder.vocab, rules=EncodingRules(net_scores_ends=True)),
        device=torch.device("cpu"),
    )
    assert old.encoder.rules.label() == "old-ends"
    budget = Budget.matrix()
    before = leaf.ended
    new_payoff = port.batched_payoffs(reg, pos, row, col, [leaf], budget=budget)[0][0]
    old_payoff = port.batched_payoffs(reg, pos, row, col, [old], budget=budget)[0][0]
    assert old.ended == leaf.ended - before > 0
    cells = _cells(reg, pos, row, col)
    for (i, j), want in cells.items():
        if want is None:
            assert old_payoff[i, j] == new_payoff[i, j]
        else:
            assert new_payoff[i, j] == want and abs(old_payoff[i, j] - want) > 0.05
