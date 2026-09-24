"""The production roads have no Python resolver behind them (IKA-209): switched off, they stop.

`POKEURAOU_RUST_NODE=0` used to mean "resolve in Python". The roads generation, the analyser
and the selection solve take (`pokeuraou.port`) have nothing to fall back to, so with the
bridge switched off they raise `PortUnavailable` rather than quietly doing something else.
resolve.py's own `batched_payoffs` -- the tools' road, until IKA-212 -- still honours it.
"""

from __future__ import annotations

import pytest

from pokeuraou import port, rustnode
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster


@pytest.fixture()
def reg():  # noqa: ANN201
    loaded = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(loaded)
    return loaded


def _sets():  # noqa: ANN202
    sheet = {entry.species: entry for entry in load_roster("rizabanadohido").sets}
    own = [sheet[n] for n in ("venusaur", "sylveon", "charizard", "garchomp")]
    foe = [sheet[n] for n in ("incineroar", "toxapex", "garchomp", "venusaur")]
    return own, foe


def test_the_roads_answer_with_the_bridge_on(reg, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """The control: the same calls answer when the bridge is on (the default)."""
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.delenv(rustnode.ENV_ENABLE, raising=False)
    rustnode.reset()
    pos = position_from_sets(reg, *_sets())
    row = narrow(reg, pos, 0, limit=2).actions
    col = narrow(reg, pos, 1, limit=2).actions
    payoff, _notes = port.batched_payoff(reg, pos, row, col, HP_SHARE.batch, budget=Budget.matrix())
    assert payoff.shape == (len(row), len(col))
    rustnode.reset()


def test_switched_off_the_roads_stop(reg, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.delenv(rustnode.ENV_ENABLE, raising=False)
    rustnode.reset()
    pos = position_from_sets(reg, *_sets())
    row = narrow(reg, pos, 0, limit=2).actions
    col = narrow(reg, pos, 1, limit=2).actions
    rustnode.reset()
    monkeypatch.setenv(rustnode.ENV_ENABLE, "0")
    try:
        with pytest.raises(rustnode.PortUnavailable):
            position_from_sets(reg, *_sets())
        with pytest.raises(rustnode.PortUnavailable):
            port.batched_payoff(reg, pos, row, col, HP_SHARE.batch, budget=Budget.matrix())
        with pytest.raises(rustnode.PortUnavailable):
            port.replacements_needed(reg, pos)
    finally:
        rustnode.reset()
