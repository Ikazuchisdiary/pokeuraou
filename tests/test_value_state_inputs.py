"""A revision-2 net reads a revision-3 encoding as before; a grown one reads the state (IKA-425).

Encoding revision 3 appends state columns to each block. Three things are held here:

* a net without `state_inputs` -- every model trained before -- gives the same logits, to
  the bit, on a revision-3 batch as on the same batch cut back to the revision-2 widths,
  which is what the encoder used to hand it. The control: a net that read the appended
  columns would move (a `state_inputs` net with nonzero state weights does).
* `grow_state_inputs` gives a net that answers as the one it grew from (zero weights on
  the new columns), and `load_model` round-trips both kinds with their own widths.
* the grown net's answer depends on the state columns once their weights are nonzero,
  including the two move ids through the move embedding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="value function needs the optional learn group")

from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.value import (  # noqa: E402
    ValueConfig,
    build,
    grow_state_inputs,
    load_model,
    save_model,
)
from tests.test_encode_state_port import FORMAT, _base, _with_state  # noqa: E402


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    encoder = Encoder(load_regulation(FORMAT))
    base = _base()
    positions = [
        Position.from_json(p)
        for p in (base, _with_state(base, durations=True), _with_state(base, durations=False))
    ]
    encoded = encoder.encode_positions(positions)
    batch = {
        name: torch.from_numpy(getattr(encoded, name))
        for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    }
    torch.manual_seed(425)
    net = build(encoder, ValueConfig()).eval()
    return encoder, batch, net


def _cut(batch: dict, widths: dict) -> dict:
    out = dict(batch)
    for name in ("mon", "side", "field"):
        out[name] = batch[name][..., : widths[name]].contiguous()
    return out


def test_a_revision_two_net_reads_only_the_leading_columns(setup) -> None:  # noqa: ANN001
    encoder, batch, net = setup
    assert net.in_widths == encoder.base_widths
    with torch.no_grad():
        full = net(batch)
        narrow = net(_cut(batch, encoder.base_widths))
    assert full.numpy().tobytes() == narrow.numpy().tobytes()


def test_growing_answers_as_before_and_the_state_then_moves_it(setup) -> None:  # noqa: ANN001
    encoder, batch, net = setup
    grown = grow_state_inputs(net, encoder).eval()
    assert grown.in_widths == encoder.widths
    with torch.no_grad():
        before = net(batch)
        after = grown(batch)
    assert torch.allclose(before, after, atol=1e-5, rtol=0)
    # The positions with state and without differ only in the state columns; with
    # nonzero weights on them the grown net tells them apart, the old one cannot.
    base_w = encoder.base_widths
    mon_old = net.mon_mlp[0].in_features
    # Random weights, not a constant: a constant column adds the same to every unit, and
    # the LayerNorm after each first layer takes that away again.
    gen = torch.Generator().manual_seed(4250)

    def fill(weight: torch.Tensor) -> None:
        weight.copy_(torch.randn(weight.shape, generator=gen) * 0.3)

    with torch.no_grad():
        # Only the two move ids' embedding columns: the last 2 * move_dim inputs.
        ids_only = grow_state_inputs(net, encoder).eval()
        k = 2 * ids_only.config.move_dim
        fill(ids_only.mon_mlp[0].weight[:, -k:])
        moved_ids = ids_only(batch) - after
        numeric = grow_state_inputs(net, encoder).eval()
        fill(numeric.mon_mlp[0].weight[:, mon_old : mon_old + 3])
        fill(numeric.side_mlp[0].weight[:, -(encoder.widths["side"] - base_w["side"]) :])
        fill(numeric.head[0].weight[:, -1:])
        moved_numeric = numeric(batch) - after
    assert float(moved_ids[0].abs()) < 1e-6  # no locked or last move in the base position
    assert float(moved_ids[1].abs()) > 1e-4
    assert float(moved_numeric[0].abs()) < 1e-6
    assert float(moved_numeric[1].abs()) > 1e-4


def test_both_kinds_save_and_load_with_their_own_widths(setup, tmp_path: Path) -> None:  # noqa: ANN001
    encoder, batch, net = setup
    grown = grow_state_inputs(net, encoder).eval()
    for name, model in (("old", net), ("grown", grown)):
        path = tmp_path / f"{name}.pt"
        save_model(path, model, model.state_dict(), encoder.vocab, model.config, meta={})
        loaded, _meta = load_model(path, encoder)
        assert loaded.in_widths == model.in_widths
        with torch.no_grad():
            assert loaded(batch).numpy().tobytes() == model(batch).numpy().tobytes()
    blob = torch.load(tmp_path / "old.pt", weights_only=False)
    assert blob["widths"] == encoder.base_widths
