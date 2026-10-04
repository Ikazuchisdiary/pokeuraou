"""A state net reads a revision-4 encoding as before; a grown one reads the bind columns (IKA-429).

Encoding revision 4 appends the bind columns (縛り) after the state ones. Held here:

* a `state_inputs` net -- value-mc4st and q-mc4st's trunk -- gives the same logits, to the
  bit, on a revision-4 batch as on the same batch cut back to the revision-3 widths, and a
  plain net the same on the revision-2 cut. The control: a net that read the appended
  columns moves (a `bind_inputs` net with nonzero bind weights does).
* `grow_bind_inputs` gives a net that answers as the one it grew from, and `load_model`
  round-trips it with its own widths.
* `bind_inputs` needs `state_inputs`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="value function needs the optional learn group")

from pokeuraou.encode import BIND_MON_FEATURES, BIND_SIDE_FEATURES, Encoder  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.value import (  # noqa: E402
    ValueConfig,
    build,
    grow_bind_inputs,
    grow_state_inputs,
    load_model,
    net_widths,
    save_model,
)
from tests.test_encode_bind_port import FORMAT, _one_hp  # noqa: E402
from tests.test_encode_state_port import _base, _with_state  # noqa: E402

NAMES = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    from pokeuraou.position import Position

    encoder = Encoder(load_regulation(FORMAT))
    base = _base()
    raw = [base, _one_hp(base, 1), _with_state(_one_hp(base, 0), durations=True)]
    encoded = encoder.encode_positions([Position.from_json(p) for p in raw])
    assert encoded.mon[..., -len(BIND_MON_FEATURES) :].any()
    batch = {name: torch.from_numpy(getattr(encoded, name)) for name in NAMES}
    torch.manual_seed(429)
    plain = build(encoder, ValueConfig()).eval()
    state = grow_state_inputs(plain, encoder).eval()
    # Nonzero weights on the state columns, so the state net is not the plain one.
    gen = torch.Generator().manual_seed(4290)
    with torch.no_grad():
        for weight in (state.mon_mlp[0].weight, state.side_mlp[0].weight, state.head[0].weight):
            weight.copy_(torch.randn(weight.shape, generator=gen) * 0.3)
    return encoder, batch, plain, state


def _cut(batch: dict, widths: dict) -> dict:
    out = dict(batch)
    for name in ("mon", "side", "field"):
        out[name] = batch[name][..., : widths[name]].contiguous()
    return out


def test_older_nets_read_only_their_columns(setup) -> None:  # noqa: ANN001
    encoder, batch, plain, state = setup
    assert state.in_widths == encoder.state_widths
    with torch.no_grad():
        assert state(batch).numpy().tobytes() == state(_cut(batch, encoder.state_widths)).numpy().tobytes()
        assert plain(batch).numpy().tobytes() == plain(_cut(batch, encoder.base_widths)).numpy().tobytes()


def test_growing_answers_as_before_and_the_bind_columns_then_move_it(setup) -> None:  # noqa: ANN001
    encoder, batch, _plain, state = setup
    grown = grow_bind_inputs(state, encoder).eval()
    assert grown.in_widths == encoder.widths
    with torch.no_grad():
        before = state(batch)
        after = grown(batch)
    assert torch.allclose(before, after, atol=1e-5, rtol=0)
    gen = torch.Generator().manual_seed(4291)
    k_mon, k_side = len(BIND_MON_FEATURES), len(BIND_SIDE_FEATURES)
    with torch.no_grad():
        moved = grow_bind_inputs(state, encoder).eval()
        w = moved.mon_mlp[0].weight
        w[:, -k_mon:] = torch.randn(w[:, -k_mon:].shape, generator=gen) * 0.3
        s = moved.side_mlp[0].weight
        s[:, -k_side:] = torch.randn(s[:, -k_side:].shape, generator=gen) * 0.3
        shift = (moved(batch) - after).abs()
        # The same batch with the bind columns zeroed answers as `state` does.
        zeroed = dict(batch)
        zeroed["mon"] = batch["mon"].clone()
        zeroed["side"] = batch["side"].clone()
        zeroed["mon"][..., -k_mon:] = 0
        zeroed["side"][..., -k_side:] = 0
        same = (moved(zeroed) - state(zeroed)).abs()
    assert float(shift.max()) > 1e-4
    assert float(same.max()) < 1e-5


def test_a_bind_net_saves_and_loads_with_its_widths(setup, tmp_path: Path) -> None:  # noqa: ANN001
    encoder, batch, _plain, state = setup
    grown = grow_bind_inputs(state, encoder).eval()
    path = tmp_path / "bind.pt"
    save_model(path, grown, grown.state_dict(), encoder.vocab, grown.config, meta={})
    loaded, _meta = load_model(path, encoder)
    assert loaded.config.bind_inputs and loaded.in_widths == encoder.widths
    with torch.no_grad():
        assert loaded(batch).numpy().tobytes() == grown(batch).numpy().tobytes()


def test_bind_inputs_need_state_inputs() -> None:
    encoder = Encoder(load_regulation(FORMAT))
    with pytest.raises(ValueError, match="state_inputs"):
        net_widths(encoder, ValueConfig(bind_inputs=True))
    with pytest.raises(ValueError, match="state columns first"):
        grow_bind_inputs(build(encoder, ValueConfig()), encoder)
