"""IKA-433: `--attention` warm-starts together with the state, bind and auxiliary options.

A plain net grown into attention + state + bind columns + auxiliary heads answers as the
plain net did (the attention output projection and the new columns start at zero), and the
attention branch is live: once its output projection is moved, the answer moves.
"""

from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch", reason="value function needs the optional learn group")

from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation, repo_root  # noqa: E402
from pokeuraou.value import ValueConfig, build, save_model  # noqa: E402
from tests.test_encode_bind_port import FORMAT, _one_hp  # noqa: E402
from tests.test_encode_state_port import _base, _with_state  # noqa: E402

NAMES = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def _tool():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "train_value_tool_433", repo_root() / "tools" / "train_value.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attention_grows_with_state_bind_and_aux(tmp_path) -> None:  # noqa: ANN001
    encoder = Encoder(load_regulation(FORMAT))
    base = _base()
    raw = [base, _one_hp(base, 1), _with_state(_one_hp(base, 0), durations=True)]
    encoded = encoder.encode_positions([Position.from_json(p) for p in raw])
    batch = {name: torch.from_numpy(getattr(encoded, name)) for name in NAMES}
    torch.manual_seed(433)
    plain = build(encoder, ValueConfig()).eval()
    parent = tmp_path / "parent.pt"
    save_model(parent, plain, plain.state_dict(), encoder.vocab, ValueConfig(), meta={"m": 1},
               widths=plain.in_widths)
    run = dict(
        epochs=1, batch_size=8, lr=1e-4, seed=1, split_seed=0, keep="last", attention=True,
        state_inputs=True, bind_inputs=True, aux_weight=5.0, aux_targets="search:1,ahead2:5",
    )
    net, _record, config = _tool().warm_start(parent, encoder, run)
    net.eval()
    assert config.attention and net.config.attention and net.config.bind_inputs
    assert hasattr(net, "aux") and hasattr(net, "aux_heads") and hasattr(net, "mon_attention")
    with torch.no_grad():
        assert torch.allclose(net(batch), plain(batch), atol=1e-5, rtol=0)
        # Positive control: the attention branch is on the road. Moving its output
        # projection moves the answer, so the agreement above is not a dead branch.
        torch.nn.init.normal_(net.mon_attention.out.weight, std=0.3)
        assert float((net(batch) - plain(batch)).abs().max()) > 1e-4
