"""IKA-428: the auxiliary targets beyond the final material (`ValueConfig.aux_targets`).

- each target in `AUX_TARGETS` read off the right rows of a synthetic dataset whose values
  say which row they came from (`aux_target_arrays`), and refused on rows out of order;
- each head keeps its target's symmetry under exchanging the sides, and the win logit is
  `forward`'s;
- the losses reach the trunk; off (the default) builds and trains what it did before;
- `tools/train_value.py --aux-targets` saves a plain model that code without the field reads.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="value function needs the optional learn group")

from pokeuraou.encode import ENCODING_REVISION, Encoder, side_feature_names  # noqa: E402
from pokeuraou.regulation import load_regulation, repo_root  # noqa: E402
from pokeuraou.selfplay import position_from_sets  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402
from pokeuraou.value import (  # noqa: E402
    AUX_TARGETS,
    Dataset,
    ValueConfig,
    aux_target_arrays,
    build,
    load_model,
    parse_aux_targets,
    save_dataset,
    save_model,
    train,
)

MC = "gen9championsvgc2026regmc"
CPU = torch.device("cpu")
ALIVE, HP = (side_feature_names().index(n) for n in ("alive_fraction", "team_hp_fraction"))
TURNS = (1, 2, 2, 4)  # a replacement decision shares turn 2 with the move before it
GAMES = 8


def _dataset(encoder: Encoder) -> Dataset:
    reg = encoder.reg
    sets = load_roster("rizabanadohido").sets
    n = reg.meta.picked_team_size
    rng = np.random.default_rng(428)
    positions, game, outcome = [], [], []
    for g in range(GAMES):
        order = rng.permutation(len(sets))
        own = [sets[i] for i in order[:n]]
        foe = [sets[i] for i in order[len(sets) - n :]]
        # side 0 wins games 0, 2, 4; side 1 wins 1, 3, 5; 6 and 7 are draws
        result = 0.5 if g >= 6 else float(g % 2 == 0)
        for _ in TURNS:
            positions.append(position_from_sets(reg, own, foe))
            game.append(g)
            outcome.append(result)
    rows = len(positions)
    encoded = encoder.encode_positions(positions)
    fainted = encoder.mon_names.index("fainted")
    for r in range(rows):
        # Material that names its row: side 0 alive = 1 - r/100, side 1 = 0.9 - r/200.
        encoded.side[r, 0, ALIVE] = 1.0 - r / 100
        encoded.side[r, 1, ALIVE] = 0.9 - r / 200
        encoded.side[r, 0, HP] = 0.8 - r / 300
        encoded.side[r, 1, HP] = 0.7 - r / 400
    for g in range(GAMES):
        last = 4 * g + 3
        # The last row lists side 0's Pokemon in another order, and the one that is now
        # first (slot 2 before) has fainted: a target read by slot instead of by species
        # would mark the wrong Pokemon.
        for name in ("species", "ability", "item", "moves", "mon", "mask"):
            array = getattr(encoded, name)
            array[last, 0, [0, 2]] = array[last, 0, [2, 0]]
        encoded.mon[last, 0, 0, fainted] = 1.0
        encoded.mon[last, 1, 3, fainted] = 1.0  # side 1's slot 3, in place
    return Dataset(
        encoded=encoded,
        outcome=np.array(outcome, np.float32),
        game=np.array(game, np.int32),
        turn=np.tile(np.array(TURNS), GAMES).astype(np.int16),
        search_value=np.linspace(0.05, 0.95, rows).astype(np.float32),
        hp_share=np.full(rows, 0.5, np.float32),
        kind=np.zeros(rows, np.int8),
        foe=np.zeros(rows, np.int32),
        foe_names=("roster",),
    )


@pytest.fixture(scope="module")
def mc():  # noqa: ANN201
    encoder = Encoder(load_regulation(MC))
    return encoder, _dataset(encoder)


def _material(r: int) -> np.ndarray:
    return np.array([[1.0 - r / 100, 0.8 - r / 300], [0.9 - r / 200, 0.7 - r / 400]], np.float32)


def _end(g: int) -> np.ndarray:
    end = _material(4 * g + 3)
    if g < 6:
        end[1 if g % 2 == 0 else 0] = 0.0  # the loser has nothing left
    return end


def test_parse_refuses_unknown_names_and_bad_weights() -> None:
    assert parse_aux_targets("") == []
    assert parse_aux_targets("ahead2:1.5,search:0.5") == [("ahead2", 1.5), ("search", 0.5)]
    for bad in ("nope:1", "ahead2", "ahead2:0", "ahead2:1,ahead2:2"):
        with pytest.raises(ValueError):
            parse_aux_targets(bad)


def test_each_target_reads_the_rows_it_names(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    t = aux_target_arrays(dataset, list(AUX_TARGETS), encoder)
    for name, (kind, dim, _loss) in AUX_TARGETS.items():
        want = {"anti": (dim,), "sym": (dim,), "side": (2, dim), "mon": (2, 4)}[kind]
        assert t[name].shape == (len(dataset), *want) and t[name].dtype == np.float32, name
    for g in range(GAMES):
        r0 = 4 * g
        end = _end(g)
        for i in range(4):
            assert np.allclose(t["end_side"][r0 + i], end, atol=1e-6)
        # rows of turns 1, 2, 2, 4. ahead1 from turn 1 is the first turn-2 row (r0+1); from
        # either turn-2 row it is the turn-4 row; from turn 4 the end.
        a1 = [_material(r0 + 1), _material(r0 + 3), _material(r0 + 3), end]
        a2 = [_material(r0 + 3), _material(r0 + 3), _material(r0 + 3), end]
        a4 = [end, end, end, end]
        for name, later in (("ahead1", a1), ("ahead2", a2), ("ahead4", a4)):
            got = t[name][r0 : r0 + 4]
            assert np.allclose(got, [m[0] - m[1] for m in later], atol=1e-6), (name, g)
        now = [_material(r0 + i) for i in range(4)]
        ko = [(now[i][:, 0] - a1[i][:, 0]) * 4 for i in range(4)]
        assert np.allclose(t["ko_next"][r0 : r0 + 4, :, 0], np.maximum(ko, 0), atol=1e-5)
        assert np.allclose(t["turns_left"][r0 : r0 + 4, 0], [4 / 8, 3 / 8, 3 / 8, 1 / 8])
        # mon_end by species: side 0's slot 2 (slot 0 in the last row) fainted, side 1's
        # slot 3. The loser's whole side is down; a draw keeps the last row.
        mons = t["mon_end"][r0]
        side0 = [1.0, 1.0, 0.0, 1.0]
        side1 = [1.0, 1.0, 1.0, 0.0]
        if g < 6 and g % 2 == 0:
            side1 = [0.0] * 4
        elif g < 6:
            side0 = [0.0] * 4
        assert mons.tolist() == [side0, side1], g
        last = t["mon_end"][r0 + 3]
        assert last[0].tolist() == [side0[2], side0[1], side0[0], side0[3]]  # re-ordered row
    assert np.allclose(t["search"][:, 0], dataset.search_value)


def test_rows_out_of_order_are_refused(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    fields = {f: getattr(dataset, f) for f in Dataset.__dataclass_fields__}
    turns = dataset.turn.copy()
    turns[1], turns[2] = 4, 1  # a turn that goes back inside game 0
    with pytest.raises(ValueError):
        aux_target_arrays(Dataset(**{**fields, "turn": turns}), ["ahead2"], encoder)
    with pytest.raises(ValueError):
        aux_target_arrays(
            Dataset(**{**fields, "game": dataset.game[::-1].copy()}), ["end_side"], encoder
        )
    with pytest.raises(ValueError, match="encoder"):
        aux_target_arrays(dataset, ["mon_end"])


def test_each_head_keeps_its_symmetry_and_the_logit_is_forwards(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    spec = ",".join(f"{name}:1" for name in AUX_TARGETS)
    torch.manual_seed(5)
    net = build(encoder, ValueConfig(aux_weight=0.5, aux_targets=spec, state_inputs=True)).eval()
    batch = dataset.tensors(np.arange(len(dataset)), CPU)
    flipped = {
        k: (v.flip(1) if v.dim() > 1 and v.shape[1] == 2 else v) for k, v in batch.items()
    }
    with torch.no_grad():
        logit, aux, heads = net.forward_heads(batch)
        logit_f, aux_f, heads_f = net.forward_heads(flipped)
        assert torch.allclose(logit, net(batch), atol=1e-6)
        assert torch.allclose(aux, net.forward_aux(batch)[1], atol=1e-6)
    assert set(heads) == set(AUX_TARGETS)
    for name, out in heads.items():
        kind = AUX_TARGETS[name][0]
        if kind == "anti":
            assert torch.allclose(out, -heads_f[name], atol=1e-5), name
        elif kind == "sym":
            assert torch.allclose(out, heads_f[name], atol=1e-5), name
        else:  # side, mon: the sides trade places
            assert torch.allclose(out, heads_f[name].flip(1), atol=1e-5), name
        assert float(out.abs().max()) > 0, name


def _fit(encoder, dataset, config, **kwargs):  # noqa: ANN001, ANN202
    torch.manual_seed(config.seed)
    net = build(encoder, config)
    _h, weights = train(net, dataset, config, device=CPU, holdout=0.25, **kwargs)
    return net, weights


def _config(**kw) -> ValueConfig:  # noqa: ANN003
    return ValueConfig(epochs=2, batch_size=8, keep="last", seed=3, split_seed=0, **kw)


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_off_builds_and_trains_the_old_net(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    assert ValueConfig().aux_targets == ""
    torch.manual_seed(1)
    plain = build(encoder, ValueConfig())
    assert not hasattr(plain, "aux_heads")
    torch.manual_seed(1)
    headed = build(encoder, ValueConfig(aux_targets="ahead2:1"))
    # The heads come after every other module: the rest draws the same initialisation.
    shared = {k: v for k, v in headed.state_dict().items() if not k.startswith("aux_heads.")}
    assert _same(plain.state_dict(), shared)
    _n, a = _fit(encoder, dataset, _config())
    _n, b = _fit(encoder, dataset, _config(), aux_targets={})
    assert _same(a, b)


@pytest.mark.parametrize("name", sorted(AUX_TARGETS))
def test_each_loss_reaches_the_trunk(mc, name) -> None:  # noqa: ANN001
    encoder, dataset = mc
    targets = aux_target_arrays(dataset, [name], encoder)
    _n, plain = _fit(encoder, dataset, _config())
    net, headed = _fit(encoder, dataset, _config(aux_targets=f"{name}:5"), aux_targets=targets)
    trunk = [k for k in plain if k.startswith("mon_mlp.0")]
    assert trunk and not _same({k: plain[k] for k in trunk}, {k: headed[k] for k in trunk})
    assert any(k.startswith(f"aux_heads.{name}.") for k in headed)


def test_the_targets_must_match_the_config(mc) -> None:  # noqa: ANN001
    encoder, dataset = mc
    targets = aux_target_arrays(dataset, ["ahead2"], encoder)
    with pytest.raises(ValueError, match="same targets"):
        _fit(encoder, dataset, _config(aux_targets="ahead2:1"))
    with pytest.raises(ValueError, match="same targets"):
        _fit(encoder, dataset, _config(), aux_targets=targets)


def _tool():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "train_value_tool_428", repo_root() / "tools" / "train_value.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_tool_trains_the_heads_and_saves_a_plain_model(mc, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    encoder, dataset = mc
    data = tmp_path / "mc.npz"
    save_dataset(data, dataset, meta={"format_id": MC, "encoding_revision": ENCODING_REVISION})
    torch.manual_seed(7)
    net = build(encoder, ValueConfig()).eval()
    parent = tmp_path / "parent.pt"
    save_model(parent, net, net.state_dict(), encoder.vocab, ValueConfig(), meta={"m": 1},
               widths=net.in_widths)

    def run(out, *flags) -> None:  # noqa: ANN001, ANN002
        monkeypatch.setattr(
            sys, "argv",
            ["train_value.py", "--data", str(data), "--init-from", str(parent), "--epochs", "1",
             "--batch-size", "8", "--keep", "last", "--lr", "5e-4", "--out", str(out),
             "--device", "cpu", "--state-inputs", *flags],
        )
        _tool().main()

    base, headed = tmp_path / "base.pt", tmp_path / "headed.pt"
    run(base)
    run(headed, "--aux-weight", "2", "--aux-targets", "mon_end:1,ahead2:2")
    blob = torch.load(headed, weights_only=False)
    assert "aux_targets" not in blob["config"] and blob["config"]["aux_weight"] == 0.0
    assert not any(k.startswith("aux") for k in blob["weights"])  # both kinds of head dropped
    assert torch.load(base, weights_only=False)["weights"].keys() == blob["weights"].keys()
    plain_net, _ = load_model(base, encoder)
    headed_net, _ = load_model(headed, encoder)
    assert not hasattr(headed_net, "aux_heads") and headed_net.config.state_inputs
    sa, sb = plain_net.state_dict(), headed_net.state_dict()
    assert not _same(sa, sb)  # the heads moved the trunk
