"""IKA-461: the Rust inference server answers what torch answers, to within float32 noise.

A server of `pokeuraou-damage inference-server` is started on nets with random weights
(`rustnet.write_net`), and a request goes to it the way `RemoteValue` sends one. Every kernel
has its own check against torch through the op `stages`, which hands back the first net's
activations after each step; the end-to-end checks ask through `RemoteValue.from_encoded`.

The inputs are real encoded positions that are then fuzzed array by array -- absent Pokemon,
empty move slots, random ids, random numeric columns -- so a kernel's reading of a mask or a
padding id is exercised, which starting positions alone never do. A control: the same request
against torch nets with other weights must differ, so the comparison can fail.

Needs a CUDA card and CUDA 12's DLLs (the `.venv`'s torch has them), so it skips without a
card, with the reason `tools/ci_skip_audit.py` counts as `cuda`.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoded, Encoder  # noqa: E402
from pokeuraou.inference import ARRAYS, RemoteValue, _plan  # noqa: E402
from pokeuraou.regulation import regulation_dir  # noqa: E402
from pokeuraou.rustnet import write_net  # noqa: E402
from pokeuraou.value import BatchedValue, ValueConfig, _masked_max, _masked_mean, build  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs need a card (the Rust inference server)"
)

FORMAT_ID = "gen9championsvgc2026regmc"
CONFIGS = {
    "base": ValueConfig(),
    "state": ValueConfig(state_inputs=True),
    "bind": ValueConfig(state_inputs=True, bind_inputs=True),
}


def _positions(regulation, count: int):
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import all_selections, load_roster

    roster = load_roster("rizabanadohido")
    picks = list(all_selections(regulation.meta.team_size, regulation.meta.picked_team_size))
    out = []
    for index in range(count):
        ours = [roster.sets[i] for i in picks[index % len(picks)]]
        theirs = [roster.sets[i] for i in picks[(index + 3) % len(picks)]]
        out.append(position_from_sets(regulation, ours, theirs))
    return out


def _net(encoder, config: ValueConfig, seed: int):
    torch.manual_seed(seed)
    net = build(encoder, config).eval()
    with torch.no_grad():
        # Layer norms start as the identity; move them so a wrong gamma or beta shows.
        for module in net.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.add_(0.3 * torch.randn_like(module.weight))
                module.bias.add_(0.3 * torch.randn_like(module.bias))
    return net


def _fuzz(encoded: Encoded, net, encoder, rows: int, seed: int) -> Encoded:
    """`rows` rows made from the positions' encoding, then changed array by array."""
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(encoded.species), rows)
    arrays = {name: np.ascontiguousarray(getattr(encoded, name)[pick]) for name in ARRAYS}
    sizes = encoder.vocab.sizes
    m = arrays["species"].shape[-1]
    for name in ("species", "ability", "item"):
        arrays[name] = rng.integers(0, sizes[name], arrays[name].shape).astype(arrays[name].dtype)
    moves = rng.integers(0, sizes["move"], arrays["moves"].shape)
    moves[rng.random(moves.shape) < 0.25] = 0
    arrays["moves"] = moves.astype(arrays["moves"].dtype)
    arrays["mask"] = (rng.random((rows, 2, m)) < 0.8).astype(np.float32)
    mon = arrays["mon"].astype(np.float32) + 0.5 * rng.standard_normal(arrays["mon"].shape).astype(np.float32)
    mon[..., net._active_feature] = (rng.random((rows, 2, m)) < 0.4).astype(np.float32)  # noqa: SLF001
    if net._move_ids:  # noqa: SLF001
        ids = rng.integers(0, sizes["move"], (rows, 2, m, net._move_ids))  # noqa: SLF001
        ids[rng.random(ids.shape) < 0.3] = 0
        lo = net.in_widths["mon"] - net._move_ids  # noqa: SLF001
        if net._bind:  # noqa: SLF001
            lo = encoder.state_widths["mon"] - net._move_ids  # noqa: SLF001
        mon[..., lo : lo + net._move_ids] = ids  # noqa: SLF001
    arrays["mon"] = mon
    for name in ("side", "field"):
        noise = 0.5 * rng.standard_normal(arrays[name].shape).astype(np.float32)
        arrays[name] = arrays[name].astype(np.float32) + noise
    return Encoded(**arrays, unknown_volatiles={})


class RustServer:
    def __init__(self, arms: dict[str, list[Path]], tmp_path: Path, slots: int = 2) -> None:
        exe = rustnode.binary_path()
        assert exe.exists(), f"{exe} is not built: cd rust && cargo build --release"
        dlls = Path(torch.__file__).parent / "lib"
        command = [
            str(exe), "inference-server", "--regulation", str(regulation_dir() / f"{FORMAT_ID}.json"),
            "--dll-dir", str(dlls), "--slots", str(slots), "--report-every", "0",
        ]
        for name, files in arms.items():
            command += ["--arm", name, *map(str, files)]
        self.log = (tmp_path / "server.log").open("w")
        self.process = subprocess.Popen(  # noqa: S603
            command, stdout=subprocess.PIPE, stderr=self.log, text=True
        )
        self.address = self.process.stdout.readline().strip()
        assert self.address, f"the server did not start; see {tmp_path / 'server.log'}"

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=20)
        self.log.close()


def _write(net, encoder, path: Path, name: str) -> Path:
    write_net(net, encoder, path, source=name)
    return path


@pytest.fixture(scope="module")
def encoder(reg):
    register_mega_stones(reg)
    return Encoder(reg)


@pytest.fixture(scope="module")
def base_encoded(reg, encoder):
    return encoder.encode_positions(_positions(reg, 40))


@pytest.fixture(scope="module")
def served(encoder, tmp_path_factory):
    """One server, an arm per config, an ensemble arm of two nets, and a control arm."""
    tmp = tmp_path_factory.mktemp("rust-server")
    nets = {name: _net(encoder, config, seed=11 + i) for i, (name, config) in enumerate(CONFIGS.items())}
    other = _net(encoder, CONFIGS["bind"], seed=99)
    files = {name: _write(net, encoder, tmp / f"{name}.rnet", f"{name}.pt") for name, net in nets.items()}
    files["bind2"] = _write(other, encoder, tmp / "bind2.rnet", "bind2.pt")
    arms = {name: [path] for name, path in files.items()}
    arms["pair"] = [files["bind"], files["bind2"]]
    server = RustServer(arms, tmp)
    yield server, {**nets, "bind2": other}
    server.close()


def _ask(address: str, request: dict) -> dict:
    host, _, port = address.rpartition(":")
    with socket.create_connection((host, int(port))) as sock:
        file = sock.makefile("rwb")
        file.write((json.dumps(request) + "\n").encode())
        file.flush()
        return json.loads(file.readline())


def _batch(encoded: Encoded) -> dict:
    return {name: torch.from_numpy(np.asarray(getattr(encoded, name))) for name in ARRAYS}


def _stages_from_server(address: str, arm: str, encoded: Encoded) -> dict[str, np.ndarray]:
    layout, used = _plan(encoded)
    rows = len(encoded.species)
    result_offset = (used + 63) & ~63
    stage_offset = result_offset + rows * 8
    capacity = 64 << 20
    block = shared_memory.SharedMemory(create=True, size=stage_offset + capacity)
    try:
        for item in layout:
            array = np.ascontiguousarray(getattr(encoded, item["name"]))
            target = np.frombuffer(block.buf, dtype=array.dtype, count=array.size, offset=item["offset"])
            target[:] = array.reshape(-1)
            del target
        reply = _ask(address, {
            "op": "stages", "model": arm, "shm": block.name, "rows": rows, "layout": layout,
            "result_offset": result_offset, "stage_offset": stage_offset, "stage_bytes": capacity,
        })
        assert reply["ok"], reply
        out = {}
        for stage in reply["stages"]:
            values = np.frombuffer(
                block.buf, dtype=np.float32, count=stage["count"], offset=stage["offset"]
            ).copy()
            out[stage["name"]] = values.reshape(-1, stage["width"])
        return out
    finally:
        block.close()
        block.unlink()


def _torch_stages(net, batch: dict) -> dict[str, np.ndarray]:
    with torch.no_grad():
        features, _ = net.mon_inputs(batch)
        mon = net.mon_mlp(features)
        present = batch["mask"].unsqueeze(-1)
        is_active = batch["mon"][..., net._active_feature].unsqueeze(-1)  # noqa: SLF001
        active, bench = present * is_active, present * (1.0 - is_active)
        pooled = torch.cat(
            [_masked_mean(mon, active), _masked_max(mon, active), _masked_mean(mon, bench),
             _masked_max(mon, bench), net.columns(batch, "side")], dim=-1)
        sides = net.side_mlp(pooled)
        ours, theirs = sides[:, 0], sides[:, 1]
        field = net.columns(batch, "field")
        head_in = torch.cat([torch.cat([ours, theirs, field], -1), torch.cat([theirs, ours, field], -1)], 0)
        head = net.head[:-1](head_in)
        probability = torch.sigmoid(net(batch))
    flat = lambda t: t.reshape(-1, t.shape[-1]).numpy()  # noqa: E731
    return {
        "features": flat(features), "mon": flat(mon), "pooled": flat(pooled), "sides": flat(sides),
        "head_in": flat(head_in), "head": flat(head), "probability": probability.numpy()[:, None],
    }


#: The largest difference per stage the test allows (absolute; the stages are O(1) after a
#: layer norm). The measured values are in records/IKA-461.md.
TOLERANCE = {
    "features": 0.0, "mon": 2e-5, "pooled": 2e-5, "sides": 2e-5, "head_in": 2e-5, "head": 5e-5,
    "probability": 2e-6,
}


@pytest.mark.parametrize("config", list(CONFIGS))
def test_every_kernel_matches_torch_on_fuzzed_rows(served, encoder, base_encoded, config):
    server, nets = served
    net = nets[config]
    encoded = _fuzz(base_encoded, net, encoder, rows=300, seed=5)
    got = _stages_from_server(server.address, config, encoded)
    want = _torch_stages(net, _batch(encoded))
    assert set(got) == set(want)
    worst = {}
    for name in want:
        assert got[name].shape == want[name].shape, name
        worst[name] = float(np.abs(got[name] - want[name]).max())
    print(f"{config}: max |rust - torch| per stage {worst}")
    for name, limit in TOLERANCE.items():
        assert worst[name] <= limit, f"{config} {name}: {worst[name]:.3e} > {limit:.1e} ({worst})"
    # The stages are not all zeros or all equal: a comparison of two blank arrays passes.
    for name in ("features", "pooled", "head"):
        assert np.abs(want[name]).max() > 0.1 and want[name].std() > 0.05, name


def test_the_comparison_can_fail(served, encoder, base_encoded):
    """The same rows against torch nets with other weights: the probability must differ."""
    server, nets = served
    encoded = _fuzz(base_encoded, nets["bind"], encoder, rows=300, seed=5)
    got = _stages_from_server(server.address, "bind", encoded)["probability"]
    other = _torch_stages(nets["bind2"], _batch(encoded))["probability"]
    assert np.abs(got - other).max() > 1e-2


def _remote(address: str, arm: str, encoder, encoded: Encoded) -> np.ndarray:
    with RemoteValue(address, arm, encoder, buffer_bytes=128 << 20) as remote:
        return remote.from_encoded(encoded)


@pytest.mark.parametrize("rows", [1, 7, 64, 65, 600, 1025, 4096, 5000])
def test_remote_value_matches_the_torch_leaf_at_every_size(served, encoder, base_encoded, rows):
    server, nets = served
    net = nets["bind"]
    encoded = _fuzz(base_encoded, net, encoder, rows=rows, seed=rows)
    got = _remote(server.address, "bind", encoder, encoded)
    device = torch.device("cuda")
    want = BatchedValue(net.to(device), encoder, device=device).from_encoded(encoded)
    net.cpu()
    assert got.shape == want.shape == (rows,)
    assert got.dtype == np.float64
    assert np.abs(got - want).max() < 5e-6, f"{np.abs(got - want).max():.3e}"
    assert rows == 1 or got.std() > 0.01


def test_an_ensemble_arm_averages_the_logits(served, encoder, base_encoded):
    server, nets = served
    encoded = _fuzz(base_encoded, nets["bind"], encoder, rows=700, seed=3)
    got = _remote(server.address, "pair", encoder, encoded)
    device = torch.device("cuda")
    pair = BatchedValue([nets["bind"].to(device), nets["bind2"].to(device)], encoder, device=device)
    want = pair.from_encoded(encoded)
    nets["bind"].cpu()
    nets["bind2"].cpu()
    assert np.abs(got - want).max() < 5e-6, f"{np.abs(got - want).max():.3e}"
    single = _remote(server.address, "bind", encoder, encoded)
    assert np.abs(got - single).max() > 1e-3  # the second net matters


def test_several_workers_at_once_get_what_they_get_alone(served, encoder, base_encoded):
    server, nets = served
    sets = [_fuzz(base_encoded, nets["bind"], encoder, rows=300 + 37 * i, seed=100 + i) for i in range(6)]
    alone = [_remote(server.address, "bind", encoder, e) for e in sets]
    got: dict[int, list[np.ndarray]] = {}

    def work(i: int) -> None:
        with RemoteValue(server.address, "bind", encoder, buffer_bytes=64 << 20) as remote:
            got[i] = [remote.from_encoded(sets[i]) for _ in range(8)]

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    for i in range(6):
        for answer in got[i]:
            # Rows are padded to a bucket, so an answer may move in the last place with the
            # batch; the same request twice is the same answer.
            assert np.abs(answer - alone[i]).max() < 1e-6
            assert np.array_equal(answer, got[i][0])


def test_ping_describe_and_errors(served, encoder):
    server, _ = served
    models = _ask(server.address, {"op": "ping"})["models"]
    assert sorted(models) == ["base", "bind", "bind2", "pair", "state"]
    arms = _ask(server.address, {"op": "describe"})["arms"]
    assert arms["pair"] == ["bind.pt", "bind2.pt"]
    reply = _ask(server.address, {
        "op": "score", "model": "nope", "shm": "x", "rows": 1, "layout": [], "result_offset": 0,
    })
    assert not reply["ok"] and "nope" in reply["error"]
    assert not _ask(server.address, {"op": "frobnicate"})["ok"]
    # A Q request with no Q server to pass it to is refused by name, not dropped.
    reply = _ask(server.address, {"op": "q", "model": "q"})
    assert not reply["ok"] and "Q" in reply["error"]


def test_the_server_prints_the_library_versions(served):
    server, _ = served
    log = Path(server.log.name).read_text()
    assert "NVRTC 12." in log and "cuBLAS 12." in log, log
    assert sys.platform  # the line is for the record
