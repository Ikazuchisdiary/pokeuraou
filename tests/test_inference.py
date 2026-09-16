"""The server has one job beyond being a server: it must not change any answer.

A leaf differing by 1.9e-06 moves an equilibrium, so "about as good" is not a thing this
can be. The test is equality, not closeness, and it is equality against the local
`BatchedValue` the search uses today.

The second test is the one the design turns on. A CUDA answer depends on how many rows were
in the call -- 5.96e-08 between a row alone and the same row among eight -- so the server
must chunk on the same boundary `BatchedValue` does, and must not merge one worker's
request with another's. Two clients asking at once must get what they would have got alone.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.inference import RemoteValue, load_models, serve  # noqa: E402
from pokeuraou.value import BatchedValue, ValueConfig, build  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def parts(reg):
    register_mega_stones(reg)
    encoder = Encoder(reg)
    torch.manual_seed(7)
    net = build(encoder, ValueConfig()).eval()
    return reg, encoder, net


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


@pytest.mark.parametrize("device_name", DEVICES)
def test_the_server_returns_exactly_what_the_local_leaf_returns(parts, device_name):
    regulation, encoder, net = parts
    device = torch.device(device_name)
    local = BatchedValue(net.to(device), encoder, device=device)
    positions = _positions(regulation, 40)
    expected = local(positions)

    from pokeuraou.inference import local_model

    server, address = serve({"value": local_model(net.to(device), device)})
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=8 << 20) as remote:
            got = remote(positions)
            assert remote.evaluated == len(positions)
    finally:
        server.shutdown()

    assert np.array_equal(got, expected), (
        f"max difference {np.abs(got - expected).max():.3e} -- the server must not change "
        f"an answer, because 1.9e-06 moves an equilibrium"
    )


@pytest.mark.parametrize("device_name", DEVICES)
def test_two_workers_at_once_get_what_they_would_get_alone(parts, device_name):
    """The whole reason requests are not merged across workers."""
    regulation, encoder, net = parts
    device = torch.device(device_name)
    local = BatchedValue(net.to(device), encoder, device=device)
    # Deliberately different row counts: merging them would make one batch of 52, and a
    # CUDA answer depends on the count.
    first = _positions(regulation, 12)
    second = _positions(regulation, 40)
    alone = {12: local(first), 40: local(second)}

    from pokeuraou.inference import local_model

    server, address = serve({"value": local_model(net.to(device), device)})
    results: dict[int, np.ndarray] = {}
    try:

        def ask(positions):
            with RemoteValue(address, "value", encoder, buffer_bytes=8 << 20) as remote:
                for _ in range(6):
                    results[len(positions)] = remote(positions)

        threads = [
            threading.Thread(target=ask, args=(first,)),
            threading.Thread(target=ask, args=(second,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        server.shutdown()

    for rows, expected in alone.items():
        assert np.array_equal(results[rows], expected), (
            f"{rows} rows changed when another worker was asking at the same time"
        )


def test_a_worker_dying_leaves_the_server_up(parts):
    regulation, encoder, net = parts
    device = torch.device("cpu")
    from pokeuraou.inference import local_model

    server, address = serve({"value": local_model(net.to(device), device)})
    try:
        casualty = RemoteValue(address, "value", encoder, buffer_bytes=4 << 20)
        casualty(_positions(regulation, 4))
        casualty._sock.close()  # noqa: SLF001 -- the ungraceful kind of departure

        with RemoteValue(address, "value", encoder, buffer_bytes=4 << 20) as survivor:
            got = survivor(_positions(regulation, 4))
        assert len(got) == 4
    finally:
        server.shutdown()


def test_a_batch_too_large_for_the_buffer_refuses_rather_than_splitting(parts):
    """Splitting would change the batch size, which changes the answer on CUDA."""
    regulation, encoder, net = parts
    from pokeuraou.inference import local_model

    server, address = serve({"value": local_model(net.to(torch.device("cpu")),
                                                  torch.device("cpu"))})
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=1 << 16) as remote:
            with pytest.raises(RuntimeError, match="raise buffer_bytes"):
                remote(_positions(regulation, 40))
    finally:
        server.shutdown()


def test_an_unknown_model_is_named_rather_than_guessed(parts):
    regulation, encoder, net = parts
    from pokeuraou.inference import local_model

    server, address = serve({"value": local_model(net.to(torch.device("cpu")),
                                                  torch.device("cpu"))})
    try:
        with RemoteValue(address, "baseline", encoder, buffer_bytes=4 << 20) as remote:
            with pytest.raises(RuntimeError, match="no model named"):
                remote(_positions(regulation, 4))
    finally:
        server.shutdown()
