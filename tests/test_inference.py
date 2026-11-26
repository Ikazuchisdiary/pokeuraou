"""The client of the inference server (`inference.RemoteValue`) against the tests' own stand-in
server (`tests/_served.py`, the protocol of the Rust server).

The Python server this file used to test (`served_model` on a torch net, its CUDA graphs, its
eager-pass gate) is gone (IKA-460); the Rust server's answers against torch are
`tests/test_rust_inference_server.py`'s. What stays is the client's side: a batch cut where
`BatchedValue` cuts it, blocks scored as their own requests, an arm that says what it holds, a
named refusal, a worker that dies without taking the server with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.inference import RemoteValue

from ._served import encoded_of, hp_share_model, serve, subset


@pytest.fixture(scope="module")
def parts(reg):
    register_mega_stones(reg)
    return reg, Encoder(reg)


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


def _server(encoder, **kwargs):  # noqa: ANN202
    share = hp_share_model(encoder)

    def model(arrays, rows):  # noqa: ANN001, ANN202
        return share(encoded_of(arrays))

    return serve({"value": model}, **kwargs), share


def test_the_client_returns_what_the_server_scored(parts):
    regulation, encoder = parts
    (server, address), share = _server(encoder)
    positions = _positions(regulation, 9)
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=4 << 20) as remote:
            got = remote(positions)
    finally:
        server.shutdown()
    assert np.array_equal(got, share(encoder.encode_positions(positions)))
    # A control that can fail: a model answering a constant is not what comes back.
    assert not np.array_equal(got, np.full(len(got), 0.25))


def test_a_batch_longer_than_one_chunk_is_cut_at_batch_size(parts):
    """The server is asked in chunks of the client's `batch_size` (a card's answer depends on
    the row count of its call, so the cut is part of the leaf)."""
    regulation, encoder = parts
    sizes: list[int] = []
    share = hp_share_model(encoder)

    def model(arrays, rows):  # noqa: ANN001, ANN202
        sizes.append(rows)
        return share(encoded_of(arrays))

    server, address = serve({"value": model})
    positions = _positions(regulation, 45)
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=8 << 20, batch_size=7) as remote:
            got = remote(positions)
    finally:
        server.shutdown()
    assert sizes == [7] * 6 + [3]
    assert np.array_equal(got, share(encoder.encode_positions(positions)))


def test_a_worker_dying_leaves_the_server_up(parts):
    regulation, encoder = parts
    (server, address), _share = _server(encoder)
    try:
        casualty = RemoteValue(address, "value", encoder, buffer_bytes=4 << 20)
        casualty(_positions(regulation, 4))
        casualty._sock.close()  # noqa: SLF001 -- the ungraceful kind of departure
        with RemoteValue(address, "value", encoder, buffer_bytes=4 << 20) as survivor:
            got = survivor(_positions(regulation, 4))
        assert len(got) == 4
    finally:
        server.shutdown()


def test_a_chunk_too_large_for_the_buffer_is_refused(parts):
    regulation, encoder = parts
    (server, address), _share = _server(encoder)
    try:
        with (
            RemoteValue(address, "value", encoder, buffer_bytes=1 << 16) as remote,
            pytest.raises(RuntimeError, match="raise buffer_bytes"),
        ):
            remote(_positions(regulation, 40))
    finally:
        server.shutdown()


def test_an_unknown_model_is_named_rather_than_guessed(parts):
    regulation, encoder = parts
    (server, address), _share = _server(encoder)
    try:
        with (
            RemoteValue(address, "baseline", encoder, buffer_bytes=4 << 20) as remote,
            pytest.raises(RuntimeError, match="no model named"),
        ):
            remote(_positions(regulation, 4))
    finally:
        server.shutdown()


def test_an_arm_says_what_it_holds_rather_than_what_it_is_called(parts):
    """`describe` is how a match can record its leaf honestly: a worker is told which arm to
    play and never what that arm holds."""
    _regulation, encoder = parts
    share = hp_share_model(encoder)

    def model(arrays, rows):  # noqa: ANN001, ANN202
        return share(encoded_of(arrays))

    server, address = serve(
        {"value": model, "baseline": model},
        arms={"value": ["value-all.pt", "value-all-s1.pt"], "baseline": ["value-gen8.pt"]},
    )
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=1 << 20) as remote:
            assert remote.describe() == ["value-all.pt", "value-all-s1.pt"]
        with RemoteValue(address, "baseline", encoder, buffer_bytes=1 << 20) as remote:
            assert remote.describe() == ["value-gen8.pt"]
        with (
            RemoteValue(address, "nope", encoder, buffer_bytes=1 << 20) as remote,
            pytest.raises(RuntimeError, match="no arm named 'nope'"),
        ):
            remote.describe()
    finally:
        server.shutdown()


def test_blocks_in_one_round_trip_are_scored_as_their_own_requests(parts):
    """IKA-291: `from_encoded_segments` sends many blocks at once, and each is scored at its own
    size. The stand-in's answer moves with the row count of its call (as a card's does), so
    equality with one request per block says the blocks were not merged -- and the stacked
    request, the positive control, is not equal."""
    regulation, encoder = parts

    def sized(arrays, rows):  # noqa: ANN001, ANN202
        base = arrays["species"].reshape(rows, -1).sum(axis=1).astype(np.float64)
        return base * 1e-3 + rows * 1e-9

    encoded = encoder.encode_positions(_positions(regulation, 30))
    blocks, at = [], 0
    for size in [1, 5, 7, 12, 3, 2]:
        blocks.append(subset(encoded, list(range(at, at + size))))
        at += size
    server, address = serve({"value": sized})
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=4 << 20) as remote:
            alone = [remote.from_encoded(block) for block in blocks]
            before = remote.calls
            together = remote.from_encoded_segments(blocks)
            round_trips = remote.calls - before
            merged = remote.from_encoded(subset(encoded, list(range(30))))
    finally:
        server.shutdown()
    assert round_trips == 1
    assert [len(t) for t in together] == [len(b.species) for b in blocks]
    for got, want in zip(together, alone, strict=True):
        assert np.array_equal(got, want)
    assert not np.array_equal(np.concatenate(together), merged)
