"""IKA-363: the inference server's merged road.

A request that asks for it (`RemoteValue(merge=True)`) is scored in one forward pass with
the other connections' requests that are waiting when the card comes free. The stand-in
model here answers with a trace of its call's size (1e-9 a row, as a card's answer moves
with the rows in its call), and takes a moment a pass, so requests from several threads
queue behind one another.

- a lone request on the merged road is a pass of its own size: the unmerged answer, to the bit;
- requests that arrive together share passes (the positive control: fewer passes than
  requests, and each answer carries the merged size), and each client still gets its own
  rows back, in order;
- a request on the unmerged road beside them is never merged.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.inference import ARRAYS, RemoteValue, serve


class _Sized:
    """(arrays, rows) -> values: a function of each row, plus 1e-9 per row of the call."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[int] = []

    def __call__(self, arrays: dict[str, np.ndarray], rows: int) -> np.ndarray:
        self.calls.append(rows)
        if self.delay:
            time.sleep(self.delay)
        raw = sum(np.asarray(arrays[name]).reshape(rows, -1).astype(np.float64).sum(axis=1)
                  * (k + 1) * 1e-3 for k, name in enumerate(ARRAYS))
        return (raw % 1.0) * 0.5 + rows * 1e-9


@pytest.fixture(scope="module")
def blocks(reg):  # noqa: ANN001, ANN201
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import all_selections, load_roster

    register_mega_stones(reg)
    encoder = Encoder(reg)
    roster = load_roster("rizabanadohido")
    picks = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
    positions = [
        position_from_sets(reg, [roster.sets[i] for i in picks[n % len(picks)]],
                           [roster.sets[i] for i in picks[(n + 3) % len(picks)]])
        for n in range(24)
    ]
    # Blocks of different sizes, as a deepening's children are.
    sizes = [3, 5, 1, 7, 2, 6]
    out, at = [], 0
    for size in sizes:
        out.append(encoder.encode_positions(positions[at : at + size]))
        at += size
    return encoder, out


def _alone(model: _Sized, encoded) -> np.ndarray:  # noqa: ANN001
    arrays = {name: getattr(encoded, name) for name in ARRAYS}
    return model(arrays, len(encoded.species))


def test_a_lone_merged_request_is_the_unmerged_answer(blocks) -> None:  # noqa: ANN001
    encoder, parts = blocks
    model = _Sized()
    server, address = serve({"value": model})
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=4 << 20, merge=True) as merged, \
                RemoteValue(address, "value", encoder, buffer_bytes=4 << 20) as plain:
            for block in parts:
                a = merged.from_encoded(block)
                b = plain.from_encoded(block)
                assert np.array_equal(a, b)
                assert np.array_equal(a, _alone(_Sized(), block))
        assert server.mergers["value"].passes == len(parts)
        assert server.mergers["value"].merged == 0
    finally:
        server.shutdown()


def test_requests_that_arrive_together_share_a_pass(blocks) -> None:  # noqa: ANN001
    encoder, parts = blocks
    model = _Sized(delay=0.05)
    server, address = serve({"value": model})
    clients = 6
    rounds = 4
    got: dict[int, list[np.ndarray]] = {}
    unmerged: list[np.ndarray] = []
    start = threading.Barrier(clients + 1)

    def ask(k: int) -> None:
        with RemoteValue(address, "value", encoder, buffer_bytes=4 << 20, merge=True) as remote:
            start.wait()
            got[k] = [remote.from_encoded(parts[k]) for _ in range(rounds)]

    def plain() -> None:
        with RemoteValue(address, "value", encoder, buffer_bytes=4 << 20) as remote:
            start.wait()
            unmerged.extend(remote.from_encoded(parts[0]) for _ in range(rounds))

    try:
        threads = [threading.Thread(target=ask, args=(k,)) for k in range(clients)]
        threads.append(threading.Thread(target=plain))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        merger = server.mergers["value"]
    finally:
        server.shutdown()
    assert merger.requests == clients * rounds
    # Positive control: passes were shared, and a shared answer carries the merged size.
    assert merger.merged > 0 and merger.passes < merger.requests
    moved = 0
    for k in range(clients):
        alone = _alone(_Sized(), parts[k])
        for values in got[k]:
            assert values.shape == alone.shape
            # Each client's own rows: only the size trace moves (1e-9 a row of the pass).
            assert np.max(np.abs(values - alone)) < 1e-6
            moved += not np.array_equal(values, alone)
    assert moved > 0
    # The unmerged road beside it: its own size every time, never merged.
    for values in unmerged:
        assert np.array_equal(values, _alone(_Sized(), parts[0]))
