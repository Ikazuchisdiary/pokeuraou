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
from pokeuraou.inference import RemoteValue, serve  # noqa: E402
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

    from pokeuraou.inference import served_model

    server, address = serve({"value": served_model(BatchedValue(net.to(device), encoder, device=device))})
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

    from pokeuraou.inference import served_model

    server, address = serve({"value": served_model(BatchedValue(net.to(device), encoder, device=device))})
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
    from pokeuraou.inference import served_model

    server, address = serve({"value": served_model(BatchedValue(net.to(device), encoder, device=device))})
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
    """A single chunk that does not fit is refused; a long batch is cut like BatchedValue.

    The refusal used to be for any batch larger than the buffer, on the grounds that
    splitting changes a batch's size and a CUDA answer depends on that. But the model
    behind the server splits at 8,192 whatever happens, so the rule forbade work without
    protecting anything -- a self-switch node of 1,048,576 leaves killed a worker in a real
    match while the direct path had been evaluating that node in 128 pieces all along. What
    is left is the case the buffer genuinely cannot serve: one chunk that does not fit.
    """
    regulation, encoder, net = parts
    from pokeuraou.inference import served_model

    server, address = serve(
        {"value": served_model(
            BatchedValue(net.to(torch.device("cpu")), encoder, device=torch.device("cpu"))
        )}
    )
    try:
        with (
            RemoteValue(address, "value", encoder, buffer_bytes=1 << 16) as remote,
            pytest.raises(RuntimeError, match="raise buffer_bytes"),
        ):
            remote(_positions(regulation, 40))
    finally:
        server.shutdown()


def test_an_unknown_model_is_named_rather_than_guessed(parts):
    regulation, encoder, net = parts
    from pokeuraou.inference import served_model

    server, address = serve(
        {"value": served_model(
            BatchedValue(net.to(torch.device("cpu")), encoder, device=torch.device("cpu"))
        )}
    )
    try:
        with (
            RemoteValue(address, "baseline", encoder, buffer_bytes=4 << 20) as remote,
            pytest.raises(RuntimeError, match="no model named"),
        ):
            remote(_positions(regulation, 4))
    finally:
        server.shutdown()


def test_an_arm_says_what_it_holds_rather_than_what_it_is_called(parts):
    """`describe` is how a match can record its leaf honestly.

    A worker is told which arm to play and never what that arm holds, so a name taken
    from its own command line is a name that can be wrong -- and the record it writes is
    what a rating is fitted from. This is the same failure the policy's position
    representation had: the file knew it wanted 408 numbers and not whose.
    """
    _regulation, encoder, net = parts
    device = torch.device("cpu")

    from pokeuraou.inference import served_model

    scorer = served_model(BatchedValue(net.to(device), encoder, device=device))
    server, address = serve(
        {"value": scorer, "baseline": scorer},
        arms={"value": ["value-all.pt", "value-all-s1.pt"], "baseline": ["value-gen8.pt"]},
    )
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=1 << 20) as remote:
            assert remote.describe() == ["value-all.pt", "value-all-s1.pt"]
        with RemoteValue(address, "baseline", encoder, buffer_bytes=1 << 20) as remote:
            assert remote.describe() == ["value-gen8.pt"]
        # An arm the server does not have is named, not guessed at -- the same rule the
        # score path already follows.
        with (
            RemoteValue(address, "nope", encoder, buffer_bytes=1 << 20) as remote,
            pytest.raises(RuntimeError, match="no arm named 'nope'"),
        ):
            remote.describe()
    finally:
        server.shutdown()


def test_the_leaf_has_no_operation_that_couples_rows(parts):
    """Nothing in the leaf lets one row of a batch change another's answer.

    The design rests on this twice over. It is why the server may serve one worker's
    batch while another's is in flight, and it is what would make padding sound if
    batches were ever merged -- cuda's answer depends on the batch *length*, so a fixed
    length is the fix, and a fixed length is only safe while the extra rows are inert.

    A `BatchNorm` added later would break both silently: the code would run, the numbers
    would be wrong, and no test but this one would notice.
    """
    _regulation, _encoder, net = parts
    coupling = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d,
    )
    found = [
        f"{name or '<root>'}: {type(module).__name__}"
        for name, module in net.named_modules()
        if isinstance(module, coupling)
    ]
    assert not found, (
        f"these couple rows within a batch: {found}. The inference server serves batches "
        f"as they arrive, so a row's answer must not depend on what it was batched with."
    )


@pytest.mark.parametrize("device_name", DEVICES)
def test_a_rows_answer_does_not_depend_on_what_it_was_batched_with(parts, device_name):
    """The property above, measured rather than inferred from the module list.

    Same rows, same count, different neighbours and different order: the server is
    allowed to change none of it. Held to `array_equal` rather than a tolerance, because
    a difference of 1.9e-06 moves an equilibrium and this is the floor every measurement
    on the board stands on.
    """
    regulation, encoder, net = parts
    device = torch.device(device_name)

    from pokeuraou.inference import served_model

    positions = _positions(regulation, 24)
    server, address = serve({"value": served_model(BatchedValue(net.to(device), encoder, device=device))})
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=8 << 20) as remote:
            straight = remote(positions)
            # The same 24 positions, reversed. Row i of the answer moves with its row.
            reversed_answer = remote(positions[::-1])
    finally:
        server.shutdown()

    assert np.array_equal(straight, reversed_answer[::-1]), (
        f"max difference {np.abs(straight - reversed_answer[::-1]).max():.3e} -- a row's "
        f"answer moved because its neighbours changed"
    )


@pytest.mark.parametrize("device_name", DEVICES)
def test_an_ensemble_arm_matches_the_ensemble_a_worker_would_have_built(parts, device_name):
    """The path that was not covered, and the one that was wrong.

    Every other test here uses a single net, and a single net has only one way to be
    evaluated. An ensemble has two: `stack_module_state` with `vmap`, which is what
    `BatchedValue` does, and `torch.stack([net(batch) for net in nets]).mean(0)`, which is
    what the server did. They sum a float32 in different orders. `value.py` refuses to keep
    both paths and says why in a comment; the server had restored the second one, and a
    real match through it produced identical menus and identical draws with mixed
    strategies differing at 4e-08 -- a different equilibrium, so a different agent.

    So this builds the ensemble both ways and demands they agree exactly.
    """
    _regulation, encoder, net = parts
    device = torch.device(device_name)
    torch.manual_seed(11)
    from pokeuraou.value import ValueConfig, build

    second = build(encoder, ValueConfig()).eval()
    nets = [net.to(device), second.to(device)]

    from pokeuraou.inference import served_model

    local = BatchedValue(nets, encoder, device=device)
    positions = _positions(_regulation, 24)
    expected = local(positions)

    server, address = serve({"value": served_model(BatchedValue(nets, encoder, device=device))})
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=8 << 20) as remote:
            got = remote(positions)
    finally:
        server.shutdown()

    assert np.array_equal(got, expected), (
        f"max difference {np.abs(got - expected).max():.3e} -- an ensemble served is not "
        f"the ensemble a worker builds, and 1.9e-06 moves an equilibrium"
    )


@pytest.mark.parametrize("device_name", DEVICES)
def test_an_ensemble_arm_survives_several_workers_at_once(parts, device_name):
    """Concurrency and ensembles, which were each covered and never together.

    The two-worker test above uses a single net, so it never reaches
    `torch.func.functional_call`; the ensemble test above uses one caller, so it never has
    two threads in it at once. The intersection is what production is: ten workers against
    an arm of two members. It failed five seconds in with `Tensor on device meta is not on
    the expected device cuda:0`, because `functional_call` swaps a module's parameters in
    place and two threads doing that to the same module race.

    Four callers rather than two, and several rounds each, because a race that needs an
    unlucky interleaving will not show up in one.
    """
    regulation, encoder, net = parts
    device = torch.device(device_name)
    torch.manual_seed(23)
    from pokeuraou.value import ValueConfig, build

    nets = [net.to(device), build(encoder, ValueConfig()).eval().to(device)]

    from pokeuraou.inference import served_model

    local = BatchedValue(nets, encoder, device=device)
    sizes = [8, 20, 33, 41]
    batches = {n: _positions(regulation, n) for n in sizes}
    alone = {n: local(batches[n]) for n in sizes}

    server, address = serve({"value": served_model(BatchedValue(nets, encoder, device=device))})
    results: dict[int, np.ndarray] = {}
    failures: list[BaseException] = []
    try:

        def ask(rows: int) -> None:
            try:
                with RemoteValue(address, "value", encoder, buffer_bytes=8 << 20) as remote:
                    for _ in range(8):
                        results[rows] = remote(batches[rows])
            except BaseException as error:  # noqa: BLE001 -- reported on the main thread
                failures.append(error)

        threads = [threading.Thread(target=ask, args=(n,)) for n in sizes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        server.shutdown()

    assert not failures, f"a caller raised: {failures[0]!r}"
    for rows, expected in alone.items():
        assert np.array_equal(results[rows], expected), (
            f"{rows} rows changed when three other workers were asking at the same time"
        )


@pytest.mark.parametrize("device_name", DEVICES)
def test_a_batch_longer_than_one_chunk_is_cut_where_batchedvalue_cuts(parts, device_name):
    """More rows than fit in one call, answered identically to the direct path.

    `BatchedValue` cuts at `batch_size` and the server's client now cuts at the same
    number, so the two make the same calls with the same row counts. That is the only
    reason the answers can be equal on CUDA, where the count is part of the arithmetic --
    and it is the property that lets a 1,048,576-leaf node go through at all.

    A small `batch_size` here so the cut happens over a few dozen positions instead of a
    few thousand; the boundary being small changes nothing about whether the two agree.
    """
    regulation, encoder, net = parts
    device = torch.device(device_name)

    from pokeuraou.inference import served_model

    local = BatchedValue(net.to(device), encoder, device=device, batch_size=7)
    positions = _positions(regulation, 45)
    expected = local(positions)

    server, address = serve({"value": served_model(local)})
    try:
        with RemoteValue(
            address, "value", encoder, buffer_bytes=8 << 20, batch_size=7
        ) as remote:
            got = remote(positions)
    finally:
        server.shutdown()

    assert len(got) == 45
    assert np.array_equal(got, expected), (
        f"max difference {np.abs(got - expected).max():.3e} -- the server cut a long batch "
        f"somewhere BatchedValue does not"
    )
