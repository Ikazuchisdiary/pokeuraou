"""Two invariants about cost, asserted as shape rather than measured as bytes.

The search side's question, and it is a good one: the test written when "do not split" was
fixed pins the *answer* -- 45 rows at `batch_size=7` agree with the local path -- and the
bug that followed was in the *cost*. `RemoteValue.__call__` encoded 1,048,576 positions in
one go, which is 3.9 GB, and 45 positions is far too cheap for that to show. Then the
served worker turned out to be importing torch after all, 816 MB that the option's own help
said it did not, and no answer was wrong there either.

Measuring cost in a test is brittle: peak resident memory depends on what else the machine
is doing, so the assertion either has slack enough to miss a regression or is tight enough
to fail on a busy afternoon. But both of these bugs have a *shape*, and the shape is exact:

    nothing encodes more rows at once than it sends
    a worker that scores remotely does not import torch

Neither reads a clock or a byte count, and both fail deterministically on the code that was
actually wrong.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.inference import RemoteValue, serve, served_model  # noqa: E402
from pokeuraou.value import BatchedValue, ValueConfig, build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _positions(regulation, count: int):
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import all_selections, load_roster

    roster = load_roster("rizabanadohido")
    picks = list(all_selections(regulation.meta.team_size, regulation.meta.picked_team_size))
    return [
        position_from_sets(
            regulation,
            [roster.sets[i] for i in picks[k % len(picks)]],
            [roster.sets[i] for i in picks[(k + 3) % len(picks)]],
        )
        for k in range(count)
    ]


def test_it_never_encodes_more_rows_than_it_sends(reg, monkeypatch):
    """The million-position encode, as a shape.

    `BatchedValue.__call__` encodes a chunk at a time; `RemoteValue.__call__` encoded the
    whole list. Both produce the same answers, so no comparison of answers can tell them
    apart -- but one of them asks the machine for 3.9 GB on a self-switch node.
    """
    register_mega_stones(reg)
    encoder = Encoder(reg)
    torch.manual_seed(7)
    net = build(encoder, ValueConfig()).eval()
    device = torch.device("cpu")

    biggest = [0]
    real_encode = encoder.encode_positions

    def watched(positions):
        biggest[0] = max(biggest[0], len(positions))
        return real_encode(positions)

    monkeypatch.setattr(encoder, "encode_positions", watched)

    server, address = serve({"value": served_model(
        BatchedValue(net.to(device), encoder, device=device, batch_size=9)
    )})
    try:
        with RemoteValue(address, "value", encoder, buffer_bytes=8 << 20,
                         batch_size=9) as remote:
            got = remote(_positions(reg, 50))
    finally:
        server.shutdown()

    assert len(got) == 50
    assert biggest[0] <= 9, (
        f"encoded {biggest[0]} positions at once with batch_size 9 -- the answers would "
        f"still be right and the memory would not be"
    )


WORKER = """
import sys
sys.path.insert(0, {src!r})
from pokeuraou.encode import Encoder
from pokeuraou.inference import RemoteValue
from pokeuraou.regulation import load_regulation
from pokeuraou.damage import register_mega_stones
from pokeuraou.teams import load_roster

reg = load_roster("rizabanadohido").reg
register_mega_stones(reg)
client = RemoteValue({address!r}, "value", Encoder(reg), buffer_bytes=4 << 20)
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import all_selections
picks = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
roster = load_roster("rizabanadohido")
positions = [
    position_from_sets(
        reg, [roster.sets[i] for i in picks[0]], [roster.sets[i] for i in picks[1]]
    )
] * 6
client(positions)
client.close()
print("TORCH" if "torch" in sys.modules else "NO TORCH")
"""


def test_a_worker_that_scores_remotely_does_not_import_torch(reg):
    """The option's help said this and it was not true.

    A served worker exists to not carry torch -- 816 MB of import and a CUDA context on top
    -- and `generation_match.py` imported it at module scope regardless, so the worker was
    the same size as before and no answer changed. This asserts the claim the help makes.
    """
    register_mega_stones(reg)
    encoder = Encoder(reg)
    torch.manual_seed(7)
    net = build(encoder, ValueConfig()).eval()

    server, address = serve({"value": served_model(
        BatchedValue(net.to(torch.device("cpu")), encoder, device=torch.device("cpu"))
    )})
    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c",
             textwrap.dedent(WORKER.format(src=str(ROOT / "src"), address=address))],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=300,
        )
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip().endswith("NO TORCH"), (
        "a worker scoring on the server imported torch, which is the whole cost the "
        "server exists to remove"
    )
