"""IKA-386 (`portserved`): a ladder read whose sub-games the port fills, has scored by the
inference server itself and solves, in one crossing a call, is the read that fills them in
the port, has Python ask the server (`port.score_stacked` -> `RemoteValue.from_encoded`) and
solves them in the port (`portlp`, IKA-381) -- to the bit.

The server here is `inference.serve` with a numpy stand-in that records every request it
answers (its rows, and each array's name, dtype, shape and bytes) and gives each row a value
that moves with the request's row count, as a CUDA answer does (IKA-291): so the same values
need the same rows in the same requests, and the two roads' requests are compared too.
Played on `rizabanadohido`, a few positions of one game, open roots and Bayesian roots, with
the gathering and the chunk cut small as well as at their defaults. No torch.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pokeuraou import ladder, portlp, portserved, search
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.inference import ARRAYS, RemoteValue, serve
from pokeuraou.setup import parse_scenario
from pokeuraou.teams import load_roster

from .test_ladder import _played
from .test_ladder_pool import _node, _same


class Stub:
    """A leaf in the server: each request's rows and bytes kept, and a value for each row
    that depends on the row and on how many rows came with it."""

    def __init__(self) -> None:
        self.log: list[tuple[int, str]] = []

    def __call__(self, arrays: dict[str, np.ndarray], rows: int) -> np.ndarray:
        digest = hashlib.sha256()
        for name in ARRAYS:
            array = arrays[name]
            digest.update(f"{name}{array.dtype.str}{array.shape}".encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        self.log.append((rows, digest.hexdigest()))
        mon = arrays["mon"].reshape(rows, -1).astype(np.float64)
        side = arrays["side"].reshape(rows, -1).astype(np.float64)
        species = arrays["species"].reshape(rows, -1).astype(np.float64)
        x = (mon.sum(axis=1) * 0.013 + side.sum(axis=1) * 0.07
             + (species * np.arange(1, species.shape[1] + 1)).sum(axis=1) * 1e-4)
        return 1.0 / (1.0 + np.exp(-np.tanh(x))) + rows * 1e-12


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture(scope="module")
def served(roster):  # noqa: ANN001, ANN201
    stub = Stub()
    server, address = serve({"stub": stub})
    yield stub, address
    server.shutdown()


def _read(reg, node, stages, leaf, *, side=0):  # noqa: ANN001, ANN202
    ours, theirs, items, matrices, weights, start = node
    saved = ladder._POOL
    ladder._POOL = None
    try:
        return ladder.read(reg, side, ours, theirs, items, matrices, weights, start, leaf,
                           budget=Budget.matrix(), stages=ladder.parse_ladder(stages), budget_ms=None)
    finally:
        ladder._POOL = saved


@pytest.mark.parametrize("small", [False, True])
def test_the_port_asks_the_server_as_python_did(roster, served, monkeypatch, small) -> None:  # noqa: ANN001
    """Every rung, the counted work, the notes and the server's requests (rows and bytes)
    of the `portlp` read, with the port asking the server. ``small``: a gathering of 64
    rows and requests of at most 100, so a call is scored in several batches and a batch in
    several requests. The positive control: the server's requests the ports sent."""
    stub, address = served
    reg = roster.reg
    if small:
        monkeypatch.setattr(search, "GATHER_ROWS", 64)
    nodes = [(_node(reg, pos, 5), "d2r2b3n3+d2r3ban4x+d3r2ban4/r2ban4", 0)
             for pos in _played(roster)[:2]]
    sheet = list(roster.sets)[:6]
    for pos in _played(roster)[:3]:
        spreads = {s: completions(reg, pos, s, sheet, seen=frozenset({0, 1})) for s in (0, 1)}
        if len(spreads[1]) >= 2:
            nodes.append((_node(reg, pos, 5, spreads, 1), "d2r2b3n4+d2r3ban4x", 1))
            break
    assert len(nodes) == 3
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=16 << 20,
                     batch_size=100 if small else 8192) as leaf:
        asked, trips, _ended = _compare(reg, nodes, leaf, stub)
    assert asked > 0 and trips > asked, (asked, trips)


def test_an_ended_leaf_is_its_result_on_both_roads(served) -> None:  # noqa: ANN001
    """`encode.settle` on the port's road: IKA-253's end (Garchomp at 1 HP against Kingambit
    at 1 HP), whose depth-2 sub-games are full of ended leaves, reads as on the Python road.
    The positive control: ended leaves the leaf counted on the port's road."""
    stub, address = served
    reg, pos = _chomp_gambit()
    nodes = [(_node(reg, pos, 4), "d2r2b3n3+d2r3ban4x", 0)]
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=16 << 20) as leaf:
        asked, _trips, ended = _compare(reg, nodes, leaf, stub)
    assert asked > 0 and ended > 0, (asked, ended)


def _compare(reg, nodes, leaf, stub):  # noqa: ANN001, ANN202
    """Each node read on the `portlp` road and on the port's; the same reads and the same
    requests. The port's server requests, all the server trips, and the ended leaves the
    leaf counted on the port's road."""
    asked = trips = ended = 0
    try:
        for node, how, side in nodes:
            portserved.ON[0] = False
            portlp.set_on(True)
            del stub.log[:]
            python = _read(reg, node, how, leaf, side=side)
            python_log = list(stub.log)
            assert python.rungs and python.here["portServed"] == 0
            portserved.set_on(True)
            del stub.log[:]
            before = leaf.ended
            ported = _read(reg, node, how, leaf, side=side)
            _same(ported, python)
            assert stub.log == python_log
            assert ported.here["serverTrips"] == python.here["serverTrips"]
            asked += ported.here["portServed"]
            trips += ported.here["serverTrips"]
            ended += leaf.ended - before
    finally:
        portserved.ON[0] = False
        portlp.set_on(False)
    return asked, trips, ended


def _filler(species: str, ability: str, nature: str, moves: list[str]) -> dict:
    return {"species": species, "ability": ability, "nature": nature, "moves": moves,
            "sp": {"hp": 1}, "hp": 0}


def _chomp_gambit():  # noqa: ANN202
    """IKA-253's end (`tests/test_decided_leaf.py`, which needs torch): both at 1 HP."""
    chomp = {"species": "Garchomp", "ability": "roughskin", "nature": "Jolly",
             "moves": ["Swords Dance", "Earthquake"], "sp": {"atk": 32, "spe": 32},
             "active": True, "hp": 1}
    gambit = {"species": "Kingambit", "ability": "defiant", "nature": "Adamant",
              "moves": ["Kowtow Cleave", "Sucker Punch"], "sp": {"hp": 32, "atk": 32},
              "active": True, "hp": 1}
    sides = []
    for lead in (chomp, gambit):
        dead = _filler("Incineroar", "intimidate", "Impish", ["Fake Out"])
        dead["active"] = True
        sides.append([lead, dead, _filler("Sinistcha", "hospitality", "Bold", ["Protect"]),
                      _filler("Farigiraf", "armortail", "Calm", ["Protect"])])
    data = {"regulation": "gen9championsvgc2026regmc", "turn": 1,
            "sides": [{"id": "p1", "team": sides[0]},
                      {"id": "p2", "spreadsKnown": True, "team": sides[1]}],
            "field": {}}
    sc = parse_scenario(data)
    register_mega_stones(sc.reg)
    pos = sc.position
    pp = {"swordsdance": 20, "earthquake": 12, "kowtowcleave": 12, "suckerpunch": 8}
    for side in pos.sides:
        for m in side.pokemon[0].moves:
            m.pp = m.maxpp = pp[m.id]
    return sc.reg, pos
