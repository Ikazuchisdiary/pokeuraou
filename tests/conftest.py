from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pokeuraou import rustnode
from pokeuraou.oracle import ORACLE_JS, Oracle, TeamSet, load_team
from pokeuraou.regulation import Regulation, load_regulation, regulation_dir

FORMAT_ID = "gen9championsvgc2026regmc"
FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "oracle: needs the built TypeScript Showdown oracle")
    config.addinivalue_line("markers", "slow: long-running (exhaustive sweeps, benchmarks)")
    config.addinivalue_line(
        "markers", "unserved: HP_SHARE is not served as a leaf in this test (IKA-460)"
    )


@pytest.fixture(autouse=True)
def _positions_not_held() -> Iterator[None]:
    """`rustnode.hold_positions` is process-wide and `tools/selfplay.py` turns it on for good.

    A test that runs `run_pool` in-process (test_poolplay, test_rank_scores) left it on, and
    a later test in the same worker that edits a position after sending it had the old text
    sent: five test_rust_node tests failed on the first CI run, and fail locally when run
    after test_rank_scores in one process (IKA-51).
    """
    yield
    rustnode.hold_positions(False)


@pytest.fixture(autouse=True)
def _served_leaves_closed() -> Iterator[None]:
    """The stand-in inference servers and leaves a test made (`tests/_served.py`) are closed
    when it ends."""
    yield
    from ._served import close_all

    close_all()


_REFUSE_UNSERVED = [False]


@pytest.fixture(scope="session", autouse=True)
def _hp_share_is_served() -> Iterator[None]:
    """`payoff.HP_SHARE` (and a left-out leaf, which is HP_SHARE) is a served leaf here.

    The port scores a node on the inference server, so a leaf held in the process is
    `beliefnode.UnservedLeaf` (IKA-460). Many tests take HP_SHARE as "a leaf"; this gives them
    the stand-in for the Rust server's hp-share arm (`tests/_served.hp_share_leaf`, the same
    formula read off the encoding). A test that wants the refusal marks itself
    ``@pytest.mark.unserved``. Session-wide, because module-scoped fixtures play games too."""
    from pokeuraou import beliefnode
    from pokeuraou.payoff import HP_SHARE

    from ._served import hp_share_leaf, served_from

    real = beliefnode.served_leaf

    def served_leaf(evaluate: object, what: str = "the hidden bench's node") -> object:
        owner = getattr(evaluate, "__self__", evaluate)
        if _REFUSE_UNSERVED[0]:
            return real(evaluate, what)
        if evaluate is None or owner is HP_SHARE:
            return hp_share_leaf()
        if hasattr(owner, "from_encoded") and hasattr(owner, "encoder"):
            # A stand-in that scores `Encoded` (a test's own leaf, or a net held here) is
            # served from this process, its own `from_encoded` answering the port's requests.
            return served_from(owner)
        return real(evaluate, what)

    beliefnode.served_leaf = served_leaf
    yield
    beliefnode.served_leaf = real


@pytest.fixture(autouse=True)
def _unserved_marker(request: pytest.FixtureRequest) -> Iterator[None]:
    _REFUSE_UNSERVED[0] = request.node.get_closest_marker("unserved") is not None
    yield
    _REFUSE_UNSERVED[0] = False


@pytest.fixture(autouse=True)
def _port_roads_restored() -> Iterator[None]:
    """`portlp`'s ladder switches (`portlp.ON`) are process-wide: restored after each test."""
    from pokeuraou import portlp

    roads = [
        value for name, value in vars(portlp).items()
        if name.isupper() and isinstance(value, list) and len(value) == 1
        and isinstance(value[0], bool)
    ]
    was = [road[0] for road in roads], rustnode._SPAN_BLOCKS[0]  # noqa: SLF001
    yield
    for road, flag in zip(roads, was[0], strict=True):
        road[0] = flag
    rustnode._SPAN_BLOCKS[0] = was[1]  # noqa: SLF001


@pytest.fixture(scope="session")
def reg() -> Regulation:
    if not (regulation_dir() / f"{FORMAT_ID}.json").exists():
        pytest.skip(
            "regulation config missing; run "
            "`node packages/sim-bridge/dist/cli/dump-regulation.js`"
        )
    return load_regulation(FORMAT_ID)


@pytest.fixture(scope="session")
def oracle() -> Iterator[Oracle]:
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built; run `npm run -w @pokeuraou/sim-bridge build`")
    with Oracle() as o:
        yield o


@pytest.fixture(scope="session")
def team_a() -> list[TeamSet]:
    return load_team(FIXTURES / "team_a.json")


@pytest.fixture(scope="session")
def team_b() -> list[TeamSet]:
    return load_team(FIXTURES / "team_b.json")


@pytest.fixture()
def port(reg: Regulation) -> Iterator[object]:
    """A warm port process for an oracle test's port twin (IKA-207; `_port_showdown`).

    Fails when there is no release binary (IKA-210): the port is the only engine the rule
    tests have, so a machine without it has not run them, and a skip would say it had.
    """
    from pokeuraou import rustnode

    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    node = rustnode.RustNode(reg)
    yield node
    node.close()
