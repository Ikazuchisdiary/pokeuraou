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
def _port_roads_restored() -> Iterator[None]:
    """`portlp`'s roads are process-wide, and generation's entry points turn them on for good
    (`portlp.default_on_for_generation`, IKA-455).

    test_pool_match runs `pool_match.main` in-process, which left every road on for the rest
    of its worker: a later test_q_rank counted the port's solves as shared ones, and
    test_belief_fold_port and test_port_faults compared the port against a "Python" node the
    port had solved too. Four tests failed on CI at 131d811 for that alone.
    """
    from pokeuraou import portlp

    was = (portlp.ON[0], portlp.GEN[0], portlp.FOLD[0], portlp.STACK[0], portlp.NODE[0],
           rustnode._SPAN_BLOCKS[0])  # noqa: SLF001
    yield
    (portlp.ON[0], portlp.GEN[0], portlp.FOLD[0], portlp.STACK[0], portlp.NODE[0],
     rustnode._SPAN_BLOCKS[0]) = was  # noqa: SLF001


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
