from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pokeuraou.oracle import ORACLE_JS, Oracle, TeamSet, load_team
from pokeuraou.regulation import Regulation, load_regulation, regulation_dir

FORMAT_ID = "gen9championsvgc2026regmc"
FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "oracle: needs the built TypeScript Showdown oracle")
    config.addinivalue_line("markers", "slow: long-running (exhaustive sweeps, benchmarks)")


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
