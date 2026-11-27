"""IKA-458 stage 3.7: `tools/agent_drift.py` holds the Rust workers' configs to one agent.

Generation's games and the board's are the Rust workers', and their agents are the configs
`rustworker.build_config` and `rustworker.board_config` write. `tests/_worker_agent.py` builds both
from generation's flags on the stand-in inference server and compares, by value, the keys that
decide which agent plays (the move node, the exact budget, the turn cap, the selection's notes,
each arm's width, leaf, store tag and Q). What these tests hold:

- the tree passes, and `--check` fails on a drift;
- each way of building the board's agent apart from generation's fails the comparison: a copy of
  `rustworker.py` with one line changed is swapped in for the run (the controls). Among them are
  the edits that passed the earlier `ast` reading of the source (`node |= ...`, `del node[...]`,
  an alias of the node, a write through `arm_configs[-1]`, `dict.update(node, ...)`, a spread of
  the budget, a halved turn cap, sliced notes);
- the Rust workers write only the node keys they fill per arm (`RUST_NODE_FILLS`), read from
  worker.rs and boardworker.rs, and another key written there fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._harness import load_tool
from ._worker_agent import (
    ROOT,
    RUST_NODE_FILLS,
    agent_differences,
    build_both,
    rust_node_writes,
    serving,
    write_pool,
)

drift = load_tool("agent_drift")
SOURCE = ROOT / "src" / "pokeuraou" / "rustworker.py"
BOARD_NODE = "        node = agent_node(model, arm.limit, leaf)\n"
ARM_APPENDED = '            "node": node,\n        })\n'


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):  # noqa: ANN001, ANN201
    pool_path = write_pool(tmp_path_factory.mktemp("pool"))
    with serving(pool_path) as address:
        yield address, pool_path


def _changed(tmp_path: Path, old: str, new: str, *, first: bool = False) -> Path:
    """A copy of rustworker.py with one occurrence of ``old`` replaced: the last (the board's)
    unless ``first`` (generation's)."""
    text = SOURCE.read_bytes().decode("utf-8")
    assert old in text, old
    at = text.find(old) if first else text.rfind(old)
    path = tmp_path / "rustworker.py"
    path.write_bytes((text[:at] + new + text[at + len(old):]).encode("utf-8"))
    return path


def _differences(stand_in, tmp_path: Path, source: Path | None = None) -> list[str]:  # noqa: ANN001
    address, pool_path = stand_in
    generation, board = build_both(address, pool_path, tmp_path, source)
    return agent_differences(generation, board)


def test_the_tree_builds_one_agent(stand_in, tmp_path) -> None:  # noqa: ANN001
    assert _differences(stand_in, tmp_path) == []
    assert rust_node_writes() == []
    assert drift.main(["--check"]) == 0


@pytest.mark.parametrize(("old", "new", "said"), [
    ('"exactBudget": exact_budget(),', '"exactBudget": {**exact_budget(), "maxRows": 1},',
     "the board's exactBudget"),
    ('"maxTurns": max_turns(args),', '"maxTurns": max_turns(args) // 2,', "the board's maxTurns"),
    ('"notes": selection_notes(),', '"notes": selection_notes()[:1],', "the board's notes"),
    (BOARD_NODE, BOARD_NODE + '        node |= {"budget": {}}\n', "arm 0's node"),
    (BOARD_NODE, BOARD_NODE + '        del node["budget"]\n', "arm 0's node"),
    (BOARD_NODE, BOARD_NODE + '        n2 = node\n        n2["budget"] = {}\n', "arm 0's node"),
    (BOARD_NODE, BOARD_NODE + '        dict.update(node, budget={})\n', "arm 0's node"),
    (ARM_APPENDED, ARM_APPENDED + '        arm_configs[-1]["node"]["budget"] = {}\n', "arm 0's node"),
    ('"limit": int(arm.limit),', '"limit": int(arm.limit) + 1,', "arm 0's width"),
    ('"q": next(i for i, seen in enumerate(qs) if seen is model),', '"q": 5,', "arm 0's Q"),
])
def test_a_board_agent_built_apart_fails(stand_in, tmp_path, old, new, said) -> None:  # noqa: ANN001
    problems = _differences(stand_in, tmp_path, _changed(tmp_path, old, new))
    assert any(said in p for p in problems), problems


def test_generation_s_agent_is_held_too(stand_in, tmp_path) -> None:  # noqa: ANN001
    source = _changed(tmp_path, '"notes": selection_notes(),', '"notes": [],', first=True)
    problems = _differences(stand_in, tmp_path, source)
    assert any("the board's notes" in p for p in problems), problems


def test_a_rust_node_write_outside_the_fills_fails() -> None:
    text = (ROOT / "rust" / "src" / "boardworker.rs").read_bytes().decode("utf-8")
    assert 'node["widths"] = widths.clone();' in text
    changed = text.replace('node["widths"] = widths.clone();',
                           'node["widths"] = widths.clone();\n            node["budget"] = json!({});')
    assert any("node['budget']" in p for p in rust_node_writes({"boardworker.rs": changed}))
    keyed = text.replace('node["widths"] = widths.clone();', 'node[key] = widths.clone();')
    assert any("cannot read" in p for p in rust_node_writes({"boardworker.rs": keyed}))
    assert {"server", "qServer", "widths"} <= RUST_NODE_FILLS


def test_the_check_fails_on_a_worker_drift(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(drift, "worker_drift", lambda *args: ["arm 0's node differs"])
    assert drift.main(["--check"]) == 1


def test_the_board_s_default_width_is_generation_s() -> None:
    pool_match = load_tool("pool_match")
    defaults = drift.flag_defaults(drift.ROOT / "tools" / "generate_queue.py")
    assert defaults["--limit"] == str(pool_match.GENERATION_LIMIT) == "12"
