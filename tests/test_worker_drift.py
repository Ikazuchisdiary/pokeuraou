"""IKA-458 stage 3.7: `tools/agent_drift.py` holds the Rust workers' configs to one agent.

Generation's games and the board's are the Rust workers', and their agents are the configs
`rustworker.build_config` and `rustworker.board_config` write. `worker_drift` reads the two with
`ast` and holds every agent key (`node`, `exactBudget`, `maxTurns`, `notes`) to its one function.
What these tests hold: the tree passes, and each way of building the board's agent apart from
generation's -- another budget, a turn cap of its own, a key written into an arm's node, a key
dropped beyond the ones the worker fills, a narrower default width -- fails the check (the
controls, each on a copy of `rustworker.py` with one line changed). The runtime side (an arm of
generation's leaf and Q gets generation's node) is `test_rust_board_worker.py`'s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._harness import load_tool

drift = load_tool("agent_drift")
SOURCE = drift.ROOT / "src" / "pokeuraou" / "rustworker.py"


def _changed(tmp_path: Path, old: str, new: str, *, which: int = -1) -> Path:
    """A copy of rustworker.py with one occurrence of ``old`` (the last: the board's) replaced."""
    text = SOURCE.read_bytes().decode("utf-8")
    assert old in text, old
    at = text.rfind(old) if which == -1 else text.find(old)
    path = tmp_path / "rustworker.py"
    path.write_bytes((text[:at] + new + text[at + len(old):]).encode("utf-8"))
    return path


def test_the_tree_builds_one_agent() -> None:
    assert drift.worker_drift() == []
    assert drift.main(["--check"]) == 0


@pytest.mark.parametrize(("old", "new", "said"), [
    ('"exactBudget": exact_budget(),', '"exactBudget": dump_budget(Budget.fast()),',
     "board_config's exactBudget comes from"),
    ('"maxTurns": max_turns(args),', '"maxTurns": 200,', "board_config's maxTurns comes from 200"),
    ("        node = agent_node(model, arm.limit, leaf)\n",
     "        node = agent_node(model, arm.limit, leaf)\n        node[\"budget\"] = {}\n",
     "board_config writes into an agent's node"),
    ("for key in BOARD_FILLS:", 'for key in (*BOARD_FILLS, "budget"):', "not BOARD_FILLS"),
    ("        node = agent_node(model, arm.limit, leaf)\n",
     "        node = static_node(static_request(model, (arm.limit, arm.limit), \"qServer\"), leaf,"
     " leaf.encoder, rules_of(leaf), Budget.fast())\n",
     "board_config calls static_node"),
])
def test_a_board_agent_built_apart_fails(tmp_path, old, new, said) -> None:  # noqa: ANN001
    problems = drift.worker_drift(_changed(tmp_path, old, new))
    assert any(said in p for p in problems), problems


def test_generation_s_agent_is_held_too(tmp_path) -> None:  # noqa: ANN001
    problems = drift.worker_drift(
        _changed(tmp_path, '"notes": selection_notes(),', '"notes": [],', which=0)
    )
    assert any("build_config's notes comes from []" in p for p in problems), problems


def test_the_check_fails_on_a_worker_drift(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(drift, "worker_drift", lambda path=None: ["board_config sets no node"])
    assert drift.main(["--check"]) == 1


def test_the_board_s_default_width_is_generation_s() -> None:
    pool_match = load_tool("pool_match")
    defaults = drift.flag_defaults(drift.ROOT / "tools" / "generate_queue.py")
    assert defaults["--limit"] == str(pool_match.GENERATION_LIMIT) == "12"
