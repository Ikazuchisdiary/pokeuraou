"""IKA-458 stage 3.7: `tools/agent_drift.py` holds the Rust workers' configs to one agent.

Generation's games and the board's are the Rust workers', and their agents are the configs
`rustworker.build_config` and `rustworker.board_config` write. `pokeuraou.workeragent` builds both
from generation's flags on the stand-in inference server and compares every key of both, by value,
read per board arm the way the worker fills its node; only the keys `ALLOWED_DIFFERENT` names,
each with its reason, are left out. What these tests hold:

- the tree passes, and `--check` fails on a drift;
- each way of building the board's agent apart from generation's fails the comparison: a copy of
  `rustworker.py` with one line changed is swapped in for the run (the controls). Among them are
  the edits that passed the earlier `ast` reading of the source (`node |= ...`, `del node[...]`,
  an alias of the node, a write through `arm_configs[-1]`, `dict.update(node, ...)`, a spread of
  the budget, a halved turn cap, sliced notes), and a key the board's config or an arm gains;
- a key missing from both configs, missing arms and a board that explores its selection are
  reported;
- the Rust workers' node writes outside the keys they fill are marked (`rust_node_writes`, a
  marker that reads the source text, not a proof).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pokeuraou.workeragent import (
    ALLOWED_DIFFERENT,
    ROOT,
    RUST_NODE_FILLS,
    agent_differences,
    build_both,
    rust_node_writes,
    serving,
    write_pool,
)

from ._harness import load_tool

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
     "arm 0 differs from generation at .exactBudget"),
    ('"maxTurns": max_turns(args),', '"maxTurns": max_turns(args) // 2,',
     "arm 0 differs from generation at .maxTurns"),
    ('"notes": selection_notes(),', '"notes": selection_notes()[:1],',
     "arm 0 differs from generation at .notes"),
    (BOARD_NODE, BOARD_NODE + '        node |= {"budget": {}}\n', "at .node.budget"),
    (BOARD_NODE, BOARD_NODE + '        del node["budget"]\n', "at .node.budget (only in generation)"),
    (BOARD_NODE, BOARD_NODE + '        n2 = node\n        n2["budget"] = {}\n', "at .node.budget"),
    (BOARD_NODE, BOARD_NODE + '        dict.update(node, budget={})\n', "at .node.budget"),
    (ARM_APPENDED, ARM_APPENDED + '        arm_configs[-1]["node"]["budget"] = {}\n',
     "arm 1 differs from generation at .node.budget"),
    ('"limit": int(arm.limit),', '"limit": int(arm.limit) + 1,', "at .node.widths"),
    ('"q": next(i for i, seen in enumerate(qs) if seen is model),', '"q": 5,', "at .node.qServer"),
    # A key the board's config gains is compared by default (it is in no list).
    ('        "mode": "board",\n', '        "mode": "board",\n        "patience": 3,\n',
     "at .patience (only in the board)"),
    # A key an arm gains is named until `board_view` reads it.
    ('            "binary": None if arm.rust_binary', '            "beam": 2,\n'
     '            "binary": None if arm.rust_binary', "at .unread arm keys"),
])
def test_a_board_agent_built_apart_fails(stand_in, tmp_path, old, new, said) -> None:  # noqa: ANN001
    problems = _differences(stand_in, tmp_path, _changed(tmp_path, old, new))
    assert any(said in p for p in problems), problems


def test_generation_s_agent_is_held_too(stand_in, tmp_path) -> None:  # noqa: ANN001
    source = _changed(tmp_path, '"notes": selection_notes(),', '"notes": [],', first=True)
    problems = _differences(stand_in, tmp_path, source)
    assert any("at .notes" in p for p in problems), problems


def test_missing_keys_and_arms_are_differences(stand_in, tmp_path) -> None:  # noqa: ANN001
    """A key gone from both configs (None == None), an empty or missing `arms`, and a board that
    explores its selection are reported, not passed."""
    address, pool_path = stand_in
    generation, board = build_both(address, pool_path, tmp_path)
    assert agent_differences(generation, board) == []
    gone = agent_differences({k: v for k, v in generation.items() if k != "maxTurns"},
                             {k: v for k, v in board.items() if k != "maxTurns"})
    assert "generation's config has no maxTurns" in gone and "the board's config has no maxTurns" in gone
    assert "the board's config has 0 arms, not 2" in agent_differences(generation, {**board, "arms": []})
    no_arms = {k: v for k, v in board.items() if k != "arms"}
    assert "the board's config has 0 arms, not 2" in agent_differences(generation, no_arms)
    exploring = agent_differences(generation, {**board, "epsilon": 0.25})
    assert any("epsilon is 0.25" in p for p in exploring), exploring


def test_the_servers_the_game_reads_are_compared(stand_in, tmp_path) -> None:  # noqa: ANN001
    """The leaf servers `playGame`'s context hands the replacement and self-switch nodes, and the
    buffers of `value` and `q`: a change in any of them is a difference (the review's controls)."""
    import copy

    address, pool_path = stand_in
    generation, board = build_both(address, pool_path, tmp_path)
    assert agent_differences(generation, board) == []

    def changed(config: dict, edit) -> dict:  # noqa: ANN001
        config = copy.deepcopy(config)
        edit(config)
        return config

    cases = (
        (generation, changed(board, lambda b: b["servers"][0].update(netScoresEnds=True)),
         "context.servers[leaf].netScoresEnds"),
        (generation, changed(board, lambda b: b["servers"][0].update(encoding={"megaFromSlots": True})),
         "context.servers[leaf].encoding"),
        (changed(generation, lambda g: g["context"]["servers"][0].update(batch=1)), board,
         "context.servers[leaf].batch"),
        (changed(generation, lambda g: g["value"].update(bytes=1024)), board, "value.bytes"),
        (changed(generation, lambda g: g["q"].update(model="other-q")), board, "q.model"),
        # The review of 996e827: a new key in generation's context or q, and a context server
        # its leaf does not name.
        (changed(generation, lambda g: g["context"].update(patience=3)), board,
         ".context.patience (only in generation)"),
        (changed(generation, lambda g: g["q"].update(batch=2)), board, "q.batch (only in generation)"),
        (changed(generation, lambda g: g["context"]["servers"].append({"model": "spare"})), board,
         "context.servers (not the leaf)"),
    )
    for mine, theirs, said in cases:
        problems = agent_differences(mine, theirs)
        assert any(said in p for p in problems), (said, problems)


def test_every_key_but_the_named_ones_is_compared() -> None:
    """`ALLOWED_DIFFERENT` is what is left out, each with a reason; nothing else is."""
    assert all(isinstance(why, str) and why for why in ALLOWED_DIFFERENT.values())
    generation = {"node": {}, "exactBudget": {}, "maxTurns": 40, "notes": [], "seed": 1,
                  "regulation": "r", "pool": {}, "epsilon": 0.25, "record": [1], "mystery": 1}
    board = {"exactBudget": {}, "maxTurns": 40, "notes": [], "seed": 1, "regulation": "r",
             "pool": {}, "epsilon": 0.0, "records": [2], "servers": [], "qServers": [],
             "arms": [{"node": {}, "limit": None}, {"node": {}, "limit": None}]}
    problems = agent_differences(generation, board)
    assert any(".mystery (only in generation)" in p for p in problems), problems
    assert not any("record" in p or "epsilon" in p for p in problems), problems


def test_a_rust_node_write_outside_the_fills_is_marked() -> None:
    """A marker, not a proof (records/IKA-458.md section 19.2): the forms it reads, and one it
    cannot read is still marked."""
    text = (ROOT / "rust" / "src" / "boardworker.rs").read_bytes().decode("utf-8")
    line = 'node["widths"] = widths.clone();'
    assert line in text
    for extra, said in (
        ('node["budget"] = json!({});', "node['budget']"),
        ('node["budget"]["maxRows"] = json!(1);', "node['budget']"),
        ('node [ "budget" ] = json!({});', "node['budget']"),
        ("node[key] = widths.clone();", "cannot read"),
        ('nodes[0]["budget"] = json!({});', "cannot read"),
        ('node.get_mut("budget");', "cannot read"),
        ('node.pointer_mut("/budget");', "cannot read"),
    ):
        changed = text.replace(line, f"{line}\n            {extra}")
        assert any(said in p for p in rust_node_writes({"boardworker.rs": changed})), extra
    assert {"server", "qServer", "widths"} <= RUST_NODE_FILLS


def test_the_check_fails_on_a_worker_drift(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(drift, "worker_drift", lambda *args: ["arm 0 differs"])
    assert drift.main(["--check"]) == 1


def test_the_board_s_default_width_is_generation_s() -> None:
    pool_match = load_tool("pool_match")
    defaults = drift.flag_defaults(drift.ROOT / "tools" / "generate_queue.py")
    assert defaults["--limit"] == str(pool_match.GENERATION_LIMIT) == "12"
