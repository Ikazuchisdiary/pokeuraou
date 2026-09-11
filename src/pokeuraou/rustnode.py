"""A warm Rust process that fills a whole node's payoff matrix.

`batched_payoffs` is already this project's node-level boundary -- one position and two
lists of actions in, one matrix out -- which makes it the one place a process boundary
costs nothing: the leaves never cross, only 576 floats do.

The port refuses cells it does not model (a self-switch that suspends the turn, an ability
outside its set), and names them. Those come back in `refused` and the caller resolves them
in Python, so a partial port is usable rather than merely measurable: a cell Rust fills is
Python's own answer, and a cell it declines costs Python's own time.

Only the parameter-free objectives cross. A learned value function cannot: its leaves are
the input, and a node's leaves are tens of megabytes of JSON. That boundary needs the
encoder ported too, and until then a node scored by a value net stays in Python.

Opt-in and off by default::

    POKEURAOU_RUST_NODE=1                      # use it when it is available
    POKEURAOU_RUST_NODE_BIN=/path/to/binary    # defaults to rust/target/release/

Nothing imports this unless the variable is set, and `available()` answers without raising
so a caller can fall back silently rather than fail a run that was working.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import MoveAction, PassAction, SideAction, SwitchAction
from .position import Position
from .regulation import Regulation, repo_root
from .resolve import Budget

ENV_ENABLE = "POKEURAOU_RUST_NODE"
ENV_BINARY = "POKEURAOU_RUST_NODE_BIN"


def binary_path() -> Path:
    override = os.environ.get(ENV_BINARY)
    if override:
        return Path(override)
    name = "pokeuraou-damage.exe" if sys.platform == "win32" else "pokeuraou-damage"
    return repo_root() / "rust" / "target" / "release" / name


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "") not in ("", "0", "false", "no")


def available() -> bool:
    return enabled() and binary_path().exists()


def dump_action(action: object) -> dict[str, Any]:
    if isinstance(action, MoveAction):
        return {
            "kind": "move",
            "slot": action.slot,
            "moveIndex": action.move_index,
            "moveId": action.move_id,
            "target": action.target,
            "mega": action.mega,
        }
    if isinstance(action, SwitchAction):
        return {
            "kind": "switch",
            "slot": action.slot,
            "partyIndex": action.party_index,
            "species": action.species,
        }
    if isinstance(action, PassAction):
        return {"kind": "pass", "slot": action.slot}
    raise TypeError(f"unknown action {action!r}")


def dump_budget(budget: Budget) -> dict[str, Any]:
    return {
        "damageRolls": budget.damage_rolls,
        "enumerateCrit": budget.enumerate_crit,
        "enumerateAccuracy": budget.enumerate_accuracy,
        "enumerateStatusChecks": budget.enumerate_status_checks,
        "enumerateSecondary": budget.enumerate_secondary,
        "enumerateSpeedTies": budget.enumerate_speed_ties,
        "pinnedPolicy": budget.pinned_policy,
        "maxBranches": budget.max_branches,
    }


@dataclass
class NodeResult:
    """One matrix per objective, plus what the port declined to fill."""

    payoffs: list[list[list[float]]]
    exact: list[list[bool]]
    #: (row, column, why) for every cell the port refused.
    refused: list[tuple[int, int, str]]
    unmodelled: tuple[str, ...]


class RustNode:
    """A warm subprocess. One per worker; it holds the regulation in memory."""

    def __init__(self, reg: Regulation, binary: Path | None = None) -> None:
        self.format_id = reg.meta.format_id
        self.binary = binary or binary_path()
        regulation = repo_root() / "configs" / "regulations" / f"{self.format_id}.json"
        self._process = subprocess.Popen(
            [str(self.binary), "node", str(regulation)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.stdin.close()
            self._process.wait(timeout=5)

    def fill(
        self,
        pos: Position,
        ours: list[SideAction],
        theirs: list[SideAction],
        objectives: list[str],
        budget: Budget,
    ) -> NodeResult:
        request = {
            "position": pos.to_json(),
            "ours": [[dump_action(a) for a in side.slots] for side in ours],
            "theirs": [[dump_action(a) for a in side.slots] for side in theirs],
            "budget": dump_budget(budget),
            "objectives": objectives,
        }
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read()
            raise RuntimeError(f"the Rust node process stopped: {stderr.strip()}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"the Rust node refused the request: {response['error']}")
        return NodeResult(
            payoffs=response["payoffs"],
            exact=response["exact"],
            refused=[(int(i), int(j), str(why)) for i, j, why in response["refused"]],
            unmodelled=tuple(response["unmodelled"]),
        )
