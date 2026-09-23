"""Client for the TypeScript Showdown oracle.

Used only by tests, benchmarks and replay parsing -- never on the solver's hot path. The
protocol is JSONL over stdio and every op takes a batch, so a test that checks 100k stat
spreads pays one round trip, not 100k.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .regulation import STAT_IDS, repo_root

ORACLE_JS = repo_root() / "packages" / "sim-bridge" / "dist" / "cli" / "oracle.js"


class OracleError(RuntimeError):
    pass


@dataclass
class RandomnessPolicy:
    """Pins every source of chance in a turn so comparison is an equality test.

    ``damage_roll`` indexes Showdown's 16 damage rolls: 0 is 100% (maximum), 15 is 85%
    (minimum).
    """

    damage_roll: int = 0
    accuracy: str = "hit"  # 'hit' | 'miss'
    crit: bool = False
    secondary: bool = False
    multihit: str = "min"  # 'min' | 'max'
    speed_tie: str = "keep"  # 'keep' | 'reverse'
    #: What `sample(values)` answers: 'first' or 'last' (IKA-178, a `randomNormal` foe).
    sample: str = "first"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "damageRoll": self.damage_roll,
            "accuracy": self.accuracy,
            "crit": self.crit,
            "secondary": self.secondary,
            "multihit": self.multihit,
            "speedTie": self.speed_tie,
        }
        # Only when asked for, so the request every other caller sends is unchanged.
        if self.sample != "first":
            out["sample"] = self.sample
        return out


@dataclass
class TeamSet:
    species: str
    ability: str
    nature: str
    moves: list[str]
    sp: dict[str, int] = field(default_factory=dict)
    item: str | None = None
    level: int | None = None
    gender: str | None = None
    name: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "species": self.species,
            "ability": self.ability,
            "nature": self.nature,
            "moves": list(self.moves),
            "sp": {k: v for k, v in self.sp.items() if k in STAT_IDS},
        }
        if self.item is not None:
            out["item"] = self.item
        if self.level is not None:
            out["level"] = self.level
        if self.gender is not None:
            out["gender"] = self.gender
        if self.name is not None:
            out["name"] = self.name
        return out

    @staticmethod
    def from_json(d: dict[str, Any]) -> TeamSet:
        return TeamSet(
            species=d["species"],
            ability=d["ability"],
            nature=d["nature"],
            moves=list(d["moves"]),
            sp=dict(d.get("sp", {})),
            item=d.get("item"),
            level=d.get("level"),
            gender=d.get("gender"),
            name=d.get("name"),
        )


def load_team(path: str | Path) -> list[TeamSet]:
    with Path(path).open(encoding="utf-8") as fh:
        return [TeamSet.from_json(d) for d in json.load(fh)]


class Oracle:
    """A long-lived oracle subprocess. Use as a context manager."""

    def __init__(self, node: str = "node", script: Path = ORACLE_JS) -> None:
        if not script.exists():
            raise OracleError(
                f"Oracle not built at {script}. Run: npm run -w @pokeuraou/sim-bridge build"
            )
        self._proc = subprocess.Popen(  # noqa: S603
            [node, str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(repo_root()),
        )
        self._next_id = 0

    def __enter__(self) -> Oracle:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # -- transport -----------------------------------------------------------

    def call(self, op: str, **kwargs: Any) -> Any:
        return self.batch([{"op": op, **kwargs}])[0]

    def batch(self, requests: Sequence[dict[str, Any]]) -> list[Any]:
        """Sends many requests in one write and reads all responses in order."""
        assert self._proc.stdin is not None and self._proc.stdout is not None
        ids: list[int] = []
        lines: list[str] = []
        for req in requests:
            rid = self._next_id
            self._next_id += 1
            ids.append(rid)
            lines.append(json.dumps({"id": rid, **req}, separators=(",", ":")))
        self._proc.stdin.write("\n".join(lines) + "\n")
        self._proc.stdin.flush()

        results: dict[int, Any] = {}
        while len(results) < len(ids):
            line = self._proc.stdout.readline()
            if not line:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise OracleError(f"Oracle exited unexpectedly. stderr:\n{stderr}")
            resp = json.loads(line)
            if not resp.get("ok"):
                raise OracleError(f"Oracle op failed: {resp.get('error')}")
            results[resp["id"]] = resp["result"]
        return [results[i] for i in ids]

    # -- ops -----------------------------------------------------------------

    def ping(self) -> dict[str, Any]:
        return self.call("ping")

    def spread_stats(
        self, format_id: str, rows: Sequence[tuple[str, str, dict[str, int]]], chunk: int = 20000
    ) -> list[dict[str, int]]:
        """Batch stat evaluation: (species, nature, sp) -> final stats.

        This is the authority ``pokeuraou.stats`` is tested against.
        """
        out: list[dict[str, int]] = []
        payload = [{"species": s, "nature": n, "sp": sp} for s, n, sp in rows]
        for start in range(0, len(payload), chunk):
            part = payload[start : start + chunk]
            out.extend(self.call("spread_stats", format=format_id, rows=part))
        return out

    def validate_team(self, format_id: str, team: Iterable[TeamSet]) -> list[str]:
        res = self.call("validate_team", format=format_id, team=[s.to_json() for s in team])
        return list(res["problems"])

    def create(
        self,
        format_id: str,
        team_a: Sequence[TeamSet],
        team_b: Sequence[TeamSet],
        seed: tuple[int, int, int, int] = (1, 2, 3, 4),
        policy: RandomnessPolicy | None = None,
    ) -> BattleHandle:
        res = self.call(
            "create",
            format=format_id,
            teams=[[s.to_json() for s in team_a], [s.to_json() for s in team_b]],
            seed=list(seed),
            policy=(policy or RandomnessPolicy()).to_json(),
        )
        return BattleHandle(self, res["session"], res)


@dataclass
class BattleHandle:
    oracle: Oracle
    session: int
    last: dict[str, Any]

    @property
    def position(self) -> dict[str, Any]:
        return self.last["position"]

    @property
    def requests(self) -> list[dict[str, Any] | None]:
        return self.last["requests"]

    @property
    def choice_errors(self) -> list[str]:
        return self.last["choiceErrors"]

    @property
    def rolls(self) -> list[dict[str, Any]]:
        """Rolls Showdown asked for during the last step, in order.

        ``{'kind': 'chance', 'numerator': 1, 'denominator': 8}`` for
        ``randomChance(1, 8)``, ``{'kind': 'random', 'from': 2, 'to': 5}`` for
        ``random(2, 5)``, and ``{'kind': 'sample', 'values': [...]}`` for ``sample``.

        The policy pins every answer, so this is not a way to observe outcomes -- it is a
        way to see the odds the simulator would have used. Three probabilities in the
        resolver turned out to be the base game's rather than the champions mod's (full
        paralysis 1/4 against the mod's 1/8, thawing 1/5 against 1/4), and a hardcoded
        probability is exactly the kind of thing that cannot be checked any other way.
        """
        return list(self.last.get("rolls") or [])

    @property
    def log(self) -> list[str]:
        return self.last["log"]

    @property
    def ended(self) -> bool:
        return bool(self.position["ended"])

    def step(self, choices: Sequence[str | None]) -> dict[str, Any]:
        self.last = self.oracle.call("step", session=self.session, choices=list(choices))
        return self.position

    def probe(self, side: int, candidates: Sequence[str]) -> list[dict[str, Any]]:
        return self.oracle.call(
            "probe_choices", session=self.session, side=side, candidates=list(candidates)
        )

    def action_speeds(self) -> list[dict[str, Any]]:
        """Showdown's own Speed numbers for the active Pokemon, for direct comparison."""
        return self.oracle.call("action_speeds", session=self.session)

    def refresh(self) -> dict[str, Any]:
        self.last = self.oracle.call("state", session=self.session)
        return self.position

    def close(self) -> None:
        self.oracle.call("close", session=self.session)
