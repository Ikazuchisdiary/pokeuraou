"""Each seat's read of a generation game's move decisions, solved again from the record.

A generation record (`selfplay.GameRecord`) keeps each seat's equilibrium mixture and value, not
the mixture the seat modelled for the other side or the value of each of its moves. Both come
out of the same solve, so they can be had by solving the decision again the way `play_game`
solved it: the recorded position, both sides' completions of the unseen bench priced by the
record's selection mixtures (`BenchPrior`, conditioned on the recorded ``shownIdentities`` and
the turn-1 leads), the menus built by the record's ranking, and `search.belief_solve`.

A re-solve is shown only where it is the recorded one: both seats' menus hold the recorded
actions and both mixtures and values agree with the record to `TOLERANCE`. Anything else is
reported as not matching, with the reason, and the caller keeps the record's own mixture.

What it needs that the record does not hold: the leaf (the model files behind
``searchObjective``), the Q behind ``qModel`` when the ranking is a q fill (checked against
``qModelSha256``), and the two sixes as sets (the pool file ``pool.id``, checked against
``pool.sha256``). `reread_game` takes them as objects; `load_inputs` finds them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pokeuraou import timematch
from pokeuraou.budget import Budget
from pokeuraou.hidden import completions, seen_slots, shown_species
from pokeuraou.position import Position
from pokeuraou.provenance import LEGACY_BENCH_DROP, LEGACY_DEEPEN, LEGACY_RANK_FILL
from pokeuraou.regulation import Regulation, to_id
from pokeuraou.search import belief_solve
from pokeuraou.selection_book import BenchPrior
from pokeuraou.selfplay import _believed, _menus
from pokeuraou.teams import all_selections

#: A re-solve is the recorded one when every probability of both mixtures and both values are
#: within this of the record (the fourth decimal). Measured on data/selfplay-mc3 b07 w3 game 5:
#: a CPU re-solve of a GPU-served generation game differs by at most 1.5e-6.
TOLERANCE = 5e-5


@dataclass
class Reread:
    """One move decision solved again: each seat's read (`timematch.read_summary`'s shape) when
    it matched the record, else None and why."""

    matched: bool
    reads: dict[int, dict[str, Any]]
    worst: float
    reason: str | None = None


def refused(record: dict[str, Any]) -> str | None:
    """Why this record's decisions cannot be solved again here, or None. The re-solve is
    `play_game`'s depth-1 hidden-bench move node with one agent on both seats; a setting
    outside it is refused rather than approximated."""
    if record.get("information") != "hidden-bench":
        return "裏を公開した局（open）の解き直しは未対応"
    for key, legacy in (("deepen", LEGACY_DEEPEN), ("benchDrop", LEGACY_BENCH_DROP)):
        got = record.get(key)
        if got is not None and set(got) != {legacy}:
            return f"{key} {got} の解き直しは未対応"
    if record.get("depth") is not None and set(record["depth"]) != {1}:
        return f"depth {record['depth']} の解き直しは未対応"
    if record.get("knockouts"):
        return "knockouts の解き直しは未対応"
    for key in ("rankView", "rankFill"):
        got = record.get(key)
        if got is not None and len(set(got)) != 1:
            return f"{key} が席ごとに違う局（{got}）の解き直しは未対応"
    if record.get("ranking") == "policy":
        return "方策で順位付けした局の解き直しは未対応"
    return None


def bench_priors(record: dict[str, Any]) -> tuple[BenchPrior | None, BenchPrior | None] | None:
    """Both seats' selection distributions as the game carried them: the record's
    ``ownSelectionMixture`` / ``foeSelectionMixture`` over `all_selections` (what
    `BenchPrior.of` gives for a one-class pool entry). None (uniform) without them."""
    own, foe = record.get("ownSelectionMixture"), record.get("foeSelectionMixture")
    if not own or not foe:
        return None
    selections = tuple(all_selections(len(record["ownSix"]), len(record["ownTeam"])))
    if len(own) != len(selections) or len(foe) != len(selections):
        raise ValueError(f"selection mixtures of {len(own)}/{len(foe)}, not {len(selections)}")
    return tuple(
        BenchPrior(selections=selections, probabilities=tuple(float(p) for p in mix),
                   species=tuple(six))
        for mix, six in ((own, record["ownSix"]), (foe, record["foeSix"]))
    )


def _leads(record: dict[str, Any]) -> list[frozenset[str] | None]:
    """The turn-1 leads per side as `play_game` carries them (species ids of the active slots)."""
    first = record["decisions"][0]
    pos = Position.from_json(first["position"])
    if pos.turn != 1:
        return [None, None]
    return [
        frozenset(shown_species(
            pos, i, frozenset(m.slot for m in pos.sides[i].pokemon if m.active_index is not None)
        ))
        for i in (0, 1)
    ]


def _seat_read(side: int, answer: Any, items: list, chosen: str) -> dict[str, Any]:  # noqa: ANN401
    """`timematch.read_summary` of one seat's Bayesian answer: its mixture, the other side's
    mixture it modelled (its replies averaged by the belief), the per-move values (the
    completions' matrices in the seat's orientation)."""
    row, col, built, weights = answer.node_payoff
    w = np.asarray(weights, dtype=np.float64)
    mine, other = (row, col) if side == 0 else (col, row)
    prices = [m if side == 0 else -m.T for m in built]
    model = sum(
        wk * np.asarray(y, dtype=np.float64) for wk, y in zip(w / w.sum(), answer.replies, strict=True)
    )
    ladder = SimpleNamespace(prices=prices, replies=list(answer.replies), value=answer.value)
    got = timematch.read_summary((side, mine, other, answer.strategy, model, ladder), list(w), chosen)
    got["reread"] = True
    got["classes"] = len(items)
    return got


def reread_game(
    reg: Regulation,
    record: dict[str, Any],
    *,
    sheets: tuple[list[Any], list[Any]],
    evaluate: Any,  # noqa: ANN401 - a LeafEvaluator
) -> dict[int, Reread]:
    """Every move decision of a hidden-bench generation record, solved again (by decision
    index). ``sheets`` are the two sixes in the record's ``ownSix`` / ``foeSix`` order;
    ``evaluate`` the leaf the record searched with (the Q, for a q fill, installed by the
    caller)."""
    why = refused(record)
    if why is not None:
        raise ValueError(why)
    for side, key in ((0, "ownSix"), (1, "foeSix")):
        names = [s.species for s in sheets[side]]
        if [to_id(n) for n in names] != [to_id(n) for n in record[key]]:
            raise ValueError(f"the sheet of side {side} is {names}, the record's {key} {record[key]}")
    limit = record["searchLimit"]
    limits = (limit, limit) if isinstance(limit, int) else tuple(limit)
    ranked = record.get("ranking") == "leaf"
    view = (record.get("rankView") or ["heaviest"])[0]
    fill = (record.get("rankFill") or [LEGACY_RANK_FILL])[0]
    priors = bench_priors(record)
    leads = _leads(record)
    budget = Budget.matrix()
    out: dict[int, Reread] = {}
    for index, d in enumerate(record["decisions"]):
        if d["kind"] != "move":
            continue
        pos = Position.from_json(d["position"])
        seen = [frozenset(d["shownIdentities"][i]) for i in (0, 1)]
        shown = [seen_slots(pos, i, seen[i]) for i in (0, 1)]
        spreads = {}
        for side in (0, 1):
            weights = None
            if priors is not None:
                weights = priors[side].weights(shown_species(pos, side, shown[side]), leads[side]) or None
            spreads[side] = completions(reg, pos, side, sheets[side], seen=shown[side], weights=weights)
        spreads = _believed(spreads, (LEGACY_BENCH_DROP, LEGACY_BENCH_DROP))
        ours, theirs = _menus(
            reg, pos, limits, evaluate, budget, ranked, None, spreads,
            rank_view=view, used={}, rank_fill=fill,
        )
        answers = belief_solve(reg, pos, ours, theirs, spreads, {0: evaluate, 1: evaluate},
                               budget=budget, sides=(0, 1))
        out[index] = _compare(d, answers, spreads)
    return out


def _compare(d: dict[str, Any], answers: dict[int, Any], spreads: dict[int, list]) -> Reread:  # noqa: ANN401
    """The re-solve against the record: the menus, both mixtures by action, both values."""
    worst = 0.0
    reasons = []
    for side, names_key, policy_key in ((0, "ownActions", "ownPolicy"), (1, "foeActions", "foePolicy")):
        answer = answers[side]
        mine = answer.ours if side == 0 else answer.theirs
        got = {a.to_choice(): float(p) for a, p in zip(mine, answer.strategy, strict=True)}
        want = dict(zip(d[names_key], (float(p) for p in d[policy_key]), strict=True))
        if set(got) != set(want):
            reasons.append(f"席 {side} の候補集合が記録と違う（{len(set(got) ^ set(want))} 手）")
            worst = max(worst, 1.0)
            continue
        worst = max(worst, max(abs(got[k] - want[k]) for k in want))
    values = (answers[0].value, -answers[1].value)
    recorded = (d.get("searchValue"), d.get("foeSearchValue"))
    for side in (0, 1):
        if recorded[side] is not None:
            worst = max(worst, abs(values[side] - float(recorded[side])))
    if worst > TOLERANCE and not reasons:
        reasons.append(f"混合か値が記録と {worst:.1e} 違う（許容 {TOLERANCE:g}）")
    if reasons:
        return Reread(False, {}, worst, "・".join(reasons))
    chosen = (d["ownChosen"], d["foeChosen"])
    reads = {side: _seat_read(side, answers[side], spreads[1 - side], chosen[side]) for side in (0, 1)}
    return Reread(True, reads, worst)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(
    reg: Regulation,
    record: dict[str, Any],
    *,
    models: Path,
    pools: Path,
    values: list[Path] | None = None,
    q_model: Path | None = None,
    device: str = "cpu",
) -> tuple[tuple[list[Any], list[Any]], Any]:  # noqa: ANN401
    """The sheets and the leaf a record was played with, found under ``models`` / ``pools``
    by the record's own names and checked against its hashes; the Q installed when the ranking
    is a q fill. ``values`` overrides the leaf's files (their label must still be the record's)."""
    from pokeuraou import qrank
    from pokeuraou.encode import Encoder
    from pokeuraou.humanplay import load_leaf
    from pokeuraou.pool import load_pool

    pool_info = record.get("pool") or {}
    if not pool_info.get("id"):
        raise ValueError("the record names no pool: the sixes' sets are not known")
    pool = load_pool(pools / f"{pool_info['id']}.json", reg)
    if pool_info.get("sha256") and pool.sha256 != pool_info["sha256"]:
        raise ValueError(
            f"the pool {pool_info['id']} is {pool.sha256[:12]}, the record's {pool_info['sha256'][:12]}"
        )
    by_id = {team.id: team for team in pool.teams}
    sheets = tuple(list(by_id[t].sets) for t in pool_info["teams"])
    label = str(record.get("searchObjective") or "")
    if not label.startswith("value:"):
        raise ValueError(f"the record searched with {label!r}, not a value model")
    if values is None:
        stem = label.removeprefix("value:")
        base, _, count = stem.rpartition("x")
        if base and count.isdigit():
            values = [models / f"{base}.pt"] + [models / f"{base}-s{k}.pt" for k in range(1, int(count))]
        else:
            values = [models / f"{stem}.pt"]
    for path in values:
        if not path.exists():
            raise ValueError(f"no model at {path}")
    first = Path(values[0]).stem
    made = f"value:{first}" if len(values) == 1 else f"value:{first}x{len(values)}"
    if made != label:
        raise ValueError(f"the files {[str(p) for p in values]} make {made!r}, the record's {label!r}")
    leaf, encoder, _device = load_leaf(reg, values, device=device, graphs=False)
    fill = (record.get("rankFill") or [LEGACY_RANK_FILL])[0]
    if qrank.is_q(fill):
        names = record.get("qModel") or []
        if q_model is None:
            if len(names) != 1:
                raise ValueError(f"the record's qModel is {names}: name the Q with --q-model")
            q_model = models / names[0]
        wanted = (record.get("qModelSha256") or [None])[0]
        if wanted and _sha256(q_model) != wanted:
            raise ValueError(f"{q_model} is not the record's Q (sha256 {wanted[:12]})")
        qrank.install(qrank.LocalQ(q_model, encoder or Encoder(reg), device=device), "")
    return sheets, leaf
