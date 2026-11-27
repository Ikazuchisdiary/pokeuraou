"""IKA-458 stage 3.7: the generation worker is the port's executable (`rust/src/worker.rs`).

`tools/selfplay.py --pool` played the queue's games in Python: per game it asked the port to
prepare the game (`prepGame`), solved a new pair's selection with the leaf, asked the port to
play it (`playGame`), wrote the record and told the queue. `pokeuraou-damage selfplay-worker`
does all of that itself. Python keeps what is the same for the whole run and is decided by the
flags: this module turns a worker's parsed flags into one JSON file (`build_config`) -- the pool
in the shapes the record and the store write, the seed, the exploration, the store and its tag,
the static parts of `playGame`'s request and the record's fields that no game changes, in
`GameRecord.to_json`'s order -- and `command` names the executable that reads it.

Only the shape generation plays goes this way (the leaf and the Q on an inference server, the
solved selection with its store, the leaf ranking by a `q-nocover` fill, no deepening, depth 1,
every completion believed); `unsupported` says what else a flag asks for, and the launcher stops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .budget import Budget
from .deepen import DEFAULT_DEEPEN
from .eqselect import DEFAULT_EQ_SELECT
from .hidden import DEFAULT_BENCH_DROP
from .regulation import repo_root

#: The executable's subcommand.
SUBCOMMAND = "selfplay-worker"


def unsupported(args: Any) -> list[str]:  # noqa: ANN401 - tools/selfplay.py's namespace
    """What the flags ask for that the Rust worker does not play, as the flags' words."""
    from .qrank import is_q
    from .search import rank_fill_covers

    out = []
    if args.inference is None:
        out.append("a leaf not on an inference server (--inference)")
    if getattr(args, "uniform_selection", False):
        out.append("--uniform-selection")
    if getattr(args, "no_selection_memo", False):
        out.append("--no-selection-memo")
    if not args.rank_leaf:
        out.append("the damage ranking (no --rank-leaf)")
    elif not is_q(args.rank_fill) or rank_fill_covers(args.rank_fill):
        out.append(f"--rank-fill {args.rank_fill} (only a q-nocover fill)")
    if args.bench_drop != DEFAULT_BENCH_DROP:
        out.append(f"--bench-drop {args.bench_drop}")
    if args.deepen != DEFAULT_DEEPEN:
        out.append(f"--deepen {args.deepen}")
    if args.depth != 1 or args.solve_restricted:
        out.append(f"--depth {args.depth}")
    if getattr(args, "record_rank_scores", False):
        out.append("--record-rank-scores")
    return out


def _record_template(
    leaf_label: str, rank_fill: str, limit: int, q_record: dict[str, Any]
) -> list[list[Any]]:
    """The record's keys in `GameRecord.to_json`'s order, with the values no game changes.

    The record `play_game` and `generate_pool` make for the shape this worker plays, with
    placeholders for the per-game fields (`worker.rs`'s `PER_GAME`).
    """
    from .selfplay import GameRecord

    record = GameRecord(own_team=[], foe_team=[], foe_archetype="pool")
    record.information = "hidden-bench"
    record.ranking = "leaf"
    record.rank_view = ["heaviest", "heaviest"]
    record.rank_fill = [rank_fill, rank_fill]
    record.bench_drop = [DEFAULT_BENCH_DROP, DEFAULT_BENCH_DROP]
    record.deepen = [DEFAULT_DEEPEN, DEFAULT_DEEPEN]
    record.knockouts = [False, False]
    record.rust_binary = [None, None]
    record.ally_targets = ["off", "off"]
    record.force_mega = [None, None]
    record.hold_baton = [None, None]
    record.baton_hold = [None, None]
    record.depth = [1, 1]
    record.solve_restricted = [False, False]
    record.eq_select = [DEFAULT_EQ_SELECT, DEFAULT_EQ_SELECT]
    record.selection_source = "solved"
    record.leads = [[], []]
    record.final_position = {}
    record.end_reason = "wipeout"
    payload = record.to_json(objective=leaf_label, search_limit=limit)
    payload["pool"] = {}
    payload.update(q_record)
    payload["gameIndex"] = 0
    return [[key, value] for key, value in payload.items()]


def build_config(  # noqa: PLR0913 - the run's settings
    args: Any,  # noqa: ANN401 - tools/selfplay.py's namespace
    pool: Any,  # noqa: ANN401 - pool.Pool
    evaluate: Any,  # noqa: ANN401 - inference.RemoteValue
    leaf_label: str,
    q_record: dict[str, Any],
    pairs: list[tuple[int, int]] | None,
    store: Path,
) -> dict[str, Any]:
    """The worker's config for these flags (`unsupported` empty), with the leaf and the Q this
    process connected to; their blocks' names are the worker's own (`worker.rs` writes them)."""
    from . import qrank
    from .beliefnode import _stack_leaf
    from .encode import rules_of
    from .inference import BUFFER_BYTES
    from .playgame import static_node
    from .qrank import Q_BUFFER_BYTES
    from .rootmenus import static_request
    from .rustnode import dump_budget
    from .selection import NOTES
    from .selection_book import set_to_json
    from .selfplay import MAX_TURNS, _HiddenBench, _self_switch_request, _set_json

    reg = pool.reg
    model = qrank.installed(qrank.q_name(args.rank_fill))
    rules = rules_of(evaluate)
    leaf = _stack_leaf(getattr(evaluate, "__self__", evaluate), rules)
    if leaf is None or type(model) is not qrank.RemoteQ:
        raise SystemExit(
            "the Rust worker plays the leaf and the Q of an inference server (--inference, --q-arm)"
        )
    ready = _self_switch_request((evaluate, evaluate), _HiddenBench((), [], None, [None, None], 0))
    if ready is None:
        raise SystemExit("the Rust worker needs the leaf of an inference server")
    context = ready[0]
    limits = (args.limit, args.limit)
    node = static_node(static_request(model, limits, "qServer"), leaf, leaf.encoder, rules, Budget.matrix())

    def sheet(s: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"species": s.species, "ability": s.ability, "item": s.item, "nature": s.nature,
                "sp": dict(s.sp), "moves": list(s.moves)}

    return {
        "regulation": str(repo_root() / "configs" / "regulations" / f"{reg.meta.format_id}.json"),
        "seed": int(args.seed),
        "maxTurns": int(getattr(args, "max_turns", MAX_TURNS)),
        "epsilon": float(args.explore_epsilon),
        "temperature": float(args.explore_temperature),
        "store": str(Path(store).resolve()),
        "tag": f"{pool.sha256}|{leaf_label}",
        "model": leaf_label,
        "notes": list(NOTES),
        "pool": {
            "id": pool.id,
            "sha256": pool.sha256,
            "teams": [
                {
                    "id": team.id,
                    "name": team.name,
                    "sets": [sheet(s) for s in team.sets],
                    "recordSets": [_set_json(reg, s) for s in team.sets],
                    "bookSets": [set_to_json(s) for s in team.sets],
                }
                for team in pool.teams
            ],
            "pairs": [[a, b] for a, b in pool.pairs],
        },
        "pairs": None if pairs is None else [[a, b] for a, b in pairs],
        "value": {"model": leaf.model, "bytes": int(BUFFER_BYTES)},
        "q": {"model": model.model, "bytes": int(Q_BUFFER_BYTES)},
        "context": context,
        "node": node,
        "exactBudget": dump_budget(Budget.exact()),
        "record": _record_template(leaf_label, args.rank_fill, args.limit, q_record),
    }


#: The board's worker (`rust/src/boardworker.rs`).
BOARD_SUBCOMMAND = "pool-match-worker"


def board_unsupported(args: Any) -> list[str]:  # noqa: ANN401 - tools/pool_match.py's namespace
    """What a board's flags ask for that the Rust worker does not play, as the flags' words."""
    from .qrank import is_q
    from .search import rank_fill_covers

    out = []
    if args.inference is None:
        out.append("leaves loaded in the worker (--value, --baseline): the leaves of an inference "
                   "server only (--inference, --baseline-inference-arm)")
    for prefix, flag in (("", "--"), ("baseline_", "--baseline-")):
        def got(name: str, prefix: str = prefix) -> Any:  # noqa: ANN401
            return getattr(args, prefix + name)

        if not got("rank_leaf"):
            out.append(f"the damage ranking (no {flag}rank-leaf)")
        elif not is_q(got("rank_fill")) or rank_fill_covers(got("rank_fill")):
            out.append(f"{flag}rank-fill {got('rank_fill')} (only a q-nocover fill)")
        if got("bench_drop") != DEFAULT_BENCH_DROP:
            out.append(f"{flag}bench-drop {got('bench_drop')}")
        if got("deepen") != DEFAULT_DEEPEN:
            out.append(f"{flag}deepen {got('deepen')}")
        if got("depth") != 1 or got("solve_restricted"):
            out.append(f"{flag}depth {got('depth')}")
        if got("eq_select") != DEFAULT_EQ_SELECT:
            out.append(f"{flag}eq-select {got('eq_select')}")
        for name in ("knockouts", "line1", "hold_baton", "bench_oracle", "net_scores_ends"):
            if got(name):
                out.append(f"{flag}{name.replace('_', '-')}")
        if got("ally_targets") != "off":
            out.append(f"{flag}ally-targets {got('ally_targets')}")
    return out


def _board_template(  # noqa: PLR0913 - one seat's record
    which: int, arms: tuple[Any, Any], arm_label: str, pool: Any,  # noqa: ANN401
    args: Any, q_record: dict[str, Any],  # noqa: ANN401
) -> list[list[Any]]:
    """The record `tools/pool_match.py` wrote for a game with the tested arm at side ``which``, in
    `write_game`'s order, with placeholders for the per-game fields (`boardworker.rs`'s `PER_GAME`)."""
    from .payoff import HP_SHARE
    from .poolplay import SOLVED
    from .provenance import provenance
    from .selfplay import GameRecord, _fingerprint

    side_arms = (arms[0], arms[1]) if which == 0 else (arms[1], arms[0])
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="pool")
    record.information = "hidden-bench"
    record.ranking = "leaf"
    record.rank_view = ["heaviest", "heaviest"]
    record.rank_fill = [arm.rank_fill for arm in side_arms]
    record.bench_drop = [DEFAULT_BENCH_DROP, DEFAULT_BENCH_DROP]
    record.deepen = [DEFAULT_DEEPEN, DEFAULT_DEEPEN]
    record.rust_binary = [None if arm.rust_binary is None else _fingerprint(arm.rust_binary)
                          for arm in side_arms]
    record.eq_select = [DEFAULT_EQ_SELECT, DEFAULT_EQ_SELECT]
    sources = tuple(arm.selection for arm in side_arms)
    record.selection_source = sources[0] if sources[0] == sources[1] else "mixed"
    record.leads = [[], []]
    record.final_position = {}
    record.end_reason = "wipeout"
    leaves = tuple(arm.name for arm in side_arms)
    limits = tuple(arm.limit for arm in side_arms)
    payload = record.to_json(
        objective=leaves[0] if leaves[0] == HP_SHARE.name else f"value:{leaves[0]}",
        search_limit=limits,
    )
    payload["provenance"] = provenance(
        "pool-match",
        seat=f"{arm_label} = side {which}",
        leaves=leaves,
        limits=limits,
        rankings=("leaf", "leaf"),
        rank_fills=tuple(arm.rank_fill for arm in side_arms),
        bench_drops=(DEFAULT_BENCH_DROP, DEFAULT_BENCH_DROP),
        deepens=(DEFAULT_DEEPEN, DEFAULT_DEEPEN),
        depths=(1, 1),
        solvers=("full", "full"),
        books=sources,
        information=("hidden-bench", "hidden-bench"),
        beliefs=tuple("uniform" if arm.solver is None else SOLVED for arm in side_arms),
    )
    payload.update({
        "gameIndex": 0,
        "seatIndex": which,
        "pool": {"id": pool.id, "sha256": pool.sha256, "pair": 0, "teams": [], "mirror": False,
                 "picks": [], "epsilon": args.explore_epsilon, "temperature": args.explore_temperature},
        **q_record,
    })
    return [[key, value] for key, value in payload.items()]


def board_config(  # noqa: PLR0913 - the board's settings
    args: Any,  # noqa: ANN401 - tools/pool_match.py's namespace
    pool: Any,  # noqa: ANN401 - pool.Pool
    arms: tuple[Any, Any],  # noqa: ANN401 - poolplay.PoolArm, tested then other
    arm_label: str,
    q_record: dict[str, Any],
    pairs: list[tuple[int, int]] | None,
) -> dict[str, Any]:
    """The board worker's config for these flags (`board_unsupported` empty), with the leaves and
    the Qs this process connected to; their blocks and addresses are the worker's own."""
    from . import qrank
    from .beliefnode import _stack_leaf
    from .encode import rules_of
    from .playgame import static_node
    from .rootmenus import static_request
    from .rustnode import dump_budget
    from .selection import NOTES
    from .selection_book import set_to_json
    from .selfplay import MAX_TURNS, _HiddenBench, _self_switch_request, _set_json

    reg = pool.reg
    ready = _self_switch_request(
        (arms[0].evaluate, arms[1].evaluate), _HiddenBench((), [], None, [None, None], 0)
    )
    if ready is None:
        raise SystemExit("the Rust worker plays the leaves of an inference server")
    context, owners = ready
    servers = []
    for server, owner in zip(context["servers"], owners, strict=True):
        servers.append({**server, "role": "inference" if owner.address == args.inference
                        else "baseline-inference"})
    qs: list[Any] = []
    arm_configs = []
    for k, arm in enumerate(arms):
        leaf = _stack_leaf(getattr(arm.evaluate, "__self__", arm.evaluate), rules_of(arm.evaluate))
        model = qrank.installed(qrank.q_name(arm.rank_fill))
        if leaf is None or type(model) is not qrank.RemoteQ:
            raise SystemExit("the Rust worker plays the leaves and the Qs of an inference server")
        if not any(model is seen for seen in qs):
            qs.append(model)
        node = static_node(static_request(model, (arm.limit, arm.limit), "qServer"), leaf,
                           leaf.encoder, rules_of(leaf), Budget.matrix())
        for key in ("server", "qServer", "widths"):
            node.pop(key)
        solver = arm.solver
        arm_configs.append({
            "name": arm.name,
            "leaf": context["leaf"][k],
            "q": next(i for i, seen in enumerate(qs) if seen is model),
            "limit": int(arm.limit),
            "store": None if solver is None else str(Path(solver.store).resolve()),
            "tag": None if solver is None else solver.tag,
            "model": None if solver is None else solver.model,
            "binary": None if arm.rust_binary is None else str(Path(arm.rust_binary).resolve()),
            "node": node,
        })

    def sheet(s: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"species": s.species, "ability": s.ability, "item": s.item, "nature": s.nature,
                "sp": dict(s.sp), "moves": list(s.moves)}

    return {
        "mode": "board",
        "regulation": str(repo_root() / "configs" / "regulations" / f"{reg.meta.format_id}.json"),
        "seed": int(args.seed),
        "maxTurns": int(getattr(args, "max_turns", MAX_TURNS)),
        "epsilon": float(args.explore_epsilon),
        "temperature": float(args.explore_temperature),
        "notes": list(NOTES),
        "label": arm_label,
        "pool": {
            "id": pool.id,
            "sha256": pool.sha256,
            "teams": [
                {
                    "id": team.id,
                    "name": team.name,
                    "sets": [sheet(s) for s in team.sets],
                    "recordSets": [_set_json(reg, s) for s in team.sets],
                    "bookSets": [set_to_json(s) for s in team.sets],
                }
                for team in pool.teams
            ],
            "pairs": [[a, b] for a, b in pool.pairs],
        },
        "pairs": None if pairs is None else [[a, b] for a, b in pairs],
        "servers": servers,
        "qServers": [{"model": q.model, "bytes": int(q.buffer_bytes)} for q in qs],
        "arms": arm_configs,
        "exactBudget": dump_budget(Budget.exact()),
        "records": [_board_template(which, arms, arm_label, pool, args, q_record) for which in (0, 1)],
    }


def board_command(config: Path, inference: str, games_out: Path, summary: Path, *,
                  baseline_inference: str | None = None, queue: str | None = None,
                  first_index: int = 0, indices: int = 0) -> list[str]:
    """The board worker's command line: the queue's indices, or ``indices`` from ``first_index``."""
    from .rustnode import binary_path, require_current_binary

    require_current_binary()
    tail = (["--queue", queue] if queue
            else ["--first-index", str(first_index), "--indices", str(indices)])
    return [str(binary_path()), BOARD_SUBCOMMAND, "--config", str(config), "--inference", inference,
            *(["--baseline-inference", baseline_inference] if baseline_inference else []),
            "--games-out", str(games_out), "--out", str(summary), *tail]


def write_config(config: dict[str, Any], path: Path) -> None:
    """Bytes, so the file is LF on Windows too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(config, ensure_ascii=False) + "\n").encode("utf-8"))


def command(config: Path, inference: str, out: Path, *, queue: str | None = None,
            first_game: int = 0, games: int = 0) -> list[str]:
    """The worker's command line: the queue's games, or ``games`` from ``first_game``."""
    from .rustnode import binary_path, require_current_binary

    require_current_binary()
    tail = ["--queue", queue] if queue else ["--first-game", str(first_game), "--games", str(games)]
    return [str(binary_path()), SUBCOMMAND, "--config", str(config), "--inference", inference,
            "--out", str(out), *tail]
