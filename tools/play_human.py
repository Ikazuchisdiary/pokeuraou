"""Play one game against the agent at the terminal, on a clock (IKA-330, stage 1).

    uv run python tools/play_human.py --seconds 45 --cores 8
    uv run python tools/play_human.py --seconds 15 --human-team 3 --agent-team 17 --human-side 1
    uv run python tools/play_human.py --seconds 5 --person random --seed 4     # a stand-in person
    uv run python tools/play_human.py --person record:data/human/games.jsonl --clock count

The game, the view and the clock are `pokeuraou.humanplay`'s (its docstring says what the
person is shown and how the agent spends its seconds). This file loads the pieces:

* the teams: two of the M-C pool's (``--human-team`` / ``--agent-team``, by index or id;
  both drawn from the seed when neither is given) or a roster file each
  (``--human-team-file`` / ``--agent-team-file``, `teams.load_roster`'s shape);
* the leaf: the M-C ensemble (`DEFAULT_VALUE`) when it is there, ``--value`` to name
  another, ``--hp-share`` for none -- said at the start either way;
* the menus: ``q-nocover`` when a Q is there (``--q-model``, default `DEFAULT_Q`), the
  default fill otherwise, with a note; ``--rank-fill`` overrides;
* the port's threads: ``--cores`` (`rustnode.set_port_threads`, IKA-32).

The person: ``terminal`` (you), ``first`` / ``random`` (stand-ins, for smoke runs),
``script:<file>`` (one answer per line: the selection as party numbers, then each choice
as its number in the legal list or its choice string), or ``record:<games.jsonl>[@n]``
(the inputs of the n-th game in a record: that game again, byte for byte on the count
clock).

Each game appends one line to ``--out`` and its clock to ``<out>.clock.jsonl``.

The screen (IKA-332): ``--view`` serves a page at http://127.0.0.1:8332/ (``--view-port``)
that shows the board, the agent's answer as it forms (its mixture, the value over time, the
cells and levels read, the opponent's mixture and the bench belief, the principal variation
as a tree) and takes the person's input with ``--person web``. ``--live-out`` keeps the
page's frames (`liveview.FileSink`); ``tools/live_view.py`` shows such a file again.

    uv run python tools/play_human.py --seconds 15 --cores 8 --person web --view
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import humanplay, liveview, qrank, rustnode  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.hidden import DEFAULT_BENCH_DROP, parse_bench_drop  # noqa: E402
from pokeuraou.names import localiser  # noqa: E402
from pokeuraou.pool import draw_pair, load_pool  # noqa: E402
from pokeuraou.regulation import repo_root  # noqa: E402
from pokeuraou.search import DEFAULT_RANK_FILL, parse_rank_fill  # noqa: E402
from pokeuraou.selfplay import MAX_TURNS  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402

#: The M-C leaf that ships (CLAUDE.md): the two-model ensemble.
DEFAULT_VALUE = ("data/models/value-mc0.pt", "data/models/value-mc0-s1.pt")
#: Where the Q that q-nocover ranks by is looked for when --q-model is not given. IKA-274
#: stage 3 is still training it; until a file is here the menus fall back, with a note.
DEFAULT_Q = "data/models/q-mc0.pt"
#: The fill the user chose on 9/26 for boards and human play (IKA-274 stage 2).
Q_FILL = "q-nocover"


def leaf_name(files: list[Path]) -> str:
    """A model stem, `-sN` dropped, `xN` for an ensemble (`pool_match.leaf_name`)."""
    stem = re.sub(r"-s\d+$", "", Path(files[0]).stem)
    return stem if len(files) == 1 else f"{stem}x{len(files)}"


def _team(pool, key: str):  # noqa: ANN001, ANN202
    if key.isdigit():
        return pool.teams[int(key)]
    for team in pool.teams:
        if team.id == key:
            return team
    raise SystemExit(f"no team {key!r} in {pool.id}")


def _person(spec: str, reg, loc, seed: int, server=None):  # noqa: ANN001, ANN202
    if spec == "web":
        if server is None:
            raise SystemExit("--person web answers from the page: give --view")
        return liveview.WebPerson(server)
    if spec == "terminal":
        return humanplay.TerminalPerson(reg, loc)
    if spec in ("first", "random"):
        return humanplay.PolicyPerson(spec, seed)
    if spec.startswith("script:"):
        return humanplay.ScriptPerson.from_file(Path(spec[len("script:"):]))
    if spec.startswith("record:"):
        path, _, line = spec[len("record:"):].partition("@")
        return humanplay.ScriptPerson.from_record(Path(path), int(line or 0))
    raise SystemExit(
        f"--person is terminal, web, first, random, script:<file> or record:<file>[@n], not {spec!r}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="regmc-matchupweb")
    ap.add_argument("--human-team", default=None, help="the person's pool team, by index or id")
    ap.add_argument("--agent-team", default=None, help="the agent's pool team, by index or id")
    ap.add_argument("--human-team-file", type=Path, default=None, help="a roster file instead")
    ap.add_argument("--agent-team-file", type=Path, default=None, help="a roster file instead")
    ap.add_argument("--human-side", type=int, default=0, choices=(0, 1))
    ap.add_argument("--seconds", type=float, default=45.0, help="the agent's budget per move")
    ap.add_argument("--cores", type=int, default=1,
                    help="the port's threads (IKA-32: 8 is the practical best)")
    ap.add_argument("--clock", default="wall", choices=humanplay.CLOCKS,
                    help="wall: stop the deepening at the budget by the clock (default); "
                    "count: spend it at measured prices, reproducible by seed")
    ap.add_argument("--width-only", action="store_true",
                    help="no deepening: the width rule alone (a baseline; how NODE_TIME is measured)")
    ap.add_argument("--max-levels", type=int, default=None,
                    help="the deepening's depth guard (IKA-307; default deepen.MAX_LEVELS). "
                    "Given, each move's record says why the deepening and its lines stopped")
    ap.add_argument("--child-q", type=int, default=None,
                    help="the deepening's child menus: each side's k best by the Q (IKA-307), "
                    "instead of narrow's damage-ranked 8")
    ap.add_argument("--value", type=Path, nargs="+", default=None,
                    help=f"the agent's leaf (one model or an ensemble). Default: {' '.join(DEFAULT_VALUE)}")
    ap.add_argument("--hp-share", action="store_true", help="no leaf: the hp-share proxy")
    ap.add_argument("--rank-fill", default=None,
                    help=f"how the agent's menus are ranked. Default: {Q_FILL} when a Q is there, "
                    f"else {DEFAULT_RANK_FILL}")
    ap.add_argument("--q-model", type=Path, default=None, help=f"the Q. Default: {DEFAULT_Q}")
    ap.add_argument("--bench-drop", default=DEFAULT_BENCH_DROP)
    ap.add_argument("--person", default="terminal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game-index", type=int, default=0)
    ap.add_argument("--games", type=int, default=1, help="games in a row (game index counts up)")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--out", type=Path, default=Path("data/human/games.jsonl"))
    ap.add_argument("--locale", default="ja")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true", help="no board on the terminal (stand-in persons)")
    ap.add_argument("--view", action="store_true",
                    help="serve the page that shows the game and the agent's reading (IKA-332)")
    ap.add_argument("--view-host", default="127.0.0.1")
    ap.add_argument("--view-port", type=int, default=8332)
    ap.add_argument("--live-out", type=Path, default=None,
                    help="keep the page's frames in this file (tools/live_view.py shows it again)")
    ap.add_argument("--sprite-url", default=None,
                    help="the page's images, {id} = Showdown's sprite id (default: Showdown's server; "
                    "\"\" for none, name cards)")
    ap.add_argument("--interval-ms", type=float, default=100.0,
                    help="the least time between two steps sent while the agent thinks")
    args = ap.parse_args(argv)
    if args.person == "web":
        args.view = True
    parse_bench_drop(args.bench_drop)

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    loc = localiser(reg, args.locale)
    say = (lambda text: None) if args.quiet else (lambda text: print(text, file=sys.stderr))

    # The leaf.
    evaluate = None
    name = "hp-share"
    encoder = None
    device = args.device
    values = args.value
    if values is None and not args.hp_share:
        found = [repo_root() / p for p in DEFAULT_VALUE]
        if all(p.exists() for p in found):
            values = found
        else:
            say(f"note: no {DEFAULT_VALUE[0]} here, so the agent plays hp-share (--value names a leaf)")
    if values and not args.hp_share:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_ensemble

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        encoder = Encoder(reg)
        nets, _ = load_ensemble(values, encoder)
        evaluate = BatchedValue([n.to(device) for n in nets], encoder, device=torch.device(device))
        name = leaf_name(values)

    # The menus.
    fill = args.rank_fill
    q_files: list[str] = []
    q_path = args.q_model or (repo_root() / DEFAULT_Q)
    if fill is None:
        fill = Q_FILL if q_path.exists() and encoder is not None else DEFAULT_RANK_FILL
        if fill != Q_FILL:
            why = "no leaf to read its encoding" if encoder is None else f"no Q at {q_path}"
            say(f"note: menus ranked by {fill}, not {Q_FILL} ({why}; --q-model names one)")
    parse_rank_fill(fill)
    if qrank.is_q(fill):
        if encoder is None:
            raise SystemExit(f"{fill} needs a leaf's encoder (not --hp-share)")
        if not q_path.exists():
            raise SystemExit(f"{fill} needs a Q: no file at {q_path}")
        model = qrank.LocalQ(q_path, encoder, device=device or "cpu")
        qrank.install(model)
        q_files = model.describe()

    if args.cores > 1:
        rustnode.set_port_threads(args.cores)
    agent = humanplay.Agent(
        reg=reg, evaluate=evaluate, name=name, seconds=args.seconds, cores=args.cores,
        clock=args.clock, rank_fill=fill, bench_drop=args.bench_drop,
        width_only=args.width_only, max_levels=args.max_levels, child_q=args.child_q,
    )
    if args.child_q is not None and not qrank.is_q(fill):
        raise SystemExit("--child-q ranks the children by the Q: it needs a Q (a q rank fill)")
    say(f"agent: leaf {name} / menus {fill}" + (f" ({', '.join(q_files)})" if q_files else "")
        + f" / {args.seconds:g} s a move on {args.cores} core(s), {args.clock} clock"
        + (" / width only" if args.width_only else "") + " / bench hidden")

    server = None
    if args.view or args.live_out is not None:
        sink = liveview.FileSink(args.live_out) if args.live_out is not None else None
        server = liveview.LiveServer(
            args.view_host, args.view_port if args.view else 0, sink=sink,
            sprite_url=args.sprite_url,
        ).start()
        if args.view:
            print(f"画面: {server.url}", file=sys.stderr)
    person = _person(args.person, reg, loc, args.seed, server)
    for n in range(args.games):
        index = args.game_index + n
        if args.human_team_file or args.agent_team_file:
            if not (args.human_team_file and args.agent_team_file):
                raise SystemExit("give both --human-team-file and --agent-team-file")
            human, mine = load_roster(args.human_team_file), load_roster(args.agent_team_file)
        elif args.human_team is not None or args.agent_team is not None:
            if args.human_team is None or args.agent_team is None:
                raise SystemExit("give both --human-team and --agent-team")
            human, mine = _team(pool, args.human_team), _team(pool, args.agent_team)
        else:
            import numpy as np

            _k, a, b = draw_pair(np.random.default_rng([args.seed, index]), pool.pairs)
            human, mine = pool.teams[a], pool.teams[b]
        agent_side = 1 - args.human_side
        teams = (mine, human) if agent_side == 0 else (human, mine)
        started = time.perf_counter()
        payload, clock, game = humanplay.play(
            agent, person, teams, agent_side=agent_side, seed=args.seed, game_index=index,
            max_turns=args.max_turns, loc=loc,
            out=None if args.quiet or args.person == "web" else sys.stdout,
            listener=None if server is None else server.listener, interval_ms=args.interval_ms,
        )
        payload["pool"] = {"id": pool.id, "sha256": pool.sha256}
        if q_files:
            payload["qModel"] = q_files
        clock["gameSeconds"] = round(time.perf_counter() - started, 3)
        humanplay.write_line(args.out, payload)
        humanplay.write_line(humanplay.clock_path(args.out), clock)
        outcome = payload["outcome"]
        you = 1 - agent_side
        result = ("引き分け・打ち切り" if outcome is None
                  else "あなたの勝ち" if (outcome > 0.5) == (you == 0) else "あなたの負け")
        moves = [d for d in clock["decisions"] if d["kind"] == "move"]
        ratios = [d["ratio"] for d in moves]
        print(
            f"game {index}: {result} ({payload['endReason']}, {payload['turns']} turns). "
            f"agent moves {len(moves)}, seconds/budget "
            + (f"mean {sum(ratios) / len(ratios):.3f} min {min(ratios):.3f} max {max(ratios):.3f}"
               if ratios else "-")
            + f"; written to {args.out}",
            file=sys.stderr,
        )
        del game
    if server is not None:
        if server.sink is not None:
            server.sink.close()
        if args.view and not args.quiet:
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input("画面を閉じるには Enter（Ctrl+C）> ")
        server.close()


if __name__ == "__main__":
    main()
