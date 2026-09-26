"""One worker of an M-C match: two arms, both seats drawn from a pool (IKA-259).

`tools/generation_match.py` plays one roster (our six) against a standings field with a
selection book, which is the M-B board. M-C has no own side: generation draws both seats
from the pool's 65 teams and solves each pair's selection with the leaf where the game
starts (`pokeuraou.poolplay`, IKA-81). This is that game with two agents in it.

What an arm is, per arm, and so in whichever seat it sits:

* its leaf (`--inference-arm` / `--baseline-inference-arm` on a server, or `--value` /
  `--baseline` loaded here; `--baseline-hp-share` plays hp-share, the M-C origin, and
  one of the three is required -- a left-out leaf was silently hp-share until IKA-335),
* its width (`--limit` / `--baseline-limit`) and narrowing (`--rank-leaf` /
  `--baseline-rank-leaf`), and how its leaf ranking fills its cells (`--rank-fill` /
  `--baseline-rank-fill`, IKA-268; a `-nocover` label builds its leaf-ranked menu without
  the cover, IKA-323; `q` / `q-nocover` rank it by a learned Q instead, IKA-274, named by
  `--q-arm` on the server or `--q-model` here),
* whether its leaf scores a finished battle by the net instead of as its result
  (`--net-scores-ends` / `--baseline-net-scores-ends`: IKA-253 undone, for measuring it),
* its selection: an arm with a leaf solves the pair's selection game with THAT leaf,
  draws its four from its side of the solve, and believes the opponent's bench from the
  same solve. Solves are shared between workers through one directory per arm
  (`--selection-store` / `--baseline-selection-store`), never mixed: the tag names the
  leaf and a store solved for another leaf stops the run. An arm without a leaf draws four
  of six uniformly and holds the uniform belief.

Index `i` is game `i // 2` with the tested arm at side `i % 2`; the pair and its seats
come from the game's seed, so the two seats of a game are the same teams with the arms
swapped. Records carry `gameIndex`, `seatIndex` and a provenance whose `seat` names the
tested arm's side, which is what `pokeuraou.sprt` pairs on.

Run through `tools/match_queue.py --pool`, which starts the inference servers and the
queue. The worker prints, per arm, what it plays, and at the end what each arm did in
each seat -- the echo that says a setting reached both seats.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import narrow as narrowing  # noqa: E402
from pokeuraou import qrank  # noqa: E402
from pokeuraou.benchflags import add_bench_flags, require_bench  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.deepen import DEFAULT_DEEPEN, deepen_spec, parse_deepen  # noqa: E402
from pokeuraou.encode import Encoder, EncodingRules  # noqa: E402
from pokeuraou.hidden import DEFAULT_BENCH_DROP, parse_bench_drop  # noqa: E402
from pokeuraou.payoff import HP_SHARE  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game  # noqa: E402
from pokeuraou.provenance import open_games, provenance, write_game  # noqa: E402
from pokeuraou.search import DEFAULT_RANK_FILL, parse_rank_fill, rank_fill_covers  # noqa: E402
from pokeuraou.selfplay import MAX_TURNS  # noqa: E402
from pokeuraou.workqueue import WorkClient  # noqa: E402

#: The width M-C generation plays (IKA-77).
GENERATION_LIMIT = 12


def leaf_name(files: Sequence[str | Path]) -> str:
    """A model stem, `-sN` dropped, `xN` for an ensemble -- `generation_match`'s names."""
    stem = re.sub(r"-s\d+$", "", Path(files[0]).stem)
    return stem if len(files) == 1 else f"{stem}x{len(files)}"


def build_leaves(
    args: argparse.Namespace, encoder: Encoder, other_encoder: Encoder | None = None
) -> tuple[object, object | None, str, str | None]:
    """(tested leaf, other leaf or None, their names). None is hp-share.

    `other_encoder` is the other arm's, when its rules differ (IKA-253's `net_scores_ends`):
    the rules travel with the leaf, so the same weights under other rules are another leaf
    object -- never the tested arm's, or both seats would play one rule.
    """
    other_encoder = other_encoder or encoder
    if args.inference is not None:
        from pokeuraou.inference import RemoteValue

        value = RemoteValue(args.inference, args.inference_arm, encoder)
        baseline = None
        if args.baseline_inference_arm:
            baseline = (
                value
                if args.baseline_inference_arm == args.inference_arm
                and other_encoder is encoder
                else RemoteValue(args.inference, args.baseline_inference_arm, other_encoder)
            )
        # Asked of the server: a worker is told an arm's name, never what it holds.
        names = [leaf_name(value.describe())]
        names.append(leaf_name(baseline.describe()) if baseline is not None else None)
        return value, baseline, names[0], names[1]
    import torch

    from pokeuraou.value import BatchedValue, load_ensemble

    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nets, _ = load_ensemble(args.value, encoder)
    value = BatchedValue([n.to(device) for n in nets], encoder, device=device)
    baseline = None
    if args.baseline:
        if (
            [p.resolve() for p in args.baseline] == [p.resolve() for p in args.value]
            and other_encoder is encoder
        ):
            baseline = value
        else:
            base_nets, _ = load_ensemble(args.baseline, other_encoder)
            baseline = BatchedValue(
                [n.to(device) for n in base_nets], other_encoder, device=device
            )
    return (
        value, baseline, leaf_name(args.value),
        leaf_name(args.baseline) if args.baseline else None,
    )


def luck_leaf(args: argparse.Namespace, value: object, encoder: Encoder) -> object:
    """The evaluator `--aivat` scores the luck with (IKA-193 stage 3): the tested arm's own
    weights through their own caller, so the arm's `calls` and `ended` stay the arm's."""
    if args.inference is not None:
        from pokeuraou.inference import RemoteValue

        return RemoteValue(args.inference, args.inference_arm, encoder)
    from pokeuraou.value import BatchedValue

    return BatchedValue(value.nets, value.encoder, device=value.device)  # type: ignore[attr-defined]


def play_with_luck(ledger_leaf: object | None, name: str, play: object) -> tuple:
    """`play()`'s record and sides, and the game's luck when there is a ledger leaf."""
    if ledger_leaf is None:
        return (*play(), None)  # type: ignore[operator]
    from pokeuraou import luck

    started = time.perf_counter()
    with luck.collecting(ledger_leaf, name) as ledger:  # type: ignore[arg-type]
        record, sides = play()  # type: ignore[operator]
    block = ledger.to_json()
    block["gameSeconds"] = round(time.perf_counter() - started, 4)
    return record, sides, block


def _ends_rule(leaf: object) -> str:
    """How this leaf scores a finished battle: by the net (IKA-253 undone) or its result."""
    rules = getattr(getattr(leaf, "encoder", None), "rules", None)
    return "by the net" if getattr(rules, "net_scores_ends", False) else "as the result"


def _install_q(args: argparse.Namespace, encoder: Encoder, ap: argparse.ArgumentParser) -> list[str]:
    """The Qs the q rank fills rank by (IKA-274): installed, and their files as the server
    (or this worker) holds them -- for the records and the echo; a named Q's (stage 3,
    ``q-nocover.NAME``) as ``NAME=file``. Empty without a q fill."""
    for fill, leafy, label in ((args.rank_fill, args.rank_leaf, "--rank-fill"),
                               (args.baseline_rank_fill, args.baseline_rank_leaf,
                                "--baseline-rank-fill")):
        if qrank.is_q(fill) and not leafy:
            # A damage-ranked arm never reaches the ranking, so the label would be recorded
            # and played by nobody.
            ap.error(f"{label} {fill} ranks the leaf-ranked menu: it needs that arm's rank-leaf")
    # A deepen label whose oracle probes by a Q (IKA-322's q<k>) wants the default Q; it
    # is named to `install_from_args` as a q fill would be, so the flags' checks hold for it.
    probing = ["q" for label in (args.deepen, args.baseline_deepen)
               if deepen_spec(label).q_probe is not None]
    models = qrank.install_from_args(
        args, encoder, (args.rank_fill, args.baseline_rank_fill, *probing), ap.error
    )
    if not models:
        return []
    files = qrank.describe_installed(models)
    print(f"  Q: {', '.join(files)} "
          + (f"(served by {args.inference})" if args.inference else "(loaded here)"),
          file=sys.stderr)
    return files


def _q_delta(before: dict[str, dict[str, int]], label: str) -> tuple[int, int]:
    """(rankings, cells) the Q ranking did under `label` since `before` (`qrank.QRANKED`)."""
    now = qrank.QRANKED.get(label, {})
    was = before.get(label, {})
    return (now.get("rankings", 0) - was.get("rankings", 0),
            now.get("cells", 0) - was.get("cells", 0))


def check_other_arm(args: argparse.Namespace, error: Callable[[str], object]) -> None:
    """The other arm's leaf is named, never defaulted (IKA-335).

    Without a baseline leaf the other arm is hp-share. That used to be what leaving the
    flag out meant, and IKA-296's null control -- an arm against itself -- ran as the arm
    against hp-share because its driver left `--baseline` out. Leaving a leaf out now
    stops the run; hp-share is `--baseline-hp-share`.
    """
    named = args.baseline or args.baseline_inference_arm
    if args.baseline_hp_share and named:
        error("--baseline-hp-share contradicts the baseline leaf "
              f"{args.baseline_inference_arm or [str(p) for p in args.baseline]}: "
              "one of them is not what you meant")
    if not args.baseline_hp_share and not named:
        error("the other arm's leaf is not named: pass --baseline (loaded here) or "
              "--baseline-inference-arm (on a server), or --baseline-hp-share to play "
              "hp-share and mean it (IKA-335)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="data/pool/<id>.json, e.g. regmc-matchupweb")
    ap.add_argument("--value", type=Path, nargs="+", default=None,
                    help="the tested arm's leaf, loaded here (one model or an ensemble)")
    ap.add_argument("--baseline", type=Path, nargs="+", default=None,
                    help="the other arm's leaf, loaded here. One of this, "
                    "--baseline-inference-arm or --baseline-hp-share is required")
    ap.add_argument("--inference", default=None, metavar="HOST:PORT")
    ap.add_argument("--inference-arm", default="value")
    ap.add_argument("--baseline-inference-arm", default=None,
                    help="the server's name for the other arm")
    ap.add_argument("--baseline-hp-share", action="store_true",
                    help="the other arm has no leaf: hp-share, the M-C origin, and you mean "
                    "it. Leaving the other arm's leaf out used to mean this silently, and a "
                    "null control meant as an arm against itself played hp-share (IKA-335)")
    ap.add_argument("--limit", type=int, default=GENERATION_LIMIT)
    ap.add_argument("--baseline-limit", type=int, default=None, help="default: --limit")
    ap.add_argument("--rank-leaf", action="store_true",
                    help="the tested arm narrows by its leaf (what M-C generation does)")
    ap.add_argument("--baseline-rank-leaf", action="store_true", help="same for the other arm")
    ap.add_argument("--rank-fill", default=DEFAULT_RANK_FILL,
                    help="how the tested arm's leaf ranking fills its cells: refs<N> replies "
                    "at the matrix budget, refs<N>-fast at Budget.fast (IKA-268); a -nocover "
                    "suffix builds its leaf-ranked menu without the cover (IKA-323)")
    ap.add_argument("--baseline-rank-fill", default=DEFAULT_RANK_FILL,
                    help="same for the other arm")
    qrank.add_q_flags(ap)
    ap.add_argument("--bench-drop", default=DEFAULT_BENCH_DROP,
                    help="which completions of the opponent's unseen slots the tested "
                    "arm's belief leaves out: w<P> under P%% of the weight, m<P> beyond "
                    "the heaviest P%%; the heaviest stays (IKA-283)")
    ap.add_argument("--baseline-bench-drop", default=DEFAULT_BENCH_DROP,
                    help="same for the other arm")
    ap.add_argument("--deepen", default=DEFAULT_DEEPEN,
                    help="how the tested arm deepens each move decision best first after "
                    "the depth-1 solve, where no bench is hidden: m<N> / r<N> spend N "
                    "cells and read the root whole / restricted (IKA-33); m<N>o<W> / "
                    "m<N>oall also widen the root by a double oracle over the rest of "
                    "the width-W menu / every legal action, s<W> / sall swapping a "
                    "weightless action out for each, b<N>... the oracle without "
                    "deepening (IKA-293); none is off")
    ap.add_argument("--baseline-deepen", default=DEFAULT_DEEPEN,
                    help="same for the other arm")
    ap.add_argument("--depth", type=int, default=1, choices=(1, 2),
                    help="the tested arm's search depth at move nodes. 2 needs "
                    "--solve-restricted: under a hidden bench that is the only depth-2 "
                    "reading `belief_solve` has (IKA-111)")
    ap.add_argument("--baseline-depth", type=int, default=1, choices=(1, 2),
                    help="same for the other arm")
    ap.add_argument("--solve-restricted", action="store_true",
                    help="at depth 2, the tested arm reads its strategy off the refined "
                    "rectangle solved as its own game (IKA-68; per completion under a "
                    "hidden bench, IKA-111)")
    ap.add_argument("--baseline-solve-restricted", action="store_true",
                    help="same for the other arm")
    add_bench_flags(ap)
    ap.add_argument("--net-scores-ends", action="store_true",
                    help="the tested arm's leaf scores a finished battle by the net, not as "
                    "its result (IKA-253 undone; only for measuring the fix)")
    ap.add_argument("--baseline-net-scores-ends", action="store_true",
                    help="same for the other arm")
    ap.add_argument("--selection-store", type=Path, default=None,
                    help="the tested arm's shared solves. Default <games-out dir>/"
                    "selection-<leaf>")
    ap.add_argument("--baseline-selection-store", type=Path, default=None)
    ap.add_argument("--explore-epsilon", type=float, default=0.0,
                    help="selection exploration in the draw and the belief alike. 0: the "
                    "pure equilibrium, as a rating wants (generation plays 0.25)")
    ap.add_argument("--explore-temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--games", type=int, default=100, help="games per seat, without --queue")
    ap.add_argument("--queue", default=None)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--games-out", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="one summary line per seat")
    ap.add_argument("--device", default=None)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--report-every", type=int, default=25)
    ap.add_argument("--aivat", action="store_true",
                    help="write each game's luck (AIVAT's chance and action terms, scored "
                    "by the tested arm's leaf; `pokeuraou.luck`, IKA-193) into its record as "
                    "`aivat`. The games are the same with or without it")
    args = ap.parse_args(argv)
    hide_bench = require_bench(args)
    if (args.inference is None) == (args.value is None):
        ap.error("one of --inference or --value names the tested arm's leaf")
    if args.inference is not None and args.baseline:
        ap.error("--baseline is loaded here; with --inference use --baseline-inference-arm")
    check_other_arm(args, ap.error)
    for fill in (args.rank_fill, args.baseline_rank_fill):
        try:
            parse_rank_fill(fill)
        except ValueError as problem:
            ap.error(str(problem))
    for label in (args.deepen, args.baseline_deepen):
        try:
            parse_deepen(label)
        except ValueError as problem:
            ap.error(str(problem))
    for drop in (args.bench_drop, args.baseline_bench_drop):
        try:
            parse_bench_drop(drop)
        except ValueError as problem:
            ap.error(str(problem))
    for depth, restricted in ((args.depth, args.solve_restricted),
                              (args.baseline_depth, args.baseline_solve_restricted)):
        if restricted != (depth == 2):
            # `play_game` refuses anything else under a hidden bench; the open game would
            # take depth 2 in the mixed reading, which this tool has no use for.
            ap.error("--depth 2 goes with --solve-restricted and depth 1 without it "
                     "(per arm; IKA-111)")

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    # One encoder per arm when their rules differ: the rules travel with the leaf.
    tested_rules = EncodingRules(net_scores_ends=args.net_scores_ends)
    other_rules = EncodingRules(net_scores_ends=args.baseline_net_scores_ends)
    encoder = Encoder(reg, rules=tested_rules)
    other_encoder = (
        encoder if other_rules == tested_rules
        else Encoder(reg, vocab=encoder.vocab, rules=other_rules)
    )
    value, baseline, value_name, baseline_name = (
        build_leaves(args, encoder)
        if other_encoder is encoder
        else build_leaves(args, encoder, other_encoder)
    )
    q_files = _install_q(args, encoder, ap)
    home = args.games_out.parent if args.games_out is not None else Path(".")

    def solver_for(leaf: object, name: str, store: Path | None) -> SolvedSelections:
        # The tag is generation's (`pool sha | value:<leaf>`), so a store generation
        # solved with the same leaf can be read, and one solved with another cannot.
        return SolvedSelections(
            reg, pool.teams, leaf, model=f"value:{name}",
            store=store or home / f"selection-{name}", tag=f"{pool.sha256}|value:{name}",
        )

    tested = PoolArm(
        name=value_name, evaluate=value,
        solver=solver_for(value, value_name, args.selection_store),
        limit=args.limit, rank_by_leaf=args.rank_leaf, rank_fill=args.rank_fill,
        bench_drop=args.bench_drop, deepen=args.deepen, depth=args.depth,
        solve_restricted=args.solve_restricted,
    )
    other_limit = args.limit if args.baseline_limit is None else args.baseline_limit
    if baseline is None:
        other = PoolArm(name=HP_SHARE.name, evaluate=None, solver=None,
                        limit=other_limit, rank_by_leaf=args.baseline_rank_leaf,
                        rank_fill=args.baseline_rank_fill,
                        bench_drop=args.baseline_bench_drop,
                        deepen=args.baseline_deepen,
                        depth=args.baseline_depth,
                        solve_restricted=args.baseline_solve_restricted)
    else:
        assert baseline_name is not None
        # One solver per arm even over one leaf: shared, the second arm would reuse the
        # first one's memo and the echo could not say that each arm solved its own.
        other = PoolArm(
            name=baseline_name, evaluate=baseline,
            solver=(
                tested.solver
                if baseline is value and args.baseline_selection_store is None
                else solver_for(baseline, baseline_name, args.baseline_selection_store)
            ),
            limit=other_limit, rank_by_leaf=args.baseline_rank_leaf,
            rank_fill=args.baseline_rank_fill,
            bench_drop=args.baseline_bench_drop,
            deepen=args.baseline_deepen,
            depth=args.baseline_depth,
            solve_restricted=args.baseline_solve_restricted,
        )
    arms = (tested, other)
    print(pool.summary(), file=sys.stderr)
    print(f"pool match / {reg.meta.format_id} / bench {'hidden' if hide_bench else 'OPEN'} / "
          f"selection eps={args.explore_epsilon}, T={args.explore_temperature}",
          file=sys.stderr)
    for label, arm in (("tested arm", tested), ("other arm", other)):
        store = arm.solver.store if arm.solver is not None else None
        print(
            f"  {label}: leaf {arm.name} / width {arm.limit} / "
            f"{'leaf' if arm.rank_by_leaf else 'damage'} ranking"
            + (f" (fill {arm.rank_fill})" if arm.rank_by_leaf else "")
            + (f" / bench drop {arm.bench_drop}" if hide_bench else "")
            + f" / deepen {arm.deepen}"
            + (f" / ends {_ends_rule(arm.evaluate)}" if arm.evaluate is not None else "")
            + (f" / depth {arm.depth} restricted" if arm.depth != 1 else "")
            + " / selection "
            f"{arm.selection}" + (f" by its own leaf, store {store}" if store else "")
            + f" / belief {'solved' if arm.solver is not None and hide_bench else 'uniform'}",
            file=sys.stderr,
        )
    ranking = tuple("leaf" if arm.rank_by_leaf else "damage" for arm in arms)
    tags = ""
    if tested.limit != other.limit:
        tags += f"@w{tested.limit}"
    if ranking[0] != ranking[1]:
        tags += {"leaf": "@leafrank", "damage": "@damagerank"}[ranking[0]]
    if tested.rank_by_leaf and tested.rank_fill != other.rank_fill:
        tags += f"@rankfill:{tested.rank_fill}"
    if hide_bench and tested.bench_drop != other.bench_drop:
        tags += f"@benchdrop:{tested.bench_drop}"
    if tested.deepen != other.deepen:
        tags += f"@deepen:{tested.deepen}"
    if tested_rules != other_rules:
        tags += f"@enc:{tested_rules.label()}"
    if tested.depth != other.depth:
        tags += f"@d{tested.depth}"
    arm_label = f"{tested.name}{tags}"

    client = WorkClient(args.queue) if args.queue else None
    ledger_leaf = luck_leaf(args, value, encoder) if args.aivat else None
    if ledger_leaf is not None:
        print(f"  aivat: luck scored by {value_name} (the tested arm's leaf) on the true "
              "position, written as `aivat`", file=sys.stderr)

    def work() -> Iterator[int]:
        if client is None:
            yield from range(2 * args.games)
            return
        while (index := client.take()) is not None:
            yield index

    games_file = open_games(args.games_out)
    # Per seat (tested at side 0, side 1): wins, played, unfinished, seconds.
    tally = [[0, 0, 0, 0.0], [0, 0, 0, 0.0]]
    # The echo, per seat and per ARM (0 tested, 1 other): what each arm's side was given.
    echo = [[{"selection": {}, "belief": {}, "leaf": set(), "fill": {}, "drop": {},
              "deepen": {}, "deepened": 0, "widened": 0, "swapped": 0, "oracle": 0,
              "depth": {}, "coverless": {"menus": 0, "dropping": 0, "dropped": 0},
              "q": [0, 0], "qprobe": [0, 0], "calls": 0}
             for _ in arms]
            for _ in range(2)]
    done = 0
    started_all = time.perf_counter()
    for index in work():
        which, game_index = index % 2, index // 2
        calls_before = [getattr(arm.evaluate, "calls", 0) for arm in arms]
        coverless_before = {k: dict(v) for k, v in narrowing.COVERLESS.items()}
        q_before = {k: dict(v) for k, v in qrank.QRANKED.items()}
        started = time.perf_counter()
        record, sides, luck_block = play_with_luck(
            ledger_leaf, value_name,
            lambda game_index=game_index, which=which: pool_match_game(
                reg, pool, arms, seed=args.seed, game_index=game_index, which=which,
                hide_bench=hide_bench, max_turns=args.max_turns,
                epsilon=args.explore_epsilon, temperature=args.explore_temperature,
            ),
        )
        done += 1
        if done % args.report_every == 0:
            print(f"    {done} played, {(time.perf_counter() - started_all) / done:.2f} s/game",
                  flush=True)
        for arm_index in (0, 1):
            side = which if arm_index == 0 else 1 - which
            bucket = echo[which][arm_index]
            for key, value_ in (("selection", sides["selections"][side]),
                                ("belief", sides["beliefs"][side])):
                bucket[key][value_] = bucket[key].get(value_, 0) + 1
            bucket["leaf"].add(sides["leaves"][side])
            # What `play_game` itself recorded as that side's fill, not what the arm holds:
            # the echo has to be read off the game, or it only repeats the command line.
            played_fill = record.rank_fill[side]
            bucket["fill"][played_fill] = bucket["fill"].get(played_fill, 0) + 1
            played_drop = record.bench_drop[side]
            bucket["drop"][played_drop] = bucket["drop"].get(played_drop, 0) + 1
            played_deepen = record.deepen[side]
            bucket["deepen"][played_deepen] = bucket["deepen"].get(played_deepen, 0) + 1
            # Decisions this side actually deepened, read off the game (IKA-33).
            bucket["deepened"] += sum(
                1 for d in record.decisions
                if d.deepened is not None and d.deepened[side] is not None
            )
            # And where the root's double oracle was asked, and what it added (IKA-293).
            for d in record.decisions:
                got = d.deepened[side] if d.deepened is not None else None
                if got is not None and "widened" in got:
                    bucket["oracle"] += 1
                    bucket["widened"] += got["widened"]
                    bucket["swapped"] += got.get("swapped", 0)
                    # The Q that narrowed the probe (IKA-322): inferences, full probes.
                    bucket["qprobe"][0] += got.get("q", 0)
                    bucket["qprobe"][1] += got.get("qfull", 0)
            played_depth = (
                f"{record.depth[side]}"
                + ("r" if record.depth[side] != 1 and record.solve_restricted[side] else "")
            )
            bucket["depth"][played_depth] = bucket["depth"].get(played_depth, 0) + 1
            if arms[0].evaluate is not arms[1].evaluate:
                bucket["calls"] += getattr(arms[arm_index].evaluate, "calls", 0) - calls_before[
                    arm_index]
            # Menus this arm built without the cover, counted by `_menus` under its label
            # (IKA-323): the positive control that -nocover reached the menu. Two arms with
            # the same label share one count.
            arm_fill = arms[arm_index].rank_fill
            if arms[arm_index].rank_by_leaf and not rank_fill_covers(arm_fill):
                now = narrowing.COVERLESS.get(arm_fill, {})
                was = coverless_before.get(arm_fill, {})
                for key in bucket["coverless"]:
                    bucket["coverless"][key] += now.get(key, 0) - was.get(key, 0)
            # Rankings this arm asked of the Q (IKA-274), under its label: the positive
            # control that a q fill reached the menu. Two arms with one label share it.
            if arms[arm_index].rank_by_leaf and qrank.is_q(arm_fill):
                got = _q_delta(q_before, arm_fill)
                bucket["q"] = [bucket["q"][0] + got[0], bucket["q"][1] + got[1]]
        if record.outcome is None:
            tally[which][2] += 1
            if client is not None:
                client.finish(index)
            continue
        side_leaves = sides["leaves"]
        write_game(
            games_file,
            record,
            # Side 0's leaf: `searchValue` in the body is side 0's search.
            objective=side_leaves[0] if side_leaves[0] == HP_SHARE.name
            else f"value:{side_leaves[0]}",
            search_limit=sides["limits"],
            extra={
                "gameIndex": game_index,
                "seatIndex": which,
                "pool": {
                    "id": pool.id, "sha256": pool.sha256, "pair": sides["pair"],
                    "teams": [pool.teams[t].id for t in sides["teams"]],
                    "mirror": sides["mirror"],
                    "picks": [[int(i) for i in p] for p in sides["picks"]],
                    "epsilon": args.explore_epsilon,
                    "temperature": args.explore_temperature,
                },
                # The Q behind a q rank fill (IKA-274), as its holder names it.
                **({"qModel": q_files} if q_files else {}),
                **({"aivat": luck_block} if luck_block is not None else {}),
            },
            source=provenance(
                "pool-match",
                seat=f"{arm_label} = side {which}",
                leaves=side_leaves,
                limits=sides["limits"],
                rankings=sides["rankings"],
                rank_fills=sides["rank_fills"],
                bench_drops=sides["bench_drops"],
                deepens=sides["deepens"],
                depths=sides["depths"],
                solvers=sides["solvers"],
                books=sides["selections"],
                information=("hidden-bench", "hidden-bench") if hide_bench
                else ("open", "open"),
                beliefs=sides["beliefs"],
            ),
        )
        tally[which][1] += 1
        tested_won = record.outcome > 0.5 if which == 0 else record.outcome < 0.5
        tally[which][0] += int(tested_won)
        tally[which][3] += time.perf_counter() - started
        if client is not None:
            client.finish(index)
    if client is not None:
        client.close()
    games_file.close()

    names = (f"tested {tested.name}", f"other {other.name}")
    for which in (0, 1):
        for arm_index in (0, 1):
            bucket = echo[which][arm_index]
            if not bucket["leaf"]:
                continue
            print(
                f"  echo, {arm_label} = side {which}: {names[arm_index]} sat at side "
                f"{which if arm_index == 0 else 1 - which}, leaf {sorted(bucket['leaf'])}, "
                f"selection {bucket['selection']}, belief {bucket['belief']}, "
                f"rank fill {bucket['fill']}, bench drop {bucket['drop']}, "
                f"deepen {bucket['deepen']} ({bucket['deepened']:,} decisions deepened"
                + (
                    f", oracle asked at {bucket['oracle']:,}, {bucket['widened']:,} actions "
                    f"widened, {bucket['swapped']:,} swapped out"
                    + (
                        f", Q probes {bucket['qprobe'][0]:,} inferences "
                        f"({bucket['qprobe'][1]:,} steps probed in full)"
                        if bucket["qprobe"][0]
                        else ""
                    )
                    if bucket["oracle"]
                    else ""
                )
                + "), "
                f"depth {bucket['depth']}"
                + (
                    f", coverless menus {bucket['coverless']['menus']:,} "
                    f"({bucket['coverless']['dropping']:,} leaving options off, "
                    f"{bucket['coverless']['dropped']:,} options)"
                    if bucket["coverless"]["menus"]
                    else ""
                )
                + (
                    f", Q rankings {bucket['q'][0]:,} ({bucket['q'][1]:,} cells)"
                    if bucket["q"][0]
                    else ""
                )
                + (f", leaf requests {bucket['calls']:,}" if bucket["calls"] else ""),
                file=sys.stderr,
            )
    # What each arm's leaf did with the finished battles it met, counted by the leaf itself
    # (IKA-253): the echo that says the rule reached the arm, not the command line again.
    for arm, label in zip(arms, names, strict=True):
        if arm.evaluate is not None:
            print(f"  ends of {label}: {getattr(arm.evaluate, 'ended', 0):,} ended leaves, "
                  f"scored {_ends_rule(arm.evaluate)}", file=sys.stderr)
            if arms[0].evaluate is arms[1].evaluate:
                break
    for arm, label in zip(arms, names, strict=True):
        if arm.solver is not None:
            s = arm.solver
            print(f"  selection of {label}: {s.solves} solved here, {s.loaded} read from "
                  f"{s.store}, {s.reused} reused", file=sys.stderr)
    wins = played = 0
    for which in (0, 1):
        seat_wins, seat_played, unfinished, elapsed = tally[which]
        rate = seat_wins / seat_played if seat_played else float("nan")
        print(f"  {arm_label} = side {which}: {seat_played} games, won {rate * 100:.1f}% "
              f"({elapsed / max(seat_played, 1):.2f} s/game, unfinished {unfinished})",
              flush=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({
                    "seat": f"{arm_label} = side {which}", "seed": args.seed,
                    "model": tested.name, "baseline": other.name, "pool": pool.id,
                    "limit": tested.limit, "baselineLimit": other.limit,
                    "played": seat_played, "wins": seat_wins, "unfinished": unfinished,
                    "seconds": elapsed,
                }) + "\n")
        wins += seat_wins
        played += seat_played
    if played:
        print(f"\n  both seats, {played} games: {arm_label} won {wins / played * 100:.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
