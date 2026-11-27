"""Writes the config of an M-C board's worker, `pokeuraou-damage pool-match-worker` (IKA-259, IKA-458).

M-C has no own side: generation draws both seats from the pool's 65 teams and solves each
pair's selection with the leaf where the game starts (IKA-81). A board is that game with two
agents in it. The games are played by the Rust worker (`rust/src/boardworker.rs`);
`tools/match_queue.py` runs this once per board to turn the flags into the JSON every worker
reads (`pokeuraou.rustworker.board_config`). The Python loop that played them here was
removed in IKA-458 stage 3.7, with the research flags only it played (a leaf loaded here, the
damage ranking, the deepening, depth 2, the bench drop, the knock-out branch, the ally
targets, the equilibrium selection, line 1, the Baton hold, the bench oracle, the net's ends).

What an arm is, per arm, and so in whichever seat it sits:

* its leaf on an inference server (`--inference-arm` / `--baseline-inference-arm`, the other
  arm's possibly on `--baseline-inference`); `--baseline-hp-share` plays hp-share, the M-C
  origin, and one of the two is required -- a left-out leaf was silently hp-share until
  IKA-335,
* its width (`--limit` / `--baseline-limit`) and its leaf ranking (`--rank-leaf` /
  `--baseline-rank-leaf`, required) by a Q (`--rank-fill` / `--baseline-rank-fill`: a
  `q-nocover` fill; unnamed plays `q-nocover` by `data/models/q-mc6.pt`, IKA-338),
* the port executable its move-node search reads (`--rust-binary` / `--baseline-rust-binary`,
  IKA-413),
* its selection: an arm with a leaf solves the pair's selection game with THAT leaf, draws its
  four from its side of the solve, and believes the opponent's bench from the same solve.
  Solves are shared through one directory per arm (`--selection-store` /
  `--baseline-selection-store`), never mixed: the tag names the leaf and a store solved for
  another leaf stops the run. hp-share draws four of six uniformly and holds the uniform belief.

Index `i` is game `i // 2` with the tested arm at side `i % 2`; the pair and its seats come from
the game's seed, so the two seats of a game are the same teams with the arms swapped. Records
carry `gameIndex`, `seatIndex` and a provenance whose `seat` names the tested arm's side, which
is what `pokeuraou.sprt` pairs on.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# IKA-431: OpenBLAS commits a buffer per thread it may run when it loads. Set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import qrank, rustworker  # noqa: E402
from pokeuraou.benchflags import add_bench_flags, require_bench  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.payoff import HP_SHARE  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402
from pokeuraou.poolplay import PoolArm, SolvedSelections  # noqa: E402
from pokeuraou.search import (  # noqa: E402
    SHIPPED_RANK_FILL,
    parse_rank_fill,
    resolve_rank_fill,
)
from pokeuraou.selfplay import MAX_TURNS  # noqa: E402

#: The width M-C generation plays (IKA-77).
GENERATION_LIMIT = 12


def leaf_name(files: Sequence[str | Path]) -> str:
    """A model stem, `-sN` dropped, `xN` for an ensemble -- `generation_match`'s names."""
    stem = re.sub(r"-s\d+$", "", Path(files[0]).stem)
    return stem if len(files) == 1 else f"{stem}x{len(files)}"


def build_leaves(
    args: argparse.Namespace, encoder: Encoder
) -> tuple[object, object | None, str, str | None]:
    """(tested leaf, other leaf or None, their names), both on inference servers."""
    from pokeuraou.inference import RemoteValue

    value = RemoteValue(args.inference, args.inference_arm, encoder)
    baseline = None
    if args.baseline_inference_arm:
        # IKA-461: the baseline arm may be on another server (`--baseline-inference`).
        other = args.baseline_inference or args.inference
        baseline = (
            value
            if args.baseline_inference_arm == args.inference_arm and other == args.inference
            else RemoteValue(other, args.baseline_inference_arm, encoder)
        )
    # Asked of the server: a worker is told an arm's name, never what it holds.
    names = [leaf_name(value.describe())]
    names.append(leaf_name(baseline.describe()) if baseline is not None else None)
    return value, baseline, names[0], names[1]


def _install_q(
    args: argparse.Namespace, encoder: Encoder, ap: argparse.ArgumentParser
) -> dict[str, list[str]]:
    """The Qs the q rank fills rank by (IKA-274): installed, and what the records name
    them by (`qrank.record_fields`): their files as the server holds them, a named Q's
    (``q-nocover.NAME``) as ``NAME=file``, and each file's sha256 (IKA-340)."""
    models = qrank.install_from_args(
        args, encoder, (args.rank_fill, args.baseline_rank_fill), ap.error
    )
    if not models:
        return {}
    fields = qrank.record_fields(models)
    print(f"  Q: {', '.join(fields['qModel'])} (served by {args.inference})", file=sys.stderr)
    return fields


#: The inference server's arm that answers `payoff.HP_SHARE` (`inference-server --hp-share-arm`,
#: IKA-460 step 4). A game's nodes are scored by the port on the server, so hp-share is a served
#: leaf like any other.
HP_SHARE_ARM = "hp-share"


def check_other_arm(args: argparse.Namespace, error: Callable[[str], object]) -> None:
    """The other arm's leaf is named, never defaulted (IKA-335).

    Without a baseline leaf the other arm is hp-share. That used to be what leaving the
    flag out meant, and IKA-296's null control -- an arm against itself -- ran as the arm
    against hp-share because its driver left `--baseline` out. Leaving a leaf out now
    stops the run; hp-share is `--baseline-hp-share`.
    """
    named = args.baseline_inference_arm
    if args.baseline_hp_share and named:
        error(f"--baseline-hp-share contradicts the baseline leaf {named}: "
              "one of them is not what you meant")
    if args.baseline_hp_share:
        args.baseline_inference_arm = HP_SHARE_ARM
    if not args.baseline_hp_share and not named:
        error("the other arm's leaf is not named: pass --baseline-inference-arm, or "
              "--baseline-hp-share to play hp-share and mean it (IKA-335)")


def focus_pairs(
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    focus: set[int],
    field: set[int] | None,
) -> list[tuple[int, int]]:
    """The pairs of a ``focus`` team against a ``field`` team (any team when None), in order
    (`--pairs-focus`, IKA-443)."""
    return [(a, b) for a, b in pairs
            if (a in focus and (field is None or b in field))
            or (b in focus and (field is None or a in field))]


def board_pairs(args: argparse.Namespace, pool: object, error: Callable[[str], object]
                ) -> list[tuple[int, int]] | None:
    """The pairs the `--pairs-*` flags leave, or None for every pair."""
    reg = pool.reg
    pairs = None
    if args.pairs_with:
        unknown = [s for s in args.pairs_with if s not in reg.species]
        if unknown:
            error(f"--pairs-with: not species ids of {reg.meta.format_id}: {unknown}")
        holding = {t for t, team in enumerate(pool.teams)
                   if any(s.species in args.pairs_with for s in team.sets)}
        pairs = [(a, b) for a, b in pool.pairs if a in holding or b in holding]
        if not pairs:
            error(f"--pairs-with {args.pairs_with}: no team of the pool has one")
    if args.pairs_focus:
        index = {team.id: t for t, team in enumerate(pool.teams)}
        unknown = [i for i in [*args.pairs_focus, *(args.pairs_field or [])] if i not in index]
        if unknown:
            error(f"--pairs-focus / --pairs-field: not team ids of the pool: {unknown}")
        pairs = focus_pairs(pairs if pairs is not None else pool.pairs,
                            {index[i] for i in args.pairs_focus},
                            {index[i] for i in args.pairs_field} if args.pairs_field else None)
        if not pairs:
            error("--pairs-focus: no pair left")
    elif args.pairs_field:
        error("--pairs-field needs --pairs-focus")
    if args.pairs_no_mirror:
        pairs = [(a, b) for a, b in (pairs if pairs is not None else pool.pairs) if a != b]
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="data/pool/<id>.json, e.g. regmc-matchupweb")
    ap.add_argument("--inference", required=True, metavar="HOST:PORT")
    ap.add_argument("--inference-arm", default="value")
    ap.add_argument("--baseline-inference-arm", default=None,
                    help="the server's name for the other arm")
    ap.add_argument("--baseline-inference", default=None, metavar="HOST:PORT",
                    help="IKA-461: the other arm's own inference server, when it is not --inference")
    ap.add_argument("--baseline-hp-share", action="store_true",
                    help="the other arm has no leaf: hp-share, the M-C origin, and you mean "
                    "it. Leaving the other arm's leaf out used to mean this silently, and a "
                    "null control meant as an arm against itself played hp-share (IKA-335)")
    ap.add_argument("--limit", type=int, default=GENERATION_LIMIT)
    ap.add_argument("--baseline-limit", type=int, default=None, help="default: --limit")
    ap.add_argument("--rank-leaf", action="store_true",
                    help="the tested arm narrows by its leaf (what M-C generation does; required)")
    ap.add_argument("--baseline-rank-leaf", action="store_true", help="same for the other arm")
    ap.add_argument("--rank-fill", default=None,
                    help="how the tested arm's leaf ranking fills its cells: a q-nocover fill "
                    f"ranks by a Q (IKA-274). Default with --rank-leaf: {SHIPPED_RANK_FILL}, "
                    "what M-C generation plays (IKA-338)")
    ap.add_argument("--baseline-rank-fill", default=None,
                    help="same for the other arm")
    qrank.add_q_flags(ap)
    ap.add_argument("--rust-binary", type=Path, default=None,
                    help="the tested arm's move-node search reads the port executable at this "
                    "path (IKA-413: a port change measured against the port before it); the "
                    "game itself, the replacements and the selection stay with the worker's "
                    "own. Default: the worker's own")
    ap.add_argument("--baseline-rust-binary", type=Path, default=None,
                    help="same for the other arm")
    ap.add_argument("--pairs-with", nargs="+", default=None, metavar="SPECIES",
                    help="draw only the pool's pairs where either team has one of these "
                    "species ids (IKA-411: a match about a change only some teams meet)")
    ap.add_argument("--pairs-focus", nargs="+", default=None, metavar="TEAM_ID",
                    help="draw only the pairs of one of these teams against one of "
                    "--pairs-field (IKA-443: candidate teams against the pool)")
    ap.add_argument("--pairs-field", nargs="+", default=None, metavar="TEAM_ID",
                    help="with --pairs-focus: the opponents (default: every team)")
    ap.add_argument("--pairs-no-mirror", action="store_true",
                    help="leave out the pairs of a team against itself (IKA-419d)")
    add_bench_flags(ap)
    ap.add_argument("--selection-store", type=Path, default=None,
                    help="the tested arm's shared solves. Default <games-out dir>/"
                    "selection-<leaf>")
    ap.add_argument("--baseline-selection-store", type=Path, default=None)
    ap.add_argument("--explore-epsilon", type=float, default=0.0,
                    help="selection exploration in the draw and the belief alike. 0: the "
                    "pure equilibrium, as a rating wants (generation plays 0.25)")
    ap.add_argument("--explore-temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--games", type=int, default=100, help="games per seat (not in the config)")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--games-out", type=Path, default=None,
                    help="where the games go; only places the default selection stores")
    ap.add_argument("--rust-config", type=Path, required=True,
                    help="IKA-458 stage 3.7: write the Rust board worker's config for these flags to "
                    "this file (tools/match_queue.py does this once per board)")
    args = ap.parse_args(argv)
    require_bench(args)
    check_other_arm(args, ap.error)
    # Each arm's fill as generation resolves it (IKA-338): named, or q-nocover for a
    # leaf-ranked menu.
    args.rank_fill = resolve_rank_fill(args.rank_fill, args.rank_leaf)
    args.baseline_rank_fill = resolve_rank_fill(args.baseline_rank_fill, args.baseline_rank_leaf)
    for fill in (args.rank_fill, args.baseline_rank_fill):
        try:
            parse_rank_fill(fill)
        except ValueError as problem:
            ap.error(str(problem))
    refused = rustworker.board_unsupported(args)
    if refused:
        ap.error(f"the Rust worker does not play {', '.join(refused)}")

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    pairs = board_pairs(args, pool, ap.error)
    encoder = Encoder(reg)
    value, baseline, value_name, baseline_name = build_leaves(args, encoder)
    q_record = _install_q(args, encoder, ap)
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
        rust_binary=args.rust_binary,
    )
    other_limit = args.limit if args.baseline_limit is None else args.baseline_limit
    if args.baseline_hp_share:
        # The M-C origin: no leaf of its own to solve a selection with, so the selection is
        # drawn; its nodes are scored by the server's hp-share arm.
        other = PoolArm(name=HP_SHARE.name, evaluate=baseline, solver=None,
                        limit=other_limit, rank_by_leaf=args.baseline_rank_leaf,
                        rank_fill=args.baseline_rank_fill,
                        rust_binary=args.baseline_rust_binary)
    else:
        assert baseline_name is not None
        other = PoolArm(
            name=baseline_name, evaluate=baseline,
            solver=(
                tested.solver
                if baseline is value and args.baseline_selection_store is None
                else solver_for(baseline, baseline_name, args.baseline_selection_store)
            ),
            limit=other_limit, rank_by_leaf=args.baseline_rank_leaf,
            rank_fill=args.baseline_rank_fill,
            rust_binary=args.baseline_rust_binary,
        )
    arms = (tested, other)
    print(pool.summary(), file=sys.stderr)
    print(f"pool match / {reg.meta.format_id} / bench hidden / "
          f"selection eps={args.explore_epsilon}, T={args.explore_temperature}",
          file=sys.stderr)
    if pairs is not None:
        print(f"  pairs: {len(pairs)} of {len(pool.pairs)}", file=sys.stderr)
    for label, arm in (("tested arm", tested), ("other arm", other)):
        store = arm.solver.store if arm.solver is not None else None
        print(
            f"  {label}: leaf {arm.name} / width {arm.limit} / leaf ranking (fill {arm.rank_fill})"
            + (f" / port {arm.rust_binary}" if arm.rust_binary is not None else "")
            + f" / selection {arm.selection}"
            + (f" by its own leaf, store {store}" if store else "")
            + " / belief " + ("solved" if arm.solver is not None else "uniform"),
            file=sys.stderr,
        )
    # The tags name what differs between the arms, as the board's records always did.
    tags = ""
    if tested.limit != other.limit:
        tags += f"@w{tested.limit}"
    if tested.rank_fill != other.rank_fill:
        tags += f"@rankfill:{tested.rank_fill}"
    if tested.rust_binary != other.rust_binary:
        tags += "@port" if tested.rust_binary is not None else "@ownport"
    arm_label = f"{tested.name}{tags}"
    rustworker.write_config(
        rustworker.board_config(args, pool, arms, arm_label, q_record, pairs), args.rust_config
    )
    print(f"  rust board worker config -> {args.rust_config}", file=sys.stderr)


if __name__ == "__main__":
    main()
