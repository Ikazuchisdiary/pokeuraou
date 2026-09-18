"""Plays one generation's search against another's, which is the only fair comparison.

Validation AUC cannot compare generations. Each one plays differently, so each generates a
different distribution of positions, and a higher AUC may only mean "this generation
created positions that are easier to judge". Winning games against the previous generation
is not confounded that way.

Both sides get the same teams, the same selections and the same seed; only the leaf
evaluation differs. Sides are swapped halfway, because a residual seat advantage would
otherwise be credited to whichever generation sat in the better seat -- the speed-tie bug
was exactly that, and it was worth 9 points in a mirror.

    uv run --group learn python tools/generation_match.py --value data/models/value-worlds.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.policy import load_policy
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.provenance import open_games, provenance, write_game
from pokeuraou.selection_book import SelectionBook, draw_across
from pokeuraou.selfplay import play_game
from pokeuraou.standings import find_cached_standings, load_standings, sample_standings_team
from pokeuraou.teams import all_selections, load_roster
from pokeuraou.workqueue import WorkClient

# `pokeuraou.value` imports torch, so it is imported where it is used rather than here.
# A worker scoring on an inference server needs neither, and torch is 816 MB of the 863
# such a worker was measured at against the 213 it weighs without.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--value",
        type=Path,
        nargs="+",
        default=[Path("data/models/value-worlds.pt")],
        help="one model, or several to average as one leaf. Several because a training "
        "run moves more than the settings under test do: six seeds of the same data and "
        "configuration spanned 0.024 of held-out AUC where the configurations differ by "
        "0.004. Two single runs measure seed luck; two ensembles of three measure the "
        "change.",
    )
    ap.add_argument("--games", type=int, default=200, help="games per seat, so twice this in total")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument(
        "--baseline",
        type=Path,
        nargs="+",
        default=None,
        help="the older generation's *model*. Given, the match is model against model, "
        "which is what every generation after the second needs -- the first comparison "
        "was a value function against hp-share and that is all --objective can express. "
        "`play_game` already takes a leaf per side.",
    )
    ap.add_argument("--objective", default="hp-share", help="the leaf the older generation used")
    ap.add_argument(
        "--depth",
        type=int,
        default=1,
        help="search depth for the arm under test. With --baseline pointing at the same "
        "model, --depth 2 --baseline-depth 1 makes this a pure depth comparison: same "
        "leaf, same width, same teams, same seed, one side looking a ply further.",
    )
    ap.add_argument("--baseline-depth", type=int, default=1, help="depth for the other arm")
    ap.add_argument(
        "--baseline-limit",
        type=int,
        default=None,
        help="candidate width for the other arm; defaults to --limit. With the same model "
        "on both sides this makes the match a pure width comparison, which is how "
        "'does a wider menu actually win' gets answered without building anything.",
    )
    ap.add_argument(
        "--rank-leaf",
        action="store_true",
        help="the arm under test ranks candidates with the leaf instead of the damage "
        "score. With the same model and width on both sides this is a pure ranking "
        "comparison.",
    )
    ap.add_argument("--baseline-rank-leaf", action="store_true", help="same for the other arm")
    ap.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="the arm under test orders its candidates with a learned policy "
        "(tools/policy_train.py) instead of with the leaf or the damage score. It "
        "supersedes --rank-leaf for that arm, because both answer the same question. "
        "Measured off the board it reaches the leaf ordering's width-24 regret at width "
        "16 for half the time; whether that converts into games is what this flag exists "
        "to find out, and nothing here assumes it does.",
    )
    ap.add_argument(
        "--baseline-policy", type=Path, default=None, help="same for the other arm"
    )
    ap.add_argument(
        "--policy-value",
        type=Path,
        default=None,
        help="the value function whose frozen `side_vectors` the policy was trained "
        "against. Required for a policy that uses a position representation, and not "
        "inferrable: every model here has the same width, so the wrong one runs at full "
        "speed and orders by numbers that mean something else. It is loaded separately "
        "from the arms' leaves, which are free to be anything.",
    )
    ap.add_argument(
        "--policy-device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="the CPU is 16x quicker on a menu-sized batch (0.14 ms against 2.25), "
        "because a batch of thirty-five rows never reaches the card's arithmetic.",
    )
    ap.add_argument(
        "--solve-sparsely",
        action="store_true",
        help="the arm under test proves the equilibrium from about a fifth of the matrix "
        "instead of filling all of it. Both settle on an equilibrium of the same game "
        "(exploitability 7.7e-08), so this is not a strength setting in theory -- what "
        "differs is which vertex of a degenerate optimum gets played, and a maximin "
        "strategy only guarantees the value. Against an opponent who is not playing the "
        "equilibrium, two equilibria can take different amounts, and that is measurable.",
    )
    ap.add_argument(
        "--baseline-solve-sparsely", action="store_true", help="same for the other arm"
    )
    ap.add_argument(
        "--selection-book",
        type=Path,
        default=None,
        help="draw both sides' four of six from a cached selection equilibrium instead of "
        "uniformly. Every rating measured so far used a uniform draw on both sides, which "
        "is worth -22.1 points against the advice, so an agent measured that way is an "
        "agent playing a selection nobody would play. Given, the book's own opponent "
        "sheets and spread classes are used, so the arms face what the book was solved "
        "against. Exploration is deliberately *not* applied: epsilon exists to keep "
        "generation's coverage wide, and a rating wants the strategy rather than the "
        "training noise.",
    )
    ap.add_argument(
        "--baseline-selection-book",
        type=Path,
        default=None,
        help="a second book, so each arm draws its own selection. An agent here is a "
        "model, a search *and* a book -- the book is worth +141 Elo and every rating so "
        "far was measured with a uniform draw on both sides, which is an agent nobody "
        "runs. This costs the pairing: with one book the two arms play the same teams "
        "and the same selections and differ only in the leaf, and with two they play "
        "their own strategies against the same field. That is the right comparison for a "
        "rating and the wrong one for isolating a model, so both exist.",
    )
    ap.add_argument(
        "--hide-bench",
        action="store_true",
        help="neither side's search is shown the other's unplayed bench, which is the "
        "condition a model trained on hidden-bench data is meant to be used in. Comparing "
        "such a model with an open-information one *without* this measures which model "
        "suits the open game, not which is better: the evaluation has to be the condition "
        "the answer is for. Games say so in their provenance and the rating keeps them on "
        "their own scale.",
    )
    ap.add_argument(
        "--baseline-uniform-selection",
        action="store_true",
        help="the other arm draws its four uniformly while this one uses its book. "
        "Without it `--selection-book` applies to both arms, which is right for holding "
        "the selection fixed and cannot express the comparison the book itself is for: "
        "the +141 Elo attributed to the advice was measured on single models against an "
        "older book, and on the current floor it is unmeasured.",
    )
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument(
        "--queue",
        default=None,
        help="address of a work queue to take (seat, game) indices from, instead of "
        "playing a fixed block. A block ends when the unluckiest worker does, and a "
        "game's cost varies eightfold: generation 8 left 25%% of the machine idle that "
        "way. Games are seeded from their index so the work is the same whoever plays it.",
    )
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="append one JSON line per seat as it completes, so a stopped run is not lost "
        "and several runs can be pooled by reading the files.",
    )
    ap.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="print a progress line every N games. Redirected stdout is fully buffered, so "
        "these are flushed explicitly or they are invisible until the process exits.",
    )
    ap.add_argument(
        "--games-out",
        type=Path,
        default=None,
        help="also append every game played, as self-play-shaped JSONL with a provenance "
        "block. These carry real outcomes and are training data that has already been "
        "paid for; the block records the leaf and width *per side*, because the two are "
        "deliberately mismatched here and a dataset must be able to say so.",
    )
    ap.add_argument(
        "--inference",
        default=None,
        metavar="HOST:PORT",
        help="score both arms' leaves on a shared inference server instead of loading "
        "the models here. A match worker holds two leaves, which is why it weighs 4.0 GB "
        "of commit and 1.5 GB of VRAM against a generation worker's one -- and why eight "
        "of them did not fit on this machine. Through the server a worker is 542 MB and "
        "imports no torch. Answers are unchanged: the server runs each request as it "
        "arrives and never merges one worker's batch with another's, so the batch length "
        "is the one the worker asked for and cuda returns the same numbers it would have.",
    )
    ap.add_argument(
        "--inference-arm",
        default="value",
        help="the server's name for the arm under test",
    )
    ap.add_argument(
        "--baseline-inference-arm",
        default=None,
        help="the server's name for the other arm; omit for a match against --objective",
    )
    # Resolved after parsing, not here: asking torch whether there is a card imports
    # torch, and a worker scoring on a server has no use for it. The help above promises
    # such a worker holds none, and this line was quietly breaking that promise -- 816 MB
    # of it, against the 213 MB such a worker otherwise weighs.
    ap.add_argument("--device", default=None, help="cuda or cpu; default is cuda if present")
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="threads per worker. 1 is right for a parallel run: N processes each spawning "
        "a pool on the same cores is slower than one process. Raise it for a single run.",
    )
    args = ap.parse_args()
    if args.inference is None:
        # Only the arm that scores here needs torch at all.
        import torch

        torch.set_num_threads(args.torch_threads)
        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device is None:
        args.device = "cuda"

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    book = SelectionBook.read(args.selection_book) if args.selection_book else None
    other_book = (
        SelectionBook.read(args.baseline_selection_book)
        if args.baseline_selection_book
        else book
    )
    if args.baseline_selection_book:
        print(f"selection: each arm draws its own ({args.selection_book.name} against "
              f"{args.baseline_selection_book.name})", file=sys.stderr)
    if book is not None:
        print(f"selection: {args.selection_book.name} ({len(book)} teams), no exploration",
              file=sys.stderr)

    # One name for the whole leaf. An ensemble's name has to say how many members it has,
    # because "three seeds averaged" and "one seed" are different agents on the rating
    # scale and the difference between them is larger than most things it measures.
    def leaf_name(paths: Sequence[Path | str]) -> str:
        stem = Path(paths[0]).stem
        stem = re.sub(r"-s\d+$", "", stem)
        return stem if len(paths) == 1 else f"{stem}x{len(paths)}"

    encoder = Encoder(reg)
    objective = OBJECTIVES[args.objective]
    if args.inference is not None:
        # Both arms live on the server, named. A match is the case that needs names: it
        # holds two leaves, and holding two is what made a match worker twice the weight
        # of a generation worker and put eight of them over the machine.
        from pokeuraou.inference import RemoteValue

        meta = base_meta = {}
        value = RemoteValue(args.inference, args.inference_arm, encoder)
        baseline = (
            RemoteValue(args.inference, args.baseline_inference_arm, encoder)
            if args.baseline_inference_arm
            else None
        )
        # The names come back from the server, not from this command line. A worker is
        # told which arm to play, never what that arm holds, and the name it records is
        # what a rating is fitted from.
        value_files = value.describe()
        baseline_files = baseline.describe() if baseline is not None else []
        print(
            f"leaf: {args.inference_arm}={leaf_name(value_files)}"
            + (
                f" vs {args.baseline_inference_arm}={leaf_name(baseline_files)}"
                if baseline is not None
                else ""
            )
            + f" on {args.inference} (this worker holds no model)",
            file=sys.stderr,
        )
    else:
        value_files = list(args.value)
        baseline_files = list(args.baseline) if args.baseline else []
        import torch

        from pokeuraou.value import BatchedValue, load_ensemble

        nets, metas = load_ensemble(args.value, encoder)
        meta = metas[0]
        device = torch.device(args.device)
        value = BatchedValue([n.to(device) for n in nets], encoder, device=device)
        baseline = None
        if args.baseline is not None:
            missing = [p for p in args.baseline if not p.exists()]
            if missing:
                raise SystemExit(f"no baseline model at {missing}")
            base_nets, base_metas = load_ensemble(args.baseline, encoder)
            base_meta = base_metas[0]
            baseline = BatchedValue(
                [n.to(device) for n in base_nets], encoder, device=device
            )

    def book_matches_leaf(files, loaded, which: str) -> None:  # noqa: ANN001
        """Whether this arm's book was solved from this arm's leaf, said out loud.

        A book is the equilibrium of the selection game as one model's value function
        defines it, so an arm drawing from another model's book selects by an evaluation
        it does not play with. That is sometimes the right configuration -- a bridge to an
        already recorded corpus has to be spelled like the corpus -- so this reports
        rather than refuses. It reports in BOTH directions, because a line that appears
        only when something is wrong is a line nobody has read when it matters.

        The book header stores the model files it was solved from, so the comparison is
        exact and catches the case a name would miss: an ensemble leaf whose book was
        solved from one of its members. Those have the same stem and are different agents.
        """
        if loaded is None or not files:
            return
        want = "+".join(getattr(p, "name", str(p)) for p in files)
        if loaded.model == want:
            print(f"  {which}: leaf and book agree ({want})", file=sys.stderr)
        else:
            print(
                f"  !! {which}: the leaf is {want}, and its book was solved from "
                f"{loaded.model or '(unrecorded)'}.\n"
                f"     That arm selects by one evaluation and plays by another. "
                f"Deliberate for a bridge to a\n"
                f"     recorded corpus; a mistake anywhere else.",
                file=sys.stderr,
            )

    book_matches_leaf(value_files, book, "tested arm")
    book_matches_leaf(baseline_files, other_book, "other arm")

    # The policy's position representation comes from its own value function, not from
    # either arm's leaf: an arm may be an ensemble, or a generation the policy never saw,
    # and the representation is a property of the policy rather than of the match.
    policy_net = None
    policy_name = ""
    if args.policy or args.baseline_policy:
        if args.policy_value is None:
            raise SystemExit(
                "--policy needs --policy-value: the position representation belongs to "
                "one value function and nothing in the file says which."
            )
        import torch

        from pokeuraou.value import load_ensemble

        policy_nets, _ = load_ensemble([args.policy_value], encoder)
        policy_net = policy_nets[0].to(torch.device(args.device))
        policy_name = args.policy_value.stem

    def load_ranker(path: Path | None):
        if path is None:
            return None
        return load_policy(path, reg, args.policy_device, policy_net, policy_name)

    policies = (load_ranker(args.policy), load_ranker(args.baseline_policy))
    if policy_net is not None:
        print(
            f"policy: {args.policy.stem if args.policy else '-'} vs "
            f"{args.baseline_policy.stem if args.baseline_policy else '-'} "
            f"({args.policy_device}, position from {policy_name})",
            file=sys.stderr,
        )

    print(
        f"new leaf: {leaf_name(value_files)} (trained on {meta.get('games', '?')} games, "
        f"val AUC {meta.get('val_auc', float('nan')):.4f})",
        file=sys.stderr,
    )
    if baseline is not None:
        print(
            f"old leaf: {leaf_name(baseline_files)} (trained on "
            f"{base_meta.get('games', '?')} games, val AUC "
            f"{base_meta.get('val_auc', float('nan')):.4f})",
            file=sys.stderr,
        )
        print(
            "  同じ探索・同じチーム・同じ選出・同じ乱数で、葉だけが違います。"
            "検証 AUC は世代間で比較できないので、勝率だけが判定です",
            file=sys.stderr,
        )
    else:
        print(f"old leaf: {args.objective}（評価軸）", file=sys.stderr)

    # Seat A: the value function is side 0. Seat B: it is side 1. Both seats use our six
    # against the field, so a seat advantage cancels when the two are combined.
    print(f"\n  {'seat':>30}  {'games':>6}  {'new win':>8}  {'95%':>6}  {'s/game':>7}")
    wins = played = 0
    book_misses = 0
    # Per arm label: the ordered fours that arm drew for itself, and the ones its
    # opponent drew. A book match is supposed to differ from a uniform one in the first
    # of those and not the second, and the first version of `--baseline-uniform-selection`
    # differed in both -- 330 distinct foe fours against 799 -- with nothing in the output
    # saying so. Counted here because the number that would have exposed it is two lines.
    drawn_ours: dict[str, set[tuple[int, ...]]] = {}
    drawn_theirs: dict[str, set[tuple[int, ...]]] = {}
    games_file = open_games(args.games_out)
    new_name = leaf_name(value_files)
    old_name = leaf_name(baseline_files) if baseline_files else args.objective
    # The seat label names the model, not "gen2". It is stamped into every recorded
    # game's provenance, and a reader of that dataset a month from now has no way to know
    # which generation "gen2" meant on the day the match ran -- the `leaves` pair is
    # authoritative, but a label that contradicts it is worse than no label.
    other_limit = args.limit if args.baseline_limit is None else args.baseline_limit
    # Both arms draw from the same book, so it is not what distinguishes them -- but it is
    # part of what each *is*, and a rating that cannot tell a book-selected agent from a
    # uniform one pools two different strengths under one name.
    selection_label = args.selection_book.stem if args.selection_book else "uniform"
    other_label = (
        args.baseline_selection_book.stem
        if args.baseline_selection_book
        else selection_label
    )

    # What each side's ordering is *called*, which is what a rating is fitted from. A
    # policy names itself: two policies are two agents, and "policy" alone would pool them.
    # It is one name per arm rather than a flag per mechanism, because the three orderings
    # are alternatives -- an arm ranking by policy is not also ranking by damage, and a
    # label built by adding a tag per flag said exactly that.
    def ranking_name(path: Path | None, leaf_ranked: bool) -> str:
        if path is not None:
            return f"policy:{path.stem}"
        return "leaf" if leaf_ranked else "damage"

    ranking_names = (
        ranking_name(args.policy, args.rank_leaf),
        ranking_name(args.baseline_policy, args.baseline_rank_leaf),
    )
    tags = ""
    if args.depth != args.baseline_depth:
        tags += f"@d{args.depth}"
    if other_limit != args.limit:
        tags += f"@w{args.limit}"
    if ranking_names[0] != ranking_names[1]:
        # The old spellings for the two orderings that already appear in recorded seats.
        tags += {"leaf": "@leafrank", "damage": "@damagerank"}.get(
            ranking_names[0], "@" + ranking_names[0].replace("policy:", "")
        )
    if args.solve_sparsely != args.baseline_solve_sparsely:
        tags += "@sparse" if args.solve_sparsely else "@fullmatrix"
    arm = f"{new_name}{tags}" if tags else new_name
    seats = (
        (f"{arm} = side 0", (value, baseline), (args.depth, args.baseline_depth),
         (args.limit, other_limit), (args.rank_leaf, args.baseline_rank_leaf),
         (args.solve_sparsely, args.baseline_solve_sparsely),
         policies, ranking_names),
        (f"{arm} = side 1", (baseline, value), (args.baseline_depth, args.depth),
         (other_limit, args.limit), (args.baseline_rank_leaf, args.rank_leaf),
         (args.baseline_solve_sparsely, args.solve_sparsely),
         policies[::-1], ranking_names[::-1]),
    )
    # Per-seat accumulators, indexed the same way as `seats`, because with a queue the two
    # seats are interleaved rather than run one after the other.
    tally = [[0, 0, 0, 0.0] for _ in seats]  # wins, played, unfinished, seconds
    client = WorkClient(args.queue) if args.queue else None
    if client is not None:
        print(f"queue: {args.queue}", file=sys.stderr)

    def work() -> Iterator[int]:
        """The (seat, game) pairs this process should play, as one index each.

        A queue hands out work in a race, so an index has to name the whole job: seat and
        game together. Draining game indices per seat would let the first seat empty the
        queue and leave the second with nothing, and the pairing between the two seats --
        which is the entire reason for playing both -- would be gone.
        """
        if client is None:
            yield from range(len(seats) * args.games)
            return
        while True:
            index = client.take()
            if index is None:
                return
            yield index

    done = 0
    overall_started = time.perf_counter()
    for index in work():
        which = index % len(seats)
        game_index = index // len(seats)
        seat, leaves, depths, limits, ranks, sparse, rankers, ranknames = seats[which]
        side_leaves = (
            (new_name, old_name) if leaves[0] is value else (old_name, new_name)
        )
        # Seeded from the index, not streamed through the seat. A queue makes the order a
        # race, and an RNG streamed through a block would make game seventeen whatever the
        # sixteen before it left behind -- different every run, and different between the
        # two seats that are supposed to play the same game.
        rng = np.random.default_rng([args.seed, game_index])
        started = time.perf_counter()
        if True:
            done += 1
            if done % args.report_every == 0:
                print(
                    f"    {done} played, "
                    f"{(time.perf_counter() - overall_started) / done:.2f} s/game",
                    flush=True,
                )
            team = pool[int(rng.integers(len(pool)))]
            # One book per SIDE, not one per game.
            #
            # This was `seat_book = book if leaves[0] is value else other_book`, with a
            # comment saying the draw decides *our* four and side 0 is always our six.
            # True as far as it goes, and it left the other side's four coming out of the
            # same entry -- so `--baseline-selection-book` only ever swapped which single
            # book governed BOTH arms, and only in the half of the games where the
            # baseline sat at side 0. The provenance said each arm drew its own
            # throughout, and so did the line this tool prints when the flag is passed.
            #
            # What showed it: two matches of the same pair, one with own books and one
            # with a shared book, returned the identical 380/848 in the seat they shared.
            # The seat's numbers cannot be identical if the opponent's selection came
            # from a different book in the two runs.
            # `--baseline-uniform-selection` means the baseline arm HAS NO BOOK, so the
            # rule travels with the arm instead of with the seat. It used to replace our
            # pick, and side 0 is always our roster, so the arm drew uniformly only where
            # it sat at side 0 and took the book's column strategy in the other seat --
            # while the label said "uniform" for both.
            tested_book = book
            baseline_book = None if args.baseline_uniform_selection else other_book
            side0_book = tested_book if leaves[0] is value else baseline_book
            side1_book = baseline_book if leaves[0] is value else tested_book
            # The other arm draws *our* four uniformly, and nothing else changes: the
            # opponent's six and the opponent's four still come from the book, so both
            # arms face the same field and the only difference is the draw being priced.
            #
            # The first version set `seat_book = None` here instead, which also moved the
            # opponent -- in that arm the foe drew uniformly too. So the match compared
            # `book/book` against `uniform/uniform` and reported +0.3, next to the +2.1
            # that pair was already known to be worth, while the arm it claimed to be
            # measuring is worth +18.9. Nothing in the record said which arm had run: the
            # provenance labels are written from the options, not from the draw.
            # One label per *arm*, then ordered by seat. Deriving them per seat is how
            # the first version got it wrong.
            tested_label = selection_label if book is not None else "uniform"
            if args.baseline_uniform_selection:
                # Plain "uniform", because an agent is its model together with ITS OWN
                # book and the opponent's book is the opponent's identity.
                #
                # This said `uniform-against-<their book>` for exactly as long as it took
                # to fit a rating and look at the components. The argument for it was that
                # a uniform arm facing a book arm has the harder field, so pooling would
                # charge that difficulty to it as weakness -- but Bradley-Terry subtracts
                # the opponent's rating, which is what "harder field" means here, so the
                # bias it guarded against does not exist. What it did instead was make the
                # book corpus unbridgeable: every match that put a book arm against a
                # uniform one renamed the uniform one after the book, so no edge could
                # ever reach the anchored component, and ~13,000 games had no zero.
                #
                # `agent_name` folds the old spelling back, so the records already written
                # are joined without replaying them.
                others_label = "uniform"
            elif other_book is None:
                others_label = "uniform"
            else:
                others_label = other_label
            seat_labels = (
                (tested_label, others_label)
                if leaves[0] is value
                else (others_label, tested_label)
            )
            entry0 = side0_book.get(team) if side0_book is not None else None
            entry1 = side1_book.get(team) if side1_book is not None else None
            # A side with no book draws uniformly; a side whose book does not hold this
            # team is a miss, and the whole game falls back. The two are different and
            # were one condition before.
            missing = (side0_book is not None and entry0 is None) or (
                side1_book is not None and entry1 is None
            )
            if not missing and (entry0 is not None or entry1 is not None):
                # epsilon 0: the rating asks what the strategy is worth, and exploration
                # is a property of generation rather than of the agent.
                #
                # `draw_across` rather than `entry.draw` even when both sides read one
                # book, so that one book, two books and a bookless arm are the same code
                # path. A branch taken only in the unusual case is a branch nobody's runs
                # exercise. Our pick comes off the stream before the class, so the arm
                # drawing uniformly cannot see the opponent's private type even through
                # its position in the stream.
                drawn = draw_across(entry0, entry1, rng, epsilon=0.0, temperature=1.0)
                foe_six = list(drawn.foe_six)
                own_pick = drawn.our_pick
                foe_pick = drawn.foe_pick
            else:
                foe_six = sample_standings_team(rng, reg, prior, team)
                own_pick = selections[int(rng.integers(len(selections)))]
                foe_pick = selections[int(rng.integers(len(selections)))]
                if side0_book is not None or side1_book is not None:
                    book_misses += 1
            drawn_ours.setdefault(seat_labels[0], set()).add(tuple(own_pick))
            drawn_theirs.setdefault(seat_labels[0], set()).add(tuple(foe_pick))
            record = play_game(
                reg,
                rng,
                [roster.sets[i] for i in own_pick],
                [foe_six[j] for j in foe_pick],
                seat,
                objective=objective,
                search_limit=limits,
                max_turns=args.max_turns,
                evaluate=leaves,
                depth=depths,
                rank_by_leaf=ranks,
                policy=rankers,
                solve_sparsely=sparse,
                # Our six is the roster; theirs is the sheet the book drew them from, or
                # the standings team when it did not. Both are public in Champions, and
                # both are what makes the four uncertain rather than unknown.
                sheets=(list(roster.sets), list(foe_six)) if args.hide_bench else None,
                # Without this the record keeps `selectionSource: "uniform"` and an empty
                # ownPick whatever the book did, and every rating row since generation 10
                # says a uniform draw for games the book actually chose. `provenance.books`
                # was right the whole time, so the two halves of the same file disagreed --
                # which the comment beside `books=` says is worse than no record at all,
                # because the rating is fitted from it.
                selection=(
                    [entry.species for entry in roster.sets],
                    [entry.species for entry in foe_six],
                    tuple(own_pick),
                    tuple(foe_pick),
                ),
            )
            record.selection_source = (
                "book" if (entry0 is not None or entry1 is not None) else "uniform"
            )
            if record.outcome is None:
                tally[which][2] += 1
                if client is not None:
                    client.finish(index)
                continue
            write_game(
                games_file,
                record,
                objective=f"value:{leaf_name(value_files)}",
                search_limit=args.limit,
                # The pair, written down. Index `i` is game `i // 2` in seat `i % 2` and
                # every game seeds from its own index, so `gameIndex` names the two games
                # that are the same matchup with the arms swapped. Nothing could find
                # them before, and every interval on a queued match was computed as if
                # 2N paired games were 2N independent ones.
                extra={"gameIndex": game_index, "seatIndex": which},
                source=provenance(
                    "generation-match",
                    seat=seat,
                    leaves=side_leaves,
                    limits=limits,
                    depths=depths,
                    rankings=ranknames,
                    solvers=tuple("sparse" if x else "full" for x in sparse),
                    # What each side's draw actually came from, not what the command
                    # line asked for. The first version built this from the labels and
                    # recorded `book, book` for a match where one arm was drawing
                    # uniformly -- a record that disagrees with the game is worse than no
                    # record, because a rating is fitted from it.
                    books=seat_labels,
                    information=(
                        ("hidden-bench", "hidden-bench")
                        if args.hide_bench
                        else ("open", "open")
                    ),
                    note=(
                        f"search depth {depths[0]} vs {depths[1]} by side"
                        if depths[0] != depths[1]
                        else ""
                    ),
                ),
            )
            tally[which][1] += 1
            # `outcome` is side 0's result, so flip it when the value function sits at 1.
            new_won = record.outcome > 0.5 if leaves[0] is value else record.outcome < 0.5
            tally[which][0] += int(new_won)
            tally[which][3] += time.perf_counter() - started
            if client is not None:
                client.finish(index)

    if client is not None:
        client.close()
    # A book that holds none of the teams played is a book that did nothing, and the
    # match would otherwise finish quietly and report a difference of zero as a result
    # about the book rather than about the lookup. It was counted and never printed.
    for label in sorted(drawn_ours):
        print(
            f"  selection as {label}: {len(drawn_ours[label])} distinct fours drawn for "
            f"us, {len(drawn_theirs[label])} for the opponent",
            file=sys.stderr,
        )
    if book_misses:
        print(
            f"  selection book: {book_misses:,} of {args.games:,} games found no entry "
            f"for their team and drew uniformly instead",
            file=sys.stderr,
        )
    # What the worker saw, so the server's own report can be subtracted from it. The gap
    # between `waited` here and (lock wait + lock hold) there is the transport: the socket,
    # the JSON, and the server's own parsing before it reaches the model.
    for arm in (value, baseline):
        if arm is not None and hasattr(arm, "calls") and arm.calls:
            print(
                f"  leaf traffic: {arm.calls:,} requests, per call "
                f"{1000 * arm.copied / arm.calls:.2f} ms filling the buffer, "
                f"{1000 * arm.waited / arm.calls:.2f} ms awaiting the reply",
                file=sys.stderr,
            )
    for which, (seat, _leaves, _d, _l, _r, _s, _p, _n) in enumerate(seats):
        seat_wins, seat_played, unfinished, elapsed = tally[which]
        rate = seat_wins / seat_played if seat_played else float("nan")
        half = (
            1.96 * (rate * (1 - rate) / seat_played) ** 0.5
            if seat_played
            else float("nan")
        )
        print(
            f"  {seat:>30}  {seat_played:>6}  {rate * 100:>7.1f}%  +-{half * 100:.1f}  "
            f"{elapsed / max(seat_played, 1):>7.2f}   (打ち切り {unfinished})",
            flush=True,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "seat": seat,
                            "seed": args.seed,
                            "model": leaf_name(value_files),
                            "baseline": (
                                leaf_name(baseline_files) if baseline_files else args.objective
                            ),
                            "objective": args.objective,
                            "limit": args.limit,
                            "depth": args.depth,
                            "baselineDepth": args.baseline_depth,
                            "played": seat_played,
                            "gen2_wins": seat_wins,
                            "unfinished": unfinished,
                            "seconds": elapsed,
                        }
                    )
                    + "\n"
                )
        wins += seat_wins
        played += seat_played

    rate = wins / played if played else float("nan")
    half = 1.96 * (rate * (1 - rate) / played) ** 0.5 if played else float("nan")
    print(
        f"\n  両席あわせて {played} ゲーム: {arm} の勝率 {rate * 100:.1f}% +-{half * 100:.1f}",
        flush=True,
    )
    if rate - half > 0.5:
        print(f"  → {arm} が有意に強い。次世代のデータ生成に使える。")
    elif rate + half < 0.5:
        print(
            f"  → {arm} が有意に弱い。この設定は採用しない。"
            "生成する前に原因を出すべき。"
        )
    else:
        print(
            "  → 差は有意でない。区間の幅が "
            f"{half * 200:.1f} ポイントなので、判定するにはゲーム数を増やす必要がある。"
        )


if __name__ == "__main__":
    main()
