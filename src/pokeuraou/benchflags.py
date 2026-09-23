"""The information condition on a command line: hidden bench, or open by name only.

The game this project solves hides the opponent's bench: a player sees six on the sheet
and the two that lead, and learns the other two only when they come in. The code grew the
other way round -- the open game first, the hidden bench added later as `--hide-bench` --
so for months the unmarked command played the easier, wrong game, and every run that was
meant to be the real one had to remember a flag (IKA-116, IKA-123).

Neither condition is a default any more. A tool that can play both takes exactly one of
`--hide-bench` (what ships) or `--open-bench` (the opponent's four shown to the search:
a reference, nothing a real game gives), and stops when given neither. It stops rather than
choosing hidden because a command recorded before IKA-123 without either flag meant OPEN,
and silently replaying it hidden would hand back a different measurement under the old
command line. So an old recorded command either keeps its meaning -- it carried
`--hide-bench` -- or fails here with the sentence that says what to add.

The same rule is in `play_game`: without `sheets` it wants `open_information=True`.
"""

from __future__ import annotations

import argparse
import sys

#: What a tool says when it was given neither flag.
MISSING = (
    "say which game to play: --hide-bench (the opponent's unplayed bench is hidden from "
    "the search, which is what ships) or --open-bench (the search is shown the opponent's "
    "four -- a reference only). Neither is a default since IKA-123. A command recorded "
    "before then without either flag played OPEN: replay it with --open-bench."
)

#: Printed by a tool that has no hidden-bench path at all, beside its results.
OPEN_REFERENCE = (
    "  information: OPEN (reference) -- this tool cannot hide the bench, so its search is "
    "shown the opponent's four. A number from it is about the open game, not the one that "
    "ships (IKA-123)."
)


def add_bench_flags(
    ap: argparse.ArgumentParser,
    *,
    hidden_help: str | None = None,
    open_help: str | None = None,
) -> None:
    """`--hide-bench` / `--open-bench`, one of them required, both writing `hide_bench`.

    Not argparse's `required=True` on the group: that message cannot say that an old
    command without either flag meant open. `require_bench` says it after parsing.
    """
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--hide-bench",
        dest="hide_bench",
        action="store_const",
        const=True,
        default=None,
        help=hidden_help
        or "neither side's search is shown the other's unplayed bench: each solves over "
        "every four the opponent's sheet still allows. The condition that ships. One of "
        "this or --open-bench is required.",
    )
    group.add_argument(
        "--open-bench",
        dest="hide_bench",
        action="store_const",
        const=False,
        help=open_help
        or "the search is handed the opponent's whole four -- the open game, a reference "
        "only. What omitting both flags meant before IKA-123; now it has to be asked for.",
    )


def require_bench(args: argparse.Namespace) -> bool:
    """`args.hide_bench` as a bool, or a stop naming both flags when it was not given."""
    if args.hide_bench is None:
        raise SystemExit(MISSING)
    return bool(args.hide_bench)


def bench_argv(hide_bench: bool) -> list[str]:
    """The flag that passes this condition on to a child process, always explicit."""
    return ["--hide-bench"] if hide_bench else ["--open-bench"]


def say_open_reference() -> None:
    """For a tool with no hidden path: say, on stderr, that its numbers are reference."""
    print(OPEN_REFERENCE, file=sys.stderr, flush=True)


__all__ = [
    "MISSING",
    "OPEN_REFERENCE",
    "add_bench_flags",
    "bench_argv",
    "require_bench",
    "say_open_reference",
]
