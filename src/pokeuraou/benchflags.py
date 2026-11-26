"""The information condition on a command line: the bench is hidden, and only that.

The game this project solves hides the opponent's bench: a player sees six on the sheet
and the two that lead, and learns the other two only when they come in. Until IKA-460 a
tool could also play the open game (`--open-bench`, the opponent's four shown to the
search: a reference, nothing a real game gives). That game is gone from the code (IKA-460),
so the hidden bench is the only condition.

`--hide-bench` stays as a flag so a recorded command that carried it still runs; it is the
default and changes nothing. `--open-bench` stops with the sentence that says why, rather
than being an unknown argument, so an old command fails with its reason.
"""

from __future__ import annotations

import argparse

#: What a tool says when it was given `--open-bench`.
OPEN_REMOVED = (
    "--open-bench: the open game (the search shown the opponent's four) was removed in "
    "IKA-460. The bench is hidden in every game; drop the flag (`--hide-bench` is the "
    "default and still accepted)."
)


class _OpenBenchRemoved(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):  # noqa: ANN001, ANN204
        parser.error(OPEN_REMOVED)


def add_bench_flags(
    ap: argparse.ArgumentParser,
    *,
    hidden_help: str | None = None,
    open_help: str | None = None,  # noqa: ARG001 - the callers' older keyword, now unused
) -> None:
    """`--hide-bench` (the default, kept for recorded commands) and a stop for `--open-bench`."""
    ap.add_argument(
        "--hide-bench",
        dest="hide_bench",
        action="store_const",
        const=True,
        default=True,
        help=hidden_help
        or "neither side's search is shown the other's unplayed bench: each solves over "
        "every four the opponent's sheet still allows. The only condition since IKA-460; "
        "this flag changes nothing.",
    )
    ap.add_argument(
        "--open-bench",
        nargs=0,
        action=_OpenBenchRemoved,
        help=argparse.SUPPRESS,
    )


def require_bench(args: argparse.Namespace) -> bool:  # noqa: ARG001
    """The bench is always hidden."""
    return True


def bench_argv(hide_bench: bool) -> list[str]:  # noqa: ARG001
    """The flag that passes this condition on to a child process, always explicit."""
    return ["--hide-bench"]


__all__ = ["OPEN_REMOVED", "add_bench_flags", "bench_argv", "require_bench"]
