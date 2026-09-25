"""What the suite skipped, and whether it was allowed to.

A green suite is not the claim CI makes here. The claim is that a named set of checks
*ran*, and a skip is how a check stops running without anyone noticing: `test_rust_node`
compares the Rust node against the Python one cell for cell, and on a machine with no
binary it is nine dots that mean nothing. The same is true of every oracle test -- the
differential against Showdown itself -- and of every test that needs the usage data.

So the workflow builds those and this refuses the run if they skipped anyway. Two failure
modes, both of which have to be caught:

- a skip whose *reason* names something CI supplies. The build step passed and the test
  skipped regardless, which means the test looks somewhere the build did not write.
- a skip whose reason is not in this file at all. That is a new kind of skip, and the
  point of the exercise is that nobody decides silently that it is fine.

What is *not* pinned is how often a data-shaped skip fires ("this position offers no
priority move"). Those depend on the contents of a fetched file rather than on the
machine, and pinning them would turn a usage-stats refresh into a red build for no reason.
They are counted and printed instead.

    uv run python tools/ci_skip_audit.py --junit reports/pytest.xml
    uv run python tools/ci_skip_audit.py --junit reports/pytest.xml --absent standings=9

``--absent NAME=N`` is the declaration that this machine does not have NAME, and that
exactly N tests may say so. It is spelt in the workflow rather than defaulted here, so the
file that decides what CI provides is also the file that lists what it does not.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Class:
    """One reason a test can skip, and how to tell whether it was entitled to."""

    name: str
    pattern: re.Pattern[str]
    why: str
    #: Environment classes are about the machine: it either built the thing or it did not,
    #: and CI is expected to have built it. Non-environment classes are about the contents
    #: of the data, which no build step can guarantee.
    environment: bool = True
    #: What the machine would have to hold for this class not to fire. Checked when the
    #: class is required, so that "no skips" cannot be satisfied by a suite that never ran.
    probe: Callable[[], bool] | None = field(default=None, compare=False)
    #: Whether `--absent` may declare it. The Rust binary may not (IKA-210): the port is the
    #: only engine the rule tests have, so without it they fail rather than skip, and a
    #: machine that lacks it has not run the suite at all.
    declarable: bool = True


def _oracle_built() -> bool:
    return (ROOT / "packages" / "sim-bridge" / "dist" / "cli" / "oracle.js").exists()


def _rust_built() -> bool:
    release = ROOT / "rust" / "target" / "release"
    return (release / "pokeuraou-damage").exists() or (
        release / "pokeuraou-damage.exe"
    ).exists()


def _regulation_present() -> bool:
    return any((ROOT / "configs" / "regulations").glob("*.json"))


def _priors_present() -> bool:
    raw = ROOT / "data" / "priors" / "raw"
    return raw.exists() and any(raw.glob("*.json*"))


def _standings_present() -> bool:
    standings = ROOT / "data" / "standings"
    return standings.exists() and any(standings.glob("*.json*"))


def _names_present() -> bool:
    return (ROOT / "configs" / "names" / "ja.json").exists()


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _cuda_present() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


#: Every skip reason the suite can produce, and which of them is a build's job to prevent.
#: Ordered: the first pattern that matches wins, so the specific ones come first.
CLASSES: tuple[Class, ...] = (
    Class(
        "oracle",
        re.compile(r"oracle not built"),
        "the TypeScript bridge to Showdown; `npm ci && npm run build`, and Showdown "
        "itself has to be built inside vendor/ first",
        probe=_oracle_built,
    ),
    Class(
        "regulation",
        re.compile(r"regulation config missing"),
        "configs/regulations/*.json, which is committed",
        probe=_regulation_present,
    ),
    Class(
        "rust",
        re.compile(
            r"no Rust binary at|the shared path is only taken with the port"
        ),
        "rust/target/release/pokeuraou-damage; `cargo build --release`. Since IKA-210 the "
        "suite fails instead of skipping without it, so a match here is a test that still "
        "skips -- it should fail -- and the class cannot be declared absent",
        probe=_rust_built,
        declarable=False,
    ),
    Class(
        "priors",
        re.compile(r"no cached usage stats|no usage data"),
        "data/priors/raw/*.json.gz; `tools/fetch_priors.py`",
        probe=_priors_present,
    ),
    Class(
        "standings",
        re.compile(r"no cached standings"),
        "data/standings/*.json.gz; `tools/fetch_standings.py 2026 worlds`",
        probe=_standings_present,
    ),
    Class(
        "names",
        re.compile(r"no ja names"),
        "configs/names/ja.json, which is committed",
        probe=_names_present,
    ),
    Class(
        "vendor",
        re.compile(r"vendor submodule not initialised"),
        "vendor/pokemon-showdown as a checkout of its own; `git submodule update --init`. "
        "A git worktree does not get one, so a worktree run declares it absent",
        environment=True,
        probe=lambda: (ROOT / "vendor" / "pokemon-showdown" / ".git").exists(),
    ),
    Class(
        "torch",
        re.compile(r"could not import 'torch'|needs the optional learn group"),
        "the optional learn group; `uv sync --group learn`",
        probe=_torch_present,
    ),
    Class(
        "cuda",
        re.compile(r"CUDA graphs need a card"),
        "a CUDA card; the server's CUDA-graph replays (IKA-291) have nothing to replay on "
        "without one, and GitHub's runner has none",
        probe=_cuda_present,
    ),
    Class(
        "cpu-batch",
        re.compile(r"CPU kernels let a row's answer depend on its batch"),
        "a CPU whose torch kernels keep a row's answer independent of its batch; the "
        "AVX512 box does, GitHub's runner does not (IKA-51)",
        probe=lambda: not os.environ.get("POKEURAOU_CPU_BATCH_VARIES"),
    ),
    Class(
        "fixtures",
        re.compile(r"\.json missing$|scenario-turn\d+\.json missing"),
        "a committed example or fixture file",
        environment=True,
        probe=lambda: (ROOT / "examples" / "scenario-turn1.json").exists(),
    ),
    # Below here: the data's contents decide, not the machine. Counted, never pinned.
    Class(
        "no-such-case",
        re.compile(
            r"no species with enough particles"
            r"|every legal species appears in the usage data"
            r"|no class in this position merged more than one particle"
            r"|this regulation has no "
            r"|no single-target damaging move"
            r"|no usable move"
            r"|calculator flags"
            r"|the hidden Pokemon has no priority move"
            r"|no clean single-cause damage turn was produced"
            r"|fixture slot has no Protect"
            r"|the target did not survive the hit"
            r"|no usable archetype"
            r"|no game finished in this small sample"
            r"|only \d+ (phases|turns) compared"
            # test_beliefnode's refused-cell tests: no move is left for the port to refuse
            # (IKA-208), and the Python fallback they hold goes in IKA-209.
            r"|the port refuses no move"
        ),
        "the fixture or the sampled data holds no instance of the case",
        environment=False,
    ),
)


def parse(
    junit: Path,
) -> tuple[list[tuple[str, str]], dict[str, int], list[tuple[str, str]]]:
    """(test id, skip reason) per skip, the run's own totals, and the xfails apart.

    A `pytest.skip` inside a test puts its reason in ``message``. A module-scope
    `importorskip` does not: the whole module is one synthetic case whose message is the
    constant "collection skipped", and the reason is in the element's text. Both are read,
    or the four modules that need torch would be four skips of unknown cause -- which is
    the failure this file exists to refuse.

    An xfail is written the same way, as ``<skipped type="pytest.xfail">``, but it is not a
    check that stopped running: the test ran and failed as declared, and a strict one fails
    the suite the day the defect it holds is fixed (IKA-158 holds the hazards on the wrong
    side this way). So it is kept out of the skips, and listed on its own.
    """
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    skips: list[tuple[str, str]] = []
    xfails: list[tuple[str, str]] = []
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.get(key, 0) or 0)
        for case in suite.iter("testcase"):
            for skipped in case.findall("skipped"):
                where = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
                message = (skipped.get("message") or "").strip()
                body = " ".join((skipped.text or "").split())
                reason = f"{message} {body}".strip() if body else message
                if skipped.get("type") == "pytest.xfail":
                    xfails.append((where, reason))
                else:
                    skips.append((where, reason))
    return skips, totals, xfails


def classify(reason: str) -> Class | None:
    for cls in CLASSES:
        if cls.pattern.search(reason):
            return cls
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--junit", type=Path, required=True, help="pytest --junitxml report")
    ap.add_argument(
        "--absent",
        action="append",
        default=[],
        metavar="NAME=N",
        help="this machine does not supply NAME, and exactly N tests may say so",
    )
    args = ap.parse_args(argv)

    absent: dict[str, int] = {}
    for spec in args.absent:
        name, _, count = spec.partition("=")
        known = {c.name for c in CLASSES if c.environment}
        if name not in known:
            print(f"  --absent {name}: not an environment class ({', '.join(sorted(known))})")
            return 2
        if not next(c for c in CLASSES if c.name == name).declarable:
            print(f"  --absent {name}: the suite requires it and fails without it (IKA-210)")
            return 2
        absent[name] = int(count) if count else 0

    if not args.junit.exists():
        print(f"  no report at {args.junit}; pytest did not get far enough to write one")
        return 2

    skips, totals, xfails = parse(args.junit)
    print(
        f"  {totals['tests']} tests, {totals['failures']} failed, "
        f"{totals['errors']} errored, {len(skips)} skipped"
        + (f", {len(xfails)} xfailed" if xfails else "")
    )
    for where, reason in xfails:
        print(f"    xfail (ran, a known defect held): {where}: {reason}")
    if totals["tests"] == 0:
        print("  no tests ran, so 'nothing skipped' means nothing")
        return 1

    counted: dict[str, list[tuple[str, str]]] = {}
    unknown: list[tuple[str, str]] = []
    for where, reason in skips:
        cls = classify(reason)
        if cls is None:
            unknown.append((where, reason))
        else:
            counted.setdefault(cls.name, []).append((where, reason))

    bad = 0
    print()
    for cls in CLASSES:
        got = counted.get(cls.name, [])
        if not cls.environment:
            if got:
                print(f"  {len(got):>3}  {cls.name:<12} allowed -- {cls.why}")
            continue
        budget = absent.get(cls.name)
        if budget is None:
            # Nothing said this machine lacks it, so the build was supposed to supply it.
            present = cls.probe() if cls.probe else True
            if not present:
                bad += 1
                print(
                    f"    -  {cls.name:<12} MISSING from this machine and not declared "
                    f"with --absent -- {cls.why}"
                )
            elif got:
                bad += 1
                print(
                    f"  {len(got):>3}  {cls.name:<12} SKIPPED although it is built -- "
                    f"the test looks somewhere the build did not write"
                )
                for where, reason in got[:5]:
                    print(f"         {where}: {reason}")
            else:
                print(f"    0  {cls.name:<12} built, and nothing skipped for it")
            continue
        if len(got) != budget:
            bad += 1
            print(
                f"  {len(got):>3}  {cls.name:<12} DECLARED ABSENT with a budget of "
                f"{budget}, and {len(got)} skipped -- {cls.why}"
            )
        else:
            print(f"  {len(got):>3}  {cls.name:<12} declared absent, budget {budget} -- ok")

    if unknown:
        bad += 1
        print(f"\n  {len(unknown)} skip(s) whose reason this audit does not know:")
        for where, reason in unknown:
            print(f"    {where}: {reason}")
        print(
            "  Add it to CLASSES in tools/ci_skip_audit.py, with which kind it is.\n"
            "  A skip nobody classified is a check nobody decided to stop running."
        )

    if bad:
        print(f"\n  {bad} problem(s) with what this run skipped.")
        return 1
    print("\n  every skip was one this run declared in advance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
