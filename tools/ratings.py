"""One scale for every agent, fitted from every match ever recorded.

Each measurement so far has been a pair: "+2.9 [+0.5, +5.3] against value-gen234". That is
the right answer to the question it was asked and it does not compose. It cannot say how
much stronger the current model is than the one three generations back without playing
that match, it cannot put width 48 and the candidate ranking on the same axis, and every
new arm needs a baseline chosen for it in advance.

A rating is that answer generalised. The model is Bradley-Terry with a side term:

    P(side 0 wins) = sigmoid(r[agent on side 0] - r[agent on side 1] + s)

which is exactly the R-and-a decomposition every match here has done by hand, with two
players replaced by as many as have ever played. `s` is the seat advantage -- side 0 is
always our roster and side 1 always a tournament team, so it is real and has been measured
between 50.7% and 54.9% -- and fitting it rather than assuming it away is what lets games
from different matchups pool.

An *agent* is a model together with the search that ran it, because that is what played:
width 48 beat width 24 with the same model by +5.5, so a rating keyed on the model alone
would pool two things that are five points apart. `provenance.agent_name` builds the key.

## What it cannot do

A Bradley-Terry rating assumes strength is one number and that beating A implies as much
against B as it does against C. Matchup cycles break that, and this game has them -- a
three-way cycle among selections is documented in `configs/knowledge/`. The fit reports
its own residuals per pair so a reader can see where the assumption fails rather than
learn it later; a pair whose observed win rate sits far from the fitted one is a cycle
showing through, not noise.

## A seed played twice

Games are dealt from `[seed, index]`, so a pairing run twice at one seed -- a crash and its
rerun, a run and its fixed rerun -- is dealt the same games twice, and counting both would
halve the variance behind its interval for nothing. Such a replay is counted once, from the
newest run that did not fail (`repeated_draws` has the measurements behind that choice).
Which convention made a table is printed on it, and `--shared-seed independent` reproduces
the count from before IKA-44.

Nothing here plays a game. It reads the games that were already played and recorded.

    uv run python tools/ratings.py
    uv run python tools/ratings.py --anchor value-gen2345.pt/w24 --logit
    uv run python tools/ratings.py --shared-seed independent   # the count before IKA-44
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.provenance import agent_name

#: (side 0 agent, side 1 agent, side 0's wins, games played). A single game is one of
#: these with `played` 1, which is what lets a per-seat total and a per-game record sit in
#: the same list.
Observation = tuple[str, str, float, int]

#: Elo's scale: 400 points is a factor of ten in odds. Logits are what the model is in.
ELO_PER_LOGIT = 400.0 / math.log(10.0)

#: Zero is the parameter-free objective. It is the only competitor here that cannot drift:
#: `hp-share` has no weights, no training set and no version, so a rating measured against
#: it means the same thing next month as it does today -- which a rating anchored on
#: whichever model happens to be current does not. It also reads as the quantity that
#: matters: what the learned machinery is worth over having none of it.
ANCHOR = "hp-share/w24"

#: How a pairing played twice on one seed's draws is counted (IKA-44). `newest` counts the
#: draws once, from the newest run that did not fail; `independent` counts every run, which
#: is what this tool did before. Printed above every table, because the choice moves the
#: intervals and a number can only be reused together with the convention it came from.
SHARED_SEED = ("newest", "independent")

#: Two runs of one pairing at one seed replay each other when at least this share of the
#: smaller run's draws are also the other's. Measured on 2026-09-23: 100% in each of the
#: three replays on disk (1,696 of 1,696), against 13.9% and 15.6% for two pairings each run
#: twice at DIFFERENT seeds, where the same team and fours come up again by chance, and 0%
#: for runs sharing seed 20260919 while drawing their fours another way. Half is in the empty
#: middle, so nothing on disk depends on its exact value.
REPLAY_SHARE = 0.5


def recover_old_axes(source: dict) -> int:
    """Fill in `depths` and `rankings` for records written before they existed.

    Those two axes were once carried only in the free-text seat label -- `...@leafrank =
    side 0`, `...@d2 = side 1` -- and a record that lacks them reads as the default
    configuration. Which means the ranking match's two arms, the depth match's two arms and
    every plain match all arrive under one name, and a rating pools three things that were
    measured 2.9 and 2.1 points apart. The first fit did exactly that: five agents, and one
    of them holding 9,240 games.

    So the label is parsed, once, here, and the number of records that needed it is
    reported. This is archaeology on data already on disk, not a format: anything written
    from now on carries the fields, and `agent_name` reads only those.

    Returns 1 if the record needed repairing, 0 otherwise.
    """
    if "rankings" in source and "depths" in source:
        return 0
    label = str(source.get("seat", ""))
    arm, _, side_text = label.partition(" = side ")
    side = 0 if side_text.strip() == "0" else 1
    other = 1 - side
    depths = [1, 1]
    rankings = ["damage", "damage"]
    if "@leafrank" in arm:
        rankings[side] = "leaf"
    if "@d2" in arm:
        depths[side] = 2
        depths[other] = 1
    source.setdefault("depths", depths)
    source.setdefault("rankings", rankings)
    return 1


def draw_key(pair: tuple[str, str], game: dict[str, Any]) -> str:
    """What the seed dealt one game, as a short digest -- empty when the record cannot say.

    The two agents in their seats and the two fours they played. A match deals a game's
    team and both selections off `default_rng([seed, index])` before either side moves, so
    two runs of one pairing at one seed deal the same keys, which is what makes the second
    one a replay; a run that shares the seed but draws its fours from another book, another
    roster or uniformly deals other keys. The index is deliberately left out: records
    written before 9/19 do not carry it, and a replay of one of those by a run that does
    must still be found.
    """
    own, foe = game.get("ownTeam"), game.get("foeTeam")
    if not own or not foe:
        return ""
    text = json.dumps([pair[0], pair[1], own, foe], sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()


def read_games(
    root: Path,
    cache_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Every games file as the fit sees it, and the runs whose DONE marker says FAILED.

    A file comes back as a dict: `rows`, one observation per matchup with its games summed;
    `repaired` and `builds`, below; `stamp`, its size and mtime; and `draws`, one
    `draw_key` per game, which is what lets `repeated_draws` recognise a replay. Which
    files the fit then counts is the caller's decision, and `main` makes it.

    The build matters and the name cannot carry it. An agent here is a model together
    with the search that ran it, and a fix to the search makes a different agent -- but
    `agent_name` is built from the provenance's settings, and a settings block says
    nothing about whether the code honoured them. Three did not, and were found in one
    day: a menu handed to the wrong side, a hidden bench that dropped two arguments, a
    replacement node solved with one leaf.

    So the engine fingerprint each game already carries is counted and printed. It is
    over-sensitive on purpose -- a docstring edit moves it -- which makes it useless as a
    key and honest as a warning: games under different hashes were not played by the same
    program, and the fit pools them anyway.
    """
    files: dict[str, dict[str, Any]] = {}
    # Directories whose own DONE marker says the run failed. Their games are real and stay
    # in the fit -- a crash at game 107 does not make the first 107 fictional -- but a run
    # that stopped for a reason is not the run its command line describes, and nothing
    # said so: `genmatch-value-gen8x3-vs-value-allx3` records "FAILED (exit 1), 0 games"
    # over 107 games that have been on the scale ever since.
    #
    # Keyed by the directory relative to `root`, because a replay's two runs are told
    # apart by it: the one that failed is not the one kept.
    failed: dict[str, str] = {}
    for marker in sorted(root.glob("**/DONE")):
        try:
            said = marker.read_text(encoding="utf-8")
        except OSError:
            continue
        if "FAIL" in said.upper():
            where = str(marker.parent.relative_to(root)).replace("\\", "/")
            failed[where] = f"{marker.parent.name}: {said.strip().splitlines()[0]}"
    # Both layouts. A match dealt in fixed blocks names its files by seed and a queued one
    # by worker, and this read only the first -- so every match run since the queue landed
    # was missing from the scale, which is every measurement taken on the ensemble floor.
    # `run_and_report.sh` had the same bug and was fixed; this copy was not.
    paths = sorted(
        set(root.glob("**/games-seed*.jsonl")) | set(root.glob("**/games-worker*.jsonl"))
    )

    # Parsed once per file and kept, keyed on the file's size and mtime.
    #
    # Eight gigabytes of JSON is six minutes, every time, and a refit is the thing anyone
    # wants to do twice in a row -- after a new match, after a naming fix, with a
    # different anchor. A tool nobody runs because it is slow is a tool whose answer
    # nobody has.
    #
    # What is cached is the AGGREGATE per matchup, not the games: `fit` consumes
    # `(side 0, side 1, wins, played)` and a single game is that row with played 1, so
    # summing a file's identical rows changes nothing it computes -- including the
    # per-matchup residual table, which expands the counts back out. It also makes the
    # cache small enough to be JSON.
    #
    # The draws are the exception, one short key per game, because a replay is a fact about
    # games and not about totals. An entry written before they were kept is read again:
    # trusting it would find no replay anywhere and quietly restore the count this tool
    # made before IKA-44.
    if cache_path is None:
        cache_path = root / ".ratings-cache.json"
    cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    fresh: dict[str, Any] = {}
    for path in paths:
        stat = path.stat()
        key = str(path.relative_to(root)).replace("\\", "/")
        stamp = [stat.st_size, int(stat.st_mtime)]
        hit = cache.get(key)
        if hit and hit.get("stamp") == stamp and "draws" in hit:
            fresh[key] = files[key] = hit
            continue
        rows: dict[tuple[str, str], list[float]] = {}
        here: Counter[str] = Counter()
        draws: list[str] = []
        needed = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record = game.get("provenance")
                outcome = game.get("outcome")
                if not record or outcome is None:
                    continue
                needed += recover_old_axes(record)
                engine = game.get("engine") or {}
                here[str(engine.get("sources", "unrecorded"))] += 1
                pair = (agent_name(record, 0), agent_name(record, 1))
                got = rows.setdefault(pair, [0.0, 0])
                got[0] += float(outcome)
                got[1] += 1
                draws.append(draw_key(pair, game))
        listed = [[a, b, wins, n] for (a, b), (wins, n) in sorted(rows.items())]
        fresh[key] = files[key] = {
            "stamp": stamp,
            "repaired": needed,
            "builds": dict(here),
            "rows": listed,
            "draws": draws,
        }
    # The cache is an optimisation, so a read-only checkout or a full disk must cost the
    # rebuild rather than the answer.
    with contextlib.suppress(OSError):
        cache_path.write_text(json.dumps(fresh), encoding="utf-8")
    return files, failed


def _row_seeds(path: Path) -> set[object]:
    """The seeds a per-seat row file names -- empty if it names none or cannot be read."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    seeds: set[object] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "seed" in row and "played" in row:
            seeds.add(row["seed"])
    return seeds


def run_seeds(root: Path, keys: Iterable[str]) -> dict[str, object]:
    """The seed each games file was dealt from, read off the per-seat rows beside it.

    A game record does not carry its seed; the rows do, one per seat, so `games-X.jsonl`
    takes the seed its `X.jsonl` names. A worker that died wrote games and no row -- and a
    crashed run is the likeliest replay there is -- so a file with no row of its own takes
    its directory's seed when every row there names the same one, which a queued run's
    always do: one `--seed` goes to every worker. Anything else maps to None, and a file
    without a seed is never taken for a replay; `main` says how many games that leaves
    unchecked rather than guessing.
    """
    by_directory: dict[str, dict[str, set[object]]] = {}
    out: dict[str, object] = {}
    for key in keys:
        directory, _, name = key.rpartition("/")
        rows = by_directory.get(directory)
        if rows is None:
            base = root / directory if directory else root
            rows = {
                path.name: _row_seeds(path)
                for pattern in ("worker*.jsonl", "seed*.jsonl")
                for path in base.glob(pattern)
            }
            by_directory[directory] = rows
        own = rows.get(name.removeprefix("games-"), set())
        everywhere: set[object] = set().union(*rows.values()) if rows else set()
        if len(own) == 1:
            out[key] = next(iter(own))
        elif not own and len(everywhere) == 1:
            out[key] = next(iter(everywhere))
        else:
            out[key] = None
    return out


def _root(parent: list[int], i: int) -> int:
    """Union-find: the representative of `i`'s group, halving the path on the way."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def repeated_draws(
    files: dict[str, dict[str, Any]],
    seeds: dict[str, object],
    failed: dict[str, str],
    *,
    convention: str = "newest",
) -> tuple[set[str], list[str]]:
    """The games files a replay keeps out of the fit, and a line for every shared seed.

    A run is one directory's games at one seed. Two runs *replay* each other when they
    share the seed, field the same pairings, and at least `REPLAY_SHARE` of the smaller
    one's draws (`draw_key`) are the other's too: the same agents dealt the same games.
    Games are seeded from `[seed, index]`, so such a second run is a second look at the
    first one's draws, and a fit that counts both reads twice the games -- an interval of
    about 1/root 2 -- for information it does not have.

    What the second look was worth, measured on the three replays on disk on 2026-09-23,
    every game matched to its twin, both seats:

        seed 20261052  gen11L-vs-gen10-ownroster-uniform + crashed twin   100% identical
        seed 20261041  h12-vs-o12-hidden + crashed twin      64% identical, rho 0.79
        seed 31337     hidden-gen11h-vs-gen10 + -fixed        11% identical, rho 0.19

    where rho is the correlation, between the two runs, of one game's score summed over
    both seats -- the quantity the rating's variance is made of. The first is a duplicate.
    The other two are a later build replaying an earlier build's draws: the h12 rerun has
    a fix to the bench belief (a Mega's base form stayed in the candidate pool,
    `hidden.py`) and `-fixed` two fixes to the blind search, which its own CMD names.

    So under `newest` a replay is counted once, from the newest run that did not fail.
    That is exact for the duplicate, and in the other two it is the only choice that keeps
    a search nobody runs any more out of the agents' ratings. Weighting both runs by
    1/(1 + rho) would be the honest way to pool them if the two runs were one agent; here
    they are not, and records before 9/19 carry no game index to pair them by. `independent`
    counts every run, which is what this tool did before IKA-44, and is kept so that the
    old numbers can be reproduced.

    A seed shared by runs of different pairings is not a replay -- no pairing is dealt the
    same games twice -- and neither is one pairing at one seed with other draws (another
    roster, another book, a uniform draw). A shared seed does not by itself share a draw:
    `anchor-gen11L-vs-hpshare` shares seed 20260919 with two book matches and not one of
    its 1,696 draws, because it draws its fours uniformly and they draw theirs from a book.
    """
    if convention not in SHARED_SEED:
        raise ValueError(f"no shared-seed convention {convention!r}; one of {SHARED_SEED}")
    runs: dict[tuple[str, object], dict[str, Any]] = {}
    for key, entry in files.items():
        seed = seeds.get(key)
        if seed is None:
            continue
        directory = key.rpartition("/")[0] or "."
        run = runs.setdefault(
            (directory, seed),
            {"files": [], "pairings": set(), "draws": Counter(), "games": 0, "newest": 0},
        )
        run["files"].append(key)
        for a, b, _wins, n in entry["rows"]:
            run["pairings"].add(tuple(sorted((a, b))))
            run["games"] += int(n)
        run["draws"].update(d for d in entry["draws"] if d)
        run["newest"] = max(run["newest"], int(entry["stamp"][1]))

    by_seed: dict[object, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for (directory, seed), run in sorted(runs.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
        by_seed[seed].append((directory, run))

    skip: set[str] = set()
    lines: list[str] = []
    for seed, members in by_seed.items():
        if len(members) < 2:
            continue
        parent = list(range(len(members)))
        near: list[tuple[int, int, float]] = []
        for i, j in itertools.combinations(range(len(members)), 2):
            a, b = members[i][1], members[j][1]
            if a["pairings"] != b["pairings"]:
                continue
            smaller = min(a["draws"].total(), b["draws"].total())
            common = (a["draws"] & b["draws"]).total()
            if smaller and common >= REPLAY_SHARE * smaller:
                parent[_root(parent, j)] = _root(parent, i)
            else:
                near.append((i, j, common / max(smaller, 1)))
        components: dict[int, list[int]] = defaultdict(list)
        for i in range(len(members)):
            components[_root(parent, i)].append(i)
        replays = [c for c in components.values() if len(c) > 1]
        # One pairing at one seed that nonetheless drew mostly other games: independent, and
        # said so with the share, because "the same seed" was the whole of the old warning.
        # Only across groups -- two runs joined through a third are one replay, not this.
        unlike = [
            f"seed {seed}: {members[i][0]} and {members[j][0]} are one pairing, but only "
            f"{share:.0%} of the smaller one's draws are the other's -- both counted"
            for i, j, share in near
            if _root(parent, i) != _root(parent, j)
        ]
        # The newest run that did not fail is the one counted. The files' modification time
        # is the only order the record carries -- a copy that resets it changes the choice,
        # which is why the choice is printed.
        order = [(d in failed, -r["newest"], d) for d, r in members]
        for component in replays:
            ranked = sorted(component, key=order.__getitem__)
            kept_dir, kept = members[ranked[0]]
            others: list[str] = []
            for i in ranked[1:]:
                directory, run = members[i]
                share = (kept["draws"] & run["draws"]).total() / max(run["draws"].total(), 1)
                tag = ", FAILED" if directory in failed else ""
                others.append(
                    f"{directory} ({run['games']:,} games{tag}, {share:.0%} of its draws "
                    f"also in {kept_dir})"
                )
                if convention == "newest":
                    skip.update(run["files"])
            if convention == "newest":
                lines.append(
                    f"seed {seed}: counted {kept_dir} ({kept['games']:,} games); "
                    f"not counted: {'; '.join(others)}"
                )
            else:
                lines.append(
                    f"seed {seed}: {'; '.join(others)} replaying {kept_dir} "
                    f"({kept['games']:,} games) -- all counted, as independent"
                )
        lines.extend(unlike)
        rest = len(members) - sum(len(c) for c in replays)
        if rest == len(members) and not unlike:
            lines.append(
                f"seed {seed}: shared by {rest} runs of {rest} different pairings, so no "
                "pairing is dealt the same draws twice -- all counted"
            )
        elif rest and not unlike:
            lines.append(f"seed {seed}: also {rest} run(s) of other pairings -- counted")
    return skip, lines


def read_summaries(root: Path) -> list[Observation]:
    """Matches that kept only their per-seat totals, which is everything before yesterday.

    `--games-out` was wired into the launcher a day ago, so the earlier matches -- the
    width comparison that established 24 over 16, and the first two generation matches --
    left a row per seat and nothing else. A row is "n games, k of them won by this arm",
    which is exactly what the likelihood consumes; one game is the same row with n = 1. So
    the whole history goes on the scale, including the only games ever played at width 16.

    A directory that has game-level records is skipped entirely. Its summaries describe the
    same games, and counting a match twice would shrink its interval by root two while
    telling the reader nothing new.

    Two schemas, because the width match and the generation match were written apart:
    `wide`/`narrow`/`wide_wins` against `model`/`baseline`/`limit`/`gen2_wins`. Both say
    which side the named arm sat on in `seat`, which is the part that matters.
    """
    out: list[Observation] = []
    # Files whose name begins with `seed`, per directory, AND the loose ones at the top.
    #
    # The scan was `**/seed*.jsonl`, and the two oldest matches are not named that way:
    # `width48-vs-16.jsonl` and `gen2-vs-proxy.jsonl` sit directly under `data/matches/`
    # and hold 336 and 480 games. The second is `value-gen2` against the hp-share
    # objective -- an edge into the anchor's own family, which is exactly what a fit
    # reporting disconnected groups is short of.
    #
    # A row without `played` is skipped below, so the per-game files that also live at the
    # top level (`book-check*.jsonl`, `cycle-match.jsonl`) contribute nothing here and are
    # read by `read_games` instead. Nothing is counted twice: no loose file has a
    # `games-*` sibling naming the same games.
    candidates: list[Path] = [
        p for p in sorted(root.glob("*.jsonl")) if ".part" not in p.name
    ]
    for directory in sorted({p.parent for p in root.glob("**/seed*.jsonl")}):
        if any(directory.glob("games-seed*.jsonl")):
            continue
        candidates.extend(sorted(directory.glob("seed*.jsonl")))
    for path in candidates:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            played = int(row.get("played", 0))
            if not played:
                continue
            first = str(row.get("seat", "")).endswith("side 0")

            def stem(name: object) -> str:
                # The same normalisation `agent_name` does, for the same reason: these
                # rows spell `value-gen23.pt` where a game record spells `value-gen23`,
                # and the split makes one agent into two nodes with no edge between them
                # -- which is what the disconnected-groups warning has been reporting.
                text = str(name)
                return text[: -len(".pt")] if text.endswith(".pt") else text

            if "wide" in row and "narrow" in row:
                model = stem(row.get("model", "?"))
                arm = f"{model}/w{row['wide']}"
                other = f"{model}/w{row['narrow']}"
                wins = int(row.get("wide_wins", 0))
            else:
                limit = row.get("limit", "?")
                arm = f"{stem(row.get('model', '?'))}/w{limit}"
                other = (
                    f"{stem(row.get('baseline') or row.get('objective', '?'))}/w{limit}"
                )
                depth = int(row.get("depth", 1))
                other_depth = int(row.get("baselineDepth", 1))
                if depth != 1:
                    arm += f"/d{depth}"
                if other_depth != 1:
                    other += f"/d{other_depth}"
                wins = int(row.get("gen2_wins", row.get("wins", 0)))
            # The stored count is the named arm's; the fit wants side 0's.
            if first:
                out.append((arm, other, float(wins), played))
            else:
                out.append((other, arm, float(played - wins), played))
    return out


def fit(
    games: list[Observation],
    *,
    anchor: str | None = None,
    iterations: int = 500,
    prior: float = 1.0,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Ratings in logits, the seat advantage, and a standard error per agent.

    Gradient ascent on the log likelihood rather than a solver, because the problem is
    tiny and a dependency is not worth it. The prior is a weak pull towards zero: two
    agents that only ever beat each other would otherwise run off to infinity, and an
    agent with one game would be reported with more confidence than it has earned.

    The error is of the *difference from the anchor*, which needs the whole covariance
    rather than its diagonal. Only differences are observable -- adding a constant to
    every rating changes no prediction -- so a marginal error printed beside an anchored
    rating would be answering a question nobody can ask. The anchor's own is exactly zero,
    which is what holding it there means.
    """
    names = sorted({a for a, _b, _w, _n in games} | {b for _a, b, _w, _n in games})
    index = {name: i for i, name in enumerate(names)}
    rating = np.zeros(len(names))
    seat = 0.0
    rows = np.array([index[a] for a, _b, _w, _n in games])
    cols = np.array([index[b] for _a, b, _w, _n in games])
    wins = np.array([w for _a, _b, w, _n in games], dtype=np.float64)
    count = np.array([n for _a, _b, _w, n in games], dtype=np.float64)
    total = float(count.sum())

    step = 0.05
    for _ in range(iterations):
        predicted = 1.0 / (1.0 + np.exp(-(rating[rows] - rating[cols] + seat)))
        # An aggregated row of n games with k wins contributes exactly what those n games
        # would have, which is why a summary and its games are interchangeable here -- and
        # why counting both would be counting the same games twice.
        error = wins - count * predicted
        gradient = np.zeros(len(names))
        np.add.at(gradient, rows, error)
        np.add.at(gradient, cols, -error)
        gradient -= prior * rating
        rating += step * gradient / max(total / len(names), 1.0)
        seat += step * error.sum() / total

    # Curvature at the optimum, for a standard error per agent. Each game contributes
    # p(1-p) to both of its players; the prior contributes its own weight.
    predicted = 1.0 / (1.0 + np.exp(-(rating[rows] - rating[cols] + seat)))
    weight = count * predicted * (1.0 - predicted)
    hessian = np.diag(np.full(len(names), float(prior)))
    np.add.at(hessian, (rows, rows), weight)
    np.add.at(hessian, (cols, cols), weight)
    np.add.at(hessian, (rows, cols), -weight)
    np.add.at(hessian, (cols, rows), -weight)
    covariance = np.linalg.inv(hessian)

    if anchor is None:
        errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    else:
        if anchor not in index:
            raise SystemExit(f"no agent named {anchor!r}; try one of {names}")
        a = index[anchor]
        rating = rating - rating[a]
        errors = np.sqrt(
            np.clip(
                np.diag(covariance) + covariance[a, a] - 2.0 * covariance[:, a], 0.0, None
            )
        )
    return (
        dict(zip(names, rating, strict=True)),
        seat,
        dict(zip(names, errors, strict=True)),
    )


def components(games: list[Observation]) -> dict[str, int]:
    """Which connected group each agent belongs to, largest first.

    A Bradley-Terry fit only relates agents that are joined by a chain of matches. Agents
    in different groups have no defined difference at all, and the fit expresses that as a
    very wide interval rather than as an error -- so the table prints them in one column
    and a reader who skims past a +-150 compares numbers that were never comparable. The
    group is the honest version of that warning.
    """
    neighbours: dict[str, set[str]] = defaultdict(set)
    for a, b, _w, _n in games:
        neighbours[a].add(b)
        neighbours[b].add(a)
    groups: list[set[str]] = []
    seen: set[str] = set()
    for start in neighbours:
        if start in seen:
            continue
        stack, group = [start], set()
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            seen.add(node)
            stack.extend(n for n in neighbours[node] if n not in group)
        groups.append(group)
    groups.sort(key=len, reverse=True)
    return {name: index for index, group in enumerate(groups) for name in group}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matches", type=Path, default=Path("data/matches"),
        help="directory of recorded matches",
    )
    ap.add_argument(
        "--anchor",
        default=ANCHOR,
        help="agent held at zero. Empty string for a mean-zero scale instead.",
    )
    ap.add_argument("--logit", action="store_true", help="print logits instead of Elo")
    ap.add_argument("--min-games", type=int, default=1)
    ap.add_argument(
        "--shared-seed",
        choices=SHARED_SEED,
        default="newest",
        help="how a pairing dealt one seed's draws twice is counted: once, from its newest "
        "run that did not fail (newest, since IKA-44), or once per run (independent, the "
        "count before). Printed with the table, because it moves the intervals.",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="the parsed-games cache; default <matches>/.ratings-cache.json. Point it "
        "elsewhere to fit a corpus this process may not write to.",
    )
    args = ap.parse_args()

    files, failed = read_games(args.matches, args.cache)
    seeds = run_seeds(args.matches, files)
    skip, replays = repeated_draws(files, seeds, failed, convention=args.shared_seed)
    games: list[Observation] = []
    repaired = 0
    builds: Counter[str] = Counter()
    for key, entry in files.items():
        if key in skip:
            continue
        games.extend((a, b, float(w), int(n)) for a, b, w, n in entry["rows"])
        repaired += int(entry.get("repaired", 0))
        builds.update(entry.get("builds", {}))
    skipped = sum(int(n) for key in skip for _a, _b, _w, n in files[key]["rows"])
    unseeded = [key for key in files if seeds.get(key) is None]
    # A run none of whose games are counted is not "in this fit", whatever its DONE says.
    counted = {key.rpartition("/")[0] or "." for key in files if key not in skip}
    gone = {key.rpartition("/")[0] or "." for key in skip} - counted
    summaries = read_summaries(args.matches)
    games += summaries
    if not games:
        raise SystemExit(f"no recorded matches under {args.matches}")
    played: Counter[str] = Counter()
    for a, b, _w, n in games:
        played[a] += n
        played[b] += n

    rating, seat, errors = fit(games, anchor=args.anchor or None)
    scale = 1.0 if args.logit else ELO_PER_LOGIT
    unit = "logit" if args.logit else "Elo"
    names = sorted(rating, key=lambda n: -rating[n])

    total = sum(n for _a, _b, _w, n in games)
    from_rows = sum(n for _a, _b, _w, n in summaries)
    print(f"{total} games, {len(rating)} agents")
    if from_rows:
        print(
            f"  {from_rows} of them from matches that kept only per-seat totals, which is "
            "every match run before the games themselves were recorded"
        )
    if repaired:
        print(
            f"  {repaired} of them predate the per-side depth and ranking fields; "
            "their configuration was read back out of the seat label"
        )
    # Which convention made this table, on every table: the intervals depend on it, and a
    # number copied out without it cannot be reproduced (IKA-44).
    if args.shared_seed == "newest":
        print(
            '  shared seeds: convention "newest" (IKA-44) -- a pairing dealt one seed\'s '
            "draws twice counts them once, from its newest run that did not fail; "
            f"{skipped:,} games not counted"
        )
    else:
        print(
            '  shared seeds: convention "independent" (before IKA-44) -- every run counts '
            "as its own draws, replays included"
        )
    for line in replays:
        print(f"    {line}")
    if unseeded:
        lost = sum(int(n) for key in unseeded for _a, _b, _w, n in files[key]["rows"])
        unchecked = sorted({key.rpartition("/")[0] or "." for key in unseeded})
        print(
            f"    {lost:,} games in {len(unchecked)} run(s) name no seed in their per-seat "
            f"rows, so whether they replay another run is not checked: {', '.join(unchecked)}"
        )
    for where, line in failed.items():
        if where not in gone:
            print(f"  ! a run in this fit says it FAILED -- {line}")
    if len(builds) > 1:
        top = ", ".join(f"{h[:8]}={n}" for h, n in builds.most_common(6))
        print(
            f"  played by {len(builds)} different builds of the engine: {top}"
            + ("..." if len(builds) > 6 else "")
        )
        print(
            "  A search fix makes a different agent and this fit cannot separate them --\n"
            "  the name is built from the settings, and a settings block does not say\n"
            "  whether the code honoured them. On 2026-09-19 three did not: a candidate\n"
            "  menu handed to the opposing side, a hidden bench that dropped the depth and\n"
            "  the ranking, and a replacement node solved with one arm's leaf for both."
        )
    print(
        f"seat advantage {seat:+.3f} logit = side 0 wins "
        f"{100 / (1 + math.exp(-seat)):.1f}% between equals"
    )
    if args.anchor:
        print(
            f"anchored at {args.anchor} = 0, so a rating is what an agent is worth "
            "over the parameter-free objective"
        )
    island = components(games)
    groups = Counter(island.values())
    if len(groups) > 1:
        print(
            f"\n  ! {len(groups)} disconnected groups, sizes "
            + ", ".join(str(n) for _g, n in groups.most_common())
            + ".\n    A rating compares agents only within a group. Nothing has ever "
            "played across, so a\n    difference between groups is not uncertain, it is "
            "undefined -- the fit says so with\n    a very wide interval, which is easy to "
            "skim past."
        )
    # The condition, printed rather than left to be inferred from a suffix.
    #
    # An agent with no `/hidden-bench` in its name searched a game where the opponent's
    # four was visible to it. That is a different game and an easier one, and the game
    # this project is solving for hides the bench -- so those rows are reference. They are
    # the bulk of the corpus and they are not deleted, because the only measurement that
    # survives from six generations back is in that condition and a drifting yardstick is
    # worse than a limited one. But a reader who takes the top of this table as "the
    # strongest agent" has read the wrong column, which is why it is a column.
    # From the observations, not from `played`. `played` adds a game's `n` to BOTH agents
    # (it answers "how many games has this agent played"), while `total` counts each game
    # once -- so summing `played` over the hidden-bench agents counts every hidden game
    # twice, and the line printed 23,744 against a truth of 11,872 on the day it was
    # written. Two counters in the same sentence, in different units.
    hidden = sum(
        n
        for a, b, _w, n in games
        if a.endswith("/hidden-bench") or b.endswith("/hidden-bench")
    )
    print(
        f"\n  {hidden} games hid the bench and {total - hidden} did not. The open ones are "
        "REFERENCE:\n  their search saw the opponent's four, which is information a real "
        "game does not give.\n  They carry their own zero, so read that scale with "
        "--anchor hp-share/w24/hidden-bench."
    )
    print(
        f"\n  {'agent':<34} {unit:>9}  {'+-':>6}  {'games':>6}  {'grp':>3}  bench"
    )
    for name in names:
        if played[name] < args.min_games:
            continue
        half = 1.96 * errors[name] * scale
        bench = "hidden" if name.endswith("/hidden-bench") else "open(ref)"
        print(
            f"  {name:<34} {rating[name] * scale:>9.1f}  {half:>6.1f}  "
            f"{played[name]:>6}  {island.get(name, 0):>3}  {bench}"
        )

    # Where the one-number assumption is failing, if it is.
    pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for a, b, w, n in games:
        pairs[(a, b)].extend([1.0] * int(round(w)) + [0.0] * int(n - round(w)))
    print("\n  fitted against observed, per matchup (a gap is a cycle, not noise)")
    print(f"  {'side 0':<26} {'side 1':<26} {'n':>5} {'seen':>6} {'fit':>6}  {'gap':>6}")
    worst = sorted(
        pairs.items(),
        key=lambda kv: -abs(
            float(np.mean(kv[1]))
            - 1.0 / (1.0 + math.exp(-(rating[kv[0][0]] - rating[kv[0][1]] + seat)))
        ),
    )
    for (a, b), results in worst[:10]:
        seen = float(np.mean(results))
        expected = 1.0 / (1.0 + math.exp(-(rating[a] - rating[b] + seat)))
        print(
            f"  {a:<26} {b:<26} {len(results):>5} {seen * 100:>5.1f}% "
            f"{expected * 100:>5.1f}% {(seen - expected) * 100:>+6.1f}"
        )


if __name__ == "__main__":
    main()
