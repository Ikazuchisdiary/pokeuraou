"""Do the port's new commands answer as Python's resolver does? (IKA-211)

`diff_node` holds the port's `resolve` and `fill` to Python. This holds the commands that
took over what only Python answered -- a turn stopped for a mid-turn replacement and its
continuation, the replacement phase, the leads' switch-ins -- on the positions where each
of them actually happens, read from recorded games:

* `pause`: a move decision followed by a `selfswitch` one. The turn is resolved whole
  (`turn` with `full`) and every branch and every pause is compared; then, for the recorded
  pause, the likeliest and the first, every replacement is resumed (`alternatives`) and
  each resumed turn compared the same way, nested pauses one level further. The recorded
  replacement is also resumed on its own (`turn` with a pause), and `select` is held to the
  outcome it names.
* `replacement`: a `replacement` decision, both sides' recorded choices, with no generator
  (a draw takes the first option, noted) and with a seeded one -- where the generator's
  state afterwards is compared too, since the point of the draw protocol is that a game
  draws the same numbers either way.
* `leads`: a game's opening, rebuilt from its teams before any lead ability ran, the same
  two ways.

Compared as `diff_node` compares a branch: the position (every field of `to_json`), the
probability, the notes (`unmodelled`) and `exact`, in order.

    python tools/diff_commands.py --games-dir data/ika73/w12 --samples 40
    python tools/diff_commands.py --games-dir <M-C games> --samples 60 --jobs 8 \\
        --exes rust/target/release/pokeuraou-damage.exe,<another build>

Each sample is independent, so `--jobs N` hands them to N processes (spawned: each opens
its own nodes). Python's answer is computed once per sample and every exe in `--exes` is
compared against it. The summary does not depend on `--jobs`, which is its null control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.actions import (  # noqa: E402
    PassAction,
    SideAction,
    side_actions,
    switch_actions_after_faint,
)
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.resolve import (  # noqa: E402
    Budget,
    TurnResult,
    apply_lead_abilities,
    replacements_needed,
    resolve_replacements,
    resolve_turn,
    resume_alternatives,
    resume_turn,
)

KINDS = ("pause", "replacement", "leads")

# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def collect(dirs: list[str], format_id: str, kinds: set[str], per_kind: int) -> list[dict]:
    """The first `per_kind` samples of each kind, in file and game order."""
    found: dict[str, list[dict]] = {kind: [] for kind in kinds}
    for directory in dirs:
        for path in sorted(Path(directory).glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    if all(len(found[k]) >= per_kind for k in kinds):
                        return [s for k in KINDS if k in kinds for s in found[k]]
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    decisions = record.get("decisions") or []
                    if not decisions or decisions[0]["position"].get("format") != format_id:
                        continue
                    where = f"{path.name}:{line_number}"
                    if "leads" in kinds and len(found["leads"]) < per_kind:
                        found["leads"].append(
                            {
                                "kind": "leads",
                                "where": where,
                                "own": record["ownTeam"],
                                "foe": record["foeTeam"],
                                "first": decisions[0]["position"],
                            }
                        )
                    for index, decision in enumerate(decisions):
                        kind = decision.get("kind")
                        after = decisions[index + 1] if index + 1 < len(decisions) else None
                        sample = {
                            "where": f"{where}#{index}",
                            "position": decision["position"],
                            "own": decision.get("ownChosen"),
                            "foe": decision.get("foeChosen"),
                        }
                        if (
                            "pause" in kinds
                            and kind == "move"
                            and after is not None
                            and after.get("kind") == "selfswitch"
                            and len(found["pause"]) < per_kind
                        ):
                            found["pause"].append(
                                {
                                    **sample,
                                    "kind": "pause",
                                    "paused": after["position"],
                                    "answer": [after.get("ownChosen"), after.get("foeChosen")],
                                }
                            )
                        if (
                            "replacement" in kinds
                            and kind == "replacement"
                            and len(found["replacement"]) < per_kind
                        ):
                            found["replacement"].append({**sample, "kind": "replacement"})
    return [s for k in KINDS if k in kinds for s in found[k]]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class Differ(Exception):  # noqa: N818 - a verdict, raised to stop at the first difference
    pass


def _same_position(mine: Position, theirs: Position, what: str) -> None:
    left, right = mine.to_json(), theirs.to_json()
    if left != right:
        keys = [k for k in left if left.get(k) != right.get(k)]
        raise Differ(f"{what}: position differs in {keys}")


def _same_probability(mine: float, theirs: float, what: str, tally: Counter) -> None:
    if mine == theirs:
        tally["bit-identical probabilities"] += 1
    elif abs(mine - theirs) <= 1e-12:
        tally["probabilities within 1e-12"] += 1
    else:
        raise Differ(f"{what}: probability python {mine!r}, port {theirs!r}")


def compare_turn(py: TurnResult, port: rustnode.PortTurn, what: str, tally: Counter) -> None:
    """Every branch and every pause, in order, with the notes and `exact`."""
    if tuple(sorted(py.unmodelled)) != tuple(sorted(port.unmodelled)):
        raise Differ(f"{what}: notes python {sorted(py.unmodelled)}, port {sorted(port.unmodelled)}")
    if bool(py.exact) != bool(port.exact):
        raise Differ(f"{what}: exact python {py.exact}, port {port.exact}")
    if len(py.branches) != len(port.outcomes) or len(py.suspended) != len(port.pauses):
        raise Differ(
            f"{what}: python {len(py.branches)} branches + {len(py.suspended)} pauses, "
            f"port {len(port.outcomes)} + {len(port.pauses)}"
        )
    for index, (mine, theirs) in enumerate(zip(py.branches, port.outcomes, strict=True)):
        _same_probability(mine.probability, theirs.probability, f"{what} branch {index}", tally)
        _same_position(mine.position, theirs.position, f"{what} branch {index}")
        tally["branches"] += 1
    for index, (mine, theirs) in enumerate(zip(py.suspended, port.pauses, strict=True)):
        _same_probability(mine.probability, theirs.probability, f"{what} pause {index}", tally)
        _same_position(mine.position, theirs.position, f"{what} pause {index}")
        tally["pauses"] += 1


def compare_alternatives(
    reg: Regulation, node: rustnode.RustNode, pause_py, pause_port, what: str, tally: Counter, depth: int  # noqa: ANN001
) -> None:
    chooser, mine = resume_alternatives(reg, pause_py)
    answered = node.resume_alternatives(pause_port)
    if answered is None:
        raise Differ(f"{what}: the port refused the alternatives")
    port_chooser, theirs = answered
    if chooser != port_chooser:
        raise Differ(f"{what}: chooser python {chooser}, port {port_chooser}")
    if [o.to_choice() for o, _ in mine] != [o.to_choice() for o, _ in theirs]:
        raise Differ(f"{what}: options differ")
    tally["alternatives"] += 1
    for (option, resumed), (_o, port_resumed) in zip(mine, theirs, strict=True):
        label = f"{what} [{option.to_choice()}]"
        compare_turn(resumed, port_resumed, label, tally)
        tally["resumed turns"] += 1
        if depth < 1:
            for index, (inner_py, inner_port) in enumerate(
                zip(resumed.suspended, port_resumed.pauses, strict=True)
            ):
                if index >= 2:
                    break
                compare_alternatives(
                    reg, node, inner_py, inner_port, f"{label} pause {index}", tally, depth + 1
                )


def _nearest(pauses: list, recorded: dict) -> int:  # noqa: ANN001
    target = [mon["hp"] for side in recorded["sides"] for mon in side["pokemon"]]

    def distance(index: int) -> int:
        mons = [m for side in pauses[index].position.sides for m in side.pokemon]
        return sum(abs(m.hp - want) for m, want in zip(mons, target, strict=False))

    return min(range(len(pauses)), key=distance)


def _choice(options: list[SideAction], wanted: str | None) -> SideAction | None:
    return next((a for a in options if a.to_choice() == wanted), None)


# ---------------------------------------------------------------------------
# One sample, Python once, every exe against it
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def _open(format_id: str, exes: list[str], pauses: int) -> None:
    _STATE["reg"] = reg = load_regulation(format_id)
    # As every production caller does before its first turn (`diff_node` too).
    register_mega_stones(reg)
    _STATE["nodes"] = [rustnode.RustNode(reg, binary=Path(exe)) for exe in exes]
    _STATE["pauses"] = pauses


def _python_pause(reg: Regulation, sample: dict) -> dict:
    pos = Position.from_json(sample["position"])
    chosen = [
        _choice(side_actions(reg, pos, side), sample["own" if side == 0 else "foe"])
        for side in (0, 1)
    ]
    if None in chosen:
        return {"skip": "recorded choice not legal now"}
    result = resolve_turn(reg, pos, chosen, budget=Budget.exact())
    if not result.suspended:
        return {"skip": "the turn does not pause now"}
    return {"pos": pos, "chosen": chosen, "result": result}


def _run_pause(reg: Regulation, node: rustnode.RustNode, sample: dict, py: dict, tally: Counter) -> None:
    pos, chosen, result = py["pos"], py["chosen"], py["result"]
    port = node.turn(pos, chosen, Budget.exact(), full=True)
    if port is None:
        raise Differ("refused: the port refused the turn")
    compare_turn(result, port, "turn", tally)
    picked = [0, _nearest(list(result.suspended), sample["paused"])]
    picked.append(max(range(len(result.suspended)), key=lambda k: result.suspended[k].probability))
    picked = list(dict.fromkeys(picked))[: _STATE["pauses"]]
    for k in picked:
        pause_py, pause_port = result.suspended[k], port.pauses[k]
        # `select` names this pause by its index after the branches.
        selected = node.turn(pos, chosen, Budget.exact(), select=len(result.branches) + k)
        if selected is None or selected.pause is None or selected.pause.raw != pause_port.raw:
            raise Differ(f"select {len(result.branches) + k} is not pause {k}")
        compare_alternatives(reg, node, pause_py, pause_port, f"pause {k}", tally, 0)
        # The recorded replacement, resumed on its own.
        owed = _self_switch_flags(pause_py.position)
        chooser = 0 if any(owed[0]) else 1
        options = switch_actions_after_faint(reg, pause_py.position, chooser, list(owed[chooser]))
        option = _choice(options, sample["answer"][chooser])
        if option is None:
            continue
        passes = SideAction(
            slots=tuple(PassAction(slot=i) for i in range(len(pause_py.position.sides[1 - chooser].active)))
        )
        choices = [option, passes] if chooser == 0 else [passes, option]
        resumed = node.resume(pause_port, choices, full=True)
        if resumed is None:
            raise Differ(f"pause {k}: the port refused the recorded replacement")
        compare_turn(resume_turn(reg, pause_py, choices), resumed, f"pause {k} recorded", tally)
        tally["recorded replacements resumed"] += 1
        if resumed.outcomes:
            one = node.resume(pause_port, choices, select=0)
            if one is None or one.position is None:
                raise Differ(f"pause {k}: select 0 of the resumed turn gave nothing")
            _same_position(resumed.outcomes[0].position, one.position, f"pause {k} select 0")


def _self_switch_flags(pos: Position):  # noqa: ANN202
    from pokeuraou.resolve import self_switches_needed

    return self_switches_needed(pos)


def _python_phase(reg: Regulation, sample: dict) -> dict:
    if sample["kind"] == "replacement":
        pos = Position.from_json(sample["position"])
        owed = replacements_needed(pos)
        chosen = [
            _choice(
                switch_actions_after_faint(reg, pos, side, list(owed[side])),
                sample["own" if side == 0 else "foe"],
            )
            for side in (0, 1)
        ]
        if None in chosen:
            return {"skip": "recorded choice not legal now"}

        def run(rng):  # noqa: ANN001, ANN202
            return resolve_replacements(reg, pos, chosen, rng=rng)

        return {"pos": pos, "chosen": chosen, "plain": run(None), "seeded": _seeded(run)}
    pos = _opening(reg, sample)
    first = Position.from_json(sample["first"])

    def lead(rng):  # noqa: ANN001, ANN202
        return apply_lead_abilities(reg, pos, rng=rng)

    plain = lead(None)
    return {
        "pos": pos,
        "plain": plain,
        "seeded": _seeded(lead),
        "record": plain.position.to_json() == first.to_json(),
    }


SEEDS = (0, 1, 2)


def _seeded(run) -> list:  # noqa: ANN001
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        phase = run(rng)
        out.append((phase, json.dumps(rng.bit_generator.state, sort_keys=True)))
    return out


def _opening(reg: Regulation, sample: dict) -> Position:
    """The opening before any lead ability ran: `position_from_sets` with the phase off."""
    from pokeuraou import selfplay
    from pokeuraou.priors import SampledSet
    from pokeuraou.resolve import ReplacementResult

    def sets(team: list[dict]) -> list[SampledSet]:
        return [
            SampledSet(
                species=t["species"], ability=t["ability"], item=t.get("item"),
                nature=t["nature"], sp=dict(t.get("sp") or {}), moves=list(t["moves"]),
            )
            for t in team
        ]

    applied = selfplay.apply_lead_abilities
    selfplay.apply_lead_abilities = lambda _reg, pos, rng=None: ReplacementResult(position=pos)  # noqa: ARG005
    try:
        return selfplay.position_from_sets(reg, sets(sample["own"]), sets(sample["foe"]))
    finally:
        selfplay.apply_lead_abilities = applied


def _run_phase(node: rustnode.RustNode, sample: dict, py: dict, tally: Counter) -> None:
    def ask(rng):  # noqa: ANN001, ANN202
        if sample["kind"] == "replacement":
            return node.resolve_replacements(py["pos"], py["chosen"], rng=rng)
        return node.apply_lead_abilities(py["pos"], rng=rng)

    port = ask(None)
    if port is None:
        raise Differ("refused: the port refused the phase")
    mine = py["plain"]
    if tuple(sorted(mine.unmodelled)) != tuple(sorted(port.unmodelled)):
        raise Differ(f"notes python {sorted(mine.unmodelled)}, port {sorted(port.unmodelled)}")
    _same_position(mine.position, port.position, "phase")
    tally["positions"] += 1
    if sample["kind"] == "replacement":
        needed = node.replacements_needed(py["pos"])
        if needed != tuple(tuple(f) for f in replacements_needed(py["pos"])):
            raise Differ("replacements_needed differs")
    for seed, (phase, state) in zip(SEEDS, py["seeded"], strict=True):
        rng = np.random.default_rng(seed)
        theirs = ask(rng)
        if theirs is None:
            raise Differ(f"refused: seed {seed}")
        _same_position(phase.position, theirs.position, f"seed {seed}")
        if tuple(sorted(phase.unmodelled)) != tuple(sorted(theirs.unmodelled)):
            raise Differ(f"seed {seed}: notes differ")
        if json.dumps(rng.bit_generator.state, sort_keys=True) != state:
            raise Differ(f"seed {seed}: the generator drew differently")
        tally["seeded positions"] += 1
    if any("the first; not branched" in u for u in mine.unmodelled):
        tally["phases with a draw"] += 1


def work(sample: dict) -> dict:
    """One sample: Python's answer once, then each exe's verdict against it."""
    reg = _STATE["reg"]
    started = time.perf_counter()
    try:
        py = _python_pause(reg, sample) if sample["kind"] == "pause" else _python_phase(reg, sample)
    except Exception as exc:  # noqa: BLE001 - a record Python cannot answer is reported
        py = {"skip": f"python raised {type(exc).__name__}"}
    python_s = time.perf_counter() - started
    out: dict[str, Any] = {"kind": sample["kind"], "where": sample["where"], "python_s": python_s}
    if "skip" in py:
        out["skip"] = py["skip"]
        return out
    if "record" in py:
        out["record"] = py["record"]
    verdicts = []
    for node in _STATE["nodes"]:
        tally: Counter = Counter()
        try:
            if sample["kind"] == "pause":
                _run_pause(reg, node, sample, py, tally)
            else:
                _run_phase(node, sample, py, tally)
            verdict = "same"
            detail = ""
        except Differ as exc:
            text = str(exc)
            verdict = "refused" if text.startswith("refused") else "differ"
            detail = text
        except Exception as exc:  # noqa: BLE001 - an exe without the commands lands here
            verdict, detail = "failed", f"{type(exc).__name__}: {exc}"[:200]
        verdicts.append({"verdict": verdict, "detail": detail, "tally": dict(tally)})
    out["verdicts"] = verdicts
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def summarise(results: list[dict], exes: list[str]) -> dict:
    summary: dict[str, Any] = {"exes": exes, "kinds": {}}
    for kind in KINDS:
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        entry: dict[str, Any] = {
            "samples": len(rows),
            "skipped": dict(Counter(r["skip"] for r in rows if "skip" in r)),
            "python_seconds": round(sum(r["python_s"] for r in rows), 1),
            "per_exe": [],
        }
        if kind == "leads":
            entry["python_matches_record"] = sum(1 for r in rows if r.get("record"))
        for index in range(len(exes)):
            verdicts = Counter()
            details = Counter()
            tally: Counter = Counter()
            for r in rows:
                if "skip" in r:
                    continue
                v = r["verdicts"][index]
                verdicts[v["verdict"]] += 1
                if v["detail"]:
                    details[v["detail"].split(":")[0] if v["verdict"] != "differ" else v["detail"][:120]] += 1
                tally.update(v["tally"])
            entry["per_exe"].append(
                {
                    "verdicts": dict(verdicts),
                    "tally": dict(sorted(tally.items())),
                    "details": dict(details.most_common(8)),
                }
            )
        summary["kinds"][kind] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games-dir", action="append", required=True)
    parser.add_argument("--format", default=None, help="regulation; default: the first record's")
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--samples", type=int, default=20, help="per kind")
    parser.add_argument("--pauses", type=int, default=3, help="pauses resumed per interrupted turn")
    parser.add_argument("--exes", default=str(rustnode.binary_path()))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", default=None, help="write the summary as JSON here")
    args = parser.parse_args()

    format_id = args.format
    if format_id is None:
        first = next(p for d in args.games_dir for p in sorted(Path(d).glob("*.jsonl")))
        with first.open(encoding="utf-8") as handle:
            format_id = json.loads(handle.readline())["decisions"][0]["position"]["format"]
    kinds = {k for k in args.kinds.split(",") if k}
    exes = [str(Path(e).resolve()) for e in args.exes.split(",") if e]
    started = time.perf_counter()
    samples = collect(args.games_dir, format_id, kinds, args.samples)
    print(f"{len(samples)} samples ({format_id}): {dict(Counter(s['kind'] for s in samples))}", flush=True)
    if args.jobs <= 1:
        _open(format_id, exes, args.pauses)
        results = [work(s) for s in samples]
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=get_context("spawn"),
            initializer=_open,
            initargs=(format_id, exes, args.pauses),
        ) as pool:
            results = list(pool.map(work, samples, chunksize=1))
    summary = summarise(results, exes)
    summary["wall_seconds"] = round(time.perf_counter() - started, 1)
    summary["jobs"] = args.jobs
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).write_bytes(json.dumps(summary, indent=1, ensure_ascii=False).encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
