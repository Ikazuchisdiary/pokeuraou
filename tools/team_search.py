"""IKA-443: search for teams stronger than the pool's by recombining its members.

The pool's 65 teams hold 390 finished sets (moves, item, ability, nature, spread). A
candidate is a pool team with members swapped for sets from other pool teams; the sets
themselves are never changed, so every set is one the value net has seen in play. The first
stage scores a candidate by the selection game's value against each of the 65 teams (the
90x90 equilibrium of `selection.solve_selection`, side 0 the candidate), which IKA-442
found ranks pool teams at r 0.944. Later stages play real games.

    team_search.py base --store <selection store> [--pool regmc-matchupweb] --out base.json
        each pool team's mean selection value over the 65 teams, from a solved store
    team_search.py neighbors --base base.json --bases K --swaps 1 --out cands.json
        every candidate one swap away from the K teams with the highest value that passes
        Species and Item Clause (`roster_from_data`), deduplicated
    team_search.py value --candidates cands.json --out <dir> --workers W --model M [M ...]
        each candidate's selection value against the 65 teams, one file per candidate,
        served by two inference servers; a rerun skips the files already written
    team_search.py report --base base.json --candidates cands.json --values <dir> --out r.json

A candidate's team data is a pool file's team entry, so `pool_with` can write a pool file
holding the 65 teams and chosen candidates for `match_queue.py --pool`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.pool import find_pool, load_pool  # noqa: E402
from pokeuraou.teams import TeamError, roster_from_data  # noqa: E402


def _write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes((json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def _pool_data(pool: str) -> dict[str, Any]:
    return json.loads(find_pool(pool).read_bytes().decode("utf-8"))


def set_key(member: dict[str, Any]) -> str:
    """A set's identity: everything but how the source wrote the species."""
    kept = {k: v for k, v in member.items() if k != "writtenAs"}
    return json.dumps(kept, sort_keys=True, ensure_ascii=False)


def play_key(member: dict[str, Any]) -> str:
    """What a set plays as: `set_key` without the gender and with the moves in order. Two
    sheets that list one set's moves in another order are one candidate (IKA-443: 718 of the
    first 6,491 candidates repeated another so; 5,771 remain)."""
    kept = {k: v for k, v in member.items() if k not in ("writtenAs", "gender")}
    kept["moves"] = sorted(kept.get("moves", []))
    return json.dumps(kept, sort_keys=True, ensure_ascii=False)


def team_id(members: list[dict[str, Any]]) -> str:
    """A candidate's id: the hash of its six sets in order (order decides nothing in a
    selection, but keeping it keeps the base team's slots readable)."""
    blob = json.dumps([set_key(m) for m in members], ensure_ascii=False).encode("utf-8")
    return "x" + hashlib.sha256(blob).hexdigest()[:15]


def distinct_sets(teams: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """Every distinct set in the pool with the id of the first team that has it."""
    seen: dict[str, tuple[dict[str, Any], str]] = {}
    for team in teams:
        for member in team["team"]:
            seen.setdefault(set_key(member), (member, team["id"]))
    return list(seen.values())


def base_values(store: Path, n: int) -> list[float]:
    """Each team's mean selection value over all n teams (its mirror counts 0.5)."""
    total = [0.0] * n
    for a in range(n):
        for b in range(a, n):
            path = store / f"{a:03d}-{b:03d}.json"
            if not path.exists():
                raise FileNotFoundError(f"{path}: the store is not complete")
            v = float(json.loads(path.read_bytes())["entry"]["value"])
            if a == b:
                total[a] += 0.5
            else:
                total[a] += v
                total[b] += 1.0 - v
    return [t / n for t in total]


def neighbors(
    data: dict[str, Any], base_ids: list[str], swaps: int,
    extra: list[dict[str, Any]] | None = None, new_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Candidates ``swaps`` (1 or 2) members away from each base team that pass the
    roster checks, with counts of what was refused.

    ``extra`` teams (IKA-450: tournament teams, or candidates found before) add their sets
    to the ones swapped in and may be bases. With ``new_only``, every set swapped in is one
    no team of ``data`` has, so a search the pool's own sets already covered is not redone."""
    if swaps not in (1, 2):
        raise ValueError("swaps is 1 or 2")
    extra = extra or []
    teams = {t["id"]: t for t in [*data["teams"], *extra]}
    pool_keys = {tuple(sorted(play_key(m) for m in t["team"])) for t in [*data["teams"], *extra]}
    sets = distinct_sets([*data["teams"], *extra])
    if new_only:
        old = {play_key(m) for t in data["teams"] for m in t["team"]}
        sets = [(m, src) for m, src in sets if play_key(m) not in old]
    from pokeuraou.regulation import load_regulation

    reg = load_regulation(str(data["regulation"]))
    out: dict[str, dict[str, Any]] = {}
    counts = {"tried": 0, "refused": 0, "duplicate": 0, "poolTeam": 0}
    played: set[tuple[str, ...]] = set()

    def consider(base: dict[str, Any], members: list[dict[str, Any]], swap: list[dict]) -> None:
        counts["tried"] += 1
        key = tuple(sorted(play_key(m) for m in members))
        if key in pool_keys:
            counts["poolTeam"] += 1
            return
        if key in played:
            counts["duplicate"] += 1
            return
        cid = team_id(sorted(members, key=set_key))
        entry = {"id": cid, "name": f"{base['name']} / " + ", ".join(
            f"{s['out']} -> {s['in']}" for s in swap), "team": members,
            "base": base["id"], "swap": swap}
        try:
            roster_from_data({**entry, "regulation": data["regulation"]}, default_id=cid, reg=reg)
        except (TeamError, KeyError, ValueError):
            counts["refused"] += 1
            return
        out[cid] = entry
        played.add(key)

    for bid in base_ids:
        base = teams[bid]
        own = {play_key(m) for m in base["team"]}
        for slot, old in enumerate(base["team"]):
            for new, source in sets:
                if play_key(new) in own:
                    continue
                members = list(base["team"])
                members[slot] = new
                first = {"slot": slot, "out": old["species"], "in": new["species"], "from": source}
                if swaps == 1:
                    consider(base, members, [first])
                    continue
                for slot2 in range(slot + 1, 6):
                    for new2, source2 in sets:
                        if play_key(new2) in own or play_key(new2) == play_key(new):
                            continue
                        second = list(members)
                        second[slot2] = new2
                        consider(base, second, [first, {
                            "slot": slot2, "out": base["team"][slot2]["species"],
                            "in": new2["species"], "from": source2}])
    return list(out.values()), counts


def combine(
    data: dict[str, Any], base: dict[str, Any], singles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """IKA-447: two-member swaps built from one base's one-member swaps ``singles`` (each a
    candidate whose ``base`` is ``base`` and whose ``swap`` has one entry): every pair on two
    different slots that passes the roster checks. A full two-swap neighbourhood is about
    150,000 teams a base; pairing only the singles the value already liked keeps it small."""
    from pokeuraou.regulation import load_regulation

    reg = load_regulation(str(data["regulation"]))
    pool_keys = {tuple(sorted(play_key(m) for m in t["team"])) for t in data["teams"]}
    out: dict[str, dict[str, Any]] = {}
    counts = {"tried": 0, "refused": 0, "duplicate": 0, "poolTeam": 0, "sameSlot": 0}
    played: set[tuple[str, ...]] = set()
    for i, a in enumerate(singles):
        for b in singles[i + 1:]:
            sa, sb = a["swap"][0], b["swap"][0]
            if a["base"] != base["id"] or b["base"] != base["id"]:
                raise ValueError("every single must be a swap of the base")
            if sa["slot"] == sb["slot"]:
                counts["sameSlot"] += 1
                continue
            counts["tried"] += 1
            members = list(base["team"])
            members[sa["slot"]] = a["team"][sa["slot"]]
            members[sb["slot"]] = b["team"][sb["slot"]]
            key = tuple(sorted(play_key(m) for m in members))
            if key in pool_keys:
                counts["poolTeam"] += 1
                continue
            if key in played:
                counts["duplicate"] += 1
                continue
            swap = sorted([sa, sb], key=lambda s: s["slot"])
            cid = team_id(sorted(members, key=set_key))
            entry = {"id": cid, "name": f"{base['name']} / " + ", ".join(
                f"{s['out']} -> {s['in']}" for s in swap), "team": members,
                "base": base["id"], "swap": swap, "singles": [a["id"], b["id"]]}
            try:
                roster_from_data({**entry, "regulation": data["regulation"]}, default_id=cid, reg=reg)
            except (TeamError, KeyError, ValueError):
                counts["refused"] += 1
                continue
            out[cid] = entry
            played.add(key)
    return list(out.values()), counts


def pool_with(data: dict[str, Any], candidates: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """A pool file holding the pool's teams and ``candidates`` (each kept with its base and
    swaps, which the loader ignores)."""
    out = dict(data)
    out["id"] = name
    out["notes"] = (f"IKA-443: the pool's {len(data['teams'])} teams and {len(candidates)} "
                    "recombined candidates")
    out["teams"] = list(data["teams"]) + list(candidates)
    return out


# -- value: the selection game against every pool team ------------------------------------


def _value_worker(args: argparse.Namespace) -> int:
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.inference import RemoteValue
    from pokeuraou.selection import SpreadClass, solve_selection

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    value = RemoteValue(args.addr, "value", Encoder(reg))
    cands = json.loads(Path(args.candidates).read_bytes())["candidates"]
    if args.ids:
        keep = set(args.ids.split(","))
        cands = [c for c in cands if c["id"] in keep]
    opponents = list(range(len(pool.teams)))
    if args.subset:
        import numpy as np

        opponents = sorted(np.random.default_rng(args.subset_seed).choice(
            len(pool.teams), size=args.subset, replace=False).tolist())
    out = Path(args.out)
    done = 0
    for i, cand in enumerate(cands):
        if i % args.workers != args.worker:
            continue
        path = out / f"{cand['id']}.json"
        if path.exists():
            continue
        roster = roster_from_data({**cand, "regulation": reg.meta.format_id}, default_id=cand["id"], reg=reg)
        values = []
        for o in opponents:
            opp = pool.teams[o]
            analysis = solve_selection(
                reg, roster.sets, [SpreadClass(weight=1.0, sets=tuple(opp.sets), label="sheet")], value
            )
            values.append(analysis.value)
        _write(path, {"id": cand["id"], "model": args.model_name, "pool": pool.sha256,
                      "opponents": [pool.teams[o].id for o in opponents], "values": values,
                      "mean": sum(values) / len(values)})
        done += 1
    print(f"worker {args.worker}: {done} candidates", flush=True)
    return 0


def _value(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = str(_pool_data(args.pool)["regulation"])
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    servers, addrs = [], []
    for i in range(args.servers):
        log = open(out / f"server{i}.log", "w", encoding="utf-8")  # noqa: SIM115
        p = subprocess.Popen(
            [sys.executable, str(ROOT / "tools/inference_server.py"), "--device", "cuda",
             "--regulation", reg, "--arm", "value", *args.model],
            env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=log, text=True)
        servers.append(p)
        addrs.append(p.stdout.readline().strip())
    name = "+".join(Path(m).stem for m in args.model)
    kids = [
        subprocess.Popen(
            [sys.executable, __file__, "value-worker", "--pool", args.pool, "--candidates",
             args.candidates, "--out", str(out), "--workers", str(args.workers), "--worker",
             str(k), "--addr", addrs[k % len(addrs)], "--model-name", name,
             "--subset", str(args.subset), "--subset-seed", str(args.subset_seed),
             *(["--ids", args.ids] if args.ids else [])],
            env=env, cwd=str(ROOT))
        for k in range(args.workers)
    ]
    rc = max(k.wait() for k in kids)
    for p in servers:
        p.terminate()
    print(f"files {len(list(out.glob('x*.json')))} rc {rc}", flush=True)
    return rc


def _report(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base).read_bytes())
    cands = json.loads(Path(args.candidates).read_bytes())["candidates"]
    rows = []
    for cand in cands:
        path = Path(args.values) / f"{cand['id']}.json"
        if path.exists():
            v = json.loads(path.read_bytes())
            rows.append({"id": cand["id"], "name": cand["name"], "base": cand["base"],
                         "mean": v["mean"]})
    rows.sort(key=lambda r: -r["mean"])
    best_pool = max(base["teams"], key=lambda t: t["value"])
    _write(Path(args.out), {"valued": len(rows), "of": len(cands), "bestPool": best_pool,
                            "top": rows[: args.top]})
    print(f"{len(rows)} of {len(cands)} valued; best pool team {best_pool['name']} {best_pool['value']:.4f}")
    for r in rows[: args.top]:
        print(f"  {r['mean']:.4f}  {r['id']}  {r['name']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base")
    b.add_argument("--pool", default="regmc-matchupweb")
    b.add_argument("--store", required=True)
    b.add_argument("--out", required=True)
    n = sub.add_parser("neighbors")
    n.add_argument("--pool", default="regmc-matchupweb")
    n.add_argument("--base", help="base.json: the --bases teams with the highest value")
    n.add_argument("--bases", type=int, default=5)
    n.add_argument("--base-ids", nargs="+", default=[], help="these bases instead of --base")
    n.add_argument("--extra", nargs="+", default=[],
                   help="pool or candidate files whose sets join the swaps (their teams may be bases)")
    n.add_argument("--new-only", action="store_true",
                   help="swap in only sets that no team of --pool has")
    n.add_argument("--swaps", type=int, default=1, choices=(1, 2))
    n.add_argument("--out", required=True)
    c = sub.add_parser("combine", help="IKA-447: pair a base's best one-member swaps")
    c.add_argument("--pool", default="regmc-matchupweb")
    c.add_argument("--base-id", required=True)
    c.add_argument("--base-file", default=None,
                   help="a pool or candidate file holding the base, if the pool does not")
    c.add_argument("--singles", nargs="+", required=True, help="candidate files of one-member swaps")
    c.add_argument("--values", nargs="+", required=True, help="value directories for the singles")
    c.add_argument("--top", type=int, default=40, help="pair this many singles, best value first")
    c.add_argument("--out", required=True)
    v = sub.add_parser("value")
    v.add_argument("--pool", default="regmc-matchupweb")
    v.add_argument("--candidates", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--workers", type=int, default=14)
    v.add_argument("--servers", type=int, default=2)
    v.add_argument("--model", nargs="+", required=True)
    w = sub.add_parser("value-worker")
    for flag in ("--pool", "--candidates", "--out", "--addr", "--model-name"):
        w.add_argument(flag, required=True)
    w.add_argument("--workers", type=int, required=True)
    w.add_argument("--worker", type=int, required=True)
    for parser in (v, w):
        parser.add_argument("--subset", type=int, default=0,
                            help="value against this many pool teams drawn with --subset-seed "
                            "(0: all of them)")
        parser.add_argument("--subset-seed", type=int, default=0)
        parser.add_argument("--ids", default="", help="only these candidate ids (comma-separated)")
    r = sub.add_parser("report")
    for flag in ("--base", "--candidates", "--values", "--out"):
        r.add_argument(flag, required=True)
    r.add_argument("--top", type=int, default=20)
    p = sub.add_parser("pool-with")
    p.add_argument("--pool", default="regmc-matchupweb")
    p.add_argument("--candidates", required=True)
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "base":
        data = _pool_data(args.pool)
        values = base_values(Path(args.store), len(data["teams"]))
        teams = [{"id": t["id"], "name": t["name"], "value": v}
                 for t, v in zip(data["teams"], values, strict=True)]
        _write(Path(args.out), {"pool": args.pool, "store": str(args.store), "teams": teams})
        for t in sorted(teams, key=lambda t: -t["value"])[:10]:
            print(f"  {t['value']:.4f}  {t['id']}  {t['name']}")
        return 0
    if args.cmd == "neighbors":
        data = _pool_data(args.pool)
        if args.base_ids:
            ids = list(args.base_ids)
        elif args.base:
            base = json.loads(Path(args.base).read_bytes())
            ids = [t["id"] for t in sorted(base["teams"], key=lambda t: -t["value"])[: args.bases]]
        else:
            ap.error("neighbors needs --base or --base-ids")
        extra: list[dict[str, Any]] = []
        for path in args.extra:
            got = json.loads(Path(path).read_bytes().decode("utf-8"))
            extra += got.get("teams") or got.get("candidates") or []
        cands, counts = neighbors(data, ids, args.swaps, extra=extra, new_only=args.new_only)
        _write(Path(args.out), {"pool": args.pool, "bases": ids, "swaps": args.swaps,
                                "extra": args.extra, "newOnly": args.new_only,
                                "counts": counts, "candidates": cands})
        print(f"{len(cands)} candidates from {ids}: {counts}")
        return 0
    if args.cmd == "combine":
        data = _pool_data(args.pool)
        teams = {t["id"]: t for t in data["teams"]}
        if args.base_file:
            got = json.loads(Path(args.base_file).read_bytes().decode("utf-8"))
            teams.update({t["id"]: t for t in got.get("teams") or got.get("candidates") or []})
        base = teams[args.base_id]
        valued: dict[str, tuple[float, dict[str, Any]]] = {}
        for path in args.singles:
            for cand in json.loads(Path(path).read_bytes().decode("utf-8"))["candidates"]:
                if cand.get("base") != base["id"] or len(cand.get("swap", [])) != 1:
                    continue
                for d in args.values:
                    vpath = Path(d) / f"{cand['id']}.json"
                    if vpath.exists():
                        valued[cand["id"]] = (float(json.loads(vpath.read_bytes())["mean"]), cand)
                        break
        ranked = [c for _v, c in sorted(valued.values(), key=lambda vc: (-vc[0], vc[1]["id"]))]
        cands, counts = combine(data, base, ranked[: args.top])
        _write(Path(args.out), {"pool": args.pool, "base": base["id"], "top": args.top,
                                "singlesValued": len(valued), "counts": counts, "candidates": cands})
        print(f"{len(cands)} two-swaps of {base['id']} from the top {args.top} "
              f"of {len(valued)} singles: {counts}")
        return 0
    if args.cmd == "value":
        return _value(args)
    if args.cmd == "value-worker":
        return _value_worker(args)
    if args.cmd == "report":
        return _report(args)
    if args.cmd == "pool-with":
        data = _pool_data(args.pool)
        cands = {c["id"]: c for c in json.loads(Path(args.candidates).read_bytes())["candidates"]}
        _write(Path(args.out), pool_with(data, [cands[i] for i in args.ids], args.name))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
