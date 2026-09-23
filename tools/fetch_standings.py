"""Caches a tournament's standings, teams included, from Reportworm's public API.

Why this replaces the curated archetype list as the primary source: the API carries every
entrant's actual six with ability, nature, item and all four moves. 395 real teams *are*
the tournament metagame, so sampling one uniformly needs no archetype abstraction and no
pairwise model -- both of which were approximations standing in for exactly this data.

What it does not carry is the SP spread, which is correct rather than missing: Champions
open team sheets reveal the nature and blank the investment, so the spread is hidden
information by design and keeps coming from usage, per position, in the belief layer.

Same handling as the Smogon usage data: fetched manually, cached locally, identified by an
explicit User-Agent, and not redistributed -- the raw file is gitignored.

    uv run python tools/fetch_standings.py 2026 worlds
    uv run python tools/fetch_standings.py 2027 --list    # the season's events and formats

``--list`` is how a regulation's events are found: the season endpoint names every event
with its format ("Regulation M-C") and whether it has been processed. It writes nothing.
Whether an event's entries carry team lists is only visible in the event file itself.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.regulation import repo_root

BASE = "https://standings.reportworm.com/api/v1"
#: An explicit, identifying User-Agent, so the operator can see who is calling and block it
#: if they want to. The default urllib string is both anonymous and often refused.
USER_AGENT = (
    "pokeuraou/0.0 (Pokemon Champions doubles solver; personal research; "
    "one-off manual fetch)"
)


def standings_dir() -> Path:
    return repo_root() / "data" / "standings"


def fetch(url: str) -> bytes:
    """Fetches over TLS, falling back to curl when the local CA store is stale.

    On this machine Python's bundled trust store rejects the certificate as expired while
    curl and the browser accept it, so the problem is a stale local store rather than the
    host. Falling back to a client with a current store keeps verification on; disabling
    it would silently make every later fetch unauthenticated, which is a much worse
    trade for a tool that writes files other code then trusts.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.read()
    except urllib.error.URLError as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error.reason):
            raise
        curl = shutil.which("curl")
        if curl is None:
            raise SystemExit(
                f"TLS verification failed ({error.reason}) and curl is not available. "
                "The local CA store is out of date; update it rather than disabling "
                "verification."
            ) from error
        print(f"  urllib: {error.reason}; retrying with curl (verification still on)")
        done = subprocess.run(  # noqa: S603
            [curl, "-sS", "--fail", "--max-time", "90", "-A", USER_AGENT, url],
            capture_output=True,
            check=False,
        )
        if done.returncode != 0:
            raise SystemExit(f"curl failed: {done.stderr.decode(errors='replace')}") from error
        return done.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("season", nargs="?", default="2026")
    ap.add_argument("event", nargs="?", default="worlds")
    ap.add_argument("--usage", action="store_true", help="also fetch the event's own usage")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--list",
        action="store_true",
        help="print the season's events (dates, code, format, processed, players); no file",
    )
    args = ap.parse_args()

    if args.list:
        url = f"{BASE}/{args.season}"
        print(f"fetching {url}")
        try:
            events = json.loads(fetch(url))
        except urllib.error.HTTPError as error:
            raise SystemExit(f"{url}: HTTP {error.code}") from error
        for ev in sorted(events, key=lambda e: (str(e.get("start")), str(e.get("code")))):
            print(
                f"  {ev.get('start')}  {str(ev.get('code')):<15} {str(ev.get('format')):<17}"
                f" processed={str(bool(ev.get('processed'))):<5} "
                f"players={ev.get('playerCount', '-')!s:<5} {ev.get('name')}"
            )
        formats: dict[str, int] = {}
        for ev in events:
            formats[str(ev.get("format"))] = formats.get(str(ev.get("format")), 0) + 1
        print(f"{len(events)} events: " + ", ".join(f"{k} x{v}" for k, v in sorted(formats.items())))
        return

    out_dir = standings_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [("", f"{args.season}-{args.event}.json.gz")]
    if args.usage:
        targets.append(("/usage", f"{args.season}-{args.event}-usage.json.gz"))

    for suffix, filename in targets:
        path = out_dir / filename
        if path.exists() and not args.force:
            print(f"{path.name} already cached ({path.stat().st_size / 1e6:.2f} MB); --force to refetch")
            continue
        url = f"{BASE}/{args.season}/{args.event}{suffix}"
        print(f"fetching {url}")
        try:
            raw = fetch(url)
        except urllib.error.HTTPError as error:
            raise SystemExit(f"{url}: HTTP {error.code}") from error
        # Parse before writing, so a cached file is always valid JSON.
        parsed = json.loads(raw)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(parsed, handle, ensure_ascii=False)
        # The standings endpoint returns an object; the usage endpoint returns a bare list.
        event = parsed.get("event") or {} if isinstance(parsed, dict) else {}
        rows = (
            len(parsed)
            if isinstance(parsed, list)
            else len(parsed.get("standings") or parsed.get("usage") or {})
        )
        print(
            f"-> {path} ({path.stat().st_size / 1e6:.2f} MB)\n"
            f"   {event.get('name', '?')} / format {event.get('format', '?')} / "
            f"{event.get('playerCount', '?')} players / {rows} rows"
        )


if __name__ == "__main__":
    main()
