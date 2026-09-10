"""Downloads Smogon usage statistics for a Champions format.

Run at most once a month -- the stats are published monthly and the files are cached under
``data/priors/raw/``. The stats directory carries no explicit licence; it is a public
directory that third-party tools consume routinely. Raw files are gitignored and not
redistributed.

    uv run python tools/fetch_priors.py --month 2026-08 --cutoff 1760
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.priors import priors_dir  # noqa: E402

BASE = "https://www.smogon.com/stats"
USER_AGENT = "pokeuraou/0.0 (Champions doubles solver; monthly usage-stats fetch)"


def fetch(month: str, format_id: str, cutoff: int, *, force: bool = False) -> Path:
    out_dir = priors_dir() / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{format_id}-{cutoff}.json.gz"
    if out.exists() and not force:
        print(f"cached: {out} ({out.stat().st_size / 1e6:.2f} MB)")
        return out

    url = f"{BASE}/{month}/chaos/{format_id}-{cutoff}.json.gz"
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        payload = resp.read()
    out.write_bytes(payload)
    print(f"wrote {out} ({len(payload) / 1e6:.2f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default="2026-08", help="YYYY-MM")
    ap.add_argument("--format", dest="format_id", default="gen9championsvgc2026regmb")
    ap.add_argument(
        "--cutoff", type=int, default=1760, help="ladder rating cutoff: 0, 1500, 1630 or 1760"
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    fetch(args.month, args.format_id, args.cutoff, force=args.force)


if __name__ == "__main__":
    main()
