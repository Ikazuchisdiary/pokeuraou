"""Show a recorded game on the page again, or read its frames (IKA-332).

    uv run python tools/live_view.py data/human/live.bin                # the page, at the pace it was played
    uv run python tools/live_view.py data/human/live.bin --speed 4      # four times as fast
    uv run python tools/live_view.py data/human/live.bin --dump | head  # the frames as JSON lines

The file is what ``tools/play_human.py --live-out`` keeps (`pokeuraou.liveview.FileSink`):
the page's own frames with the seconds each was sent at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import liveview  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8332)
    ap.add_argument("--speed", type=float, default=1.0, help="0 sends everything at once")
    ap.add_argument("--dump", action="store_true", help="print the frames as JSON lines instead")
    args = ap.parse_args(argv)
    frames = list(liveview.read_record(args.record))
    if args.dump:
        decoder = liveview.Decoder()
        for seconds, frame in frames:
            got = decoder.feed(frame)
            if got is not None:
                sys.stdout.write(json.dumps({"t": round(seconds, 4), **got}, ensure_ascii=False) + "\n")
        return
    server = liveview.LiveServer(args.host, args.port).start()
    print(f"画面: {server.url}（{len(frames)} フレーム）", file=sys.stderr)
    try:
        input("ページを開いたら Enter で再生 > ")
        liveview.Replay(server, frames, speed=args.speed).run()
        input("再生した。閉じるには Enter > ")
    except (EOFError, KeyboardInterrupt):
        pass
    server.close()


if __name__ == "__main__":
    main()
