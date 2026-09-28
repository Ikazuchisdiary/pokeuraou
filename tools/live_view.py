"""Show a recorded game on the page again, or read its frames (IKA-332).

    uv run python tools/live_view.py data/human/live.bin                # the page, at the pace it was played
    uv run python tools/live_view.py data/human/live.bin --speed 4      # four times as fast
    uv run python tools/live_view.py data/human/live.bin --dump | head  # the frames as JSON lines

The file is what ``tools/play_human.py --live-out`` keeps (`pokeuraou.liveview.FileSink`):
the page's own frames with the seconds each was sent at. A record kept before IKA-345 (the
labels were plain strings then) is read into today's frames as it is loaded
(`pokeuraou.liveview.upgrade_record`): its labels become one part a slot, and its chance
branches keep their one-line text (the draws and the field were not recorded).
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
    ap.add_argument("--sprite-url", default=None,
                    help="where the page takes images, {id} = Showdown's sprite id "
                    "(default: Showdown's server; \"\" for none)")
    ap.add_argument("--analysis-url", default=None,
                    help="the analysis page's address, for the page's links to it (default: the "
                    "launcher's port 8337 on the same host)")
    ap.add_argument("--dump", action="store_true", help="print the frames as JSON lines instead")
    args = ap.parse_args(argv)
    frames = list(liveview.read_record(args.record))
    frames, upgraded = liveview.upgrade_record(frames)
    if upgraded:
        print(f"IKA-345 より前の形の記録を今の形に直して読みます（{upgraded} フレーム）", file=sys.stderr)
    if args.dump:
        decoder = liveview.Decoder()
        for seconds, frame in frames:
            got = decoder.feed(frame)
            if got is not None:
                sys.stdout.write(json.dumps({"t": round(seconds, 4), **got}, ensure_ascii=False) + "\n")
        return
    server = liveview.LiveServer(
        args.host, args.port, sprite_url=args.sprite_url,
        links={"analysis-url": args.analysis_url} if args.analysis_url else None,
    ).start()
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
