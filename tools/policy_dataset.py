"""Turning recorded games into training data for an ordering.

Every recorded decision already carries what a policy needs to learn from: the menu it
chose from (`ownActions`) and the equilibrium's mixed strategy over that menu
(`ownPolicy`). Nothing has to be generated -- 85,228 decisions and 217,769 played actions
sit in gen-7 alone, and the search side can keep its machine.

What the ordering has to do is put the support at the top. Measured on those recordings,
the leaf ordering -- the better of the two that exist, and the one gen-7 was generated with
-- gets 73.9% of the equilibrium's weight into its top six, where a perfect ordering would
get 100% because the support is never larger than six. That gap is the whole prize:
`tools/policy_ceiling.py` prices it at 36 cells against 2,304.

The actions are recovered rather than parsed. A recorded action is a Showdown choice string
like `move 2 2, move 1`, and rebuilding the position's legal actions gives back real
`SideAction` objects whose `to_choice()` matches -- so the features come from the objects,
and a string this cannot match is dropped and counted rather than guessed at.

    uv run --group learn python tools/policy_dataset.py --games-dir data/selfplay-gen7 \\
        --out data/policy-gen7.npz --games 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.narrow import _bridged_scores, score_action  # noqa: E402

# The feature layout lives in `pokeuraou.policy` because the search uses it too, and a
# model is only correct if the rows it is shown at search time are built by the same
# code that built the rows it was trained on.
from pokeuraou.policy import (  # noqa: E402
    FEATURES,
    IDS,
    features_for,
    ids_for,
)
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402


def _load_trunk(path: Path, encoder: Encoder, device_name: str):
    """A frozen position representation: what the value function's own head is given.

    `side_vectors` is (B, 2, side_dim) and the head sees `[ours, theirs, field]`, so that
    is what an ordering should see too -- the position as the function that fills the
    matrix understands it. The weights are frozen: this asks whether the representation
    helps, not whether it can be improved.
    """
    import torch

    from pokeuraou.value import load_model

    net, _meta = load_model(path, encoder)
    device = torch.device(device_name)
    net = net.to(device).eval()

    @torch.no_grad()
    def embed(items: list[tuple[Position, int]]) -> np.ndarray:
        positions = [p for p, _side in items]
        encoded = encoder.encode_positions(positions)
        batch = {
            "species": torch.from_numpy(encoded.species),
            "ability": torch.from_numpy(encoded.ability),
            "item": torch.from_numpy(encoded.item),
            "moves": torch.from_numpy(encoded.moves),
            "mon": torch.from_numpy(encoded.mon),
            "mask": torch.from_numpy(encoded.mask),
            "side": torch.from_numpy(encoded.side),
            "field": torch.from_numpy(encoded.field),
        }
        batch = {k: v.to(device) for k, v in batch.items()}
        sides = net.side_vectors(batch)
        index = torch.tensor([s for _p, s in items], device=device)
        rows = torch.arange(len(items), device=device)
        ours = sides[rows, index]
        theirs = sides[rows, 1 - index]
        return torch.cat([ours, theirs, batch["field"]], dim=-1).cpu().numpy()

    return embed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, default=2000, help="games to read")
    ap.add_argument("--min-turn", type=int, default=1)
    ap.add_argument(
        "--value",
        type=Path,
        default=None,
        help="a trained value function, used frozen as a position representation. Its "
        "`side_vectors` is what its own head consumes, so it is the position as the thing "
        "that fills the matrix sees it -- and the ordering's job is to guess what that "
        "will conclude.",
    )
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--chunk", type=int, default=2048, help="positions encoded at once")
    args = ap.parse_args()

    reg: Regulation | None = None
    encoder: Encoder | None = None
    identities: list[np.ndarray] = []
    menus: list[np.ndarray] = []
    #: (position, acting side) per menu, embedded in chunks so the positions are not all
    #: held at once.
    pending: list[tuple[Position, int]] = []
    embeddings: list[np.ndarray] = []
    net = None
    targets: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    unmatched = 0
    decisions = 0
    games = 0

    for path in sorted(args.games_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                games += 1
                for turn in record.get("decisions", ()):
                    if turn.get("kind") != "move":
                        continue
                    if int(turn.get("turn", 0)) < args.min_turn:
                        continue
                    pos = Position.from_json(turn["position"])
                    if reg is None:
                        reg = load_regulation(pos.format)
                        register_mega_stones(reg)
                        encoder = Encoder(reg)
                    for side, (names, weights) in enumerate(
                        (
                            (turn["ownActions"], turn["ownPolicy"]),
                            (turn["foeActions"], turn["foePolicy"]),
                        )
                    ):
                        policy = np.asarray(weights, dtype=np.float32)
                        if policy.sum() <= 0 or len(names) < 2:
                            continue
                        pool = side_actions(reg, pos, side)
                        legal = {a.to_choice(): a for a in pool}
                        chosen = [legal.get(n) for n in names]
                        if any(a is None for a in chosen):
                            unmatched += 1
                            continue
                        scored = _bridged_scores(reg, pos, side, pool)
                        if scored is None:
                            scored = [
                                score_action(reg, pos, side, a, battlers=None)
                                for a in pool
                            ]
                        by_choice = {c.action.to_choice(): c.score for c in scored}
                        scores = [by_choice.get(n, 0.0) for n in names]
                        menus.append(
                            features_for(
                                reg, pos, side, chosen, scores,
                                turn=int(turn.get("turn", 0)),
                            )
                        )
                        identities.append(ids_for(encoder, pos, side, chosen))
                        if args.value is not None:
                            pending.append((pos, side))
                            if len(pending) >= args.chunk:
                                if net is None:
                                    net = _load_trunk(args.value, encoder, args.device)
                                embeddings.append(net(pending))
                                pending = []
                        targets.append(policy / policy.sum())
                        ranks.append(np.arange(len(names), dtype=np.int16))
                        decisions += 1
                if games >= args.games:
                    break
        if games >= args.games:
            break

    if pending:
        if net is None:
            net = _load_trunk(args.value, encoder, args.device)
        embeddings.append(net(pending))
        pending = []

    if not menus:
        raise SystemExit("no usable decisions found")

    lengths = np.array([len(m) for m in menus], dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if embeddings:
        stacked = np.concatenate(embeddings).astype(np.float16)
        if len(stacked) != len(lengths):
            raise SystemExit(
                f"{len(stacked)} embeddings for {len(lengths)} menus -- they are per "
                f"menu and must line up"
            )
        extra["position"] = stacked
        # Which value function produced those 408 numbers. Nothing downstream can work it
        # out -- every model in `data/models/` has the same `side_vectors` width, so a
        # policy handed the wrong one at search time gets a well-shaped representation
        # that means something else, and neither the loader nor the game would say so.
        extra["value_model"] = np.array(args.value.stem)
    np.savez_compressed(
        args.out,
        features=np.concatenate(menus).astype(np.float32),
        ids=np.concatenate(identities).astype(np.int32),
        **extra,
        targets=np.concatenate(targets).astype(np.float32),
        ranks=np.concatenate(ranks).astype(np.int16),
        lengths=lengths,
        species_vocab=len(encoder.vocab.species) + 1,
        move_vocab=len(encoder.vocab.moves) + 1,
    )
    support = [int((t > 1e-9).sum()) for t in targets]
    print(f"{decisions:,} decisions from {games:,} games -> {args.out}")
    print(f"  menu {lengths.mean():.1f} wide on average, {lengths.sum():,} rows, "
          f"{FEATURES} features and {IDS} identities"
          + (f", {extra['position'].shape[1]} position dims" if extra else ""))
    print(f"  support {np.mean(support):.2f} actions on average, "
          f"max {max(support)}")
    if unmatched:
        print(f"  {unmatched:,} menus dropped: a recorded choice string had no legal "
              f"action to match (a position the record and the rules disagree about)")


if __name__ == "__main__":
    main()
