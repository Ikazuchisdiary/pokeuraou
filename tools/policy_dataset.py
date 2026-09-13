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

from pokeuraou.actions import SideAction, side_actions, target_names  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.narrow import _bridged_scores, score_action  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402

#: Per slot: is it a move, a switch, or nothing; which move; where it points; is it a mega.
#: Kept deliberately small and cheap -- anything here has to be computable at search time,
#: for every candidate, without touching the resolver.
SLOT_FEATURES = 12
#: Features describing the pair as a whole.
PAIR_FEATURES = 4
#: The damage score `narrow` already computes, plus where it ranks the action. Both are
#: free at search time -- `_bridged_scores` runs on every narrow call whatever the ordering
#: is -- and they are the only position-specific thing here, so a model without them can
#: only learn which *shapes* of action tend to be played.
SCORE_FEATURES = 4
#: The position, summarised. Cheap counts rather than the encoder's arrays: this asks
#: whether a small model on free features can beat the leaf ordering, not whether a large
#: one can.
POSITION_FEATURES = 6
FEATURES = 2 * SLOT_FEATURES + PAIR_FEATURES + SCORE_FEATURES + POSITION_FEATURES

#: Per slot: who is acting, which move, what it switches to, and what it points at -- as
#: vocabulary indices, for the model to embed. The first attempt had only the move's *slot
#: number*, which says nothing across different Pokemon, and lost to the leaf ordering by
#: twenty points. Identity is the thing that was missing.
IDS_PER_SLOT = 4
IDS = 2 * IDS_PER_SLOT


def _slot_features(action: SideAction, index: int, pos: Position, side: int) -> np.ndarray:
    out = np.zeros(SLOT_FEATURES, dtype=np.float32)
    slots = action.slots
    if index >= len(slots):
        out[0] = 1.0  # absent
        return out
    slot = slots[index]
    kind = type(slot).__name__
    move_index = getattr(slot, "move_index", None)
    target = getattr(slot, "target", None)
    out[1] = float(kind == "MoveAction")
    out[2] = float(kind == "SwitchAction")
    out[3] = float(getattr(slot, "mega", False))
    if move_index is not None:
        # Move slots are one-hot: a move's identity matters more than its number, but the
        # number is what is free here, and the position tells the model which move it is.
        out[4 + min(max(int(move_index) - 1, 0), 3)] = 1.0
    if target is not None:
        out[8] = float(int(target) > 0)      # aimed at the far side
        out[9] = float(int(target) < 0)      # aimed at our own side
        out[10] = float(abs(int(target)) == 2)
    else:
        out[11] = 1.0                        # spread, or self, or no target
    return out


def _pair_features(action: SideAction, pos: Position, side: int) -> np.ndarray:
    out = np.zeros(PAIR_FEATURES, dtype=np.float32)
    kinds = [type(s).__name__ for s in action.slots]
    out[0] = float(kinds.count("SwitchAction"))
    out[1] = float(any(getattr(s, "mega", False) for s in action.slots))
    targets = [getattr(s, "target", None) for s in action.slots]
    known = [int(t) for t in targets if t is not None]
    # Both slots aimed at the same opponent -- focusing fire is a real doubles decision and
    # the damage score cannot express it, because it scores the slots separately.
    out[2] = float(len(known) == 2 and known[0] == known[1] and known[0] > 0)
    out[3] = float(len(known) == 2 and known[0] != known[1] and min(known) > 0)
    return out


def _position_features(pos: Position, side: int, turn: int) -> np.ndarray:
    out = np.zeros(POSITION_FEATURES, dtype=np.float32)
    ours, theirs = pos.sides[side], pos.sides[1 - side]
    out[0] = sum(1 for m in ours.pokemon if not m.fainted) / 4.0
    out[1] = sum(1 for m in theirs.pokemon if not m.fainted) / 4.0
    alive_ours = [m for m in ours.pokemon if not m.fainted]
    alive_theirs = [m for m in theirs.pokemon if not m.fainted]
    out[2] = (
        float(np.mean([m.hp / max(m.maxhp, 1) for m in alive_ours])) if alive_ours else 0.0
    )
    out[3] = (
        float(np.mean([m.hp / max(m.maxhp, 1) for m in alive_theirs]))
        if alive_theirs
        else 0.0
    )
    out[4] = min(turn, 30) / 30.0
    out[5] = float(ours.mega_used)
    return out


def ids_for(
    encoder: Encoder,
    pos: Position,
    side: int,
    actions: list[SideAction],
) -> np.ndarray:
    """Vocabulary indices for each action: actor, move, switch target, aim."""
    vocab = encoder.vocab
    names = target_names(pos, side)
    active = pos.sides[side].active_pokemon()
    out = np.zeros((len(actions), IDS), dtype=np.int32)
    for row, action in enumerate(actions):
        for index, slot in enumerate(action.slots[:2]):
            base = index * IDS_PER_SLOT
            actor = active[index] if index < len(active) else None
            if actor is not None:
                out[row, base] = vocab.species.get(actor.species, 0)
            move_id = getattr(slot, "move_id", None)
            if move_id is not None:
                out[row, base + 1] = vocab.moves.get(move_id, 0)
            switch_to = getattr(slot, "species", None)
            if switch_to is not None:
                out[row, base + 2] = vocab.species.get(switch_to, 0)
            target = getattr(slot, "target", None)
            if target is not None:
                aimed = names.species_for(int(target))
                if aimed is not None:
                    out[row, base + 3] = vocab.species.get(aimed, 0)
    return out


def features_for(
    reg: Regulation,
    pos: Position,
    side: int,
    actions: list[SideAction],
    scores: list[float] | None = None,
    turn: int = 0,
):
    rows = np.zeros((len(actions), FEATURES), dtype=np.float32)
    position = _position_features(pos, side, turn)
    if scores is not None:
        values = np.asarray(scores, dtype=np.float32)
        order = np.argsort(np.argsort(-values))          # 0 is the best-scoring action
        spread = float(values.max() - values.min()) or 1.0
        normalised = (values - values.min()) / spread
    else:
        values = np.zeros(len(actions), dtype=np.float32)
        order = np.zeros(len(actions), dtype=np.int64)
        normalised = values
    base = 2 * SLOT_FEATURES + PAIR_FEATURES
    for i, action in enumerate(actions):
        rows[i, :SLOT_FEATURES] = _slot_features(action, 0, pos, side)
        rows[i, SLOT_FEATURES : 2 * SLOT_FEATURES] = _slot_features(action, 1, pos, side)
        rows[i, 2 * SLOT_FEATURES : base] = _pair_features(action, pos, side)
        rows[i, base] = values[i]
        rows[i, base + 1] = normalised[i]
        rows[i, base + 2] = min(int(order[i]), 47) / 47.0
        rows[i, base + 3] = float(order[i] == 0)
        rows[i, base + SCORE_FEATURES :] = position
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, default=2000, help="games to read")
    ap.add_argument("--min-turn", type=int, default=1)
    args = ap.parse_args()

    reg: Regulation | None = None
    encoder: Encoder | None = None
    identities: list[np.ndarray] = []
    menus: list[np.ndarray] = []
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
                        targets.append(policy / policy.sum())
                        ranks.append(np.arange(len(names), dtype=np.int16))
                        decisions += 1
                if games >= args.games:
                    break
        if games >= args.games:
            break

    if not menus:
        raise SystemExit("no usable decisions found")

    lengths = np.array([len(m) for m in menus], dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.concatenate(menus).astype(np.float32),
        ids=np.concatenate(identities).astype(np.int32),
        targets=np.concatenate(targets).astype(np.float32),
        ranks=np.concatenate(ranks).astype(np.int16),
        lengths=lengths,
        species_vocab=len(encoder.vocab.species) + 1,
        move_vocab=len(encoder.vocab.moves) + 1,
    )
    support = [int((t > 1e-9).sum()) for t in targets]
    print(f"{decisions:,} decisions from {games:,} games -> {args.out}")
    print(f"  menu {lengths.mean():.1f} wide on average, {lengths.sum():,} rows, "
          f"{FEATURES} features and {IDS} identities")
    print(f"  support {np.mean(support):.2f} actions on average, "
          f"max {max(support)}")
    if unmatched:
        print(f"  {unmatched:,} menus dropped: a recorded choice string had no legal "
              f"action to match (a position the record and the rules disagree about)")


if __name__ == "__main__":
    main()
