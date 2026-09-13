"""A learned ordering for `narrow`: which candidates the equilibrium will actually play.

`narrow` has to cut several hundred legal pairs down to a menu of a few dozen, and every
action it drops is an action the search cannot choose. Two orderings existed before this
one -- expected damage, and the leaf's own opinion of where a candidate leads -- and both
are heuristics about the equilibrium rather than predictions of it. This one is trained
directly on recorded equilibria: the features of a candidate go in, the weight the solved
equilibrium put on it comes out.

The features live here rather than in `tools/policy_dataset.py` because the model is only
correct if the rows it is shown at search time are built by the same code that built the
rows it was trained on. Two copies of that arithmetic would agree on the day they were
written and drift silently afterwards, and the failure would look like a model that got
worse for no reason.

`load_policy` returns `rank(pos, side, actions, scored=None)`. `narrow` wants
`rank(pool, scored)`, so `policy_ranking` binds the position and the side -- the same shape
`search.leaf_ranking` has, and for the same reason: a menu belongs to the agent that built
it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .actions import SideAction, target_names
from .position import Position
from .regulation import Regulation

if TYPE_CHECKING:  # pragma: no cover - imports for annotations only
    from .encode import Encoder

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

#: Columns of `ids_for` that index the species vocabulary, and the ones that index moves.
#: The loader embeds them with two different tables, so the split is a property of the
#: layout above and belongs next to it rather than in the model's forward pass.
SPECIES_IDS = [0, 2, 3, 4, 6, 7]
MOVE_IDS = [1, 5]


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


def load_policy(
    path: str | Path,
    reg: Regulation,
    device_name: str = "cpu",
    value_net: Any = None,
    value_name: str = "",
) -> Callable[..., np.ndarray]:
    """The learned ordering, as a function from (position, side, actions) to scores.

    On the CPU, whatever the value function is using. The model is small and a menu is
    about thirty-five rows, which on CUDA is 2.25 ms of pure launch latency -- flat from 34
    rows to 138 -- against 0.14 ms on the CPU. Sixteen times, for the same arithmetic, and
    it is the same fact as everything else in `rust/README.md`: a small batch never reaches
    the card's arithmetic at all.

    ``value_net`` supplies the frozen position representation, and it has to be the *same*
    value function the policy was trained against. Every model in `data/models/` produces
    a `side_vectors` of the same width, so the wrong one loads cleanly, runs at full speed
    and orders candidates by a representation that means something else. ``value_name``
    is checked against the one the blob records when it records one; models trained before
    that field existed record nothing and are taken on the caller's word.
    """
    import torch
    from torch import nn

    from .encode import Encoder

    blob = torch.load(path, map_location="cpu", weights_only=False)
    encoder = Encoder(reg)
    device = torch.device(device_name)

    embed = 24
    species = nn.Embedding(int(blob["species_vocab"]), embed, padding_idx=0)
    moves = nn.Embedding(int(blob["move_vocab"]), embed, padding_idx=0)
    width = int(blob["hidden"])
    state = blob["state"]
    position_dims = int(blob.get("position_dims", 0) or 0)
    position = None
    position_width = 0
    if position_dims:
        if value_net is None:
            raise ValueError(
                "this policy was trained with a position representation, so it needs the "
                "value function that produced it -- pass a value net"
            )
        trained_with = str(blob.get("position_value") or "")
        if trained_with and value_name and trained_with != value_name:
            raise ValueError(
                f"{Path(path).name} was trained on {trained_with}'s position "
                f"representation and was handed {value_name}'s. They are the same width, "
                f"so this would have run -- and ordered candidates by numbers that mean "
                f"something else."
            )
        position_width = state["position.0.weight"].shape[0]
        position = nn.Sequential(nn.Linear(position_dims, position_width), nn.ReLU())
        position.load_state_dict(
            {k[len("position.") :]: v for k, v in state.items()
             if k.startswith("position.")}
        )
    trunk = nn.Sequential(
        nn.Linear(int(blob["features"]) + 2 * 4 * embed + position_width, width),
        nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(width, width), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(width, 1),
    )
    species.load_state_dict({"weight": state["species.weight"]})
    moves.load_state_dict({"weight": state["moves.weight"]})
    trunk.load_state_dict(
        {k[len("trunk.") :]: v for k, v in state.items() if k.startswith("trunk.")}
    )
    for module in (species, moves, trunk):
        module.to(device).eval()
    if position is not None:
        position.to(device).eval()

    # The value function stays wherever it already is -- it is doing the leaf work and
    # is not this model's to move -- so the embedding is computed there and carried over.
    value_device = next(value_net.parameters()).device if value_net is not None else device

    @torch.no_grad()
    def embed_position(pos: Position, side: int):
        """The value function's own view of the position, from the acting side."""
        encoded = encoder.encode_positions([pos])
        batch = {
            name: torch.from_numpy(getattr(encoded, name)).to(value_device)
            for name in ("species", "ability", "item", "moves", "mon", "mask", "side",
                         "field")
        }
        sides = value_net.side_vectors(batch)
        return torch.cat(
            [sides[0, side], sides[0, 1 - side], batch["field"][0]], dim=-1
        ).unsqueeze(0).to(device)

    @torch.no_grad()
    def rank(pos: Position, side: int, actions, scored=None) -> np.ndarray:
        # `narrow` computes the damage candidates before it calls a ranker and now hands
        # them over, so there is no second crossing to the port for them.
        if scored is None:
            from .narrow import _bridged_scores, score_action

            scored = _bridged_scores(reg, pos, side, list(actions))
            if scored is None:
                scored = [score_action(reg, pos, side, a, battlers=None) for a in actions]
        by_choice = {c.action.to_choice(): c.score for c in scored}
        scores = [by_choice.get(a.to_choice(), 0.0) for a in actions]
        x = torch.from_numpy(
            features_for(reg, pos, side, list(actions), scores)
        ).to(device)
        k = torch.from_numpy(
            ids_for(encoder, pos, side, list(actions)).astype("int64")
        ).to(device)
        parts = [
            x,
            species(k[:, SPECIES_IDS]).flatten(-2),
            moves(k[:, MOVE_IDS]).flatten(-2),
        ]
        if position is not None:
            summary = position(embed_position(pos, side))
            parts.append(summary.expand(x.shape[0], summary.shape[-1]))
        return trunk(torch.cat(parts, dim=-1)).squeeze(-1).cpu().numpy()

    return rank


def policy_ranking(
    rank: Callable[..., np.ndarray], pos: Position, side: int
) -> Callable[..., np.ndarray]:
    """Bind a loaded policy to one position and side, in the shape `narrow` wants.

    The same adaptation `search.leaf_ranking` performs, and for the same reason: `narrow`
    calls its ranker with the pool it is about to cut and the damage candidates it has
    already computed, and knows nothing about which position it is in.
    """

    def bound(pool: list[SideAction], scored: object = None) -> np.ndarray:
        if not pool:
            return np.zeros(0)
        return rank(pos, side, pool, scored)

    return bound


__all__ = [
    "FEATURES",
    "IDS",
    "features_for",
    "ids_for",
    "load_policy",
    "policy_ranking",
]
