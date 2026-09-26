"""A light candidate model: Q(s, a, b) for every pair of both sides' actions (IKA-274).

The leaf ranking (`search.leaf_ranking`) scores each candidate against two damage replies,
and IKA-310/311 measured what that misses: the width-12 menu leaks the equilibrium's
actions because which combination is needed is decided by the opponent's *mix*, and only
ranking against a mix solved in this position (IKA-311's `coverq`, AUC 0.92) cuts the
leak. This model is meant to supply that mix cheaply: it predicts the whole depth-1 matrix
M[a, b] of a position -- every legal pair of each side -- from the position and the two
action lists, so a small game on it gives the opponent's mix and each candidate's value
against it.

Three parts live here, and nothing on a shipped road imports them:

* **The views** (`decision_views`): which position a recorded decision's matrix is built
  on. Under a hidden bench each side ranked from its own heaviest completion
  (`selfplay._menus.views`, recorded as `rankViews`), so the teaching matrix of side s is
  built on side s's completion and the model's input is that same position. A side that
  sees the whole opposing four reads the true position; when both do, one matrix serves.
* **The action encoding** (`encode_actions`): each candidate pair tied to the rows of the
  position's `encode.Encoded` arrays -- who acts, with which move (a vocabulary id, the
  same table the leaf embeds), at whom, switching to whom, with or without mega. The
  leaf's own encoding is not touched.
* **The net** (`QNet`): the position through its own small per-Pokemon encoder, each
  action from the rows it names, and a pair head. Output is side 0's win probability, the
  matrix's own units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actions import MoveAction, SideAction, SwitchAction, side_actions
from .encode import Encoded, Vocabulary
from .position import Position

#: Columns of one slot of an encoded action.
ACTION_FIELDS = ("kind", "actor", "move", "mega", "target", "target_row", "switch_row")
KIND_PASS, KIND_MOVE, KIND_SWITCH = 0, 1, 2
TARGET_NONE, TARGET_FOE, TARGET_ALLY = 0, 1, 2
#: Slots in a side action (doubles).
SLOTS = 2


# --------------------------------------------------------------------------------------
# the pool and the views


def legal_pool(reg: Any, pos: Position, side: int) -> list[SideAction]:  # noqa: ANN401
    """Every candidate `narrow` would rank: `side_actions` less `drop_dead_actions`."""
    from .narrow import drop_dead_actions

    return drop_dead_actions(reg, pos, side, side_actions(reg, pos, side))


@dataclass(frozen=True, slots=True)
class View:
    """One position a teaching matrix is built on, and whose reading it is.

    ``side`` is the side that ranks from it (0 or 1), or 2 when both sides see the whole
    board and one position serves both. ``completion`` is the index into the completions
    of the other side's bench that the ranking read (`rankViews`), -1 when nothing hid.
    """

    side: int
    position: Position
    completion: int
    species: tuple[str, ...]


def decision_views(
    reg: Any,  # noqa: ANN401
    pool: Any,  # noqa: ANN401
    game: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> list[View]:
    """The positions each side of a recorded move decision ranked its menu from.

    Rebuilt as `play_game` built them: the other side's completions from its six on the
    pool's sheet, with the slots it has shown (`seen_slots` of the carried
    `shownIdentities`), and the recorded index picked. The species at that index must be
    the recorded ones, or this is not the completion the game read and it stops.
    """
    from .hidden import completions, seen_slots

    pos = Position.from_json(decision["position"])
    if game.get("information") != "hidden-bench":
        return [View(2, pos, -1, ())]
    named = game["pool"]
    if named["sha256"] != pool.sha256:
        raise ValueError(f"game from pool {named['sha256']}, not {pool.sha256}")
    shown = decision["shownIdentities"]
    ranked = decision["rankViews"]
    views: list[View] = []
    exact = [False, False]
    for side in (0, 1):
        other = 1 - side
        team = next(t for t in pool.teams if t.id == named["teams"][other])
        items = completions(reg, pos, other, list(team.sets), seen=seen_slots(pos, other, shown[other]))
        index, species = int(ranked[side][0]), tuple(ranked[side][1])
        item = items[index]
        if tuple(item.species) != species:
            raise ValueError(f"completion {index} of side {other} is {item.species}, recorded {species}")
        exact[side] = item.exact
        views.append(View(side, item.position, index if not item.exact else -1, species))
    if exact[0] and exact[1]:
        return [View(2, pos, -1, ())]
    return views


# --------------------------------------------------------------------------------------
# actions


def _row_of_party(side: Any, party_index: int) -> int:  # noqa: ANN401
    """The `Encoded` row (list position) of the Pokemon a choice's party index names."""
    for row, mon in enumerate(side.pokemon):
        if mon.slot == party_index - 1:
            return row
    raise ValueError(f"no Pokemon in party slot {party_index}")


def encode_actions(
    vocab: Vocabulary,
    pos: Position,
    side: int,
    actions: Sequence[SideAction],
) -> np.ndarray:
    """(N, 2, 7) int32: each action's two slots as rows of `pos`'s `Encoded` arrays.

    Per slot: kind (pass / move / switch), the acting Pokemon's row on `side` (-1 for
    none), the move's vocabulary id (0 unknown or none), mega (0/1), the target's kind
    (none / foe / ally), the target Pokemon's row on its side (-1 for an empty position or
    no target), and the switch-in's row on `side` (-1).
    """
    own = pos.sides[side]
    foe = pos.sides[1 - side]
    out = np.full((len(actions), SLOTS, len(ACTION_FIELDS)), -1, dtype=np.int32)
    for n, action in enumerate(actions):
        for t, slot_action in enumerate(action.slots[:SLOTS]):
            row = out[n, t]
            actor = own.active[slot_action.slot] if slot_action.slot < len(own.active) else None
            row[1] = -1 if actor is None else int(actor)
            row[2] = 0
            row[3] = 0
            row[4] = TARGET_NONE
            if isinstance(slot_action, MoveAction):
                row[0] = KIND_MOVE
                row[2] = vocab.moves.get(slot_action.move_id, 0)
                row[3] = 1 if slot_action.mega else 0
                target = slot_action.target
                if target is not None and target > 0:
                    row[4] = TARGET_FOE
                    at = foe.active[target - 1] if target - 1 < len(foe.active) else None
                    row[5] = -1 if at is None else int(at)
                elif target is not None and target < 0:
                    row[4] = TARGET_ALLY
                    at = own.active[-target - 1] if -target - 1 < len(own.active) else None
                    row[5] = -1 if at is None else int(at)
            elif isinstance(slot_action, SwitchAction):
                row[0] = KIND_SWITCH
                row[6] = _row_of_party(own, slot_action.party_index)
            else:
                row[0] = KIND_PASS
    return out


# --------------------------------------------------------------------------------------
# what a move is, and what a candidate does here


MOVE_TARGETS = (
    "normal",
    "self",
    "allAdjacentFoes",
    "all",
    "allAdjacent",
    "any",
    "allySide",
    "randomNormal",
    "adjacentAlly",
    "foeSide",
    "allies",
    "adjacentAllyOrSelf",
)
MOVE_FLAGS = (
    "contact",
    "sound",
    "punch",
    "bite",
    "pulse",
    "slicing",
    "heal",
    "protect",
    "bullet",
    "wind",
    "powder",
    "dance",
)


def move_table(reg: Any, vocab: Vocabulary) -> np.ndarray:  # noqa: ANN401, C901
    """(moves vocabulary, P) float32: each move's fixed properties from the dex.

    Type, category, power, accuracy, priority, the kind of target, the dex flags, and a
    summary of what else it does: a protecting move (`stallingMove`), switching its user
    out, recoil or drain, a secondary effect's chance, a status or volatile it inflicts,
    stat stages it moves on the user or the target, multi-hit, a raised critical ratio, a
    side condition or field effect it sets. Row 0 (unknown or none) is zeros.
    """
    types = list(vocab.types)
    width = len(types) + 3 + 4 + 1 + len(MOVE_TARGETS) + 1 + len(MOVE_FLAGS) + 14
    table = np.zeros((max(vocab.moves.values(), default=0) + 1, width), dtype=np.float32)
    for move_id, row_index in vocab.moves.items():
        move = reg.moves.get(move_id)
        if move is None:
            continue
        raw = move.raw
        row = table[row_index]
        at = 0
        if move.type in types:
            row[at + types.index(move.type)] = 1.0
        at += len(types)
        row[at + ("Physical", "Special", "Status").index(move.category)] = 1.0
        at += 3
        row[at] = move.base_power / 150.0
        row[at + 1] = 1.0 if move.accuracy is None else move.accuracy / 100.0
        row[at + 2] = 1.0 if move.accuracy is None else 0.0
        row[at + 3] = move.priority / 5.0
        at += 4
        row[at] = 1.0 if move.category != "Status" and move.base_power == 0 else 0.0  # variable
        at += 1
        if move.target in MOVE_TARGETS:
            row[at + MOVE_TARGETS.index(move.target)] = 1.0
        else:
            row[at + len(MOVE_TARGETS)] = 1.0
        at += len(MOVE_TARGETS) + 1
        flags = raw.get("flags") or {}
        for i, flag in enumerate(MOVE_FLAGS):
            row[at + i] = 1.0 if flag in flags else 0.0
        at += len(MOVE_FLAGS)
        secondaries = raw.get("secondaries") or ([raw["secondary"]] if raw.get("secondary") else [])
        boosts_self = raw.get("self", {}).get("boosts") if isinstance(raw.get("self"), dict) else None
        summary = [
            bool(raw.get("stallingMove")),
            bool(raw.get("selfSwitch")),
            bool(raw.get("recoil")) or bool(raw.get("mindBlownRecoil")) or bool(raw.get("hasCrashDamage")),
            bool(raw.get("drain")),
            max((s.get("chance", 100) for s in secondaries), default=0) / 100.0,
            bool(raw.get("status")) or any(s.get("status") for s in secondaries),
            bool(raw.get("volatileStatus")) or any(s.get("volatileStatus") for s in secondaries),
            sum((raw.get("boosts") or {}).values()) / 2.0 if raw.get("target") == "self" else 0.0,
            sum((raw.get("boosts") or {}).values()) / 2.0 if raw.get("target") != "self" else 0.0,
            sum((boosts_self or {}).values()) / 2.0,
            bool(raw.get("multihit")),
            max(0, int(move.crit_ratio) - 1) / 2.0,
            bool(raw.get("sideCondition")) or bool(raw.get("slotCondition")),
            bool(raw.get("weather")) or bool(raw.get("terrain")) or bool(raw.get("pseudoWeather")),
        ]
        row[at : at + len(summary)] = np.asarray(summary, dtype=np.float32)
    return table


#: Numbers per candidate from the port's `qfeatures` command (rust/src/qfeatures.rs).
FEATURE_WIDTH = 34


def port_features(
    reg: Any,  # noqa: ANN401
    pos: Position,
    pools: tuple[Sequence[SideAction], Sequence[SideAction]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Each candidate's damage, knock-out, speed and switch numbers, from the port.

    One crossing for both sides' pools. None when the port refuses the position (the
    same guard as the ranking's damage score).
    """
    from . import rustnode

    node = rustnode.node_for(reg)
    if node is None:
        raise rustnode.PortUnavailable("no Rust node for the candidate features")
    response = node._exchange(  # noqa: SLF001 - one request, the same pipe `score` takes
        {
            "kind": "qfeatures",
            "position": rustnode._position(pos),  # noqa: SLF001
            "candidates": [
                [[rustnode.dump_action(a) for a in c.slots] for c in pools[side]] for side in (0, 1)
            ],
        }
    )
    if response.get("refused"):
        return None
    if int(response["width"]) != FEATURE_WIDTH:
        raise ValueError(f"the port answers {response['width']} features, this reads {FEATURE_WIDTH}")
    out = []
    for side in (0, 1):
        rows = np.asarray(response["features"][side], dtype=np.float32)
        out.append(rows.reshape(len(pools[side]), FEATURE_WIDTH))
    return out[0], out[1]


# --------------------------------------------------------------------------------------
# the net


@dataclass(slots=True)
class QConfig:
    act_dim: int = 128
    pair_dim: int = 128
    rank: int = 32
    dropout: float = 0.1
    #: False: moves by their vocabulary id only. True: the id plus the dex's fixed
    #: properties (`move_table`) and the port's per-candidate numbers (`port_features`).
    properties: bool = False
    #: One round of attention over the eight Pokemon rows after the trunk.
    attend: bool = True


def _torch() -> Any:  # noqa: ANN401
    import torch

    return torch


def build_net(  # noqa: C901, PLR0915
    encoder: Any,  # noqa: ANN401
    config: QConfig,
    table: np.ndarray | None = None,
) -> Any:  # noqa: ANN401
    """A `QNet` over `encoder`'s arrays.

    **The trunk is the leaf's.** The position is read by a `value.ValueNet` built for the
    same encoder (its vocabulary fingerprint binds both): the same embeddings, the same
    per-Pokemon MLP and the same side pooling, so its weights can be the shipped leaf's
    (`load_trunk`) and a later model can share one trunk between V and Q. Its value head
    is carried and not used. A candidate is tied to the trunk's Pokemon rows it names.

    **Antisymmetric by construction.** With u the row actions' vectors and v the column
    actions', the logit is g(u_i, v_j) - g(v_j, u_i). The trunk's rows and side vectors of
    the mirrored position are this position's with the side axis flipped (every layer is
    per Pokemon, per side, or a permutation-equivariant attention over the eight), so the
    mirrored position's row actions are exactly this one's v and its columns this one's u:
    Q(mirror s, b, a) = 1 - Q(s, a, b) holds at initialisation and after any training, and
    one teaching matrix teaches both sides.

    With `config.properties` it needs `table` (`move_table`), kept as a buffer so a saved
    net carries the dex it was trained with, and its forward takes the port's features.
    """
    torch = _torch()
    nn = torch.nn
    from .value import ValueConfig, build

    if config.properties and table is None:
        raise ValueError("a net with properties needs the move table")
    props = int(table.shape[1]) if config.properties and table is not None else 0
    extra = FEATURE_WIDTH if config.properties else 0
    widths = encoder.widths

    class QNet(nn.Module):
        """The leaf's trunk -> Pokemon rows; action -> the rows it names; pair -> logit."""

        def __init__(self) -> None:
            super().__init__()
            c = config
            self.config = c
            self.trunk = build(encoder, ValueConfig())
            tc = self.trunk.config
            mon_dim, side_dim, move_dim = tc.mon_dim, tc.side_dim, tc.move_dim
            if props:
                self.register_buffer("move_props", torch.from_numpy(np.asarray(table, dtype=np.float32)))
                # Added to the trunk's rows, zero at the start, so a trunk from the leaf
                # reads the position exactly as the leaf does until training moves it.
                self.mon_props = nn.Linear(props, mon_dim)
                nn.init.zeros_(self.mon_props.weight)
                nn.init.zeros_(self.mon_props.bias)
            if c.attend:
                self.attend = nn.MultiheadAttention(mon_dim, 4, batch_first=True)
                self.attend_norm = nn.LayerNorm(mon_dim)
            self.context = nn.Sequential(
                nn.Linear(2 * side_dim + widths["field"], c.act_dim),
                nn.LayerNorm(c.act_dim),
                nn.GELU(),
            )
            self.kind = nn.Embedding(3, 16)
            self.target = nn.Embedding(3, 8)
            self.slot_pos = nn.Embedding(SLOTS, c.act_dim)
            slot_in = 16 + mon_dim + move_dim + props + 1 + 8 + mon_dim + mon_dim
            self.slot_mlp = nn.Sequential(
                nn.Linear(slot_in, c.act_dim),
                nn.LayerNorm(c.act_dim),
                nn.GELU(),
                nn.Linear(c.act_dim, c.act_dim),
            )
            self.act_mlp = nn.Sequential(
                nn.Linear(2 * c.act_dim + extra, c.act_dim),
                nn.LayerNorm(c.act_dim),
                nn.GELU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.act_dim, c.act_dim),
                nn.LayerNorm(c.act_dim),
                nn.GELU(),
            )
            self.first_pair = nn.Linear(c.act_dim, c.pair_dim)
            self.second_pair = nn.Linear(c.act_dim, c.pair_dim, bias=False)
            self.pair_out = nn.Linear(c.pair_dim, 1)
            self.first_low = nn.Linear(c.act_dim, c.rank)
            self.second_low = nn.Linear(c.act_dim, c.rank)

        def mons(self, batch: dict[str, Any]) -> tuple[Any, Any]:  # noqa: ANN401
            """(B, 2, M, mon_dim) rows and (B, 2, side_dim) side vectors, by the leaf's trunk."""
            t = self.trunk
            moves = t.move(batch["moves"])
            move_mask = (batch["moves"] > 0).float().unsqueeze(-1)
            move_pooled = (moves * move_mask).sum(dim=3) / move_mask.sum(dim=3).clamp(min=1.0)
            features = torch.cat(
                [
                    t.species(batch["species"]),
                    t.ability(batch["ability"]),
                    t.item(batch["item"]),
                    move_pooled,
                    batch["mon"],
                ],
                dim=-1,
            )
            mon = t.mon_mlp(features)
            if props:
                pooled = (self.move_props[batch["moves"]] * move_mask).sum(3) / move_mask.sum(3).clamp(
                    min=1.0
                )
                mon = mon + self.mon_props(pooled)
            present = batch["mask"].unsqueeze(-1)
            is_active = batch["mon"][..., t._active_feature].unsqueeze(-1)  # noqa: SLF001
            active = present * is_active
            bench = present * (1.0 - is_active)
            from .value import _masked_max, _masked_mean

            pooled_sides = torch.cat(
                [
                    _masked_mean(mon, active),
                    _masked_max(mon, active),
                    _masked_mean(mon, bench),
                    _masked_max(mon, bench),
                    batch["side"],
                ],
                dim=-1,
            )
            sides = t.side_mlp(pooled_sides)
            if self.config.attend:
                b, s, m, d = mon.shape
                flat = mon.reshape(b, s * m, d)
                here = batch["mask"].reshape(b, s * m) > 0
                attended, _w = self.attend(flat, flat, flat, key_padding_mask=~here)
                mon = self.attend_norm(flat + torch.nan_to_num(attended)).reshape(b, s, m, d)
            return mon, sides

        def encode_actions(self, h: Any, ctx: Any, side: int, acts: Any, feats: Any = None) -> Any:  # noqa: ANN401
            """(B, N, act_dim) for `side`'s actions (B, N, 2, 7) and their features (B, N, F)."""
            b, n = acts.shape[:2]
            zero = torch.zeros(b, 1, h.shape[-1], device=h.device, dtype=h.dtype)

            def rows(of_side: Any, index: Any) -> Any:  # noqa: ANN401
                table = torch.cat([of_side, zero], dim=1)  # row M is "nobody"
                safe = torch.where(index < 0, torch.full_like(index, of_side.shape[1]), index)
                flat = safe.reshape(b, -1)
                got = torch.gather(table, 1, flat.unsqueeze(-1).expand(-1, -1, table.shape[-1]))
                return got.reshape(*index.shape, table.shape[-1])

            own = h[:, side]
            foe = h[:, 1 - side]
            kind = acts[..., 0].clamp(min=0)
            actor = rows(own, acts[..., 1])
            move_ids = acts[..., 2].clamp(min=0)
            move = self.trunk.move(move_ids)
            if props:
                move = torch.cat([move, self.move_props[move_ids]], dim=-1)
            mega = acts[..., 3].clamp(min=0).unsqueeze(-1).to(h.dtype)
            tkind = acts[..., 4].clamp(min=0)
            trow = acts[..., 5]
            at_foe = rows(foe, torch.where(tkind == TARGET_FOE, trow, torch.full_like(trow, -1)))
            at_ally = rows(own, torch.where(tkind == TARGET_ALLY, trow, torch.full_like(trow, -1)))
            switch = rows(own, acts[..., 6])
            x = torch.cat(
                [self.kind(kind), actor, move, mega, self.target(tkind), at_foe + at_ally, switch],
                dim=-1,
            )
            slots = self.slot_mlp(x) + self.slot_pos.weight  # (B,N,2,act)
            pair = slots.sum(2)
            parts = [pair, ctx.unsqueeze(1).expand(-1, n, -1)]
            if extra:
                parts.append(feats)
            return self.act_mlp(torch.cat(parts, dim=-1))

        def pair(self, first: Any, second: Any) -> Any:  # noqa: ANN401
            """g(first_i, second_j): (B, N, K) for first (B, N, d) against second (B, K, d)."""
            hidden = torch.nn.functional.gelu(
                self.first_pair(first).unsqueeze(2) + self.second_pair(second).unsqueeze(1)
            )
            additive = self.pair_out(hidden).squeeze(-1)
            low = torch.einsum("bir,bjr->bij", self.first_low(first), self.second_low(second))
            return additive + low / (self.config.rank**0.5)

        def forward(
            self, batch: dict[str, Any], acts0: Any, acts1: Any, feats0: Any = None, feats1: Any = None
        ) -> Any:  # noqa: ANN401
            """(B, N0, N1) logits of side 0 winning, antisymmetric under the mirror."""
            h, sides = self.mons(batch)
            field = batch["field"]
            ctx0 = self.context(torch.cat([sides[:, 0], sides[:, 1], field], dim=-1))
            ctx1 = self.context(torch.cat([sides[:, 1], sides[:, 0], field], dim=-1))
            u = self.encode_actions(h, ctx0, 0, acts0, feats0)
            v = self.encode_actions(h, ctx1, 1, acts1, feats1)
            return self.pair(u, v) - self.pair(v, u).transpose(1, 2)

    return QNet()


def load_trunk(net: Any, path: str | Path, encoder: Any) -> None:  # noqa: ANN401
    """Copy a trained leaf's weights into `net`'s trunk (read only; the leaf is untouched)."""
    from .value import load_model

    leaf, _meta = load_model(path, encoder)
    net.trunk.load_state_dict(leaf.state_dict())


def config_dict(config: QConfig) -> dict[str, Any]:
    return asdict(config)


def load_q(path: str | Path, device: str = "cpu") -> Any:  # noqa: ANN401
    """A trained Q (`tools/q_train.py`'s file) in eval mode on `device`.

    Stops when the file was trained on another vocabulary than its regulation's today:
    the action encoding indexes the same move table the trunk embeds.
    """
    import torch

    from .encode import Encoder
    from .regulation import load_regulation

    blob = torch.load(path, map_location=device, weights_only=False)
    encoder = Encoder(load_regulation(blob["regulation"]))
    if blob.get("vocab_fingerprint") != encoder.vocab.fingerprint():
        raise ValueError(f"{path} was trained on another vocabulary than this regulation's")
    net = build_net(encoder, QConfig(**blob["config"]), blob.get("move_table"))
    net.load_state_dict(blob["state"])
    net.vocab_fingerprint = blob["vocab_fingerprint"]
    return net.to(device).eval()


#: The position arrays a Q reads, in `Encoded`'s names.
POSITION_ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def q_matrix(net: Any, arrays: dict[str, np.ndarray], device: Any) -> np.ndarray:  # noqa: ANN401
    """(N0, N1) float64: side 0's win probability for every pair, from one request's arrays.

    ``arrays`` holds one position's `POSITION_ARRAYS` (one row each), ``acts0`` / ``acts1``
    (`encode_actions`) and, for a net with properties, ``feats0`` / ``feats1``
    (`port_features`). The one expression `tools/q_menus.py` measured with, so a served
    Q and a local one answer alike.
    """
    torch = _torch()
    batch = {
        name: torch.from_numpy(np.ascontiguousarray(arrays[name])).to(device)
        for name in POSITION_ARRAYS
    }
    acts = [
        torch.from_numpy(np.ascontiguousarray(arrays[f"acts{s}"]).astype(np.int64))[None].to(device)
        for s in (0, 1)
    ]
    feats: list[Any] = [None, None]
    if net.config.properties:
        feats = [
            torch.from_numpy(np.ascontiguousarray(arrays[f"feats{s}"], dtype=np.float32))[None].to(device)
            for s in (0, 1)
        ]
    with torch.no_grad():
        logits = net(batch, acts[0], acts[1], feats[0], feats[1])
    return torch.sigmoid(logits[0]).double().cpu().numpy()


def encoded_row(encoded: Encoded, index: int) -> dict[str, np.ndarray]:
    """One position's arrays, as the teaching shards store them."""
    return {
        name: getattr(encoded, name)[index]
        for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    }


__all__ = [
    "ACTION_FIELDS",
    "KIND_MOVE",
    "KIND_PASS",
    "KIND_SWITCH",
    "TARGET_ALLY",
    "TARGET_FOE",
    "TARGET_NONE",
    "QConfig",
    "View",
    "build_net",
    "config_dict",
    "load_trunk",
    "move_table",
    "port_features",
    "decision_views",
    "encode_actions",
    "encoded_row",
    "legal_pool",
    "load_q",
    "q_matrix",
]
