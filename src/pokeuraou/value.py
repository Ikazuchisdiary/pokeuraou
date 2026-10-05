"""The learned value function: a win probability, and the parts that make it honest.

The target is the real outcome of a game played to a win or a loss, so the number this
produces is a win probability and may be printed as one. That is the whole reason the
self-play generator discards unfinished games instead of labelling them.

Three design decisions are load-bearing, and each is here to stop a specific way of being
quietly wrong.

**Antisymmetry is architectural, not learned.** The game is zero-sum, so
``V(position) + V(mirrored position) = 1`` must hold. The head is evaluated on both side
orderings and the logit is the difference, which makes the identity exact at every point
in training. Learning it from data instead would spend capacity on a known fact and would
still be violated in exactly the positions with little data.

**The split is by game, never by decision.** Every decision in one game carries the same
label, and 13 decisions from the same game are 13 copies of one outcome. Splitting
decisions at random puts near-duplicates on both sides of the split, and the validation
loss then measures memorisation. This is the difference between a believable AUC and a
flattering one.

**Evaluation is per turn.** Measuring the `hp-share` proxy first showed why: it scores
AUC 0.910 over all decisions but 0.769 at turn 1 and 0.961 by turn 13. Material *is* the
answer late, so an aggregate number is dominated by positions nobody needs help with. The
job is the early game, and a single average hides whether it was done.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from . import timing
from .encode import (
    BIND_MON_FEATURES,
    MOVE_ID_FEATURES,
    Encoded,
    Encoder,
    Vocabulary,
    settle,
)
from .slotswap import SwapSlots, swap_batch


@dataclass(slots=True)
class ValueConfig:
    """Sizes and optimiser settings, saved next to the weights."""

    species_dim: int = 48
    ability_dim: int = 24
    item_dim: int = 24
    move_dim: int = 32
    mon_dim: int = 160
    side_dim: int = 192
    head_dim: int = 256
    #: 0.4, from `tools/sweep_value.py` at three seeds: 0.3991 +-0.008 validation log loss
    #: against 0.4102 +-0.002 at the original 0.1, and monotone in between (0.0 -> 0.4172,
    #: 0.2 -> 0.4074, 0.3 -> 0.4071, 0.5 -> 0.4045). Raising the learning rate to 4e-3 wins
    #: by a similar margin on its own and adds nothing on top of this, so the two were the
    #: same effect measured twice: at 13,395 games the model fits too easily.
    dropout: float = 0.4
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-2
    batch_size: int = 1024
    epochs: int = 30
    #: Stop when validation loss has not improved for this many epochs.
    patience: int = 4
    seed: int = 0
    #: Which games are held out, separately from `seed`.
    #:
    #: `seed` moves three things at once -- initialisation, batch order and the held-out
    #: games -- so two runs that differ by it differ in what they learned AND in what they
    #: were marked against, and the second is not noise a reader can average away: it
    #: changes the meaning of the number, not just its value. None keeps the old behaviour
    #: of taking the split from `seed`, so every model trained before this was added is
    #: still described by what its record says -- including one read back out of a stored
    #: config, which predates the field and so arrives at this default.
    #:
    #: `tools/train_value.py` defaults its flag to 0 rather than to None, because a run
    #: that passes `--seed` and nothing else should still be marked against the games
    #: every other run was. The two defaults agree on every run in the record: no
    #: invocation of that tool has ever passed `--seed`, so `seed` was 0 there too.
    #:
    #: Not a way to decompose run-to-run variance -- that was ruled out, because the
    #: comparisons that matter here change the pool, so no shared split exists across
    #: them, and `tools/sweep_value.py` already computes one split and hands it to every
    #: configuration when the pool IS shared. This exists so a record can say which split
    #: it used.
    split_seed: int | None = None
    #: Which weights `train` hands back (IKA-86). ``"best"`` stops after `patience`
    #: epochs without a better validation loss and returns the best epoch's weights --
    #: the recipe of every model up to value-gen11L, which stopped at epoch 8-11 of a
    #: 30-epoch OneCycle and so never left the top of the learning rate. ``"last"`` runs
    #: all `epochs` and returns the final (or averaged) weights, so the schedule's decay
    #: is the part that is kept.
    keep: str = "best"
    #: ``"none"``, ``"ema"`` (a moving average of the weights after every step, decay
    #: `ema_decay`) or ``"swa"`` (the plain mean of the end-of-epoch weights from epoch
    #: ``ceil(swa_from * epochs)`` on). Only with ``keep="last"``.
    average: str = "none"
    ema_decay: float = 0.999
    swa_from: float = 0.5
    #: OneCycle's warm-up share. 0.2 is what every model so far used.
    pct_start: float = 0.2
    #: IKA-318: read each move's fixed properties from the dex (`qhead.move_table`: type,
    #: category, power, accuracy, priority, target, flags, what else it does) beside its
    #: id embedding. Off: the net, its parameters and its answers are exactly those of
    #: every model before this field existed (a stored config without it reads back off).
    move_properties: bool = False
    #: Width of one move's property vector after its own small layer, when on.
    move_property_dim: int = 32
    #: IKA-405: how many equal groups each hidden layer's LayerNorm normalises separately.
    #: 1 is the plain LayerNorm over the whole layer -- every model before this field. A
    #: net grown by `widen_net` to k times the width has k groups of the original width, so
    #: group 0 is the original layer and its normalisation does not see the added units.
    width_groups: int = 1
    #: IKA-90: one residual attention layer over the (up to) 2 x picked_team_size Pokemon
    #: tokens, after `mon_mlp` and before the pooling. Its output projection starts at zero,
    #: so a net warm-started from a model without it answers exactly as that model until
    #: training moves it. Off: parameters, keys and answers are those of every earlier model.
    attention: bool = False
    attention_heads: int = 4
    #: IKA-425: read the encoder's state columns (encoding revision 3: turns left of Trick
    #: Room, Tailwind and the screens, the sleep and toxic counters, Perish Song's count,
    #: the locked and the last move). Off: the net reads the revision-2 columns only
    #: (`Encoder.base_widths`), so every model before this field -- a stored config without
    #: it reads back off -- answers bit for bit as it did on a revision-3 encoding. On: the
    #: numeric state columns join each block's first layer, and the two move ids are looked
    #: up in the move embedding and join `mon_mlp`'s first layer. A warm start from a model
    #: without it gives all of those input columns zero weights (`grow_state_inputs`).
    state_inputs: bool = False
    #: IKA-425 stage 2a: dropout on the per-Pokemon vectors (after `mon_mlp`, before the
    #: pooling) and on each side's vector (after `side_mlp`), in training only. 0.0 adds no
    #: module and draws nothing from the random stream, so the net, its answers and a
    #: training run are those of every earlier model.
    encoder_dropout: float = 0.0
    #: IKA-429: read the bind columns too (encoding revision 4, `BIND_MON_FEATURES` and
    #: `BIND_SIDE_FEATURES`). Needs `state_inputs`. Off: the net reads what it read before
    #: (a stored config without it reads back off). On: the Pokemon's bind columns join
    #: `mon_mlp`'s first layer after the move ids' embeddings, the side's join `side_mlp`'s.
    #: A warm start from a net without it gives them zero weights (`grow_bind_inputs`).
    bind_inputs: bool = False
    #: IKA-425 stage 2a: weight of the auxiliary regression. Above 0, the net has a small
    #: extra head (`aux`) that reads the same hidden layer as the win logit and regresses the
    #: end-of-game material of the game the row belongs to (`final_material_targets`: the
    #: difference in alive fraction and in team HP fraction, side 0 minus side 1), also
    #: antisymmetric. `train` adds `aux_weight` x MSE to the cross entropy. The head is
    #: never read by `forward`; a model saved for use drops it. 0.0 builds nothing.
    aux_weight: float = 0.0
    #: IKA-428: more auxiliary targets, ``"name:weight,name:weight"`` from `AUX_TARGETS`
    #: (each built from the dataset's rows by `aux_target_arrays`). Each has its own small
    #: head under `aux_heads`, read by training only; a model saved for use drops them and
    #: this field. "" builds nothing, so the net, its answers and a training run are those
    #: of every earlier model.
    aux_targets: str = ""


class _GroupLayerNorm(nn.Module):
    """LayerNorm applied to each of `groups` equal slices of the last dimension.

    The parameters are `weight` and `bias` of the whole width, named and shaped as in
    `nn.LayerNorm`, so a state dict reads across. With one group it is `nn.LayerNorm`
    (`_norm` returns that class then, not this one).
    """

    def __init__(self, dim: int, groups: int, eps: float = 1e-5) -> None:
        super().__init__()
        if dim % groups:
            raise ValueError(f"{dim} is not divisible into {groups} groups")
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        shaped = x.unflatten(-1, (self.groups, -1))
        normed = nn.functional.layer_norm(shaped, shaped.shape[-1:], eps=self.eps)
        return normed.flatten(-2) * self.weight + self.bias


def _norm(dim: int, groups: int) -> nn.Module:
    return nn.LayerNorm(dim) if groups == 1 else _GroupLayerNorm(dim, groups)


class _MonAttention(nn.Module):
    """Pre-norm residual self-attention across all Pokemon of both sides (IKA-90).

    A token attends to every present token, its own side's and the other side's. Which
    side a key is on is read only as "same as the query's or not" (a learned per-head term
    added to the score by `rel[query, key]`), never as side 0 or 1, so swapping the two
    sides permutes the tokens and nothing else: the layer commutes with the side flip, and
    the head's `head(o, t) - head(t, o)` keeps the value antisymmetric. `out` starts at
    zero, so the layer is the identity until it is trained.
    """

    def __init__(self, dim: int, heads: int, tokens_per_side: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"attention width {dim} does not split into {heads} heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        # Per head: what a query looks for in a same-side key and in an opposing key.
        self.rel = nn.Parameter(torch.randn(2, heads, self.head_dim) * 0.02)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        side = torch.arange(2 * tokens_per_side) // tokens_per_side
        self.register_buffer("rel_index", (side[:, None] != side[None, :]).long(), persistent=False)

    def forward(self, mon: Tensor, present: Tensor) -> Tensor:
        """`mon` (B,2,M,D), `present` (B,2,M,1) -> (B,2,M,D)."""
        b, sides, m, d = mon.shape
        n = sides * m
        x = mon.reshape(b, n, d)
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)

        def split(t: Tensor) -> Tensor:
            return t.view(b, n, self.heads, self.head_dim).transpose(1, 2)  # (B,H,N,hd)

        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-1, -2)  # (B,H,N,N)
        # q . rel[same/opposite] for each query, then picked per (query, key) pair.
        per_rel = torch.einsum("bhnd,rhd->bhnr", q, self.rel)  # (B,H,N,2)
        index = self.rel_index.unsqueeze(0).unsqueeze(0).expand(b, self.heads, n, n)
        scores = scores + per_rel.gather(-1, index)
        scores = scores / self.head_dim**0.5
        keys = present.reshape(b, 1, 1, n) > 0
        # A large negative, not -inf: an absent query has no key of its own to fall back on.
        attn = torch.softmax(scores.masked_fill(~keys, -1e9), dim=-1)
        mixed = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return mon + self.out(mixed).view(b, sides, m, d)


class ValueNet(nn.Module):
    """Per-Pokemon encoder, order-invariant pooling, antisymmetric head.

    Pooling is masked mean *and* max, taken separately over the active Pokemon and the
    bench. Mean alone cannot express "one of my Pokemon can survive this", which is a
    property of the best member rather than the average one; max alone loses how much
    material is left. Active and bench are pooled apart because being on the field is not
    a matter of degree.
    """

    def __init__(self, encoder: Encoder, config: ValueConfig) -> None:
        super().__init__()
        self.config = config
        sizes = encoder.vocab.sizes
        # IKA-425: the columns of each block this net reads -- the revision-2 prefix, the
        # revision-3 prefix with `state_inputs`, all of them with `bind_inputs` (IKA-429).
        widths = net_widths(encoder, config)
        self.in_widths = widths
        self._move_ids = len(MOVE_ID_FEATURES) if config.state_inputs else 0
        # IKA-429: where the move ids sit in a full-width row (before the bind columns),
        # and how many bind columns follow them. 0 for a net without `bind_inputs`.
        self._bind = len(BIND_MON_FEATURES) if config.bind_inputs else 0
        self._ids_end = encoder.state_widths["mon"]
        groups = config.width_groups

        self.species = nn.Embedding(sizes["species"], config.species_dim, padding_idx=0)
        self.ability = nn.Embedding(sizes["ability"], config.ability_dim, padding_idx=0)
        self.item = nn.Embedding(sizes["item"], config.item_dim, padding_idx=0)
        self.move = nn.Embedding(sizes["move"], config.move_dim, padding_idx=0)

        mon_in = (
            config.species_dim
            + config.ability_dim
            + config.item_dim
            + config.move_dim
            + widths["mon"]
            - self._move_ids
            + self._move_ids * config.move_dim
        )
        self.mon_mlp = nn.Sequential(
            nn.Linear(mon_in, config.mon_dim),
            _norm(config.mon_dim, groups),
            nn.GELU(),
            nn.Linear(config.mon_dim, config.mon_dim),
            _norm(config.mon_dim, groups),
            nn.GELU(),
        )
        if config.move_properties:
            # IKA-318. Each move's dex properties go through a small layer of their own and
            # are pooled over the four moves like the ids are; the pool joins the first
            # layer of `mon_mlp` as extra input columns -- written as a second linear added
            # to that layer's output, which is the same map as concatenating the inputs.
            # Those columns start at zero, so a net warm-started from an id-only model reads
            # every position exactly as that model does until training moves them.
            table = move_property_table(encoder)
            self.register_buffer("move_props", torch.from_numpy(table))
            self.move_prop_embed = nn.Sequential(
                nn.Linear(table.shape[1], config.move_property_dim), nn.GELU()
            )
            self.move_prop_in = nn.Linear(config.move_property_dim, config.mon_dim)
            nn.init.zeros_(self.move_prop_in.weight)
            nn.init.zeros_(self.move_prop_in.bias)
        if config.attention:
            self.mon_attention = _MonAttention(
                config.mon_dim, config.attention_heads, encoder.mons_per_side
            )
        # active mean, active max, bench mean, bench max
        pooled = 4 * config.mon_dim + widths["side"]
        self.side_mlp = nn.Sequential(
            nn.Linear(pooled, config.side_dim),
            _norm(config.side_dim, groups),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * config.side_dim + widths["field"], config.head_dim),
            _norm(config.head_dim, groups),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_dim, config.head_dim),
            _norm(config.head_dim, groups),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_dim, 1),
        )
        if config.aux_weight > 0:
            self.aux = nn.Linear(config.head_dim, AUX_DIM)
        if config.aux_targets:
            # IKA-428: one head per named target (`AUX_TARGETS`), built after every other
            # module so that the modules above draw the same initialisation as without.
            heads = {}
            for name, _weight in parse_aux_targets(config.aux_targets):
                kind, dim, _loss = AUX_TARGETS[name]
                if kind == "mon":
                    heads[name] = nn.Linear(config.mon_dim + config.head_dim, 1)
                else:
                    heads[name] = nn.Linear(config.head_dim, dim)
            self.aux_heads = nn.ModuleDict(heads)

    def columns(self, batch: dict[str, Tensor], name: str) -> Tensor:
        """`batch[name]` ("mon", "side" or "field") cut to the columns this net reads.

        The whole array when it is already that wide (a revision-2 encoding read by a
        revision-2 net, or any encoding read by a `state_inputs` net), so nothing is
        copied on that road; the leading columns otherwise (IKA-425).
        """
        x = batch[name]
        width = self.in_widths[name]
        return x if x.shape[-1] == width else x[..., :width]

    def mon_inputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """(the first layer's input (B,2,M,mon_in), the move mask (B,2,M,4,1)).

        Shared with `qhead.QNet.mons`, which runs the same trunk.
        """
        moves = self.move(batch["moves"])  # (B,2,M,4,move_dim)
        move_mask = (batch["moves"] > 0).float().unsqueeze(-1)
        move_pooled = (moves * move_mask).sum(dim=3) / move_mask.sum(dim=3).clamp(min=1.0)
        mon = self.columns(batch, "mon")
        parts = [
            self.species(batch["species"]),
            self.ability(batch["ability"]),
            self.item(batch["item"]),
            move_pooled,
        ]
        if self._bind:
            # IKA-429: the bind columns after the move ids; they join the first layer last,
            # so a net grown from a `state_inputs` one keeps its inputs' places.
            k, end = self._move_ids, self._ids_end
            parts.append(mon[..., : end - k])
            parts.append(self.move(mon[..., end - k : end].long()).flatten(-2))
            parts.append(mon[..., end:])
        elif self._move_ids:
            # IKA-425: the locked and the last move, ids carried in the last columns,
            # read through the move embedding (index 0, none, is the zero row).
            k = self._move_ids
            parts.append(mon[..., :-k])
            parts.append(self.move(mon[..., -k:].long()).flatten(-2))
        else:
            parts.append(mon)
        return torch.cat(parts, dim=-1), move_mask

    def side_vectors(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, 2, side_dim), computed without reference to which side is which."""
        return self._sides_and_mons(batch)[0]

    def _sides_and_mons(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """`side_vectors` and the per-Pokemon vectors it pooled, (B, 2, M, mon_dim)."""
        features, move_mask = self.mon_inputs(batch)
        if self.config.move_properties:
            props = self.move_prop_embed(self.move_props[batch["moves"]])
            props_pooled = (props * move_mask).sum(dim=3) / move_mask.sum(dim=3).clamp(min=1.0)
            first = self.mon_mlp[0](features) + self.move_prop_in(props_pooled)
            mon = self.mon_mlp[1:](first)
        else:
            mon = self.mon_mlp(features)  # (B,2,M,mon_dim)
        if self.config.encoder_dropout > 0:
            mon = nn.functional.dropout(mon, self.config.encoder_dropout, self.training)

        present = batch["mask"].unsqueeze(-1)
        if self.config.attention:
            mon = self.mon_attention(mon, present)
        is_active = batch["mon"][..., self._active_feature].unsqueeze(-1)
        active = present * is_active
        bench = present * (1.0 - is_active)
        pooled = torch.cat(
            [
                _masked_mean(mon, active),
                _masked_max(mon, active),
                _masked_mean(mon, bench),
                _masked_max(mon, bench),
                self.columns(batch, "side"),
            ],
            dim=-1,
        )
        sides = self.side_mlp(pooled)
        if self.config.encoder_dropout > 0:
            sides = nn.functional.dropout(sides, self.config.encoder_dropout, self.training)
        return sides, mon

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Logit of side 0 winning, antisymmetric by construction."""
        sides = self.side_vectors(batch)
        ours, theirs = sides[:, 0], sides[:, 1]
        field = self.columns(batch, "field")
        forward = self.head(torch.cat([ours, theirs, field], dim=-1))
        mirrored = self.head(torch.cat([theirs, ours, field], dim=-1))
        return (forward - mirrored).squeeze(-1)

    def forward_aux(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """The win logit of :meth:`forward` and the auxiliary regression, (B,) and (B, AUX_DIM).

        The same two passes through the head, taken up to its last layer so the hidden layer
        feeds both outputs; the regression is antisymmetric the same way (side 0 minus side 1
        of what it reads), so a target that is a difference between the sides changes sign
        when the sides are exchanged. Training only (needs `aux_weight` > 0); `forward` does
        not read the head.
        """
        sides = self.side_vectors(batch)
        ours, theirs = sides[:, 0], sides[:, 1]
        field = self.columns(batch, "field")
        last = self.head[-1]
        hidden = self.head[:-1]
        h_forward = hidden(torch.cat([ours, theirs, field], dim=-1))
        h_mirrored = hidden(torch.cat([theirs, ours, field], dim=-1))
        logit = (last(h_forward) - last(h_mirrored)).squeeze(-1)
        return logit, self.aux(h_forward) - self.aux(h_mirrored)

    def forward_heads(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        """The win logit, the `aux` regression (None without it) and each `aux_heads` output.

        The same passes as :meth:`forward_aux`. Each named head reads the head's hidden layer
        the way its kind says (`AUX_TARGETS`), so every output keeps the symmetry of its
        target when the sides are exchanged:

        * ``anti`` (B, dim): ``f(h_forward) - f(h_mirrored)``, side 0 minus side 1.
        * ``sym`` (B, dim): ``f(h_forward) + f(h_mirrored)``, the same for both sides.
        * ``side`` (B, 2, dim): ``f(h_forward)`` for side 0, ``f(h_mirrored)`` for side 1.
        * ``mon`` (B, 2, M): one logit per Pokemon from its own vector beside its side's
          hidden layer.

        Training only (IKA-428).
        """
        sides, mon = self._sides_and_mons(batch)
        ours, theirs = sides[:, 0], sides[:, 1]
        field = self.columns(batch, "field")
        last = self.head[-1]
        hidden = self.head[:-1]
        h_forward = hidden(torch.cat([ours, theirs, field], dim=-1))
        h_mirrored = hidden(torch.cat([theirs, ours, field], dim=-1))
        logit = (last(h_forward) - last(h_mirrored)).squeeze(-1)
        aux = self.aux(h_forward) - self.aux(h_mirrored) if hasattr(self, "aux") else None
        outputs: dict[str, Tensor] = {}
        for name, head in self.aux_heads.items():
            kind = AUX_TARGETS[name][0]
            if kind == "anti":
                outputs[name] = head(h_forward) - head(h_mirrored)
            elif kind == "sym":
                outputs[name] = head(h_forward) + head(h_mirrored)
            elif kind == "side":
                outputs[name] = torch.stack([head(h_forward), head(h_mirrored)], dim=1)
            else:  # mon
                per_side = torch.stack([h_forward, h_mirrored], dim=1)  # (B,2,head_dim)
                context = per_side.unsqueeze(2).expand(-1, -1, mon.shape[2], -1)
                outputs[name] = head(torch.cat([mon, context], dim=-1)).squeeze(-1)
        return logit, aux, outputs

    #: Index of `is_active` inside the per-Pokemon numeric block. Set by :func:`build`.
    _active_feature: int = 0


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    filled = values.masked_fill(mask <= 0, float("-inf"))
    out = filled.max(dim=2).values
    # A side with nothing in the group (no bench left) would otherwise be -inf.
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def move_property_table(encoder: Encoder) -> np.ndarray:
    """(move vocabulary rows, P) float32: `qhead.move_table` for the encoder's dex.

    One row per row of the move embedding (index 0, unknown or none, is zeros), so a move
    id indexes both. The table is IKA-274's, imported rather than copied, so the leaf and
    the candidate model read a move the same way.
    """
    from .qhead import move_table

    table = move_table(encoder.reg, encoder.vocab)
    rows = encoder.vocab.sizes["move"]
    if table.shape[0] < rows:
        table = np.concatenate(
            [table, np.zeros((rows - table.shape[0], table.shape[1]), np.float32)]
        )
    return np.ascontiguousarray(table[:rows], dtype=np.float32)


def net_widths(encoder: Encoder, config: ValueConfig) -> dict[str, int]:
    """The columns of each block a net with `config` reads: revision 2's prefix, revision
    3's with `state_inputs` (IKA-425), every column with `bind_inputs` (IKA-429)."""
    if config.bind_inputs:
        if not config.state_inputs:
            raise ValueError("bind_inputs needs state_inputs: the bind columns follow the state ones")
        return dict(encoder.widths)
    return dict(encoder.state_widths if config.state_inputs else encoder.base_widths)


def build(encoder: Encoder, config: ValueConfig) -> ValueNet:
    net = ValueNet(encoder, config)
    net._active_feature = encoder.mon_names.index("is_active")
    return net


def widen_net(net: ValueNet, encoder: Encoder, factor: int, *, seed: int = 0) -> ValueNet:
    """`net` with every hidden layer `factor` times as wide, answering exactly as `net` (IKA-405).

    The hidden layers are `mon_dim`, `side_dim` and `head_dim`; the embeddings are not
    widened. The grown net keeps the old units as group 0 of each layer and adds
    `factor - 1` groups. Two things make the old units' values unchanged by the added ones:

    * every layer is followed by a LayerNorm, which over a wider layer would take its mean
      and variance over the added units too. The grown net normalises each group of the
      original width on its own (`ValueConfig.width_groups`, `_GroupLayerNorm`), so group 0
      sees exactly its old inputs.
    * the weights from added input units into the old units are zero, and the final layer
      reads only the old head units at first. The added units are read by nothing that the
      answer depends on until training moves those weights.

    The added units' own incoming weights (from every input, old or added) are the fresh
    default initialisation at `seed`; their LayerNorm is the identity-start (1, 0). The
    value is identical up to float32 rounding: the matrix products run over a longer sum of
    which the extra terms are exact zeros. `net` is not modified.
    """
    config = net.config
    if factor < 1:
        raise ValueError(f"factor {factor}")
    if config.move_properties:
        raise ValueError("widen_net does not handle the move-property branch")
    if config.attention:
        raise ValueError("widen_net does not handle the attention layer")
    if config.width_groups != 1:
        raise ValueError("widen_net grows a net that has not been widened")
    from dataclasses import replace

    wide_config = replace(
        config,
        mon_dim=config.mon_dim * factor,
        side_dim=config.side_dim * factor,
        head_dim=config.head_dim * factor,
        width_groups=factor,
    )
    torch.manual_seed(seed)
    wide = build(encoder, wide_config)
    wide._active_feature = net._active_feature
    widths = net.in_widths
    d, s, h = config.mon_dim, config.side_dim, config.head_dim
    big_d, big_s, big_h = wide_config.mon_dim, wide_config.side_dim, wide_config.head_dim
    mon_in = wide.mon_mlp[0].in_features
    # Linear layer -> (input blocks in the old net, the same blocks in the wide net). A
    # block that is a widened layer's output keeps its first `old` columns in place.
    blocks = {
        "mon_mlp.0": ([mon_in], [mon_in]),
        "mon_mlp.3": ([d], [big_d]),
        "side_mlp.0": ([d] * 4 + [widths["side"]], [big_d] * 4 + [widths["side"]]),
        "head.0": ([s, s, widths["field"]], [big_s, big_s, widths["field"]]),
        "head.4": ([h], [big_h]),
        "head.8": ([h], [big_h]),
    }
    old_state, new_state = net.state_dict(), wide.state_dict()
    with torch.no_grad():
        for key, value in old_state.items():
            target = new_state[key]
            layer = key.rsplit(".", 1)[0]
            if value.shape == target.shape:
                target.copy_(value)
            elif layer in blocks and key.endswith(".weight"):
                old_blocks, new_blocks = blocks[layer]
                columns = []
                offset = 0
                for old_size, new_size in zip(old_blocks, new_blocks, strict=True):
                    columns.append(torch.arange(old_size) + offset)
                    offset += new_size
                target[: value.shape[0]] = 0.0
                target[: value.shape[0], torch.cat(columns)] = value
            else:
                # bias of a Linear, weight and bias of a LayerNorm: first `old` entries.
                target[: value.shape[0]] = value
    wide.load_state_dict(new_state)
    return wide


def grow_state_inputs(net: ValueNet, encoder: Encoder) -> ValueNet:
    """`net` given IKA-425's state columns as zero-weight inputs, answering as `net` did.

    The new columns come last in each first layer's input: after the revision-2 numeric
    columns in `mon_mlp.0` (the numeric state columns, then the two move ids' embeddings),
    after the side vector's revision-2 columns in `side_mlp.0`, after the field's in
    `head.0`. Their weights are zero, every other weight is copied, so before the first step
    the answer is `net`'s up to float32 rounding (the products run over a longer sum whose
    extra terms are exact zeros). `net` is not modified.
    """
    from dataclasses import replace

    if net.config.state_inputs:
        raise ValueError("the net already reads the state columns")
    return _grown_inputs(net, encoder, replace(net.config, state_inputs=True))


def grow_bind_inputs(net: ValueNet, encoder: Encoder) -> ValueNet:
    """`net` (a `state_inputs` net) given IKA-429's bind columns as zero-weight inputs.

    As `grow_state_inputs`: the bind columns come last in `mon_mlp.0` (after the move ids'
    embeddings) and in `side_mlp.0` (after the side's state columns), with zero weights, so
    the grown net answers as `net` did until training moves them.
    """
    from dataclasses import replace

    if not net.config.state_inputs:
        raise ValueError("grow the state columns first (grow_state_inputs)")
    if net.config.bind_inputs:
        raise ValueError("the net already reads the bind columns")
    return _grown_inputs(net, encoder, replace(net.config, bind_inputs=True))


def _grown_inputs(net: ValueNet, encoder: Encoder, config: ValueConfig) -> ValueNet:
    """`net` rebuilt under `config`, whose first layers read more columns, each appended
    after the ones `net` read: old weights copied to the old places, the new ones 0."""
    grown = build(encoder, config)
    grown._active_feature = net._active_feature
    old_state, new_state = net.state_dict(), grown.state_dict()
    d, s = net.config.mon_dim, net.config.side_dim
    old_w, new_w = net.in_widths, grown.in_widths
    mon_old = net.mon_mlp[0].in_features
    # Linear -> the input positions, in the grown layer, of the old layer's columns.
    keep = {
        "mon_mlp.0.weight": torch.arange(mon_old),
        "side_mlp.0.weight": torch.arange(4 * d + old_w["side"]),
        "head.0.weight": torch.cat([torch.arange(2 * s), 2 * s + torch.arange(old_w["field"])]),
    }
    assert grown.side_mlp[0].in_features == 4 * d + new_w["side"]
    assert grown.head[0].in_features == 2 * s + new_w["field"]
    with torch.no_grad():
        for key, value in old_state.items():
            target = new_state[key]
            if key in keep:
                target.zero_()
                target[:, keep[key]] = value
            else:
                target.copy_(value)
    grown.load_state_dict(new_state)
    return grown


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Dataset:
    """Encoded positions with their real outcomes, and where each came from."""

    encoded: Encoded
    #: 1.0 if side 0 won the game this decision came from.
    outcome: np.ndarray
    #: Which game each decision belongs to, so a split can be by game.
    game: np.ndarray
    turn: np.ndarray
    #: The value the search reported at this decision, in the units of whatever leaf
    #: generated the game. It was named `proxy` when every generation was produced with
    #: the `hp-share` objective and the two were the same number; they stopped being the
    #: same the moment generation moved to a learned leaf, and the name did not. It is now
    #: the *previous generation's model, backed by a full one-ply equilibrium search* --
    #: a much stronger baseline than any parameter-free objective, and one a freshly
    #: trained raw evaluation is expected to sit slightly below, because search is what
    #: the difference is made of. Never a target.
    search_value: np.ndarray
    #: The parameter-free `hp-share` of the position, which is the baseline that answers
    #: "is a learned value function worth having at all". Zero-length for datasets encoded
    #: before it was stored.
    hp_share: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: 'move' or 'replacement', as an index into :attr:`kinds`.
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    kinds: tuple[str, ...] = ("move", "replacement")
    #: Opponent pool label per decision, for per-archetype reporting.
    foe: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    foe_names: tuple[str, ...] = ()
    #: Side 1's own value at this decision, in side 0's units -- the record's
    #: `foeSearchValue` (IKA-127) -- NaN where the decision carries none: the open game, a
    #: self-switch, a hidden-bench record written before IKA-127. Under a hidden bench the
    #: two sides solve different games, and `search_value` is side 0's alone. Zero-length
    #: for a dataset encoded before it was stored. Never a target; see `td_target`.
    foe_search_value: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )

    def __len__(self) -> int:
        return int(self.outcome.shape[0])

    def split_by_game(self, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Indices for train and validation, disjoint at the level of whole games.

        Decisions from one game all carry that game's single outcome, so a split over
        decisions would put 13 copies of the same label on both sides and the validation
        number would measure memorisation instead of generalisation.
        """
        games = np.unique(self.game)
        rng = np.random.default_rng(seed)
        rng.shuffle(games)
        cut = int(len(games) * (1.0 - holdout))
        # The same answer as a Python membership test per decision, without a list of
        # 12 million bools (IKA-426).
        is_train = np.isin(self.game, games[:cut])
        return np.flatnonzero(is_train), np.flatnonzero(~is_train)

    def tensors(self, index: np.ndarray, device: torch.device) -> dict[str, Tensor]:
        from .packed import PackedFloat

        e = self.encoded

        def ids(array: np.ndarray) -> Tensor:
            # A packed dataset (`load_dataset`) holds ids narrow; the batch is widened back
            # to the int64 the encoder produced, so the net sees the same tensors either way.
            return torch.from_numpy(np.ascontiguousarray(array[index])).to(device).long()

        def floats(array: Any) -> Tensor:
            # Packed float rows are widened on `device` (IKA-426): the same bits.
            if isinstance(array, PackedFloat):
                return array.tensor(index, device)
            return torch.from_numpy(np.ascontiguousarray(array[index])).to(device)

        return {
            "species": ids(e.species),
            "ability": ids(e.ability),
            "item": ids(e.item),
            "moves": ids(e.moves),
            "mon": floats(e.mon),
            "mask": floats(e.mask),
            "side": floats(e.side),
            "field": floats(e.field),
        }


def split_for(
    dataset: Dataset, holdout: float, config: ValueConfig
) -> tuple[np.ndarray, np.ndarray]:
    """The held-out games a configuration asks for, resolved in exactly one place.

    Which seed decides the split is a rule, and it used to be written down three times --
    in :func:`train`, in the training tool, and again in that tool's learning curve. A
    rule spelled three times is one that will eventually be spelled two ways, and the two
    ways would not look different in any output: both produce a validation number, and
    only the games behind it would have moved.

    ``None`` means "follow ``seed``", which is what every model trained before
    :attr:`ValueConfig.split_seed` existed did, so a stored config read back at its
    default still describes its own run.
    """
    seed = config.seed if config.split_seed is None else config.split_seed
    return dataset.split_by_game(holdout, seed)


def concat_datasets(parts: Sequence[Dataset]) -> Dataset:
    """Joins per-generation shards into one dataset.

    Two index spaces are local to a shard and have to be rebuilt, or the join is silently
    wrong rather than loudly wrong:

    * ``game`` numbers restart at zero in every shard. Left alone, generation 6's game 12
      and generation 7's game 12 would be one game, and :meth:`Dataset.split_by_game`
      would put half of each on both sides of the validation split -- the exact leak that
      splitting by game exists to prevent.
    * ``foe`` indexes into that shard's ``foe_names``, so the same number means different
      opponents in different shards.

    ``hp_share`` is refused rather than zero-filled when a shard predates it. A zero there
    reads as "the opponent has everything", and it would go straight into the baseline
    this project measures the value function against.
    """
    if not parts:
        raise ValueError("nothing to concatenate")
    if len(parts) == 1:
        return parts[0]
    missing = [i for i, p in enumerate(parts) if len(p.hp_share) != len(p)]
    if missing:
        raise ValueError(
            f"shard(s) {missing} carry no hp_share; re-encode them rather than joining, "
            "because a zero there is a real value (the opponent has everything left)"
        )
    kinds = parts[0].kinds
    if any(p.kinds != kinds for p in parts):
        raise ValueError("shards disagree on `kinds`")

    names: list[str] = []
    index: dict[str, int] = {}
    foes: list[np.ndarray] = []
    games: list[np.ndarray] = []
    offset = 0
    for part in parts:
        remap = np.empty(len(part.foe_names), dtype=np.int32)
        for i, name in enumerate(part.foe_names):
            if name not in index:
                index[name] = len(names)
                names.append(name)
            remap[i] = index[name]
        foes.append(remap[part.foe] if len(part.foe) else part.foe)
        games.append(part.game.astype(np.int64) + offset)
        # +1 because game ids are dense from zero; an empty shard leaves the offset alone.
        offset += int(part.game.max()) + 1 if len(part.game) else 0

    unknown: Counter[str] = Counter()
    for part in parts:
        unknown.update(part.encoded.unknown_volatiles)
    from .packed import PackedFloat

    def joined(name: str) -> Any:
        arrays = [getattr(p.encoded, name) for p in parts]
        # Packed shards join packed, without a dense join in between (`load_dataset`).
        if any(isinstance(a, PackedFloat) for a in arrays):
            return PackedFloat.concat(arrays)
        return np.concatenate(arrays)

    encoded = Encoded(
        **{
            name: joined(name)
            for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
        },
        unknown_volatiles=dict(unknown),
    )
    return Dataset(
        encoded=encoded,
        outcome=np.concatenate([p.outcome for p in parts]),
        game=np.concatenate(games).astype(np.int32),
        turn=np.concatenate([p.turn for p in parts]),
        search_value=np.concatenate([p.search_value for p in parts]),
        hp_share=np.concatenate([p.hp_share for p in parts]),
        kind=np.concatenate([p.kind for p in parts]),
        kinds=kinds,
        foe=np.concatenate(foes),
        foe_names=tuple(names),
        # A shard encoded before the column existed has none of these values, which is
        # what NaN says; a zero-length column would misalign every shard after it. When
        # no shard has it the join has none either, as each shard did.
        foe_search_value=(
            np.concatenate(
                [
                    p.foe_search_value.astype(np.float32)
                    if len(p.foe_search_value) == len(p)
                    else np.full(len(p), np.nan, dtype=np.float32)
                    for p in parts
                ]
            )
            if any(len(p.foe_search_value) == len(p) for p in parts)
            else np.zeros(0, dtype=np.float32)
        ),
    )


def td_target(dataset: Dataset, lam: float, *, from_game: int | None = None) -> np.ndarray:
    """``(1 - lam) * outcome + lam * search_value``, the label to fit instead of the outcome.

    The outcome is the truth but it is an extremely noisy sample of it. Every decision in
    a game carries that game's single win or loss, so 13 decisions share one label and the
    independent information in a pool is the number of *games*, not decisions. At turn 1
    the label is close to a coin flip about a position that is not.

    :attr:`Dataset.search_value` is the other end of that trade: the value the generating
    search reported at this very decision, from a full one-ply equilibrium over the
    previous model. It is biased -- it is a model's opinion, and fitting it alone would
    only reproduce the model it came from -- but it is not noisy, and it is already
    recorded at every decision. On the same validation games the generating search scores
    0.8850 / 0.4184 against the raw network's 0.8809 / 0.4221, so the opinion being mixed
    in is a better one than the network currently holds.

    ``lam = 0`` is the outcome and nothing else, which is what every generation so far was
    trained on. ``lam = 1`` is pure distillation of the previous generation and cannot
    exceed it. The useful settings are in between, and which one is a measurement.

    Both arrays are side-0 relative -- all three places that record ``searchValue`` write
    a value in side 0's units, the same orientation as ``outcome`` -- so no flip is needed
    and none is applied.

    *Whose* value it is depends on what the search could see (IKA-127). In the open game
    one agent's two sides solve one matrix, and side 0's equilibrium value is side 1's as
    well. Under a hidden bench they do not: each side solves the Bayesian game over its
    own belief about the other's back two, and ``searchValue`` is side 0's answer alone --
    side 0's win probability as side 0 believes it, conditioned on what side 0 has seen of
    side 1. Side 1's answer to its own game, in the same units, is recorded beside it as
    ``foeSearchValue`` and read into :attr:`Dataset.foe_search_value`. At a self-switch
    it is the chooser's expected score, in side 0's units, and there is no second value.
    So on a hidden-bench pool this target is the outcome mixed with *side 0's* opinion.

    That is kept as it is: changing it changes what every ``--td-lambda`` run learns, and
    which is right is a measurement, not a docstring. The choices, for whoever makes it:

    * side 0's value, what this does. The value net is asked for side 0's win probability
      from the *true* position, and side 0's belief is one of the two opinions of it.
    * the mean of the two, ``(searchValue + foeSearchValue) / 2``, symmetric in the
      seats, so a pool's target does not depend on which seat the book-drawn side sat in.
    * each side's value weighted by how much of the other it had seen -- the side that
      saw more is the better-informed opinion of the true position.

    A decision without ``foeSearchValue`` (open, self-switch, before IKA-127) has NaN
    there, and any mixture must fall back to ``searchValue`` for it.

    :param from_game: when given, rows of games numbered below it keep the outcome alone
        (``lam = 0`` there) and the mix applies from that game on (IKA-403). In a pool of
        several generations the teacher of each shard is the previous generation's leaf,
        and the first one (value-gen11L, log loss 0.4744 on mc2's held-out games) is worse
        than the student while the later ones are better (0.4087 against 0.4150).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if len(dataset.search_value) != len(dataset):
        raise ValueError("the dataset carries no search_value to mix in")
    outcome = dataset.outcome.astype(np.float32)
    if lam == 0.0:
        return outcome
    mixed = ((1.0 - lam) * outcome + lam * dataset.search_value.astype(np.float32)).astype(
        np.float32
    )
    if from_game is None:
        return mixed
    return np.where(dataset.game >= from_game, mixed, outcome).astype(np.float32)


#: Outputs of the auxiliary regression head (`ValueConfig.aux_weight`): the difference in
#: alive fraction and in team HP fraction, side 0 minus side 1.
AUX_DIM = 2


def final_material_targets(dataset: Dataset) -> np.ndarray:
    """(rows, AUX_DIM) float32: each row's game's last recorded position's material difference.

    A game's rows are contiguous and in turn order (checked: ``game`` never goes back to an
    earlier id, ``turn`` never decreases inside a game), so the game's last row is its last
    decision, taken before that turn was resolved. Its ``alive_fraction`` and
    ``team_hp_fraction`` (side 0 minus side 1) are the end-of-game material the auxiliary
    head regresses, repeated on every row of the game. It is the last decision's position,
    not the final board: the turn that ends the game is not in it.
    """
    game = dataset.game
    if len(game) == 0:
        return np.zeros((0, AUX_DIM), np.float32)
    steps = np.diff(game.astype(np.int64))
    if (steps < 0).any():
        raise ValueError("a game's rows must be contiguous and in order for the final material")
    starts = np.r_[0, np.flatnonzero(steps != 0) + 1]
    ends = np.r_[starts[1:], len(game)]
    if len(np.unique(game[starts])) != len(starts):
        raise ValueError("a game id appears in two separate runs of rows")
    from .encode import side_feature_names

    names = side_feature_names()
    columns = [names.index("alive_fraction"), names.index("team_hp_fraction")]
    last = np.asarray(dataset.encoded.side[ends - 1], np.float32)  # (games, 2, side)
    per_game = last[:, 0, :][:, columns] - last[:, 1, :][:, columns]
    return np.repeat(per_game, ends - starts, axis=0).astype(np.float32)


#: IKA-428: the auxiliary targets `ValueConfig.aux_targets` can name, as (kind, dim, loss).
#: The kind is how the head reads the hidden layer (`ValueNet.forward_heads`); the loss is
#: the mean squared error ("mse") or the cross entropy against a probability ("bce").
#: Every target is built from the rows of the encoded dataset (`aux_target_arrays`).
AUX_TARGETS: dict[str, tuple[str, int, str]] = {
    # Each side's alive fraction and team HP fraction at the game's end (`end_material`).
    "end_side": ("side", 2, "mse"),
    # The material difference (alive, HP; side 0 minus side 1) k turns on: the first row
    # of the same game at least k turns later, or the end when there is none.
    "ahead1": ("anti", 2, "mse"),
    "ahead2": ("anti", 2, "mse"),
    "ahead4": ("anti", 2, "mse"),
    # How many of each side's Pokemon faint before the next turn's decision (or the end).
    "ko_next": ("side", 1, "mse"),
    # Turns left in the game, (last row's turn - this turn + 1) / 8. The same for both sides.
    "turns_left": ("sym", 1, "mse"),
    # Whether each Pokemon is alive at the game's end, matched by species (a team holds
    # one of each). Absent slots are not scored.
    "mon_end": ("mon", 1, "bce"),
    # The generating search's value at this decision (`Dataset.search_value`) as a second,
    # separate win probability.
    "search": ("anti", 1, "bce"),
    # IKA-434: the generating search's value minus the generating evaluation model's static
    # value, both as logits of side 0 winning (what a one-turn look ahead adds to the leaf).
    # Not built from the rows: `tools/lookahead_target.py` writes it (NaN where the row's
    # generating leaf was not a learned model) and `aux_target_arrays` takes it as `external`.
    "lookahead": ("anti", 1, "mse_nan"),
}
#: Targets read from outside the dataset's rows (`aux_target_arrays(external=...)`).
EXTERNAL_AUX_TARGETS = frozenset({"lookahead"})


def parse_aux_targets(spec: str) -> list[tuple[str, float]]:
    """``"name:weight,..."`` -> [(name, weight)], refusing an unknown name or a weight <= 0."""
    out: list[tuple[str, float]] = []
    for part in (p for p in spec.split(",") if p):
        name, _, weight = part.partition(":")
        if name not in AUX_TARGETS or not weight or float(weight) <= 0:
            raise ValueError(
                f"aux target {part!r}: want name:weight with a name of {sorted(AUX_TARGETS)} "
                "and a weight above 0"
            )
        out.append((name, float(weight)))
    if len({n for n, _ in out}) != len(out):
        raise ValueError(f"aux targets {spec!r} name one target twice")
    return out


def _game_runs(game: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(starts, ends, run of each row) of the contiguous runs of one game id each."""
    steps = np.diff(game.astype(np.int64))
    if (steps < 0).any():
        raise ValueError("a game's rows must be contiguous and in order")
    starts = np.r_[0, np.flatnonzero(steps != 0) + 1]
    ends = np.r_[starts[1:], len(game)]
    if len(np.unique(game[starts])) != len(starts):
        raise ValueError("a game id appears in two separate runs of rows")
    run = np.repeat(np.arange(len(starts)), ends - starts)
    return starts, ends, run


def _rows_in_chunks(array: Any, index: np.ndarray, pick: Any, chunk: int = 500_000) -> np.ndarray:
    """``pick(np.asarray(array[index]))`` a chunk of rows at a time, concatenated."""
    parts = [
        pick(np.asarray(array[index[s : s + chunk]], np.float32))
        for s in range(0, len(index), chunk)
    ]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def end_material(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(material per row (rows, 2, 2), end material per game (games, 2, 2), ends, run).

    Material is a side's ``alive_fraction`` and ``team_hp_fraction``. The end of a game is
    its last row's material with the losing side's set to 0: the last row is the last
    decision, taken before the turn that ended the game, and the game ended because the
    loser had nothing left. A drawn game (outcome 0.5) keeps the last row as it is.
    """
    from .encode import side_feature_names

    names = side_feature_names()
    columns = [names.index("alive_fraction"), names.index("team_hp_fraction")]
    _starts, ends, run = _game_runs(dataset.game)
    material = _rows_in_chunks(
        dataset.encoded.side, np.arange(len(dataset)), lambda a: a[:, :, columns]
    )
    end = material[ends - 1].copy()
    outcome = np.asarray(dataset.outcome, np.float32)[ends - 1]
    end[outcome == 1.0, 1, :] = 0.0
    end[outcome == 0.0, 0, :] = 0.0
    return material, end, ends, run


def _ahead(dataset: Dataset, k: int, ends: np.ndarray, run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row index of the first row of the same game at least k turns on, whether there is one)."""
    turn = np.asarray(dataset.turn, np.int64)
    if (turn < 0).any() or turn.max(initial=0) >= 1 << 20:
        raise ValueError("turns out of range")
    key = run.astype(np.int64) << 20 | turn
    if (np.diff(key) < 0).any():
        raise ValueError("turns must not decrease inside a game")
    j = np.searchsorted(key, key + k, side="left")
    there = j < ends[run]
    return np.minimum(j, len(key) - 1), there


def aux_target_arrays(
    dataset: Dataset,
    names: Sequence[str],
    encoder: Encoder | None = None,
    external: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Per-row float32 targets of the named `AUX_TARGETS`, in the shapes their heads output.

    ``mon_end`` is (rows, 2, M) with 1 alive, 0 fainted and -1 for an empty slot (not
    scored); it needs the `encoder` for the per-Pokemon column names. The rows of a game
    must be contiguous with turns that never decrease (checked), which every encoded
    dataset is (IKA-425).
    """
    out: dict[str, np.ndarray] = {}
    if not names:
        return out
    material = end = ends = run = None
    if any(n not in EXTERNAL_AUX_TARGETS for n in names):
        material, end, ends, run = end_material(dataset)
    for name in names:
        if name in EXTERNAL_AUX_TARGETS:
            column = None if external is None else external.get(name)
            if column is None or column.shape != (len(dataset),):
                raise ValueError(f"aux target {name!r} needs a column of one value per row")
            if np.isinf(column).any():
                raise ValueError(f"aux target {name!r} has an infinite value (NaN means no target)")
            out[name] = np.asarray(column, np.float32)[:, None]
        elif name == "end_side":
            out[name] = end[run]
        elif name.startswith("ahead") or name == "ko_next":
            k = 1 if name == "ko_next" else int(name[len("ahead"):])
            j, there = _ahead(dataset, k, ends, run)
            later = np.where(there[:, None, None], material[j], end[run])
            if name == "ko_next":
                # alive_fraction is alive / M; the count that faints, never below 0.
                m = dataset.encoded.mask.shape[-1]
                fainted = (material[:, :, 0] - later[:, :, 0]) * m
                out[name] = np.maximum(fainted, 0.0)[:, :, None].astype(np.float32)
            else:
                out[name] = (later[:, 0, :] - later[:, 1, :]).astype(np.float32)
        elif name == "turns_left":
            turn = np.asarray(dataset.turn, np.float32)
            last = turn[ends - 1][run]
            out[name] = ((last - turn + 1.0) / 8.0)[:, None].astype(np.float32)
        elif name == "mon_end":
            if encoder is None:
                raise ValueError("mon_end needs the encoder")
            out[name] = _mon_end(dataset, ends, run, encoder.mon_names.index("fainted"))
        elif name == "search":
            value = np.asarray(dataset.search_value, np.float32)
            if len(value) != len(dataset) or not np.isfinite(value).all():
                raise ValueError("the dataset's search_value is missing or not finite")
            out[name] = np.clip(value, 0.0, 1.0)[:, None]
        else:
            raise ValueError(f"unknown aux target {name!r}")
    return out


def _mon_end(dataset: Dataset, ends: np.ndarray, run: np.ndarray, fainted_col: int) -> np.ndarray:
    """(rows, 2, M): each present Pokemon's alive (1) or fainted (0) at its game's end."""
    e = dataset.encoded
    last = ends - 1
    species_last = np.asarray(e.species[last]).astype(np.int64)  # (games, 2, M)
    present_last = _rows_in_chunks(e.mask, last, lambda a: a) > 0
    alive_last = present_last & (
        _rows_in_chunks(e.mon, last, lambda a: a[..., fainted_col], chunk=100_000) == 0
    )
    outcome = np.asarray(dataset.outcome, np.float32)[last]
    alive_last[outcome == 1.0, 1, :] = False
    alive_last[outcome == 0.0, 0, :] = False
    out = np.empty((len(dataset), *species_last.shape[1:]), np.float32)
    for s in range(0, len(dataset), 500_000):
        rows = np.arange(s, min(s + 500_000, len(dataset)))
        species = np.asarray(e.species[rows]).astype(np.int64)  # (n, 2, M)
        present = np.asarray(e.mask[rows]) > 0
        g = run[rows]
        same = species[:, :, :, None] == species_last[g][:, :, None, :]  # (n, 2, M, M)
        alive = (same & alive_last[g][:, :, None, :]).any(axis=-1)
        found = same.any(axis=-1)
        out[rows] = np.where(present & found, alive.astype(np.float32), -1.0)
    return out


def aux_loss(name: str, output: Tensor, wanted: Tensor) -> Tensor:
    """The loss of one `AUX_TARGETS` head on a batch (IKA-428)."""
    kind, _dim, loss = AUX_TARGETS[name]
    if loss == "mse":
        return ((output - wanted) ** 2).mean()
    if loss == "mse_nan":
        # IKA-434: a NaN target is a row without one; the mean is over the rows with one.
        scored = ~torch.isnan(wanted)
        err = (output - torch.where(scored, wanted, output.detach())) ** 2
        return (err * scored).sum() / scored.sum().clamp(min=1)
    if kind == "mon":
        scored = wanted >= 0
        each = nn.functional.binary_cross_entropy_with_logits(
            output, wanted.clamp(min=0.0), reduction="none"
        )
        return (each * scored).sum() / scored.sum().clamp(min=1)
    return nn.functional.binary_cross_entropy_with_logits(output, wanted)


def game_weights(game: np.ndarray, spec: str) -> np.ndarray:
    """Per-row float32 weights from ``"FROM:W,FROM:W,..."``: a row of game g gets the weight of
    the last entry whose FROM is at most g (IKA-425). FROM must start at 0 and increase."""
    entries = [(int(a), float(b)) for a, b in (part.split(":") for part in spec.split(","))]
    froms = [a for a, _ in entries]
    if froms[0] != 0 or froms != sorted(set(froms)) or any(w < 0 for _, w in entries):
        raise ValueError(f"{spec!r}: FROM must start at 0 and increase, weights be >= 0")
    weights = np.array([w for _, w in entries], np.float32)
    return weights[np.searchsorted(froms, game, side="right") - 1]


def load_ensemble(
    paths: Sequence[str | Path], encoder: Encoder
) -> tuple[list[ValueNet], list[dict[str, Any]]]:
    """Several trained nets to be averaged as one leaf.

    They must share a vocabulary and feature widths -- `load_model` checks both -- because
    averaging logits from nets that mean different things by embedding index 41 would be
    averaging noise.
    """
    nets: list[ValueNet] = []
    metas: list[dict[str, Any]] = []
    for path in paths:
        net, meta = load_model(path, encoder)
        nets.append(net)
        metas.append(meta)
    return nets, metas


def load_dataset(path: str | Path, *, packed: bool = True, cache: bool = True) -> Dataset:
    """Reads a cache written by ``tools/encode_dataset.py``.

    With ``packed`` (the default) the big arrays are kept small and exact: ids in the
    narrowest integer type that holds them, float columns as bits, small codes or float32
    (:mod:`pokeuraou.packed`), each inflated a chunk at a time so the dense arrays are
    never in memory. :meth:`Dataset.tensors` widens a batch back to the dense types, so a
    net sees the same numbers either way. ``packed=False`` is the plain read.

    With ``cache`` (the default, packed only) the packed arrays are written once beside
    the file (`packed.cache_dir`) and read back memory-mapped (IKA-426). The second read
    takes seconds instead of two minutes, adds the arrays to no process's commit, and
    several runs on one file share one copy in memory. The ``.npz`` itself is unchanged.
    A cache that cannot be written (a read-only directory) is reported and skipped.
    """
    from . import packed as packing

    data = np.load(Path(path), allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    keys = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    if packed:
        arrays = None
        directory = packing.cache_dir(path) if cache else None
        if directory is not None:
            arrays = packing.load_cache(directory)
        if arrays is None:
            arrays = {key: _read_packed(Path(path), key) for key in keys}
            if directory is not None:
                try:
                    packing.save_cache(directory, arrays)
                except OSError as exc:
                    print(f"load_dataset: no packed cache for {path} ({exc})")
                else:
                    # Read back mapped, and let the private copy go.
                    arrays = packing.load_cache(directory)
        assert arrays is not None
        arrays = {key: arrays[key] for key in keys}
    else:
        arrays = {
            key: data[key]
            for key in ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
        }
    encoded = Encoded(**arrays, unknown_volatiles=meta.get("unknown_volatiles", {}))
    return Dataset(
        encoded=encoded,
        outcome=data["outcome"],
        game=data["game"],
        turn=data["turn"],
        # `proxy` is the old name for the same array. Files written before the rename are
        # still readable, and are still the generating search's value rather than
        # hp-share, whatever their key says.
        search_value=data["search_value"] if "search_value" in data else data["proxy"],
        hp_share=data["hp_share"] if "hp_share" in data else np.zeros(0, np.float32),
        kind=data["kind"],
        foe=data["foe"],
        foe_names=tuple(meta["foe_names"]),
        foe_search_value=(
            data["foe_search_value"]
            if "foe_search_value" in data
            else np.zeros(0, np.float32)
        ),
    )


def _read_packed(path: Path, key: str) -> Any:
    """One array of a cache in its small exact form (`load_dataset`)."""
    from .packed import NpzMember, PackedFloat, narrow_ints

    member = NpzMember(path, key)
    if member.dtype.kind in "iu":
        return narrow_ints(member.chunks(), member.shape[0], member.shape[1:])
    if member.dtype == np.float32 and len(member.shape) >= 2:
        return PackedFloat.from_chunks(member.chunks(), member.shape)
    return np.load(path, allow_pickle=False)[key]


def save_dataset(path: str | Path, dataset: Dataset, meta: dict[str, Any]) -> None:
    from .packed import AsType, PackedFloat, save_npz_streamed

    e = dataset.encoded
    packed = any(isinstance(getattr(e, k), PackedFloat) for k in ("mon", "mask", "side", "field"))
    # A packed dataset keeps its ids narrow; the file keeps the encoder's int64.
    ids = {
        k: AsType(getattr(e, k), np.int64) if packed else getattr(e, k)
        for k in ("species", "ability", "item", "moves")
    }
    entries = dict(
        **ids,
        mon=e.mon,
        mask=e.mask,
        side=e.side,
        field=e.field,
        outcome=dataset.outcome,
        game=dataset.game,
        turn=dataset.turn,
        search_value=dataset.search_value,
        hp_share=dataset.hp_share,
        kind=dataset.kind,
        foe=dataset.foe,
        foe_search_value=dataset.foe_search_value,
        meta_json=np.asarray(
            json.dumps(
                {
                    **meta,
                    "foe_names": list(dataset.foe_names),
                    "unknown_volatiles": e.unknown_volatiles,
                }
            )
        ),
    )
    if packed:
        save_npz_streamed(Path(path), entries)
    else:
        np.savez_compressed(Path(path), **entries)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EpochReport:
    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float
    seconds: float


def train(
    net: ValueNet,
    dataset: Dataset,
    config: ValueConfig,
    *,
    device: torch.device,
    holdout: float = 0.15,
    log: Any = None,
    train_index: np.ndarray | None = None,
    val_index: np.ndarray | None = None,
    target: np.ndarray | None = None,
    snapshots: dict[str, dict[str, Tensor]] | None = None,
    swap_slots: SwapSlots | None = None,
    aux_target: np.ndarray | None = None,
    row_weight: np.ndarray | None = None,
    aux_targets: dict[str, np.ndarray] | None = None,
) -> tuple[list[EpochReport], dict[str, Tensor]]:
    """Fits the network and returns the epoch history and the best weights.

    "Best" is by validation loss rather than by validation AUC. AUC only asks whether
    winning positions score above losing ones; the tool needs the number itself to be a
    probability, so the criterion has to be the one that punishes miscalibration.

    :param target: what to fit, per decision, when it should not be the game's outcome --
        see :func:`td_target`. Validation is *always* against the real outcome, whatever
        this is, or the number stops meaning "how often does this position win" and rows
        with different targets stop being comparable to each other.
    :param snapshots: when given, filled with the final weights (``"last"``) and both
        averages (``"ema"``, ``"swa"``) of a ``keep="last"`` run, so one fit answers all
        three (`tools/sweep_value.py`).

    :param swap_slots: when given, each training row is shown in a random arrangement of its
        sides' left and right Pokemon, drawn afresh every epoch (IKA-412, `slotswap`). The
        validation rows are never exchanged. Off by default: nothing then changes, nor does
        the random stream the batch order draws from.

    :param aux_target: (rows, AUX_DIM) per-row regression targets (`final_material_targets`),
        needed when ``config.aux_weight`` > 0: the loss is the cross entropy plus
        ``aux_weight`` x the mean squared error of the net's auxiliary head (IKA-425).
    :param row_weight: per-row weights of the cross entropy (`game_weights`); a batch's loss is
        the weighted mean. None keeps the plain mean and the exact computation of every run
        before this was added (IKA-425).
    :param aux_targets: per-row targets of each head named by ``config.aux_targets``
        (`aux_target_arrays`), needed exactly when it is not empty: the loss adds each
        head's weight x `aux_loss` (IKA-428).

    ``epochs=0`` trains nothing and returns the weights the net came in with -- the null
    control of a warm start (`tools/train_value.py --init-from`, IKA-194).
    """
    import time

    if (config.aux_weight > 0) != (aux_target is not None):
        raise ValueError("aux_weight > 0 and aux_target go together")
    if aux_target is not None and not hasattr(net, "aux"):
        raise ValueError("the net has no auxiliary head; build it with aux_weight > 0")
    if row_weight is not None and len(row_weight) != len(dataset):
        raise ValueError("row_weight is one weight per dataset row")
    head_weights = dict(parse_aux_targets(config.aux_targets))
    if set(head_weights) != set(aux_targets or {}):
        raise ValueError("config.aux_targets and aux_targets must name the same targets")
    if head_weights and swap_slots is not None and "mon_end" in head_weights:
        raise ValueError("--swap-slots moves the Pokemon but not the mon_end targets")

    if config.keep not in ("best", "last") or config.average not in ("none", "ema", "swa"):
        raise ValueError(f"keep={config.keep!r} average={config.average!r}")
    if config.average != "none" and config.keep != "last":
        raise ValueError("an average is of the weights the schedule ends on: keep='last'")
    if config.epochs == 0:
        return [], {k: v.detach().clone() for k, v in net.state_dict().items()}
    torch.manual_seed(config.seed)
    if train_index is None or val_index is None:
        train_idx, val_idx = split_for(dataset, holdout, config)
    else:
        # Given explicitly by the learning curve, which varies the training set while
        # holding the validation games fixed so the rows can be compared to each other.
        train_idx, val_idx = train_index, val_index
    outcome = torch.from_numpy(dataset.outcome.astype(np.float32))
    fitted = outcome if target is None else torch.from_numpy(target.astype(np.float32))

    optimiser = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps = max(1, len(train_idx) // config.batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=config.lr,
        total_steps=config.epochs * steps,
        pct_start=config.pct_start,
    )
    averaging = config.keep == "last" and (snapshots is not None or config.average != "none")
    ema = _Averages(net, config) if averaging else None
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(config.seed)
    swap_rng = np.random.default_rng([config.seed, 412]) if swap_slots is not None else None

    history: list[EpochReport] = []
    best = {k: v.detach().clone() for k, v in net.state_dict().items()}
    best_loss = float("inf")
    stale = 0

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        net.train()
        order = train_idx.copy()
        rng.shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, len(order) - config.batch_size + 1, config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            batch = dataset.tensors(batch_idx, device)
            if swap_slots is not None:
                batch = swap_batch(batch, swap_slots.draw(swap_rng, batch_idx), swap_slots)
            heads: dict[str, Tensor] = {}
            if aux_target is None and row_weight is None and not head_weights:
                logit = net(batch)
                loss = loss_fn(logit, fitted[batch_idx].to(device))
            else:
                if head_weights:
                    logit, aux, heads = net.forward_heads(batch)
                elif aux_target is None:
                    logit = net(batch)
                else:
                    logit, aux = net.forward_aux(batch)
                each = nn.functional.binary_cross_entropy_with_logits(
                    logit, fitted[batch_idx].to(device), reduction="none"
                )
                if row_weight is None:
                    loss = each.mean()
                else:
                    weight = torch.from_numpy(row_weight[batch_idx]).to(device)
                    loss = (each * weight).sum() / weight.sum().clamp(min=1e-6)
                if aux_target is not None:
                    wanted = torch.from_numpy(aux_target[batch_idx]).to(device)
                    loss = loss + config.aux_weight * ((aux - wanted) ** 2).mean()
                for name, output in heads.items():
                    assert aux_targets is not None
                    wanted = torch.from_numpy(aux_targets[name][batch_idx]).to(device)
                    loss = loss + head_weights[name] * aux_loss(name, output, wanted)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            if ema is not None:
                ema.step(net)
            total += loss.item() * len(batch_idx)
            seen += len(batch_idx)

        val_logit = predict(net, dataset, val_idx, device=device)
        val_loss = float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(val_logit), outcome[val_idx]
            )
        )
        val_auc = auc(val_logit, dataset.outcome[val_idx])
        report = EpochReport(
            epoch=epoch,
            train_loss=total / max(seen, 1),
            val_loss=val_loss,
            val_auc=val_auc,
            seconds=time.perf_counter() - started,
        )
        history.append(report)
        if log is not None:
            log(report)

        if ema is not None:
            ema.epoch_end(net, epoch)
        if config.keep == "last":
            continue
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if config.keep == "last":
        last = {k: v.detach().clone() for k, v in net.state_dict().items()}
        chosen = {"last": last}
        if ema is not None:
            chosen |= {"ema": ema.ema_weights(), "swa": ema.swa_weights()}
        if snapshots is not None:
            snapshots.update(chosen)
        best = chosen["last" if config.average == "none" else config.average]
    return history, best


class _Averages:
    """The two weight averages of a ``keep="last"`` run (IKA-86), kept side by side.

    EMA after every optimiser step; SWA as the running mean of the end-of-epoch weights
    from epoch ``ceil(swa_from * epochs)``. Both in float32 on the net's device. The net
    has no running statistics (LayerNorm, not BatchNorm), so averaging the state dict is
    the whole of it -- nothing needs recomputing afterwards.
    """

    def __init__(self, net: nn.Module, config: ValueConfig) -> None:
        import math

        self.decay = config.ema_decay
        self.swa_start = max(1, math.ceil(config.swa_from * config.epochs))
        self.ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
        self.swa: dict[str, Tensor] | None = None
        self.swa_n = 0

    @torch.no_grad()
    def step(self, net: nn.Module) -> None:
        for k, v in net.state_dict().items():
            self.ema[k].mul_(self.decay).add_(v.float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def epoch_end(self, net: nn.Module, epoch: int) -> None:
        if epoch < self.swa_start:
            return
        state = net.state_dict()
        if self.swa is None:
            self.swa = {k: v.detach().clone().float() for k, v in state.items()}
        else:
            for k, v in state.items():
                self.swa[k].mul_(self.swa_n / (self.swa_n + 1)).add_(
                    v.float(), alpha=1.0 / (self.swa_n + 1)
                )
        self.swa_n += 1

    def ema_weights(self) -> dict[str, Tensor]:
        return {k: v.clone() for k, v in self.ema.items()}

    def swa_weights(self) -> dict[str, Tensor]:
        assert self.swa is not None
        return {k: v.clone() for k, v in self.swa.items()}


@torch.no_grad()
def predict(
    net: ValueNet,
    dataset: Dataset,
    index: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    net.eval()
    out = np.empty(len(index), dtype=np.float32)
    for start in range(0, len(index), batch_size):
        chunk = index[start : start + batch_size]
        out[start : start + len(chunk)] = (
            net(dataset.tensors(chunk, device)).float().cpu().numpy()
        )
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(a won position scores above a lost one), ties counting a half."""
    wins = labels > 0.5
    if not wins.any() or wins.all():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inverse, weights=ranks) / counts)[inverse]
    n_pos = int(wins.sum())
    n_neg = len(scores) - n_pos
    return float((ranks[wins].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def save_model(
    path: str | Path,
    net: ValueNet,
    weights: dict[str, Tensor],
    vocab: Vocabulary,
    config: ValueConfig,
    meta: dict[str, Any],
    widths: dict[str, int] | None = None,
) -> None:
    """Writes weights with the fingerprint of the vocabulary they were trained on."""
    torch.save(
        {
            "weights": weights,
            # IKA-428: `aux_targets` is written only when set, so a model trained without
            # it (or saved for use, which drops it) reads back in code from before it.
            "config": {
                k: v for k, v in asdict(config).items() if k != "aux_targets" or v
            },
            "format_id": vocab.format_id,
            "vocab_fingerprint": vocab.fingerprint(),
            # Rows per table, index 0 included -- what `load_model` cuts the current
            # vocabulary back to before comparing fingerprints (IKA-82). A model saved
            # before this key existed is read from its embedding shapes instead.
            "vocab_sizes": dict(vocab.sizes),
            "active_feature": net._active_feature,
            # The columns the net reads (`ValueNet.in_widths`) unless told otherwise.
            "widths": dict(widths or net.in_widths),
            "meta": meta,
        },
        Path(path),
    )


#: Embedding table of each vocabulary, as `ValueNet` names them.
_EMBEDDINGS = {"species": "species", "ability": "ability", "item": "item", "move": "move"}


def _model_vocab_sizes(blob: dict[str, Any]) -> dict[str, int]:
    """Rows per table the model was trained with: stored, or read off its embeddings."""
    stored = blob.get("vocab_sizes")
    if stored:
        return {k: int(v) for k, v in stored.items()}
    weights = blob["weights"]
    return {kind: int(weights[f"{name}.weight"].shape[0]) for kind, name in _EMBEDDINGS.items()}


def _grown(
    weights: dict[str, Tensor], have: dict[str, int], want: dict[str, int]
) -> dict[str, Tensor]:
    """The weights with each embedding table given zero rows up to the current size.

    Zero, not the mean of the trained rows and not random: deterministic, and for species,
    ability and item a zero row is exactly what index 0 ("absent or unknown") contributes,
    so a new id reads as the unknown id did before it was appended -- which is how the old
    vocabulary encoded it. For moves it is not quite that, because the move pool divides
    by the number of nonzero indices; a new move is "a move that adds nothing to the
    average". Either way no existing row moves, so a position without a new id scores
    bit for bit as before (tests/test_value.py, IKA-82).
    """
    out = dict(weights)
    for kind, name in _EMBEDDINGS.items():
        extra = want[kind] - have[kind]
        if extra:
            table = weights[f"{name}.weight"]
            out[f"{name}.weight"] = torch.cat([table, table.new_zeros(extra, table.shape[1])])
    return out


def _refuse_unless_extended(blob: dict[str, Any], encoder: Encoder) -> None:
    """A model from another regulation loads only onto a vocabulary that extends its own.

    M-C's committed order begins with M-B's (`configs/vocab/`, IKA-82), so the M-C
    vocabulary cut back to an M-B model's table sizes, and named M-B, is the vocabulary
    that model was trained on -- its fingerprint is the one the model stored. Any other
    pair of regulations, or M-C in the id-sorted order it had before, gives another
    fingerprint and is refused: there the same integer is a different Pokemon.
    """
    have, want = _model_vocab_sizes(blob), encoder.vocab.sizes
    if all(have[k] <= want[k] for k in want):
        cut = encoder.vocab.prefix(have, blob["format_id"]).fingerprint()
        if cut == blob["vocab_fingerprint"]:
            return
    raise ValueError(
        f"model was trained on {blob['format_id']}, encoder is for "
        f"{encoder.vocab.format_id}, and the encoder's vocabulary does not begin with the "
        "model's: the same integer would mean a different Pokemon"
    )


def _checked_move_props(stored: Tensor, encoder: Encoder) -> Tensor:
    """The encoder's move property table, if the model was trained on the same rows of it.

    A model with `move_properties` carries the table it learned from (IKA-318). The rows
    it has must be the current dex's rows, bit for bit: the fingerprint pins which integer
    is which move, and this pins what the model was told each move does. A grown
    vocabulary's appended rows come from the current dex -- which is the point of reading
    properties rather than ids alone, an unseen move is not blank. A changed column layout
    or a changed row is refused, as a moved id is.
    """
    current = move_property_table(encoder)
    rows, width = int(stored.shape[0]), int(stored.shape[1])
    if width != current.shape[1] or rows > current.shape[0]:
        raise ValueError(
            f"the model's move property table is {rows}x{width} and this dex gives "
            f"{current.shape[0]}x{current.shape[1]}: the columns changed meaning"
        )
    if not np.array_equal(stored.cpu().numpy(), current[:rows]):
        changed = int((stored.cpu().numpy() != current[:rows]).any(axis=1).sum())
        raise ValueError(
            f"{changed} move(s) have other properties in this dex than the model was "
            "trained on; the same move no longer reads as the same move"
        )
    return torch.from_numpy(current)


def load_model(path: str | Path, encoder: Encoder) -> tuple[ValueNet, dict[str, Any]]:
    """Loads weights, refusing a vocabulary they were not trained against.

    The refusal is the point. A model trained on M-B and loaded against M-C would run
    perfectly and answer about the wrong Pokemon, since the same integer indexes a
    different species.

    One change is not a refusal: a vocabulary that has only grown at the end (IKA-82).
    The order is append-only (`configs/vocab/`), so the current vocabulary cut back to
    the model's table sizes is the one it was trained on, and its fingerprint must be the
    one the model stored. Then the embedding tables get zero rows for the appended ids
    (`_grown`) and every other weight loads as it is. Anything else -- an index that moved,
    a table that shrank -- still fails the fingerprint and is refused.

    The same holds across regulations whose order extends another's: an M-B model loads
    onto M-C (`_refuse_unless_extended`) and `meta["vocab_extended_from"]` names M-B.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    weights = blob["weights"]
    if blob["format_id"] != encoder.vocab.format_id:
        _refuse_unless_extended(blob, encoder)
    if blob["vocab_fingerprint"] != encoder.vocab.fingerprint():
        have, want = _model_vocab_sizes(blob), encoder.vocab.sizes
        shrank = [k for k in want if have[k] > want[k]]
        prefix = None if shrank else encoder.vocab.prefix(have, blob["format_id"]).fingerprint()
        if prefix != blob["vocab_fingerprint"]:
            detail = (
                f"its tables are larger ({', '.join(shrank)})"
                if shrank
                else f"the current one cut back to its sizes is {prefix}, so it is not a "
                "prefix: an existing id changed its integer"
            )
            raise ValueError(
                f"vocabulary fingerprint {encoder.vocab.fingerprint()} does not match the "
                f"model's {blob['vocab_fingerprint']} and {detail}; the same integer no "
                "longer means the same Pokemon"
            )
        weights = _grown(weights, have, want)
        blob["meta"] = {
            **(blob.get("meta") or {}),
            "vocab_grown_from": {k: have[k] for k in want if have[k] != want[k]},
        }
        if blob["format_id"] != encoder.vocab.format_id:
            blob["meta"]["vocab_extended_from"] = blob["format_id"]
    # The fingerprint covers the vocabulary -- which species is which integer -- and not
    # the numeric feature blocks beside it. Adding a side feature shifts every feature
    # after it, and a model loaded across that change would read boosts where it expects
    # HP. A shape mismatch would eventually raise from `load_state_dict`, but only if the
    # change happened to alter a width the weights touch, and the message would name a
    # tensor rather than the cause.
    #
    # IKA-425 appended state columns to each block (revision 3). A model without
    # `state_inputs` reads the revision-2 prefix, so what it stored must be that prefix;
    # one with it reads every column.
    # IKA-429 appended the bind columns (revision 4); only a `bind_inputs` model reads them.
    config = ValueConfig(**blob["config"])
    expected = net_widths(encoder, config)
    stored = blob.get("widths") or {}
    if stored and stored != expected:
        raise ValueError(
            f"feature widths {expected} do not match the model's {stored}; the "
            "encoder gained or lost a feature, so the same column no longer means the "
            "same quantity"
        )
    if "move_props" in weights:
        weights = dict(weights)
        weights["move_props"] = _checked_move_props(weights["move_props"], encoder)
    net = build(encoder, config)
    net._active_feature = int(blob["active_feature"])
    net.load_state_dict(weights)
    net.eval()
    return net, blob.get("meta", {})


__all__ = [
    "BatchedValue",
    "Dataset",
    "EpochReport",
    "ValueConfig",
    "ValueNet",
    "auc",
    "build",
    "grow_bind_inputs",
    "grow_state_inputs",
    "net_widths",
    "load_dataset",
    "load_model",
    "move_property_table",
    "predict",
    "save_dataset",
    "save_model",
    "split_for",
    "train",
    "widen_net",
]


# ---------------------------------------------------------------------------
# Evaluating positions
# ---------------------------------------------------------------------------


#: The arrays of an `Encoded` the net reads.
_ENCODED_ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


class BatchedValue:
    """Win probabilities for many positions in one forward pass.

    Both the selection resolver and the next generation of self-play need the same thing:
    hundreds or thousands of positions scored at once. Scoring them one at a time would be
    absurd, so the batch is the unit. That part has not changed.

    The numbers that used to stand here have. They were `to_json` 58 ms, `encode` 30 ms
    and forward 3 ms for 256 positions, and a 16x16 node on which the forward pass was
    0.8% of the cost. Both are history rather than argument now: `to_json` has not been
    on this path since positions started going straight to `encode_positions`, so the
    largest of those three terms no longer exists, and the width is 24.

    Measured again 2026-09-20 over whole generation runs rather than one node
    (`tools/profile_stages.py generation`, 300 games, eight workers, width 24,
    `value-gen11L`, one CUDA card), as a share of a worker's wall clock:

        bridge off   branch generation 85.8%, encoding 8.1%, forward 0.4%
        bridge on    branch generation 26.6%, encoding 2.9%, forward 32.3%

    **Those two lines are from earlier the same day than the code below them**, and the
    32.3% is the part that has since moved. The forward pass had not got slower: the
    bridge refuses about 2.7% of a node's cells, each refused cell was filled here as a
    1x1 node, and each paid a forward pass of its own -- 45,777 forward passes for 2,673
    nodes over those 300 games, and on a 60-game run that counted rows as well as calls,
    94.9% of the calls carrying 6.0% of the rows, 1,110 rows a call for a whole node
    against 3.9 for a refused cell.

    IKA-52 and IKA-53 landed that afternoon and took it away: a node's refused cells go
    to `batched_payoffs` in one call instead of one each, and the two names behind 88% of
    the refusals joined the port's list. On the same 60-game run the forward passes went
    10,930 to 776 and the run 21.8s to 10.3s.

    The paragraph is kept rather than deleted because it is what this class is for. The
    batching was being undone one cell at a time downstream of here, and nothing visible
    from inside this class could have shown it -- only a breakdown taken across the whole
    run could, which is the reason `pokeuraou.timing` exists.
    """

    def __init__(
        self,
        net: ValueNet | Sequence[ValueNet],
        encoder: Encoder,
        *,
        device: torch.device | None = None,
        batch_size: int = 8192,
    ) -> None:
        #: One net, or several to average. Several because a *training run* moves more
        #: than the settings being compared do: the same data and configuration at three
        #: seeds spanned 0.9489 to 0.9725 on one held-out set, where the configurations
        #: under test differed by 0.004. A comparison of two single runs measures seed
        #: luck. Averaging removes it from the leaf, so what is left to measure is the
        #: thing that was changed.
        self.nets = [net.eval()] if isinstance(net, ValueNet) else [n.eval() for n in net]
        self.net = self.nets[0]
        self.encoder = encoder
        self.device = device or next(self.nets[0].parameters()).device
        self.batch_size = batch_size
        #: Positions scored so far, so a caller can report the cost it incurred.
        self.evaluated = 0
        #: Of those, how many the battle had ended in (IKA-253): scored as their result,
        #: or by the net under `rules.net_scores_ends`. What a match echoes per arm.
        self.ended = 0
        #: Members stacked into one call. Three members run one after another cost 2.9x a
        #: single net rather than the 1.15x their arithmetic would suggest, because the
        #: GPU here is bound by launch latency -- 2.1 ms whether the batch is 8 rows or
        #: 2,048 -- so three calls pay the fixed cost three times. Stacking the weights
        #: and mapping over them makes it one call: 1.13x to 1.19x at the batch sizes a
        #: node actually produces.
        self._stacked: Any = None
        if len(self.nets) > 1:
            self._stack()

    def _stack(self) -> None:
        import copy

        from torch.func import functional_call, stack_module_state

        members = [n.to(self.device) for n in self.nets]
        params, buffers = stack_module_state(members)
        base = copy.deepcopy(members[0]).to("meta")

        def one(p, b, x):  # noqa: ANN001, ANN202
            return functional_call(base, (p, b), (x,))

        mapped = torch.vmap(one, in_dims=(0, 0, None))
        self._stacked = lambda batch: mapped(params, buffers, batch).mean(0)

    def _mean_logit(self, batch: dict[str, Tensor]) -> Tensor:
        """The average of the members' logits, which stays antisymmetric.

        Each member returns `head(ours, theirs) - head(theirs, ours)`, so mirroring a
        position negates its logit exactly. A mean of negated logits is the negation of
        the mean, and `V(x) + V(mirror x) = 1` survives the ensemble unchanged. Averaging
        probabilities would preserve it too; logits are averaged because that is where the
        model is linear and a confident member does not get flattened by an unsure one.
        """
        if len(self.nets) == 1:
            return self.nets[0](batch)
        # The stacked path and the sequential one differ in the last places of a float32
        # sum -- measured at 1.9e-06 -- and this project treats that as a real difference:
        # a changed payoff is a changed equilibrium. So there is one path, not a fast one
        # and a reference one to fall back on.
        return self._stacked(batch)

    @timing.timed("forward")
    @torch.no_grad()
    def __call__(self, positions: list[Any]) -> np.ndarray:
        """(N,) probability that side 0 wins, for Position objects or their JSON form."""
        if not positions:
            return np.zeros(0, dtype=np.float64)
        # Position objects go straight to `encode_positions`; only a caller that already
        # has JSON pays the conversion, and `to_json` was 64% of the per-leaf cost.
        as_json = bool(positions) and isinstance(positions[0], dict)
        out = np.empty(len(positions), dtype=np.float64)
        for start in range(0, len(positions), self.batch_size):
            chunk = positions[start : start + self.batch_size]
            encoded = (
                self.encoder.encode(chunk)
                if as_json
                else self.encoder.encode_positions(chunk)
            )
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
            batch = {k: v.to(self.device) for k, v in batch.items()}
            values = torch.sigmoid(self._mean_logit(batch)).double().cpu().numpy()
            # An ended position is its result, not the net's guess at it (IKA-253).
            self.ended += settle(values, encoded, self.encoder.rules)
            out[start : start + len(chunk)] = values
        self.evaluated += len(positions)
        timing.count("leaves", len(positions))
        timing.count("forward.passes", -(-len(positions) // self.batch_size))
        return out

    @timing.timed("forward")
    @torch.no_grad()
    def from_encoded(self, encoded: Encoded) -> np.ndarray:
        """(N,) win probability for a batch that is already encoded.

        A caller that can produce the arrays some other way -- the Rust port does, from
        the leaves it already holds -- should not have to hand back positions for this to
        re-encode. The 11% encoding and 5% forward pass this used to cite were measured
        before the port encoded anything; on 2026-09-20 the same generation run is 2.9%
        encoding here plus 3.3% in the port, against 32.3% of forward pass.

        The forward pass stays here on purpose. It is float32 matrix arithmetic, and a
        second implementation would sum it in a different order; a difference in the last
        places of a win probability is a difference in the payoff, which is a difference in
        the equilibrium. Only the input crosses.
        """
        n = len(encoded.species)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            batch = {
                "species": torch.from_numpy(encoded.species[start:stop]),
                "ability": torch.from_numpy(encoded.ability[start:stop]),
                "item": torch.from_numpy(encoded.item[start:stop]),
                "moves": torch.from_numpy(encoded.moves[start:stop]),
                "mon": torch.from_numpy(encoded.mon[start:stop]),
                "mask": torch.from_numpy(encoded.mask[start:stop]),
                "side": torch.from_numpy(encoded.side[start:stop]),
                "field": torch.from_numpy(encoded.field[start:stop]),
            }
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out[start:stop] = torch.sigmoid(self._mean_logit(batch)).double().cpu().numpy()
        # An ended position is its result, not the net's guess at it (IKA-253). The port
        # says which leaves ended (`EncodedNode.unpack`); an encoding with no word on it
        # (a training set) is left as the net scored it.
        self.ended += settle(out, encoded, self.encoder.rules)
        self.evaluated += n
        timing.count("leaves", n)
        timing.count("forward.passes", -(-n // self.batch_size))
        return out

    @timing.timed("forward")
    @torch.no_grad()
    def from_encoded_segments(self, segments: Sequence[Encoded]) -> list[np.ndarray]:
        """`from_encoded` of each block, with one wait for the device instead of one each.

        IKA-291: a depth-2 pass scores dozens of sub-games, each a few hundred rows. Stacking
        them into one batch would be one forward pass, but not the same answers -- a row's
        value moves with the number of rows in its call (37 different answers for the same
        eight rows between 8 and 988 rows on the card), and depth 2 was accepted on games
        these answers made. So every block is cut and run exactly as `from_encoded` would
        run it alone, and only the copy back waits for the device, once, at the end.
        """
        outs = [np.empty(len(encoded.species), dtype=np.float64) for encoded in segments]
        pending: list[tuple[int, int, int, Tensor]] = []
        passes = 0
        for index, encoded in enumerate(segments):
            n = len(encoded.species)
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                batch = {
                    name: torch.from_numpy(getattr(encoded, name)[start:stop]).to(self.device)
                    for name in _ENCODED_ARRAYS
                }
                pending.append(
                    (index, start, stop, torch.sigmoid(self._mean_logit(batch)).double())
                )
                passes += 1
        if pending:
            host = torch.cat([scores for *_where, scores in pending]).cpu().numpy()
            at = 0
            for index, start, stop, _scores in pending:
                outs[index][start:stop] = host[at : at + stop - start]
                at += stop - start
        for encoded, out in zip(segments, outs, strict=True):
            # Each block's ended leaves are their results, as `from_encoded` settles them.
            self.ended += settle(out, encoded, self.encoder.rules)
        rows = sum(len(out) for out in outs)
        self.evaluated += rows
        timing.count("leaves", rows)
        timing.count("forward.passes", passes)
        return outs

    def objective(self, name: str = "win") -> Any:  # noqa: ANN401
        """The value function as a single-position :class:`~pokeuraou.payoff.Objective`.

        `__call__` takes one position at a time and is correspondingly slow; `batch`
        carries this object's own batch form, so a caller holding a whole node's leaves
        pays one forward pass rather than one per leaf -- 1.86x on a 24x24 matrix over
        four spread classes, measured through the analyser.
        """
        from .payoff import _Objective

        return _Objective(
            name=name,
            formula="学習した価値関数（勝敗を教師に学習した勝率）",
            blind_to="学習に使った探索の強さと相手プールに条件付き",
            _value=lambda pos: float(self([pos])[0]),
            _batch=lambda positions: self(list(positions)),
            from_encoded=self.from_encoded,
        )
