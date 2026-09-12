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

from .encode import Encoded, Encoder, Vocabulary


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
        widths = encoder.widths

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
        )
        self.mon_mlp = nn.Sequential(
            nn.Linear(mon_in, config.mon_dim),
            nn.LayerNorm(config.mon_dim),
            nn.GELU(),
            nn.Linear(config.mon_dim, config.mon_dim),
            nn.LayerNorm(config.mon_dim),
            nn.GELU(),
        )
        # active mean, active max, bench mean, bench max
        pooled = 4 * config.mon_dim + widths["side"]
        self.side_mlp = nn.Sequential(
            nn.Linear(pooled, config.side_dim),
            nn.LayerNorm(config.side_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * config.side_dim + widths["field"], config.head_dim),
            nn.LayerNorm(config.head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_dim, config.head_dim),
            nn.LayerNorm(config.head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_dim, 1),
        )

    def side_vectors(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, 2, side_dim), computed without reference to which side is which."""
        moves = self.move(batch["moves"])  # (B,2,M,4,move_dim)
        move_mask = (batch["moves"] > 0).float().unsqueeze(-1)
        move_pooled = (moves * move_mask).sum(dim=3) / move_mask.sum(dim=3).clamp(min=1.0)

        features = torch.cat(
            [
                self.species(batch["species"]),
                self.ability(batch["ability"]),
                self.item(batch["item"]),
                move_pooled,
                batch["mon"],
            ],
            dim=-1,
        )
        mon = self.mon_mlp(features)  # (B,2,M,mon_dim)

        present = batch["mask"].unsqueeze(-1)
        is_active = batch["mon"][..., self._active_feature].unsqueeze(-1)
        active = present * is_active
        bench = present * (1.0 - is_active)
        pooled = torch.cat(
            [
                _masked_mean(mon, active),
                _masked_max(mon, active),
                _masked_mean(mon, bench),
                _masked_max(mon, bench),
                batch["side"],
            ],
            dim=-1,
        )
        return self.side_mlp(pooled)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Logit of side 0 winning, antisymmetric by construction."""
        sides = self.side_vectors(batch)
        ours, theirs = sides[:, 0], sides[:, 1]
        field = batch["field"]
        forward = self.head(torch.cat([ours, theirs, field], dim=-1))
        mirrored = self.head(torch.cat([theirs, ours, field], dim=-1))
        return (forward - mirrored).squeeze(-1)

    #: Index of `is_active` inside the per-Pokemon numeric block. Set by :func:`build`.
    _active_feature: int = 0


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    filled = values.masked_fill(mask <= 0, float("-inf"))
    out = filled.max(dim=2).values
    # A side with nothing in the group (no bench left) would otherwise be -inf.
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def build(encoder: Encoder, config: ValueConfig) -> ValueNet:
    net = ValueNet(encoder, config)
    net._active_feature = encoder.mon_names.index("is_active")
    return net


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
        train_games = set(games[:cut].tolist())
        is_train = np.array([g in train_games for g in self.game])
        return np.flatnonzero(is_train), np.flatnonzero(~is_train)

    def tensors(self, index: np.ndarray, device: torch.device) -> dict[str, Tensor]:
        e = self.encoded
        return {
            "species": torch.from_numpy(np.ascontiguousarray(e.species[index])).to(device),
            "ability": torch.from_numpy(np.ascontiguousarray(e.ability[index])).to(device),
            "item": torch.from_numpy(np.ascontiguousarray(e.item[index])).to(device),
            "moves": torch.from_numpy(np.ascontiguousarray(e.moves[index])).to(device),
            "mon": torch.from_numpy(np.ascontiguousarray(e.mon[index])).to(device),
            "mask": torch.from_numpy(np.ascontiguousarray(e.mask[index])).to(device),
            "side": torch.from_numpy(np.ascontiguousarray(e.side[index])).to(device),
            "field": torch.from_numpy(np.ascontiguousarray(e.field[index])).to(device),
        }


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
    encoded = Encoded(
        **{
            name: np.concatenate([getattr(p.encoded, name) for p in parts])
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
    )


def load_dataset(path: str | Path) -> Dataset:
    """Reads a cache written by ``tools/encode_dataset.py``."""
    data = np.load(Path(path), allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    encoded = Encoded(
        species=data["species"],
        ability=data["ability"],
        item=data["item"],
        moves=data["moves"],
        mon=data["mon"],
        mask=data["mask"],
        side=data["side"],
        field=data["field"],
        unknown_volatiles=meta.get("unknown_volatiles", {}),
    )
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
    )


def save_dataset(path: str | Path, dataset: Dataset, meta: dict[str, Any]) -> None:
    e = dataset.encoded
    np.savez_compressed(
        Path(path),
        species=e.species,
        ability=e.ability,
        item=e.item,
        moves=e.moves,
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
        meta_json=json.dumps(
            {
                **meta,
                "foe_names": list(dataset.foe_names),
                "unknown_volatiles": e.unknown_volatiles,
            }
        ),
    )


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
) -> tuple[list[EpochReport], dict[str, Tensor]]:
    """Fits the network and returns the epoch history and the best weights.

    "Best" is by validation loss rather than by validation AUC. AUC only asks whether
    winning positions score above losing ones; the tool needs the number itself to be a
    probability, so the criterion has to be the one that punishes miscalibration.
    """
    import time

    torch.manual_seed(config.seed)
    if train_index is None or val_index is None:
        train_idx, val_idx = dataset.split_by_game(holdout, config.seed)
    else:
        # Given explicitly by the learning curve, which varies the training set while
        # holding the validation games fixed so the rows can be compared to each other.
        train_idx, val_idx = train_index, val_index
    outcome = torch.from_numpy(dataset.outcome.astype(np.float32))

    optimiser = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps = max(1, len(train_idx) // config.batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=config.lr, total_steps=config.epochs * steps, pct_start=0.2
    )
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(config.seed)

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
            target = outcome[batch_idx].to(device)
            logit = net(batch)
            loss = loss_fn(logit, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            schedule.step()
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

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    return history, best


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
            "config": asdict(config),
            "format_id": vocab.format_id,
            "vocab_fingerprint": vocab.fingerprint(),
            "active_feature": net._active_feature,
            "widths": dict(widths or {}),
            "meta": meta,
        },
        Path(path),
    )


def load_model(path: str | Path, encoder: Encoder) -> tuple[ValueNet, dict[str, Any]]:
    """Loads weights, refusing a vocabulary they were not trained against.

    The refusal is the point. A model trained on M-B and loaded against M-C would run
    perfectly and answer about the wrong Pokemon, since the same integer indexes a
    different species.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if blob["format_id"] != encoder.vocab.format_id:
        raise ValueError(
            f"model was trained on {blob['format_id']}, encoder is for "
            f"{encoder.vocab.format_id}"
        )
    if blob["vocab_fingerprint"] != encoder.vocab.fingerprint():
        raise ValueError(
            f"vocabulary fingerprint {encoder.vocab.fingerprint()} does not match the "
            f"model's {blob['vocab_fingerprint']}; the regulation dump changed under it, "
            "so the same integer no longer means the same Pokemon"
        )
    # The fingerprint covers the vocabulary -- which species is which integer -- and not
    # the numeric feature blocks beside it. Adding a side feature shifts every feature
    # after it, and a model loaded across that change would read boosts where it expects
    # HP. A shape mismatch would eventually raise from `load_state_dict`, but only if the
    # change happened to alter a width the weights touch, and the message would name a
    # tensor rather than the cause.
    stored = blob.get("widths") or {}
    if stored and stored != encoder.widths:
        raise ValueError(
            f"feature widths {encoder.widths} do not match the model's {stored}; the "
            "encoder gained or lost a feature, so the same column no longer means the "
            "same quantity"
        )
    config = ValueConfig(**blob["config"])
    net = build(encoder, config)
    net._active_feature = int(blob["active_feature"])
    net.load_state_dict(blob["weights"])
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
    "load_dataset",
    "load_model",
    "predict",
    "save_dataset",
    "save_model",
    "train",
]


# ---------------------------------------------------------------------------
# Evaluating positions
# ---------------------------------------------------------------------------


class BatchedValue:
    """Win probabilities for many positions in one forward pass.

    Both the selection resolver and the next generation of self-play need the same thing:
    hundreds or thousands of positions scored at once. Scoring them one at a time would be
    absurd -- measured on a 16x16 node, the forward pass is 0.8% of the cost and the
    per-position work around it is the rest -- so the batch is the unit.

    The cost breakdown is worth keeping in view, because it says where optimisation
    belongs: for 256 positions, `to_json` 58 ms, `encode` 30 ms, forward 3 ms.
    """

    def __init__(
        self,
        net: ValueNet,
        encoder: Encoder,
        *,
        device: torch.device | None = None,
        batch_size: int = 8192,
    ) -> None:
        self.net = net.eval()
        self.encoder = encoder
        self.device = device or next(net.parameters()).device
        self.batch_size = batch_size
        #: Positions scored so far, so a caller can report the cost it incurred.
        self.evaluated = 0

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
            out[start : start + len(chunk)] = (
                torch.sigmoid(self.net(batch)).double().cpu().numpy()
            )
        self.evaluated += len(positions)
        return out

    @torch.no_grad()
    def from_encoded(self, encoded: Encoded) -> np.ndarray:
        """(N,) win probability for a batch that is already encoded.

        The encoding is 11% of a generation run and the forward pass is 5%, so a caller
        that can produce the arrays some other way -- the Rust port does, from the leaves
        it already holds -- should not have to hand back positions for this to re-encode.

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
            out[start:stop] = torch.sigmoid(self.net(batch)).double().cpu().numpy()
        self.evaluated += n
        return out

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
