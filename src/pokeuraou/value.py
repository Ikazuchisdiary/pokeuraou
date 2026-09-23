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


def td_target(dataset: Dataset, lam: float) -> np.ndarray:
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
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if len(dataset.search_value) != len(dataset):
        raise ValueError("the dataset carries no search_value to mix in")
    outcome = dataset.outcome.astype(np.float32)
    if lam == 0.0:
        return outcome
    return ((1.0 - lam) * outcome + lam * dataset.search_value.astype(np.float32)).astype(
        np.float32
    )


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
        foe_search_value=(
            data["foe_search_value"]
            if "foe_search_value" in data
            else np.zeros(0, np.float32)
        ),
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
        foe_search_value=dataset.foe_search_value,
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
    target: np.ndarray | None = None,
    snapshots: dict[str, dict[str, Tensor]] | None = None,
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

    ``epochs=0`` trains nothing and returns the weights the net came in with -- the null
    control of a warm start (`tools/train_value.py --init-from`, IKA-194).
    """
    import time

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
            logit = net(batch)
            loss = loss_fn(logit, fitted[batch_idx].to(device))
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
            "config": asdict(config),
            "format_id": vocab.format_id,
            "vocab_fingerprint": vocab.fingerprint(),
            # Rows per table, index 0 included -- what `load_model` cuts the current
            # vocabulary back to before comparing fingerprints (IKA-82). A model saved
            # before this key existed is read from its embedding shapes instead.
            "vocab_sizes": dict(vocab.sizes),
            "active_feature": net._active_feature,
            "widths": dict(widths or {}),
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
    "load_dataset",
    "load_model",
    "predict",
    "save_dataset",
    "save_model",
    "split_for",
    "train",
]


# ---------------------------------------------------------------------------
# Evaluating positions
# ---------------------------------------------------------------------------


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
            out[start : start + len(chunk)] = (
                torch.sigmoid(self._mean_logit(batch)).double().cpu().numpy()
            )
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
        self.evaluated += n
        timing.count("leaves", n)
        timing.count("forward.passes", -(-n // self.batch_size))
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
