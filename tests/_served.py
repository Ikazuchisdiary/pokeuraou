"""A leaf the port can ask: an inference server of the tests' own, and the `RemoteValue` on it.

Since IKA-460 the hidden bench's node, the replacement node and a game's steps are the port's, and
the port scores them on the inference server (`inference.RemoteValue`); a leaf held in the test's
process is `beliefnode.UnservedLeaf`. This module serves a test's stand-in from a thread of the
test's own process with the real protocol (a socket and shared memory, `inference.serve`), with
no torch and no GPU, so the search and the game's roads run in CI.

* `serving(reg, from_encoded)`: a context manager that serves a function of an `Encoded` batch and
  yields the `RemoteValue` on it (its `__call__` encodes positions and asks the server, as the
  real leaf does).
* `hp_share_model(encoder)`: the stand-in for `payoff.HP_SHARE` read off the encoding (remaining
  HP of each side's live members over each side's total, side 0's share of the two): what a
  Rust inference server's hp-share arm gives, and what the tests that took `HP_SHARE` as "a leaf"
  use. It agrees with `HP_SHARE` to the rounding of the encoder's HP scale.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from pokeuraou.encode import HP_SCALE, Encoded, Encoder
from pokeuraou.inference import RemoteValue, serve

_ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def encoded_of(arrays: dict[str, np.ndarray]) -> Encoded:
    """The server's arrays as the `Encoded` batch a leaf scores."""
    return Encoded(**{name: arrays[name] for name in _ARRAYS}, unknown_volatiles={}, decided=None)


def model_of(from_encoded: Callable[[Encoded], np.ndarray]) -> Callable[[dict, int], np.ndarray]:
    """A server model (``(arrays, rows) -> values``) from a function of an `Encoded` batch."""

    def model(arrays: dict[str, np.ndarray], rows: int) -> np.ndarray:
        del rows
        return np.asarray(from_encoded(encoded_of(arrays)), dtype=np.float64)

    return model


@contextmanager
def serving(
    reg: Any,  # noqa: ANN401 - a Regulation
    from_encoded: Callable[[Encoded], np.ndarray] | None = None,
    *,
    model: Callable[[dict, int], np.ndarray] | None = None,
    encoder: Encoder | None = None,
    batch_size: int | None = None,
    buffer_bytes: int = 32 << 20,
) -> Iterator[RemoteValue]:
    """A `RemoteValue` on a server of this process that scores with ``from_encoded`` (or the
    server model ``model``)."""
    encoder = encoder or Encoder(reg)
    if model is None:
        assert from_encoded is not None
        model = model_of(from_encoded)
    shut, address = serve({"stub": model})
    extra = {} if batch_size is None else {"batch_size": batch_size}
    try:
        with RemoteValue(address, "stub", encoder, buffer_bytes=buffer_bytes, **extra) as leaf:
            yield leaf
    finally:
        shut.shutdown()
        shut.server_close()


_OPEN: list[Callable[[], None]] = []


def leaf_of(
    scores: Any,  # noqa: ANN401 - an object with ``from_encoded(Encoded) -> values``
    encoder: Encoder,
    *,
    batch_size: int | None = None,
    buffer_bytes: int = 32 << 20,
) -> RemoteValue:
    """A `RemoteValue` on a server of this process scoring with ``scores.from_encoded``. It is
    closed at the end of the test that made it (`close_all`, called by `conftest`); the scorer
    is ``leaf.scores``, so a test can read what the server was asked."""
    shut, address = serve({"stub": model_of(scores.from_encoded)})
    extra = {} if batch_size is None else {"batch_size": batch_size}
    leaf = RemoteValue(address, "stub", encoder, buffer_bytes=buffer_bytes, **extra)
    leaf.scores = scores

    def close() -> None:
        leaf.close()
        shut.shutdown()
        shut.server_close()

    _OPEN.append(close)
    return leaf


def close_all() -> None:
    """Every leaf `leaf_of` made, closed."""
    while _OPEN:
        _OPEN.pop()()


_HP_LEAF: list[RemoteValue] = []
_HP_CLOSE: list[Callable[[], None]] = []


def hp_share_leaf() -> RemoteValue:
    """One `RemoteValue` of this process on a server of `hp_share_model`: the stand-in, in the
    tests, for the Rust inference server's hp-share arm (`payoff.HP_SHARE` as a served leaf).
    Made on first use and closed when the process ends."""
    if not _HP_LEAF:
        import atexit

        from pokeuraou.regulation import load_regulation

        encoder = Encoder(load_regulation("gen9championsvgc2026regmc"))
        shut, address = serve({"stub": model_of(hp_share_model(encoder))})
        leaf = RemoteValue(address, "stub", encoder, buffer_bytes=64 << 20)

        def close() -> None:
            leaf.close()
            shut.shutdown()
            shut.server_close()

        atexit.register(close)
        _HP_LEAF.append(leaf)
    return _HP_LEAF[0]


_FROM: dict[int, tuple[Any, RemoteValue]] = {}


def served_from(owner: Any) -> RemoteValue:  # noqa: ANN401 - a leaf with `from_encoded`, `encoder`
    """A `RemoteValue` of this process on a server answering with ``owner.from_encoded`` (one per
    leaf, kept for the life of the process: a module fixture may hold the leaf for many tests)."""
    found = _FROM.get(id(owner))
    if found is None:
        import atexit

        model = model_of(owner.from_encoded)
        shut, address = serve({"stub": model})
        leaf = RemoteValue(address, "stub", owner.encoder, buffer_bytes=64 << 20)

        def close() -> None:
            leaf.close()
            shut.shutdown()
            shut.server_close()

        atexit.register(close)
        found = _FROM[id(owner)] = (owner, leaf)  # the owner is kept, so its id is not reused
    return found[1]


_STEEP_LEAF: list[RemoteValue] = []


def steep_leaf() -> RemoteValue:
    """A second served leaf that disagrees with `hp_share_leaf` (antisymmetric too): the HP
    share through a steeper curve. For the tests that play two arms with different leaves."""
    if not _STEEP_LEAF:
        import atexit

        from pokeuraou.regulation import load_regulation

        encoder = Encoder(load_regulation("gen9championsvgc2026regmc"))
        share = hp_share_model(encoder)

        def steep(encoded: Encoded) -> np.ndarray:
            s = np.clip(share(encoded), 1e-9, 1 - 1e-9)
            return 1.0 / (1.0 + np.exp(-np.log(s / (1.0 - s)) * 3.0))

        shut, address = serve({"stub": model_of(steep)})
        leaf = RemoteValue(address, "stub", encoder, buffer_bytes=64 << 20)

        def close() -> None:
            leaf.close()
            shut.shutdown()
            shut.server_close()

        atexit.register(close)
        _STEEP_LEAF.append(leaf)
    return _STEEP_LEAF[0]


def hp_share_model(encoder: Encoder) -> Callable[[Encoded], np.ndarray]:
    """`payoff.HP_SHARE` read off an encoded batch."""
    mon_k = {name: k for k, name in enumerate(encoder.mon_names)}

    def score(encoded: Encoded) -> np.ndarray:
        mon = encoded.mon.astype(np.float64)
        present = encoded.mask > 0
        maxhp = np.rint(mon[..., mon_k["maxhp_scaled"]] * HP_SCALE)
        hp = np.rint(mon[..., mon_k["hp_fraction"]] * maxhp)
        alive = present & ~(mon[..., mon_k["fainted"]] > 0)
        live = np.where(alive, hp, 0.0).sum(axis=-1)
        total = np.where(present, maxhp, 0.0).sum(axis=-1)
        remaining = np.where(total > 0, live / np.where(total > 0, total, 1.0), 0.0)
        denominator = remaining[:, 0] + remaining[:, 1]
        return np.where(denominator > 0, remaining[:, 0] / np.where(denominator > 0, denominator, 1.0), 0.5)

    return score


def subset(encoded: Encoded, rows: list[int]) -> Encoded:
    """Rows ``rows`` of an `Encoded`, in that order (what the Python road cut a lent fill with)."""
    return Encoded(
        **{name: getattr(encoded, name)[rows] for name in _ARRAYS},
        unknown_volatiles=dict(encoded.unknown_volatiles),
        decided=None if encoded.decided is None else encoded.decided[rows],
    )
