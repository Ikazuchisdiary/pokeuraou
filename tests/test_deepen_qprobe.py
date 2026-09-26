"""IKA-322: the root's double oracle with its probe narrowed by a Q (the label's ``q<k>``).

What these tests hold:

- **the label**: ``m1200sq3h`` reads as ``sall`` (every legal action a candidate) with the
  probe narrowed to a Q's three best a side; ``q<k>`` goes wherever a width does;
- **the short list is the Q's top k** against the root's current equilibrium, read as the
  real cells are read (rows against the column strategy, per completion's reply on a
  Bayesian root; columns by what they save, the ``max 0`` per completion);
- **the oracle still ends at the whole game's equilibrium**, with a Q that knows the game
  and with one that is noise: when the short list gains nothing every outside action is
  probed once, so nothing that gains is left outside;
- **a perfect Q probes fewer cells** than the unnarrowed oracle, and the real cells
  decide: a noise Q reaches the same value;
- **the null control**: a short list as long as the outside is the unnarrowed oracle --
  the same actions join in the same order, the same cells are probed;
- **the Q is asked once per node** (one per completion on a Bayesian root) and counted,
  never charged to a budget of cells; `Cost` prices it and the probed cells apart;
- **in a game** ``q<k>`` with a k past every outside list plays the ``all`` game byte for
  byte but for the label and the counts, and a real ``q3`` game is reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import port, qrank
from pokeuraou.equilibrium import solve, solve_bayesian
from pokeuraou.position import Position

from .test_deepen import _Act, _hidden_game, _payload, setup  # noqa: F401


def test_the_q_label_parses() -> None:
    spec = deepen_mod.deepen_spec("m1200sq3h")
    assert spec == deepen_mod.DeepenSpec(
        "mixed", 1200, deepen_mod.ALL_ACTIONS, swap=True, hidden=True, q_probe=3
    )
    assert deepen_mod.deepen_spec("m100oq2") == deepen_mod.DeepenSpec(
        "mixed", 100, deepen_mod.ALL_ACTIONS, swap=False, hidden=False, q_probe=2
    )
    assert deepen_mod.deepen_spec("b50sq1h").reading == "breadth"
    # The labels before it read as they did: no Q.
    for label in ("m1200sallh", "m100sall", "m60s6", "none", "m60h"):
        assert deepen_mod.deepen_spec(label).q_probe is None
    for bad in ("m60q3", "m60sq0", "r60sq3", "m60sq", "m60sq3x", "m60sqall"):
        with pytest.raises(ValueError):
            deepen_mod.deepen_spec(bad)


# --------------------------------------------------------------------------------------
# Known matrices (no port): the Bayesian oracle, with its fill and its Q injected


def _belief_run(full, w, rows, cols, q=None, k=None, swap=False):  # noqa: ANN001, ANN202
    """The Bayesian oracle run to the end; cells are filled from `full` and counted."""
    names_r = [_Act(f"r{i}") for i in range(full[0].shape[0])]
    names_c = [_Act(f"c{j}") for j in range(full[0].shape[1])]
    prices = [m[np.ix_(rows, cols)] for m in full]
    root = deepen_mod._BeliefRoot(
        0, [names_r[i] for i in rows], [names_c[j] for j in cols], [None] * len(full), w,
        prices, solve_bayesian(prices, w),
    )

    def fill(r, c):  # noqa: ANN001, ANN202
        ri = [int(a.name[1:]) for a in r]
        ci = [int(b.name[1:]) for b in c]
        return [m[np.ix_(ri, ci)] for m in full]

    asked = []

    def ask_q(own_c, other_c):  # noqa: ANN001, ANN202
        asked.append((len(own_c), len(other_c)))
        ri = [int(a.name[1:]) for a in own_c]
        ci = [int(b.name[1:]) for b in other_c]
        return [m[np.ix_(ri, ci)] for m in q]

    oracle = deepen_mod._BeliefOracle(
        root, (names_r, names_c), fill, swap=swap, q_probe=k,
        ask_q=None if q is None else ask_q,
    )
    meter = deepen_mod._Meter(None)
    added = []
    for _ in range(1_000):
        before = (len(root.own), len(root.other))
        if not oracle.step(meter):
            break
        added.append("row" if len(root.own) > before[0] else "col")
    else:
        raise AssertionError("the oracle never stopped")
    return root, oracle, meter, added, asked


def _is_whole_equilibrium(root, full, w) -> None:  # noqa: ANN001
    k = len(full)
    want = float(solve_bayesian(full, w).value)
    assert root.equilibrium.value == pytest.approx(want, abs=1e-6)
    x = np.zeros(full[0].shape[0])
    for n, a in enumerate(root.own):
        x[int(a.name[1:])] = root.equilibrium.row_strategy[n]
    ys = []
    for y_k in root.equilibrium.col_strategies:
        y = np.zeros(full[0].shape[1])
        for n, b in enumerate(root.other):
            y[int(b.name[1:])] = y_k[n]
        ys.append(y)
    guarantee = sum(w[q] * (x @ full[q]).min() for q in range(k))
    best_row = max(sum(w[q] * (full[q] @ ys[q]) for q in range(k)))
    assert guarantee >= want - 1e-6 and best_row <= want + 1e-6


@pytest.mark.parametrize("swap", [False, True])
def test_the_narrowed_oracle_reaches_the_whole_game_with_any_q(swap) -> None:  # noqa: ANN001
    """A Q that is the game and a Q that is noise both end at the whole Bayesian game's
    equilibrium: the real cells decide, and an empty short list probes everything.

    A Q that is the game ranks each side's truly best first, so every step adds what the
    unnarrowed oracle adds, in the same order, and asks a subset of its cells (only the
    last step, the proof, asks every outside action). The count of cells filled need
    not be smaller: a probe fills a rectangle whole, known cells again."""
    rng = np.random.default_rng(322)
    probed = {"all": 0, "perfect": 0}
    fell_back = 0
    for _ in range(40):
        k = int(rng.integers(1, 4))
        full = [rng.random((12, 11)) for _ in range(k)]
        w = rng.random(k) + 0.1
        w = w / w.sum()
        noise = [rng.random((12, 11)) for _ in range(k)]
        base = _belief_run(full, w, [0, 1], [0], swap=swap)
        _is_whole_equilibrium(base[0], full, w)
        probed["all"] += len(base[1].known)
        for name, q in (("perfect", full), ("noise", noise)):
            root, oracle, meter, added, asked = _belief_run(
                full, w, [0, 1], [0], q=q, k=2, swap=swap
            )
            _is_whole_equilibrium(root, full, w)
            # Asked once, for every completion, over both whole candidate lists.
            assert asked == [(12, 11)] and meter.qs == k
            if name == "perfect":
                assert added == base[3]
                assert [a.name for a in root.own] == [a.name for a in base[0].own]
                assert [b.name for b in root.other] == [b.name for b in base[0].other]
                assert set(oracle.known) <= set(base[1].known)
                probed["perfect"] += len(oracle.known)
            else:
                fell_back += oracle.fallbacks
    assert probed["perfect"] < probed["all"], probed
    assert fell_back > 0, "the noise Q's short list never came up empty"


def test_a_short_list_as_long_as_the_outside_is_the_plain_oracle() -> None:
    """The null control: k past every outside list asks the same cells in the same
    order and adds the same actions as the unnarrowed oracle."""
    rng = np.random.default_rng(3221)
    compared = 0
    for _ in range(30):
        k = int(rng.integers(1, 4))
        full = [rng.random((8, 7)) for _ in range(k)]
        w = rng.random(k) + 0.1
        w = w / w.sum()
        plain = _belief_run(full, w, [0, 1], [0], swap=True)
        wide = _belief_run(full, w, [0, 1], [0], q=[rng.random((8, 7)) for _ in range(k)],
                           k=100, swap=True)
        assert wide[3] == plain[3]
        assert [a.name for a in wide[0].own] == [a.name for a in plain[0].own]
        assert [b.name for b in wide[0].other] == [b.name for b in plain[0].other]
        assert wide[1].probed == plain[1].probed and wide[2].cells == plain[2].cells
        assert wide[1].fallbacks == 0
        compared += len(plain[3])
    assert compared > 30


def test_the_short_list_is_the_qs_best_against_the_bayesian_answer() -> None:
    rng = np.random.default_rng(3222)
    for _ in range(30):
        k = int(rng.integers(2, 5))
        full = [rng.random((9, 8)) for _ in range(k)]
        q = [rng.random((9, 8)) for _ in range(k)]
        w = rng.random(k) + 0.1
        w = w / w.sum()
        rows, cols = [0, 1, 2], [0, 1]
        names_r = [_Act(f"r{i}") for i in range(9)]
        names_c = [_Act(f"c{j}") for j in range(8)]
        prices = [m[np.ix_(rows, cols)] for m in full]
        eq = solve_bayesian(prices, w)
        root = deepen_mod._BeliefRoot(
            0, [names_r[i] for i in rows], [names_c[j] for j in cols], [None] * k, w,
            prices, eq,
        )
        oracle = deepen_mod._BeliefOracle(
            root, (names_r, names_c), lambda r, c: [], q_probe=2,
            ask_q=lambda o, t, q=q: [m.copy() for m in q],
        )
        short = oracle._shortlist(deepen_mod._Meter(None))
        x = eq.row_strategy
        out_r = [i for i in range(9) if i not in rows]
        out_c = [j for j in range(8) if j not in cols]
        row_gain = {
            i: sum(root.w[n] * float(q[n][i, cols] @ eq.col_strategies[n]) for n in range(k))
            for i in out_r
        }
        col_gain = {
            j: sum(
                root.w[n] * max(
                    0.0,
                    float(x @ q[n][np.ix_(rows, cols)] @ eq.col_strategies[n])
                    - float(x @ q[n][rows, j]),
                )
                for n in range(k)
            )
            for j in out_c
        }
        want_r = sorted(sorted(out_r, key=lambda i: -row_gain[i])[:2])
        want_c = sorted(sorted(out_c, key=lambda j: -col_gain[j])[:2])
        assert sorted(int(a.name[1:]) for a in short[0]) == want_r
        assert sorted(int(b.name[1:]) for b in short[1]) == want_c


# --------------------------------------------------------------------------------------
# The open root's oracle: the port faked by a matrix, the process's Q installed


class _MatrixQ:
    """A Q that reads a matrix by the actions' names (side 0's value)."""

    def __init__(self, table: np.ndarray) -> None:
        self.table = table
        self.calls = 0

    def matrix(self, reg, pos, pools):  # noqa: ANN001, ANN201
        self.calls += 1
        ri = [int(a.name[1:]) for a in pools[0]]
        ci = [int(b.name[1:]) for b in pools[1]]
        return self.table[np.ix_(ri, ci)]


def _open_run(full, q, k, monkeypatch, swap=False):  # noqa: ANN001, ANN202
    names_r = [_Act(f"r{i}") for i in range(full.shape[0])]
    names_c = [_Act(f"c{j}") for j in range(full.shape[1])]

    def payoffs(reg, pos, rows, cols, evaluators, budget, cells):  # noqa: ANN001, ANN202
        ri = [int(a.name[1:]) for a in rows]
        ci = [int(b.name[1:]) for b in cols]
        return [full[np.ix_(ri, ci)]], set(), True

    def payoff(reg, pos, rows, cols, evaluate, budget):  # noqa: ANN001, ANN202
        ri = [int(a.name[1:]) for a in rows]
        ci = [int(b.name[1:]) for b in cols]
        return full[np.ix_(ri, ci)], set()

    monkeypatch.setattr(port, "batched_payoffs", payoffs)
    monkeypatch.setattr(port, "batched_payoff", payoff)
    model = None if q is None else _MatrixQ(q)
    if model is not None:
        monkeypatch.setattr(qrank, "_INSTALLED", [model])
    rows, cols = [0, 1], [0]
    payoff0 = full[np.ix_(rows, cols)]
    root = deepen_mod._Node(
        pos=None, rows=[names_r[i] for i in rows], cols=[names_c[j] for j in cols],
        payoff=payoff0.copy(), equilibrium=solve(payoff0), level=0,
    )
    oracle = deepen_mod._Oracle(root, (names_r, names_c), swap=swap, q_probe=k)
    meter = deepen_mod._Meter(None)
    for _ in range(1_000):
        if not oracle.step(None, None, None, meter, set()):
            break
    else:
        raise AssertionError("the oracle never stopped")
    return root, oracle, meter, model


def test_the_open_oracle_narrowed_reaches_the_whole_game(monkeypatch) -> None:  # noqa: ANN001
    rng = np.random.default_rng(3223)
    probed = {"all": 0, "perfect": 0}
    for _ in range(30):
        full = rng.random((12, 11))
        want = solve(full).value
        base = _open_run(full, None, None, monkeypatch)
        assert base[0].equilibrium.value == pytest.approx(want, abs=1e-7)
        probed["all"] += base[1].probed
        for q in (full, rng.random((12, 11))):
            root, oracle, meter, model = _open_run(full, q, 2, monkeypatch, swap=True)
            assert root.equilibrium.value == pytest.approx(want, abs=1e-7)
            assert model.calls == 1 and meter.qs == 1
            if q is full:
                probed["perfect"] += oracle.probed
        # Probed cells are the probe's; the budget in cells does not see the Q.
        assert meter.probed == oracle.probed and meter.probed <= meter.cells
        assert meter.spent == meter.refines + meter.cells
    assert probed["perfect"] < probed["all"], probed


def test_the_full_probe_is_asked_once_per_menu() -> None:
    """After a full probe finds nothing, the next steps ask the short list alone until an
    action joins: the deepening's steps do not probe everything again."""
    rng = np.random.default_rng(3224)
    checked = 0
    for _ in range(20):
        k = int(rng.integers(1, 4))
        full = [rng.random((10, 9)) for _ in range(k)]
        w = rng.random(k) + 0.1
        w = w / w.sum()
        noise = [rng.random((10, 9)) for _ in range(k)]
        root, oracle, meter, _added, _asked = _belief_run(full, w, [0, 1], [0], q=noise, k=2)
        if not oracle.proved:
            continue  # it ended with the short list covering the whole outside
        checked += 1
        fallbacks, probed = oracle.fallbacks, meter.probed
        for _step in range(3):
            assert not oracle.step(meter)
        assert oracle.fallbacks == fallbacks and oracle.proved
        # Only the short list was asked, against a support that did not move: nothing new.
        assert meter.probed == probed
    assert checked >= 10


def test_a_cost_prices_the_q_and_the_probe_apart() -> None:
    plain = deepen_mod.Cost(fill=5.0, refine=1.5, cell=0.1)
    # Without the two new prices a probed cell is any cell and a Q is free: as before.
    assert plain.ms(3, 2, 100, probed=40, qs=7) == plain.ms(3, 2, 100)
    priced = deepen_mod.Cost(fill=5.0, refine=1.5, cell=0.1, probe=0.05, q=5.2)
    assert priced.ms(3, 2, 100, probed=40, qs=7) == pytest.approx(
        3 * 5.0 + 2 * 1.5 + 60 * 0.1 + 40 * 0.05 + 7 * 5.2
    )
    meter = deepen_mod._Meter(priced)
    meter.filled(40, probe=True)
    meter.filled(60)
    meter.asked_q(2)
    assert meter.probed == 40 and meter.cells == 100 and meter.qs == 2
    assert meter.spent == pytest.approx(priced.ms(2, 0, 100, 40, 2) / 0.1)


# --------------------------------------------------------------------------------------
# In a game


class _HashQ:
    """A deterministic stand-in Q: a matrix from the actions' choices and the position."""

    def __init__(self) -> None:
        self.calls = 0

    def matrix(self, reg, pos, pools):  # noqa: ANN001, ANN201
        import hashlib

        self.calls += 1
        seed = hashlib.sha256(
            repr(pos.to_json()).encode() + b"|".join(a.to_choice().encode() for a in pools[0])
            + b"#" + b"|".join(b.to_choice().encode() for b in pools[1])
        ).digest()
        rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
        return rng.random((len(pools[0]), len(pools[1])))


def _strip_q(payload: dict) -> dict:
    """The payload without the Q's counts (a copy: the record shares its dicts)."""
    import copy

    payload = copy.deepcopy(payload)
    for d in payload["decisions"]:
        for side in d.get("deepened") or []:
            if side:
                side.pop("q", None)
                side.pop("qfull", None)
    return payload


def test_in_a_game_a_list_past_the_outside_plays_the_all_game(setup, monkeypatch) -> None:  # noqa: ANN001, F811
    model = _HashQ()
    monkeypatch.setattr(qrank, "_INSTALLED", [model])
    plain = _payload(_hidden_game(setup, "m60sallh"))
    wide = _payload(_hidden_game(setup, "m60sq9999h"))
    assert model.calls > 0
    asked = [side["q"] for d in wide["decisions"] for side in (d.get("deepened") or [])
             if side and "q" in side]
    assert asked and sum(asked) > 0
    assert all(side.get("qfull") == 0 for d in wide["decisions"]
               for side in (d.get("deepened") or []) if side and "qfull" in side)
    assert wide["deepen"] == ["m60sq9999h"] * 2
    wide["deepen"] = plain["deepen"]
    assert _strip_q(wide) == plain


def test_in_a_game_q3_narrows_and_is_reproducible(setup, monkeypatch) -> None:  # noqa: ANN001, F811
    monkeypatch.setattr(qrank, "_INSTALLED", [_HashQ()])
    game = _hidden_game(setup, "m60sq1h")
    again = _hidden_game(setup, "m60sq1h")
    assert _payload(game) == _payload(again)
    plain = _hidden_game(setup, "m60sallh")
    # The short list changed what was probed, so the game moved (the positive control
    # that the label reached the oracle).
    assert _strip_q(_payload(game))["decisions"] != _payload(plain)["decisions"]
    hidden = short = 0
    for d in game.decisions:
        if d.deepened is None:
            continue
        pos = Position.from_json(d.position)
        shown = all(len(d.shown[i]) == len(pos.sides[i].pokemon) for i in (0, 1))
        for side in (0, 1):
            report = d.deepened[side]
            if report is None:
                continue
            classes = report.get("classes", 0)
            # One Q per completion on a Bayesian root, one for an open root.
            assert report["q"] == (classes if classes else 1)
            hidden += bool(classes)
            assert not shown or not classes
            # More widenings than full probes: some came off the short list alone.
            short += report["widened"] > report["qfull"]
    assert hidden > 0 and short > 0
