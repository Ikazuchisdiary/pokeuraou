"""An agent's name carries which Q its leaf ranking read, and the fit is converged.

On 2026-10-01 the M-C table put the production value-mc3e6x2 (254.7) under value-mc2x2
(261.7) although it had beaten it through value-mc3e4x2 (+15.4 and +59.2). Two things were
checked. The arm label `.new` means "this match's candidate arm", not a Q, so q-mc2 was
`.new` in IKA-400 and the plain arm in IKA-402 and the name held two Qs. And the fit
stopped after 500 gradient steps along a chain joined by few games, 100 Elo short. The
second was the cause of the 9 to 13 point gaps; both are pinned here.
"""

from __future__ import annotations

import math

from pokeuraou.provenance import agent_name, q_identity

from ._harness import load_tool

ratings = load_tool("ratings")


def record(fills: list[str], q_models: list[str] | None) -> dict:
    source: dict = {
        "leaves": ["value-m", "value-m"],
        "limits": [12, 12],
        "rankings": ["leaf", "leaf"],
        "rankFills": fills,
    }
    if q_models is not None:
        source["qModels"] = [q_identity(q_models, f) for f in fills]
    return source


def test_q_identity_reads_the_arm_the_label_names() -> None:
    q = ["q-mc0.pt", "new=q-mc2.pt"]
    assert q_identity(q, "q-nocover") == "q-mc0"
    assert q_identity(q, "q-nocover.new") == "q-mc2"
    assert q_identity(["q-mc2.pt"], "q-nocover") == "q-mc2"
    assert q_identity(["q-mc2.pt"], "q-nocover.new") is None
    assert q_identity(None, "q-nocover") is None


def test_the_same_q_is_one_agent_whichever_arm_it_was() -> None:
    """IKA-400's `.new` arm and IKA-402's plain arm both read q-mc2: one agent. The control
    (below) is that the two arms of one match, which read different Qs, are two."""
    ika400_new = record(["q-nocover", "q-nocover.new"], ["q-mc0.pt", "new=q-mc2.pt"])
    ika402_plain = record(["q-nocover", "q-nocover.new"], ["q-mc2.pt", "new=q-mc3.pt"])
    assert agent_name(ika400_new, 1) == agent_name(ika402_plain, 0)
    assert agent_name(ika400_new, 1).endswith("/rankfill:q-nocover(q-mc2)")
    assert agent_name(ika400_new, 0) != agent_name(ika400_new, 1)
    # a null match, both arms the same file, is one agent in both seats
    null = record(["q-nocover", "q-nocover.new"], ["q-mc0.pt", "new=q-mc0.pt"])
    assert agent_name(null, 0) == agent_name(null, 1)


def test_a_record_that_cannot_say_keeps_its_label_and_is_marked() -> None:
    old = record(["q-nocover", "q-nocover.new"], None)
    assert agent_name(old, 1).endswith("/rankfill:q-nocover.new(q?)")
    # no Q at all (refs2): unchanged from before
    refs = record(["refs2", "refs2"], None)
    assert "rankfill" not in agent_name(refs, 0)


def test_the_fit_is_converged_along_a_thin_chain() -> None:
    """anchor -- m2 (1000 games) -- m3 (765 games, 58% for m3, both seats): the m3-m2 difference is
    ln(.58/.42) logit up to a small prior. The 500 gradient steps this replaced left it at
    about a fifth of that, which is how the production model sat below its predecessor."""
    games = [
        ("m2", "hp", 0.80 * 500, 500),
        ("hp", "m2", 0.20 * 500, 500),
        ("m3", "m2", 0.58 * 400, 400),
        ("m2", "m3", 0.42 * 365, 365),
        # the bulk of a real corpus: many games elsewhere, which sets the old step size
        ("a", "hp", 0.60 * 20000, 20000),
        ("hp", "a", 0.40 * 20000, 20000),
    ]
    rating, _seat, _err = ratings.fit(games, anchor="hp")
    assert abs((rating["m3"] - rating["m2"]) - math.log(0.58 / 0.42)) < 0.01
    assert abs(rating["m2"] - math.log(0.8 / 0.2)) < 0.01
    assert ratings.fit.steps < 100
