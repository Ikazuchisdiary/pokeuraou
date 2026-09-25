"""The vocabulary order is committed and append-only, and a model survives it growing (IKA-82).

Until 9/23 every id's integer came from sorting the dump's ids, so a re-dump that added one
move renumbered every move after it and the shipping net stopped loading (IKA-136's
`meteorassault`, 240 moves moved). The order now lives in `configs/vocab/`, both encoders
read it, and `value.load_model` loads a model whose vocabulary is a prefix of the current one
by growing its embedding tables.

What is pinned here:

- **nothing that existed moved**: the committed order, cut back to the sizes of 9/23, has
  the fingerprints every model up to then was saved with -- which stays true for as long
  as the order is only appended to;
- **an inserted id goes at the end** of a scratch dump's vocabulary, whatever its place in
  the dump, and the sorted numbering this replaced is shown to fail the same check;
- **the null control**: a model grown onto the longer vocabulary scores every position
  without the new id bit for bit as before -- with a positive control, a shifted index,
  showing the comparison can fail;
- **refusal**: a vocabulary that is not a prefix, and a dump id the order does not list;
- **the port reads the same order** (skipped without a Rust binary).
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.encode import (
    VOCAB_TABLES,
    Encoder,
    Vocabulary,
    build_vocabulary,
    read_vocab_order,
    vocab_order_path,
)
from pokeuraou.regulation import load_regulation, regulation_dir, repo_root
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

# `pokeuraou.value` imports torch at module scope; without the learn group this file
# raised at collection (the suite job's first CI run, IKA-51).
torch = pytest.importorskip("torch", reason="the dataset needs the optional learn group")

from pokeuraou.value import ValueConfig, build, load_model, save_model  # noqa: E402

FORMAT = "gen9championsvgc2026regmb"
NEW_MOVE = "ika82testmove"

#: Fingerprints and table sizes (index 0 included) on 9/23, before the order was committed.
#: `value-gen11L` stores the M-B one.
#:
#: M-C's 9/23 fingerprint was 9616b72545058306 at these sizes. On 9/24 its order was
#: rewritten to begin with M-B's (`vocab_order.py --extend`, IKA-82), with no M-C model yet
#: trained, and the pin below is the rewritten order at its sizes that day -- from which it
#: is append-only like any other.
PINNED = {
    "gen9championsvgc2026regmb": (
        "848731f359e4b3a6",
        {"species": 358, "ability": 317, "item": 149, "move": 515},
    ),
    "gen9championsvgc2026regmc": (
        "c2557340f3ed460f",
        {"species": 393, "ability": 317, "item": 167, "move": 516},
    ),
}
MC = "gen9championsvgc2026regmc"


def _write_order(path: Path, order: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({"formatId": FORMAT, **order}).encode("utf-8"))


def _scratch(tmp_path: Path, *, order: str = "append", drop: str | None = None) -> str:
    """A dump that gained NEW_MOVE (listed where an id-sorted re-dump puts it) and an order.

    `order` is "append" (what `tools/vocab_order.py --append` writes), "sorted" (the
    numbering this replaced) or "stale" (the committed order, not yet appended to).
    """
    dump = json.loads((regulation_dir() / f"{FORMAT}.json").read_text(encoding="utf-8"))
    template = next(m for m in dump["moves"] if m["id"] == "closecombat")
    dump["moves"].append({**copy.deepcopy(template), "id": NEW_MOVE, "name": "IKA-82 Test"})
    dump["moves"].sort(key=lambda m: m["id"])
    if drop:
        dump["moves"] = [m for m in dump["moves"] if m["id"] != drop]
    path = tmp_path / "configs" / "regulations" / f"{FORMAT}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(json.dumps(dump).encode("utf-8"))

    committed = read_vocab_order(vocab_order_path(regulation_dir() / f"{FORMAT}.json"))
    assert committed is not None
    grown = {t: list(committed[t]) for t in VOCAB_TABLES}
    if order == "append":
        grown["moves"].append(NEW_MOVE)
    elif order == "sorted":
        grown["moves"] = sorted(grown["moves"] + [NEW_MOVE])
    _write_order(vocab_order_path(path), grown)
    return str(path)


@pytest.mark.parametrize("format_id", sorted(PINNED))
def test_the_committed_order_cut_back_to_9_23_is_the_vocabulary_of_9_23(format_id: str) -> None:
    fingerprint, sizes = PINNED[format_id]
    vocab = build_vocabulary(load_regulation(format_id))
    assert vocab.prefix(sizes).fingerprint() == fingerprint
    assert all(vocab.sizes[k] >= sizes[k] for k in sizes)


@pytest.mark.parametrize("format_id", sorted(PINNED))
def test_every_dump_id_is_in_the_committed_order(format_id: str) -> None:
    # `build_vocabulary` would refuse otherwise; this names the check `--check` makes.
    path = vocab_order_path(regulation_dir() / f"{format_id}.json")
    assert path.parent == repo_root() / "configs" / "vocab"
    order = read_vocab_order(path, format_id)
    assert order is not None
    reg = load_regulation(format_id)
    for table, ids in (("species", reg.species), ("abilities", reg.abilities),
                       ("items", reg.items), ("moves", reg.moves)):
        assert set(ids) <= set(order[table]), table


def test_an_inserted_id_goes_at_the_end_and_nothing_moves(tmp_path: Path) -> None:
    before = build_vocabulary(load_regulation(FORMAT))
    after = build_vocabulary(load_regulation(FORMAT, _scratch(tmp_path)))
    assert after.moves[NEW_MOVE] == len(before.moves) + 1
    assert {k: after.moves[k] for k in before.moves} == before.moves
    assert after.species == before.species and after.items == before.items
    assert after.prefix(before.sizes).fingerprint() == before.fingerprint()


def test_the_sorted_numbering_it_replaced_moves_existing_ids(tmp_path: Path) -> None:
    # Fail-before: numbering by sort -- the old `build_vocabulary` -- is what an order
    # file in sorted order reproduces, and it fails the check the test above passes.
    before = build_vocabulary(load_regulation(FORMAT))
    after = build_vocabulary(load_regulation(FORMAT, _scratch(tmp_path, order="sorted")))
    moved = [k for k in before.moves if after.moves[k] != before.moves[k]]
    assert len(moved) > 0
    assert after.prefix(before.sizes).fingerprint() != before.fingerprint()


def test_a_dump_id_the_order_does_not_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vocab_order.py --append"):
        build_vocabulary(load_regulation(FORMAT, _scratch(tmp_path, order="stale")))


def test_a_retired_id_keeps_its_slot(tmp_path: Path) -> None:
    before = build_vocabulary(load_regulation(FORMAT))
    after = build_vocabulary(load_regulation(FORMAT, _scratch(tmp_path, drop="closecombat")))
    assert after.moves["closecombat"] == before.moves["closecombat"]
    assert after.moves[NEW_MOVE] == len(before.moves) + 1


# -- the model side --------------------------------------------------------------------


def _positions(reg) -> list:  # noqa: ANN001
    roster = load_roster("rizabanadohido")
    n = reg.meta.picked_team_size
    sets = roster.sets
    return [
        position_from_sets(reg, sets[:n], sets[len(sets) - n :]),
        position_from_sets(reg, sets[len(sets) - n :], sets[:n]),
        position_from_sets(reg, sets[1 : n + 1], sets[:n]),
    ]


def _with_new_move(position):  # noqa: ANN001, ANN202
    from pokeuraou.position import Position

    doc = position.to_json()
    doc["sides"][0]["pokemon"][0]["moves"][0]["id"] = NEW_MOVE
    return Position.from_json(doc)


def _scores(net, encoder: Encoder, positions: list) -> np.ndarray:  # noqa: ANN001
    e = encoder.encode_positions(positions)
    batch = {k: torch.as_tensor(getattr(e, k), dtype=torch.long)
             for k in ("species", "ability", "item", "moves")}
    batch |= {k: torch.as_tensor(getattr(e, k)) for k in ("mon", "mask", "side", "field")}
    with torch.no_grad():
        return net(batch).numpy()


@pytest.fixture()
def saved(tmp_path: Path):  # noqa: ANN201
    """A seeded, untrained net saved under today's vocabulary: every row is distinct."""
    encoder = Encoder(load_regulation(FORMAT))
    torch.manual_seed(82)
    net = build(encoder, ValueConfig()).eval()
    path = tmp_path / "value.pt"
    save_model(path, net, net.state_dict(), encoder.vocab, ValueConfig(), meta={"m": 1})
    return path, encoder, net


@pytest.mark.parametrize("legacy", [False, True])
def test_a_model_loads_onto_a_grown_vocabulary_and_scores_old_positions_bit_for_bit(
    saved, tmp_path: Path, legacy: bool  # noqa: ANN001
) -> None:
    path, old_encoder, old_net = saved
    if legacy:
        # Saved before `vocab_sizes` existed, like value-gen11L: sizes come off the shapes.
        blob = torch.load(path, weights_only=False)
        del blob["vocab_sizes"]
        torch.save(blob, path)
    encoder = Encoder(load_regulation(FORMAT, _scratch(tmp_path)))
    net, meta = load_model(path, encoder)
    assert meta["vocab_grown_from"] == {"move": old_encoder.vocab.sizes["move"]}
    assert meta["m"] == 1
    new_row = encoder.vocab.moves[NEW_MOVE]
    assert net.move.weight.shape[0] == new_row + 1
    assert torch.equal(net.move.weight[new_row], torch.zeros_like(net.move.weight[new_row]))

    positions = _positions(encoder.reg)
    before = _scores(old_net, old_encoder, _positions(old_encoder.reg))
    after = _scores(net, encoder, positions)
    assert before.tobytes() == after.tobytes()

    # The new id is encoded at its appended index and is read (it is not index 0).
    carrier = _with_new_move(positions[0])
    assert (encoder.encode_positions([carrier]).moves == new_row).any()


def test_the_bit_comparison_can_fail(saved) -> None:  # noqa: ANN001
    # Positive control for the test above: one existing index moved changes the scores.
    path, encoder, net = saved
    positions = _positions(encoder.reg)
    base = _scores(net, encoder, positions)
    first = positions[0].sides[0].pokemon[0].moves[0].id
    moves = dict(encoder.vocab.moves)
    other = next(k for k in moves if k != first)
    moves[first], moves[other] = moves[other], moves[first]
    shifted = Encoder(encoder.reg)
    object.__setattr__(shifted, "vocab", Vocabulary(
        format_id=encoder.vocab.format_id, species=encoder.vocab.species,
        abilities=encoder.vocab.abilities, items=encoder.vocab.items, moves=moves,
        types=encoder.vocab.types,
    ))
    assert base.tobytes() != _scores(net, shifted, positions).tobytes()


def test_a_vocabulary_that_is_not_a_prefix_is_refused(saved, tmp_path: Path) -> None:  # noqa: ANN001
    # Grown by one like the append, but numbered by sort: the tables are larger, yet an
    # existing id changed its integer, so it is refused rather than grown.
    path, _encoder, _net = saved
    encoder = Encoder(load_regulation(FORMAT, _scratch(tmp_path, order="sorted")))
    with pytest.raises(ValueError, match="not a prefix"):
        load_model(path, encoder)


def test_a_vocabulary_smaller_than_the_model_is_refused(saved, tmp_path: Path) -> None:  # noqa: ANN001
    path, encoder, net = saved
    blob = torch.load(path, weights_only=False)
    blob["vocab_sizes"] = {**blob["vocab_sizes"], "move": blob["vocab_sizes"]["move"] + 1}
    blob["vocab_fingerprint"] = "0" * 16
    torch.save(blob, path)
    with pytest.raises(ValueError, match="larger"):
        load_model(path, encoder)


# -- M-C extends M-B -------------------------------------------------------------------


def _vocab_order_tool():  # noqa: ANN202
    import importlib.util

    path = repo_root() / "tools" / "vocab_order.py"
    spec = importlib.util.spec_from_file_location("vocab_order_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mc_numbers_every_mb_id_as_mb_does() -> None:
    mb = build_vocabulary(load_regulation(FORMAT))
    mc = build_vocabulary(load_regulation(MC))
    for table in ("species", "abilities", "items", "moves"):
        assert {k: getattr(mc, table)[k] for k in getattr(mb, table)} == getattr(mb, table)
    # Cut back to M-B's sizes and named M-B, M-C is M-B: today's, and value-gen11L's of 9/23.
    assert mc.prefix(mb.sizes, FORMAT).fingerprint() == mb.fingerprint()
    fingerprint, sizes = PINNED[FORMAT]
    assert mc.prefix(sizes, FORMAT).fingerprint() == fingerprint
    # Under its own name it is not M-B's vocabulary: the name is part of the fingerprint.
    assert mc.prefix(mb.sizes).fingerprint() != mb.fingerprint()


def test_the_extension_check_passes_and_can_fail() -> None:
    tool = _vocab_order_tool()
    path = vocab_order_path(regulation_dir() / f"{MC}.json")
    order = read_vocab_order(path, MC)
    extends = json.loads(path.read_text(encoding="utf-8"))["extends"]
    assert extends["formatId"] == FORMAT
    assert tool.extension_problems(MC, order, extends) == []
    # Positive control: two M-B species swapped in M-C's list is reported, at the first.
    broken = {t: list(v) for t, v in order.items()}
    broken["species"][3], broken["species"][9] = broken["species"][9], broken["species"][3]
    problems = tool.extension_problems(MC, broken, extends)
    assert len(problems) == 1 and "species" in problems[0] and "index 4" in problems[0]


def _mc_only_carrier(position):  # noqa: ANN001, ANN202
    from pokeuraou.position import Position

    doc = position.to_json()
    doc["sides"][0]["pokemon"][0]["species"] = "salamence"
    doc["sides"][0]["pokemon"][0]["item"] = "salamencite"
    return Position.from_json(doc)


def test_an_mb_model_loads_onto_mc_and_scores_mb_positions_bit_for_bit(saved) -> None:  # noqa: ANN001
    path, mb_encoder, mb_net = saved
    encoder = Encoder(load_regulation(MC))
    net, meta = load_model(path, encoder)
    assert meta["vocab_extended_from"] == FORMAT
    have, want = mb_encoder.vocab.sizes, encoder.vocab.sizes
    assert meta["vocab_grown_from"] == {k: have[k] for k in want if have[k] != want[k]}
    assert set(meta["vocab_grown_from"]) == {"species", "item"}
    for table in ("species", "item"):
        rows = getattr(net, table).weight[have[table]:]
        assert rows.shape[0] == want[table] - have[table] and torch.count_nonzero(rows) == 0

    positions = _positions(mb_encoder.reg)
    before = _scores(mb_net, mb_encoder, positions)
    assert before.tobytes() == _scores(net, encoder, positions).tobytes()

    # Positive control: an M-C-only species and item are read from the new rows. Writing
    # those rows moves the carrier's score and leaves every M-B position where it was.
    carrier = [_mc_only_carrier(positions[0])]
    e = encoder.encode_positions(carrier)
    assert e.species[0, 0, 0] == encoder.vocab.species["salamence"] >= have["species"]
    assert e.item[0, 0, 0] == encoder.vocab.items["salamencite"] >= have["item"]
    zero_rows = _scores(net, encoder, carrier)
    with torch.no_grad():
        net.species.weight[have["species"]:] = 1.0
        net.item.weight[have["item"]:] = 1.0
    assert zero_rows.tobytes() != _scores(net, encoder, carrier).tobytes()
    assert before.tobytes() == _scores(net, encoder, positions).tobytes()


def test_an_mb_model_is_refused_by_mc_in_its_old_sorted_order(saved, tmp_path: Path) -> None:  # noqa: ANN001
    # Fail-before: M-C numbered by sorting its ids -- its order until 9/24 -- is not an
    # extension of M-B, and the load refuses it rather than growing.
    path, _encoder, _net = saved
    src = regulation_dir() / f"{MC}.json"
    dump = tmp_path / "configs" / "regulations" / src.name
    dump.parent.mkdir(parents=True)
    dump.write_bytes(src.read_bytes())
    order = read_vocab_order(vocab_order_path(src), MC)
    vocab_order_path(dump).parent.mkdir(parents=True)
    vocab_order_path(dump).write_bytes(json.dumps(
        {"formatId": MC, **{t: sorted(order[t]) for t in VOCAB_TABLES}}
    ).encode("utf-8"))
    with pytest.raises(ValueError, match="does not begin with"):
        load_model(path, Encoder(load_regulation(MC, str(dump))))


# -- the port --------------------------------------------------------------------------


def _rust_encode(dump: str, fixture: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [str(rustnode.binary_path()), "encode", dump, str(fixture), str(out)],
        capture_output=True, text=True, timeout=60, check=False,
    )


def test_the_port_reads_the_same_order(tmp_path: Path) -> None:
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    dump = _scratch(tmp_path)
    reg = load_regulation(FORMAT, dump)
    encoder = Encoder(reg)
    positions = _positions(reg)
    positions.append(_with_new_move(positions[0]))
    fixture = tmp_path / "turns.json"
    fixture.write_bytes(json.dumps(
        {"format_id": FORMAT, "positions": [p.to_json() for p in positions]}
    ).encode("utf-8"))
    out = tmp_path / "encoded.bin"
    done = _rust_encode(dump, fixture, out)
    assert done.returncode == 0, done.stderr
    raw = out.read_bytes()
    newline = raw.index(b"\n")
    header = json.loads(raw[:newline])
    n, m = header["positions"], header["monsPerSide"]
    count = n * 2 * m * 4
    got = np.frombuffer(raw[newline + 1 :], dtype=np.int32, count=count, offset=3 * n * 2 * m * 4)
    want = encoder.encode_positions(positions).moves
    assert np.array_equal(got.reshape(want.shape), want)
    assert (want[-1] == encoder.vocab.moves[NEW_MOVE]).any()

    # M-C's committed order, which extends M-B's and carries an `extends` record, is read
    # the same by the port: species and item of M-B positions and of an M-C-only carrier.
    mc_dump = str(regulation_dir() / f"{MC}.json")
    mc_encoder = Encoder(load_regulation(MC))
    mc_positions = _positions(reg) + [_mc_only_carrier(positions[0])]
    mc_fixture = tmp_path / "turns-mc.json"
    mc_fixture.write_bytes(json.dumps(
        {"format_id": MC, "positions": [p.to_json() for p in mc_positions]}
    ).encode("utf-8"))
    done = _rust_encode(mc_dump, mc_fixture, tmp_path / "encoded-mc.bin")
    assert done.returncode == 0, done.stderr
    raw = (tmp_path / "encoded-mc.bin").read_bytes()
    newline = raw.index(b"\n")
    n = json.loads(raw[:newline])["positions"]
    want = mc_encoder.encode_positions(mc_positions)
    for k, name in ((0, "species"), (2, "item")):
        got = np.frombuffer(raw[newline + 1 :], dtype=np.int32, count=n * 2 * m,
                            offset=k * n * 2 * m * 4)
        assert np.array_equal(got.reshape(n, 2, m), getattr(want, name)), name
    assert want.species[-1, 0, 0] == mc_encoder.vocab.species["salamence"] > 357

    stale = tmp_path / "stale"
    stale_dump = _scratch(stale, order="stale")
    refused = _rust_encode(stale_dump, fixture, tmp_path / "refused.bin")
    assert refused.returncode != 0
    assert "vocab_order.py --append" in refused.stderr
