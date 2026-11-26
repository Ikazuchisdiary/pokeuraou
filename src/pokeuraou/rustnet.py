"""IKA-461: a trained value net as the one file the Rust inference server reads (`.rnet`).

Training stays in torch; the Rust server (`pokeuraou-damage inference-server`) has its own
forward pass, so the weights cross as raw float32 and a header that says how the net reads
a row. The layout is

    8 bytes   b"RNET0001"
    8 bytes   the header's length in bytes, little-endian
    header    UTF-8 JSON, padded with spaces to a multiple of 64 bytes from the file's start
    data      every tensor as float32, little-endian, each at a multiple of 64 bytes

The header names the net's sizes, the columns of each block it reads, where the move ids
sit among the Pokemon's columns, and every tensor's name, shape and offset (from the start
of the data). `export_net` refuses the nets the Rust forward pass does not implement
(attention, the move-property layers, grouped LayerNorm, a head of more than two hidden
layers) rather than writing a file that answers differently.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"RNET0001"
ALIGN = 64


def _aligned(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def net_header(net: Any, encoder: Any) -> dict[str, Any]:  # noqa: ANN401
    """The header's fields that describe how the net reads a row (no tensors yet)."""
    config = net.config
    if config.attention or config.move_properties or config.width_groups != 1:
        raise ValueError(
            "the Rust forward pass has no attention, move-property layers or grouped LayerNorm"
        )
    if config.head_layers != 2:
        raise ValueError(f"the Rust forward pass has a head of 2 hidden layers, not {config.head_layers}")
    widths = net.in_widths
    k = int(net._move_ids)  # noqa: SLF001 -- the number of move-id columns
    cut = int(widths["mon"])
    if net._bind:  # noqa: SLF001
        ids_end = int(encoder.state_widths["mon"])
        id_lo = ids_end - k
    elif k:
        id_lo = cut - k
    else:
        id_lo = cut
    mon_in = (
        config.species_dim + config.ability_dim + config.item_dim + config.move_dim
        + cut - k + k * config.move_dim
    )
    return {
        "version": 1,
        "dims": {
            "species": config.species_dim, "ability": config.ability_dim,
            "item": config.item_dim, "move": config.move_dim,
            "mon": config.mon_dim, "side": config.side_dim, "head": config.head_dim,
        },
        "mon_cut": cut,
        "side_cut": int(widths["side"]),
        "field_cut": int(widths["field"]),
        "id_lo": int(id_lo),
        "n_ids": k,
        "active_feature": int(net._active_feature),  # noqa: SLF001
        "mon_in": int(mon_in),
        "pooled": int(4 * config.mon_dim + widths["side"]),
        "head_in": int(2 * config.side_dim + widths["field"]),
    }


def export_net(path: str | Path, encoder: Any, out: str | Path) -> dict[str, Any]:  # noqa: ANN401
    """Write `path` (a `.pt` value net) as the `.rnet` file `out`; returns the header."""
    from .value import load_model

    path = Path(path)
    net, _meta = load_model(path, encoder)
    return write_net(net, encoder, out, source=path.name, sha256=file_sha256(path))


def write_net(
    net: Any,  # noqa: ANN401
    encoder: Any,  # noqa: ANN401
    out: str | Path,
    *,
    source: str,
    sha256: str = "",
) -> dict[str, Any]:
    """Write a loaded `ValueNet` as the `.rnet` file `out`; returns the header."""
    header = net_header(net, encoder)
    header["source"] = source
    header["sha256"] = sha256
    header["format_id"] = encoder.vocab.format_id
    header["vocab_fingerprint"] = encoder.vocab.fingerprint()
    named = [
        (name, tensor.detach().cpu().numpy())
        for name, tensor in net.state_dict().items()
        if not name.startswith(("aux", "move_props"))
    ]
    _pack(header, named, out)
    return header


def _pack(header: dict[str, Any], named: list[tuple[str, np.ndarray]], out: str | Path) -> None:
    """Write `header` and the tensors as one `.rnet` file."""
    tensors = []
    chunks: list[tuple[int, bytes]] = []
    offset = 0
    for name, array in named:
        array = np.ascontiguousarray(array, dtype="<f4")
        offset = _aligned(offset)
        tensors.append({"name": name, "shape": list(array.shape), "offset": offset})
        chunks.append((offset, array.tobytes()))
        offset += array.nbytes
    header["tensors"] = tensors
    body = json.dumps(header, sort_keys=True).encode("utf-8")
    head_len = _aligned(len(MAGIC) + 8 + len(body) + 1) - len(MAGIC) - 8
    padded = body + b" " * (head_len - len(body) - 1) + b"\n"
    data = bytearray(offset)
    for at, raw in chunks:
        data[at : at + len(raw)] = raw
    blob = MAGIC + struct.pack("<Q", len(padded)) + padded + bytes(data)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".part")
    tmp.write_bytes(blob)
    tmp.replace(out)


def write_q(
    net: Any,  # noqa: ANN401
    encoder: Any,  # noqa: ANN401
    out: str | Path,
    *,
    source: str,
    sha256: str = "",
) -> dict[str, Any]:
    """Write a loaded Q net (`qhead.build_net`) as the `.rnet` file `out`; returns the header.

    The header is the trunk's (`net_header`) with `kind` "q" and the Q's own sizes. Only the
    net the Rust forward pass implements is written: move properties and attention on.
    """
    from . import qhead

    config = net.config
    if not (config.properties and config.attend):
        raise ValueError("the Rust Q reads a net with move properties and attention (q-mc4 on)")
    header = net_header(net.trunk, encoder)
    header.update({
        "kind": "q",
        "source": source,
        "sha256": sha256,
        "format_id": encoder.vocab.format_id,
        "vocab_fingerprint": encoder.vocab.fingerprint(),
        "q": {
            "act_dim": config.act_dim, "pair_dim": config.pair_dim, "rank": config.rank,
            "features": qhead.FEATURE_WIDTH, "heads": 4, "slots": qhead.SLOTS,
            "properties": bool(config.properties),
        },
    })
    named = [
        (name, tensor.detach().cpu().numpy())
        for name, tensor in net.state_dict().items()
        if not name.startswith(("trunk.head.", "trunk.aux", "trunk.move_props"))
    ]
    _pack(header, named, out)
    return header


def ensure_exported_q(pt: str | Path, out_dir: str | Path) -> Path:
    """The `.rnet` of the Q file `pt` in `out_dir`, written when missing or its source changed."""
    from . import qhead
    from .encode import Encoder
    from .regulation import load_regulation

    pt = Path(pt)
    digest = file_sha256(pt)
    out = Path(out_dir) / f"{pt.stem}.rnet"
    if out.exists():
        try:
            if read_header(out).get("sha256") == digest:
                return out
        except (ValueError, OSError):
            pass
    net = qhead.load_q(pt, "cpu")
    import torch

    blob = torch.load(pt, map_location="cpu", weights_only=False)
    encoder = Encoder(load_regulation(blob["regulation"]))
    write_q(net, encoder, out, source=pt.name, sha256=digest)
    return out


def server_command(
    regulation: str,
    arms: dict[str, list[str | Path]],
    q_arms: dict[str, str | Path],
    out_dir: str | Path,
) -> list[str]:
    """The command of one `pokeuraou-damage inference-server`, the nets exported first.

    `arms` maps an arm's name to its `.pt` files (several: an ensemble, logits averaged);
    `q_arms` maps a Q arm's name to its `.pt`. The `.rnet` files go to `out_dir` (reused when
    their source hash is unchanged). The CUDA 12 DLLs are the server's own search
    (`--dll-dir`, `POKEURAOU_CUDA_DLLS`, `CUDA_PATH_V12*`, the Toolkit's folder); the `.venv`'s
    torch's are passed only when none of those is set.
    """
    import importlib.util
    import os
    import sys

    from .regulation import regulation_dir, repo_root

    # Not `rustnode`: importing it makes a launcher write a timing report of its own.
    exe = Path(
        os.environ.get("POKEURAOU_RUST_NODE_BIN")
        or repo_root() / "rust" / "target" / "release"
        / ("pokeuraou-damage.exe" if sys.platform == "win32" else "pokeuraou-damage")
    )
    if not exe.exists():
        raise SystemExit(f"{exe} is not built: cd rust && cargo build --release")
    command = [str(exe), "inference-server", "--regulation", str(regulation_dir() / f"{regulation}.json")]
    try:
        for name, files in arms.items():
            rnets = [ensure_exported(f, regulation, Path(out_dir) / "rnet") for f in files]
            command += ["--arm", name, *map(str, rnets)]
        for name, file in q_arms.items():
            command += ["--q-arm", name, str(ensure_exported_q(file, Path(out_dir) / "rnet"))]
    except ValueError as problem:
        raise SystemExit(
            f"the Rust inference server cannot take this net: {problem}. "
            "Use the Python server (--python-server)."
        ) from None
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin and not os.environ.get("POKEURAOU_CUDA_DLLS"):
        lib = Path(spec.origin).parent / "lib"
        if lib.is_dir() and not any(k.startswith("CUDA_PATH_V12") for k in os.environ):
            command += ["--dll-dir", str(lib)]
    return command


def read_header(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        if fh.read(8) != MAGIC:
            raise ValueError(f"{path} is not an .rnet file")
        (length,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(length))


def ensure_exported(pt: str | Path, regulation: str, out_dir: str | Path) -> Path:
    """The `.rnet` of `pt` in `out_dir`, written when it is missing or its source changed."""
    from .encode import Encoder
    from .regulation import load_regulation

    pt = Path(pt)
    out = Path(out_dir) / f"{pt.stem}.{regulation}.rnet"
    digest = file_sha256(pt)
    if out.exists():
        try:
            if read_header(out).get("sha256") == digest:
                return out
        except (ValueError, OSError):
            pass
    export_net(pt, Encoder(load_regulation(regulation)), out)
    return out
