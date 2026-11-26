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
    tensors = []
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in net.state_dict().items():
        if name.startswith(("aux", "move_props")):
            continue
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        offset = _aligned(offset)
        tensors.append({"name": name, "shape": list(array.shape), "offset": offset})
        chunks.append((offset, array.tobytes()))  # type: ignore[arg-type]
        offset += array.nbytes
    header["tensors"] = tensors
    body = json.dumps(header, sort_keys=True).encode("utf-8")
    # The padded header ends where the data starts, a multiple of 64 from the file's start.
    head_len = _aligned(len(MAGIC) + 8 + len(body) + 1) - len(MAGIC) - 8
    padded = body + b" " * (head_len - len(body) - 1) + b"\n"
    data = bytearray(offset)
    for at, raw in chunks:  # type: ignore[misc]
        data[at : at + len(raw)] = raw
    blob = MAGIC + struct.pack("<Q", len(padded)) + padded + bytes(data)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".part")
    tmp.write_bytes(blob)
    tmp.replace(out)
    return header


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
