#!/usr/bin/env python3
"""Minimal safetensors reader/writer shared by the E82 head scripts.

Everything here is byte-level on purpose: E82 has to prove *provenance*
claims about published heads (byte-identical tensor payloads, bit-exact BF16
islands), and any library that silently upcasts or reorders would destroy the
evidence. Tensors are returned as raw memoryviews plus dtype/shape so a caller
can hash the exact stored bytes as well as do numeric work.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NP = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U8": np.dtype("<u1"),
    "U32": np.dtype("<u4"),
    "BOOL": np.dtype("?"),
}


@dataclass(frozen=True)
class Entry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


class SafeTensors:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        self.data_start = 8 + header_len
        self.metadata = header.pop("__metadata__", None)
        self.entries = {
            name: Entry(name, v["dtype"], tuple(v["shape"]), v["data_offsets"][0], v["data_offsets"][1])
            for name, v in header.items()
        }
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def names(self) -> list[str]:
        return list(self.entries)

    def raw(self, name: str) -> np.ndarray:
        e = self.entries[name]
        return self._mm[self.data_start + e.begin : self.data_start + e.end]

    def sha256(self, name: str) -> str:
        return hashlib.sha256(self.raw(name).tobytes()).hexdigest()

    def array(self, name: str) -> np.ndarray:
        """Raw stored values. BF16 is returned as uint16 (no upcast)."""
        e = self.entries[name]
        buf = self.raw(name)
        if e.dtype == "BF16":
            return buf.view(np.uint16).reshape(e.shape)
        return buf.view(_NP[e.dtype]).reshape(e.shape)

    def f32(self, name: str) -> np.ndarray:
        e = self.entries[name]
        if e.dtype == "BF16":
            return bf16_to_f32(self.array(name))
        return self.array(name).astype(np.float32)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even BF16, matching MLX/torch cast semantics."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    rounded = u.astype(np.uint64) + np.uint64(0x7FFF) + lsb.astype(np.uint64)
    return (rounded >> np.uint64(16)).astype(np.uint16)


def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray], metadata: dict | None = None) -> int:
    """Write tensors in insertion order. uint16 arrays are stored as BF16."""
    dtype_of = {
        np.dtype("<u2"): "BF16",
        np.dtype("<u4"): "U32",
        np.dtype("<i4"): "I32",
        np.dtype("<f4"): "F32",
    }
    header: dict[str, object] = {}
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        header[name] = {
            "dtype": dtype_of[arr.dtype.newbyteorder("<")],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-(len(blob) + 8)) % 8
    blob += b" " * pad
    path = Path(path)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for name, arr in tensors.items():
            fh.write(np.ascontiguousarray(arr).tobytes())
    return path.stat().st_size


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tree_digest(directory: str | Path) -> tuple[str, int]:
    """The runner's head tree digest: sha256 over "<file sha256>  <rel path>\\n"
    lines, LC_ALL=C sorted, top-level README.md excluded (fetch-declared-head.sh:42-59).
    """
    directory = Path(directory)
    rels = sorted(
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != "README.md"
    )
    total = 0
    lines = []
    for rel in rels:
        p = directory / rel
        total += p.stat().st_size
        lines.append(f"{file_sha256(p)}  {rel}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest(), total
