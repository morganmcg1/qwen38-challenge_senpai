"""Minimal safetensors reader/writer for E144 head requantization.

Zero dependencies beyond numpy. Reads the header without mapping payload, and
memory-maps individual tensors on demand so a 428 MB or 850 MB head can be
walked tensor by tensor.
"""

import json
import mmap
import struct

import numpy as np

_DTYPES = {
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
_INVERSE_DTYPES = {v.str: k for k, v in _DTYPES.items()}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Widen raw BF16 bit patterns (as uint16) to float32 losslessly."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Narrow float32 to BF16 bit patterns with round-to-nearest-even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((bits + rounding) >> 16).astype(np.uint16)


class SafeTensors:
    """Read-only handle over a .safetensors file."""

    def __init__(self, path):
        self.path = str(path)
        self._file = open(self.path, "rb")
        (header_len,) = struct.unpack("<Q", self._file.read(8))
        header = json.loads(self._file.read(header_len))
        self.metadata = header.pop("__metadata__", {})
        self.header = header
        self._payload_start = 8 + header_len
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def names(self):
        return sorted(self.header)

    def info(self, name):
        entry = self.header[name]
        start, end = entry["data_offsets"]
        return entry["dtype"], tuple(entry["shape"]), end - start

    def raw(self, name) -> np.ndarray:
        """Return the tensor's bytes as its declared dtype, BF16 kept as uint16."""
        entry = self.header[name]
        start, end = entry["data_offsets"]
        dtype = _DTYPES.get(entry["dtype"], np.dtype("<u2"))
        buffer = self._map[self._payload_start + start : self._payload_start + end]
        return np.frombuffer(buffer, dtype=dtype).reshape(entry["shape"])

    def float32(self, name) -> np.ndarray:
        """Return the tensor widened to float32, handling BF16 explicitly."""
        entry = self.header[name]
        if entry["dtype"] == "BF16":
            return bf16_to_f32(self.raw(name))
        return self.raw(name).astype(np.float32)

    def close(self):
        self._map.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def save(path, tensors, metadata=None):
    """Write a .safetensors file, preserving insertion order of `tensors`.

    BF16 tensors are passed in as uint16 bit patterns tagged by wrapping them in
    a (array, "BF16") pair, because numpy has no native bfloat16.
    """
    header = {}
    offset = 0
    payload = []
    for name, value in tensors.items():
        if isinstance(value, tuple):
            array, dtype_tag = value
        else:
            array, dtype_tag = value, _INVERSE_DTYPES[np.dtype(value.dtype).str]
        array = np.ascontiguousarray(array)
        nbytes = array.nbytes
        header[name] = {
            "dtype": dtype_tag,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        payload.append(array)
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    # safetensors requires the payload to start 8-byte aligned.
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for array in payload:
            handle.write(array.tobytes())
