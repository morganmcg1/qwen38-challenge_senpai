#!/usr/bin/env python3
"""E111: emit a real (scale, bias, code) blob for one scored linear shape.

The rung-1 harness must run on the checkpoint's own metadata, not on synthetic
values, because the payoff of the 1-byte recoding depends on the true code
distribution (3.12 % of groups carry a non-zero correction).

The packer is fail closed. It re-derives the BF16 bias from (scale, code) with
exactly the integer sequence the Metal arm uses and refuses to write the blob
unless every group reproduces the stored bias bit for bit.

Blob layout, little endian:

    magic   'E111'      uint32
    version 1           uint32
    K                   uint32
    N                   uint32
    group_size          uint32
    n_groups = N*K/64   uint32
    scales   uint16[n_groups]   BF16 bits, row major
    biases   uint16[n_groups]   BF16 bits, row major
    codes    uint8[n_groups]    (corr + 1) << 4 | z
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x31313145  # 'E111' little endian

SHAPES = {
    # name: (ordered checkpoint tensor list template, K, N)
    "lm_head": (["language_model.lm_head"], 5120, 248320),
    "mlp.gate_up": (
        [
            "language_model.model.layers.{L}.mlp.gate_proj",
            "language_model.model.layers.{L}.mlp.up_proj",
        ],
        5120,
        34816,
    ),
    "mlp.down": (["language_model.model.layers.{L}.mlp.down_proj"], 17408, 5120),
    "gdn.in_proj": (
        [
            "language_model.model.layers.{L}.linear_attn.in_proj_qkv",
            "language_model.model.layers.{L}.linear_attn.in_proj_z",
            "language_model.model.layers.{L}.linear_attn.in_proj_a",
            "language_model.model.layers.{L}.linear_attn.in_proj_b",
        ],
        5120,
        16480,
    ),
    "fa.qkv": (
        [
            "language_model.model.layers.{L}.self_attn.q_proj",
            "language_model.model.layers.{L}.self_attn.k_proj",
            "language_model.model.layers.{L}.self_attn.v_proj",
        ],
        5120,
        14336,
    ),
}


def read_header(path: Path):
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def rne_bits(x: np.ndarray) -> np.ndarray:
    u = x.view(np.uint32)
    return (((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))) >> np.uint32(16)).astype(
        np.uint16
    )


def build_codes(scale_bits: np.ndarray, bias_bits: np.ndarray):
    """Return (codes, ok_mask). Raw uint16 correction, so the kernel can add it
    to the rounded bit pattern with one integer add and no sign handling."""
    scale = bf16_to_f32(scale_bits)
    best_code = np.zeros(scale.size, dtype=np.uint8)
    ok = np.zeros(scale.size, dtype=bool)
    for z in range(16):
        cand = rne_bits(np.float32(-z) * scale)
        diff = (bias_bits.astype(np.int32) - cand.astype(np.int32)) & 0xFFFF
        # raw uint16 delta of -1, 0 or +1
        for corr, raw in ((-1, 0xFFFF), (0, 0), (1, 1)):
            hit = (diff == raw) & ~ok
            if not hit.any():
                continue
            best_code[hit] = np.uint8(((corr + 1) << 4) | z)
            ok |= hit
    return best_code, ok


def verify(scale_bits: np.ndarray, bias_bits: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Reproduce the Metal arm's integer sequence exactly."""
    c = codes.astype(np.uint32)
    z = (c & np.uint32(0xF)).astype(np.float32)
    prod = (-z) * bf16_to_f32(scale_bits)
    u = prod.view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    u = u + ((c & np.uint32(0x30)) << np.uint32(12)) - np.uint32(0x10000)
    return (u >> np.uint32(16)).astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=sorted(SHAPES))
    ap.add_argument("--layer", type=int, default=38)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names, K, N = SHAPES[args.shape]
    names = [n.format(L=args.layer) for n in names]
    wdir = Path(args.weights)
    wmap = json.loads((wdir / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {s: read_header(wdir / s) for s in sorted(set(wmap.values()))}

    groups_per_row = K // 64
    scales, biases = [], []
    rows = 0
    for name in names:
        shard = wmap[name + ".scales"]
        hdr, off = headers[shard]
        s_info, b_info = hdr[name + ".scales"], hdr[name + ".biases"]
        assert s_info["shape"][1] == groups_per_row, (name, s_info["shape"], groups_per_row)
        rows += s_info["shape"][0]
        with (wdir / shard).open("rb") as fh:
            for info, sink in ((s_info, scales), (b_info, biases)):
                fh.seek(off + info["data_offsets"][0])
                n = int(np.prod(info["shape"]))
                sink.append(np.frombuffer(fh.read(2 * n), dtype=np.uint16))
    assert rows == N, (rows, N)
    scale_bits = np.concatenate(scales)
    bias_bits = np.concatenate(biases)
    n_groups = scale_bits.size
    assert n_groups == N * groups_per_row

    codes, ok = build_codes(scale_bits, bias_bits)
    if not ok.all():
        print(f"FAIL: {int((~ok).sum())} of {n_groups} groups have no (z, corr) code", file=sys.stderr)
        return 1
    got = verify(scale_bits, bias_bits, codes)
    bad = int(np.count_nonzero(got != bias_bits))
    if bad:
        print(f"FAIL: {bad} of {n_groups} groups do not reproduce the stored bias", file=sys.stderr)
        return 1

    # positive control: one damaged code must break the reconstruction
    damaged = codes.copy()
    damaged[0] ^= np.uint8(1)
    ctrl = verify(scale_bits, bias_bits, damaged)
    if ctrl[0] == bias_bits[0]:
        print("FAIL: positive control did not fire", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(struct.pack("<IIIIII", MAGIC, 1, K, N, 64, n_groups))
        fh.write(scale_bits.tobytes())
        fh.write(bias_bits.tobytes())
        fh.write(codes.tobytes())

    zs = codes & 0x0F
    corr = (codes >> 4).astype(np.int32) - 1
    print(
        f"{args.shape}: K={K} N={N} groups={n_groups:,} "
        f"z_range=[{int(zs.min())},{int(zs.max())}] "
        f"corr_nonzero={float(np.count_nonzero(corr) / n_groups):.6f} "
        f"bit_exact=1.000000 control_fired=yes -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
