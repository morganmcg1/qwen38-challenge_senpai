#!/usr/bin/env python3
"""E111 rung 0: is the stored affine-4 bias losslessly reconstructible from
(scale, small code)?

MLX's affine quantizer (quantized.h, affine_quantize) builds each group record
as

    scale_raw = max((w_max - w_min) / 15, eps)      # sign flipped when
    side      = |w_min| > |w_max|                   # side == false
    edge      = side ? w_min : w_max
    q0        = round(edge / scale)
    scale     = (q0 == 0) ? scale : edge / q0
    bias      = (q0 == 0) ? 0     : edge

so the stored bias equals q0 * scale in fp32 before both are rounded to BF16
independently. With z = -q0 this predicts

    stored_bias_bits == bf16_rne(-z * stored_scale) + correction,
    |correction| <= 1 BF16 ordinal.

This script measures, for every group of every scored target trunk tensor:
the recovered z, whether a (z, correction) pair reproduces the stored bias bit
for bit, whether that z is unique, and the correction distribution. It reads
only `scales` and `biases`, never the packed weights.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

GROUP_SIZE = 64
BITS = 4

# family -> (checkpoint suffixes contributing to the fused scored linear, K, N)
FAMILIES = {
    "gdn.in_proj": (
        (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
        ),
        5120,
        16480,
    ),
    "fa.qkv": (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), 5120, 14336),
    "mlp.gate_up": (("mlp.gate_proj", "mlp.up_proj"), 5120, 34816),
    "lm_head": (("lm_head",), 5120, 248320),
    "gdn.out_proj": (("linear_attn.out_proj",), 6144, 5120),
    "fa.o_proj": (("self_attn.o_proj",), 6144, 5120),
    "mlp.down": (("mlp.down_proj",), 17408, 5120),
}
UNSCORED = {"embed_tokens": (("embed_tokens",), 5120, 248320)}


def read_header(path: Path):
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> BF16 bit pattern (no NaN inputs here)."""
    u = x.view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return rounded.astype(np.uint16)


def signed_ordinal(bits: np.ndarray) -> np.ndarray:
    """Sign-magnitude BF16 bit pattern -> monotone integer ordinal.

    +0 and -0 both map to 0, so a sign flip at zero costs no ordinals.
    """
    mag = (bits & np.uint16(0x7FFF)).astype(np.int32)
    neg = (bits & np.uint16(0x8000)) != 0
    return np.where(neg, -mag, mag)


class Stats:
    __slots__ = (
        "groups",
        "match_any",
        "match_unique",
        "corr_hist",
        "z_hist",
        "zero_bias",
        "max_abs_delta",
        "z_over_15",
        "no_match_examples",
    )

    def __init__(self):
        self.groups = 0
        self.match_any = 0
        self.match_unique = 0
        self.corr_hist = Counter()
        self.z_hist = Counter()
        self.zero_bias = 0
        self.max_abs_delta = 0
        self.z_over_15 = 0
        self.no_match_examples = []

    def merge(self, other: "Stats"):
        self.groups += other.groups
        self.match_any += other.match_any
        self.match_unique += other.match_unique
        self.corr_hist.update(other.corr_hist)
        self.z_hist.update(other.z_hist)
        self.zero_bias += other.zero_bias
        self.max_abs_delta = max(self.max_abs_delta, other.max_abs_delta)
        self.z_over_15 += other.z_over_15
        self.no_match_examples.extend(other.no_match_examples[:4])


def audit_chunk(scale_bits: np.ndarray, bias_bits: np.ndarray, zmax: int, st: Stats, tag: str):
    n = scale_bits.size
    st.groups += n
    scale = bf16_to_f32(scale_bits)
    bias_ord = signed_ordinal(bias_bits)
    st.zero_bias += int(np.count_nonzero((bias_bits & np.uint16(0x7FFF)) == 0))

    # bias == -z * scale, so z ~= -bias / scale. scale is never zero (eps floor).
    bias_f = bf16_to_f32(bias_bits)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_est = np.rint(-bias_f / scale)
    z_est = np.nan_to_num(z_est, nan=0.0, posinf=float(zmax), neginf=0.0)
    z_est = np.clip(z_est, 0, zmax).astype(np.int32)

    best_delta = np.full(n, 1 << 20, dtype=np.int32)
    best_z = np.zeros(n, dtype=np.int32)
    match_count = np.zeros(n, dtype=np.int8)

    for off in (-1, 0, 1):
        z = np.clip(z_est + off, 0, zmax)
        prod = (-z.astype(np.float32)) * scale
        cand_bits = f32_to_bf16_rne(prod)
        delta = bias_ord - signed_ordinal(cand_bits)
        hit = np.abs(delta) <= 1
        if off == -1:
            match_count += hit.astype(np.int8)
        else:
            # a duplicate z (from clipping) must not be counted twice
            dup = z == np.clip(z_est + (off - 1), 0, zmax)
            match_count += (hit & ~dup).astype(np.int8)
        better = np.abs(delta) < np.abs(best_delta)
        best_delta = np.where(better, delta, best_delta)
        best_z = np.where(better, z, best_z)

    any_match = match_count >= 1
    st.match_any += int(np.count_nonzero(any_match))
    st.match_unique += int(np.count_nonzero(match_count == 1))
    st.max_abs_delta = max(st.max_abs_delta, int(np.abs(best_delta).max(initial=0)))
    st.z_over_15 += int(np.count_nonzero(best_z[any_match] > 15))

    corr = best_delta[any_match]
    for v, c in zip(*np.unique(corr, return_counts=True)):
        st.corr_hist[int(v)] += int(c)
    zs = best_z[any_match]
    for v, c in zip(*np.unique(zs, return_counts=True)):
        st.z_hist[int(v)] += int(c)

    if not any_match.all() and len(st.no_match_examples) < 4:
        idx = np.flatnonzero(~any_match)[:4]
        for i in idx:
            st.no_match_examples.append(
                {
                    "tensor": tag,
                    "group": int(i),
                    "scale_bits": int(scale_bits[i]),
                    "bias_bits": int(bias_bits[i]),
                    "scale": float(scale[i]),
                    "bias": float(bias_f[i]),
                    "z_est": int(z_est[i]),
                    "best_delta": int(best_delta[i]),
                }
            )


def full_sweep(scale_bits: np.ndarray, bias_bits: np.ndarray, zmax: int) -> np.ndarray:
    """Exhaustive z search. Returns the number of matching z per group."""
    scale = bf16_to_f32(scale_bits)
    bias_ord = signed_ordinal(bias_bits)
    count = np.zeros(scale.size, dtype=np.int16)
    for z in range(zmax + 1):
        cand = f32_to_bf16_rne(np.float32(-z) * scale)
        count += (np.abs(bias_ord - signed_ordinal(cand)) <= 1).astype(np.int16)
    return count


def run_verification(wdir: Path, wmap: dict, headers: dict, sample: int, seed: int) -> dict:
    """Two checks the neighbour search cannot make on its own.

    1. Exhaustive z in [0, 15] on a random sample: proves the recovered z is
       the only one that matches, so 4 bits of z are sufficient and unambiguous.
    2. Positive control: the same sample with the stored bias moved two BF16
       ordinals away must produce zero matches. This proves the test can fail.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for fam, (suffixes, _k, _n) in FAMILIES.items():
        names = sorted(
            {
                k[: -len(".scales")]
                for k in wmap
                if k.endswith(".scales") and any(k[: -len(".scales")].endswith(s) for s in suffixes)
            }
        )
        name = names[len(names) // 2]
        shard = wmap[name + ".scales"]
        hdr, off = headers[shard]
        s_info, b_info = hdr[name + ".scales"], hdr[name + ".biases"]
        n = int(np.prod(s_info["shape"]))
        take = min(sample, n)
        start = int(rng.integers(0, n - take + 1))
        with (wdir / shard).open("rb") as fh:
            fh.seek(off + s_info["data_offsets"][0] + 2 * start)
            sb = np.frombuffer(fh.read(2 * take), dtype=np.uint16)
            fh.seek(off + b_info["data_offsets"][0] + 2 * start)
            bb = np.frombuffer(fh.read(2 * take), dtype=np.uint16)

        counts = full_sweep(sb, bb, 15)
        controls = {}
        for shift in (2, 3):
            mag = (bb & np.uint16(0x7FFF)).astype(np.int32) + shift
            damaged = ((bb & np.uint16(0x8000)).astype(np.int32) | mag).astype(np.uint16)
            controls[shift] = int(np.count_nonzero(full_sweep(sb, damaged, 15) > 0))
        out[fam] = {
            "tensor": name,
            "sampled_groups": int(take),
            "exhaustive_unique": int(np.count_nonzero(counts == 1)),
            "exhaustive_none": int(np.count_nonzero(counts == 0)),
            "exhaustive_multi": int(np.count_nonzero(counts > 1)),
            # +2 ordinals still matches exactly the groups whose true correction
            # is -1, which is the tolerance window working as designed; +3
            # ordinals leaves the window for every group and must match none.
            "control_plus2_matches": controls[2],
            "control_plus3_matches": controls[3],
        }
        print(
            f"verify {fam:14s} {name}: unique={out[fam]['exhaustive_unique']}/{take} "
            f"none={out[fam]['exhaustive_none']} multi={out[fam]['exhaustive_multi']} "
            f"control+2={controls[2]} control+3={controls[3]}",
            flush=True,
        )
    return out


def audit_tensor(path: Path, data_off: int, hdr: dict, name: str, zmax: int, chunk: int) -> Stats:
    st = Stats()
    s_info = hdr[name + ".scales"]
    b_info = hdr[name + ".biases"]
    assert s_info["dtype"] == "BF16" and b_info["dtype"] == "BF16", name
    n = int(np.prod(s_info["shape"]))
    s0 = data_off + s_info["data_offsets"][0]
    b0 = data_off + b_info["data_offsets"][0]
    with path.open("rb") as fh:
        done = 0
        while done < n:
            take = min(chunk, n - done)
            fh.seek(s0 + 2 * done)
            sb = np.frombuffer(fh.read(2 * take), dtype=np.uint16)
            fh.seek(b0 + 2 * done)
            bb = np.frombuffer(fh.read(2 * take), dtype=np.uint16)
            audit_chunk(sb, bb, zmax, st, name)
            done += take
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--zmax", type=int, default=63)
    ap.add_argument("--chunk", type=int, default=4 << 20)
    ap.add_argument("--include-unscored", action="store_true")
    ap.add_argument("--verify-sample", type=int, default=1 << 20)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--out", default="research/e111-rung0-bias-recode.json")
    args = ap.parse_args()

    wdir = Path(args.weights)
    index = json.loads((wdir / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    headers = {}
    for shard in sorted(set(wmap.values())):
        headers[shard] = read_header(wdir / shard)

    fams = dict(FAMILIES)
    if args.include_unscored:
        fams.update(UNSCORED)

    tensors = sorted({k[: -len(".scales")] for k in wmap if k.endswith(".scales")})
    fam_stats = {f: Stats() for f in fams}
    fam_members = {f: [] for f in fams}
    unassigned = []
    for t in tensors:
        placed = False
        for fam, (suffixes, _k, _n) in fams.items():
            if any(t.endswith(sfx) for sfx in suffixes):
                fam_members[fam].append(t)
                placed = True
                break
        if not placed:
            unassigned.append(t)

    total_groups = 0
    for fam in fams:
        for t in fam_members[fam]:
            shard = wmap[t + ".scales"]
            hdr, off = headers[shard]
            st = audit_tensor(wdir / shard, off, hdr, t, args.zmax, args.chunk)
            fam_stats[fam].merge(st)
        s = fam_stats[fam]
        total_groups += s.groups
        print(
            f"{fam:14s} tensors={len(fam_members[fam]):3d} groups={s.groups:>12,d} "
            f"match={s.match_any / max(s.groups,1):.9f} unique={s.match_unique / max(s.groups,1):.9f} "
            f"z>15={s.z_over_15:,d} maxz={max(s.z_hist) if s.z_hist else -1} "
            f"corr={dict(sorted(s.corr_hist.items()))}",
            flush=True,
        )

    verification = run_verification(wdir, wmap, headers, args.verify_sample, args.seed)

    report = {
        "weights_dir": str(wdir),
        "verification": verification,
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "zmax_searched": args.zmax,
        "unassigned_quantized_tensors": unassigned,
        "families": {},
        "total_scored_groups": total_groups,
    }
    for fam, st in fam_stats.items():
        k, n = fams[fam][1], fams[fam][2]
        report["families"][fam] = {
            "K": k,
            "N": n,
            "tensor_count": len(fam_members[fam]),
            "groups": st.groups,
            "share_with_match": st.match_any / max(st.groups, 1),
            "share_unique_match": st.match_unique / max(st.groups, 1),
            "share_needing_correction": sum(c for v, c in st.corr_hist.items() if v != 0)
            / max(st.groups, 1),
            "correction_hist": {str(v): c for v, c in sorted(st.corr_hist.items())},
            "z_hist": {str(v): c for v, c in sorted(st.z_hist.items())},
            "z_max": max(st.z_hist) if st.z_hist else -1,
            "z_over_15": st.z_over_15,
            "zero_bias_groups": st.zero_bias,
            "max_abs_delta_ordinals": st.max_abs_delta,
            "no_match_examples": st.no_match_examples[:4],
            "bytes_per_pass_shipped": st.groups * 36,
            "bytes_saved_bias6": st.groups * 1,
            "bytes_saved_nobias": st.groups * 2,
        }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
