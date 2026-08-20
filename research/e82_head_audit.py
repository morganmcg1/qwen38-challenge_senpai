#!/usr/bin/env python3
"""E82 rung 0, steps 1-2: byte-level provenance audit of four MTP heads.

The assignment asserts three provenance claims. This script re-derives each one
from the artifacts rather than accepting them:

  1. `xkm/qwen3.8-27b-mtp-head-retrained` is a genuine fine-tune of the
     EigenLabs BF16 master: projections perturbed, every norm bit-frozen.
  2. `Kamciosz/qwen38-mtp-head-xkm-affine2-v1` is a pure graft: xkm's 15
     tensors byte-copied, plus `draft_lm_head.*` byte-copied from the pinned
     declared head, with no new training.
  3. `amal-david/qwen38-mtp-head-q2-q4-rerank-v1` (our declared head) carries
     no trained weight change at all: it is quantization and layout work on the
     untouched master.

Claim 3 gets the strongest available test. The declared head's 4-bit trunk is
dequantized and compared to the master, and -- decisively -- the master is
re-quantized here with MLX affine/g64 and the packed uint32 payload is compared
BYTE FOR BYTE against what amal-david shipped. A bit-exact match proves both
that the weights underneath are untouched and that this script's quantizer is
the same one that produced the shipped head, which is what makes the E82 build
step trustworthy.

  python3 research/e82_head_audit.py --out research/e82-head-audit.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from e82_st import SafeTensors, bf16_to_f32, file_sha256, tree_digest

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
HEADS = {
    "master_bf16": CACHE / "mtp-head/model.safetensors",
    "declared_q4": CACHE / "mtp-head-declared/model.safetensors",
    "xkm_bf16": CACHE / "e82/xkm-retrained/model.safetensors",
    "kamciosz_graft": CACHE / "e82/kamciosz-graft/model.safetensors",
}
GROUP = 64
BITS = 4


def compare_f32(a: np.ndarray, b: np.ndarray) -> dict:
    """Like-for-like numeric comparison of a candidate `a` against reference `b`."""
    d = a - b
    nb = float(np.linalg.norm(b))
    na = float(np.linalg.norm(a))
    dot = float(np.dot(a.ravel(), b.ravel()))
    return {
        "n": int(a.size),
        "pct_elements_changed": 100.0 * float(np.count_nonzero(d)) / a.size,
        "rel_l2": float(np.linalg.norm(d)) / nb if nb else 0.0,
        "cosine": dot / (na * nb) if na and nb else 1.0,
        "max_abs_diff": float(np.abs(d).max()),
        "rms_ref": nb / np.sqrt(a.size),
    }


def dequant_affine(w_u32: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int, group: int) -> np.ndarray:
    """MLX affine dequantization: w = q * scale + bias, groups packed low bits first."""
    per_word = 32 // bits
    rows, packed_cols = w_u32.shape
    mask = (1 << bits) - 1
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).astype(np.uint32)
    q = ((w_u32[:, :, None] >> shifts[None, None, :]) & np.uint32(mask)).astype(np.float32)
    q = q.reshape(rows, packed_cols * per_word)
    cols = q.shape[1]
    q = q.reshape(rows, cols // group, group)
    return (q * scales[:, :, None] + biases[:, :, None]).reshape(rows, cols)


def quantize_affine_mlx(w_f32: np.ndarray, bits: int, group: int):
    """Quantize with MLX itself, so packing matches the runtime loader exactly."""
    import mlx.core as mx

    w = mx.array(w_f32).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=group, bits=bits)
    mx.eval(q, s, b)
    return (
        np.array(q, copy=True),
        np.array(s.astype(mx.bfloat16).view(mx.uint16), copy=True),
        np.array(b.astype(mx.bfloat16).view(mx.uint16), copy=True),
    )


TRUNK = [
    "fc.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
]
NORMS = [
    "norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e82-head-audit.json")
    ap.add_argument("--skip-requant-identity", action="store_true")
    args = ap.parse_args()

    st = {k: SafeTensors(p) for k, p in HEADS.items()}
    report: dict = {"artifacts": {}}
    for name, p in HEADS.items():
        report["artifacts"][name] = {
            "path": str(p),
            "bytes": p.stat().st_size,
            "file_sha256": file_sha256(p),
            "tensors": len(st[name].entries),
            "metadata": st[name].metadata,
        }
        print(f"{name:16s} {p.stat().st_size:>12,} B  {len(st[name].entries)} tensors")

    master, xkm, kam, decl = st["master_bf16"], st["xkm_bf16"], st["kamciosz_graft"], st["declared_q4"]

    # --- claim 1: xkm is a real fine-tune of the master --------------------
    print("\n=== xkm BF16 vs EigenLabs master BF16 (like-for-like) ===")
    xkm_tbl = {}
    for name in TRUNK + NORMS:
        a, b = xkm.f32(name), master.f32(name)
        row = compare_f32(a, b)
        row["bytes_identical"] = xkm.sha256(name) == master.sha256(name)
        xkm_tbl[name] = row
        print(
            f"  {name:45s} chg {row['pct_elements_changed']:6.2f}%  relL2 {row['rel_l2']:.4e}"
            f"  cos {row['cosine']:.8f}  identical={row['bytes_identical']}"
        )
    report["xkm_vs_master"] = xkm_tbl

    # --- claim 2: Kamciosz is a byte-level graft ---------------------------
    print("\n=== Kamciosz graft: byte identity against claimed parents ===")
    graft = {}
    for name in kam.names():
        parent = "xkm_bf16" if name in xkm else "declared_q4"
        src = xkm if name in xkm else decl
        same = kam.sha256(name) == src.sha256(name) if name in src else None
        graft[name] = {"claimed_parent": parent, "bytes_identical": same, "sha256": kam.sha256(name)}
        print(f"  {name:45s} <- {parent:14s} identical={same}")
    report["kamciosz_graft"] = graft

    # --- claim 3a: declared-head islands are verbatim master rows ----------
    print("\n=== declared head precision islands vs master rows (bit-exactness) ===")
    islands = {}
    for proj in ("q", "k", "v"):
        idx = np.array(decl.array(f"precision_islands.{proj}.indices"))
        isl = decl.array(f"precision_islands.{proj}.weight")  # stored BF16 bits
        mrows = master.array(f"layers.0.self_attn.{proj}_proj.weight")[idx]
        xrows = xkm.array(f"layers.0.self_attn.{proj}_proj.weight")[idx]
        bit_exact_master = bool(np.array_equal(isl, mrows))
        f_isl, f_m, f_x = bf16_to_f32(isl), bf16_to_f32(mrows), bf16_to_f32(xrows)
        islands[proj] = {
            "rows": int(idx.size),
            "index_min": int(idx.min()),
            "index_max": int(idx.max()),
            "indices_are_identity": bool(np.array_equal(idx, np.arange(idx.size))),
            "bit_exact_vs_master": bit_exact_master,
            "vs_master": compare_f32(f_isl, f_m),
            "master_vs_xkm_same_rows": compare_f32(f_x, f_m),
        }
        print(
            f"  {proj}: rows {idx.size} bit_exact_vs_master={bit_exact_master} "
            f"relL2 {islands[proj]['vs_master']['rel_l2']:.3e} "
            f"cos {islands[proj]['vs_master']['cosine']:.8f} | xkm moved those rows by "
            f"relL2 {islands[proj]['master_vs_xkm_same_rows']['rel_l2']:.3e}"
        )
    report["declared_islands_vs_master"] = islands

    # --- claim 3b: dequantized declared trunk vs master, and bit-exact
    #     reproduction of the shipped packing from the master ---------------
    print("\n=== declared 4-bit trunk: damage vs master, and packing identity ===")
    dq = {}
    for name in TRUNK:
        stem = name[: -len(".weight")]
        w = decl.array(name)
        s = bf16_to_f32(decl.array(f"{stem}.scales"))
        b = bf16_to_f32(decl.array(f"{stem}.biases"))
        deq = dequant_affine(w, s, b, BITS, GROUP)
        ref = master.f32(name)
        row = compare_f32(deq, ref)
        if not args.skip_requant_identity:
            qq, ss, bb = quantize_affine_mlx(ref, BITS, GROUP)
            row["mlx_requant_weight_bit_identical"] = bool(np.array_equal(qq, np.array(w)))
            row["mlx_requant_scales_bit_identical"] = bool(np.array_equal(ss, np.array(decl.array(f"{stem}.scales"))))
            row["mlx_requant_biases_bit_identical"] = bool(np.array_equal(bb, np.array(decl.array(f"{stem}.biases"))))
            row["mlx_requant_weight_mismatch_frac"] = float(np.mean(qq != np.array(w)))
        dq[name] = row
        print(
            f"  {name:45s} relL2 {row['rel_l2']:.4e} cos {row['cosine']:.8f}"
            + (
                f"  packing_identical={row['mlx_requant_weight_bit_identical']}"
                f" (mismatch {row['mlx_requant_weight_mismatch_frac']:.2e})"
                if not args.skip_requant_identity
                else ""
            )
        )
    report["declared_dequant_vs_master"] = dq

    # norms are stored bare in the declared head; compare directly
    print("\n=== declared head norms vs master (bit identity) ===")
    dn = {}
    for name in NORMS:
        same = decl.sha256(name) == master.sha256(name)
        dn[name] = {"bytes_identical": same}
        print(f"  {name:45s} identical={same}")
    report["declared_norms_vs_master"] = dn

    # tree digests for the record
    report["tree_digests"] = {}
    for key, d in (("declared", CACHE / "mtp-head-declared"),):
        digest, total = tree_digest(d)
        report["tree_digests"][key] = {"dir": str(d), "sha256": digest, "bytes": total}
        print(f"\ntree digest {key}: {digest} ({total:,} B)")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
