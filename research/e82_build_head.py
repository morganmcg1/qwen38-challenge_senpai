#!/usr/bin/env python3
"""E82 rung 0, step 3: build a 4-bit g64 head from a BF16 trunk.

The three hard build constraints from the advisor's feedback are enforced here,
not documented and hoped for:

  1. `draft_lm_head.{weight,scales,biases}` are copied as raw bytes from the
     pinned declared head and asserted sha256-identical.
  2. The written artifact must land within +-2 % of the declared head's
     427,742,600 B, so the per-draft-step weight traffic channel is held fixed.
  3. `precision_islands.{q,k,v}` are recomputed against THIS trunk by the rule
     the declared head records in its own metadata -- largest per-output-row
     fp32 reconstruction SSE, Q=1024, K=all, V=all -- never copied.

`verify` mode applies the same rule to the EigenLabs master and requires it to
reproduce the declared head's island indices bit for bit. That is what licenses
the claim that this script implements the pinned head's rule rather than a
plausible guess at it.

  python3 research/e82_build_head.py verify
  python3 research/e82_build_head.py build --source soup --tag e82-xkm-soup-q4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from e82_st import SafeTensors, bf16_to_f32, file_sha256, tree_digest, write_safetensors

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
MASTER = CACHE / "mtp-head/model.safetensors"
DECLARED = CACHE / "mtp-head-declared/model.safetensors"
XKM = CACHE / "e82/xkm-retrained"
SOURCES = {
    "master": MASTER,
    "soup": XKM / "model.safetensors",
    "parent_a": XKM / "parents/parent-a-generic-plus-structured.safetensors",
    "parent_b": XKM / "parents/parent-b-plus-copytask.safetensors",
    "qat": XKM / "parents/parent-a-qat-4bit-aware.safetensors",
}
DECLARED_BYTES = 427_742_600
GROUP, BITS = 64, 4

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
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.q_norm.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
]
DRAFT = ["draft_lm_head.weight", "draft_lm_head.scales", "draft_lm_head.biases"]
ISLAND_Q_ROWS = 1024


def quantize(w_bf16_u16: np.ndarray):
    """MLX affine quantization from the stored BF16 bits, exactly as the shipped
    declared head was produced (proved bit-identical in e82_head_audit.py)."""
    import mlx.core as mx

    w = mx.array(w_bf16_u16).view(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    deq = mx.dequantize(q, s, b, group_size=GROUP, bits=BITS)
    mx.eval(q, s, b, deq)
    return (
        np.array(q, copy=True),
        np.array(s.view(mx.uint16), copy=True),
        np.array(b.view(mx.uint16), copy=True),
        np.array(deq.astype(mx.float32), copy=True),
    )


def row_sse(ref_f32: np.ndarray, deq_f32: np.ndarray) -> np.ndarray:
    d = ref_f32 - deq_f32
    return np.einsum("ij,ij->i", d, d)


def island_indices(sse: np.ndarray, count: int) -> np.ndarray:
    """Top-`count` rows by reconstruction SSE, emitted in ascending row order."""
    top = np.argpartition(sse, -count)[-count:]
    return np.sort(top).astype(np.int32)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    nb = float(np.linalg.norm(b))
    na = float(np.linalg.norm(a))
    return {
        "rel_l2": float(np.linalg.norm(d)) / nb if nb else 0.0,
        "cosine": float(np.dot(a.ravel(), b.ravel()) / (na * nb)) if na and nb else 1.0,
        "max_abs_diff": float(np.abs(d).max()),
    }


def cmd_verify(args) -> None:
    """Reproduce the declared head's island indices from the master."""
    master, decl = SafeTensors(MASTER), SafeTensors(DECLARED)
    print("island selection rule replay: master -> declared head indices")
    ok = True
    for proj, count in (("q", ISLAND_Q_ROWS), ("k", None), ("v", None)):
        name = f"layers.0.self_attn.{proj}_proj.weight"
        ref = master.f32(name)
        _, _, _, deq = quantize(master.array(name))
        sse = row_sse(ref, deq)
        want = np.array(decl.array(f"precision_islands.{proj}.indices"))
        got = island_indices(sse, count) if count else np.arange(ref.shape[0], dtype=np.int32)
        same = bool(np.array_equal(got, want))
        ok &= same
        overlap = len(set(got.tolist()) & set(want.tolist()))
        print(f"  {proj}: rows {want.size} exact_index_match={same} overlap={overlap}/{want.size}")
    print("VERIFY:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


def cmd_build(args) -> None:
    src_path = SOURCES[args.source]
    src, decl = SafeTensors(src_path), SafeTensors(DECLARED)
    out_dir = Path(os.path.expanduser(args.out_dir)) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, np.ndarray] = {}
    damage: dict[str, dict] = {}
    islands: dict[str, dict] = {}

    for name in TRUNK:
        stem = name[: -len(".weight")]
        ref = src.f32(name)
        q, s, b, deq = quantize(src.array(name))
        tensors[name] = q
        tensors[f"{stem}.scales"] = s
        tensors[f"{stem}.biases"] = b
        row = compare(deq, ref)
        # the same measurement for the pinned head, so the two are read together
        mref = SafeTensors(MASTER).f32(name)
        _, _, _, mdeq = quantize(SafeTensors(MASTER).array(name))
        row["pinned_head_rel_l2"] = compare(mdeq, mref)["rel_l2"]
        row["pinned_head_cosine"] = compare(mdeq, mref)["cosine"]
        row["finetune_rel_l2_vs_master"] = compare(ref, mref)["rel_l2"]
        damage[name] = row
        print(
            f"  {name:45s} requant relL2 {row['rel_l2']:.4e} cos {row['cosine']:.8f}"
            f" | pinned {row['pinned_head_rel_l2']:.4e} | finetune signal {row['finetune_rel_l2_vs_master']:.4e}"
        )
        if proj_of(name):
            proj = proj_of(name)
            count = ISLAND_Q_ROWS if proj == "q" else None
            sse = row_sse(ref, deq)
            idx = island_indices(sse, count) if count else np.arange(ref.shape[0], dtype=np.int32)
            tensors[f"precision_islands.{proj}.weight"] = np.ascontiguousarray(src.array(name)[idx])
            tensors[f"precision_islands.{proj}.indices"] = idx
            islands[proj] = {
                "rows": int(idx.size),
                "selected_sse_share": float(sse[idx].sum() / sse.sum()),
                "sse_total": float(sse.sum()),
            }

    for name in NORMS:
        tensors[name] = np.array(src.array(name))
    for name in DRAFT:
        tensors[name] = np.array(decl.array(name))

    ordered = {k: tensors[k] for k in sorted(tensors)}
    meta = {
        "format": "e82-xkm-requantized-q4-g64-plus-bf16-qkv-islands-v1",
        "trunk_source": f"{args.source}:{src_path.name}",
        "trunk_source_sha256": file_sha256(src_path),
        "trunk_quantization": "mlx affine, bits=4, group_size=64",
        "draft_lm_head": "byte-copied from amal-david/qwen38-mtp-head-q2-q4-rerank-v1",
        "selection": "largest per-output-row fp32 reconstruction SSE; Q=1024,K=all,V=all",
        "built_by": "senpai E82",
    }
    out_file = out_dir / "model.safetensors"
    nbytes = write_safetensors(out_file, ordered, meta)

    built = SafeTensors(out_file)
    checks = {
        "draft_lm_head_byte_identical": {n: built.sha256(n) == decl.sha256(n) for n in DRAFT},
        "norms_byte_identical_to_source": {n: built.sha256(n) == src.sha256(n) for n in NORMS},
        "bytes": nbytes,
        "declared_bytes": DECLARED_BYTES,
        "bytes_delta_pct": 100.0 * (nbytes - DECLARED_BYTES) / DECLARED_BYTES,
        "tensor_count": len(built.entries),
        "shapes_match_declared": {
            n: (built.entries[n].shape == decl.entries[n].shape and built.entries[n].dtype == decl.entries[n].dtype)
            for n in decl.names()
        },
    }
    digest, tree_bytes = tree_digest(out_dir)
    report = {
        "tag": args.tag,
        "source": str(src_path),
        "out": str(out_file),
        "file_sha256": file_sha256(out_file),
        "tree_sha256": digest,
        "tree_bytes": tree_bytes,
        "metadata": meta,
        "quantization_damage": damage,
        "islands": islands,
        "constraints": checks,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))

    all_ok = (
        all(checks["draft_lm_head_byte_identical"].values())
        and all(checks["norms_byte_identical_to_source"].values())
        and all(checks["shapes_match_declared"].values())
        and abs(checks["bytes_delta_pct"]) <= 2.0
    )
    print(
        f"\n{args.tag}: {nbytes:,} B ({checks['bytes_delta_pct']:+.3f} % vs declared),"
        f" {len(built.entries)} tensors, tree {digest}"
    )
    print("constraints:", "PASS" if all_ok else "FAIL")
    print(f"wrote {out_file}\nwrote {args.report}")

    # A run tree the CLI accepts: the head config.json is inert on the
    # declared-head branch but benchmark-qwen-mtp.sh refuses a dir without one.
    run_dir = Path(str(out_dir) + "-run")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_file = run_dir / "model.safetensors"
    if run_file.exists():
        run_file.unlink()
    os.link(out_file, run_file)
    subprocess.run(["cp", str(CACHE / "mtp-head/config.json"), str(run_dir / "config.json")], check=True)
    print(f"run tree staged at {run_dir}")
    raise SystemExit(0 if all_ok else 1)


def proj_of(name: str) -> str | None:
    for proj in ("q", "k", "v"):
        if name == f"layers.0.self_attn.{proj}_proj.weight":
            return proj
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.set_defaults(func=cmd_verify)
    b = sub.add_parser("build")
    b.add_argument("--source", choices=sorted(SOURCES), required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--out-dir", default=str(CACHE / "e82/built"))
    b.add_argument("--report", default=None)
    b.set_defaults(func=cmd_build)
    args = ap.parse_args()
    if getattr(args, "report", None) is None and args.cmd == "build":
        args.report = f"research/e82-build-{args.tag}.json"
    args.func(args)


if __name__ == "__main__":
    main()
