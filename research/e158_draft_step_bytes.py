#!/usr/bin/env python3
"""E158 R0.3 -- attribute bytes per draft step to named head tensors.

Reads the census artifact and the drafting geometry from the source, then
prices the bf16 trunk upgrade in published percent.

Every read set below is what the SHIPPED code path touches once per draft
step. The source facts it encodes, with line references on this base:

  Qwen35.swift:3488   complete K/V island coverage narrows the affine-4 pack to
                      the q rows and reads K and V from the BF16 island rows,
                      so the affine-4 k_proj/v_proj packs are never read at
                      draft time.
  Qwen35.swift:3319   the island arm defaults to `all`.
  Qwen35.swift:5688   derived index: 16 rows per leaf, 2-bit centroids.
  Qwen35.swift:2089   probe arm p15.
  Qwen35.swift:5687   32 rerank candidates against the exact affine-4 compact
                      matrix.
  Qwen35.swift:6065   with no `draft_lm_head.*` in the head tree the readout
                      falls back to the dense exact compact projection.

harness=ranked for every published-percent figure; harness=local for nothing
here, because this module measures no time.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CENSUS = ROOT / "research/e158-artifacts/head-census.json"
OUT = ROOT / "research/e158-artifacts/draft-step-bytes.json"

HIDDEN = 5_120
COMPACT_PADDED_ROWS = 98_336
ROWS_PER_LEAF = 16
PROBE_FRACTION = 0.15
RERANK_CANDIDATES = 32
BYTES_PER_PARAM_2BIT_G64 = 2 / 8 + 2 * 2 / 64
BYTES_PER_PARAM_4BIT_G64 = 4 / 8 + 2 * 2 / 64

# harness=ranked exchange rates, assignment "Baselines" block.
PCT_PER_MB_PER_DRAFT_STEP = 0.01793
# Repriced by E158 R1 F6. The old 2.6701 came from a regression that let the
# schedule move, so it charged a depth change to acceptance and overstated the
# rate 2.65x. At a fixed schedule the rate is q / (1 + a), measured 1.0093
# %/pt over [0.9688, 1.1016]; see research/e158-artifacts/f6-repricing.json.
# This constant is a DENOMINATOR here, so the correction makes every byte
# increase 2.65x harder to justify.
PCT_PER_ACCEPTANCE_POINT = 1.0093
PCT_PER_ACCEPTANCE_POINT_RANGE = (0.9688, 1.1016)
MEAN_DRAFT_STEPS_PER_ROUND = 4.382
ROOFLINE_BYTES_PER_SECOND = 567e9
MODELLED_RANKED_ROUND_US = 43_114.0


def tensor_map(head: dict) -> dict[str, int]:
    return {t["name"]: t["bytes"] for t in head["tensors"]}


def declared_trunk_reads(d: dict[str, int]) -> dict[str, int]:
    """What the declared head's trunk reads once per draft step."""
    reads = {
        "fc.weight": d["fc.weight"],
        "fc.scales": d["fc.scales"],
        "fc.biases": d["fc.biases"],
        "layers.0.self_attn.q_proj.weight": d["layers.0.self_attn.q_proj.weight"],
        "layers.0.self_attn.q_proj.scales": d["layers.0.self_attn.q_proj.scales"],
        "layers.0.self_attn.q_proj.biases": d["layers.0.self_attn.q_proj.biases"],
        "precision_islands.q.weight": d["precision_islands.q.weight"],
        "precision_islands.q.indices": d["precision_islands.q.indices"],
        # _exactKVDenseW: k and v island rows put back in output order, read as
        # one dense BF16 [2048, 5120] matmul.
        "precision_islands.k.weight": d["precision_islands.k.weight"],
        "precision_islands.v.weight": d["precision_islands.v.weight"],
        "layers.0.self_attn.o_proj.weight": d["layers.0.self_attn.o_proj.weight"],
        "layers.0.self_attn.o_proj.scales": d["layers.0.self_attn.o_proj.scales"],
        "layers.0.self_attn.o_proj.biases": d["layers.0.self_attn.o_proj.biases"],
    }
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for part in ("weight", "scales", "biases"):
            key = f"layers.0.mlp.{proj}.{part}"
            reads[key] = d[key]
    for norm in (
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "norm.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    ):
        reads[norm] = d[norm]
    return reads


def declared_unread(d: dict[str, int]) -> dict[str, int]:
    """Resident in the declared artifact, never read at draft time."""
    unread = {}
    for proj in ("k_proj", "v_proj"):
        for part in ("weight", "scales", "biases"):
            key = f"layers.0.self_attn.{proj}.{part}"
            unread[key] = d[key]
    unread["precision_islands.k.indices"] = d["precision_islands.k.indices"]
    unread["precision_islands.v.indices"] = d["precision_islands.v.indices"]
    return unread


def pinned_trunk_reads(p: dict[str, int]) -> dict[str, int]:
    """The bf16 head has no readout, so its whole tensor set is the trunk."""
    return dict(p)


def declared_readout_reads() -> dict[str, int]:
    leaves = COMPACT_PADDED_ROWS // ROWS_PER_LEAF
    probes = max(1, int(-(-PROBE_FRACTION * leaves // 1)))
    refined = probes * ROWS_PER_LEAF
    return {
        "derived centroids (2-bit g64, leaf means of draft_lm_head rows)": int(
            leaves * HIDDEN * BYTES_PER_PARAM_2BIT_G64
        ),
        "draft_lm_head rows in probed leaves (2-bit g64)": int(
            refined * HIDDEN * BYTES_PER_PARAM_2BIT_G64
        ),
        "exact compact rerank rows (4-bit g64, target lm_head gather)": int(
            RERANK_CANDIDATES * HIDDEN * BYTES_PER_PARAM_4BIT_G64
        ),
    }, {"leaves": leaves, "probes": probes, "refined_rows": refined}


def pinned_readout_reads() -> dict[str, int]:
    return {
        "dense exact compact projection (4-bit g64, target lm_head gather)": int(
            COMPACT_PADDED_ROWS * HIDDEN * BYTES_PER_PARAM_4BIT_G64
        )
    }


def mb(value: int | float) -> float:
    return value / 1e6


def main() -> int:
    census = json.loads(CENSUS.read_text())
    declared = tensor_map(census["declared_head"])
    pinned = tensor_map(census["pinned_head"])

    d_trunk = declared_trunk_reads(declared)
    d_unread = declared_unread(declared)
    d_readout, geometry = declared_readout_reads()
    p_trunk = pinned_trunk_reads(pinned)
    p_readout = pinned_readout_reads()

    d_trunk_total = sum(d_trunk.values())
    d_readout_total = sum(d_readout.values())
    p_trunk_total = sum(p_trunk.values())
    p_readout_total = sum(p_readout.values())

    # Per-matrix price of moving that matrix from its declared representation
    # to bf16. K and V are already exactly bf16 through the islands.
    upgrades = {}
    pairs = [
        ("fc", ["fc.weight", "fc.scales", "fc.biases"], "fc.weight"),
        (
            "self_attn.q_proj",
            [
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.q_proj.scales",
                "layers.0.self_attn.q_proj.biases",
                "precision_islands.q.weight",
                "precision_islands.q.indices",
            ],
            "layers.0.self_attn.q_proj.weight",
        ),
        (
            "self_attn.k_proj",
            ["precision_islands.k.weight"],
            "layers.0.self_attn.k_proj.weight",
        ),
        (
            "self_attn.v_proj",
            ["precision_islands.v.weight"],
            "layers.0.self_attn.v_proj.weight",
        ),
        (
            "self_attn.o_proj",
            [
                "layers.0.self_attn.o_proj.weight",
                "layers.0.self_attn.o_proj.scales",
                "layers.0.self_attn.o_proj.biases",
            ],
            "layers.0.self_attn.o_proj.weight",
        ),
    ]
    for proj in ("gate_proj", "up_proj", "down_proj"):
        pairs.append(
            (
                f"mlp.{proj}",
                [f"layers.0.mlp.{proj}.{p}" for p in ("weight", "scales", "biases")],
                f"layers.0.mlp.{proj}.weight",
            )
        )
    for label, declared_keys, pinned_key in pairs:
        now = sum(d_trunk[k] for k in declared_keys)
        then = pinned[pinned_key]
        delta = then - now
        upgrades[label] = {
            "declared_bytes_per_draft_step": now,
            "bf16_bytes_per_draft_step": then,
            "delta_bytes": delta,
            "delta_MB": round(mb(delta), 4),
            "published_cost_pct": round(mb(delta) * PCT_PER_MB_PER_DRAFT_STEP, 4),
        }

    trunk_delta = p_trunk_total - d_trunk_total
    trunk_delta_mb = mb(trunk_delta)
    result = {
        "harness": "ranked for every published-percent figure",
        "geometry": geometry,
        "declared_head": {
            "trunk_reads_per_draft_step": d_trunk,
            "trunk_total_bytes": d_trunk_total,
            "readout_reads_per_draft_step": d_readout,
            "readout_total_bytes": d_readout_total,
            "total_bytes_per_draft_step": d_trunk_total + d_readout_total,
            "total_MB_per_draft_step": round(mb(d_trunk_total + d_readout_total), 4),
            "resident_but_never_read_at_draft_time": d_unread,
            "resident_but_never_read_total_bytes": sum(d_unread.values()),
        },
        "pinned_bf16_head": {
            "trunk_reads_per_draft_step": p_trunk,
            "trunk_total_bytes": p_trunk_total,
            "readout_reads_per_draft_step": p_readout,
            "readout_total_bytes": p_readout_total,
            "total_bytes_per_draft_step": p_trunk_total + p_readout_total,
            "total_MB_per_draft_step": round(mb(p_trunk_total + p_readout_total), 4),
        },
        "bf16_trunk_upgrade": {
            "delta_bytes_per_draft_step": trunk_delta,
            "delta_MB_per_draft_step": round(trunk_delta_mb, 4),
            "published_cost_pct": round(trunk_delta_mb * PCT_PER_MB_PER_DRAFT_STEP, 4),
            "break_even_acceptance_points": round(
                trunk_delta_mb * PCT_PER_MB_PER_DRAFT_STEP / PCT_PER_ACCEPTANCE_POINT,
                4,
            ),
            "us_per_round_at_roofline": round(
                trunk_delta / ROOFLINE_BYTES_PER_SECOND * 1e6
                * MEAN_DRAFT_STEPS_PER_ROUND,
                1,
            ),
            "modelled_ranked_round_us": MODELLED_RANKED_ROUND_US,
            "per_matrix": upgrades,
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    print(f"declared trunk   {d_trunk_total:>12,} B = {mb(d_trunk_total):8.2f} MB / draft step")
    print(f"declared readout {d_readout_total:>12,} B = {mb(d_readout_total):8.2f} MB / draft step")
    print(f"declared TOTAL   {d_trunk_total + d_readout_total:>12,} B = {mb(d_trunk_total + d_readout_total):8.2f} MB / draft step")
    print(f"pinned   trunk   {p_trunk_total:>12,} B = {mb(p_trunk_total):8.2f} MB / draft step")
    print(f"pinned   readout {p_readout_total:>12,} B = {mb(p_readout_total):8.2f} MB / draft step")
    print(f"pinned   TOTAL   {p_trunk_total + p_readout_total:>12,} B = {mb(p_trunk_total + p_readout_total):8.2f} MB / draft step")
    print(f"bf16 trunk delta {trunk_delta:>12,} B = {trunk_delta_mb:8.2f} MB -> "
          f"{trunk_delta_mb * PCT_PER_MB_PER_DRAFT_STEP:.4f} % published, "
          f"break-even {trunk_delta_mb * PCT_PER_MB_PER_DRAFT_STEP / PCT_PER_ACCEPTANCE_POINT:.3f} pp")
    for label, row in upgrades.items():
        print(f"  {label:22s} {row['delta_MB']:9.2f} MB  {row['published_cost_pct']:7.4f} %")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
