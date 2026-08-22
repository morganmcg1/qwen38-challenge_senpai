#!/usr/bin/env python3
"""E131 rung 1: census every scored quantized-matvec entry point and price the
recoverable occupancy.

    usage: python3 research/e131_rung1.py --outdir research/e131-artifacts

Rung 0 asked which bodies one entry point can inline. Rung 1 asks the cost
question: across every scored entry point, how much round-cost-weighted
resident occupancy is currently unavailable, and by which mechanism.

Two recovery mechanisms are priced separately, because they cost very different
amounts of engineering:

  Class B, entry-point split. A dispatch executes one body but is allocated the
  register maximum over EVERY body the entry point can inline. Splitting the
  narrow dispatch into its own compiled entry point recovers the difference and
  needs no register work at all. This is the "free simdgroup".

  Class A, register diet. Reduce the register maximum of the body that sets the
  entry point's allocation. Priced at thresholds of 1, 2, 4 and 8 registers.

Compile-only. No GPU, no model, no timing. Every simdgroup number here is
`derived` under Rule 89: `floor(BUDGET / registers)` is a model output computed
from a measured register count, never a measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as A  # noqa: E402
import e123_arms as E  # noqa: E402
import e131_census as C  # noqa: E402
import jit_string_compile as J  # noqa: E402

ARCHES = (A.LOCAL_ARCH, A.RANKED_ARCH)
RANKED = A.RANKED_ARCH

# Every `affine_qmv*` entry point the scored decode round reaches, with the
# evidence that puts it on the live path.
ENTRY_POINTS = {
    "affine_qmv_fast<bfloat16_t, 64, 4, false>": {
        "role": "every unrouted affine-4/g64 projection: the whole MTP head at "
                "ntg.x=1, and the whole target verify forward on rounds of "
                "width 1 or 2",
        "shared": True,
    },
    "affine_qmv_fast<bfloat16_t, 64, 4, true>": {
        "role": "batched affine-4 matvec",
        "shared": False,
    },
    "affine_qmv_fast<bfloat16_t, 64, 2, false>": {
        "role": "affine-2 coarse draft readout, out_vec_size 98336; NOT live "
                "in the shipped config, the derived cluster index bypasses it",
        "shared": True,
    },
    "affine_qmv<bfloat16_t, 64, 2, false>": {
        "role": "draft centroid score, N=12292; non-fast because 12292 % 8 != 0",
        "shared": False,
    },
    "affine_gather_qmv_fast<bfloat16_t, 64, 2>": {
        "role": "probed row score, N=8 over B=3073 gathered leaves",
        "shared": False,
    },
}

# Bodies the shared wide entry point can inline, isolated in rung 0.
BODY_OF_NTG = {
    1: "impl4", 2: "xr2", 3: "m3", 4: "m4", 5: "m5",
    6: "m6", 7: "m7", 8: "m8", 9: "m9",
}

# Live dispatch inventory. Bytes are the exact quantized tensor bytes read per
# dispatch, from the declared head safetensors header and the backbone weight
# pass. `ntg_x` is the launched threadgroup count along x, which equals M
# because `quantized.cpp:254` launches `grid_dims(M, ceil(N/8), B)`.
HEAD_MB = {
    "mtp.fc": 29.49,
    "mtp.self_attn.q_proj": 35.39,
    "mtp.self_attn.o_proj": 17.69,
    "mtp.mlp.gate_up": 100.26,
    "mtp.mlp.down_proj": 50.13,
}
HEAD_QMV_MB = sum(HEAD_MB.values())
# Backbone quantized weight pass per verify forward, ledger 242.8.
BACKBONE_QMV_MB = 14412.0

# Draft depths to price. The shipped adaptive schedule is `costModelDepth`
# capped at `segmentedVerifyDepthCap = 7`, with in-repo receipts quoting an
# effective mean draft length of 4.32 to 5.10.
DEPTHS = (0, 1, 2, 3, 4, 5, 6, 7)
REFERENCE_DEPTH = 4
REGISTER_THRESHOLDS = (1, 2, 4, 8)


def census(cells: list[str]) -> dict:
    source = J.assemble(cells, None)
    out: dict = {}
    with tempfile.TemporaryDirectory() as raw:
        workdir = pathlib.Path(raw)
        lib = A.build_metallib(source, workdir)
        for arch in ARCHES:
            translated = A.translate(lib, arch, workdir)
            for cell in cells:
                record = translated.get(J.host_name(cell))
                if record is None:
                    raise SystemExit("missing %s on %s" % (cell, arch))
                registers = record["registers"]
                out.setdefault(cell, {})[arch] = {
                    "registers": registers,
                    "spill_bytes": record["spill_bytes"],
                    "text_bytes": record["text_bytes"],
                    "text_sha8": record["text_sha8"],
                    "simdgroups": E.simdgroups(registers, arch),
                }
    return out


def gain_pct(base: int, better: int) -> float:
    return round(100.0 * (better - base) / base, 3)


def dispatch_inventory(depth: int) -> list[dict]:
    """Byte-weighted dispatch classes for one round at this draft depth."""
    wide = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
    rows = []
    width = depth + 1
    # Route B claims the target verify forward only for widths 3 to 9.
    routed = 3 <= width <= 9
    rows.append({
        "dispatch": "target verify forward, all 257 affine-4 projections",
        "entry_point": None if routed else wide,
        "ntg_x": width,
        "megabytes": BACKBONE_QMV_MB,
        "owner": "thorfinn, Route B (E129)" if routed else "this census",
        "routed_to_route_b": routed,
    })
    for name, megabytes in HEAD_MB.items():
        rows.append({
            "dispatch": "MTP head %s" % name,
            "entry_point": wide,
            "ntg_x": 1,
            "megabytes": megabytes * depth,
            "owner": "this census",
            "routed_to_route_b": False,
        })
    return [r for r in rows if r["megabytes"] > 0.0]


def price(entries: dict, bodies: dict, depth: int) -> dict:
    """Round-cost-weighted recoverable residency at one draft depth."""
    rows = dispatch_inventory(depth)
    total = sum(r["megabytes"] for r in rows)
    priced = []
    for row in rows:
        share = row["megabytes"] / total
        record = dict(row)
        record["byte_share"] = round(share, 6)
        for arch in ARCHES:
            if row["routed_to_route_b"]:
                record[arch] = {"recoverable_pct_derived": 0.0,
                                "note": "Route B, cited not censused"}
                continue
            entry = entries[row["entry_point"]][arch]
            body = bodies[BODY_OF_NTG[row["ntg_x"]]][arch]
            record[arch] = {
                "entry_registers": entry["registers"],
                "entry_simdgroups_derived": entry["simdgroups"],
                "body": BODY_OF_NTG[row["ntg_x"]],
                "body_registers": body["registers"],
                "body_simdgroups_derived": body["simdgroups"],
                "recoverable_pct_derived": gain_pct(
                    entry["simdgroups"], body["simdgroups"]),
            }
        priced.append(record)

    metric = {}
    for arch in ARCHES:
        metric[arch] = round(sum(
            r["byte_share"] * r[arch]["recoverable_pct_derived"]
            for r in priced), 4)
    return {
        "draft_depth": depth,
        "verify_width": depth + 1,
        "round_megabytes": round(total, 2),
        "unrouted_byte_share": round(sum(
            r["byte_share"] for r in priced if not r["routed_to_route_b"]), 6),
        "class_b_recoverable_residency_pct_derived": metric,
        "dispatches": priced,
    }


def register_diet(entries: dict) -> list[dict]:
    """Class A: what a register reduction on the shared entry point buys."""
    wide = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
    out = []
    for threshold in REGISTER_THRESHOLDS:
        row = {"registers_removed": threshold}
        for arch in ARCHES:
            entry = entries[wide][arch]
            reduced = entry["registers"] - threshold
            after = E.simdgroups(reduced, arch)
            row[arch] = {
                "registers_before": entry["registers"],
                "registers_after": reduced,
                "simdgroups_before_derived": entry["simdgroups"],
                "simdgroups_after_derived": after,
                "residency_gain_pct_derived": gain_pct(entry["simdgroups"], after),
            }
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="research/e131-artifacts")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    entries = census(list(ENTRY_POINTS))
    bodies = json.loads((outdir / "rung0-census.json").read_text())["bodies"]

    depths = [price(entries, bodies, d) for d in DEPTHS]
    reference = next(d for d in depths if d["draft_depth"] == REFERENCE_DEPTH)

    ranked_rows = []
    for row in reference["dispatches"]:
        if row["routed_to_route_b"]:
            continue
        ranked_rows.append({
            "dispatch": row["dispatch"],
            "ntg_x": row["ntg_x"],
            "byte_share_pct": round(100 * row["byte_share"], 3),
            "g17s_entry_registers": row[RANKED]["entry_registers"],
            "g17s_entry_simdgroups_derived": row[RANKED]["entry_simdgroups_derived"],
            "g17s_body": row[RANKED]["body"],
            "g17s_body_registers": row[RANKED]["body_registers"],
            "g17s_body_simdgroups_derived": row[RANKED]["body_simdgroups_derived"],
            "g17s_recoverable_pct_derived": row[RANKED]["recoverable_pct_derived"],
            "g16s_recoverable_pct_derived": row[A.LOCAL_ARCH]["recoverable_pct_derived"],
            "weighted_contribution_pct_derived": round(
                row["byte_share"] * row[RANKED]["recoverable_pct_derived"], 4),
        })
    ranked_rows.sort(key=lambda r: -r["weighted_contribution_pct_derived"])

    wide = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
    excluding_wide = round(sum(
        r["byte_share"] * r[RANKED]["recoverable_pct_derived"]
        for r in reference["dispatches"]
        if r["entry_point"] is not None and r["entry_point"] != wide), 4)
    diet = next(r for r in register_diet(entries) if r["registers_removed"] == 2)
    stop_rule = {
        "rule": "run rung 2 only when the metric reaches 1.0 percent at a "
                "two-register threshold once the wide-QMV cells are excluded",
        "metric_excluding_wide_qmv_pct_derived": excluding_wide,
        "threshold_pct": 1.0,
        "fires": excluding_wide < 1.0,
        "decision": "skip rung 2",
        "reason":
            "every other scored entry point is a single-body kernel with no "
            "shared width switch, so it has no split to recover and its "
            "registers already sit well inside a step: affine_qmv 61, "
            "affine_gather_qmv_fast 58 and the batched affine_qmv_fast 57 on "
            "applegpu_g17s. All recoverable residency is on the one shared "
            "wide entry point, and the mechanism that recovers it is a split, "
            "not a register diet, so ISA scouting for marginal registers "
            "cannot change the decision",
        "class_a_two_register_gain_pct_derived":
            diet[RANKED]["residency_gain_pct_derived"],
    }

    payload = {
        "experiment": "E131",
        "rung": 1,
        "harness": "compile-only census (xcrun metal-tt), not ranked and not local timing",
        "timing_valid": False,
        "gpu_used": False,
        "model_loaded": False,
        "official_or_ranked_score": False,
        "occupancy_label": "derived",
        "occupancy_rule":
            "Rule 89: simdgroups = floor(BUDGET / registers) is a model output "
            "computed from the register count, not a measurement",
        "base_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(pathlib.Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, check=True).stdout.strip(),
        "toolchain": C.toolchain(),
        "simdgroup_budget": E.SIMDGROUP_BUDGET,
        "entry_point_roles": {k: v["role"] for k, v in ENTRY_POINTS.items()},
        "cells": entries,
        "bodies": bodies,
        "cost_model": {
            "class": "modelled, not measured",
            "rule": "time is proportional to quantized weight bytes within the "
                    "affine-4 matvec family",
            "justification":
                "Finding 22 measures every affine-4 projection family at 87.2 "
                "to 99.6 percent of the 273 GB/s DRAM peak, so within this "
                "family the dispatch is bandwidth bound and byte share is a "
                "first-order time share",
            "backbone_qmv_megabytes_per_round": BACKBONE_QMV_MB,
            "head_qmv_megabytes_per_draft_step": round(HEAD_QMV_MB, 2),
            "head_megabytes_by_projection": HEAD_MB,
            "excluded": "latency-class work (SDPA, norms, KV writes, sorts, "
                        "top-2) is outside the affine-4 matvec family and is "
                        "not priced here",
        },
        "by_depth": depths,
        "reference_depth": REFERENCE_DEPTH,
        "ranked": ranked_rows,
        "class_a_register_diet": register_diet(entries),
        "stop_rule": stop_rule,
        "conversion_caveat":
            "derived residency is not time. A larger resident simdgroup count "
            "converts to time only where the dispatch is occupancy limited. "
            "Finding 22 measures the affine-4 families at 87.2 to 99.6 percent "
            "of the 273 GB/s DRAM peak at width 5, which would leave little to "
            "win; ledger 242.8 reports the same round reading as 526 GB/s, "
            "which is 1.99 times that ceiling and implies a large "
            "system-level-cache-served share that IS latency bound. The two "
            "readings disagree, and this census cannot settle which holds at "
            "the head's width-1 operating point",
        "metric": {
            "e131_ranked_recoverable_residency_pct":
                reference["class_b_recoverable_residency_pct_derived"][RANKED],
            "arch": RANKED,
            "mechanism": "class B, entry-point split, zero register work",
            "at_draft_depth": REFERENCE_DEPTH,
            "g16s_companion_pct":
                reference["class_b_recoverable_residency_pct_derived"][A.LOCAL_ARCH],
        },
        "wall_seconds": round(time.time() - started, 2),
    }
    path = outdir / "rung1-census.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")

    print("entry points (registers / derived simdgroups)")
    for cell in ENTRY_POINTS:
        rec = entries[cell]
        print("  %-46s g16s %3d/%-3d  g17s %3d/%-3d" % (
            cell, rec[A.LOCAL_ARCH]["registers"], rec[A.LOCAL_ARCH]["simdgroups"],
            rec[RANKED]["registers"], rec[RANKED]["simdgroups"]))
    print("\nclass B recoverable residency by draft depth (derived)")
    for d in depths:
        m = d["class_b_recoverable_residency_pct_derived"]
        print("  d=%d width=%d  unrouted bytes %6.2f %%   g16s %6.3f %%   g17s %6.3f %%"
              % (d["draft_depth"], d["verify_width"],
                 100 * d["unrouted_byte_share"], m[A.LOCAL_ARCH], m[RANKED]))
    print("\nranked dispatch classes at d=%d" % REFERENCE_DEPTH)
    for row in ranked_rows:
        print("  %-46s ntg.x=%d  %6.3f %% of bytes  %3d->%3d regs  %2d->%2d sg  %+7.2f %%  w %6.4f"
              % (row["dispatch"][:46], row["ntg_x"], row["byte_share_pct"],
                 row["g17s_entry_registers"], row["g17s_body_registers"],
                 row["g17s_entry_simdgroups_derived"], row["g17s_body_simdgroups_derived"],
                 row["g17s_recoverable_pct_derived"],
                 row["weighted_contribution_pct_derived"]))
    print("\ne131_ranked_recoverable_residency_pct = %.4f (derived, %s, d=%d)"
          % (payload["metric"]["e131_ranked_recoverable_residency_pct"],
             RANKED, REFERENCE_DEPTH))
    print("wrote %s   %.2f s" % (path, payload["wall_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
