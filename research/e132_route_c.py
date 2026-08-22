#!/usr/bin/env python3
"""E132 rung 3: Route C narrow entry point, design and price. And question 3.

Every steady-state MTP head projection launches at `ntg.x == 1`. It misses both
switches in `affine_qmv_fast`, falls to `default: break` at `quantized.h:2022`
and runs `qmv_fast_impl`, which needs far fewer registers than the entry point
that contains it allocated. Route C is a narrow `MLXFast.metalKernel` holding
only that body, routed from Swift at `m == 1`.

DESIGN ONLY. This module measures, prices and emits sources. It writes nothing
into `Qwen35.swift`, which thorfinn owns this round.

Two candidate bodies are censused:

  C1  `qmv_fast_impl<bfloat16_t, 64, 4>` EXTRACTED VERBATIM from `quantized.h`.
      Only the two `const constant int&` parameters become `const int`, because
      an `MLXFast.metalKernel` body reads shapes from `x_shape` and `w_shape`
      in the thread address space. Every arithmetic line is byte-identical to
      the shipped source, so bit-exactness is a property of the extraction and
      not of a transcription this file could get wrong.
  C2  `qwen_e120_qmv_wide<1, false>` from the shipped Route B header, which
      needs no new kernel text at all.

No GPU, no model, no timing. Every simdgroup figure is derived under Rule 89.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402
import e120_g17s_census as e120  # noqa: E402
import e131_cliff_gate as gate  # noqa: E402
import e131_kernel_sources as ks  # noqa: E402
from e123_arms import SIMDGROUP_BUDGET  # noqa: E402
from jit_string_compile import assemble, host_name  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "research/e132-artifacts"
QUANTIZED_H = ("Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
               "quantized.h")
ARCHES = (agx.LOCAL_ARCH, agx.RANKED_ARCH)
RANKED = agx.RANKED_ARCH

STOCK_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
C1_NAME = "qwen35_custom_affine4_g64_qmv_narrow_v1"
C2_NAME = "qwen35_custom_affine4_g64_qmv_wide1_v1"

# `Qwen35.swift` Qwen35CustomQMV.matmul binding order.
QMV_INPUTS = [("w", "uint32_t"), ("scales", "bfloat16_t"),
              ("biases", "bfloat16_t"), ("x", "bfloat16_t")]
QMV_OUTPUTS = [("y", "bfloat16_t")]

# Verbatim slices of `quantized.h`, taken by anchor so a source move is a hard
# failure rather than a silently wrong kernel.
SLICES = (
    ("#define MLX_MTL_CONST", "\ntemplate <typename T, typename U"),
    ("template <typename T, typename U, int values_per_thread, int bits>\n"
     "inline U load_vector(", "\ntemplate <typename T, typename U, int "
     "values_per_thread, int bits>\ninline U load_vector_safe("),
    ("template <typename U, int values_per_thread, int bits>\ninline U qdot(",
     "\ntemplate <typename U, int values_per_thread, int bits>\ninline U "
     "qdot_safe("),
    ("template <typename T, int group_size, int bits>\nMETAL_FUNC void "
     "qmv_fast_impl(", "\n// Exact-order affine4/g64 multi-row QMV."),
)

# F13's local head law and work-class transfer table, ledger 243.6.
HEAD_LOCAL_BASE_US = 2560.0
HEAD_LOCAL_MARGINAL_US = 2226.5
HEAD_LOCAL_ROUND_US = 10090.0
HEAD_RANKED_ROUND_US = 1019.0
RANKED_ROUND_US = 55870.0
LOCAL_ROUND_US = 131024.0
# F13's "transfer" column is RELATIVE to the 2.345x machine ratio, not a time
# multiplier: 0.237 = (1019 / 10090) x 2.345. Using 0.237 as a time multiplier
# understates how fast F13 claims the ranked head is, by 2.345x.
MACHINE_RATIO = LOCAL_ROUND_US / RANKED_ROUND_US
HEAD_ABSOLUTE_RATIO = HEAD_RANKED_ROUND_US / HEAD_LOCAL_ROUND_US
HEAD_TRANSFER = HEAD_ABSOLUTE_RATIO * MACHINE_RATIO

# `research/e131-artifacts/rung1-census.json` cost_model, the figure the
# advisor flagged. These are the affine-4 g64 projections that reach
# `affine_qmv_fast<bfloat16_t, 64, 4, false>` at `ntg.x == 1`.
HEAD_AFFINE4_BYTES = {
    "mtp.fc": 29_491_200,
    "mtp.self_attn.q_proj": 35_389_440,
    "mtp.self_attn.o_proj": 17_694_720,
    "mtp.mlp.gate_up": 100_270_080,
    "mtp.mlp.down_proj": 50_135_040,
}
# Ledger 234.9 declared head budget.
HEAD_DECLARED_BYTES = 427_738_112
HEAD_NON_COARSE_BYTES = 270_400_512
HEAD_DENSE_COARSE_BYTES = 157_337_600
HEAD_CLUSTER_COARSE_BYTES = 59_001_600
# Ledger 24505, thorfinn's standalone bench of the dense coarse read.
COARSE_BENCH_US = 820.3
COARSE_BENCH_GBPS = 191.8
# FACT 33, edward E92: beagle's measured read-only ceiling.
BEAGLE_READ_CEILING_GBPS = 265.0

# Runbook frame for the entry-point leg. F131 measured the same coefficient
# directly on two independent designs.
C_MODELLED = 0.445
C_MEASURED = (-0.0014, 0.0105)


def slice_verbatim(text: str, start: str, end: str) -> str:
    at = text.find(start)
    if at < 0:
        raise SystemExit("quantized.h: anchor not found: %r" % start[:48])
    stop = text.find(end, at)
    if stop < 0:
        raise SystemExit("quantized.h: end anchor not found: %r" % end[:48])
    return text[at:stop].rstrip() + "\n"


def c1_library(rev: str | None) -> tuple[str, str]:
    """A self-contained Route C library, and the extracted `qmv_fast_impl`."""
    text = ks.swift_text(QUANTIZED_H, rev)
    parts = [e120.PRELUDE, "#define METAL_FUNC inline", ""]
    impl = ""
    for start, end in SLICES:
        chunk = slice_verbatim(text, start, end)
        if "qmv_fast_impl(" in start:
            # An MLXFast body reads the shapes into thread-space locals, so the
            # two `constant` references become values. Nothing else changes.
            for old, new in (("const constant int& in_vec_size,",
                              "const int in_vec_size,"),
                             ("const constant int& out_vec_size,",
                              "const int out_vec_size,")):
                if chunk.count(old) != 1:
                    raise SystemExit("qmv_fast_impl: %r is not unique" % old)
                chunk = chunk.replace(old, new)
            impl = chunk
        parts.append(chunk)
    body = """
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    qmv_fast_impl<bfloat16_t, 64, 4>(
        w, scales, biases, x, y, qmv_k, qmv_n,
        threadgroup_position_in_grid,
        simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""
    parts.append(e120.generate(C1_NAME, QMV_INPUTS, QMV_OUTPUTS, body))
    return "\n".join(parts) + "\n", impl


def c2_library(rev: str | None) -> str:
    """Route C built from the shipped Route B header at NA=1."""
    swift = ks.swift_text(ks.QWEN35, rev)
    header = ks.named_literal(swift, "qwen35E120QMVHeader")
    body = """
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    qwen_e120_qmv_wide<1, false>(
        w, scales, biases, x, nullptr, y,
        qmv_k, qmv_n, 8, 0, qmv_out_row, qmv_lid);
"""
    parts = [e120.PRELUDE, header, "",
             e120.generate(C2_NAME, QMV_INPUTS, QMV_OUTPUTS, body)]
    return "\n".join(parts) + "\n"


def route_c_census(rev: str | None) -> dict:
    c1_source, impl = c1_library(rev)
    c2_source = c2_library(rev)
    cells = {
        host_name(STOCK_CELL): {
            "library": "stock", "source": assemble((STOCK_CELL,), rev),
            "source_form": "jit_twin",
            "role": "shipped entry point the head reaches at ntg.x == 1"},
        C1_NAME: {
            "library": "c1", "source": c1_source,
            "source_form": "swift_metal_kernel",
            "role": "Route C C1: qmv_fast_impl extracted verbatim"},
        C2_NAME: {
            "library": "c2", "source": c2_source,
            "source_form": "swift_metal_kernel",
            "role": "Route C C2: shipped Route B header at NA=1"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        rows = gate.census(cells, pathlib.Path(tmp), "routec")
    return {"cells": rows, "c1_source": c1_source, "c2_source": c2_source,
            "extracted_impl_bytes": len(impl.encode())}


def price(stock: dict, narrow: dict, head_share: float) -> dict:
    """Residency-only price of Route C, with `c` a free parameter.

    Route C deletes no work per output element: the entry point it replaces
    reaches the same `qmv_fast_impl` after a handful of scalar compares that
    run once per thread, not once per k block. The only mechanism is the
    register count of the containing function, so the whole price is
    `c x delta-residency x head share` and it is zero at `c = 0`.
    """
    base_sg = stock["simdgroups_derived"]
    cand_sg = narrow["simdgroups_derived"]
    delta_pct = 100.0 * (cand_sg - base_sg) / base_sg
    def at(c: float) -> float:
        return c * delta_pct * head_share / 100.0
    return {
        "base_registers": stock["registers"],
        "candidate_registers": narrow["registers"],
        "base_simdgroups_derived": base_sg,
        "candidate_simdgroups_derived": cand_sg,
        "residency_change_pct_derived": round(delta_pct, 2),
        "head_ranked_share_pct": head_share,
        "deleted_instructions_per_output_element": 0.0,
        "ranked_gain_pct_at_c_0": 0.0,
        "ranked_gain_pct_at_c_measured_low": round(at(C_MEASURED[0]), 4),
        "ranked_gain_pct_at_c_measured_high": round(at(C_MEASURED[1]), 4),
        "ranked_gain_pct_at_c_modelled_0445": round(at(C_MODELLED), 4),
        "c_frame": "entry-point leg, runbook c = 0.445 [0.139, 0.819]",
        "c_measured": "F131, two independent designs: %s" % (C_MEASURED,),
        "gain_pct_per_point_of_head_share_at_c_0445":
            round(C_MODELLED * delta_pct / 100.0, 4),
        "head_share_note":
            "The price is linear in the head share, so question 3 rescales it "
            "directly. The assignment's 1.82 %% is used here as instructed; "
            "at the %.2f %% the byte law forces at a 546 GB/s ranked host the "
            "c = 0.445 figure becomes %+.4f %%."
            % (6.35, C_MODELLED * delta_pct * 6.35 / 100.0),
    }


def question3() -> dict:
    """Reconcile 232.96 MB per draft step against F13's 1019 us per round."""
    affine4 = sum(HEAD_AFFINE4_BYTES.values())
    mb = 1_000_000.0

    # The naive ratio the advisor flagged.
    steps = 1.0 + (HEAD_LOCAL_ROUND_US - HEAD_LOCAL_BASE_US) / \
        HEAD_LOCAL_MARGINAL_US
    naive_us = HEAD_RANKED_ROUND_US / steps
    naive_gbps = affine4 / naive_us / 1000.0

    # The local law closes against an independent bench.
    local_gbps = HEAD_DECLARED_BYTES / HEAD_LOCAL_MARGINAL_US / 1000.0
    cluster_local_gbps = (HEAD_NON_COARSE_BYTES + HEAD_CLUSTER_COARSE_BYTES) \
        / HEAD_LOCAL_MARGINAL_US / 1000.0
    bench_gbps = HEAD_DENSE_COARSE_BYTES / COARSE_BENCH_US / 1000.0

    # The same law under F13's own ranked head figure.
    ranked_marginal_us = HEAD_LOCAL_MARGINAL_US * HEAD_ABSOLUTE_RATIO
    ranked_gbps = HEAD_DECLARED_BYTES / ranked_marginal_us / 1000.0

    # The campaign has no measured ranked read bandwidth, so invert the law
    # instead of assuming one. For each candidate ranked rate, report the head
    # transfer coefficient and the ranked head share it forces.
    inversion = []
    for rate in sorted({200.0, 273.0, BEAGLE_READ_CEILING_GBPS, 400.0, 546.0,
                        round(ranked_gbps, 1)}):
        ratio = local_gbps / rate
        inversion.append({
            "ranked_read_gigabytes_per_second": rate,
            "implied_head_absolute_ratio": round(ratio, 3),
            "implied_f13_transfer_column": round(ratio * MACHINE_RATIO, 3),
            "implied_ranked_head_us_per_round":
                round(HEAD_LOCAL_ROUND_US * ratio, 0),
            "implied_ranked_head_share_pct":
                round(100.0 * HEAD_LOCAL_ROUND_US * ratio /
                      RANKED_ROUND_US, 2),
        })
    # Reported alongside, under one stated assumption rather than as the claim.
    floor_transfer = local_gbps / (2.0 * BEAGLE_READ_CEILING_GBPS)
    implied_share = 100.0 * (HEAD_LOCAL_ROUND_US * floor_transfer) / \
        RANKED_ROUND_US
    return {
        "flagged_figure_megabytes": round(affine4 / mb, 2),
        "flagged_figure_is": (
            "the affine-4 g64 subset that reaches "
            "affine_qmv_fast<bfloat16_t, 64, 4, false> at ntg.x == 1, "
            "not the whole head"),
        "affine4_projection_bytes": HEAD_AFFINE4_BYTES,
        "head_declared_bytes": HEAD_DECLARED_BYTES,
        "affine4_share_of_head_pct": round(100.0 * affine4 /
                                           HEAD_DECLARED_BYTES, 1),
        "draft_steps_implied_by_f13_local_law": round(steps, 3),
        "naive_ratio": {
            "numerator_megabytes": round(affine4 / mb, 2),
            "denominator_us": round(naive_us, 1),
            "implied_gigabytes_per_second": round(naive_gbps, 1),
            "three_errors": [
                "a per-draft-step byte count divided by a per-ROUND head "
                "time; F13's local law gives the marginal draft step as "
                "%.1f us, not head_us / d" % HEAD_LOCAL_MARGINAL_US,
                "a subset byte count treated as the whole head",
                "a local byte count divided by a RANKED time",
            ],
        },
        "local_check": {
            "head_marginal_us_per_draft_step": HEAD_LOCAL_MARGINAL_US,
            "declared_head_gigabytes_per_second": round(local_gbps, 1),
            "with_cluster_index_gigabytes_per_second":
                round(cluster_local_gbps, 1),
            "independent_bench_gigabytes_per_second": round(bench_gbps, 1),
            "bench_source": "ledger 24505, dense coarse read 157,337,600 B "
                            "in %.1f us" % COARSE_BENCH_US,
            "agreement_pct": round(100.0 * (local_gbps - bench_gbps) /
                                   bench_gbps, 2),
            "verdict": "the byte count is right; the local law and an "
                       "independent bench agree to better than 1 %",
        },
        "residual": {
            "f13_transfer_column_is_relative": (
                "0.237 = (1019 / 10090) x %.3f. The column is normalised by "
                "the machine ratio, so it is not a time multiplier. The "
                "absolute head ratio F13 claims is %.3f, meaning M5 does the "
                "head %.1fx faster than beagle."
                % (MACHINE_RATIO, HEAD_ABSOLUTE_RATIO,
                   1.0 / HEAD_ABSOLUTE_RATIO)),
            "ranked_marginal_us_per_draft_step": round(ranked_marginal_us, 1),
            "implied_ranked_gigabytes_per_second": round(ranked_gbps, 1),
            "times_beagle_read_ceiling":
                round(ranked_gbps / BEAGLE_READ_CEILING_GBPS, 2),
            "suspect_quantity": "F13's 1019 us ranked head figure, and "
                                "therefore the 1.82 % ranked head share",
            "head_transfer_floor_if_ranked_host_is_at_most_2x_beagle":
                round(floor_transfer, 3),
            "implied_minimum_ranked_head_share_pct": round(implied_share, 2),
            "inversion": inversion,
            "inversion_note":
                "The campaign has no measured ranked read bandwidth, so this "
                "inverts the law rather than assuming a rate. F13's 1019 us "
                "is only reachable if the ranked host sustains %.0f GB/s on "
                "this work. The ranked round total is held at %.0f us, so "
                "each row also moves work out of the other two classes."
                % (ranked_gbps, RANKED_ROUND_US),
        },
        "consequence_for_route_c": (
            "Do not price Route C from bytes at all. Rule 94 forbids a "
            "bandwidth price on the QMV family, and Route C removes no bytes "
            "and no instructions per output element. Its only mechanism is "
            "the register count of the containing entry point."),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rev", default=None,
                        help="read sources at this git revision")
    parser.add_argument("--emit", type=pathlib.Path,
                        help="write the two candidate Metal sources here")
    args = parser.parse_args()

    started = time.time()
    censused = route_c_census(args.rev)
    rows = censused["cells"]
    stock = rows[host_name(STOCK_CELL)]

    prices = {}
    for name in (C1_NAME, C2_NAME):
        narrow = rows[name].get(RANKED, {})
        if "error" in narrow:
            prices[name] = {"error": narrow["error"]}
            continue
        prices[name] = price(stock[RANKED], narrow,
                             HEAD_RANKED_ROUND_US / RANKED_ROUND_US * 100.0)

    receipt = {
        "schema_version": 1,
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "harness": "compile_only",
        "occupancy_label": "derived",
        "occupancy_rule": "Rule 89",
        "status": "DESIGN ONLY, nothing implemented in a shipped file",
        "tool": "research/e132_route_c.py",
        "toolchain": gate.toolchain(),
        "rev": args.rev or "working tree",
        "cells": rows,
        "price": prices,
        "dispatch": {
            "source": "quantized.cpp:251-254, qmv()",
            "host_at_m1": "bn = 8, bk = 32, group_dims(32, 2, 1), "
                          "grid_dims(1, (N + 7) / 8, B)",
            "mlxfast_equivalent": "grid: (32, (n / 8) * 2, 1), "
                                  "threadGroup: (32, 2, 1)",
            "note": "MLXFast.metalKernel takes the grid in THREADS and MLX "
                    "divides by the threadgroup, so (32, (n/8)*2, 1) over "
                    "(32, 2, 1) is exactly (1, n/8, 1) threadgroups.",
        },
        "bit_exactness": {
            C1_NAME: (
                "The three helpers and qmv_fast_impl are sliced verbatim out "
                "of quantized.h by anchor. The only edit is the two "
                "`const constant int&` parameters becoming `const int`, "
                "because an MLXFast body reads shapes into thread space. No "
                "arithmetic line is retyped, so exactness is a property of "
                "the extraction and this file asserts both substitutions are "
                "unique before it compiles."),
            C2_NAME: (
                "qwen_e120_qmv_wide<1, false> inherits the shipped Route B "
                "exactness argument unchanged. Per output element: the "
                "grouped-by-four x sum matches load_vector's "
                "`sum += x[i]+x[i+1]+x[i+2]+x[i+3]`; the four grouped "
                "products match qdot's accumulation, with qdot's "
                "`(x/2^p) * (2^p * nibble)` equal to `x * nibble` because a "
                "power-of-two scaling is exact; the per-k-block update "
                "`scale*partial + sums*bias` matches `scale*accum + bias*sum` "
                "by the commutativity of IEEE-754 multiplication; and the "
                "weight, scale and bias addressing is identical. The m axis "
                "is independent, so NA=1 needs no argument the shipped "
                "NA>=3 instantiations do not already carry."),
        },
        "question3": question3(),
        "runtime_seconds": round(time.time() - started, 2),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "rung3-route-c.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.emit:
        args.emit.mkdir(parents=True, exist_ok=True)
        (args.emit / "route-c1.metal").write_text(censused["c1_source"])
        (args.emit / "route-c2.metal").write_text(censused["c2_source"])

    print("E132 rung 3: Route C narrow entry point, DESIGN ONLY")
    print("%-46s %-14s %s" % ("entry point", "g16s", "g17s"))
    for name, row in rows.items():
        cols = []
        for arch in ARCHES:
            cell = row.get(arch, {})
            cols.append(cell["error"][:40] if "error" in cell else
                        "%3d/%-4d %2d sg" % (cell["registers"],
                                             cell["spill_bytes"],
                                             cell["simdgroups_derived"]))
        print("%-46s %-14s %s" % (name[:46], cols[0], cols[1]))

    print("\nprice, ranked, residency only, head share %.2f %%"
          % (HEAD_RANKED_ROUND_US / RANKED_ROUND_US * 100.0))
    for name, row in prices.items():
        if "error" in row:
            print("  %-44s %s" % (name[:44], row["error"][:60]))
            continue
        print("  %-44s %+6.2f %% residency -> %+.4f %% at c=0.0105, "
              "%+.4f %% at c=0.445, 0 at c=0"
              % (name[:44], row["residency_change_pct_derived"],
                 row["ranked_gain_pct_at_c_measured_high"],
                 row["ranked_gain_pct_at_c_modelled_0445"]))

    q3 = receipt["question3"]
    print("\nquestion 3")
    print("  flagged %.2f MB is %.1f %% of the declared head, not the head"
          % (q3["flagged_figure_megabytes"], q3["affine4_share_of_head_pct"]))
    print("  local law %.1f GB/s vs independent bench %.1f GB/s (%.2f %%)"
          % (q3["local_check"]["declared_head_gigabytes_per_second"],
             q3["local_check"]["independent_bench_gigabytes_per_second"],
             q3["local_check"]["agreement_pct"]))
    print("  F13's 1019 us ranked head needs %.0f GB/s, %.2fx beagle's "
          "measured read ceiling"
          % (q3["residual"]["implied_ranked_gigabytes_per_second"],
             q3["residual"]["times_beagle_read_ceiling"]))
    print("  so the suspect quantity is the ranked head figure, not the byte "
          "count. Inverting the law:")
    print("    %-14s %-10s %-10s %-12s %s"
          % ("ranked GB/s", "abs ratio", "F13 col", "head us/round",
             "ranked head share"))
    for row in q3["residual"]["inversion"]:
        print("    %-14.1f %-10.3f %-10.3f %-12.0f %.2f %%"
              % (row["ranked_read_gigabytes_per_second"],
                 row["implied_head_absolute_ratio"],
                 row["implied_f13_transfer_column"],
                 row["implied_ranked_head_us_per_round"],
                 row["implied_ranked_head_share_pct"]))
    print("\nwrote %s in %.2f s" % (path, receipt["runtime_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
