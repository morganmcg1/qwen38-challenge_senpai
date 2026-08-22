#!/usr/bin/env python3
"""E131 rung 1b: verify E131-F4's head routing table with my own instrument.

F4 hands over a ten-row claim table and asks for agreement or disagreement per
row, with my own reading cited. It also asks for three things the rung 1 census
did not carry:

  * `qmv_fast_impl<T, 64, 4>` as a FIRST-CLASS cell rather than an inlined body;
  * the metric under the assignment's LITERAL definition, which prices a
    register diet of two or fewer registers, beside the class B split value
    rung 1 published;
  * the g16s / g17s bandwidth-saturation bound as a bound on what residency can
    pay, explicitly not a reopening of F114.

Everything here is compile-only or a read of an already-committed artifact. No
GPU, no model load, no timing. Every simdgroup figure is `derived` (Rule 89).
"""

from __future__ import annotations

import argparse
import array
import json
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402
import e131_census as C  # noqa: E402
import e131_rung1 as R1  # noqa: E402
from e123_arms import SIMDGROUP_BUDGET, simdgroups  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHES = (agx.LOCAL_ARCH, agx.RANKED_ARCH)
RANKED = agx.RANKED_ARCH
HEAD = (pathlib.Path.home()
        / ".cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared/model.safetensors")
QWEN35 = ROOT / "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
QWEN35MTP = ROOT / "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift"
E116 = ROOT / "research/e116-artifacts/rung4-qmv-share-512.json"
RUNG0 = ROOT / "research/e131-artifacts/rung0-census.json"

# Affine-2 draft readout bytes per draft step, from the declared head header.
# 2-bit packs K/16 uint32 words per row; scales and biases are one BF16 each
# per group of 64.
CENTROID_ROWS = 12292
LEAF_ROWS = 24584           # 3073 probed leaves x 8 rows
RERANK_ROWS = 32
K = 5120


def affine2_mb(rows: int) -> float:
    words = rows * (K // 16) * 4
    groups = rows * (K // 64) * 2 * 2
    return (words + groups) / 1e6


def affine4_mb(rows: int) -> float:
    words = rows * (K // 8) * 4
    groups = rows * (K // 64) * 2 * 2
    return (words + groups) / 1e6


def head_header() -> dict:
    with open(HEAD, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def island_indices(name: str) -> list[int]:
    with open(HEAD, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        start, end = header[name]["data_offsets"]
        f.seek(8 + n + start)
        values = array.array("i")
        values.frombytes(f.read(end - start))
    return list(values)


def complete_permutation(values: list[int], count: int) -> bool:
    """`Qwen35Attention.isCompletePermutation`, Qwen35.swift:3008-3021."""
    return len(values) == count and sorted(values) == list(range(count))


def islands() -> dict:
    """Rows 3 and 4: does the K/V affine-4 pack ever run?"""
    header = head_header()
    out: dict = {"tensors": {}, "checks": {}}
    for part in ("q", "k", "v"):
        for kind in ("weight", "indices"):
            name = "precision_islands.%s.%s" % (part, kind)
            out["tensors"][name] = {"dtype": header[name]["dtype"],
                                    "shape": header[name]["shape"]}
    outputs = {"q": 12288, "k": 1024, "v": 1024}
    for part, count in outputs.items():
        values = island_indices("precision_islands.%s.indices" % part)
        out["checks"][part] = {
            "output_count": count,
            "index_count": len(values),
            "distinct": len(set(values)),
            "min": min(values),
            "max": max(values),
            "is_complete_permutation": complete_permutation(values, count),
        }
    kv_dense = 2 * 1024 * K * 2 / 1e6
    kv_affine4 = 2 * affine4_mb(1024)
    out["bytes_per_draft_step_megabytes"] = {
        "kv_dense_bf16_actually_streamed": round(kv_dense, 3),
        "kv_affine4_if_the_pack_ran": round(kv_affine4, 3),
        "delta": round(kv_dense - kv_affine4, 3),
    }
    out["verdict"] = (
        "AGREE with F4 rows 3 and 4. K and V index sets are complete "
        "permutations of 0...1023, so `installExactQKVRows` takes the "
        "`_exactKVDenseW` branch and K/V are one dense BF16 matmul. Q's index "
        "set is 1024 rows of a 12288-row output, so it is NOT a permutation "
        "and q_proj keeps the affine-4 path.")
    return out


def routing() -> dict:
    """Rows 1, 2, 5, 6, 7: what wrapper does each head projection enter?"""
    mtp = QWEN35MTP.read_text()
    swift = QWEN35.read_text()
    fc_sites = [mtp[:m.start()].count("\n") + 1
                for m in re.finditer(r"= fc\(\n", mtp)]
    return {
        "mtp_swift_routed_call_sites": len(re.findall(r"qwen35Routed", mtp)),
        "mtp_swift_uses_backbone_classes": {
            "Qwen35Attention": "Qwen35Attention" in mtp,
            "Qwen35FusedMLP": "Qwen35FusedMLP" in mtp,
        },
        "fc_declaration": "@ModuleInfo(key: \"fc\") var fc: Linear",
        "fc_call_lines": fc_sites,
        "fc_enters_a_routed_wrapper": False,
        "wrapper_of": {
            "q_proj": "qwen35RoutedQuantizedMM, Qwen35.swift:2870 island fast "
                      "path and :2890 otherwise",
            "o_proj": "qwen35RoutedLinear, Qwen35.swift:1322",
            "mlp gate+up fused": "qwen35RoutedQuantizedMM, Qwen35.swift:1907 "
                                 "through Qwen35FusedMLP.fusedGateUp",
            "mlp down_proj": "qwen35RoutedLinear, Qwen35.swift:1944 and :1946",
            "fc": "none",
        },
        "routable_guards": {
            "source": "Qwen35CustomQMV.routable, Qwen35.swift:1740-1766",
            "bits": 4, "group_size": 64, "mode": "affine",
            "n_mod_8": 0, "k_mod_512": 0, "n_min": 4096,
            "widths": "3...9",
        },
        "why_each_head_projection_is_rejected": {
            "q_proj": "m == 1 only; K=5120, N=12288 clear every other guard",
            "o_proj": "m == 1 only; K=6144, N=5120 clear every other guard",
            "gate_up": "m == 1 only; K=5120, N=34816 clear every other guard",
            "down_proj": "m == 1 only; K=17408, N=5120 clear every other guard",
        },
        "verdict": (
            "AGREE with F4's correction of F121, with one refinement. "
            "`Qwen35MTP.swift` has zero `qwen35Routed*` call sites, but the "
            "head's attention and MLP ARE the backbone classes, so four of the "
            "five quantized head projections do enter a routed wrapper and are "
            "rejected for `m == 1` alone. The wrapper is "
            "`qwen35RoutedQuantizedMM` for q_proj and the fused gate+up, not "
            "`qwen35RoutedLinear`. My rung 1 comment endorsed the F121 framing "
            "and was wrong in the same way; the dispatched kernel and the "
            "`ntg.x == 1` finding are unchanged."),
    }


def readout() -> dict:
    """Rows 8 and 9: which kernel does the draft readout actually reach?"""
    return {
        "fast_rule": "bool fast = N % bn == 0 && K % 512 == 0, "
                     "mlx/backend/metal/quantized.cpp:259, bn = 8",
        "singlerow_affine2_guard":
            "group_size == 64 && bits == 2 && out_vec_size == 98336 && "
            "ntg.x == 1, quantized.h:1909-1917",
        "centroid_score": {
            "K": K, "N": CENTROID_ROWS,
            "n_mod_8": CENTROID_ROWS % 8,
            "fast": CENTROID_ROWS % 8 == 0 and K % 512 == 0,
            "entry_point": "affine_qmv<bfloat16_t, 64, 2, false>",
            "f4_claim": "qmv_fast_singlerow_affine2_g64",
            "verdict": "DISAGREE with F4 row 8 on the kernel reached. "
                       "12292 %% 8 == %d, so the dispatch is NOT fast and "
                       "takes the non-fast `affine_qmv`. The singlerow body "
                       "is also gated on out_vec_size == 98336, which 12292 "
                       "is not, so it could not be reached even if the "
                       "dispatch were fast."
                       % (CENTROID_ROWS % 8),
        },
        "leaf_row_score": {
            "K": K, "rows": LEAF_ROWS, "leaves": LEAF_ROWS // 8,
            "entry_point": "affine_gather_qmv_fast<bfloat16_t, 64, 2>",
            "verdict": "AGREE with F4 row 9. The gather primitive is a "
                       "separate entry point; it is fast because the gathered "
                       "N per leaf is 8 and K is 5120.",
        },
        "compact_readout_98336": {
            "entry_point": "affine_qmv_fast<bfloat16_t, 64, 2, false>",
            "live": False,
            "verdict": "The 98336 affine-2 entry point is the DENSE FALLBACK "
                       "only. mtp-head.manifest.json states the shipped path "
                       "derives a two-level index and scores 12292 centroids "
                       "then 24584 gathered rows, so the fallback is dead in "
                       "the shipped configuration. The affine-2 TENSORS are "
                       "live; the 98336 ENTRY POINT is not.",
        },
        "megabytes_per_draft_step": {
            "centroid_score": round(affine2_mb(CENTROID_ROWS), 3),
            "leaf_row_score": round(affine2_mb(LEAF_ROWS), 3),
            "rerank_32_rows_affine4": round(affine4_mb(RERANK_ROWS), 3),
        },
    }


def impl_cell() -> dict:
    """F4 section 3: `qmv_fast_impl<T, 64, 4>` as a first-class cell."""
    cells = C.iso_kernels()
    host = "e131_iso_impl4"
    with tempfile.TemporaryDirectory() as raw:
        workdir = pathlib.Path(raw)
        library = C.emit_preamble(workdir) + cells["impl4"]
        rows = C.census_library(library, workdir, "impl4", lambda n: n == host)
    record = rows[host]
    for arch in ARCHES:
        record[arch]["simdgroups_derived"] = record[arch].pop("simdgroups")
    record["cell"] = "qmv_fast_impl<bfloat16_t, 64, 4>"
    record["inlined_bodies"] = ["qmv_fast_impl only: it is a leaf, it calls no "
                                "other body and it carries no width switch"]
    record["reached_from"] = ("`default: break` at quantized.h:2022, taken by "
                              "every dispatch with ntg.x == 1, which is every "
                              "steady-state MTP head projection")
    record["standalone_note"] = (
        "This is what a narrow entry point containing only this body would "
        "allocate. Inside the shipped shared function the body's own register "
        "count is NOT what runs: the function allocates the maximum over all "
        "inlined bodies, so these dispatches run at the entry point's count.")
    return record


def s_distribution() -> dict:
    """Row 1: does `mtp.fc`'s flush dispatch clear Route B's 3...9 guard?"""
    data = json.loads(E116.read_text())
    by_width = data["rounds_by_width"]
    mtp = {int(k[1:]): v for k, v in by_width.items() if int(k[1:]) >= 2}
    rounds = sum(mtp.values())
    mean = sum(w * n for w, n in mtp.items()) / rounds
    clears = sum(n for w, n in mtp.items() if 3 <= w <= 9)
    return {
        "source": "research/e116-artifacts/rung4-qmv-share-512.json, "
                  "rounds_by_width, local 512-token public fixture",
        "harness": "local",
        "mtp_round_width_histogram": mtp,
        "mtp_rounds": rounds,
        "mean_width": round(mean, 3),
        "rounds_clearing_route_b_widths": clears,
        "share_clearing": round(clears / rounds, 4),
        "verdict": (
            "AGREE with F4 row 1 that the flush dispatch clears the guard on "
            "most rounds: S = 1 + accepted_prev equals the next round's verify "
            "width, and %d of %d MTP rounds land in 3...9 on this fixture. "
            "Two ambiguities I will not paper over: this is the LOCAL fixture, "
            "not the ranked prompt pool, and `fc` also runs once per SEQUENTIAL "
            "draft at S == 1, so the per-round dispatch mix is one flush "
            "dispatch at S plus d dispatches at 1. No committed trace carries "
            "a per-projection ntg.x ledger, so the split between the two forms "
            "is read from the call sites, not measured."
            % (clears, rounds)),
    }


def metrics(entries: dict, bodies: dict, depth: int) -> dict:
    """The literal (class A) metric beside rung 1's class B split value."""
    rung1 = R1.price(entries, bodies, depth)
    wide = "affine_qmv_fast<bfloat16_t, 64, 4, false>"

    # Rung 1's denominator is the affine-4 QMV byte frame. Add the affine-2
    # draft readout and the dense K/V island so the sensitivity is visible.
    extra = depth * (affine2_mb(CENTROID_ROWS) + affine2_mb(LEAF_ROWS)
                     + affine4_mb(RERANK_ROWS) + 2 * 1024 * K * 2 / 1e6)
    denominator = rung1["round_megabytes"] + extra

    def next_step(registers: int, arch: str) -> tuple[int, int, float]:
        budget = SIMDGROUP_BUDGET[arch]
        current = simdgroups(registers, arch)
        target = budget // (current + 1)
        return target, registers - target, 100.0 / current

    shares = {
        wide: depth * R1.HEAD_QMV_MB / denominator,
        "affine_qmv<bfloat16_t, 64, 2, false>":
            depth * affine2_mb(CENTROID_ROWS) / denominator,
        "affine_gather_qmv_fast<bfloat16_t, 64, 2>":
            depth * affine2_mb(LEAF_ROWS) / denominator,
    }
    class_a: dict = {}
    for arch in ARCHES:
        rows = []
        for cell, share in shares.items():
            registers = entries[cell][arch]["registers"]
            target, distance, gain = next_step(registers, arch)
            rows.append({
                "cell": cell,
                "byte_share": round(share, 6),
                "registers": registers,
                "simdgroups_derived": simdgroups(registers, arch),
                "registers_for_one_more_simdgroup": target,
                "distance_registers": distance,
                "reachable_within_2": distance <= 2,
                "gain_pct_derived": round(gain, 3),
                "contribution_pct": round(
                    share * gain, 5) if distance <= 2 else 0.0,
                "owned_by_alphonse": cell == wide,
            })
        total = sum(r["contribution_pct"] for r in rows)
        excl = sum(r["contribution_pct"] for r in rows
                   if not r["owned_by_alphonse"])
        class_a[arch] = {"cells": rows,
                         "metric_pct_derived": round(total, 4),
                         "metric_excluding_wide_qmv_pct_derived": round(excl, 4)}

    class_b = rung1["class_b_recoverable_residency_pct_derived"]
    rescaled = {arch: round(class_b[arch] * rung1["round_megabytes"]
                            / denominator, 4) for arch in ARCHES}
    return {
        "draft_depth": depth,
        "affine4_qmv_frame_megabytes": rung1["round_megabytes"],
        "plus_affine2_readout_and_kv_island_megabytes": round(denominator, 2),
        "class_b_split_published_in_rung1": class_b,
        "class_b_split_on_the_wider_denominator": rescaled,
        "class_a_literal_definition": class_a,
        "stop_rule": {
            "threshold_registers": 2,
            "excluding_wide_qmv_pct_derived":
                class_a[RANKED]["metric_excluding_wide_qmv_pct_derived"],
            "fires": class_a[RANKED][
                "metric_excluding_wide_qmv_pct_derived"] < 1.0,
            "correction_to_rung_1": (
                "Rung 1 reported the excluded-cell metric as exactly 0.0 on "
                "the ground that every remaining entry point is single-body. "
                "That is the class B reading. Under the assignment's LITERAL "
                "class A definition the two affine-2 readout cells ARE within "
                "one register of a step, so the correct figure is small but "
                "not zero. The stop rule fires either way."),
        },
    }


def saturation() -> dict:
    """F4 section 4, recorded as a bound on payoff, not as a predictor."""
    return {
        "applegpu_g16s": {
            "achieved_gb_s_at_m1_target_shapes": [240.0, 250.0],
            "measured_stream_ceiling_gb_s": 227.0,
            "reading": "at or above the measured streaming ceiling, so the "
                       "path is bandwidth-saturated and extra residency "
                       "cannot pay there",
        },
        "applegpu_g17s": {
            "measured_stream_ceiling_gb_s": 542.8,
            "fraction_of_ceiling_at_the_same_achieved_bandwidth": 0.45,
            "reading": "roughly 45 % of ceiling, so residency has room to pay",
        },
        "label": "saturation bound",
        "not_f114": (
            "F114 killed achieved bandwidth as a PREDICTOR of gain. This uses "
            "the same quantity only as a BOUND on how much residency can pay, "
            "which is a different claim and does not reopen F114."),
        "rule_83": "the two architectures disagree in direction; never pool",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="research/e131-artifacts/rung1b-f4-verification.json")
    ap.add_argument("--depth", type=int, default=R1.REFERENCE_DEPTH)
    args = ap.parse_args()

    started = time.time()
    entries = R1.census(list(R1.ENTRY_POINTS))
    bodies = json.loads(RUNG0.read_text())["bodies"]

    receipt = {
        "experiment": "E131",
        "rung": "1b",
        "harness": "local",
        "timing_valid": False,
        "gpu_used": False,
        "model_loaded": False,
        "official_or_ranked_score": False,
        "occupancy_label": "derived",
        "occupancy_rule": "Rule 89",
        "base_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True, check=True).stdout.strip(),
        "toolchain": C.toolchain(),
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "f4_rows_1_2_5_6_7_routing": routing(),
        "f4_rows_3_4_precision_islands": islands(),
        "f4_rows_8_9_draft_readout": readout(),
        "f4_row_1_s_distribution": s_distribution(),
        "qmv_fast_impl_first_class_cell": impl_cell(),
        "metrics": metrics(entries, bodies, args.depth),
        "saturation_bound": saturation(),
    }
    receipt["wall_seconds"] = round(time.time() - started, 2)

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "islands_kv_complete_permutation": [
            receipt["f4_rows_3_4_precision_islands"]["checks"][p][
                "is_complete_permutation"] for p in ("q", "k", "v")],
        "impl4": {a: receipt["qmv_fast_impl_first_class_cell"][a]
                  for a in ARCHES},
        "class_a": {a: receipt["metrics"]["class_a_literal_definition"][a][
            "metric_pct_derived"] for a in ARCHES},
        "class_a_excluding_wide_qmv": {
            a: receipt["metrics"]["class_a_literal_definition"][a][
                "metric_excluding_wide_qmv_pct_derived"] for a in ARCHES},
        "class_b_rescaled": receipt["metrics"][
            "class_b_split_on_the_wider_denominator"],
        "wall_seconds": receipt["wall_seconds"],
    }, indent=2))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
