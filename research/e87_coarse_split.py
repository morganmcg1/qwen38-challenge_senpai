#!/usr/bin/env python3
"""E87 section 4 -- split the arm C coarse readout into its two stages.

The question is whether the arm C coarse pass is limited by BYTES or by
UNPACKING. Both stages read affine 2-bit group-64 weights, so a byte-limited
stage runs near the host DRAM ceiling and an unpack-limited stage does not.

This reducer is fit free. It reads command-buffer GPU intervals from an
`MLX_E80_GPU_TIME=1` census leg and reports, for each stage:

  * the ISOLATED interval, taken from `exclusive_kernels`, which holds only the
    buffers that carried exactly one kernel;
  * the WHOLE-BUFFER interval from `signatures`, which is an upper bound
    because the buffer may also carry the `argPartition` mbsort chain.

Reference points, all measured on this host family:

  * 265.0 GB/s   DRAM ceiling (edward E92 rung 1)
  * 0.21 ps      non-memory time per weight for affine-4 (MLP gate+up)
  * 0.80 ps      non-memory time per weight for the affine-2 declared
                 `draft_lm_head` (askeladd E93, 994.81 us, 158.2 GB/s)

The campaign writes those two constants as "0.21" and "0.80". Their unit is
picoseconds per weight: 37.9 us over 178.3 M weights and 401.1 us over 503.5 M
weights.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

DRAM_GB_S = 265.0
AFFINE4_PS_PER_WEIGHT = 0.21
AFFINE2_DECLARED_PS_PER_WEIGHT = 0.80
STOP_RULE_US = 60.0

CLUSTERS = 12292
ROWS_PER_CLUSTER = 8
PROBES = 3073
COMPACT_ROWS = 98336
HIDDEN = 5120

STAGE1_KEY = "affine_qmv_bfloat16_t_gs_64_b_2_batch_0 grid=1x1537x1 tg=32x2x1"
STAGE2_KEY = "affine_gather_qmv_fast_bfloat16_t_gs_64_b_2 grid=1x1x3073 tg=32x2x1"


def affine2_bytes(rows: int, cols: int) -> int:
    """Packed bytes of an affine 2-bit group-64 tensor: 16 weights per uint32,
    plus a bfloat16 scale and a bfloat16 bias for every group of 64."""
    groups = rows * cols // 64
    return rows * cols // 4 + 4 * groups


STAGE_SETS = {
    "armc": {
        # Byte derivation, stated inline so the next reader can check it.
        # Stage 1 reads every centroid once: CLUSTERS x HIDDEN affine-2 g64
        # weights. The dispatch grid 1x1537x1 covers ceil(12292/8) row blocks,
        # which confirms the row count.
        "stage1_centroids": {
            "key": STAGE1_KEY,
            "bytes": affine2_bytes(CLUSTERS, HIDDEN),
            "weights": CLUSTERS * HIDDEN,
            "shape": f"[{CLUSTERS},{HIDDEN}] affine-2 g64 dense qmv",
        },
        # Stage 2 reads ROWS_PER_CLUSTER rows for each probed cluster. The
        # dispatch grid 1x1x3073 carries the probe count directly.
        "stage2_gather": {
            "key": STAGE2_KEY,
            "bytes": affine2_bytes(PROBES * ROWS_PER_CLUSTER, HIDDEN),
            "weights": PROBES * ROWS_PER_CLUSTER * HIDDEN,
            "shape": f"[{PROBES}x{ROWS_PER_CLUSTER},{HIDDEN}] affine-2 g64 gather qmv",
        },
    },
    # The advisor-head declared dense coarse pass. One qmv over every compact
    # row, so the grid is ceil(98336/8) = 12292 row blocks.
    "declared": {
        "dense_coarse": {
            "key": "affine_qmv_fast_bfloat16_t_gs_64_b_2_batch_0 "
                   "grid=1x12292x1 tg=32x2x1",
            "bytes": affine2_bytes(COMPACT_ROWS, HIDDEN),
            "weights": COMPACT_ROWS * HIDDEN,
            "shape": f"[{COMPACT_ROWS},{HIDDEN}] affine-2 g64 dense qmv",
        },
    },
}


def steady_draft_snapshots(path: str, width_phase: str):
    rows = [json.loads(line) for line in open(path)]
    out = []
    for row in rows:
        if row.get("event") != "gputime":
            continue
        if width_phase not in row.get("by_width_phase", {}):
            continue
        if row.get("snapshot", 0) == 0:  # warmup round
            continue
        out.append(row)
    return out


def aggregate(snapshots, field: str, prefix: str):
    agg = collections.defaultdict(lambda: {"buffers": 0, "dispatches": 0, "gpu_ns": 0})
    for snap in snapshots:
        for key, value in snap.get(field, {}).items():
            if not key.startswith(prefix + "|"):
                continue
            short = key.split("|", 2)[2]
            agg[short]["buffers"] += value.get("buffers", 0)
            agg[short]["dispatches"] += value.get("dispatches", 0)
            agg[short]["gpu_ns"] += value.get("gpu_ns", 0)
    return agg


def roster(whole, exclusive, per_draft: int):
    """Every buffer signature in the phase, so no cost hides outside the stages."""
    rows = []
    for key, value in whole.items():
        # A signature is `kernel*count` joined by commas, so a buffer holds one
        # kernel exactly once only when there is a single `*1` term.
        solo = key[:-2] if key.endswith("*1") and "," not in key else None
        rows.append({
            "signature": key,
            "us_per_draft": value["gpu_ns"] / 1e3 / per_draft,
            "buffers_per_draft": value["buffers"] / per_draft,
            "dispatches_per_buffer": value["dispatches"] / max(value["buffers"], 1),
            "isolated": solo is not None and solo in exclusive,
        })
    rows.sort(key=lambda row: -row["us_per_draft"])
    return rows


def price(measured_us: float, moved_bytes: int, weights: int) -> dict:
    achieved = moved_bytes / (measured_us * 1e-6) / 1e9
    memory_us = moved_bytes / (DRAM_GB_S * 1e9) * 1e6
    non_memory_us = measured_us - memory_us
    return {
        "measured_us": measured_us,
        "moved_bytes": moved_bytes,
        "moved_mb": moved_bytes / 1e6,
        "weights": weights,
        "achieved_gb_s": achieved,
        "fraction_of_dram_ceiling": achieved / DRAM_GB_S,
        "memory_us_at_ceiling": memory_us,
        "non_memory_us": non_memory_us,
        "ps_per_weight": non_memory_us * 1e6 / weights,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("census", nargs="+")
    ap.add_argument("--width-phase", default="w9|draft_head")
    ap.add_argument("--drafts-per-round", type=int, default=8)
    ap.add_argument("--stages", default="armc", choices=sorted(STAGE_SETS))
    ap.add_argument("--json")
    args = ap.parse_args()
    stages = STAGE_SETS[args.stages]

    report = {
        "stage_set": args.stages,
        "dram_gb_s": DRAM_GB_S,
        "affine4_ps_per_weight": AFFINE4_PS_PER_WEIGHT,
        "affine2_declared_ps_per_weight": AFFINE2_DECLARED_PS_PER_WEIGHT,
        "stop_rule_us": STOP_RULE_US,
        "width_phase": args.width_phase,
        "drafts_per_round": args.drafts_per_round,
        "legs": {},
    }

    for path in args.census:
        snaps = steady_draft_snapshots(path, args.width_phase)
        if not snaps:
            print(f"=== {path}\n    no steady {args.width_phase} snapshots", file=sys.stderr)
            return 1
        n = len(snaps)
        per_draft = n * args.drafts_per_round
        exclusive = aggregate(snaps, "exclusive_kernels", args.width_phase)
        whole = aggregate(snaps, "signatures", args.width_phase)

        phase_ns = sum(s["by_width_phase"][args.width_phase]["gpu_ns"] for s in snaps)
        phase_disp = sum(
            s["by_width_phase"][args.width_phase]["dispatches"] for s in snaps
        )

        leg = {
            "census": path,
            "steady_snapshots": n,
            "snapshot_ids": [s["snapshot"] for s in snaps],
            "phase_us_per_round": phase_ns / 1e3 / n,
            "phase_us_per_draft": phase_ns / 1e3 / per_draft,
            "phase_dispatches_per_draft": phase_disp / per_draft,
            "roster": roster(whole, exclusive, per_draft),
            "stages": {},
        }

        print(f"=== {path}")
        print(f"    steady snapshots={n}  drafts/round={args.drafts_per_round}")
        print(
            "    %s: %.1f us/round  %.2f us/draft  %.1f dispatches/draft"
            % (
                args.width_phase,
                leg["phase_us_per_round"],
                leg["phase_us_per_draft"],
                leg["phase_dispatches_per_draft"],
            )
        )

        combined_non_memory = 0.0
        combined_isolated = True
        for name, spec in stages.items():
            entry = {"shape": spec["shape"], "expected_bytes": spec["bytes"]}

            hit = None
            for key, value in whole.items():
                if key.startswith(spec["key"]):
                    hit = (key, value)
                    break
            if hit is None:
                print(f"    {name}: NOT FOUND in this leg", file=sys.stderr)
                return 1
            whole_key, whole_val = hit
            entry["whole_buffer"] = {
                "signature": whole_key,
                "buffers_per_draft": whole_val["buffers"] / per_draft,
                "dispatches_per_buffer": whole_val["dispatches"]
                / max(whole_val["buffers"], 1),
                **price(
                    whole_val["gpu_ns"] / 1e3 / per_draft, spec["bytes"], spec["weights"]
                ),
            }

            iso = exclusive.get(spec["key"])
            if iso is None:
                combined_isolated = False
                entry["isolated"] = None
            else:
                entry["isolated"] = {
                    "buffers_per_draft": iso["buffers"] / per_draft,
                    "dispatches_per_buffer": iso["dispatches"] / max(iso["buffers"], 1),
                    **price(
                        iso["gpu_ns"] / 1e3 / per_draft, spec["bytes"], spec["weights"]
                    ),
                }
                combined_non_memory += entry["isolated"]["non_memory_us"]

            leg["stages"][name] = entry

            src = entry["isolated"] or entry["whole_buffer"]
            label = "ISOLATED" if entry["isolated"] else "WHOLE-BUFFER upper bound"
            print(f"    {name} [{label}]  {spec['shape']}")
            print(
                "        %.2f us/draft  %.2f MB  %.1f GB/s (%.1f%% of %.0f)"
                % (
                    src["measured_us"],
                    src["moved_mb"],
                    src["achieved_gb_s"],
                    100 * src["fraction_of_dram_ceiling"],
                    DRAM_GB_S,
                )
            )
            print(
                "        memory %.2f us  non-memory %.2f us  %.3f ps/weight"
                % (
                    src["memory_us_at_ceiling"],
                    src["non_memory_us"],
                    src["ps_per_weight"],
                )
            )
            if entry["isolated"]:
                wb = entry["whole_buffer"]
                print(
                    "        whole buffer %.2f us (%.1f dispatches) -> other-kernel residual %.2f us"
                    % (
                        wb["measured_us"],
                        wb["dispatches_per_buffer"],
                        wb["measured_us"] - src["measured_us"],
                    )
                )

        if combined_isolated:
            weights = sum(s["weights"] for s in stages.values())
            leg["combined"] = {
                "non_memory_us_per_draft": combined_non_memory,
                "ps_per_weight": combined_non_memory * 1e6 / weights,
                "stop_rule_us": STOP_RULE_US,
                "axis_closed": combined_non_memory < STOP_RULE_US,
            }
            print(
                "    COMBINED non-memory %.2f us/draft (%.3f ps/weight) -> axis_closed=%s"
                % (
                    combined_non_memory,
                    combined_non_memory * 1e6 / weights,
                    combined_non_memory < STOP_RULE_US,
                )
            )
        else:
            print("    COMBINED: at least one stage was never isolated", file=sys.stderr)

        report["legs"][path] = leg

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
