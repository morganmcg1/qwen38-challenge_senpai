#!/usr/bin/env python3
"""E196: price the second SDPA call of the qL 6..9 exactness split.

Reduces two probe files:

  research/e196-census.json   which kernel fires for call A and call B at
                              every cell, and for the R = 1..5 ladder
  research/e196-timing.json   chain-slope timing of the vector SDPA family

Pricing model, fitted on the R = 1..5 ladder that MLX's existing dispatch
already serves:

    T(R, N) = a0 + a1*N + R*(b0 + b1*N)

  a0 + a1*N   per-CALL fixed cost (launch, output write, 2pass reduction setup)
  b0          per-ROW cost that does NOT scale with the KV window
  b1*N        per-ROW cost that DOES scale with the KV window, i.e. the row's
              own traversal of the KV -- the only term a fused, row-amortised
              Route-1 kernel can remove

Today's split at width m over a committed window of kL keys pays

    T(5, kSplit) + T(m-5, kL)          kSplit = kL - (m - 5)

A Route-1 fused one-pass kernel cannot cost less than

    F(m, kL) = a0 + a1*kL + m*b0 + b1*kL

one call's fixed cost, one KV traversal, and m rows of non-KV row work. The
difference is the Route-1 recoverable time. Two brackets are also reported:

  conservative  a fused kernel with today's per-row efficiency (b unchanged)
  optimistic    a fused kernel whose extra m-5 rows are free, which equals the
                whole FINDING 497 split step

harness=local. Within-session relative measurement, no thermal gate, no score.
"""

import argparse
import json
import statistics
from collections import defaultdict

FULL_ATTENTION_LAYERS = 16
MUE_US_PER_ROUND = 567.0  # senpai/frontier-state.json
SPLIT = 5
GQA = 6
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
BYTES_PER_ELEMENT = 2  # bfloat16

# M4 Pro (applegpu_g16s, 20-core GPU) limits, for the roofline that names the
# binding resource. The vector SDPA kernel issues scalar fused multiply-adds,
# not simdgroup matrix operations, so the FMA peak is the relevant ALU limit:
# 20 cores * 128 lanes * 2 flops * 1.575 GHz.
M4PRO_DRAM_GB_PER_S = 273.0
M4PRO_BF16_TFLOP_PER_S = 8.06

# Round composition of a scored 512-token decode leg, from
# research/analysis-runP-512-confirm.json (cap-7 tree, 81 rounds).
# Key is the draft depth; rows per round m = depth + 1.
DEPTH_HISTOGRAM = {3: 1, 4: 29, 5: 2, 6: 3, 7: 46}
SEED_TOKENS = 512  # senpai/program.md: 512-token seed, then 512 decoded tokens
DECODE_TOKENS = 512


def load(path):
    with open(path) as handle:
        return json.load(handle)


def robust_min(values):
    return min(values)


def reduce_samples(timing):
    """Robust minimum microseconds per rep for every measured cell.

    External contention only ever ADDS time to a GPU block, so the per-cell
    minimum over the ABBA blocks is the contention-robust estimator (E191
    precedent).
    """
    buckets = defaultdict(list)
    for sample in timing["samples"]:
        key = (
            sample["arm"],
            sample["mode"],
            sample["chain"],
            sample["kv"],
            sample["m"],
            sample["rows"],
            sample["keys"],
            sample.get("heads", HEADS),
        )
        buckets[key].append(sample["microseconds"])
    reduced = {}
    for key, values in buckets.items():
        reduced[key] = {
            "min": robust_min(values),
            "median": statistics.median(values),
            "n": len(values),
            "spread_pct": 100.0 * (max(values) - min(values)) / max(values),
        }
    return reduced


def linear_fit(xs, ys):
    """Least-squares slope and intercept, plus the max residual."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return {
        "slope": slope,
        "intercept": intercept,
        "max_abs_residual": max(abs(r) for r in residuals),
        "points": list(zip(xs, ys)),
    }


def chain_slopes(reduced):
    """Marginal GPU cost of ONE more SDPA call, free of the eval barrier.

    For each cell, regress total microseconds on the chain length. The slope
    is the pipelined per-call cost; the intercept absorbs the blocking eval
    that FINDING 497's instrument charged to every call.
    """
    grouped = defaultdict(dict)
    for (arm, mode, chain, kv, m, rows, keys, heads), stats in reduced.items():
        grouped[(arm, mode, kv, m, rows, keys, heads)][chain] = stats["min"]
    slopes = {}
    for key, by_chain in grouped.items():
        if len(by_chain) < 2:
            continue
        chains = sorted(by_chain)
        fit = linear_fit([float(c) for c in chains], [by_chain[c] for c in chains])
        # A `pair` cell issues TWO calls per chain step; report per chain step.
        slopes[key] = {
            "per_call_us": fit["slope"],
            "eval_intercept_us": fit["intercept"],
            "max_abs_residual_us": fit["max_abs_residual"],
            "chains": chains,
            "totals_us": [by_chain[c] for c in chains],
        }
    return slopes


def ladder_model(slopes, mode):
    """Fit T(R, N) = a(N) + b(N)*R per KV window, then a(N) and b(N) in N."""
    per_kv = {}
    for (arm, m_mode, kv, m, rows, keys, heads), value in slopes.items():
        if arm != "ladder" or m_mode != mode or heads != HEADS:
            continue
        per_kv.setdefault((kv, keys), {})[rows] = value["per_call_us"]
    model = {}
    for (kv, keys), by_rows in sorted(per_kv.items()):
        rows = sorted(by_rows)
        fit = linear_fit([float(r) for r in rows], [by_rows[r] for r in rows])
        model[kv] = {
            "kv": kv,
            "keys": keys,
            "a_us": fit["intercept"],
            "b_us_per_row": fit["slope"],
            "max_abs_residual_us": fit["max_abs_residual"],
            "per_row_us": {r: by_rows[r] for r in rows},
        }
    return model


def split_by_kv_family(model, boundary=1024):
    """The 1-pass `sdpa_vector` family and the 2-pass family are different
    kernels; never fit or interpolate across the boundary."""
    one_pass = {kv: v for kv, v in model.items() if v["keys"] < boundary}
    two_pass = {kv: v for kv, v in model.items() if v["keys"] >= boundary}
    return one_pass, two_pass


def kv_scaling(model_subset, field):
    """Decompose a per-call or per-row cost into a KV-independent part and a
    KV-proportional part: value(N) = c0 + c1*N."""
    if len(model_subset) < 2:
        return None
    items = sorted(model_subset.values(), key=lambda v: v["keys"])
    fit = linear_fit([float(v["keys"]) for v in items], [v[field] for v in items])
    return {"c0": fit["intercept"], "c1_per_key": fit["slope"], "points": fit["points"]}


def kv_bytes_per_traversal(keys):
    """Bytes of K plus V a single full pass over the window must read."""
    return 2 * keys * KV_HEADS * HEAD_DIM * BYTES_PER_ELEMENT


def price_cells(slopes, model, mode, widths, kvs):
    """Per-layer pricing of today's split and of the Route-1 floor."""
    rows_out = []
    for kv in kvs:
        if kv not in model:
            continue
        fit = model[kv]
        a_us = fit["a_us"]
        b_us = fit["b_us_per_row"]
        for m in widths:
            call_a = slopes.get(("callA", mode, kv, m, SPLIT, kv + m - (m - SPLIT), HEADS))
            call_b = slopes.get(("callB", mode, kv, m, m - SPLIT, kv + m, HEADS))
            pair = slopes.get(("pair", mode, kv, m, m, kv + m, HEADS))
            if call_a is None or call_b is None:
                continue
            kL = kv + m
            k_split = kL - (m - SPLIT)
            measured_split = call_a["per_call_us"] + call_b["per_call_us"]
            measured_pair = pair["per_call_us"] if pair else None
            # The m = 5 production control, from the ladder fit at this window.
            control_five = a_us + b_us * SPLIT
            rows_out.append(
                {
                    "kv": kv,
                    "m": m,
                    "kL": kL,
                    "kSplit": k_split,
                    "call_a_us": call_a["per_call_us"],
                    "call_b_us": call_b["per_call_us"],
                    "sum_a_b_us": measured_split,
                    "pair_us": measured_pair,
                    "overlap_us": (
                        None if measured_pair is None else measured_split - measured_pair
                    ),
                    "control_m5_us": control_five,
                    "step_vs_m5_us": measured_split - control_five,
                    "a_us": a_us,
                    "b_us_per_row": b_us,
                }
            )
    return rows_out


def route1_pricing(priced, one_pass_scaling, two_pass_scaling, boundary=1024):
    """Route-1 recoverable time per cell, with its two brackets."""
    out = []
    for row in priced:
        kL = row["kL"]
        scaling = two_pass_scaling if kL >= boundary else one_pass_scaling
        a_us = row["a_us"]
        b_us = row["b_us_per_row"]
        m = row["m"]
        measured = row["pair_us"] if row["pair_us"] is not None else row["sum_a_b_us"]

        # Conservative floor: one call, today's per-row efficiency.
        floor_conservative = a_us + b_us * m
        # Optimistic floor: the extra m-5 rows are free, i.e. the fused pass
        # costs exactly what today's m = 5 call costs.
        floor_optimistic = a_us + b_us * SPLIT
        # Route-1 floor: one call's fixed cost, ONE KV traversal, and m rows of
        # the per-row work that does not scale with the KV window.
        if scaling is None:
            floor_route1 = None
        else:
            b0 = scaling["b0"]
            b1 = scaling["b1_per_key"]
            floor_route1 = a_us + m * b0 + b1 * kL

        out.append(
            {
                **row,
                "floor_conservative_us": floor_conservative,
                "floor_route1_us": floor_route1,
                "floor_optimistic_us": floor_optimistic,
                "recoverable_conservative_us": measured - floor_conservative,
                "recoverable_route1_us": (
                    None if floor_route1 is None else measured - floor_route1
                ),
                "recoverable_optimistic_us": measured - floor_optimistic,
                "kv_bytes_per_traversal": kv_bytes_per_traversal(kL),
            }
        )
    return out


def round_weights():
    """Current-tree round composition and the KV window each round sees."""
    total = sum(DEPTH_HISTOGRAM.values())
    tokens_per_round = DECODE_TOKENS / total
    rounds = []
    # Order is unknown, so spread every round evenly across the window. The KV
    # window a round sees is the cache offset before its own append.
    index = 0
    for depth, count in sorted(DEPTH_HISTOGRAM.items()):
        for _ in range(count):
            rounds.append({"m": depth + 1})
    for position, entry in enumerate(rounds):
        entry["kv"] = SEED_TOKENS + tokens_per_round * position
        entry["kL"] = entry["kv"] + entry["m"]
        entry["weight"] = 1.0 / total
        index += 1
    return rounds, total, tokens_per_round


def interpolate(priced_by_m, m, kv, field, boundary=1024):
    """Interpolate a per-cell value in the KV window WITHOUT crossing the
    1-pass / 2-pass kernel boundary. Returns (value, was_interpolated)."""
    candidates = sorted(
        (row for row in priced_by_m if row["m"] == m and row[field] is not None),
        key=lambda row: row["kv"],
    )
    same_family = [
        row
        for row in candidates
        if (row["kL"] >= boundary) == (kv + m >= boundary)
    ]
    if not same_family:
        return None, True
    if len(same_family) == 1:
        return same_family[0][field], True
    lower = [row for row in same_family if row["kv"] <= kv]
    upper = [row for row in same_family if row["kv"] >= kv]
    if lower and upper and lower[-1]["kv"] != upper[0]["kv"]:
        low, high = lower[-1], upper[0]
        span = high["kv"] - low["kv"]
        weight = (kv - low["kv"]) / span
        return low[field] + weight * (high[field] - low[field]), True
    if lower and upper and lower[-1]["kv"] == kv:
        return lower[-1][field], False
    # Extrapolate from the two nearest same-family cells.
    low, high = same_family[0], same_family[-1]
    span = high["kv"] - low["kv"]
    if span == 0:
        return low[field], True
    weight = (kv - low["kv"]) / span
    return low[field] + weight * (high[field] - low[field]), True


def weighted_aggregate(priced, field):
    rounds, total, tokens_per_round = round_weights()
    per_round_us = 0.0
    interpolated = 0
    covered = 0
    detail = defaultdict(lambda: {"rounds": 0, "us": 0.0})
    for entry in rounds:
        m = entry["m"]
        if m < 6:
            continue  # the split does not fire below width 6
        value, was_interpolated = interpolate(priced, m, entry["kv"], field)
        if value is None:
            continue
        covered += 1
        interpolated += 1 if was_interpolated else 0
        contribution = value * FULL_ATTENTION_LAYERS * entry["weight"]
        per_round_us += contribution
        detail[m]["rounds"] += 1
        detail[m]["us"] += contribution
    return {
        "field": field,
        "per_round_us": per_round_us,
        "per_round_ms": per_round_us / 1000.0,
        "mue": per_round_us / MUE_US_PER_ROUND,
        "split_rounds": covered,
        "total_rounds": total,
        "split_round_share": covered / total,
        "tokens_per_round": tokens_per_round,
        "interpolated_cells": interpolated,
        "per_m": {str(m): dict(value) for m, value in sorted(detail.items())},
    }


def census_summary(census):
    rows = []
    for cell in census["cells"]:
        for call in cell["calls"]:
            rows.append(
                {
                    "kv": cell["kv"],
                    "m": cell["m"],
                    "kL": cell["kL"],
                    "kSplit": cell["kSplit"],
                    "call": call["call"],
                    "dispatches": call["dispatches"],
                    "kernels": call["kernel_sequence"],
                    "grids": call["grids"],
                }
            )
    ladder = [
        {
            "kv": entry["kv"],
            "R": entry["R"],
            "kL": entry["kL"],
            "dispatches": entry["dispatches"],
            "kernels": entry["kernel_sequence"],
            "grids": entry["grids"],
        }
        for entry in census.get("ladder", [])
    ]
    return {"cells": rows, "ladder": ladder}


def occupancy_model(slopes, mode):
    """Is the scored `sdpa_vector` dispatch latency bound or bandwidth bound?

    Each arm holds per-threadgroup work fixed (GQA 6, head dim 256, one KV
    window) and scales the query-head count, which scales the threadgroup
    count and the total KV bytes read by the same factor. Regress per-call
    microseconds on head count and report the elasticity

        e = d ln(time) / d ln(heads)

    at the scored 24-head point. e -> 0 means the dispatch is latency bound
    and adding work per threadgroup is nearly free, so a fused kernel that
    loops m rows in one threadgroup really can amortize the KV traversal.
    e -> 1 means the dispatch is bandwidth or throughput bound, the machine
    is already saturated, and only the per-call fixed cost is recoverable.
    """
    per_cell = defaultdict(dict)
    for (arm, m_mode, kv, m, rows, keys, heads), value in slopes.items():
        if arm != "occupancy" or m_mode != mode:
            continue
        per_cell[(keys, rows)][heads] = value["per_call_us"]
    out = []
    for (keys, rows), by_heads in sorted(per_cell.items()):
        head_counts = sorted(by_heads)
        if len(head_counts) < 2:
            continue
        fit = linear_fit(
            [float(h) for h in head_counts], [by_heads[h] for h in head_counts]
        )
        at_scored = by_heads.get(HEADS)
        elasticity = (
            None if at_scored in (None, 0) else fit["slope"] * HEADS / at_scored
        )
        # Ratio form, free of any fit: doubling the heads from the scored
        # point costs this much more.
        doubling = None
        if HEADS in by_heads and 2 * HEADS in by_heads:
            doubling = by_heads[2 * HEADS] / by_heads[HEADS]
        halving = None
        if HEADS in by_heads and HEADS // 2 in by_heads:
            halving = by_heads[HEADS // 2] / by_heads[HEADS]
        out.append(
            {
                "keys": keys,
                "rows": rows,
                "per_call_us": {h: by_heads[h] for h in head_counts},
                "slope_us_per_head": fit["slope"],
                "intercept_us": fit["intercept"],
                "elasticity_at_24_heads": elasticity,
                "ratio_48_over_24": doubling,
                "ratio_12_over_24": halving,
            }
        )
    return out


def roofline(model, kv=512, m=SPLIT):
    """Which resource binds the scored `sdpa_vector` dispatch?

    The occupancy arm scales the threadgroup count and the total load traffic
    together, so it proves that a per-work resource binds but not which one.
    This does: it compares the achieved rates against the M4 Pro limits.

    Every (query head, query row) threadgroup reads the whole K and V window
    of its own KV head, so the loads are redundant by a factor of GQA times
    the row count. That redundant traffic is exactly what a fused,
    row-amortized kernel removes.
    """
    if kv not in model:
        return None
    fit = model[kv]
    keys = fit["keys"]
    seconds = (fit["a_us"] + fit["b_us_per_row"] * m) / 1e6
    unique_bytes = kv_bytes_per_traversal(keys)
    requested_bytes = unique_bytes * GQA * m
    # Q.K^T and the P.V accumulation: two multiply-adds per key element.
    flops = 2 * 2 * keys * HEAD_DIM * HEADS * m
    return {
        "kv": kv,
        "keys": keys,
        "rows": m,
        "call_us": seconds * 1e6,
        "unique_kv_bytes": unique_bytes,
        "requested_kv_bytes": requested_bytes,
        "redundancy_factor": GQA * m,
        "unique_gb_per_s": unique_bytes / seconds / 1e9,
        "requested_gb_per_s": requested_bytes / seconds / 1e9,
        "gflop_per_s": flops / seconds / 1e9,
        "m4pro_dram_gb_per_s": M4PRO_DRAM_GB_PER_S,
        "m4pro_bf16_tflop_per_s": M4PRO_BF16_TFLOP_PER_S,
        "fraction_of_dram_bandwidth": (unique_bytes / seconds / 1e9)
        / M4PRO_DRAM_GB_PER_S,
        "fraction_of_alu_peak": (flops / seconds / 1e12) / M4PRO_BF16_TFLOP_PER_S,
    }


def occupancy_verdict(rows, roof, weight_keys=517, weight_rows=5):
    """Read the occupancy elasticity together with the roofline.

    The elasticity says only whether a per-work resource binds. The roofline
    says which one. Route-1 removes redundant KV loads, so a work-scaled
    dispatch that sits far below both the DRAM and the ALU limit is bound by
    load throughput through the cache hierarchy, which is the resource
    row-amortization attacks.
    """
    deciding = next(
        (
            row
            for row in rows
            if row["keys"] == weight_keys and row["rows"] == weight_rows
        ),
        None,
    )
    low_parallelism = next(
        (row for row in rows if row["keys"] == weight_keys and row["rows"] == 1), None
    )
    if deciding is None or deciding["elasticity_at_24_heads"] is None:
        return {"cell": None, "verdict": "unmeasured"}
    elasticity = deciding["elasticity_at_24_heads"]
    scaling = (
        "latency_bound"
        if elasticity < 0.35
        else "work_scaled" if elasticity > 0.75 else "mixed"
    )
    if roof is None:
        binding = "unknown"
    elif roof["fraction_of_alu_peak"] > 0.5:
        binding = "alu"
    elif roof["fraction_of_dram_bandwidth"] > 0.5:
        binding = "dram_bandwidth"
    elif scaling == "latency_bound":
        binding = "dispatch_latency"
    else:
        binding = "redundant_load_throughput"
    return {
        "cell": deciding,
        "low_parallelism_cell": low_parallelism,
        "elasticity": elasticity,
        "scaling": scaling,
        "binding_resource": binding,
        "route1_removes_binding_resource": binding
        in {"redundant_load_throughput", "dram_bandwidth", "dispatch_latency"},
    }


def decide(recoverable_ms):
    if recoverable_ms >= 1.0:
        return "CONTINUE"
    if recoverable_ms < 0.5:
        return "CLOSE"
    return "UNCLEAR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", default="research/e196-census.json")
    parser.add_argument("--timing", default="research/e196-timing.json")
    parser.add_argument("--mode", default="serial", choices=["serial", "indep"])
    parser.add_argument("--out", default="research/e196-analysis.json")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    census = load(args.census)
    timing = load(args.timing)
    reduced = reduce_samples(timing)
    slopes = chain_slopes(reduced)

    report = {
        "probe": "e196-second-sdpa-pass-pricing",
        "harness": "local",
        "cool_gate_passed_real_gate": timing.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": timing.get("gate_qualified_for_timing"),
        "host_architecture": census.get("host_architecture"),
        "probe_needle": timing.get("probe_needle"),
        "gpu_temperature_c": timing.get("gpu_temperature_c"),
        "blocks": timing.get("blocks"),
        "reps": timing.get("reps"),
        "full_attention_layers": FULL_ATTENTION_LAYERS,
        "mue_us_per_round": MUE_US_PER_ROUND,
        "depth_histogram": {str(k): v for k, v in DEPTH_HISTOGRAM.items()},
        "census": census_summary(census),
        "modes": {},
    }

    widths = sorted({sample["m"] for sample in timing["samples"] if sample["m"] >= 6})
    kvs = sorted({sample["kv"] for sample in timing["samples"]})

    for mode in ["serial", "indep"]:
        model = ladder_model(slopes, mode)
        if not model:
            continue
        one_pass, two_pass = split_by_kv_family(model)

        def scaling_for(subset):
            a_fit = kv_scaling(subset, "a_us")
            b_fit = kv_scaling(subset, "b_us_per_row")
            if a_fit is None or b_fit is None:
                return None
            return {
                "a0": a_fit["c0"],
                "a1_per_key": a_fit["c1_per_key"],
                "b0": b_fit["c0"],
                "b1_per_key": b_fit["c1_per_key"],
                "a_points": a_fit["points"],
                "b_points": b_fit["points"],
            }

        one_pass_scaling = scaling_for(one_pass)
        two_pass_scaling = scaling_for(two_pass)
        priced = price_cells(slopes, model, mode, widths, kvs)
        priced = route1_pricing(priced, one_pass_scaling, two_pass_scaling)

        aggregates = {
            field: weighted_aggregate(priced, field)
            for field in [
                "step_vs_m5_us",
                "recoverable_conservative_us",
                "recoverable_route1_us",
                "recoverable_optimistic_us",
            ]
        }
        report["modes"][mode] = {
            "ladder_model": {str(k): v for k, v in model.items()},
            "one_pass_kv_scaling": one_pass_scaling,
            "two_pass_kv_scaling": two_pass_scaling,
            "cells": priced,
            "weighted": aggregates,
            "decision_route1": decide(
                aggregates["recoverable_route1_us"]["per_round_ms"]
            ),
            "occupancy": occupancy_model(slopes, mode),
            "roofline_m5_kv512": roofline(model, kv=512, m=SPLIT),
        }
        report["modes"][mode]["occupancy_verdict"] = occupancy_verdict(
            report["modes"][mode]["occupancy"],
            report["modes"][mode]["roofline_m5_kv512"],
        )

    # The bridge cells: the whole shipped `today` form, one call per eval,
    # which is FINDING 497's own instrument.
    bridge = []
    for (arm, mode, chain, kv, m, rows, keys, heads), stats in sorted(reduced.items()):
        if arm != "today":
            continue
        bridge.append({"kv": kv, "m": m, "us": stats["min"], "spread_pct": stats["spread_pct"]})
    by_kv_m = {(row["kv"], row["m"]): row["us"] for row in bridge}
    bridge_steps = []
    for (kv, m), value in sorted(by_kv_m.items()):
        if m < 6 or (kv, 5) not in by_kv_m:
            continue
        step_us = value - by_kv_m[(kv, 5)]
        bridge_steps.append(
            {
                "kv": kv,
                "m": m,
                "today_us": value,
                "control_m5_us": by_kv_m[(kv, 5)],
                "step_us": step_us,
                "step_ms_per_round": step_us * FULL_ATTENTION_LAYERS / 1000.0,
                "mue": step_us * FULL_ATTENTION_LAYERS / MUE_US_PER_ROUND,
            }
        )
    report["finding_497_bridge"] = bridge_steps

    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != "census"}, indent=1)[:6000])
    print(f"\nwrote {args.out}")

    if args.wandb:
        import wandb

        run = wandb.init(
            project="qwen38-mlx-challenge-senpai",
            entity="wandb-applied-ai-team",
            name="e196-second-sdpa-pass-pricing",
            job_type="probe",
            config={
                "experiment": "E196",
                "harness": "local",
                "host_architecture": report["host_architecture"],
                "full_attention_layers": FULL_ATTENTION_LAYERS,
                "mue_us_per_round": MUE_US_PER_ROUND,
                "depth_histogram": report["depth_histogram"],
                "blocks": report["blocks"],
                "reps": report["reps"],
                "cool_gate_passed_real_gate": report["cool_gate_passed_real_gate"],
                "gate_qualified_for_timing": report["gate_qualified_for_timing"],
            },
        )
        summary = {}
        for mode, payload in report["modes"].items():
            for field, aggregate in payload["weighted"].items():
                summary[f"{mode}/{field}/ms_per_round"] = aggregate["per_round_ms"]
                summary[f"{mode}/{field}/mue"] = aggregate["mue"]
            summary[f"{mode}/decision_route1"] = payload["decision_route1"]
            verdict = payload["occupancy_verdict"]
            summary[f"{mode}/occupancy/elasticity_at_24_heads"] = verdict.get(
                "elasticity"
            )
            summary[f"{mode}/occupancy/scaling"] = verdict.get("scaling")
            summary[f"{mode}/occupancy/binding_resource"] = verdict.get(
                "binding_resource"
            )
            roof = payload["roofline_m5_kv512"]
            if roof:
                summary[f"{mode}/roofline/fraction_of_alu_peak"] = roof[
                    "fraction_of_alu_peak"
                ]
                summary[f"{mode}/roofline/fraction_of_dram_bandwidth"] = roof[
                    "fraction_of_dram_bandwidth"
                ]
                summary[f"{mode}/roofline/requested_gb_per_s"] = roof[
                    "requested_gb_per_s"
                ]
                summary[f"{mode}/roofline/redundancy_factor"] = roof[
                    "redundancy_factor"
                ]
            occupancy_table = wandb.Table(
                columns=[
                    "keys",
                    "rows",
                    "heads",
                    "per_call_us",
                ]
            )
            for row in payload["occupancy"]:
                for heads, value in row["per_call_us"].items():
                    occupancy_table.add_data(
                        row["keys"], row["rows"], int(heads), value
                    )
            run.log({f"{mode}/occupancy": occupancy_table})
            table = wandb.Table(
                columns=[
                    "kv",
                    "m",
                    "kL",
                    "call_a_us",
                    "call_b_us",
                    "pair_us",
                    "control_m5_us",
                    "step_vs_m5_us",
                    "a_us",
                    "b_us_per_row",
                    "floor_route1_us",
                    "recoverable_conservative_us",
                    "recoverable_route1_us",
                    "recoverable_optimistic_us",
                ]
            )
            for cell in payload["cells"]:
                table.add_data(
                    cell["kv"],
                    cell["m"],
                    cell["kL"],
                    cell["call_a_us"],
                    cell["call_b_us"],
                    cell["pair_us"],
                    cell["control_m5_us"],
                    cell["step_vs_m5_us"],
                    cell["a_us"],
                    cell["b_us_per_row"],
                    cell["floor_route1_us"],
                    cell["recoverable_conservative_us"],
                    cell["recoverable_route1_us"],
                    cell["recoverable_optimistic_us"],
                )
            run.log({f"{mode}/cells": table})
        bridge_table = wandb.Table(
            columns=["kv", "m", "today_us", "control_m5_us", "step_ms_per_round", "mue"]
        )
        for row in bridge_steps:
            bridge_table.add_data(
                row["kv"], row["m"], row["today_us"], row["control_m5_us"],
                row["step_ms_per_round"], row["mue"],
            )
        run.log({"finding_497_bridge": bridge_table})
        run.summary.update(summary)
        print(f"wandb run: {run.url} id={run.id}")
        run.finish()


if __name__ == "__main__":
    main()
