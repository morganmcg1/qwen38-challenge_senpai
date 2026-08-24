#!/usr/bin/env python3
"""E169 marginal verified row decomposition: fit and reconcile both estimators.

harness=local for every measured number here. The in-situ census and the
bottom-up shape sweep are independent estimates of the same quantity -- the
seconds one extra verified row adds to one scored target forward -- so the
interesting output is not either number alone but their agreement.

Both estimators are fitted with the same two-term row law:

    T(M) = a + b*M + c*(activeInputGroups(M) - 1)

`b` is the smooth marginal row, and `c` is one extra pass of the whole weight
matrix. The group boundary is not a modelling choice: `Qwen35CustomQMV`
launches `ceil(M / inputsPerGroup(M))` input-row threadgroups per wide QMV
call, each of which re-reads every weight byte, and its `inputsPerGroup` table
drops from 5 at M=5 to 3 at M=6 and back to 3 at M=9. Fitting a single linear
term across that step would average a per-row price with a per-weight-pass
price, and LAW 377 values those two channels 187x apart on the ranked host.
"""

from __future__ import annotations

import argparse
import json
import collections
import math
import statistics
import sys
from pathlib import Path

BLOCK_PREFIX = "E169_BLOCK "

# Mirrors Qwen35CustomQMV.inputsPerGroup, Qwen35.swift:1715-1729. M=1 is not in
# `Qwen35CustomQMV.widths` and runs the library kernel instead, so it is a
# different mechanism and is excluded from every routed fit below.
INPUTS_PER_GROUP = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}


def qmv_groups(m):
    if m not in INPUTS_PER_GROUP:
        return 1
    ipg = INPUTS_PER_GROUP[m]
    return (m + ipg - 1) // ipg


# ---------------------------------------------------------------- fitting


def _solve(ata, atb):
    """Gaussian elimination with partial pivoting on a small dense system."""
    n = len(atb)
    m = [row[:] + [atb[i]] for i, row in enumerate(ata)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-18:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col] / m[col][col]
            for k in range(col, n + 1):
                m[row][k] -= factor * m[col][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_row_law(points):
    """Least squares fit of T(M) = a + b*M + c*(groups(M)-1) over (M, seconds).

    Drops the group column when the measured widths never cross a boundary,
    because an identically zero column makes the design singular.
    """
    widths = [m for m, _ in points]
    groups = [qmv_groups(m) - 1 for m in widths]
    use_group = len(set(groups)) > 1
    basis = [
        [1.0, float(m)] + ([float(g)] if use_group else [])
        for m, g in zip(widths, groups)
    ]
    ys = [t for _, t in points]

    n = len(basis[0])
    if len(ys) <= n:
        return None
    ata = [[sum(r[i] * r[j] for r in basis) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] * y for r, y in zip(basis, ys)) for i in range(n)]
    beta = _solve(ata, atb)
    if beta is None:
        return None

    preds = [sum(b * v for b, v in zip(beta, row)) for row in basis]
    resid = [y - p for y, p in zip(ys, preds)]
    ss_res = sum(r * r for r in resid)
    mean_y = sum(ys) / len(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    dof = max(1, len(ys) - n)

    def se(index):
        col = _solve(ata, [1.0 if i == index else 0.0 for i in range(n)])
        return math.sqrt(max(0.0, ss_res / dof * col[index])) if col else float("nan")

    return {
        "intercept_s": beta[0],
        "per_row_s": beta[1],
        "per_group_s": beta[2] if use_group else 0.0,
        "group_term_fitted": use_group,
        "per_row_se_s": se(1),
        "per_group_se_s": se(2) if use_group else 0.0,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse_s": math.sqrt(ss_res / len(ys)),
        "max_abs_resid_s": max(abs(r) for r in resid),
        "n_points": len(ys),
        "fitted_widths": widths,
    }


def observed_marginals(curve):
    """Measured step from M to M+1, split by whether it crosses a group edge."""
    widths = sorted(curve)
    within, crossing = [], []
    steps = {}
    for lo, hi in zip(widths, widths[1:]):
        if hi != lo + 1:
            continue
        delta = curve[hi] - curve[lo]
        crossed = qmv_groups(hi) > qmv_groups(lo)
        steps[f"{lo}->{hi}"] = {"delta_s": delta, "crosses_group_edge": crossed}
        # M=1 runs the library kernel, so 1->2 is a kernel change, not a row.
        if lo >= 2:
            (crossing if crossed else within).append(delta)
    return {
        "steps": steps,
        "within_group_mean_s": statistics.mean(within) if within else None,
        "within_group_median_s": statistics.median(within) if within else None,
        "group_crossing_mean_s": statistics.mean(crossing) if crossing else None,
        "group_crossing_values_s": crossing,
    }


# ---------------------------------------------------- in-situ census input


def load_census_blocks(paths):
    blocks = []
    for path in paths:
        for line in Path(path).read_text(errors="replace").splitlines():
            index = line.find(BLOCK_PREFIX)
            if index < 0:
                continue
            try:
                blocks.append(json.loads(line[index + len(BLOCK_PREFIX):]))
            except json.JSONDecodeError:
                continue
    return blocks


def analyze_census(blocks, drop_width_one=True):
    by_arm = {}
    for block in blocks:
        by_arm.setdefault(block["arm"], []).append(block)

    arms = {}
    for arm, records in sorted(by_arm.items()):
        # Pool the ascending and descending passes at each width: monotone
        # thermal drift inside one arm cancels to first order in the mean.
        by_width = {}
        for record in records:
            by_width.setdefault(record["width"], []).append(record["seconds_median"])
        curve = {w: statistics.mean(v) for w, v in sorted(by_width.items())}
        points = [(w, t) for w, t in curve.items() if not (drop_width_one and w == 1)]
        fit = fit_row_law(points)
        if fit is None:
            continue
        spread = {
            w: (max(v) - min(v)) / statistics.mean(v) if len(v) > 1 else 0.0
            for w, v in sorted(by_width.items())
        }
        temps = [b["gpu_temp_entry_c"] for b in records if "gpu_temp_entry_c" in b]
        arms[arm] = {
            **fit,
            "seconds_by_width": {str(w): t for w, t in curve.items()},
            "observed_marginals": observed_marginals(curve),
            "pass_spread_frac_by_width": {str(w): s for w, s in spread.items()},
            "max_pass_spread_frac": max(spread.values()) if spread else 0.0,
            "gpu_temp_entry_c_min": min(temps) if temps else None,
            "gpu_temp_entry_c_max": max(temps) if temps else None,
            "blocks": len(records),
        }
    return arms


# Which arms are disjoint single families, and what the census can never see.
CENSUS_FAMILIES = {
    "lm_head": "lm_head",
    "mlp_all": "mlp",
    "fa_o_proj": "attn_o_proj",
    "gdn_out_proj": "gdn_out_proj",
}


def census_per_width_deltas(blocks):
    """Direct per-width arm deltas, the estimate that needs no row-law fit.

    Each (width, arm) cell is measured ABBA as baseline, arm, arm, baseline, so
    the local baseline mean already absorbs monotone drift. The null control is
    the same wrapper at full width, so subtracting it leaves only the pinned
    family. An arm whose null-corrected delta is inside the baseline pass spread
    did not reach the scored path at all; the vendored fast path calls
    `qwen35RoutedLinear` on the module's arrays instead of dispatching through
    `callAsFunction`, so module surgery cannot intercept it.
    """
    cells = collections.defaultdict(list)
    for block in blocks:
        cells[(block["width"], block["arm"])].append(block["seconds_median"])

    widths = sorted({w for w, _ in cells})
    serial = statistics.fmean(cells[(1, "baseline")]) if (1, "baseline") in cells else None

    out = {"width_1_baseline_s": serial, "widths": {}}
    for width in widths:
        base_values = cells.get((width, "baseline"))
        if not base_values:
            continue
        base = statistics.fmean(base_values)
        spread = (max(base_values) - min(base_values)) / base if base else None
        null_values = cells.get((width, "null"))
        null_delta = statistics.fmean(null_values) - base if null_values else None
        width_tax = base - serial if serial is not None else None

        entry = {
            "baseline_s": base,
            "baseline_blocks": len(base_values),
            "baseline_spread_frac": spread,
            "null_delta_s": null_delta,
            "width_tax_vs_serial_s": width_tax,
            "active_input_groups": qmv_groups(width),
            "inputs_per_group": INPUTS_PER_GROUP.get(width),
            "arms": {},
        }
        for (cell_width, arm), values in cells.items():
            if cell_width != width or arm in ("baseline", "null"):
                continue
            delta = statistics.fmean(values) - base
            corrected = delta - null_delta if null_delta is not None else None
            reached = (
                corrected is not None and spread is not None
                and abs(corrected) > spread * base
            )
            entry["arms"][arm] = {
                "raw_delta_s": delta,
                "null_corrected_delta_s": corrected,
                "family_width_tax_s": -corrected if corrected is not None else None,
                "share_of_width_tax": (
                    -corrected / width_tax
                    if corrected is not None and width_tax else None
                ),
                "reached_scored_path": reached,
                "blocks": len(values),
            }
        out["widths"][str(width)] = entry
    return out


def census_shares(arms):
    """Family share of the marginal row, referenced to the null control."""
    if "null" not in arms:
        return {"error": "null control missing; wrapper cost cannot be subtracted"}
    ref_row = arms["null"]["per_row_s"]
    ref_group = arms["null"]["per_group_s"]
    baseline = arms.get("baseline", {})

    out = {
        "reference_arm": "null",
        "reference_per_row_s": ref_row,
        "reference_per_group_s": ref_group,
        "baseline_per_row_s": baseline.get("per_row_s"),
        "baseline_per_group_s": baseline.get("per_group_s"),
        # A large null-vs-baseline gap means the wrapper itself is not free and
        # every share below is only as trustworthy as that subtraction.
        "wrapper_overhead_frac_of_baseline": (
            (ref_row - baseline["per_row_s"]) / baseline["per_row_s"]
            if baseline.get("per_row_s") else None
        ),
        "families": {},
    }
    attributed_row = 0.0
    attributed_group = 0.0
    for arm, family in CENSUS_FAMILIES.items():
        if arm not in arms:
            continue
        drop_row = ref_row - arms[arm]["per_row_s"]
        drop_group = ref_group - arms[arm]["per_group_s"]
        se_row = math.hypot(arms[arm]["per_row_se_s"], arms["null"]["per_row_se_s"])
        out["families"][family] = {
            "arm": arm,
            "per_row_s": arms[arm]["per_row_s"],
            "per_group_s": arms[arm]["per_group_s"],
            "marginal_row_s_attributed": drop_row,
            "share_of_marginal_row": drop_row / ref_row if ref_row else None,
            "group_s_attributed": drop_group,
            "share_of_group_pass": drop_group / ref_group if ref_group else None,
            "row_se_s": se_row,
            "significant_2se": abs(drop_row) > 2 * se_row,
        }
        attributed_row += drop_row
        attributed_group += drop_group
    out["attributed_row_s"] = attributed_row
    out["attributed_row_share"] = attributed_row / ref_row if ref_row else None
    out["residual_row_s"] = ref_row - attributed_row
    out["residual_row_share"] = (ref_row - attributed_row) / ref_row if ref_row else None
    out["attributed_group_s"] = attributed_group
    out["residual_group_s"] = ref_group - attributed_group

    if "all_interceptable" in arms:
        joint = ref_row - arms["all_interceptable"]["per_row_s"]
        out["additivity"] = {
            "joint_arm_marginal_row_s": joint,
            "sum_of_single_arms_s": attributed_row,
            "difference_s": joint - attributed_row,
            "difference_frac_of_joint": (
                (joint - attributed_row) / joint if joint else None),
        }
    return out


# --------------------------------------------------------- bottom-up input


def analyze_deconfound(payload):
    """Separate the output-width effect from the reduction-depth effect.

    The scored shape set cannot do this: every fast scored shape has k=5120 and
    every slow one has k>5120. Two arms share the k=5120, n=5120 anchor. If
    throughput tracks `n` and is flat in `k`, the deficit is a launch-geometry
    effect, because the Y grid is `n/8` threadgroups (`Qwen35.swift:1834`) and
    each threadgroup owns 8 output rows (`Qwen35.swift:1590`). If it tracks `k`
    and is flat in `n`, the deficit is in the reduction and the output mapping
    is innocent.
    """
    records = payload.get("deconfound_shapes")
    if not records:
        return None

    out = {"arms": {}, "threadgroups_y_formula": "n / 8", "anchor": None}
    for record in records:
        arm = out["arms"].setdefault(record["family"], {})
        cells = {}
        for row in record["rows"]:
            cells.setdefault(row["m"], []).append(row)
        widths = {}
        for m, rows in sorted(cells.items()):
            seconds = statistics.fmean(r["seconds_per_call"] for r in rows)
            routed = all(r["routed_to_custom_qmv"] for r in rows)
            widths[str(m)] = {
                "seconds_per_call": seconds,
                "tflops": rows[0]["flops_per_call"] / seconds / 1e12,
                "routed_to_custom_qmv": routed,
                "pass_spread_frac": (
                    (max(r["seconds_per_call"] for r in rows)
                     - min(r["seconds_per_call"] for r in rows)) / seconds
                ),
            }
        arm[record["name"]] = {
            "k": record["k"],
            "n": record["n"],
            "threadgroups_y": record["n"] // 8,
            "widths": widths,
        }
        if record["k"] == 5120 and record["n"] == 5120:
            out["anchor"] = record["name"]

    # Verdict per width: does throughput move more along n or along k?
    verdicts = {}
    sweep_n = out["arms"].get("sweep_n", {})
    sweep_k = out["arms"].get("sweep_k", {})
    all_widths = set()
    for group in (sweep_n, sweep_k):
        for cell in group.values():
            all_widths.update(cell["widths"])
    for width in sorted(all_widths, key=int):
        def span(group):
            values = [
                cell["widths"][width]["tflops"]
                for cell in group.values() if width in cell["widths"]
            ]
            return (min(values), max(values)) if values else None

        # The anchor cell (k=5120, n=5120) belongs to both arms: it is the
        # k=5120 end of the k sweep and the n=5120 end of the n sweep.
        anchor_cell = sweep_n.get(out["anchor"]) if out["anchor"] else None
        anchor = (
            anchor_cell["widths"][width]["tflops"]
            if anchor_cell and width in anchor_cell["widths"] else None
        )
        n_span, k_span = span(sweep_n), span(sweep_k)
        if not n_span or not k_span:
            continue
        if anchor is not None:
            k_span = (min(k_span[0], anchor), max(k_span[1], anchor))
        n_range = n_span[1] / n_span[0]
        k_range = k_span[1] / k_span[0]
        verdicts[width] = {
            "sweep_n_tflops_min": n_span[0],
            "sweep_n_tflops_max": n_span[1],
            "sweep_n_ratio": n_range,
            "sweep_k_tflops_min": k_span[0],
            "sweep_k_tflops_max": k_span[1],
            "sweep_k_ratio": k_range,
            "dominant_axis": "n" if n_range > k_range else "k",
            "axis_ratio": max(n_range, k_range) / min(n_range, k_range),
        }
    out["verdict_by_width"] = verdicts
    out["byte_law_by_width"] = _deconfound_byte_law(out["arms"])
    out["matched_byte_pairs"] = _matched_byte_pairs(out["arms"])
    return out


# Affine 4-bit, group 64: k/2 packed bytes plus one bf16 scale and one bf16
# bias per group of 64, per output row.
WEIGHT_BYTES_PER_NK = 0.5 + 2.0 / 64.0 + 2.0 / 64.0


def weight_bytes(k, n):
    return n * k * WEIGHT_BYTES_PER_NK


def _deconfound_cells(arms):
    cells = {}
    for group in arms.values():
        for name, cell in group.items():
            cells[name] = cell
    return cells


def _deconfound_byte_law(arms):
    """Fit `t = t0 + mb / bandwidth` across the whole n-by-k grid.

    If one affine law in weight bytes fits every cell at both extremes of the
    grid, then time is a function of bytes alone and is symmetric in `n` and
    `k`. That refutes both readings of the seven-point scored table: neither
    output width nor reduction depth has an effect of its own, and the
    per-shape TFLOP/s column is `flops / (t0 + mb / bw)`, which falls with
    `mb` for a fixed `t0` even when the kernel is perfectly efficient.
    """
    cells = _deconfound_cells(arms)
    widths = set()
    for cell in cells.values():
        widths.update(cell["widths"])

    result = {}
    for width in sorted(widths, key=int):
        points = []
        for name, cell in cells.items():
            slot = cell["widths"].get(width)
            if slot is None or not slot["routed_to_custom_qmv"]:
                continue
            mb = weight_bytes(cell["k"], cell["n"]) / 1e6
            points.append((name, cell["k"], cell["n"], mb,
                           slot["seconds_per_call"] * 1e6))
        if len(points) < 3:
            continue
        n_pts = float(len(points))
        sx = sum(p[3] for p in points)
        sy = sum(p[4] for p in points)
        sxx = sum(p[3] * p[3] for p in points)
        sxy = sum(p[3] * p[4] for p in points)
        det = n_pts * sxx - sx * sx
        slope = (n_pts * sxy - sx * sy) / det
        intercept = (sy * sxx - sx * sxy) / det
        residuals = []
        for name, k, n, mb, us in points:
            pred = intercept + slope * mb
            residuals.append({
                "cell": name, "k": k, "n": n, "mb": mb,
                "measured_us": us, "predicted_us": pred,
                "residual_frac": (us - pred) / us,
            })
        result[width] = {
            "fixed_us": intercept,
            "us_per_mb": slope,
            "effective_gb_per_s": 1e3 / slope,
            "max_abs_residual_frac": max(abs(r["residual_frac"])
                                         for r in residuals),
            "cells": sorted(residuals, key=lambda r: r["mb"]),
        }
    return result


def _matched_byte_pairs(arms):
    """Cells with equal weight bytes but transposed `n` and `k`.

    These are the direct test. `k8192_n5120` and `k5120_n8192` stream the same
    bytes and run the same arithmetic, but the first launches 640 threadgroups
    and the second 1024 (`Qwen35.swift:1834`, grid Y is `n/8`). A launch
    geometry effect must show up here; a bytes-only law must not.
    """
    cells = _deconfound_cells(arms)
    by_bytes = {}
    for name, cell in cells.items():
        by_bytes.setdefault(weight_bytes(cell["k"], cell["n"]), []).append(
            (name, cell))

    pairs = []
    for total, members in sorted(by_bytes.items()):
        if len(members) != 2:
            continue
        (name_a, cell_a), (name_b, cell_b) = sorted(
            members, key=lambda item: item[1]["n"])
        widths = sorted(
            set(cell_a["widths"]) & set(cell_b["widths"]), key=int)
        rows = []
        for width in widths:
            slot_a = cell_a["widths"][width]
            slot_b = cell_b["widths"][width]
            if not (slot_a["routed_to_custom_qmv"]
                    and slot_b["routed_to_custom_qmv"]):
                continue
            rows.append({
                "m": int(width),
                "narrow_us": slot_a["seconds_per_call"] * 1e6,
                "wide_us": slot_b["seconds_per_call"] * 1e6,
                "wide_over_narrow": (slot_b["seconds_per_call"]
                                     / slot_a["seconds_per_call"]),
            })
        if not rows:
            continue
        pairs.append({
            "mb": total / 1e6,
            "narrow_output": {"name": name_a, "k": cell_a["k"],
                              "n": cell_a["n"],
                              "threadgroups_y": cell_a["threadgroups_y"]},
            "wide_output": {"name": name_b, "k": cell_b["k"],
                            "n": cell_b["n"],
                            "threadgroups_y": cell_b["threadgroups_y"]},
            "threadgroup_ratio": (cell_b["threadgroups_y"]
                                  / cell_a["threadgroups_y"]),
            "widths": rows,
            "max_abs_deviation_frac": max(
                abs(r["wide_over_narrow"] - 1.0) for r in rows),
        })
    return pairs


# Weight bytes the target forward streams, from the transformed safetensors
# headers (`research/e169_flop_budget.py`): mlp + gdn + attn + lm_head. The
# embedding table is a gather, not a stream, and is excluded.
TARGET_FORWARD_WEIGHT_MB = 14412.35

# FLOPs one row of arithmetic performs per MB of affine-4 group-64 weight,
# counting one multiply and one add per weight element.
FLOP_PER_ROW_PER_MB = 2.0 / WEIGHT_BYTES_PER_NK * 1e6


def stream_vs_arithmetic(byte_law):
    """Split the byte-law slope into a weight-pass term and a per-row term.

    `slope(M) = alpha * activeInputGroups(M) + beta * M`, in microseconds per
    megabyte of weight. `alpha` is paid once per input-row group and is the
    weight stream. `beta` is paid once per row and is arithmetic on weights
    already resident. This is the measurement that answers whether the round
    is bandwidth-bound or compute-bound, because the two terms are separated
    inside one kernel on one host rather than compared across hosts.
    """
    rows = []
    for width, law in byte_law.items():
        rows.append((float(qmv_groups(int(width))), float(width),
                     law["us_per_mb"]))
    if len(rows) < 3:
        return None

    a11 = sum(r[0] * r[0] for r in rows)
    a12 = sum(r[0] * r[1] for r in rows)
    a22 = sum(r[1] * r[1] for r in rows)
    b1 = sum(r[0] * r[2] for r in rows)
    b2 = sum(r[1] * r[2] for r in rows)
    det = a11 * a22 - a12 * a12
    alpha = (b1 * a22 - a12 * b2) / det
    beta = (a11 * b2 - a12 * b1) / det

    fitted = []
    for groups, width, slope in sorted(rows, key=lambda r: r[1]):
        pred = alpha * groups + beta * width
        fitted.append({
            "m": int(width), "active_input_groups": int(groups),
            "measured_us_per_mb": slope, "predicted_us_per_mb": pred,
            "residual_frac": (slope - pred) / slope,
        })

    forward = {}
    for width in range(1, 10):
        groups = qmv_groups(width)
        stream_ms = alpha * groups * TARGET_FORWARD_WEIGHT_MB / 1e3
        arith_ms = beta * width * TARGET_FORWARD_WEIGHT_MB / 1e3
        forward[str(width)] = {
            "active_input_groups": groups,
            "weight_stream_ms": stream_ms,
            "arithmetic_ms": arith_ms,
            "sum_ms": stream_ms + arith_ms,
        }

    return {
        "model": "slope(M) = alpha*activeInputGroups(M) + beta*M, us per MB",
        "alpha_us_per_mb_per_weight_pass": alpha,
        "beta_us_per_mb_per_row": beta,
        "weight_stream_gb_per_s": 1e3 / alpha,
        "arithmetic_tflop_per_s": FLOP_PER_ROW_PER_MB / (beta * 1e-6) / 1e12,
        "max_abs_residual_frac": max(abs(f["residual_frac"]) for f in fitted),
        "fit": fitted,
        "target_forward_weight_mb": TARGET_FORWARD_WEIGHT_MB,
        "target_forward_projection": forward,
        "marginal_row_arithmetic_ms": beta * TARGET_FORWARD_WEIGHT_MB / 1e3,
        "marginal_row_weight_stream_ms": 0.0,
    }


# Roofs. Local pair is edward's measured peak from
# `research/e63-artifacts/e63-cost-curve.json`. Ranked pair is the published
# M5 Max 40-core bandwidth and a third-party measured bf16 GEMM rate; both are
# recorded at `senpai/campaign-ledger.md:16924-16935`.
LOCAL_PEAK_GB_S = 226.035
LOCAL_PEAK_TFLOPS = 7.506
RANKED_PEAK_GB_S = 614.0
RANKED_PEAK_TFLOPS = 53.0

GFLOP_PER_ROW = 51.696893952
# Prefill computes logits for the last row only, so it runs `lm_head` once
# instead of once per row.
GFLOP_PER_PREFILL_ROW = GFLOP_PER_ROW - 2.5427968


def group_partition(m):
    """Vector widths the shipped table instantiates for one verify of width M.

    `inputsPerGroup` fixes the group size; the last group takes the remainder,
    and `Qwen35.swift:1553` widens a one-row tail to two.
    """
    ipg = INPUTS_PER_GROUP.get(m)
    if ipg is None:
        return None
    widths = []
    first = 0
    while first < m:
        remaining = m - first
        widths.append(ipg if remaining >= ipg else max(remaining, 2))
        first += ipg
    return widths


def pass_cost_model(curve_by_width):
    """Fit the target forward as a fixed cost plus one cost per weight pass.

    `T(M) = F + sum over groups of pass(NA_group)`. The shipped table reuses
    the same `NA` at several `M`, so `NA = 3` appears at M = 3, 6 and 9 with
    one, two and three passes. That over-determines the model and lets the
    per-pass cost be separated from the per-forward cost without assuming
    anything about what the row is made of.
    """
    rows = []
    for width, seconds in sorted(curve_by_width.items(), key=lambda kv: kv[0]):
        partition = group_partition(width)
        if partition is None:
            continue
        rows.append((width, partition, seconds * 1e3))
    if len(rows) < 5:
        return None

    nas = sorted({na for _, partition, _ in rows for na in partition})
    columns = ["fixed"] + [f"pass_na{na}" for na in nas]
    design = []
    target = []
    for _, partition, ms in rows:
        design.append([1.0] + [float(partition.count(na)) for na in nas])
        target.append(ms)

    size = len(columns)
    ata = [[sum(r[i] * r[j] for r in design) for j in range(size)]
           for i in range(size)]
    atb = [sum(r[i] * y for r, y in zip(design, target)) for i in range(size)]
    solution = _solve(ata, atb)
    if solution is None:
        return None

    coeffs = dict(zip(columns, solution))
    fitted = []
    for (width, partition, ms), row in zip(rows, design):
        pred = sum(c * v for c, v in zip(solution, row))
        fitted.append({
            "m": width, "partition": partition, "measured_ms": ms,
            "predicted_ms": pred, "residual_frac": (ms - pred) / ms,
        })

    passes = {}
    for na in nas:
        pass_ms = coeffs[f"pass_na{na}"]
        passes[str(na)] = {
            "pass_ms": pass_ms,
            "gb_per_s": TARGET_FORWARD_WEIGHT_MB / pass_ms,
            "frac_of_local_measured_roof": (
                TARGET_FORWARD_WEIGHT_MB / pass_ms / LOCAL_PEAK_GB_S),
            "frac_of_local_spec_roof":
                TARGET_FORWARD_WEIGHT_MB / pass_ms / 273.0,
            "ms_per_row_in_pass": pass_ms / na,
        }

    return {
        "model": "T(M) = fixed + sum over groups of pass(NA)",
        "fixed_ms": coeffs["fixed"],
        "passes": passes,
        "max_abs_residual_frac": max(abs(f["residual_frac"]) for f in fitted),
        "fit": fitted,
        "marginal_row_inside_a_pass_ms": {
            f"{a}->{b}": passes[str(b)]["pass_ms"] - passes[str(a)]["pass_ms"]
            for a, b in zip(nas, nas[1:])
        },
    }


def ranked_roofline():
    """Place the ranked prefill and decode rounds against both ranked roofs.

    Every input is a ranked measurement or checkpoint arithmetic. No local
    number enters, so Rule 83 does not apply to the conclusion.
    """
    weight_gb = TARGET_FORWARD_WEIGHT_MB / 1e3
    rows = {}

    prefill_s = 0.5271
    rows["prefill_512_rows"] = {
        "seconds": prefill_s,
        "rows": 512,
        "weight_stream_gb_per_s": weight_gb / prefill_s,
        "frac_of_bandwidth_roof": weight_gb / prefill_s / RANKED_PEAK_GB_S,
        "tflop_per_s": 512 * GFLOP_PER_PREFILL_ROW / prefill_s / 1e3,
        "frac_of_compute_roof": (512 * GFLOP_PER_PREFILL_ROW / prefill_s / 1e3
                                 / RANKED_PEAK_TFLOPS),
    }

    for name, seconds, width in (("beagle_round", 0.044983, 5.3818),
                                 ("essays_round", 0.048930, 6.0870)):
        rows[name] = {
            "seconds": seconds,
            "rows": width,
            "weight_stream_gb_per_s": weight_gb / seconds,
            "frac_of_bandwidth_roof": weight_gb / seconds / RANKED_PEAK_GB_S,
            "tflop_per_s": width * GFLOP_PER_ROW / seconds / 1e3,
            "frac_of_compute_roof": (width * GFLOP_PER_ROW / seconds / 1e3
                                     / RANKED_PEAK_TFLOPS),
        }

    # If the weight stream were confined to the M-independent term of the
    # ranked law, that term alone would have to carry the whole stream.
    fixed_s = 16.1585e-3
    return {
        "harness": "ranked",
        "weight_gb_per_forward": weight_gb,
        "bandwidth_roof_gb_per_s": RANKED_PEAK_GB_S,
        "compute_roof_tflop_per_s": RANKED_PEAK_TFLOPS,
        "rounds": rows,
        "stream_floor_seconds_at_roof": weight_gb / RANKED_PEAK_GB_S,
        "fixed_term_seconds": fixed_s,
        "implied_gb_per_s_if_stream_inside_fixed_term":
            weight_gb / fixed_s,
        "stream_fits_inside_fixed_term":
            weight_gb / fixed_s <= RANKED_PEAK_GB_S,
    }


def arithmetic_efficiency_by_width(scored_rows, byte_law):
    """Effective arithmetic rate of the routed kernel against the local roof.

    `fixed_cost_removed_tflops` already divides out the per-call fixed cost,
    so what remains is the rate at which the kernel turns resident weights
    into results. `NA` is the vector width the kernel is instantiated at, from
    `inputsPerGroup`, and it sets how many FLOP each dequantized nibble
    serves.
    """
    out = {}
    for width in sorted(byte_law, key=int):
        rates = [row["widths"][width]["fixed_cost_removed_tflops"]
                 for row in scored_rows if width in row["widths"]]
        if not rates:
            continue
        mean_rate = statistics.fmean(rates)
        na = INPUTS_PER_GROUP.get(int(width))
        out[width] = {
            "inputs_per_group": na,
            "mean_tflop_per_s": mean_rate,
            "spread_frac": (max(rates) - min(rates)) / mean_rate,
            "frac_of_local_compute_roof": mean_rate / LOCAL_PEAK_TFLOPS,
        }
    return out


def scored_shapes_against_byte_law(payload, byte_law):
    """Out-of-sample test of the byte law on the seven scored shapes.

    The law is fitted only on the synthetic de-confounding grid, so every
    scored shape here is a held-out point. `head.lm_head` at 715 MB is far
    outside the fitted range and is the strongest test of all.
    """
    rows = []
    for record in payload.get("shapes", []):
        mb = weight_bytes(record["k"], record["n"]) / 1e6
        cells = {}
        for row in record["rows"]:
            cells.setdefault(row["m"], []).append(row)
        widths = {}
        for m, group in sorted(cells.items()):
            law = byte_law.get(str(m))
            if law is None or not all(r["routed_to_custom_qmv"]
                                      for r in group):
                continue
            us = statistics.fmean(r["seconds_per_call"] for r in group) * 1e6
            pred = law["fixed_us"] + law["us_per_mb"] * mb
            flops = group[0]["flops_per_call"]
            widths[str(m)] = {
                "measured_us": us,
                "predicted_us": pred,
                "residual_frac": (us - pred) / us,
                "raw_tflops": flops / (us * 1e-6) / 1e12,
                "fixed_cost_removed_tflops": (
                    flops / (max(us - law["fixed_us"], 1e-9) * 1e-6) / 1e12
                ),
            }
        if widths:
            rows.append({
                "name": record["name"], "k": record["k"], "n": record["n"],
                "mb": mb, "widths": widths,
            })
    return rows


def analyze_bottom_up(payload):
    entries = list(payload.get("shapes", []))
    for key in ("gated_delta_recurrence", "top_two_readout"):
        if key in payload:
            entries.append(payload[key])

    shapes = {}
    total_row = 0.0
    total_group = 0.0
    total_intercept = 0.0
    curve_by_width = {}
    for entry in entries:
        by_m = {}
        for row in entry["rows"]:
            by_m.setdefault(row["m"], []).append(row["seconds_per_call"])
        curve = {m: statistics.mean(v) for m, v in sorted(by_m.items())}
        routed = entry.get("family") in {"mlp", "gdn", "attn", "lm_head"}
        points = [(m, t) for m, t in curve.items() if not (routed and m == 1)]
        fit = fit_row_law(points)
        if fit is None:
            continue
        calls = entry["calls_per_verify"]
        for m, seconds in curve.items():
            curve_by_width[m] = curve_by_width.get(m, 0.0) + seconds * calls
        total_row += fit["per_row_s"] * calls
        total_group += fit["per_group_s"] * calls
        total_intercept += fit["intercept_s"] * calls
        shapes[entry["name"]] = {
            "family": entry["family"],
            "calls_per_verify": calls,
            "per_call_row_s": fit["per_row_s"],
            "per_call_group_s": fit["per_group_s"],
            "per_call_intercept_s": fit["intercept_s"],
            "per_verify_row_s": fit["per_row_s"] * calls,
            "per_verify_group_s": fit["per_group_s"] * calls,
            "per_verify_intercept_s": fit["intercept_s"] * calls,
            "r2": fit["r2"],
            "rmse_frac_of_mean": (
                fit["rmse_s"] / statistics.mean([t for _, t in points])
                if points else None),
            "per_row_se_s": fit["per_row_se_s"] * calls,
            "excluded_width_one": routed,
            "width_one_s": curve.get(1),
            "routed_widths": sorted(
                {r["m"] for r in entry["rows"] if r.get("routed_to_custom_qmv")}),
            "k": entry.get("k"),
            "n": entry.get("n"),
        }

    families = {}
    for name, rec in shapes.items():
        slot = families.setdefault(
            rec["family"],
            {"per_verify_row_s": 0.0, "per_verify_group_s": 0.0,
             "per_verify_intercept_s": 0.0, "shapes": []})
        slot["per_verify_row_s"] += rec["per_verify_row_s"]
        slot["per_verify_group_s"] += rec["per_verify_group_s"]
        slot["per_verify_intercept_s"] += rec["per_verify_intercept_s"]
        slot["shapes"].append(name)
    for slot in families.values():
        slot["share_of_marginal_row"] = (
            slot["per_verify_row_s"] / total_row if total_row else None)
        slot["share_of_group_pass"] = (
            slot["per_verify_group_s"] / total_group if total_group else None)

    return {
        "shapes": shapes,
        "families": families,
        "bottom_up_marginal_row_s": total_row,
        "bottom_up_group_pass_s": total_group,
        "bottom_up_intercept_s": total_intercept,
        "modelled_verify_seconds_by_width": {
            str(m): t for m, t in sorted(curve_by_width.items())},
        "modelled_observed_marginals": observed_marginals(curve_by_width),
        "device": payload.get("device"),
        "custom_qmv_arm": payload.get("custom_qmv_arm"),
    }


# ------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-log", action="append", default=[])
    parser.add_argument("--bottom-up", default=None)
    parser.add_argument("--static-budget", default="research/out/e169/flop_budget.json")
    parser.add_argument("--out", default="research/out/e169/decomposition.json")
    args = parser.parse_args()

    report = {
        "experiment": "e169-marginal-verified-row-decomposition",
        "harness": "local",
        "row_law": "T(M) = a + b*M + c*(activeInputGroups(M)-1)",
        "inputs_per_group_table": INPUTS_PER_GROUP,
        "active_input_groups": {str(m): qmv_groups(m) for m in range(1, 10)},
    }

    if args.census_log:
        blocks = load_census_blocks(args.census_log)
        arms = analyze_census(blocks)
        baseline = arms.get("baseline", {}).get("seconds_by_width", {})
        report["in_situ"] = {
            "blocks_parsed": len(blocks),
            "arms": arms,
            "per_width_deltas": census_per_width_deltas(blocks),
            "attribution": census_shares(arms),
            "pass_cost_model": pass_cost_model(
                {int(w): s for w, s in baseline.items()}),
        }

    if args.bottom_up:
        payload = json.loads(Path(args.bottom_up).read_text())
        report["bottom_up"] = analyze_bottom_up(payload)
        deconfound = analyze_deconfound(payload)
        if deconfound:
            byte_law = deconfound["byte_law_by_width"]
            scored = scored_shapes_against_byte_law(payload, byte_law)
            report["deconfound"] = deconfound
            report["scored_shapes_vs_byte_law"] = scored
            report["stream_vs_arithmetic"] = stream_vs_arithmetic(byte_law)
            report["arithmetic_efficiency"] = arithmetic_efficiency_by_width(
                scored, byte_law)

    report["ranked_roofline"] = ranked_roofline()

    static_path = Path(args.static_budget)
    if static_path.exists():
        report["static_budget"] = json.loads(static_path.read_text())

    # Cross-estimator reconciliation. Two methods with different systematic
    # errors landing on the same number is the only real evidence here; a large
    # gap is a finding about the estimators, not a number to average away.
    in_situ = report.get("in_situ", {}).get("arms", {}).get("baseline")
    bottom = report.get("bottom_up")
    if in_situ and bottom:
        report["cross_estimator"] = {
            "in_situ_marginal_row_s": in_situ["per_row_s"],
            "bottom_up_marginal_row_s": bottom["bottom_up_marginal_row_s"],
            "row_ratio_bottom_up_over_in_situ": (
                bottom["bottom_up_marginal_row_s"] / in_situ["per_row_s"]
                if in_situ["per_row_s"] else None),
            "in_situ_group_pass_s": in_situ["per_group_s"],
            "bottom_up_group_pass_s": bottom["bottom_up_group_pass_s"],
            "group_ratio_bottom_up_over_in_situ": (
                bottom["bottom_up_group_pass_s"] / in_situ["per_group_s"]
                if in_situ["per_group_s"] else None),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    print(f"E169_ANALYSIS_OUT {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
