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
        report["in_situ"] = {
            "blocks_parsed": len(blocks),
            "arms": arms,
            "per_width_deltas": census_per_width_deltas(blocks),
            "attribution": census_shares(arms),
        }

    if args.bottom_up:
        report["bottom_up"] = analyze_bottom_up(
            json.loads(Path(args.bottom_up).read_text()))

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
