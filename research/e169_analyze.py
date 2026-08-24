#!/usr/bin/env python3
"""E169 marginal verified row decomposition: fit and reconcile both estimators.

harness=local for every measured number here. The in-situ census and the
bottom-up shape sweep are independent estimates of the same quantity -- the
seconds one extra verified row adds to one scored target forward -- so the
interesting output is not either number alone but their agreement.

In-situ model, fitted per arm over the measured widths:

    T(M) = a + b*M + c*(ceil(M/5) - 1)

`b` is the marginal row cost. `c` absorbs the SDPA >=5-row split in the 16
full-attention layers (AttentionUtils.swift `let split = 5`), which adds a
whole extra segment rather than a smooth per-row term; leaving it out would
smear a step function across the slope.

A family's share is read from the drop in `b` when that family alone is pinned
to one row, measured against the null control rather than the raw baseline so
the wrapper's own cost cancels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

BLOCK_PREFIX = "E169_BLOCK "
SDPA_SPLIT = 5


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


def fit_row_law(points, with_segment_term=True):
    """Least squares fit of T(M) = a + b*M + c*(ceil(M/5)-1) over (M, seconds).

    Falls back to a two-term fit when the widths never cross a split boundary,
    because a column that is identically zero makes the design singular.
    """
    widths = [m for m, _ in points]
    segs = [math.ceil(m / SDPA_SPLIT) - 1 for m in widths]
    use_seg = with_segment_term and len(set(segs)) > 1
    basis = [[1.0, float(m)] + ([float(s)] if use_seg else []) for m, s in zip(widths, segs)]
    ys = [t for _, t in points]

    n = len(basis[0])
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
    # Standard error of the slope from the (A^T A)^-1 diagonal.
    inv_col = _solve(ata, [1.0 if i == 1 else 0.0 for i in range(n)])
    se_b = math.sqrt(max(0.0, ss_res / dof * inv_col[1])) if inv_col else float("nan")

    return {
        "intercept_s": beta[0],
        "per_row_s": beta[1],
        "per_segment_s": beta[2] if use_seg else 0.0,
        "segment_term_fitted": use_seg,
        "per_row_se_s": se_b,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse_s": math.sqrt(ss_res / len(ys)),
        "n_points": len(ys),
    }


# ---------------------------------------------------- in-situ census input


def load_census_blocks(paths):
    blocks = []
    for path in paths:
        text = Path(path).read_text(errors="replace")
        for line in text.splitlines():
            index = line.find(BLOCK_PREFIX)
            if index < 0:
                continue
            try:
                blocks.append(json.loads(line[index + len(BLOCK_PREFIX):]))
            except json.JSONDecodeError:
                continue
    return blocks


def analyze_census(blocks):
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
        points = [(w, statistics.mean(v)) for w, v in sorted(by_width.items())]
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
            "widths": [w for w, _ in points],
            "seconds_by_width": {str(w): t for w, t in points},
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


def census_shares(arms):
    """Family share of the marginal row, referenced to the null control."""
    if "null" not in arms:
        return {"error": "null control missing; wrapper cost cannot be subtracted"}
    ref = arms["null"]["per_row_s"]
    baseline = arms.get("baseline", {}).get("per_row_s")

    out = {
        "reference_arm": "null",
        "reference_per_row_s": ref,
        "baseline_per_row_s": baseline,
        # A large null-vs-baseline gap means the wrapper itself is not free and
        # every share below is only as trustworthy as that subtraction.
        "wrapper_overhead_frac_of_baseline": (
            (ref - baseline) / baseline if baseline else None
        ),
        "families": {},
    }
    attributed = 0.0
    for arm, family in CENSUS_FAMILIES.items():
        if arm not in arms:
            continue
        drop = ref - arms[arm]["per_row_s"]
        se = math.hypot(arms[arm]["per_row_se_s"], arms["null"]["per_row_se_s"])
        out["families"][family] = {
            "arm": arm,
            "per_row_s": arms[arm]["per_row_s"],
            "marginal_s_attributed": drop,
            "share_of_marginal_row": drop / ref if ref else None,
            "se_s": se,
            "significant_2se": abs(drop) > 2 * se,
        }
        attributed += drop
    out["attributed_s"] = attributed
    out["attributed_share"] = attributed / ref if ref else None
    out["residual_s"] = ref - attributed
    out["residual_share"] = (ref - attributed) / ref if ref else None

    if "all_interceptable" in arms:
        joint = ref - arms["all_interceptable"]["per_row_s"]
        out["additivity"] = {
            "joint_arm_marginal_s": joint,
            "sum_of_single_arms_s": attributed,
            "difference_s": joint - attributed,
            "difference_frac_of_joint": (joint - attributed) / joint if joint else None,
        }
    return out


# --------------------------------------------------------- bottom-up input


def analyze_bottom_up(payload):
    entries = list(payload.get("shapes", []))
    for key in ("gated_delta_recurrence", "top_two_readout"):
        if key in payload:
            entries.append(payload[key])

    shapes = {}
    total = 0.0
    for entry in entries:
        by_m = {}
        for row in entry["rows"]:
            by_m.setdefault(row["m"], []).append(row["seconds_per_call"])
        points = [(m, statistics.mean(v)) for m, v in sorted(by_m.items())]
        # A single isolated shape has no SDPA split, so the segment term would
        # only fit noise here.
        fit = fit_row_law(points, with_segment_term=False)
        if fit is None:
            continue
        calls = entry["calls_per_verify"]
        per_row = fit["per_row_s"] * calls
        total += per_row
        shapes[entry["name"]] = {
            "family": entry["family"],
            "calls_per_verify": calls,
            "per_call_row_s": fit["per_row_s"],
            "per_call_intercept_s": fit["intercept_s"],
            "per_verify_row_s": per_row,
            "per_verify_intercept_s": fit["intercept_s"] * calls,
            "r2": fit["r2"],
            "per_row_se_s": fit["per_row_se_s"] * calls,
            "routed_to_custom_qmv": all(
                r.get("routed_to_custom_qmv", True) for r in entry["rows"]
            ),
            "k": entry.get("k"),
            "n": entry.get("n"),
        }

    families = {}
    for name, rec in shapes.items():
        slot = families.setdefault(rec["family"], {"per_verify_row_s": 0.0, "shapes": []})
        slot["per_verify_row_s"] += rec["per_verify_row_s"]
        slot["shapes"].append(name)
    for slot in families.values():
        slot["share_of_marginal_row"] = (
            slot["per_verify_row_s"] / total if total else None
        )

    return {
        "shapes": shapes,
        "families": families,
        "bottom_up_marginal_row_s": total,
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

    report = {"experiment": "e169-marginal-verified-row-decomposition", "harness": "local"}

    if args.census_log:
        blocks = load_census_blocks(args.census_log)
        arms = analyze_census(blocks)
        report["in_situ"] = {
            "blocks_parsed": len(blocks),
            "arms": arms,
            "attribution": census_shares(arms),
        }

    if args.bottom_up:
        report["bottom_up"] = analyze_bottom_up(
            json.loads(Path(args.bottom_up).read_text())
        )

    static_path = Path(args.static_budget)
    if static_path.exists():
        report["static_budget"] = json.loads(static_path.read_text())

    # Cross-estimator reconciliation. Two methods with different systematic
    # errors landing on the same number is the only real evidence here; a large
    # gap is a finding about the estimators, not a number to average away.
    in_situ = report.get("in_situ", {}).get("attribution", {}).get("reference_per_row_s")
    bottom = report.get("bottom_up", {}).get("bottom_up_marginal_row_s")
    if in_situ and bottom:
        report["cross_estimator"] = {
            "in_situ_marginal_row_s": in_situ,
            "bottom_up_marginal_row_s": bottom,
            "ratio_bottom_up_over_in_situ": bottom / in_situ,
            "difference_s": bottom - in_situ,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    print(f"E169_ANALYSIS_OUT {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
