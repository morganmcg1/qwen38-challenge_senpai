#!/usr/bin/env python3
"""E74 rung 0: working-threadgroup arithmetic, ordering checks, pre-registration.

    usage: research/e74_rung0.py [--report research/out/e71-census-r1/report.json]
                                 [--json research/out/e74-rung0.json]

No GPU work. Everything here is arithmetic over E71's measured census and over
the E33 microbenchmark table already in the campaign ledger. The predictions
this file emits are committed before the E74 census session runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
import subprocess
import sys

# --- shipped dispatch, read from source ---------------------------------------
#
# Host grid is frozen stock MLX:
#   Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:250-254
#     int bn = 8; MTL::Size group_dims(32, 2, 1);
#     MTL::Size grid_dims(M, (N + bn - 1) / bn, B);
# so the host always launches M * ceil(n/8) * B threadgroups, independent of IPG.
#
# The kernel retires the surplus x-groups immediately:
#   Vendor/.../kernels/quantized.h:1171-1174
#     const int first_m = int(tid.x) * IPG;
#     if (first_m >= M) { return; }
# so working (non-early-return) threadgroups = ceil(M/IPG) * ceil(n/8) * B.
# Every census family has B = 1, which is why the assignment's formula holds.
#
# IPG per width for out_vec_size >= 4096, quantized.h:1922-1978.
SHIPPED_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 4, 9: 5}
# For 1024 <= out_vec_size < 4096 the promoted pair kernel runs at every width,
# quantized.h:1980-2023, so IPG = 2 there. M == 1 and M >= 10 leave both
# switches for stock qmv_fast_impl, quantized.h:2026.
PAIR_BAND_IPG = 2
WIDE_BAND_MIN_N = 4096  # quantized.h:1918
PAIR_BAND_MIN_N = 1024  # quantized.h:1917

OUT_ROWS_PER_TG = 8  # quantized.h:1175, out_row = tid.y * 8 + simd_gid * 4
THREADS_PER_TG = 64  # quantized.cpp:252, group_dims(32, 2, 1)

# --- E33 microbenchmark, senpai/campaign-ledger.md item 130, table at :3457 ----
#
# The assignment cites this table as "ledger item 137". Item 137 (:3763) is the
# under-powered-null entry and carries no table; the eight-shape table is item
# 130 (:3379, table at :3457). The numbers quoted in the assignment are correct.
#
# The E33 arm went <T,6,3,true> -> <T,6,6,true,2>: IPG 3 -> 6 halves the working
# x-groups, and ROWS_PER_SIMD 4 -> 2 is a second, shape-independent change.
# Ledger 157 prices that second change at +10.54 % against +2.93 % for grid
# thinning. A per-shape fit must therefore carry a free level term C; only the
# shape-dependent part of the E33 ratio is usable as a grid signal.
E33_SHAPES = [
    # name, n, k, shipped working TGs, arm working TGs, observed cost ratio
    ("head.lm_head", 248320, 5120, 62080, 31040, 0.9830),
    ("head.compact_draft_vocab", 98336, 5120, 24584, 12292, 0.9868),
    ("mlp.gate_up_fused", 34816, 5120, 8704, 4352, 0.9941),
    ("linear_attn.in_proj", 16480, 5120, 4120, 2060, 0.9947),
    ("full_attn.qkv_proj", 14336, 5120, 3584, 1792, 1.0148),
    ("full_attn.o_proj", 5120, 6144, 1280, 640, 1.0414),
    ("linear_attn.out_proj", 5120, 6144, 1280, 640, 1.0492),
    ("mlp.down", 5120, 17408, 1280, 640, 1.0592),
]
E33_CORES = 20  # ledger :652, same host class as this one

CENSUS_FAMILIES = ["lm_head", "mlp_gate_up", "gdn_out_proj", "fa_o_proj", "mlp_down"]
NEW_WIDTHS = [7, 8]
OLD_WIDTHS = [4, 5, 6, 9]

# E71 cells that failed their width's null control. They are reported, never
# used as measurements. research/e71-results.md, rung 2 null table.
E71_NOT_RESOLVED = {(4, "fa_o_proj"), (4, "gdn_out_proj"), (4, "lm_head"),
                    (5, "fa_o_proj"), (5, "lm_head")}


def gpu_core_count() -> tuple[int, str]:
    """Read the GPU core count from the device. Never assume it."""
    out = subprocess.run(["ioreg", "-l"], capture_output=True, text=True,
                         errors="replace").stdout
    hits = re.findall(r'"gpu-core-count"\s*=\s*(\d+)', out)
    if hits:
        return int(hits[0]), 'ioreg -l "gpu-core-count"'
    out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                         capture_output=True, text=True).stdout
    hits = re.findall(r"Total Number of Cores:\s*(\d+)", out)
    if hits:
        return int(hits[0]), "system_profiler SPDisplaysDataType"
    raise SystemExit("e74_rung0: could not read the GPU core count from the device")


def ipg_for(m: int, n: int) -> int:
    if n >= WIDE_BAND_MIN_N:
        return SHIPPED_IPG[m]
    if n >= PAIR_BAND_MIN_N:
        return PAIR_BAND_IPG
    return 1  # stock qmv_fast_impl: one x-group does the whole reduction


def working_tgs(m: int, n: int) -> int:
    return math.ceil(m / ipg_for(m, n)) * math.ceil(n / 8)


def launched_tgs(m: int, n: int) -> int:
    return m * math.ceil(n / 8)


def kendall_tau_b(xs, ys):
    """Kendall tau-b. Ties are kept, not broken, and are counted in the norm."""
    n = len(xs)
    conc = disc = tx = ty = txy = 0
    for i, j in itertools.combinations(range(n), 2):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        if dx == 0 and dy == 0:
            txy += 1
        elif dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx * dy > 0:
            conc += 1
        else:
            disc += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - tx - txy) * (n0 - ty - txy))
    return {"tau_b": (conc - disc) / denom if denom else float("nan"),
            "concordant": conc, "discordant": disc,
            "ties_x_only": tx, "ties_y_only": ty, "ties_both": txy, "pairs": int(n0)}


def fit_overhead_model(shapes):
    """cost/byte = base * (1 + a/t). Halving t gives ratio C*(1+2a/t)/(1+a/t)."""
    best = None
    for a in [x / 4 for x in range(1, 40001)]:  # a in 0.25 .. 10000 TGs
        preds = [(1 + 2 * a / t_old) / (1 + a / t_old) for _, _, _, t_old, _, _ in shapes]
        obs = [r for *_, r in shapes]
        c = sum(o / p for o, p in zip(obs, preds)) / len(obs)
        sse = sum((o - c * p) ** 2 for o, p in zip(obs, preds))
        if best is None or sse < best["sse"]:
            best = {"a_tgs": a, "level_C": c, "sse": sse}
    t_flip = None
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        val = best["level_C"] * (1 + 2 * best["a_tgs"] / mid) / (1 + best["a_tgs"] / mid)
        if val > 1.0:
            lo = mid
        else:
            hi = mid
        t_flip = mid
    best["sign_flip_working_tgs"] = t_flip
    best["rmse"] = math.sqrt(best["sse"] / len(shapes))
    return best


def fit_hard_knee(shapes):
    """cost/byte penalty = s * max(0, ln(t_knee) - ln(t)); halving t moves along it."""
    def pen(t, knee):
        return max(0.0, math.log(knee) - math.log(t))

    best = None
    for knee in [200 * 1.02 ** i for i in range(300)]:  # 200 .. ~76000
        d = [pen(t_old / 2, knee) - pen(t_old, knee) for _, _, _, t_old, _, _ in shapes]
        obs = [math.log(r) for *_, r in shapes]
        mean_d, mean_o = statistics.fmean(d), statistics.fmean(obs)
        var = sum((x - mean_d) ** 2 for x in d)
        if var == 0:
            continue
        s = sum((x - mean_d) * (o - mean_o) for x, o in zip(d, obs)) / var
        c = mean_o - s * mean_d
        sse = sum((o - (c + s * x)) ** 2 for x, o in zip(d, obs))
        if best is None or sse < best["sse"]:
            best = {"knee_working_tgs": knee, "slope_s": s, "log_level_C": c, "sse": sse}
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        d = max(0.0, math.log(best["knee_working_tgs"]) - math.log(mid / 2)) - \
            max(0.0, math.log(best["knee_working_tgs"]) - math.log(mid))
        val = best["log_level_C"] + best["slope_s"] * d
        if val > 0:
            lo = mid
        else:
            hi = mid
    best["sign_flip_working_tgs"] = mid
    best["level_C"] = math.exp(best["log_level_C"])
    best["rmse_log"] = math.sqrt(best["sse"] / len(shapes))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="research/out/e71-census-r1/report.json")
    ap.add_argument("--json", default="research/out/e74-rung0.json")
    args = ap.parse_args()

    cores, core_source = gpu_core_count()
    e71 = json.load(open(args.report))
    shapes = e71["shapes"]

    out = {
        "experiment": "e74-in-situ-threadgroup-knee",
        "rung": 0,
        "harness": "local",
        "official_or_ranked_score": False,
        "device": {"gpu_cores": cores, "core_count_source": core_source,
                   "architecture": e71["identity"]["device"]["architecture"]},
        "dispatch": {"shipped_ipg_wide_band": SHIPPED_IPG,
                     "pair_band_ipg": PAIR_BAND_IPG,
                     "wide_band_min_n": WIDE_BAND_MIN_N,
                     "pair_band_min_n": PAIR_BAND_MIN_N,
                     "out_rows_per_threadgroup": OUT_ROWS_PER_TG,
                     "threads_per_threadgroup": THREADS_PER_TG,
                     "working_tgs_formula": "ceil(M/IPG) * ceil(n/8) * B, B=1 here",
                     "launched_tgs_formula": "M * ceil(n/8) * B"},
    }

    # --- 1. the grid table for every measured and planned cell ----------------
    grid = {}
    for fam in CENSUS_FAMILIES:
        n = shapes[fam]["n"]
        grid[fam] = {"n": n, "k": shapes[fam]["k"], "k_blocks": shapes[fam]["k_blocks"],
                     "gb": shapes[fam]["gb"], "calls": shapes[fam]["calls"],
                     "by_width": {}}
        for m in sorted(set(OLD_WIDTHS + NEW_WIDTHS)):
            w = working_tgs(m, n)
            grid[fam]["by_width"][m] = {
                "ipg": ipg_for(m, n), "groups": math.ceil(m / ipg_for(m, n)),
                "working_tgs": w, "tgs_per_core": w / cores,
                "launched_tgs": launched_tgs(m, n),
                "idle_tgs": launched_tgs(m, n) - w,
                "ms_per_gb": shapes[fam]["by_width"].get(str(m), {}).get("ms_per_gb"),
                "resolved": (m, fam) not in E71_NOT_RESOLVED
                            and str(m) in shapes[fam]["by_width"],
            }
    out["grid"] = grid

    # --- 2. ordering of measured ms/GB against working threadgroups -----------
    per_width = {}
    for m in OLD_WIDTHS:
        fams = [f for f in CENSUS_FAMILIES if grid[f]["by_width"][m]["ms_per_gb"] is not None]
        ms = [grid[f]["by_width"][m]["ms_per_gb"] for f in fams]
        per_width[m] = {
            "families": fams,
            "ms_per_gb": ms,
            "vs_working_tgs": kendall_tau_b([grid[f]["by_width"][m]["working_tgs"] for f in fams], ms),
            "vs_k_blocks": kendall_tau_b([grid[f]["k_blocks"] for f in fams], ms),
            "vs_n": kendall_tau_b([grid[f]["n"] for f in fams], ms),
            "vs_gb": kendall_tau_b([grid[f]["gb"] for f in fams], ms),
            "resolved_families": [f for f in fams if grid[f]["by_width"][m]["resolved"]],
        }
        res = per_width[m]["resolved_families"]
        if len(res) >= 3:
            msr = [grid[f]["by_width"][m]["ms_per_gb"] for f in res]
            per_width[m]["vs_working_tgs_resolved_only"] = kendall_tau_b(
                [grid[f]["by_width"][m]["working_tgs"] for f in res], msr)
        per_width[m]["spread_abs"] = max(ms) - min(ms)
        per_width[m]["spread_ratio"] = max(ms) / min(ms)
        per_width[m]["mean_ms_per_gb"] = statistics.fmean(ms)
    out["ordering_per_width"] = per_width

    # --- 3. E33 cross-check ---------------------------------------------------
    e33_tau = kendall_tau_b([s[3] for s in E33_SHAPES], [s[5] for s in E33_SHAPES])
    e33_tau_desc = kendall_tau_b([-s[3] for s in E33_SHAPES], [s[5] for s in E33_SHAPES])
    tied = [s for s in E33_SHAPES if s[3] == 1280]
    out["e33"] = {
        "ledger_citation": "senpai/campaign-ledger.md item 130, table at :3457",
        "assignment_cited_item_137_which_is_wrong": True,
        "shapes": [{"name": s[0], "n": s[1], "k": s[2], "k_blocks": s[2] // 64,
                    "shipped_tgs": s[3], "arm_tgs": s[4], "ratio": s[5],
                    "shipped_tgs_per_core": s[3] / E33_CORES,
                    "arm_tgs_per_core": s[4] / E33_CORES} for s in E33_SHAPES],
        "tau_ratio_vs_shipped_tgs": e33_tau,
        "tau_ratio_vs_negated_tgs": e33_tau_desc,
        "confound": ("the E33 arm changed IPG 3->6 and ROWS_PER_SIMD 4->2 together; "
                     "ledger 157 prices the row-blocking change at +10.54 % against "
                     "+2.93 % for grid thinning, so only the shape-dependent part of "
                     "the E33 ratio is a grid signal and the fits carry a free level"),
        "within_tie_residual": {
            "shapes": [s[0] for s in tied],
            "k_blocks": [s[2] // 64 for s in tied],
            "ratios": [s[5] for s in tied],
            "note": ("three shapes share 1280 shipped threadgroups and their ratios "
                     "still spread 1.8 %, ordered by k. A threadgroup-only model "
                     "cannot explain that residual; reduction depth can."),
        },
        "fit_overhead_1_over_t": fit_overhead_model(E33_SHAPES),
        "fit_hard_knee": fit_hard_knee(E33_SHAPES),
        "observed_sign_flip_between_working_tgs": [1792, 2060],
        "observed_sign_flip_between_tgs_per_core": [1792 / E33_CORES, 2060 / E33_CORES],
    }

    # --- 4. pre-registration --------------------------------------------------
    #
    # The level of ms/GB at a new width is a nuisance. Anchor it on E71's own
    # measured verify curve: mean ms/GB across the five families divided by the
    # measured width tax F(M)-F(1) is 0.0680 at M=6 and 0.0669 at M=9.
    curve = e71["rung1_curve"]
    f = {int(k): v["mean_ms"] for k, v in curve.items()}
    tax = {m: f[m] - f[1] for m in f}
    anchor = statistics.fmean([per_width[6]["mean_ms_per_gb"] / tax[6],
                               per_width[9]["mean_ms_per_gb"] / tax[9]])
    level = {m: anchor * tax[m] for m in (7, 8)}

    def d_stat(m):
        g = lambda fam: grid[fam]["by_width"][m]["ms_per_gb"]
        return statistics.fmean([g("fa_o_proj"), g("gdn_out_proj")]) - \
            statistics.fmean([g("lm_head"), g("mlp_gate_up")])

    def r_stat(m):
        g = lambda fam: grid[fam]["by_width"][m]["ms_per_gb"]
        return g("mlp_down") - statistics.fmean([g("fa_o_proj"), g("gdn_out_proj")])

    measured_d = {m: d_stat(m) for m in OLD_WIDTHS}
    measured_r = {m: r_stat(m) for m in OLD_WIDTHS}
    measured_dn = {m: measured_d[m] / per_width[m]["mean_ms_per_gb"] for m in OLD_WIDTHS}

    # Decompose the measured M=6 cross-family pattern into a grid term, a
    # reduction-depth term and a family residual, then step each term to M=7
    # and M=8 under the two hypotheses. The knee location comes from the E33
    # fit; the slopes come from this host's own M=6 census.
    knee = out["e33"]["fit_hard_knee"]["knee_working_tgs"]
    pen = lambda t: max(0.0, math.log(knee) - math.log(t))
    mean6 = per_width[6]["mean_ms_per_gb"]
    dev6 = {fam: grid[fam]["by_width"][6]["ms_per_gb"] / mean6 - 1.0 for fam in CENSUS_FAMILIES}
    pen6 = {fam: pen(grid[fam]["by_width"][6]["working_tgs"]) for fam in CENSUS_FAMILIES}
    pen78 = {fam: pen(grid[fam]["by_width"][7]["working_tgs"]) for fam in CENSUS_FAMILIES}
    # A: relative ms/GB penalty per unit log deficit below the knee, from D(6).
    a_grid = measured_dn[6] / (pen6["fa_o_proj"] - pen6["lm_head"])
    # B: relative ms/GB penalty per k-block, from R(6), at fixed n and fixed grid.
    b_depth = (measured_r[6] / mean6) / (grid["mlp_down"]["k_blocks"] - grid["fa_o_proj"]["k_blocks"])
    b_depth_low = (measured_r[9] / per_width[9]["mean_ms_per_gb"]) / \
        (grid["mlp_down"]["k_blocks"] - grid["fa_o_proj"]["k_blocks"])
    resid = {fam: dev6[fam] - a_grid * pen6[fam]
             - b_depth * (grid[fam]["k_blocks"] - 96) for fam in CENSUS_FAMILIES}
    resid_mean = statistics.fmean(resid.values())
    resid = {fam: resid[fam] - resid_mean for fam in CENSUS_FAMILIES}

    def scenario(m, grid_responds, cliff_resolves):
        b = b_depth_low if cliff_resolves else b_depth
        p = pen78 if grid_responds else pen6
        raw = {fam: 1.0 + a_grid * p[fam] + b * (grid[fam]["k_blocks"] - 96) + resid[fam]
               for fam in CENSUS_FAMILIES}
        scale = level[m] / statistics.fmean(raw.values())
        return {fam: round(raw[fam] * scale, 3) for fam in CENSUS_FAMILIES}

    scenarios = {
        "grid_responds_and_cliff_resolves": {str(m): scenario(m, True, True) for m in (7, 8)},
        "grid_responds_and_cliff_persists": {str(m): scenario(m, True, False) for m in (7, 8)},
        "no_grid_term_and_cliff_resolves": {str(m): scenario(m, False, True) for m in (7, 8)},
        "no_grid_term_and_cliff_persists": {str(m): scenario(m, False, False) for m in (7, 8)},
    }

    groups1_dn = statistics.fmean([measured_dn[m] for m in (4, 5, 6)])
    groups1_sd = statistics.stdev([measured_dn[m] for m in (4, 5, 6)])

    out["preregistration"] = {
        "written_before_any_e74_gpu_work": True,
        "level_anchor": {
            "method": "mean ms/GB across five families = anchor * (F(M)-F(1))",
            "anchor_ms_per_gb_per_ms": anchor,
            "measured_width_tax_ms": {str(m): tax[m] for m in (4, 5, 6, 7, 8, 9)},
            "predicted_mean_ms_per_gb": {str(m): level[m] for m in (7, 8)},
            "uncertainty": "+/- 10 %, from the 0.0680 against 0.0669 anchor spread",
        },
        "why_m7_and_m8": (
            "ceil(M/IPG) steps 1 -> 2 between M=6 and M=7 with no source edit, so "
            "every family's working threadgroup count doubles while n, k and bytes "
            "are unchanged. The step is confounded: IPG also falls 6 -> 4, so a "
            "register-pressure term moves at the same width. The two mechanisms are "
            "separated by which families respond, not by the aggregate level."
        ),
        "model": {
            "form": "ms/GB(fam,M) = L(M) * (1 + A*max(0, ln(knee) - ln(working_tgs)) + B*(k_blocks-96) + residual)",
            "knee_working_tgs_from_e33": knee,
            "A_grid_per_log_deficit": a_grid,
            "B_depth_per_k_block_at_m6": b_depth,
            "B_depth_per_k_block_at_m9": b_depth_low,
            "family_residual": resid,
            "note": ("A is calibrated on this host at M=6 and is 2.0x the slope the E33 "
                     "microbenchmark implies, s=0.0885. The knee location is E33's; the "
                     "magnitude is mine. If the two disagree the location survives and "
                     "the magnitude does not."),
        },
        "primary_statistic_D": {
            "definition": "mean ms/GB{fa_o_proj, gdn_out_proj} - mean ms/GB{lm_head, mlp_gate_up}",
            "why": ("both pairs sit at 80 to 96 k-blocks, so D is nearly free of the "
                    "reduction-depth term. The pairs differ 6.8x to 48x in n and so in "
                    "working threadgroups. D is the grid contrast, and it excludes "
                    "mlp_down entirely, so no part of it can be the IPG=6 register story."),
            "measured": measured_d,
            "measured_normalised_by_width_mean": measured_dn,
            "groups1_mean_D_over_level": groups1_dn,
            "groups1_sd_D_over_level": groups1_sd,
            "groups2_measured_D_over_level_m9": measured_dn[9],
            "H_knee_prediction": {
                "D_over_level_7_8": [0.02, 0.17],
                "point": a_grid * pen78["fa_o_proj"],
                "D_7_ms_per_gb": [0.10, 0.85], "D_8_ms_per_gb": [0.11, 0.97],
                "basis": ("doubling the working threadgroups of the n=5120 families cuts "
                          "their log deficit below the knee from 1.293 to 0.599, so D/level "
                          "falls by 54 %. M=9, the only measured groups=2 width, gives 0.061."),
            },
            "H_null_prediction": {
                "D_over_level_7_8": [0.20, 0.42],
                "point": groups1_dn,
                "D_7_ms_per_gb": [0.99, 2.09], "D_8_ms_per_gb": [1.14, 2.40],
                "basis": ("the three groups=1 widths give D/level = 0.256, 0.410, 0.224, "
                          "mean 0.297. With no grid term D/level stays in that band while "
                          "the level rises, so D itself grows."),
            },
            "falsifies_the_knee": (
                "D/level at 0.20 or above at both M=7 and M=8. That says doubling the "
                "working threadgroup count of the n=5120 families does not reduce their "
                "per-byte penalty, so the M=9 collapse is a width effect and the grid "
                "term does not exist in situ."),
            "unresolved_band": "D/level between 0.17 and 0.20, or the two new widths disagreeing across that boundary",
            "e33_magnitude_check": {
                "e33_implied_D_over_level_7_8": measured_dn[6] - 0.0885 * (pen6["fa_o_proj"] - pen78["fa_o_proj"]),
                "note": ("if the E33 slope is the true magnitude and the rest of D(6) is "
                         "some other n-dependent term, D/level lands near 0.16, inside my "
                         "unresolved band. I record that risk before measuring: this "
                         "session can confirm a strong knee or falsify any knee, and a "
                         "weak E33-sized knee is the outcome it cannot separate."),
            },
        },
        "secondary_statistic_R": {
            "definition": "ms/GB{mlp_down} - mean ms/GB{fa_o_proj, gdn_out_proj}",
            "why": ("all three share n=5120 and therefore share the working threadgroup "
                    "count at every width. R is free of the grid term by construction and "
                    "isolates reduction depth, 272 k-blocks against 96."),
            "measured": measured_r,
            "H_register_cliff_prediction": {
                "R_7": [-0.20, 0.60], "R_8": [-0.20, 0.70],
                "basis": "M=6 at IPG=6 is the only anomalous width. R(4)=0.389, R(5)=-0.034, R(9)=0.192 against R(6)=1.263"},
            "H_depth_is_width_driven_prediction": {
                "R_7": [1.30, 2.00], "R_8": [1.50, 2.30],
                "basis": "if the depth penalty rises with the level rather than with IPG, R keeps growing"},
        },
        "per_family_point_predictions": scenarios,
        "dispersion_prediction": {
            "measured_absolute_spread": {str(m): per_width[m]["spread_abs"] for m in OLD_WIDTHS},
            "measured_groups": {str(m): math.ceil(m / SHIPPED_IPG[m]) for m in OLD_WIDTHS},
            "H_knee_absolute_spread_7_8": [0.40, 1.60],
            "H_null_absolute_spread_7_8": [2.40, 3.60],
            "status": ("confirmatory only. The spread mixes the grid term and the depth "
                       "term, so it cannot separate them. D and R are the decisive "
                       "statistics and they are orthogonal by construction."),
        },
        "stop_rules_from_the_assignment": [
            "stop if rung 0 shows the four existing widths contradict the E33 ordering",
            "stop if the rung 2 positive control fails",
            "stop if the rung 1 null control fails at both new widths",
        ],
        "expected_noise": {
            "session_null_ms": e71["session_null_ms"],
            "note": ("a 0.1 ms block error is 0.35 ms/GB on fa_o_proj at 0.283 GB but "
                     "only 0.12 ms/GB on gdn_out_proj and 0.03 ms/GB on mlp_down. D "
                     "carries half of the fa_o_proj error, so its floor is about "
                     "0.19 ms/GB. The H bands above are wider than that floor."),
        },
    }

    json.dump(out, open(args.json, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("device", "dispatch")}, indent=1))
    print(f"\nwrote {args.json}")
    return 0




if __name__ == "__main__":
    sys.exit(main())
