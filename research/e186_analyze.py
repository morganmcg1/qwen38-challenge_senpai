#!/usr/bin/env python3
"""E186 -- fit per-family width curves and attribute E177's M^2 round term.

Reads the probe JSON written by `E186DecodeWidthProbeTests.widthSweep`, fits
`us(m) = a + b*m + c*m^2` per family cell, reconstructs a decode round from the
per-layer invocation counts, and compares the reconstructed quadratic term with
E177's fitted round model.

    R(M) = 30.640 - 0.939*M + 0.576*M^2 ms      (M = draft count)
         = 32.155 - 2.091*m + 0.576*m^2 ms      (m = M + 1 evaluated rows)

harness=local. No thermal gate, no score.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

# E177 round model, in evaluated-row form.
E177_A_M, E177_B_M, E177_C_M = 32.155, -2.091, 0.576

# Invocations of each timed family in one decode round of the 64-layer target:
# 48 Gated DeltaNet layers, 16 full-attention layers, 64 MLPs, one readout.
CELL_COUNTS = {
    "mlp.gate_up": 64,
    "mlp.down": 64,
    "gdn.in_proj": 48,
    "gdn.out_proj": 48,
    "fa.qkv": 16,
    "fa.o_proj": 16,
    "lm_head": 1,
}


def quad_fit(m, y):
    """OLS fit of y = a + b*m + c*m^2 with t-based 95% CIs per coefficient."""
    x = np.vstack([np.ones_like(m), m, m * m]).T
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - 3
    if dof <= 0:
        return beta, np.full(3, math.nan), math.nan, math.nan
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    # 97.5th percentile of the t distribution, adequate for dof >= 8.
    tcrit = 1.96 + 2.4 / dof
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else math.nan
    return beta, tcrit * se, r2, math.sqrt(s2)


def cell_key(sample):
    return (sample["family"], sample["cell"])


def summarize(payload, floor_subtract):
    floor = payload.get("eval_floor_microseconds", 0.0) if floor_subtract else 0.0
    by_cell = defaultdict(lambda: defaultdict(list))
    for s in payload["samples"]:
        by_cell[cell_key(s)][s["m"]].append(max(s["microseconds"] - floor, 0.0))

    out = {}
    for key, per_m in by_cell.items():
        widths = sorted(per_m)
        m = np.array([w for w in widths for _ in per_m[w]], dtype=float)
        y = np.array([v for w in widths for v in per_m[w]], dtype=float)
        beta, ci, r2, rmse = quad_fit(m, y)
        out[key] = {
            "family": key[0],
            "cell": key[1],
            "a_us": beta[0],
            "b_us_per_m": beta[1],
            "c_us_per_m2": beta[2],
            "a_ci95": ci[0],
            "b_ci95": ci[1],
            "c_ci95": ci[2],
            "r2": r2,
            "rmse_us": rmse,
            "n_per_cell": {str(w): len(per_m[w]) for w in widths},
            "mean_us": {str(w): float(np.mean(per_m[w])) for w in widths},
            "sd_us": {str(w): float(np.std(per_m[w], ddof=1)) for w in widths},
            "min_us": {str(w): float(np.min(per_m[w])) for w in widths},
        }
    return out, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", help="probe JSON from widthSweep")
    ap.add_argument("--dispatch", help="dispatch JSON from dispatchProof")
    ap.add_argument("--out", help="write the analysis JSON here")
    ap.add_argument("--wandb-project",
                    default="wandb-applied-ai-team/qwen38-mlx-challenge-senpai")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--keep-eval-floor", action="store_true",
                    help="do not subtract the measured per-eval floor")
    args = ap.parse_args()

    with open(args.probe) as f:
        payload = json.load(f)

    fits, floor = summarize(payload, not args.keep_eval_floor)

    # Layer-level reconstruction: mixers + MLPs + readout, no double counting.
    layer_total = np.zeros(3)
    layer_ci2 = np.zeros(3)
    layer_rows = []
    for key, n in [(("gdn_layer", "nConfirmed=1"), 48), (("mlp", "fused"), 64),
                   (("qmv_inpath", "lm_head"), 1)]:
        fit = fits.get(key)
        if fit is None:
            continue
        coef = np.array([fit["a_us"], fit["b_us_per_m"], fit["c_us_per_m2"]])
        ci = np.array([fit["a_ci95"], fit["b_ci95"], fit["c_ci95"]])
        layer_total += n * coef
        layer_ci2 += (n * ci) ** 2
        layer_rows.append({"key": "/".join(key), "invocations": n,
                           "c_us_per_m2": fit["c_us_per_m2"],
                           "c_round_ms_per_m2": n * fit["c_us_per_m2"] / 1e3})
    # Full attention: whichever kv length was measured, both reported.
    for key, fit in fits.items():
        if key[0] != "fa_layer":
            continue
        coef = np.array([fit["a_us"], fit["b_us_per_m"], fit["c_us_per_m2"]])
        layer_rows.append({"key": "/".join(key), "invocations": 16,
                           "c_us_per_m2": fit["c_us_per_m2"],
                           "c_round_ms_per_m2": 16 * fit["c_us_per_m2"] / 1e3})
    fa_keys = sorted(k for k in fits if k[0] == "fa_layer")
    if fa_keys:
        fit = fits[fa_keys[0]]
        coef = np.array([fit["a_us"], fit["b_us_per_m"], fit["c_us_per_m2"]])
        ci = np.array([fit["a_ci95"], fit["b_ci95"], fit["c_ci95"]])
        layer_total += 16 * coef
        layer_ci2 += (16 * ci) ** 2
    layer_total /= 1e3
    layer_ci = np.sqrt(layer_ci2) / 1e3

    # Cell-level reconstruction from the routed QMV arms alone.
    cell_total = np.zeros(3)
    cell_ci2 = np.zeros(3)
    cell_rows = []
    for cell, n in CELL_COUNTS.items():
        fit = fits.get(("qmv_inpath", cell))
        if fit is None:
            continue
        coef = np.array([fit["a_us"], fit["b_us_per_m"], fit["c_us_per_m2"]])
        ci = np.array([fit["a_ci95"], fit["b_ci95"], fit["c_ci95"]])
        cell_total += n * coef
        cell_ci2 += (n * ci) ** 2
        cell_rows.append({"cell": cell, "invocations": n,
                          "c_us_per_m2": fit["c_us_per_m2"],
                          "c_round_ms_per_m2": n * fit["c_us_per_m2"] / 1e3})
    cell_total /= 1e3
    cell_ci = np.sqrt(cell_ci2) / 1e3

    analysis = {
        "harness": "local",
        "cool_gate_passed_real_gate": payload.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": payload.get("gate_qualified_for_timing"),
        "official_or_ranked_score": False,
        "eval_floor_microseconds": payload.get("eval_floor_microseconds"),
        "eval_floor_subtracted": not args.keep_eval_floor,
        "qmv_arm": payload.get("qmv_arm"),
        "active_input_groups": payload.get("active_input_groups"),
        "blocks": payload.get("blocks"),
        "e177_round_model_m_form": {"a_ms": E177_A_M, "b_ms": E177_B_M,
                                    "c_ms": E177_C_M},
        "fits": [dict(v) for v in fits.values()],
        "layer_reconstruction": {
            "a_ms": layer_total[0], "b_ms": layer_total[1],
            "c_ms": layer_total[2],
            "a_ci95_ms": layer_ci[0], "b_ci95_ms": layer_ci[1],
            "c_ci95_ms": layer_ci[2],
            "c_share_of_e177": layer_total[2] / E177_C_M,
            "rows": layer_rows,
        },
        "cell_reconstruction": {
            "a_ms": cell_total[0], "b_ms": cell_total[1], "c_ms": cell_total[2],
            "a_ci95_ms": cell_ci[0], "b_ci95_ms": cell_ci[1],
            "c_ci95_ms": cell_ci[2],
            "c_share_of_e177": cell_total[2] / E177_C_M,
            "rows": cell_rows,
        },
        "temperatures": payload.get("temperatures"),
    }

    # E182 reconciliation: isolated cost and reconstructed round share at the
    # three widths E182 instruments in path.
    recon = []
    for key, fit in fits.items():
        n = None
        if key[0] == "qmv_inpath":
            n = CELL_COUNTS.get(key[1])
        elif key == ("gdn_layer", "nConfirmed=1"):
            n = 48
        elif key[0] == "fa_layer":
            n = 16
        elif key == ("mlp", "fused"):
            n = 64
        if not n:
            continue
        row = {"key": "/".join(key), "invocations": n}
        for m in ("1", "4", "8"):
            us = fit["mean_us"].get(m)
            if us is None:
                continue
            row[f"us_m{m}"] = us
            row[f"round_ms_m{m}"] = n * us / 1e3
            mm = float(m)
            round_total = E177_A_M + E177_B_M * mm + E177_C_M * mm * mm
            row[f"round_share_m{m}"] = n * us / 1e3 / round_total
        recon.append(row)
    analysis["e182_reconciliation"] = recon

    if args.dispatch:
        with open(args.dispatch) as f:
            analysis["dispatch"] = json.load(f)

    text = json.dumps(analysis, indent=2, sort_keys=True, default=float)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)

    if not args.no_wandb:
        import wandb

        entity, project = args.wandb_project.split("/", 1)
        run = wandb.init(entity=entity, project=project,
                         name=args.wandb_name or "e186-decode-width-probes",
                         job_type="measurement",
                         tags=["e186", "harness=local", "measurement"],
                         config={
                             "harness": "local",
                             "gate_qualified_for_timing": False,
                             "cool_gate_passed_real_gate": False,
                             "official_or_ranked_score": False,
                             "eval_floor_us": payload.get(
                                 "eval_floor_microseconds"),
                             "blocks": payload.get("blocks"),
                             "qmv_arm": payload.get("qmv_arm"),
                             "widths": payload.get("widths"),
                             "kv_lengths": payload.get("kv_lengths"),
                         })
        raw = wandb.Table(columns=["family", "cell", "m", "block", "ascending",
                                   "microseconds", "reps"])
        for s in payload["samples"]:
            raw.add_data(s["family"], s["cell"], s["m"], s["block"],
                         s["ascending"], s["microseconds"], s["reps"])
        fit_table = wandb.Table(columns=[
            "family", "cell", "a_us", "b_us_per_m", "c_us_per_m2", "c_ci95",
            "r2", "rmse_us"])
        for fit in fits.values():
            fit_table.add_data(fit["family"], fit["cell"], fit["a_us"],
                               fit["b_us_per_m"], fit["c_us_per_m2"],
                               fit["c_ci95"], fit["r2"], fit["rmse_us"])
        run.log({"e186/raw_samples": raw, "e186/fits": fit_table})
        run.summary.update({
            "layer_c_ms_per_m2": layer_total[2],
            "layer_c_share_of_e177": layer_total[2] / E177_C_M,
            "cell_c_ms_per_m2": cell_total[2],
            "cell_c_share_of_e177": cell_total[2] / E177_C_M,
            "e177_c_ms_per_m2": E177_C_M,
        })
        print(f"wandb run: {run.url} id={run.id}")
        run.finish()


if __name__ == "__main__":
    main()
