#!/usr/bin/env python3
"""E186 -- per-family decode-width curves and the M^2 attribution.

Reads one or more probe JSON files written by
`E186DecodeWidthProbeTests.widthSweep` and produces:

  * raw us(m) per family cell (per-block minimum, mean, sd, n);
  * a quadratic fit a + b*m + c*m^2 per cell with t-based 95% CIs;
  * the staircase decomposition (plateau ramp vs step at the G boundaries);
  * layer-minus-parts closure for the GDN, MLP and FA layers;
  * a decode-round reconstruction weighted by invocation counts;
  * an admissibility test of the staircase form against the ranked anchors;
  * the E182 reconciliation table.

E177 ranked round law (harness=ranked, FINDING 456), M = draft count,
m = M + 1 evaluated rows:

    R(M) = 30.640 - 0.939*M + 0.576*M^2 ms
    R(m) = 32.155 - 2.091*m + 0.576*m^2 ms

Every measurement in this file is harness=local, ungated, and is not a score.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

WIDTHS = list(range(1, 10))

# Ranked law in row form.
E177 = (32.155, -2.091, 0.576)
# The ranked anchors the law was fitted through: two paid controlled receipts
# at cap-4 and cap-7 plus the held-out serial floor. M -> m = M + 1.
RANKED_ANCHORS_M = [0, 4, 7]

# True decode round from E185 / FINDING 482.
ROUND_MS = 145.963
MUE_MS = 0.567

# One decode round of the 64-layer target.
LAYER_COUNTS = {
    ("gdn_layer", "nConfirmed=1"): 48,
    ("fa_layer", "kv=1024"): 16,
    ("mlp", "fused"): 64,
    ("qmv_inpath", "lm_head"): 1,
}
CELL_COUNTS = {
    "mlp.gate_up": 64,
    "mlp.down": 64,
    "gdn.in_proj": 48,
    "gdn.out_proj": 48,
    "fa.qkv": 16,
    "fa.o_proj": 16,
    "lm_head": 1,
}


def e177_ms(m):
    a, b, c = E177
    return a + b * m + c * m * m


def load(paths):
    """Pool per-block samples from every session, keyed by (family, cell, m)."""
    samples = defaultdict(lambda: defaultdict(list))
    meta = []
    for path in paths:
        with open(path) as f:
            payload = json.load(f)
        meta.append({
            "path": path,
            "harness": payload.get("harness"),
            "cool_gate_passed_real_gate": payload.get(
                "cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": payload.get(
                "gate_qualified_for_timing"),
            "official_or_ranked_score": payload.get("official_or_ranked_score"),
            "blocks": payload.get("blocks"),
            "warmup_evals_per_cell": payload.get("warmup_evals_per_cell"),
            "eval_floor_entry_us": payload.get("eval_floor_entry_microseconds"),
            "eval_floor_exit_us": payload.get("eval_floor_microseconds"),
            "qmv_arm": payload.get("qmv_arm"),
            "active_input_groups": payload.get("active_input_groups"),
            "temperatures": payload.get("temperatures"),
            "n_samples": len(payload.get("samples", [])),
        })
        for s in payload["samples"]:
            samples[(s["family"], s["cell"])][s["m"]].append(s["microseconds"])
    return samples, meta


def quad_fit(m, y):
    """OLS a + b*m + c*m^2 with t-based 95% CIs."""
    x = np.vstack([np.ones_like(m), m, m * m]).T
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - 3
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(x.T @ x)))
    tcrit = 1.96 + 2.4 / dof
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else math.nan
    return beta, tcrit * se, r2


def curves(samples):
    """Per-cell estimator table. The per-block minimum is the headline: it is
    the estimator that survives the cross-session reproducibility check."""
    out = {}
    for key, per_m in samples.items():
        mins = np.array([min(per_m[m]) for m in WIDTHS])
        means = np.array([float(np.mean(per_m[m])) for m in WIDTHS])
        sds = np.array([float(np.std(per_m[m], ddof=1)) for m in WIDTHS])
        n = [len(per_m[m]) for m in WIDTHS]
        mv = np.array([m for m in WIDTHS for _ in per_m[m]], dtype=float)
        yv = np.array([v for m in WIDTHS for v in per_m[m]], dtype=float)
        beta, ci, r2 = quad_fit(mv, yv)
        beta_min, ci_min, r2_min = quad_fit(
            np.array(WIDTHS, dtype=float), mins)
        out[key] = {
            "family": key[0], "cell": key[1],
            "min_us": mins.tolist(), "mean_us": means.tolist(),
            "sd_us": sds.tolist(), "n": n,
            "fit_on_samples": {"a": beta[0], "b": beta[1], "c": beta[2],
                               "a_ci95": ci[0], "b_ci95": ci[1],
                               "c_ci95": ci[2], "r2": r2},
            "fit_on_minima": {"a": beta_min[0], "b": beta_min[1],
                              "c": beta_min[2], "a_ci95": ci_min[0],
                              "b_ci95": ci_min[1], "c_ci95": ci_min[2],
                              "r2": r2_min},
            # Staircase decomposition at the two G boundaries proven in the
            # dispatch section: G steps 1->2 at m=6 and 2->3 at m=9.
            "step_m6_us": float(mins[5] - mins[4]),
            "step_m9_us": float(mins[8] - mins[7]),
            "plateau_slope_m6_m8_us_per_row": float((mins[7] - mins[5]) / 2.0),
            "plateau_slope_m1_m3_us_per_row": float((mins[2] - mins[0]) / 2.0),
            "delta_from_m1_us": (mins - mins[0]).tolist(),
        }
    return out


def reconstruct(cur):
    """Round-level curve from the layer families, absolute and width-delta."""
    total = np.zeros(len(WIDTHS))
    rows = []
    for key, n in LAYER_COUNTS.items():
        fit = cur.get(key)
        if fit is None:
            continue
        v = np.array(fit["min_us"])
        total += n * v
        rows.append({
            "family": "/".join(key), "invocations": n,
            "abs_ms": (n * v / 1e3).tolist(),
            "delta_ms": (n * (v - v[0]) / 1e3).tolist(),
        })
    delta = total - total[0]
    e177 = np.array([e177_ms(m) for m in WIDTHS])
    e177_delta = e177 - e177[0]
    shares = {}
    for row in rows:
        d = np.array(row["delta_ms"])
        shares[row["family"]] = (
            float(d[7] / (delta[7] / 1e3)) if delta[7] else 0.0)
    return {
        "abs_ms": (total / 1e3).tolist(),
        "delta_ms": (delta / 1e3).tolist(),
        "e177_abs_ms": e177.tolist(),
        "e177_delta_ms": e177_delta.tolist(),
        "overshoot_factor_delta": [
            float(delta[i] / 1e3 / e177_delta[i]) if e177_delta[i] else None
            for i in range(len(WIDTHS))],
        "delta_share_at_m8": shares,
        "rows": rows,
    }


def admissibility(recon):
    """Can a two-parameter affine rescaling of the measured isolated shape
    reproduce the three ranked anchors that E177's quadratic was fitted
    through? If yes, the staircase is admissible on the ranked evidence and
    the smooth law is not uniquely determined by it."""
    shape = np.array(recon["abs_ms"])
    anchor_m = [M + 1 for M in RANKED_ANCHORS_M]
    idx = [m - 1 for m in anchor_m]
    y = np.array([e177_ms(m) for m in anchor_m])
    x = np.vstack([np.ones(len(idx)), shape[idx]]).T
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred_anchor = x @ beta
    resid = y - pred_anchor
    full = beta[0] + beta[1] * shape
    return {
        "anchor_M": RANKED_ANCHORS_M,
        "anchor_m": anchor_m,
        "anchor_e177_ms": y.tolist(),
        "alpha_ms": float(beta[0]),
        "beta": float(beta[1]),
        "anchor_pred_ms": pred_anchor.tolist(),
        "anchor_residual_ms": resid.tolist(),
        "max_abs_residual_ms": float(np.abs(resid).max()),
        "max_abs_residual_in_MUE": float(np.abs(resid).max() / MUE_MS),
        "staircase_round_ms": full.tolist(),
        "e177_round_ms": [e177_ms(m) for m in WIDTHS],
        "disagreement_ms": [float(full[i] - e177_ms(WIDTHS[i]))
                            for i in range(len(WIDTHS))],
    }


def closure(cur, layer_key, part_keys):
    """Does the layer's width dependence equal the sum of its parts'?
    Deltas from m=1 cancel the per-call eval overhead each isolated arm pays."""
    layer = np.array(cur[layer_key]["delta_from_m1_us"])
    parts = np.zeros(len(WIDTHS))
    for key in part_keys:
        parts += np.array(cur[key]["delta_from_m1_us"])
    return {
        "layer": "/".join(layer_key),
        "parts": ["/".join(k) for k in part_keys],
        "layer_delta_us": layer.tolist(),
        "parts_delta_us": parts.tolist(),
        "remainder_us": (layer - parts).tolist(),
        "remainder_fraction_at_m9": float((layer[8] - parts[8]) / layer[8]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", nargs="+", help="probe JSON files to pool")
    ap.add_argument("--dispatch", help="dispatch JSON from dispatchProof")
    ap.add_argument("--out", required=True, help="write the analysis JSON here")
    ap.add_argument("--wandb-project",
                    default="wandb-applied-ai-team/qwen38-mlx-challenge-senpai")
    ap.add_argument("--wandb-name", default="e186-decode-width-probes")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    samples, meta = load(args.probe)
    cur = curves(samples)
    recon = reconstruct(cur)
    adm = admissibility(recon)

    gdn_closure = closure(
        cur, ("gdn_layer", "nConfirmed=1"),
        [("qmv_inpath", "gdn.in_proj"), ("qmv_inpath", "gdn.out_proj"),
         ("gdn_recurrence", "T=m")])
    mlp_closure = closure(
        cur, ("mlp", "fused"),
        [("qmv_inpath", "mlp.gate_up"), ("qmv_inpath", "mlp.down")])
    fa_closure = closure(
        cur, ("fa_layer", "kv=1024"),
        [("qmv_inpath", "fa.qkv"), ("qmv_inpath", "fa.o_proj"),
         ("sdpa", "kv=1024")])

    # widthSixWall: the SDPA-only step at m=6, priced per round over 16 layers.
    width_six_wall = {}
    for kv in ("kv=512", "kv=1024"):
        v = np.array(cur[("sdpa", kv)]["min_us"])
        width_six_wall[kv] = {
            "step_m5_to_m6_us": float(v[5] - v[4]),
            "flat_m6_to_m9_us": float(v[8] - v[5]),
            "round_cost_of_step_ms": float(16 * (v[5] - v[4]) / 1e3),
            "round_share_of_step": float(16 * (v[5] - v[4]) / 1e3 / ROUND_MS),
            "step_in_MUE": float(16 * (v[5] - v[4]) / 1e3 / MUE_MS),
        }

    # E182 reconciliation at the widths E182 instruments in path.
    over = recon["overshoot_factor_delta"][7]
    e182 = []
    for key, fit in sorted(cur.items()):
        n = LAYER_COUNTS.get(key) or (
            CELL_COUNTS.get(key[1]) if key[0] == "qmv_inpath" else None)
        if not n:
            continue
        v = fit["min_us"]
        row = {"family": "/".join(key), "invocations": n}
        for m in (1, 4, 8):
            row[f"us_m{m}"] = v[m - 1]
            row[f"round_ms_m{m}"] = n * v[m - 1] / 1e3
            row[f"round_share_m{m}"] = n * v[m - 1] / 1e3 / ROUND_MS
        # In-path band: scale the isolated width delta down by the measured
        # reconstruction overshoot, widened by +/-25% for transfer risk.
        row["inpath_delta_band_ms_m1_to_m8"] = [
            n * (v[7] - v[0]) / 1e3 / (over * 1.25),
            n * (v[7] - v[0]) / 1e3 / (over * 0.8)]
        e182.append(row)

    analysis = {
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "estimator": "per-block minimum pooled across sessions",
        "round_ms": ROUND_MS,
        "mue_ms": MUE_MS,
        "e177_row_form": {"a": E177[0], "b": E177[1], "c": E177[2]},
        "sessions": meta,
        "widths": WIDTHS,
        "curves": list(cur.values()),
        "round_reconstruction": recon,
        "staircase_admissibility": adm,
        "closure": [gdn_closure, mlp_closure, fa_closure],
        "width_six_wall": width_six_wall,
        "e182_reconciliation": e182,
    }
    if args.dispatch:
        with open(args.dispatch) as f:
            analysis["dispatch"] = json.load(f)

    with open(args.out, "w") as f:
        json.dump(analysis, f, indent=2, sort_keys=True, default=float)

    print(f"round reconstruction abs ms : "
          f"{[round(v, 2) for v in recon['abs_ms']]}")
    print(f"E177 round ms               : "
          f"{[round(v, 2) for v in recon['e177_abs_ms']]}")
    print("delta overshoot factor      : "
          f"{[None if v is None else round(v, 2) for v in recon['overshoot_factor_delta']]}")
    print("delta share at m=8          : "
          f"{ {k: round(v, 3) for k, v in recon['delta_share_at_m8'].items()} }")
    print(f"staircase anchor residual ms: "
          f"{[round(v, 3) for v in adm['anchor_residual_ms']]} "
          f"(max {adm['max_abs_residual_ms']:.3f} ms = "
          f"{adm['max_abs_residual_in_MUE']:.2f} MUE)")
    print(f"staircase round ms          : "
          f"{[round(v, 2) for v in adm['staircase_round_ms']]}")
    print(f"disagreement vs E177 ms     : "
          f"{[round(v, 2) for v in adm['disagreement_ms']]}")
    print(f"widthSixWall                : {json.dumps(width_six_wall)}")
    for c in analysis["closure"]:
        print(f"closure {c['layer']:26s} remainder at m=9 "
              f"{c['remainder_us'][8]:8.1f} us "
              f"({100 * c['remainder_fraction_at_m9']:.1f}% of layer delta)")

    if args.no_wandb:
        return

    import wandb

    entity, project = args.wandb_project.split("/", 1)
    run = wandb.init(
        entity=entity, project=project, name=args.wandb_name,
        job_type="measurement",
        tags=["e186", "harness=local", "measurement", "decode-width"],
        config={
            "harness": "local",
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "experiment": "e186-decode-width-kernel-probes",
            "advisor_base_sha": "82fd65e5f8cb559fae4d4094a750581c030c6376",
            "host": "Apple M4 Pro, 20 GPU cores, macOS 26.5.2",
            "kernel_family": "non-nax (ranked M5 runs _nax)",
            "estimator": "per-block minimum pooled across sessions",
            "sessions": [m["path"] for m in meta],
            "eval_floor_entry_us": [m["eval_floor_entry_us"] for m in meta],
            "eval_floor_exit_us": [m["eval_floor_exit_us"] for m in meta],
            "round_ms": ROUND_MS, "mue_ms": MUE_MS,
        })

    raw = wandb.Table(columns=["family", "cell", "m", "block", "ascending",
                               "position", "microseconds", "reps", "session"])
    for path in args.probe:
        with open(path) as f:
            payload = json.load(f)
        for s in payload["samples"]:
            raw.add_data(s["family"], s["cell"], s["m"], s["block"],
                         s["ascending"], s.get("position", -1),
                         s["microseconds"], s["reps"], path)

    curve_table = wandb.Table(columns=[
        "family", "cell", "m", "min_us", "mean_us", "sd_us", "n"])
    for fit in cur.values():
        for i, m in enumerate(WIDTHS):
            curve_table.add_data(fit["family"], fit["cell"], m,
                                 fit["min_us"][i], fit["mean_us"][i],
                                 fit["sd_us"][i], fit["n"][i])

    fit_table = wandb.Table(columns=[
        "family", "cell", "a_us", "b_us_per_m", "c_us_per_m2", "c_ci95", "r2",
        "step_m6_us", "step_m9_us", "plateau_slope_m6_m8_us"])
    for fit in cur.values():
        f = fit["fit_on_samples"]
        fit_table.add_data(fit["family"], fit["cell"], f["a"], f["b"], f["c"],
                           f["c_ci95"], f["r2"], fit["step_m6_us"],
                           fit["step_m9_us"],
                           fit["plateau_slope_m6_m8_us_per_row"])

    round_table = wandb.Table(columns=[
        "m", "isolated_round_ms", "isolated_delta_ms", "e177_round_ms",
        "e177_delta_ms", "staircase_rescaled_ms", "disagreement_ms"])
    for i, m in enumerate(WIDTHS):
        round_table.add_data(
            m, recon["abs_ms"][i], recon["delta_ms"][i],
            recon["e177_abs_ms"][i], recon["e177_delta_ms"][i],
            adm["staircase_round_ms"][i], adm["disagreement_ms"][i])

    e182_table = wandb.Table(columns=[
        "family", "invocations", "us_m1", "us_m4", "us_m8", "round_ms_m1",
        "round_ms_m4", "round_ms_m8", "round_share_m8",
        "inpath_delta_band_lo_ms", "inpath_delta_band_hi_ms"])
    for row in e182:
        e182_table.add_data(
            row["family"], row["invocations"], row["us_m1"], row["us_m4"],
            row["us_m8"], row["round_ms_m1"], row["round_ms_m4"],
            row["round_ms_m8"], row["round_share_m8"],
            row["inpath_delta_band_ms_m1_to_m8"][0],
            row["inpath_delta_band_ms_m1_to_m8"][1])

    run.log({"e186/raw_samples": raw, "e186/curves": curve_table,
             "e186/fits": fit_table, "e186/round": round_table,
             "e186/e182_reconciliation": e182_table})
    run.summary.update({
        "staircase_max_anchor_residual_ms": adm["max_abs_residual_ms"],
        "staircase_max_anchor_residual_in_MUE":
            adm["max_abs_residual_in_MUE"],
        "delta_overshoot_at_m8": recon["overshoot_factor_delta"][7],
        "delta_share_mlp_at_m8": recon["delta_share_at_m8"].get("mlp/fused"),
        "width_six_wall_round_ms_kv512":
            width_six_wall["kv=512"]["round_cost_of_step_ms"],
        "width_six_wall_round_ms_kv1024":
            width_six_wall["kv=1024"]["round_cost_of_step_ms"],
        "gdn_recurrence_slope_us_per_row":
            cur[("gdn_recurrence", "T=m")]["plateau_slope_m1_m3_us_per_row"],
    })
    print(f"wandb run: {run.url} id={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
