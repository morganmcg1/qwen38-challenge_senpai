#!/usr/bin/env python3
"""Reduce the E126 rung-2 in-situ ABBA session and score the transfer models.

    usage: research/e126_rung2.py [--replicates N] [--label L] [--tokens N]
                                  [--out PATH]

The leg reduction is `research/e121_e2e_analyse.py`, reused rather than
rewritten: same ABBA order, same paired replicate contrast, same pooling. This
file adds the three things E126 asks for that E121 did not need.

1. `e126_in_situ_prediction_error_pp`, the absolute error in percentage points
   between the rung-1 isolated prediction and this in-situ measurement.
   Baseline 0.452, which is E121's own miss. Direction: minimize.
2. The per-arm entry and exit temperature table askeladd needs as item 3.
3. The three candidate isolated-to-in-situ transfer models, scored against
   every (isolated, in situ) pair the campaign has.

THE HEADLINE IS ABSOLUTE CANDIDATE SECONDS PER TOKEN against the fresh
`share_off` legs in the same session. The local serial-to-MTP ratio is a
secondary read: both local legs use the candidate binary, so a change that also
moves the local serial leg cancels in that ratio and cannot be priced by it.

Everything here is `harness=local`. No number in this file is an official
score, and none of it is gate qualified.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import e121_e2e_analyse as e121

# ---------------------------------------------------------------- prediction
# PRE-REGISTERED. Written before the session produced a single leg, from the
# rung-1 isolated measurement of the same contrast on the same base.
#
# Rung 1 measured `share_on` against `share_off`, percent faster, per width:
RUNG1_ON_VS_OFF_PER_WIDTH_PCT = {3: 0.30375386, 4: 1.66204542, 5: 0.14379638}
# The cost-weighted frame of the same cells, carried as a sensitivity.
RUNG1_ON_VS_OFF_COST_WEIGHTED_PCT = {3: 0.82161928, 4: 2.30805705,
                                     5: 0.15837859}
# Standing NA weights. NA = 2 is dispatched to a separate narrow function that
# this arm does not touch, so its 0.024 carries a zero effect rather than being
# renormalised away.
NA_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}
# E116's fitted wide-QMV share of the leg. F95 records that this coefficient is
# width dependent, so the width mix it is used at is stated with it.
WIDE_QMV_TO_LEG = 0.607
WIDE_QMV_TO_LEG_WIDTH_MIX = "standing NA weights, mean width 3.79"
RANKED_TRANSFER = 0.95

# E121's own miss is the baseline this experiment must beat.
PREDICTION_ERROR_BASELINE_PP = 0.452

# Every (isolated prediction, in-situ measurement) pair the campaign holds, in
# PERCENT FASTER. Positive means faster. The rung-2 point is appended once it
# is measured.
TRANSFER_POINTS = [
    {"name": "E121 rung 3", "mechanism": "gated chunk-sum share",
     "kernel_pct": 1.463, "isolated_leg_pct": 0.888, "in_situ_leg_pct": 0.4364,
     "mean_width": 3.79, "source": "e121-results.md"},
    {"name": "thorfinn rung 5e", "mechanism": "Route B replica sum table",
     "kernel_pct": 5.337, "isolated_leg_pct": 3.240, "in_situ_leg_pct": 4.249,
     "mean_width": 7.359, "source": "W&B zkcfcaxr"},
]


def round_weighted(per_width: dict[int, float]) -> float:
    return sum(NA_WEIGHTS[na] * per_width.get(na, 0.0) for na in NA_WEIGHTS)


def transfer_models(points: list[dict]) -> dict:
    """Score the three candidate isolated-to-in-situ transfer models.

    Sign convention throughout: percent FASTER, so positive is an improvement.
    A model is scored by the spread of the parameter it claims is constant. A
    model whose 'constant' varies more than the effects themselves explains
    nothing.
    """
    usable = [p for p in points if p.get("in_situ_leg_pct") is not None]
    out: dict = {"n_points": len(usable), "points": usable}
    if len(usable) < 2:
        out["verdict"] = "not enough in-situ points to separate the models"
        return out

    ratios = [p["in_situ_leg_pct"] / p["isolated_leg_pct"] for p in usable]
    offsets = [p["in_situ_leg_pct"] - p["isolated_leg_pct"] for p in usable]

    out["multiplicative"] = {
        "claim": "in_situ = k * isolated, k constant",
        "k_per_point": dict(zip([p["name"] for p in usable], ratios)),
        "k_mean": statistics.fmean(ratios),
        "k_spread": max(ratios) - min(ratios),
        "k_relative_spread": (max(ratios) - min(ratios))
                             / statistics.fmean(ratios),
    }
    out["additive"] = {
        "claim": "in_situ = isolated + c, c constant in percentage points",
        "c_per_point": dict(zip([p["name"] for p in usable], offsets)),
        "c_mean": statistics.fmean(offsets),
        "c_spread_pp": max(offsets) - min(offsets),
    }

    # Roofline distance is not directly measured for every point, so the
    # campaign's available proxy is mean draft width: a wider round streams
    # more rows per weight byte and sits closer to achieved peak. The model is
    # scored the same way, by the spread of its claimed constant.
    slopes = []
    for p in usable:
        ratio = p["in_situ_leg_pct"] / p["isolated_leg_pct"]
        slopes.append((p["mean_width"], ratio))
    if len({w for w, _ in slopes}) > 1:
        xs = [w for w, _ in slopes]
        ys = [r for _, r in slopes]
        xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
        den = sum((x - xbar) ** 2 for x in xs)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / den
        out["width_scaled"] = {
            "claim": "in_situ / isolated rises with mean draft width, the "
                     "campaign's available proxy for roofline distance",
            "slope_per_width": slope,
            "intercept": ybar - slope * xbar,
            "points": slopes,
            "note": "with %d points this is a description, not a fit"
                    % len(slopes),
        }

    best = min(("multiplicative", out["multiplicative"]["k_relative_spread"]),
               ("additive", out["additive"]["c_spread_pp"]
                / max(abs(p["in_situ_leg_pct"]) for p in usable)),
               key=lambda kv: kv[1])
    out["verdict"] = (
        "neither constant survives: multiplicative k spans %.3f to %.3f and "
        "additive c spans %+.3f to %+.3f pp"
        % (min(ratios), max(ratios), min(offsets), max(offsets))
        if len(usable) >= 2 else "insufficient")
    out["least_bad"] = best[0]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--label", default="e126r2")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("research/e126-artifacts/"
                                         "rung2-insitu.json"))
    args = ap.parse_args(argv)

    core_path = pathlib.Path("/tmp/e126-rung2-core.json")
    # The reduction defaults to E121's own row-digest artifact. Reusing it here
    # would report E121's exactness legs as though this session had run them.
    # This session's exactness evidence is the per-leg `all_tokens_matched`
    # flag on its own twenty legs, so the pin is scoped to an E126 path that
    # only exists if E126 produced one.
    rc = e121.main(["--replicates", str(args.replicates),
                    "--tokens", str(args.tokens),
                    "--label", args.label,
                    "--exactness",
                    "research/e126-artifacts/row-digest-512.json",
                    "--out", str(core_path)])
    if rc != 0 or not core_path.is_file():
        print("e126_rung2: the leg reduction failed with %s" % rc)
        return rc or 2
    core = json.loads(core_path.read_text())

    legs = core["legs"]
    summary = core["summary"]
    measured_leg_pct = -summary["mtp_spt_pct_mean"]  # percent FASTER

    predicted_kernel = round_weighted(RUNG1_ON_VS_OFF_PER_WIDTH_PCT)
    predicted_leg = predicted_kernel * WIDE_QMV_TO_LEG
    predicted_kernel_cw = round_weighted(RUNG1_ON_VS_OFF_COST_WEIGHTED_PCT)
    predicted_leg_cw = predicted_kernel_cw * WIDE_QMV_TO_LEG
    error_pp = abs(predicted_leg - measured_leg_pct)

    # Askeladd item 3: the in-situ pair with per-arm entry temperatures.
    thermal = {}
    for arm in ("base", "share"):
        entry = [leg["gpu_temp_entry_c"] for leg in legs
                 if leg["tree"] == arm and leg["gpu_temp_entry_c"] is not None]
        exits = [leg["gpu_temp_exit_c"] for leg in legs
                 if leg["tree"] == arm and leg["gpu_temp_exit_c"] is not None]
        thermal[arm] = {
            "n": len([leg for leg in legs if leg["tree"] == arm]),
            "entry_c_mean": statistics.fmean(entry) if entry else None,
            "entry_c_min": min(entry) if entry else None,
            "entry_c_max": max(entry) if entry else None,
            "entry_c_spread": (max(entry) - min(entry)) if entry else None,
            "exit_c_mean": statistics.fmean(exits) if exits else None,
            "exit_c_max": max(exits) if exits else None,
            "seconds_per_token_mean": statistics.fmean(
                leg["seconds_per_token"] for leg in legs
                if leg["tree"] == arm),
        }
    if thermal["base"]["entry_c_mean"] is not None:
        imbalance = (thermal["share"]["entry_c_mean"]
                     - thermal["base"]["entry_c_mean"])
        thermal["share_minus_base_entry_c"] = imbalance
        thermal["balance_report_fired"] = abs(imbalance) > 1.0

    points = list(TRANSFER_POINTS)
    points.append({
        "name": "E126 rung 2", "mechanism": "gated chunk-sum share, reprice",
        "kernel_pct": predicted_kernel, "isolated_leg_pct": predicted_leg,
        "in_situ_leg_pct": measured_leg_pct, "mean_width": 3.79,
        "source": "this session",
    })

    doc = {
        "experiment": "e126-rung2-insitu",
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "arms": ["share_off (base, common control)", "share_on (shipped)"],
        "g_min_ask_dropped_because":
            "at NA=4, which carries 0.667 of the round, the shipped "
            "g_split_pred body and g_min_ask have the same threadgroup access "
            "count, and g_min_ask uses 1024 threadgroup bytes against 512, 10 "
            "static fadds against 6, and 392 AIR lines against 360. Its "
            "exchange is not smaller at the scored width. It has also never "
            "been transplanted into quantized.h.",
        "core": core,
        "prediction": {
            "registered_before_session": True,
            "source": "E126 rung 1 isolated, share_on against share_off",
            "per_width_pct": RUNG1_ON_VS_OFF_PER_WIDTH_PCT,
            "na_weights": NA_WEIGHTS,
            "round_weighted_kernel_pct": predicted_kernel,
            "wide_qmv_to_leg": WIDE_QMV_TO_LEG,
            "wide_qmv_to_leg_width_mix": WIDE_QMV_TO_LEG_WIDTH_MIX,
            "predicted_leg_pct_faster": predicted_leg,
            "cost_weighted_sensitivity": {
                "round_weighted_kernel_pct": predicted_kernel_cw,
                "predicted_leg_pct_faster": predicted_leg_cw,
            },
        },
        "measured": {
            "primary": "absolute candidate mtp seconds per token against a "
                       "fresh share_off in the same session",
            "mtp_spt_base_mean_s": summary["mtp_spt_base_mean_s"],
            "mtp_spt_share_mean_s": summary["mtp_spt_share_mean_s"],
            "leg_pct_faster": measured_leg_pct,
            "ci95_pct_faster": [
                -summary["mtp_spt_pct_ci95_upper"]
                if summary["mtp_spt_pct_ci95_upper"] is not None else None,
                -summary["mtp_spt_pct_ci95_lower"]
                if summary["mtp_spt_pct_ci95_lower"] is not None else None,
            ],
            "stdev_pct": summary["mtp_spt_pct_stdev"],
            "n_replicates": len(core["per_replicate"]),
            "n_legs": summary["n_legs"],
            "local_ratio_pct_mean": summary["local_ratio_pct_mean"],
            "local_ratio_caveat":
                "both local legs use the candidate binary, so this ratio "
                "cannot price a change that also moves the local serial leg",
            "serial_spt_pct_mean": summary["serial_spt_pct_mean"],
            "schedule_invariant": summary["schedule_invariant"],
            "exactness_passed": summary["exactness_passed"],
            "ranked_frame_pct_faster": measured_leg_pct * RANKED_TRANSFER,
        },
        "secondary_metric": {
            "name": "e126_in_situ_prediction_error_pp",
            "direction": "minimize",
            "baseline": PREDICTION_ERROR_BASELINE_PP,
            "candidate": error_pp,
            "delta": error_pp - PREDICTION_ERROR_BASELINE_PP,
            "beat_baseline": error_pp < PREDICTION_ERROR_BASELINE_PP,
        },
        "thermal_per_arm": thermal,
        "transfer_models": transfer_models(points),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print("\n================ E126 rung 2, harness=local ================")
    print("arms          share_off (control) and share_on (shipped), ABBA")
    print("legs          %d over %d replicates, %d tokens"
          % (summary["n_legs"], len(core["per_replicate"]), args.tokens))
    print("base   s/tok  %.9f" % summary["mtp_spt_base_mean_s"])
    print("share  s/tok  %.9f" % summary["mtp_spt_share_mean_s"])
    print("leg    faster %+.4f %%  ci95 [%s]"
          % (measured_leg_pct,
             ", ".join("%+.4f" % v for v in doc["measured"]["ci95_pct_faster"]
                       if v is not None)))
    print("local ratio   %+.4f %% (secondary only)"
          % summary["local_ratio_pct_mean"])
    print("predicted     %+.4f %% faster from rung 1 isolated" % predicted_leg)
    print("ERROR         %.4f pp against the %.3f pp baseline -> %s"
          % (error_pp, PREDICTION_ERROR_BASELINE_PP,
             "BEAT" if error_pp < PREDICTION_ERROR_BASELINE_PP else "MISS"))
    print("schedule invariant %s, exactness %s"
          % (summary["schedule_invariant"], summary["exactness_passed"]))
    print("\n--- thermal, per arm ---")
    for arm in ("base", "share"):
        t = thermal[arm]
        print("  %-6s n=%d entry %.2f C [%.2f, %.2f] spread %.2f, exit max "
              "%.2f" % (arm, t["n"], t["entry_c_mean"], t["entry_c_min"],
                        t["entry_c_max"], t["entry_c_spread"],
                        t["exit_c_max"]))
    print("  share minus base entry %+.2f C -> %s"
          % (thermal["share_minus_base_entry_c"],
             "REPORT FIRED" if thermal["balance_report_fired"] else "MATCHED"))
    print("\n--- transfer models ---")
    tm = doc["transfer_models"]
    print("  %s" % tm["verdict"])
    for name, k in tm["multiplicative"]["k_per_point"].items():
        print("    %-18s in_situ/isolated %+.3f" % (name, k))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
