#!/usr/bin/env python3
"""E145: publish the measured width cost curve and everything built on it.

`harness=local`. Unlike E140 this experiment did hold the GPU: R1, R2 and R2b
are timed decode sessions through the real 40 C cool gate. R0, R0b, R3 and R4
are zero-GPU analysis over those legs. The run therefore carries
`gpu_used=True` and per-leg thermal evidence, but it is still a local
instrument and never an official or ranked score.

Usage:
  python3 e145_wandb_log.py --run-name e145-live-width-cost-curve
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wandb  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
ARTIFACTS = HERE / "e145-artifacts"
BASE_SHA = "2cd0d459c651de53cc4ccebb160a19fb00ed87c4"
WORKER_SHA = ("9fe1bf8076dd2832fcbc011bb81a20f8e6af49b5352c744eeb823db2"
              "552ee307")


def load(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def table(columns, rows):
    return wandb.Table(columns=list(columns), data=[list(r) for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="e145-live-width-cost-curve")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    curve = load("curve.json")
    legs_blob = load("legs.json")
    r0 = load("r0.json")
    r0b = load("r0b-anchors.json")
    r1 = load("r1.json")
    r3 = load("r3.json")
    r4m = load("r4-measured.json")
    r4r = load("r4-replayed.json")
    cross = load("r4-cross.json")
    if curve is None or legs_blob is None or r1 is None:
        raise SystemExit("the curve, the legs and R1 must all be present")
    legs = legs_blob["legs"]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.run_name,
        job_type="measurement", mode="offline" if args.offline else "online",
        tags=["e145", "width-curve", "pinned-depth", "timed", "harness=local"],
        config={
            "experiment": "E145",
            "hypothesis": "H145: the width cost curve that every depth price "
                          "rests on is replayed, not measured. Decoding at "
                          "each pinned width will show a different shape, and "
                          "the shape difference will change the best price.",
            "harness": "local",
            "gpu_used": True,
            "official_or_ranked": False,
            "commit": git_sha(),
            "base_sha": BASE_SHA,
            "worker_sha256": WORKER_SHA,
            "pr": 144,
            "host": "Mac16,11 Apple M4 Pro 48 GiB",
            "pin_env": "MLX_E145_PIN_DEPTH",
            "decode_tokens": 512,
            "cool_gate": "real 40 C gate, never bypassed",
            "r3_seeds": (r3 or {}).get("seeds"),
            "r3_windows": (r3 or {}).get("windows"),
            "r4_seeds": (r4m or {}).get("seeds"),
            "r4_windows": (r4m or {}).get("windows"),
            "width1_anchor_used": (r3 or {}).get("width1_anchor_used"),
        })

    level = curve["level_transfer"]
    cliff = curve["cliff"]
    f217 = curve["f217"]
    repair = curve["repair_model"]
    prefill = curve["ranked_prefill"]

    summary = {
        # The curve itself, and how badly the replayed one described it.
        "e145_level_transfer_k": level["k"],
        "e145_max_abs_shape_residual_pct": level["max_abs_residual_pct"],
        "e145_r3_triggered": level["r3_triggered"],
        "e145_cliff_measured_us": cliff["measured_us"],
        "e145_cliff_replayed_us": cliff["replayed_ranked_us"],
        "e145_cliff_excess_measured": cliff["cliff_excess_over_typical_step"],
        "e145_cliff_excess_replayed": cliff["replayed_cliff_excess"],
        "e145_cliff_share_of_width5_round":
            cliff["measured_share_of_width5_round"],

        # F217 and the F218 bound, on the measured curve.
        "e145_predicted_beagle_shift_us_measured":
            f217["e145_predicted_beagle_shift_us_measured"],
        "e145_f218_bound_consistent_measured":
            f217["e145_f218_bound_consistent_measured"],

        # What this design cannot identify. Reported as flags, never as
        # fitted coefficients, because both regressors are collinear.
        "e145_repair_model_fitted": repair["fitted"],
        "e145_width_replay_correlation": repair["width_replay_correlation"],
        "e145_ranked_prefill_fitted": prefill["fitted"],
        "e145_prefill_regressor_correlation": prefill["regressor_correlation"],

        # Fidelity, across every timed leg in the experiment.
        "e145_timed_legs": len(legs),
        "e145_legs_all_matched": all(l["all_tokens_matched"] for l in legs),
        "e145_total_divergence": sum(l["residual_divergence_count"]
                                     for l in legs),
        "e145_legs_through_real_gate": sum(1 for l in legs
                                           if l["real_cool_gate_taken"]),
        "e145_worker_sha256_unique": len({l["worker_sha256"]
                                          for l in legs}) == 1,
    }

    # Trace legs are not timed and never take the gate, so they carry no entry
    # temperature. Aggregating over them would report nan and hide the real
    # thermal spread of the timed legs.
    entry_temps = [l["gate_entry_temp_c"] for l in legs
                   if isinstance(l["gate_entry_temp_c"], (int, float))
                   and math.isfinite(l["gate_entry_temp_c"])]
    summary["e145_gated_legs_with_entry_temp"] = len(entry_temps)
    summary["e145_min_entry_temp_c"] = min(entry_temps)
    summary["e145_max_entry_temp_c"] = max(entry_temps)
    summary["e145_entry_temp_spread_c"] = max(entry_temps) - min(entry_temps)

    run.log({"width_curve": table(
        ["width", "pin", "measured_us", "measured_from_blocks_us",
         "replayed_us", "ratio", "shape_residual_pct", "repeat_spread_pct",
         "rounds_pooled", "pin_leak_fraction", "replayed_round_share",
         "mean_draft_len", "accepted_draft_rate"],
        [[int(w), m["pin"], m["us"], m["us_mean_from_blocks"],
          curve["replayed_ranked_us"][w], level["per_width_ratio"][w],
          level["residual_pct"][w], m["us_spread_pct"], m["rounds_pooled"],
          m["pin_leak_fraction"], m["replayed_round_share"],
          m["mean_draft_len"], m["accepted_draft_rate"]]
         for w, m in sorted(curve["measured"].items(),
                            key=lambda kv: int(kv[0]))])})

    run.log({"width_steps": table(
        ["step", "measured_us", "replayed_us", "measured_over_replayed",
         "measured_share_of_low_round", "replayed_share_of_low_round"],
        [[k, v["measured_us"], v["replayed_ranked_us"],
          v["measured_over_replayed"], v["measured_as_share_of_low_round"],
          v["replayed_as_share_of_low_round"]]
         for k, v in sorted(curve["steps"].items())])})

    # The closure test. This is the strongest evidence in the experiment: the
    # curve predicts held-out legs it never saw, and the mean-draft-length
    # model that it replaces does not.
    # The serial arm decodes at width 1 and has nothing to predict, so it
    # carries a closure record without `predicted_us`. It is the reference,
    # not a held-out test.
    closure_rows = []
    for fixture, fx in sorted(r1["fixtures"].items()):
        for arm, a in sorted(fx["arms"].items()):
            c = a["closure"]
            if "predicted_us" not in c:
                continue
            closure_rows.append([
                fixture, arm, c["predicted_us"], c["observed_us"],
                c["error_pct"], c["interp_at_mean_us"], c["interp_error_pct"],
                c["jensen_gap_us"], c["mean_width"], c["covered_mass"]])
    run.log({"closure_test": table(
        ["fixture", "arm", "predicted_us", "observed_us", "error_pct",
         "interp_at_mean_us", "interp_error_pct", "jensen_gap_us",
         "mean_width", "covered_mass"], closure_rows)})
    summary["e145_closure_worst_abs_error_pct"] = max(
        abs(r[4]) for r in closure_rows)
    summary["e145_interp_at_mean_worst_abs_error_pct"] = max(
        abs(r[6]) for r in closure_rows)
    summary["e145_jensen_gap_changes_sign"] = (
        min(r[7] for r in closure_rows) < 0 < max(r[7] for r in closure_rows))

    # R1, the regime hypothesis.
    run.log({"r1_arms": table(
        ["fixture", "arm_effect_spt_pct", "arm_effect_round_cost_pct",
         "arm_effect_blocks_pct", "predicted_arm_effect_pct",
         "prediction_error_pp", "width_mass_ge6_shift_pp",
         "width_mass_at6_shift_pp"],
        [[f, x["arm_effect_spt_pct"], x["arm_effect_round_cost_pct"],
          x["arm_effect_blocks_pct"],
          x["predicted_arm_effect_round_cost_pct"],
          x["arm_effect_prediction_error_pp"],
          x["width_mass_ge6_shift_pp"], x["width_mass_at6_shift_pp"]]
         for f, x in sorted(r1["fixtures"].items())])})
    for fixture, fx in sorted(r1["fixtures"].items()):
        summary["e145_%s_arm_effect_spt_pct" % fixture] = \
            fx["arm_effect_spt_pct"]
        summary["e145_%s_arm_effect_prediction_error_pp" % fixture] = \
            fx["arm_effect_prediction_error_pp"]
        for who, pre in sorted(fx["prereg"].items()):
            summary["e145_%s_prereg_%s_verdict" % (fixture, who)] = \
                pre["verdict"]
    run.log({"r1_prereg": table(
        ["fixture", "who", "low", "high", "verdict"],
        [[f, who, pre.get("range", pre.get("range_pp"))[0],
          pre.get("range", pre.get("range_pp"))[1], pre["verdict"]]
         for f, fx in sorted(r1["fixtures"].items())
         for who, pre in sorted(fx["prereg"].items())])})

    # R1b, the second-session replicate. `work_signatures_match` is the part
    # that makes the gap interpretable: identical round counts, draft lengths
    # and acceptance rates mean only the timing moved between sessions.
    replicate = legs_blob.get("r1b_replicate") or {}
    if replicate:
        run.log({"r1b_replicate": table(
            ["fixture", "r1_pct", "r1b_pct", "gap_pp", "r1_blocks_pct",
             "r1b_blocks_pct", "blocks_gap_pp", "same_sign",
             "work_signatures_match"],
            [[f, v["r1_pct"], v["r1b_pct"], v["gap_pp"], v["r1_blocks_pct"],
              v["r1b_blocks_pct"], v["blocks_gap_pp"], v["same_sign"],
              v["work_signatures_match"]]
             for f, v in sorted(replicate.items())])})
        for fixture, v in sorted(replicate.items()):
            summary["e145_%s_replicate_gap_pp" % fixture] = v["gap_pp"]
            summary["e145_%s_replicate_same_sign" % fixture] = v["same_sign"]
            summary["e145_%s_replicate_work_signatures_match" % fixture] = \
                v["work_signatures_match"]

    # Rule 128 asks for `wired-zh` and `warm` in every leg. The timed parent
    # swallows worker stderr, so the lines are unobtainable rather than
    # omitted. Record the failure instead of hiding it.
    summary["e145_warm_telemetry_present_any_leg"] = any(
        l["warm_telemetry_present"] for l in legs)
    summary["e145_warm_telemetry_blocked_by_parent_stderr"] = True

    noise_floor = None
    if r0 is not None:
        summary["e145_predicted_beagle_shift_us"] = \
            r0["e145_predicted_beagle_shift_us"]
        summary["e145_f218_bound_consistent"] = \
            r0["e145_f218_bound_consistent"]
        null = r0["null_control"]
        noise_floor = null["noise_floor_pct_median_eligible"]
        summary["e145_noise_floor_pct_median_eligible"] = noise_floor
        summary["e145_noise_floor_pct_all"] = null["noise_floor_pct_all"]
        summary["e145_null_control_is_true_null"] = null["is_true_null_pair"]
        summary["e145_null_control_max_draft_len_gap"] = \
            null["max_abs_draft_len_gap"]
        run.log({"r0_null_control": table(
            ["prompt", "draft_len", "draft_len_gap_vs_f219", "round_us",
             "round_us_pct_vs_f219"],
            [[p, v["draft_len"], v["draft_len_gap_vs_f219"], v["round_us"],
              v["round_us_pct_vs_f219"]]
             for p, v in sorted(null["prompts"].items())])})

    if r0b is not None:
        summary["e145_three_anchor_spread_pct"] = r0b["three_anchor_spread_pct"]
        summary["e145_three_anchor_agree"] = \
            r0b["three_anchor_agree_within_1p5_pct"]
        summary["e145_prefill_is_inside_the_scored_leg"] = \
            r0b["prefill_is_inside_the_scored_leg"]
        for key, value in r0b["ranked_prefill_estimate"].items():
            if isinstance(value, (int, float, bool)):
                summary["e145_r0b_" + key] = value

    if r3 is not None:
        summary["e145_shape_worst_disagreement_pct"] = \
            r3["shape_disagreement"]["worst_pct"]
        summary["e145_shape_worst_disagreement_width"] = \
            r3["shape_disagreement"]["worst_width"]
        summary["e145_shape_agrees_within_5pct"] = \
            r3["shape_disagreement"]["agrees_within_5_pct"]
        summary["e145_width1_anchor_us"] = \
            r3["width1_candidates_us"][r3["width1_anchor_used"]]
        if r3.get("scale_gate"):
            summary["e145_scale_gate_worst_shift_pp"] = \
                r3["scale_gate"]["worst_shift_pp"]
        for source in ("measured", "replayed"):
            tier = r3["tier_implied"][source]
            summary["e145_tier_implied_%s_vs_mean_other" % source] = \
                tier["tier_against_mean_other"]
            summary["e145_tier_implied_%s_argmax_step" % source] = \
                tier["argmax_step_index"]

        # Both curves are normalised by their own mean step, so the two
        # columns are directly comparable even though the levels differ by
        # roughly a factor of two.
        by_boundary = {}
        for source in ("measured", "replayed"):
            for row in r3["boundary_table"][source]:
                key = (row["from_width"], row["to_width"])
                by_boundary.setdefault(key, {})[source] = row
        run.log({"r3_boundary_shape": table(
            ["boundary", "measured_step_us", "measured_multiple",
             "replayed_step_us", "replayed_multiple"],
            [["%d->%d" % key,
              rows["measured"]["step_us"], rows["measured"]["share_of_mean_step"],
              rows["replayed"]["step_us"], rows["replayed"]["share_of_mean_step"]]
             for key, rows in sorted(by_boundary.items())
             if "measured" in rows and "replayed" in rows])})

        # Cell keys are "cost|price|cell", so the three cost and price
        # combinations become three columns of one row per cell.
        cells = r3["cells"]
        combos = sorted({tuple(k.split("|")[:2]) for k in cells})
        cell_names = sorted({k.split("|")[2] for k in cells})
        run.log({"r3_cells": table(
            ["cell"] + ["cost=%s price=%s" % c for c in combos],
            [[name] + [cells["%s|%s|%s" % (c[0], c[1], name)]["median_pct_mean"]
                       for c in combos]
             for name in cell_names])})

    if r4m is not None:
        summary["e145_r4_best_cell"] = r4m["best_cell"]
        summary["e145_r4_best_minus_shipped_pp"] = \
            r4m["best_minus_shipped_pp"]
        summary["e145_r4_ridge_cliff_price_spread_pct"] = \
            r4m["ridge_stats"]["cliff_price_spread_pct"]
        summary["e145_r4_ridge_shallow_price_spread_pct"] = \
            r4m["ridge_stats"]["shallow_price_spread_pct"]
        summary["e145_r4_ridge_objective_spread_pp"] = \
            r4m["ridge_stats"]["objective_spread_pp"]
        summary["e145_r4_shipped_upper_slot_spread_pp"] = \
            r4m["grid"][r4m["shipped_cell"]]["upper_slot_spread_mean"]
        summary["e145_r4_shipped_worst_slot"] = \
            r4m["grid"][r4m["shipped_cell"]]["worst_upper_slot_prompt"]
        if r4m.get("pb67_best_cell"):
            pb67_best = r4m["pb67_grid"][r4m["pb67_best_cell"]]
            summary["e145_r4_pb67_best_cell"] = r4m["pb67_best_cell"]
            summary["e145_r4_pb67_objective"] = pb67_best["objective_mean"]
            summary["e145_r4_pb67_gain_over_single_boundary_pp"] = (
                pb67_best["objective_mean"]
                - r4m["grid"][r4m["best_cell"]]["objective_mean"])
        run.log({"r4_ridge": table(
            ["h", "tier", "within_price", "cliff_price", "objective"],
            [[c["h"], c["tier"], c["within_price"], c["cliff_price"],
              c["objective_mean"]] for c in r4m["ridge"]])})

    # The number the experiment exists to produce: what did believing the
    # replayed curve actually cost, scored on the measured one?
    if cross is not None:
        summary["e145_wrong_curve_regret_pp"] = cross["regret_pp"]
        summary["e145_replayed_and_measured_pick_same_cell"] = \
            cross["same_cell"]
        summary["e145_replayed_believed_gain_pp"] = \
            cross["replayed_believed_gain_pp"]
        summary["e145_replayed_cell_true_gain_pp"] = \
            cross["replayed_cell_true_gain_pp"]
        # The headline claim of the experiment: the whole consequence of the
        # replayed curve being wrong is smaller than the noise floor the F4
        # null control established for this same measurement system.
        if noise_floor is not None:
            summary["e145_regret_below_noise_floor"] = \
                abs(cross["regret_pp"]) < noise_floor
            summary["e145_regret_over_noise_floor_ratio"] = \
                abs(cross["regret_pp"]) / noise_floor

    run.log({"timed_legs": table(
        ["slot", "fixture", "arm", "pin", "position", "rounds",
         "round_us_from_blocks", "block_us_median", "mean_draft_len",
         "accepted_draft_rate", "spt", "gate_entry_temp_c", "leg_exit_temp_c",
         "real_cool_gate_taken", "warm_telemetry_present", "all_tokens_matched",
         "residual_divergence_count"],
        [[l["slot"], l["fixture"], l["arm"], l["pin"], l["position"],
          len(l["blocks"]), l["round_us_from_blocks"], l["block_us_median"],
          l["mean_draft_len"], l["accepted_draft_rate"], l["spt"],
          l["gate_entry_temp_c"], l["leg_exit_temp_c"],
          l["real_cool_gate_taken"], l["warm_telemetry_present"],
          l["all_tokens_matched"], l["residual_divergence_count"]]
         for l in legs])})

    run.summary.update(summary)
    print("run id   %s" % run.id)
    print("run url  %s" % run.url)
    for key in sorted(summary):
        print("  %-52s %s" % (key, summary[key]))
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
