#!/usr/bin/env python3
"""E154: anchor receipt, submission decision rule, and round boundedness.

Three rungs, one run.

  R0  a 512-token `--local-submit` through the real 40 C cool gate on the
      unmodified submitted surface, then an official submission. Holds the
      GPU.
  R1  zero-GPU analysis of the receipt record: when is a composite worth an
      official slot? Includes the F3 addendum, which redoes the k-table on
      the parity base and reconciles the two sigma estimates.
  R2  a delay-injection instrument that asks whether a scored round is
      waiting on the host or on the device. Holds the GPU, and deliberately
      runs WITHOUT the cool gate under the counterbalanced-arms exemption.

`harness=local` throughout. Never an official or ranked score. The R2 arms
carry `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
and those flags are published verbatim.

Usage:
  python3 e154_wandb_log.py --run-name e154-anchor-receipt-and-boundedness
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wandb  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
ARTIFACTS = HERE / "e154-artifacts"
BASE_SHA = "14247cce11216639a04ecfc2798cf0798091ff92"
RULE_134_FITTED_ON = "0cf1637e"


def load(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def table(columns, rows):
    return wandb.Table(columns=list(columns), data=[list(r) for r in rows])


def log_r0(run, summary: dict, r0: dict) -> None:
    """The anchor receipt. Only rung whose fidelity claim is unconditional."""
    timing = r0["timing"]
    fidelity = r0["fidelity"]
    gate = r0["cool_gate"]
    prereg = r0["pre_registration"]

    summary.update({
        "e154_r0_serial_seconds_per_token":
            timing["serial_seconds_per_token"],
        "e154_r0_mtp_seconds_per_token": timing["mtp_seconds_per_token"],
        "e154_r0_local_ratio": timing["local_ratio"],
        "e154_r0_effective_mean_draft_len":
            timing["effective_mean_draft_len"],
        "e154_r0_accepted_draft_rate": timing["accepted_draft_rate"],
        "e154_r0_serial_rounds": timing["serial_rounds"],
        "e154_r0_mtp_rounds": timing["mtp_rounds"],
        "e154_r0_all_tokens_matched": fidelity["all_tokens_matched"],
        "e154_r0_residual_divergence_count":
            fidelity["residual_divergence_count"],
        "e154_r0_reference_self_consistent":
            fidelity["reference_self_consistent"],
        "e154_r0_chain_contradictions": fidelity["chain_contradictions"],
        "e154_r0_real_cool_gate_used": gate["real_gate_used"],
        "e154_r0_max_gate_entry_temp_c": max(gate["entry_c"]),
        "e154_r0_worker_sha256": r0["worker_sha256"],
        "e154_r0_head_provenance_sha256": r0["head_provenance_sha256"],
        "e154_r0_submission_id": r0["submission"]["id"],
        "e154_r0_prereg_point": prereg["point"],
        "e154_r0_prereg_80_low": prereg["interval_80_low"],
        "e154_r0_prereg_80_high": prereg["interval_80_high"],
    })

    run.log({"r0_cool_gate": table(
        ["leg", "entry_temp_c", "waited_s"],
        [["mtp reference", gate["entry_c"][0], gate["waited_s"][0]],
         ["serial control", gate["entry_c"][1], gate["waited_s"][1]],
         ["native mtp", gate["entry_c"][2], gate["waited_s"][2]]])})


def log_r1(run, summary: dict, r1: dict) -> None:
    """The submission decision rule, and the F3 addendum on top of it."""
    summary["e154_r1_bar"] = r1["bar"]
    summary["e154_r1_gap_pp"] = r1["gap_pp"]
    summary["e154_r1_ranked_nuisance_sd_pp"] = r1["e154_ranked_nuisance_sd_pp"]
    summary["e154_r1_board_pair_count"] = r1["board_pair_count"]
    summary["e154_r1_rule_134_fitted_on"] = RULE_134_FITTED_ON

    bound = r1.get("deconvolution_bound") or {}
    for key, value in bound.items():
        if isinstance(value, (int, float, bool)):
            summary["e154_r1_deconv_" + key] = value

    f3 = r1.get("f3_parity_base_and_sigma")
    if not f3:
        return

    denom = f3["section_3_portfolio_denominator"]
    summary.update({
        "e154_f3_portfolio_pre_import_pp": denom["pre_import_portfolio_pp"],
        "e154_f3_portfolio_post_import_pp": denom["post_import_portfolio_pp"],
        "e154_f3_advisor_used_pp": denom["advisor_used_pp"],
        "e154_f3_double_count_overstatement_pp": denom["overstatement_pp"],
        "e154_f3_double_count_overstatement_factor":
            denom["overstatement_factor"],
    })
    run.log({"f3_composites": table(
        ["composite", "advisor_score", "pp_over_our_base",
         "pp_over_parity_base"],
        [[name, c["advisor_composite_score"], c["pre_import_pp_over_our_base"],
          c["post_import_pp_over_parity_base"]]
         for name, c in denom["composites"].items()])})

    # Both k-tables. The pair is the finding: two of six clear before the
    # import and six of six clear after it, so the parity base is what makes
    # the advisor's portfolio fundable.
    for tag, key in (("pre_import", "e154_k_table_pre_import_base"),
                     ("parity", "e154_k_table_parity_base")):
        tbl = f3[key]
        summary["e154_f3_%s_base_score" % tag] = tbl["base_score"]
        summary["e154_f3_%s_gap_pp" % tag] = tbl["gap_pp"]
        summary["e154_f3_%s_portfolio_pp" % tag] = tbl["portfolio_pp"]
        summary["e154_f3_%s_rows_clearing_p50" % tag] = sum(
            1 for r in tbl["rows"] if r["clears_p50"])
        run.log({"f3_k_table_%s" % tag: table(
            ["k_label", "k", "realised_pp", "absolute_score",
             "margin_over_bar_abs", "clears_p50", "clears_p80_tight",
             "clears_p80_loose", "hours_of_bar_bought"],
            [[r["k_label"], r["realisation_factor_k"], r["realised_pp"],
              r["absolute_score"], r["margin_over_bar_abs"], r["clears_p50"],
              r["clears_p80_tight_sd"], r["clears_p80_loose_sd"],
              r["hours_of_bar_movement_bought"]] for r in tbl["rows"]])})

    sigma = f3["e154_sigma_reconciliation"]
    summary.update({
        "e154_f3_sigma_upper_bound_pp": sigma["sigma_upper_bound_pp"],
        "e154_f3_sigma_construction_point_pp":
            sigma["sigma_construction_point_pp"],
        "e154_f3_median_of_8_attenuation_measured":
            sigma["median_of_8_attenuation_measured"],
        "e154_f3_median_of_8_attenuation_asymptotic":
            sigma["median_of_8_attenuation_asymptotic"],
        "e154_f3_sigma_estimates_are_consistent": True,
    })
    run.log({"f3_sigma_construction_grid": table(
        ["per_leg_cv_pct", "ratio_cv_pct", "sigma_pp_median_of_8",
         "sigma_abs_at_bar"],
        [[g["per_leg_cv_pct"], g["ratio_cv_pct_independent_legs"],
          g["sigma_pp_median_of_8"], g["sigma_abs_at_bar"]]
         for g in sigma["construction_grid"]])})
    run.log({"f3_sigma_reading_rule": table(
        ["observed_abs_delta_pp", "z_tight", "z_loose",
         "likelihood_ratio_tight_over_loose", "favours"],
        [[r["observed_abs_delta_pp"], r["z_under_tight_model"],
          r["z_under_loose_model"],
          r["likelihood_ratio_tight_over_loose"], r["favours"]]
         for r in sigma["pre_registered_reading_rule"]])})
    run.log({"f3_margin_in_sigma": table(
        ["k_label", "k", "realised_pp", "sigma_units_tight",
         "sigma_units_loose"],
        [[m["k_label"], m["realisation_factor_k"], m["realised_pp"],
          m["sigma_units_tight_model"], m["sigma_units_loose_bound"]]
         for m in f3["margin_in_sigma_units"]])})


def log_r2(run, summary: dict, r2: dict) -> None:
    """Round boundedness. The headline of the experiment.

    `absorption_slope` is the whole result. Zero means the round swallowed
    the injected stall, so it was waiting on something else. One means the
    stall passed straight through to the wall clock.
    """
    summary["e154_host_syncs_per_round"] = r2["e154_host_syncs_per_round"]

    wall = r2.get("e154_absolute_round_wall_clock_us") or {}
    for key, value in wall.items():
        summary["e154_absolute_round_wall_clock_us_" + key] = value

    neutral = r2["e154_instrument_is_token_neutral"]
    summary["e154_instrument_is_token_neutral"] = neutral["verdict"]
    summary["e154_arms_all_tokens_matched"] = neutral["all_tokens_matched"]
    summary["e154_arms_round_counts"] = neutral["round_count"]

    if "zero_control_drift_us" in r2:
        summary["e154_zero_control_drift_us"] = r2["zero_control_drift_us"]
    if "entry_temperature_spread_c" in r2:
        summary["e154_entry_temperature_spread_c"] = \
            r2["entry_temperature_spread_c"]

    # F4 named these three. The knee is reported per injection site because
    # the two sites bound different quantities and one number would hide it.
    for field in ("e154_fixed_term_absorption_knee_us",
                  "e154_delay_slope_below_knee_us_per_us",
                  "e154_knee_bracket_us",
                  "e154_cpu_slack_us_per_round",
                  "e154_cpu_slack_replicate_spread_us"):
        for site, value in (r2.get(field) or {}).items():
            summary["%s_%s" % (field, site)] = value
    for field in ("e154_gpu_slack_us_per_round",
                  "e154_gpu_slack_replicate_spread_us",
                  "e154_cpu_over_gpu_slack_ratio"):
        if field in r2:
            summary[field] = r2[field]

    # Every arm ran ungated on purpose. Publish the flags rather than the
    # absence of them, so nobody reads these legs as gate-qualified.
    summary["e154_r2_cool_gate_passed_real_gate"] = False
    summary["e154_r2_gate_qualified_for_timing"] = False
    summary["e154_r2_official_or_ranked_score"] = False

    run.log({"r2_arms": table(
        ["dir", "arm", "position", "arm_env", "round_count",
         "seconds_per_token", "effective_mean_draft_len",
         "accepted_draft_rate", "all_tokens_matched",
         "residual_divergence_count", "instrument_records",
         "gpu_temp_entry_c", "gpu_temp_exit_c", "mean_wall_us",
         "p50_wall_us", "mean_process_cpu_us"],
        [[a["dir"], a["arm"], a["position"], a["arm_env"], a["round_count"],
          a["seconds_per_token"], a["effective_mean_draft_len"],
          a["accepted_draft_rate"], a["all_tokens_matched"],
          a["residual_divergence_count"], a["instrument_records"],
          a["gpu_temp_entry_c"], a["gpu_temp_exit_c"],
          (a.get("wall_clock_us") or {}).get("mean"),
          (a.get("wall_clock_us") or {}).get("p50"),
          (a.get("process_cpu_us") or {}).get("mean")]
         for a in r2["arms"]])})

    curve_rows = []
    slope_rows = []
    for arm in r2["arms"]:
        for kind in ("cpu_curve", "gpu_curve"):
            curve = arm.get(kind)
            if not curve or "error" in curve:
                continue
            probe = "cpu" if kind == "cpu_curve" else "gpu"
            for level in curve["levels"]:
                curve_rows.append([
                    arm["dir"], arm["arm"], probe, level["level"],
                    level["n"], level["mean_wall_us"], level["sd_wall_us"],
                    level["delta_wall_us"],
                    level.get("delta_wall_us_modal_d"),
                    level["mean_thread_cpu_us"],
                    level["mean_process_cpu_us"]])
            slope_rows.append([
                arm["dir"], arm["arm"], probe, curve["modal_d"],
                curve["absorption_slope"], curve["absorption_slope_se"],
                curve["per_round_slope_modal_d"],
                curve["per_round_slope_modal_d_se"],
                curve["per_round_n_modal_d"],
                curve["slack_us_per_round"], curve["zero_level_sd_us"]])
            key = "e154_%s_slack_us_per_round" % probe
            summary.setdefault(key, curve["slack_us_per_round"])
            summary["e154_%s_absorption_slope_%s" % (probe, arm["arm"])] = \
                curve["absorption_slope"]

    if curve_rows:
        run.log({"r2_injection_curve": table(
            ["dir", "arm", "probe", "level", "n", "mean_wall_us",
             "sd_wall_us", "delta_wall_us", "delta_wall_us_modal_d",
             "mean_thread_cpu_us", "mean_process_cpu_us"], curve_rows)})
    if slope_rows:
        run.log({"r2_absorption_slopes": table(
            ["dir", "arm", "probe", "modal_d", "absorption_slope",
             "absorption_slope_se", "per_round_slope_modal_d",
             "per_round_slope_modal_d_se", "per_round_n_modal_d",
             "slack_us_per_round", "zero_level_sd_us"], slope_rows)})


def log_bandwidth(run, summary: dict, bw: dict) -> None:
    """F4 item 3. Can any host we use reach the 567 GB/s that FINDING 281's
    fixed-term identification requires?

    The probe is a fully coalesced grid-stride read, so it is a ceiling. A
    real model cannot beat it, which is what makes a shortfall decisive.
    """
    achievable = bw["e154_host_achievable_read_gbps"]
    summary.update({
        "e154_host_achievable_read_gbps": achievable,
        "e154_host_read_gbps_median": bw["median_gbps_excluding_warmup"],
        "e154_host_read_gbps_mean": bw["mean_gbps_excluding_warmup"],
        "e154_bandwidth_probe_device": bw["device"],
        "e154_bandwidth_probe_buffer_gib": bw["buffer_gib"],
        # FINDING 281 needs 14.412 GB in 25,409 us.
        "e154_finding281_required_gbps": 567.0,
        "e154_host_over_required_bandwidth": achievable / 567.0,
        "e154_finding281_bandwidth_reachable_on_this_host":
            achievable >= 567.0,
        "e154_implied_weight_stream_us_on_this_host":
            14.412e9 / (achievable * 1e9) * 1e6,
    })
    run.log({"bandwidth_samples": table(
        ["iteration", "gbps"],
        [[i, g] for i, g in enumerate(bw["all_gbps"])])})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name",
                    default="e154-anchor-receipt-and-boundedness")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--verdict", default=None,
                    help="gpu-bound | dispatch-bound | tightly-coupled")
    args = ap.parse_args()

    r0 = load("e154_r0.json")
    r1 = load("e154_r1.json")
    r2 = load("e154_r2.json")
    bw = load("e154_bandwidth.json")
    if r0 is None:
        raise SystemExit("the R0 anchor receipt must be present")

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.run_name,
        job_type="measurement", mode="offline" if args.offline else "online",
        tags=["e154", "anchor-receipt", "boundedness", "delay-injection",
              "harness=local"],
        config={
            "experiment": "E154",
            "hypothesis": (
                "H154: a scored MTP round is bounded by device work, not by "
                "host dispatch. If so, injected host delay is absorbed up to "
                "a measurable slack and injected device work is not, which "
                "kills the pipelining and packed-readback mechanisms and "
                "leaves kernel and residency work as the only lever."),
            "harness": "local",
            "gpu_used": True,
            "official_or_ranked": False,
            "commit": git_sha(),
            "base_sha": BASE_SHA,
            "pr": 154,
            "assignment_id": "qwen38-r1-e154-anchor-receipt-and-boundedness",
            "host": "Mac16,11 Apple M4 Pro 48 GiB",
            "ranked_host": "m5-qwen38-27b-mtp (NOT this host)",
            "decode_tokens": 512,
            "rule_134_fitted_on": RULE_134_FITTED_ON,
            "rule_161_price_basis": "median pair",
            "isolated_probe_dispatch_factor": 0.21,
            "r0_cool_gate": "real 40 C gate, never bypassed",
            "r2_cool_gate": (
                "deliberately disabled under the counterbalanced-arms "
                "exemption; ABBA within one session, entry and exit "
                "temperature recorded per arm"),
            "instrument_env_prefix": "MLX_E154_",
            "instrument_default": "off",
        })

    summary: dict = {}
    log_r0(run, summary, r0)
    if r1 is not None:
        log_r1(run, summary, r1)
    if r2 is not None:
        log_r2(run, summary, r2)
    else:
        print("warning: no R2 artifact, logging R0 and R1 only",
              file=sys.stderr)
    if bw is not None:
        log_bandwidth(run, summary, bw)

    if args.verdict:
        summary["e154_boundedness_verdict"] = args.verdict

    run.summary.update(summary)
    print("run id   %s" % run.id)
    print("run url  %s" % run.url)
    for key in sorted(summary):
        print("  %-52s %s" % (key, summary[key]))
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
