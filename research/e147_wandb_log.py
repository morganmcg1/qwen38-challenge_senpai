#!/usr/bin/env python3
"""E147: log every required metric and the full identity tuple to one W&B run.

HARNESS LABELLING. `program.md` invalidates an unlabelled score model, and this
experiment carries two harnesses that must never be mixed:

  harness=local   the M4 Pro ABBA session. It measures the NON-NAX
                  `affine_qmm_t`, which is the seed-prefill path on a host
                  where `is_nax_available()` is false. It validates the
                  transformation. It is not a ranked estimate and it is not
                  gate qualified.
  harness=ranked  the eight published `prefill_seconds_per_token` values from
                  an official receipt, and the published-median repricing
                  derived from them. Both ranked legs pay the same absolute
                  seed prefill, so `raw' = (S - dP) / (C - dP)` and no
                  `psi_serial` term exists or may be subtracted.

Every metric name below carries its harness, and `measurement_harness` in the
config records which one produced each artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
TRACK_ID = "qwen3.8-27b-mtp-v1"

# harness=ranked. The published seed prefill at the current bar, and the modern
# band every recent row except two sits inside.
RANKED_PREFILL_SPT_BAR = 0.001028291
RANKED_BAND_LOW = 0.0010277
RANKED_BAND_HIGH = 0.0010349

# harness=ranked. Advisor value model, published median against prefill cut.
FORECAST_TABLE = {
    0.0: 0.0,
    -1.5: 0.1122,
    -3.0: 0.2251,
    -4.1: 0.3071,
    -5.1: 0.3822,
}

# Threadgroup staging footprint in bytes, derived from the instantiation
# macros and the allocation expressions rather than measured.
THREADGROUP_BYTES = {
    "non_nax_32x32x32_bf16_xs_and_ws_doubled": 10_240,
    "non_nax_32x32x32_float32_xs_and_ws_doubled": 18_432,
    "nax_64x64x64_bf16_ws_doubled": 18_432,
    "nax_64x64x64_float32_ws_doubled_would_be": 34_816,
    "nax_64x64x64_float32_kept_unpipelined": 17_408,
    "limit": 32_768,
}


def load_json(path: str):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def load_kv(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def interpolate_forecast(pct: float) -> float:
    """Linear interpolation of the advisor's ranked value table."""
    xs = sorted(FORECAST_TABLE)
    if pct >= xs[-1]:
        return FORECAST_TABLE[xs[-1]]
    if pct <= xs[0]:
        # Below the tabulated range, extend the local slope.
        lo, hi = xs[0], xs[1]
    else:
        lo = max(x for x in xs if x <= pct)
        hi = min(x for x in xs if x >= pct)
        if lo == hi:
            return FORECAST_TABLE[lo]
    span = hi - lo
    frac = (pct - lo) / span
    return FORECAST_TABLE[lo] + frac * (FORECAST_TABLE[hi] - FORECAST_TABLE[lo])


def rung_a_metrics(rung_a: dict) -> tuple[dict, dict]:
    legs = rung_a["legs"]
    base = [leg["seed_prefill_seconds"] for leg in legs if leg["arm"] == "base"]
    cand = [leg["seed_prefill_seconds"] for leg in legs if leg["arm"] == "cand"]
    mean_base = statistics.fmean(base)
    mean_cand = statistics.fmean(cand)
    sd_base = statistics.stdev(base)
    sd_cand = statistics.stdev(cand)
    stderr = math.sqrt(sd_base**2 / len(base) + sd_cand**2 / len(cand))
    diff = mean_cand - mean_base
    pct = 100.0 * diff / mean_base
    pooled = rung_a["pooled"]

    metrics = {
        "e147_rungA_local_prefill_pct": pct,
        "e147_rungA_local_prefill_pct_block_form": pooled["prefill_gain_pct"],
        "e147_rungA_local_prefill_pct_stderr": 100.0 * stderr / mean_base,
        "e147_rungA_local_prefill_t": diff / stderr,
        "e147_local_seed_prefill_seconds_base": mean_base,
        "e147_local_seed_prefill_seconds_arm": mean_cand,
        "e147_local_seed_prefill_spt_base": mean_base / 512.0,
        "e147_local_to_ranked_level_factor": (mean_base / 512.0)
        / RANKED_PREFILL_SPT_BAR,
        "e147_rungA_exactness_divergences": float(pooled["divergences"]),
        "e147_rungA_unmatched_legs": float(pooled["unmatched_legs"]),
        "e147_rungA_witness_mismatches": float(pooled["witness_mismatches"]),
        "e147_rungA_legs": float(pooled["leg_count"]),
        "e147_rungA_decode_spt_gain_pct": pooled["decode_spt_gain_pct"],
        "e147_rungA_rank_separation_complete": float(max(cand) < min(base)),
    }
    for prompt, row in rung_a["prompts"].items():
        metrics[f"e147_rungA_local_prefill_pct_{prompt}"] = row["prefill_gain_pct"]
        metrics[f"e147_rungA_prefill_base_seconds_{prompt}"] = row["prefill_base_mean"]
        metrics[f"e147_rungA_prefill_cand_seconds_{prompt}"] = row["prefill_cand_mean"]

    config = {
        "rungA_identity": rung_a["identity"],
        "rungA_estimator": rung_a["estimator"],
        "rungA_arm_selector": rung_a["arm_selector"],
        "rungA_entry_temp_c_min": pooled.get("entry_temp_c_min"),
        "rungA_entry_temp_c_max": pooled.get("entry_temp_c_max"),
        "rungA_blocks_pct": pooled["prefill_gain_pct_per_block"],
    }
    return metrics, config


BUILD_AFFECTING_PREFIXES = (
    "Package.swift",
    "Sources/",
    "Vendor/",
    "mtp-head.manifest.json",
    "mtp-head/",
)


def submitted_surface_check() -> tuple[dict, dict]:
    """Prove the ranked host builds the same worker this host measured.

    Yukon packages only `editablePaths`, so the ranked build is the
    organizer's tree at `upstream/main` with our copies of those paths
    substituted. A file that changes the built worker but is NOT submitted
    therefore exists only here, and any local measurement that depends on it
    does not transfer.

    The test is one set difference: take every build-affecting path where
    HEAD diverges from `upstream/main`, and require the unsubmitted part to
    be empty. This subsumes the specific worry for this experiment, which is
    that `Package.swift`, `jit_kernels.cpp` and `nojit_kernels.cpp` decide
    whether the live Metal source form is the JIT string or `mlx.metallib`.

    Research scripts, tests and campaign notes are excluded on purpose. They
    are not submitted and they cannot change the worker binary.
    """
    track = json.loads(pathlib.Path("benchmark.json").read_text())
    if track["trackId"] != TRACK_ID:
        raise SystemExit(f"benchmark.json is track {track['trackId']}, not {TRACK_ID}")
    editable = track["editablePaths"] + track.get("optionalEditablePaths", [])

    diverged = subprocess.run(
        ["git", "diff", "--name-only", "upstream/main", "HEAD"],
        capture_output=True, text=True, check=True).stdout.split()
    submitted, local_only = [], []
    for path in diverged:
        if not path.startswith(BUILD_AFFECTING_PREFIXES):
            continue
        covered = any(path == e or path.startswith(e.rstrip("/") + "/")
                      for e in editable)
        (submitted if covered else local_only).append(path)

    source_form_files = [
        "Vendor/mlx-swift/Package.swift",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/jit_kernels.cpp",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/nojit_kernels.cpp",
    ]
    drifted = sorted(set(source_form_files) & set(local_only))

    metrics = {
        "e147_submitted_surface_self_contained": float(not local_only),
        "e147_build_decision_files_match_upstream": float(not drifted),
        "e147_submitted_changed_file_count": float(len(submitted)),
    }
    config = {
        "submitted_changed_files": sorted(submitted),
        "local_only_build_affecting_files": sorted(local_only),
        "source_form_files_checked": source_form_files,
        "source_form_files_drifted": drifted,
        "live_source_form_note": (
            "Package.swift excludes nojit_kernels.cpp and "
            "jit_kernels.cpp get_qmm_nax_kernel concatenates "
            "metal::quantized_nax(), which is mlx-generated/quantized_nax.cpp. "
            "That twin is a submitted path and the three deciding files are "
            "organizer-owned and unmodified, so the JIT string proven live "
            "locally is the one the ranked host compiles."
        ),
    }
    return metrics, config


def rung_d_metrics(receipt: dict) -> tuple[dict, dict]:
    """harness=ranked. Read one official receipt through research/e147_rungD.py.

    Feedback 3 adds three readouts on top of the prefill headline: D-1 proves
    which depth-price column the built worker shipped, D-2 prices the
    decode-only difference against the only public pb6 anchor in whole units
    of the campaign state-step constant, and D-3 recovers the round count each
    prompt actually ran.
    """
    values = receipt["prefill_seconds_per_token"]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    pct = 100.0 * (mean / RANKED_PREFILL_SPT_BAR - 1.0)
    forecast = interpolate_forecast(pct)
    realised = receipt["realised_median_pct"]
    metrics = {
        "e147_ranked_prefill_pct": pct,
        "e147_ranked_prefill_spt_mean": mean,
        "e147_ranked_prefill_within_row_cv_pct": 100.0 * sd / mean,
        "e147_ranked_prefill_vs_band_low_pct": 100.0 * (mean / RANKED_BAND_LOW - 1.0),
        "e147_ranked_prefill_vs_band_high_pct": 100.0 * (mean / RANKED_BAND_HIGH - 1.0),
        "e147_forecast_median_pct": forecast,
        "e147_realised_median_pct": realised,
        "e147_forecast_error_pp": realised - forecast,
        "e147_realised_median_pct_model_a": receipt["realised_median_pct_model_a"],
        "e147_ranked_score": receipt["score"],
        "e147_ranked_published_median": receipt["published_median_recomputed"],
        # D-1.
        "e147_schedule_matches_pb6_column": receipt["schedule_matches_pb6_column"],
        "e147_schedule_l1_to_ship_column": receipt["schedule_column_l1_distance_ship"],
        "e147_schedule_l1_to_pb6_column": receipt["schedule_column_l1_distance_pb6"],
        "e147_ranked_non_drafting_round_count_plutarch": float(
            receipt["non_drafting_round_count_by_prompt"]["plutarch"]),
        # D-2.
        "e147_implied_state_step_us": receipt["implied_state_step_us"],
        "e147_implied_state_steps": receipt["implied_state_steps"],
        "e147_implied_state_steps_median": receipt["implied_state_steps_median"],
        "e147_state_steps_integer": receipt["state_steps_integer"],
        "e147_state_steps_integer_fraction": receipt["state_steps_integer_fraction"],
        "e147_corrected_decode_seconds_total": receipt[
            "corrected_decode_seconds_total"],
        "e147_decode_only_seconds_total": receipt["decode_only_seconds_total"],
        "e147_decode_only_seconds_total_anchor": receipt[
            "decode_only_seconds_total_anchor"],
        # D-3.
        "e147_rounds_closed_form_max_error": receipt["rounds_closed_form_max_error"],
        "e147_rounds_closed_form_reproduces_receipt": receipt[
            "rounds_closed_form_reproduces_receipt"],
        # Feedback 5. Both halves of the bit-identical zero-delta anchor pair,
        # and the measured noise the effect is judged against.
        "e147_prefill_pct_vs_684821ed": receipt["e147_prefill_pct_vs_684821ed"],
        "e147_prefill_pct_vs_f7d59543": receipt["e147_prefill_pct_vs_f7d59543"],
        "e147_prefill_anchor_spread_pp": receipt["e147_prefill_anchor_spread_pp"],
        "e147_prefill_anchor_agreement_ok": receipt[
            "e147_prefill_anchor_agreement_ok"],
        "e147_prefill_effect_z_vs_pair_se": receipt[
            "e147_prefill_effect_z_vs_pair_se"],
        "e147_prefill_effect_sd_pct": receipt["e147_prefill_effect_sd_pct"],
        "e147_prefill_effect_dispersion_ratio": receipt[
            "e147_prefill_effect_dispersion_ratio"],
        "e147_prefill_worst_residual_pct": receipt["e147_prefill_worst_residual_pct"],
        "e147_prefill_medicine_is_worst_residual": receipt[
            "e147_prefill_medicine_is_worst_residual"],
        "e147_prefill_medicine_residual_pct": receipt[
            "e147_prefill_medicine_residual_pct"],
        "e147_prefill_prompts_below_band": receipt["e147_prefill_prompts_below_band"],
    }
    for prompt in receipt["prompt_order"]:
        metrics[f"e147_ranked_mtp_spt_{prompt}"] = receipt[
            "mtp_seconds_per_token_mean_by_prompt"][prompt]
        metrics[f"e147_ranked_serial_spt_{prompt}"] = receipt[
            "serial_seconds_per_token_mean_by_prompt"][prompt]
        metrics[f"e147_prefill_pct_{prompt}"] = receipt[
            "prefill_pct_by_prompt_vs_684821ed"][prompt]
        metrics[f"e147_prefill_residual_{prompt}"] = receipt[
            "prefill_residual_by_prompt"][prompt]
        metrics[f"e147_ranked_prefill_spt_{prompt}"] = receipt[
            "prefill_seconds_per_token_by_prompt"][prompt]
        metrics[f"e147_ranked_dlen_{prompt}"] = receipt["dlen_by_prompt"][prompt]
        metrics[f"e147_ranked_raw_{prompt}"] = receipt["raw_ratio_by_prompt"][prompt]
        metrics[f"e147_implied_rounds_{prompt}"] = receipt[
            "implied_rounds_by_prompt"][prompt]
        metrics[f"e147_exact_rounds_{prompt}"] = float(
            receipt["exact_rounds_by_prompt"][prompt])
        metrics[f"e147_rounds_closed_form_error_{prompt}"] = receipt[
            "rounds_closed_form_error_by_prompt"][prompt]
        metrics[f"e147_decode_only_seconds_{prompt}"] = receipt[
            "decode_only_seconds_by_prompt"][prompt]
        step = receipt["implied_state_steps_by_prompt"][prompt]
        if step is not None:
            metrics[f"e147_implied_state_steps_{prompt}"] = step
    config = {
        "ranked_prefill_spt_per_prompt": values,
        "ranked_submission_id": receipt["submission_id"],
        "ranked_status": receipt["status"],
        "ranked_promotion_status": receipt["promotion_status"],
        "ranked_bar_id8": receipt["bar_id8"],
        "ranked_bar_published_median": receipt["bar_published_median"],
        "rungD_anchor_id8": receipt["anchor_id8"],
        "rungD_schedule_comparable_prompts": receipt["schedule_comparable_prompts"],
        "rungD_state_step_constant_us": receipt["state_step_constant_us"],
        "rungD_state_step_constant_sd_us": receipt["state_step_constant_sd_us"],
        "rungD_forecast_model": "model_b: raw' = (S - dP)/(C - dP)",
        "rungD_alternative_model": (
            "model_a: raw' = S/(C - dP); implied by program.md because the "
            "ranked serial numerator comes from a runner-owned prebuilt "
            "baseline workspace"
        ),
        "prefill_noise_se_pct": receipt["prefill_noise_se_pct"],
        "prefill_noise_sd_pct": receipt["prefill_noise_sd_pct"],
        "prefill_noise_source": (
            "feedback 5, finding 237: bit-identical ranked pair "
            "684821ed / f7d59543, ground truth exactly zero"
        ),
        "prefill_band_cluster_by_prompt": receipt["prefill_band_cluster_by_prompt"],
        "zero_delta_pair_pct_by_prompt": receipt["zero_delta_pair_pct_by_prompt"],
        "prefill_worst_residual_prompt": receipt["e147_prefill_worst_residual_prompt"],
    }
    return metrics, config


# One entry per later artifact: the file, the config key that carries it whole,
# and the scalars promoted into the metric series. A key that is absent from a
# present file is an error, because a renamed field must not silently drop a
# reported metric.
ARTIFACTS = [
    ("research/e147-dispatch.json", "dispatch", {
        "e147_dispatch_ranked_all_nax": "ranked_all_nax",
        "e147_dispatch_local_all_qmm_t_impl": "local_all_qmm_t_impl",
        "e147_dispatch_any_scored_uses_splitk": "any_scored_uses_splitk",
        "e147_dispatch_f6_section5_confirmed": "f6_section5_confirmed",
    }),
    ("research/e147-f8-followup.json", "f8_followup", {
        "e147_max_threadgroup_bytes": "e147_max_threadgroup_bytes",
        "e147_every_instantiated_shape_fits": "every_instantiated_shape_fits",
        "e147_our_rows_share_one_source_ref": "our_rows_share_one_source_ref",
    }),
    ("research/e147-base-diff.json", "base_diff", {
        "e147_base_arm_gap_explains_four_percent":
            "e147_base_arm_gap_explains_four_percent",
        "e147_base_diff_predicted_pct_additive":
            "predicted_our_leg_slower_pct_additive",
        "e147_base_diff_predicted_pct_multiplicative":
            "predicted_our_leg_slower_pct_multiplicative",
        "e147_base_diff_observed_weighted_five_pct": "observed_weighted_five_pct",
        "e147_base_diff_observed_unweighted_mean_pct": "observed_unweighted_mean_pct",
        "e147_base_diff_max_abs_residual_sigma": "max_abs_residual_sigma",
    }),
    ("research/e147-qmv-jit-census.json", "qmv_jit_census", {
        "e147_decode_entry_points_moved": "decode_entry_points_moved",
        "e147_rule_101_census_control_moved": "rule_101_positive_control_moved",
    }),
    ("research/e147-retile-arm.json", "rungE1a_retile_arm", {
        "e147_rungE_grid_stride_max_iters": "e147_rungE_grid_stride_max_iters",
        "e147_rungE_grid_stride_max_iters_probe":
            "e147_rungE_grid_stride_max_iters_probe",
        "e147_rungE1a_scored_rows_ok": "scored_rows_ok",
        "e147_rungE1a_probe_rows_ok": "probe_rows_ok",
        "e147_rungE1a_all_controls_passed": "all_controls_passed",
        "e147_rungE1a_all_source_assertions_passed": "all_source_assertions_passed",
    }),
    ("research/e147-rungE2.json", "rungE2_nax_compile", {
        "e147_rungE2_retile_compiles": "e147_rungE2_retile_compiles",
        "e147_rungE2_k_order_preserved": "e147_rungE2_k_order_preserved",
        "e147_rungE2_matmad_branch_switches": "e147_rungE2_matmad_branch_switches",
        "e147_rungE2_register_census_available":
            "e147_rungE2_register_census_available",
        "e147_rungE2_scored_shape_also_untranslatable":
            "e147_rungE2_scored_shape_also_untranslatable",
        "e147_rungE2_retile_refusal_identical_to_scored":
            "e147_rungE2_retile_refusal_identical_to_scored",
        "e147_rungE2_failopen_shape_exists": "e147_rungE2_failopen_shape_exists",
        "e147_rungE2_failopen_control_observed":
            "e147_rungE2_failopen_control_observed",
    }),
    ("research/e147-rungE1c.json", "rungE1c_nax_arm", {
        "e147_rungE1c_arm_on_compiles": "e147_rungE1c_arm_on_compiles",
        "e147_rungE1c_arm_on_differs_from_arm_off":
            "e147_rungE1c_arm_on_differs_from_arm_off",
        "e147_rungE1c_illegal_shape_rejected":
            "e147_rungE1c_illegal_shape_rejected",
        "e147_rungE1c_digest_control_moves":
            "e147_rungE1c_digest_control_moves",
        "e147_rungE1c_nax_arm_off_air_delta_bytes":
            "e147_rungE1c_nax_arm_off_air_delta_bytes",
        "e147_rungE1c_transfer_air_delta_bytes":
            "e147_rungE1c_transfer_air_delta_bytes",
        "e147_rungE1c_transfer_isa_text_delta_applegpu_g17s":
            "e147_rungE1c_transfer_isa_text_delta_applegpu_g17s",
        "e147_rungE1c_transfer_register_delta_applegpu_g17s":
            "e147_rungE1c_transfer_register_delta_applegpu_g17s",
        "e147_rungE1c_transfer_spill_delta_applegpu_g17s":
            "e147_rungE1c_transfer_spill_delta_applegpu_g17s",
        "e147_rungE1c_failopen_compiles_unguarded":
            "e147_rungE1c_failopen_compiles_unguarded",
        "e147_rungE1c_failopen_shape_rejected":
            "e147_rungE1c_failopen_shape_rejected",
        "e147_rungE1c_rule145_named_in_refusal":
            "e147_rungE1c_rule145_named_in_refusal",
        "e147_rungE1c_rule145_control_observed":
            "e147_rungE1c_rule145_control_observed",
        "e147_rungE1c_guarded_tile_sites": "e147_rungE1c_guarded_tile_sites",
        "e147_rungE1c_shipped_other_sites_compile":
            "e147_rungE1c_shipped_other_sites_compile",
    }),
    ("research/e147-qmv-jit-census.json", "jit_census_validity", {
        "e147_census_valid": "census_valid",
        "e147_qmv_jit_census_covers_nax_gemm": "covers_nax_gemm",
    }),
]

# F11 dispositions that are decisions, not measurements. They are logged so the
# result artifact and W&B agree on what this branch actually ships.
DISPOSITIONS = {
    # Rung E-1b proved the retile mechanism bit exact on hardware, but it proved
    # it for the non-NAX twin. The index algebra transfers to NAX; the
    # arithmetic, `tile_matmad_nax` operand packing, does not. It is a rehearsal.
    "e147_rungE1b_ships": 0.0,
}

# harness=local. Rung E-1b is an exactness result, never a timing result.
RUNG_E1B_FLAGS = [
    ("e147_rungE1b_retile_exact_local", "e147_rungE_reindex_exact_local"),
    ("e147_rungE1b_positive_control_caught",
     "e147_rungE1b_positive_control_caught"),
    ("e147_rungE1b_arm_contrast", "e147_rungE1b_arm_contrast_nonempty"),
    ("e147_rungE1b_tree_restored", "e147_rungE4_submitted_default_is_off"),
    ("e147_rungE1b_control_differs_from_retile",
     "e147_rungE1b_control_differs_from_retile"),
    ("e147_rungE1b_verdict", "e147_rungE1b_verdict_pass"),
]

TRUTHY = {"true": 1.0, "false": 0.0, "ok": 1.0, "1": 1.0, "0": 0.0,
          "PASS": 1.0, "FAIL": 0.0, "EMPTY": 0.0}


def artifact_metrics() -> tuple[dict, dict]:
    metrics: dict = {}
    config: dict = {}
    for path, config_key, promoted in ARTIFACTS:
        payload = load_json(path)
        if payload is None:
            continue
        config[config_key] = payload
        for metric_name, source_key in promoted.items():
            if source_key not in payload:
                raise SystemExit(f"{path} has no field {source_key!r}")
            metrics[metric_name] = float(payload[source_key])
    return metrics, config


def rung_e1b_metrics(path: str) -> tuple[dict, dict]:
    kv = load_kv(path)
    if not kv:
        return {}, {}
    metrics = {}
    for source_key, metric_name in RUNG_E1B_FLAGS:
        raw = kv.get(source_key)
        if raw is None:
            raise SystemExit(f"{path} has no field {source_key!r}")
        if raw not in TRUTHY:
            raise SystemExit(f"{path}: {source_key}={raw!r} is not a verdict")
        metrics[metric_name] = TRUTHY[raw]
    for prompt in ("beagle_a", "essays_montaigne"):
        matched = kv.get(f"e147_rungE1b_{prompt}_all_tokens_matched")
        if matched is not None:
            metrics[f"e147_rungE1b_all_tokens_matched_{prompt}"] = TRUTHY[matched]
            metrics[f"e147_rungE1b_residual_divergence_count_{prompt}"] = float(
                kv[f"e147_rungE1b_{prompt}_residual_divergence_count"])
    return metrics, {"rungE1b_detail": kv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung-a", default="research/e147-rungA.json")
    ap.add_argument("--control", default="research/e147-rungA-control.txt")
    ap.add_argument("--receipt", default="research/e147-rungD.json")
    ap.add_argument("--rung-e1b", default="research/e147-rungE1b.txt")
    ap.add_argument("--name", default="e147-nax-seed-prefill-double-buffer")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    config: dict = {
        "experiment": "E147",
        "pr": 147,
        "commit": commit,
        "base_sha": "bcc11dc6527ea4fa32be22c15b99a3a95695bfaf",
        # Scope and byte budget are measured against the PR base, not against
        # 770a3ff2. 770a3ff2 is byte-identical to the PR base on quantized_nax.h
        # and its twin, which makes it a valid SOURCE base for the NAX compile
        # rungs, but it is several hundred commits behind on quantized.h, so
        # using it as the growth base charges this experiment for other
        # students' promoted work.
        "assignment_scope_base": "bcc11dc6527ea4fa32be22c15b99a3a95695bfaf",
        "nax_compile_rung_base": "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf",
        "e147_rungE_base_composition": (
            "clean PR base bcc11dc6 on quantized.h and its twin (rung A and "
            "rung B both reverted); quantized_nax.h carries only the (128, 32) "
            "retile arm, shipped off, plus the RULE 145 tile guards"
        ),
        "mechanism_measured": (
            "software-pipeline the affine quantized transposed GEMM k-loop: "
            "double the threadgroup staging buffer, stage tile 0 in a "
            "prologue, keep one barrier per iteration, alternate halves. "
            "REFUTED on the ranked host by receipt 7226dc9a and reverted"
        ),
        "mechanism_shipped": (
            "grid-stride (128, 32) retile of the NAX transposed GEMM "
            "threadgroup tile, shipped off behind kE147NaxRetileOn"
        ),
        "reference_implementation": "fp_quantized_nax.h shipped pipelined k-loop",
        "reassociation": "none; per-element mma sequence unchanged (Rule 92)",
        "live_source_form": (
            "Metal JIT string compiled into the runtime worker; "
            "nojit_kernels.cpp is excluded from the package"
        ),
        "threadgroup_bytes": THREADGROUP_BYTES,
        "e147_max_threadgroup_bytes": max(
            v for k, v in THREADGROUP_BYTES.items()
            if k not in ("limit", "nax_64x64x64_float32_ws_doubled_would_be")
        ),
        "host": "Mac16,11 Apple M4 Pro 48 GiB applegpu_g16s",
        "nax_available_locally": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "depth_price_arm": "pb6 (compiled default, MLX_E134_DEPTH_PRICE_ARM unset)",
        "ranked_prefill_spt_bar": RANKED_PREFILL_SPT_BAR,
        "ranked_value_model": "raw' = (S - dP)/(C - dP); no psi_serial term",
    }

    metrics: dict = {
        "e147_max_threadgroup_bytes": float(config["e147_max_threadgroup_bytes"]),
        **DISPOSITIONS,
    }

    surface_m, surface_c = submitted_surface_check()
    metrics.update(surface_m)
    config.update(surface_c)

    rung_a = load_json(args.rung_a)
    if rung_a:
        rung_a_m, rung_a_c = rung_a_metrics(rung_a)
        metrics.update(rung_a_m)
        config.update(rung_a_c)

    control = load_kv(args.control)
    if control:
        fired = control.get("e147_rungA_positive_control_failed")
        if fired is not None:
            metrics["e147_rungA_positive_control_failed"] = float(fired)
        for mode in ("barrier0", "nobarrier", "wronghalf"):
            # A mode that never ran records "none" rather than a verdict, and
            # must stay out of the numeric series instead of being coerced to 0.
            value = control.get(f"e147_control_{mode}_caught")
            if value not in (None, "none"):
                metrics[f"e147_rungA_control_{mode}_caught"] = float(value)
        config["rungA_control_detail"] = control

    receipt = load_json(args.receipt)
    if receipt:
        rung_d_m, rung_d_c = rung_d_metrics(receipt)
        metrics.update(rung_d_m)
        config.update(rung_d_c)

    art_m, art_c = artifact_metrics()
    metrics.update(art_m)
    config.update(art_c)

    e1b_m, e1b_c = rung_e1b_metrics(args.rung_e1b)
    metrics.update(e1b_m)
    config.update(e1b_c)

    revert = subprocess.run(
        ["git", "diff", "--stat", "bdba19f66e84e7e2aa1f8162eeaa0a82579edc94",
         "--", "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"],
        capture_output=True, text=True, check=True).stdout.strip()
    metrics["e147_e141_revert_is_byte_exact"] = 1.0 if revert == "" else 0.0

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        config=config,
        mode="offline" if args.offline else "online",
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    print(f"e147 wandb run {run.id} {run.url}")
    for key in sorted(metrics):
        print(f"  {key} = {metrics[key]}")
    run.finish()


if __name__ == "__main__":
    main()
