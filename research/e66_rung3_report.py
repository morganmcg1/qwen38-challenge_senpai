#!/usr/bin/env python3
"""E66 rung 3: do `t55` and `t6` compose additively on a whole 512-token leg?

Three arms in one session:

  A = a_neither  `<T,5,3>` and `<T,6,3>`, wide-helper bound NA <= 5.
  B = b_t6       the merged base, `<T,5,3>` and `<T,6,6>`.
  C = c_t55_t6   the candidate, `<T,5,5>` and `<T,6,6>`.

Design: one declared and discarded warm-up leg, then
`a1 a1 b1 b1 c1 c1 c2 c2 b2 b2 a2 a2`. Leg positions are A={1,2,11,12},
B={3,4,9,10}, C={5,6,7,8}: position sum 26 and mean 6.5 for all three arms, so
the arm effect is orthogonal to linear thermal or clock drift.

`mtp_seconds_per_token ~ arm + leg_position` is fitted on the twelve timed legs
with arm as a three-level factor and A as the reference.

ONE IDENTITY MATTERS FOR READING THIS FILE. Inside a linear model,
`(C-A) = (B-A) + (C-B)` holds by construction, so this session cannot test
additivity against itself. The test is against two INDEPENDENT single-mechanism
measurements: E61 rung 3's `t6` on this same host, and E59 rung 4's `t55` on a
different host. Sub-additivity would appear as a measured `C-A` materially
smaller in magnitude than their sum.

Timing used the permitted local ungated protocol, so the three gate flags travel
verbatim and this is not a ranked or official score.

  python3 research/e66_rung3_report.py --out research/e66-rung3.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import pathlib
import statistics

import numpy as np

RUNS = pathlib.Path(".mlxfast-private/e66/runs")

# tag -> (arm, discarded)
PLAN = (
    ("warm", "b_t6", True),
    ("a1", "a_neither", False),
    ("b1", "b_t6", False),
    ("c1", "c_t55_t6", False),
    ("c2", "c_t55_t6", False),
    ("b2", "b_t6", False),
    ("a2", "a_neither", False),
)
ARMS = ("a_neither", "b_t6", "c_t55_t6")

# Preregistered single-mechanism effects, each measured elsewhere.
PREREG = {
    "B_minus_A": {"pct": -0.5251, "t": -14.19,
                  "source": "askeladd E61 rung 3, this host, t6 alone"},
    "C_minus_B": {"pct": -0.7689, "t": -10.68,
                  "source": "thorfinn E59 rung 4, different host, t55 alone"},
}
PREREG_SUM_PCT = PREREG["B_minus_A"]["pct"] + PREREG["C_minus_B"]["pct"]

NULL_CHANNELS = (
    ("serial_leg_seconds_per_token", "serial_seconds_per_token"),
    ("serial_round_cost_p50_seconds", "serial_p50_block_seconds"),
    ("serial_seed_prefill_seconds", "serial_seed_prefill_seconds"),
    ("mtp_seed_prefill_seconds", "mtp_seed_prefill_seconds"),
)


def pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0


def read_meta(tag: str) -> dict:
    out = {}
    for line in (RUNS / tag / "meta.txt").read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def read_legs() -> list[dict]:
    """One record per leg, in the order they ran."""
    legs = []
    position = 0
    for tag, arm, discarded in PLAN:
        meta = read_meta(tag)
        for path in sorted(glob.glob(str(RUNS / tag / "score-*.json"))):
            index = int(pathlib.Path(path).stem.split("-")[1])
            metrics = json.loads(pathlib.Path(path).read_text())["metrics"]
            rep = RUNS / tag / "reports" / ("leg-%d" % index)
            serial = json.loads((rep / "03-mtp-timed.json").read_text())
            mtp = json.loads((rep / "04-mtp-timed.json").read_text())
            if not serial.get("is_serial_control"):
                raise SystemExit("%s: 03-mtp-timed.json is not the serial control" % rep)
            if mtp.get("is_serial_control"):
                raise SystemExit("%s: 04-mtp-timed.json is the serial control" % rep)
            if not discarded:
                position += 1
            legs.append({
                "tag": tag,
                "arm": arm,
                "discarded": discarded,
                "leg_index": index,
                "position": None if discarded else position,
                "head_sha": meta.get("head_sha"),
                "dirty": meta.get("dirty"),
                "twin_digests": meta.get("twin_digests"),
                "binary_assert_m5_na": meta.get("e66_binary_assert_m5_na"),
                "binary_assert_m6_na": meta.get("e66_binary_assert_m6_na"),
                "binary_assert_m9_na": meta.get("e66_binary_assert_m9_na"),
                "binary_assert_wide_bound": meta.get("e66_binary_assert_wide_bound"),
                "worker_sha256_before": meta.get("leg%d_before_worker_sha256" % index),
                "worker_sha256_after": meta.get("leg%d_after_worker_sha256" % index),
                "worker_unchanged_across_leg":
                    meta.get("leg%d_worker_unchanged_across_leg" % index),
                "metallib_source_fingerprint":
                    meta.get("leg%d_before_metallib_source_fingerprint" % index),
                "thermal_before": meta.get("thermal_before") if index == 1
                    else meta.get("leg%d_thermal_after" % (index - 1)),
                "thermal_after": meta.get("leg%d_thermal_after" % index),
                "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
                "serial_seconds_per_token": metrics["serial_seconds_per_token"],
                "mtp_decode_speedup": metrics["mtp_decode_speedup"],
                "effective_mean_draft_len": metrics["effective_mean_draft_len"],
                "accepted_draft_rate": metrics["accepted_draft_rate"],
                "all_tokens_matched": metrics["all_tokens_matched"],
                "residual_divergence_count": metrics["residual_divergence_count"],
                "decode_tokens": metrics["decode_tokens"],
                "serial_p50_block_seconds": serial["p50_block_request_seconds"],
                "serial_seed_prefill_seconds": serial["seed_prefill_seconds"],
                "serial_round_count": serial["round_count"],
                "mtp_seed_prefill_seconds": mtp["seed_prefill_seconds"],
                "mtp_p50_block_seconds": mtp["p50_block_request_seconds"],
                "mtp_round_count": mtp["round_count"],
                "mtp_decode_seconds": mtp["decode_seconds"],
            })
    return legs


def gpu_temp(sample: str | None) -> float | None:
    if not sample or not sample.startswith("gpu_temp="):
        return None
    return float(sample.split("gpu_temp=")[1].split("C")[0])


def fit(timed: list[dict], key: str) -> dict:
    """OLS of `key ~ arm + leg_position`, arm A as the reference level."""
    y = np.array([l[key] for l in timed], dtype=float)
    X = np.array([[1.0,
                   1.0 if l["arm"] == "b_t6" else 0.0,
                   1.0 if l["arm"] == "c_t55_t6" else 0.0,
                   float(l["position"])] for l in timed])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * xtx_inv
    ref = statistics.fmean(l[key] for l in timed if l["arm"] == "a_neither")

    def contrast(vec: list[float]) -> dict:
        c = np.array(vec, dtype=float)
        est = float(c @ beta)
        se = float((c @ cov @ c) ** 0.5)
        return {"estimate": est, "stderr": se, "t": est / se if se else None,
                "pct_of_arm_a_mean": est / ref * 100.0,
                "stderr_pct_of_arm_a_mean": se / ref * 100.0}

    return {
        "model": "%s ~ arm + leg_position, arm A reference" % key,
        "n": len(y),
        "dof": dof,
        "arm_a_mean": ref,
        "intercept": float(beta[0]),
        "residual_sd": s2 ** 0.5,
        "position_coef_per_leg": float(beta[3]),
        "position_coef_pct_per_leg": float(beta[3]) / ref * 100.0,
        "position_coef_stderr": float(cov[3][3] ** 0.5),
        "position_coef_t": float(beta[3] / cov[3][3] ** 0.5),
        "contrasts": {
            "B_minus_A": contrast([0, 1, 0, 0]),
            "C_minus_A": contrast([0, 0, 1, 0]),
            "C_minus_B": contrast([0, -1, 1, 0]),
        },
    }


def same_arm_spreads(timed: list[dict], key: str) -> dict:
    """This session's own null for one metric, from same-arm leg pairs."""
    pairs = []
    for arm in ARMS:
        legs = [l for l in timed if l["arm"] == arm]
        for a, b in itertools.combinations(legs, 2):
            pairs.append({
                "arm": arm,
                "positions": [a["position"], b["position"]],
                "separation": abs(a["position"] - b["position"]),
                "abs_delta_pct": abs(pct(b[key], a[key])),
            })
    by_arm = {}
    for arm in ARMS:
        vals = [l[key] for l in timed if l["arm"] == arm]
        by_arm[arm] = (max(vals) - min(vals)) / statistics.fmean(vals) * 100.0
    by_sep: dict[int, float] = {}
    for p in pairs:
        by_sep[p["separation"]] = max(by_sep.get(p["separation"], 0.0),
                                      p["abs_delta_pct"])
    return {
        "largest_same_arm_spread_pct": max(by_arm.values()),
        "same_arm_spread_pct_by_arm": by_arm,
        "largest_same_arm_pair_delta_pct_by_separation":
            dict(sorted(by_sep.items())),
        "pairs": pairs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e66-rung3.json")
    args = ap.parse_args()

    legs = read_legs()
    timed = [l for l in legs if not l["discarded"]]
    warm = [l for l in legs if l["discarded"]]

    position_sums = {arm: sum(l["position"] for l in timed if l["arm"] == arm)
                     for arm in ARMS}
    position_means = {arm: statistics.fmean(
        l["position"] for l in timed if l["arm"] == arm) for arm in ARMS}

    by_arm = {}
    for arm in ARMS:
        vals = [l["mtp_seconds_per_token"] for l in timed if l["arm"] == arm]
        entries = [gpu_temp(l["thermal_before"]) for l in timed if l["arm"] == arm]
        exits = [gpu_temp(l["thermal_after"]) for l in timed if l["arm"] == arm]
        entries = [t for t in entries if t is not None]
        exits = [t for t in exits if t is not None]
        by_arm[arm] = {
            "legs": len(vals),
            "mtp_seconds_per_token_mean": statistics.fmean(vals),
            "mtp_seconds_per_token_min": min(vals),
            "mtp_seconds_per_token_max": max(vals),
            "same_arm_spread_pct": (max(vals) - min(vals)) / statistics.fmean(vals) * 100.0,
            "entry_gpu_temp_c": entries,
            "exit_gpu_temp_c": exits,
            "entry_gpu_temp_mean_c": statistics.fmean(entries) if entries else None,
        }

    all_entries = [t for a in ARMS for t in by_arm[a]["entry_gpu_temp_c"]]
    entry_spread = (max(all_entries) - min(all_entries)) if all_entries else None

    headline = fit(timed, "mtp_seconds_per_token")
    null = same_arm_spreads(timed, "mtp_seconds_per_token")
    session_null = null["largest_same_arm_spread_pct"]

    measured = {k: headline["contrasts"][k]["pct_of_arm_a_mean"]
                for k in ("B_minus_A", "C_minus_A", "C_minus_B")}
    measured_se = {k: headline["contrasts"][k]["stderr_pct_of_arm_a_mean"]
                   for k in ("B_minus_A", "C_minus_A", "C_minus_B")}

    prereg_se = {k: abs(v["pct"] / v["t"]) for k, v in PREREG.items()}
    prereg_sum_se = (prereg_se["B_minus_A"] ** 2 + prereg_se["C_minus_B"] ** 2) ** 0.5
    combined_se = (measured_se["C_minus_A"] ** 2 + prereg_sum_se ** 2) ** 0.5
    additivity_residual = measured["C_minus_A"] - PREREG_SUM_PCT

    # Per-leg arm effect: the leg value minus the position-adjusted arm A
    # prediction. Sign stability is judged on these, not on raw leg values.
    per_leg_effect = []
    for l in timed:
        pred_a = headline["intercept"] + headline["position_coef_per_leg"] * l["position"]
        per_leg_effect.append({
            "position": l["position"], "tag": l["tag"], "arm": l["arm"],
            "mtp_seconds_per_token": l["mtp_seconds_per_token"],
            "arm_effect_vs_position_adjusted_a_pct":
                (l["mtp_seconds_per_token"] - pred_a) / headline["arm_a_mean"] * 100.0,
        })
    c_effects = [e["arm_effect_vs_position_adjusted_a_pct"]
                 for e in per_leg_effect if e["arm"] == "c_t55_t6"]
    b_effects = [e["arm_effect_vs_position_adjusted_a_pct"]
                 for e in per_leg_effect if e["arm"] == "b_t6"]

    channels = {}
    for name, key in NULL_CHANNELS:
        f = fit(timed, key)
        n = same_arm_spreads(timed, key)
        channels[name] = {
            "metric": key,
            "arm_means": {arm: statistics.fmean(
                l[key] for l in timed if l["arm"] == arm) for arm in ARMS},
            "contrasts_pct": {k: v["pct_of_arm_a_mean"]
                              for k, v in f["contrasts"].items()},
            "own_largest_same_arm_spread_pct": n["largest_same_arm_spread_pct"],
            "own_spread_by_separation": n["largest_same_arm_pair_delta_pct_by_separation"],
            "quiet": all(abs(v["pct_of_arm_a_mean"]) < n["largest_same_arm_spread_pct"]
                         for v in f["contrasts"].values()),
            "position_coef_pct_per_leg": f["position_coef_pct_per_leg"],
            "enters_headline": False,
        }

    schedule_identical = (
        len({l["effective_mean_draft_len"] for l in timed}) == 1
        and len({l["accepted_draft_rate"] for l in timed}) == 1
        and len({l["mtp_round_count"] for l in timed}) == 1)
    correctness_ok = (
        all(l["all_tokens_matched"] for l in legs)
        and all(l["residual_divergence_count"] == 0 for l in legs)
        and all(l["decode_tokens"] == 512 for l in legs))
    headline_excludes_prefill = all(
        abs(l["mtp_decode_seconds"] / l["decode_tokens"]
            - l["mtp_seconds_per_token"]) < 1e-12 for l in timed)
    rung1_ok = all(l["worker_unchanged_across_leg"] == "true" for l in timed)
    dispatch_live = all(
        (l["binary_assert_m5_na"], l["binary_assert_m6_na"]) == {
            "a_neither": ("3", "3"), "b_t6": ("3", "6"),
            "c_t55_t6": ("5", "6")}[l["arm"]] for l in timed)

    replication_ok = abs(measured["B_minus_A"] - PREREG["B_minus_A"]["pct"]) < session_null

    report = {
        "experiment": "qwen38-r1-e66-composition-certification",
        "rung": 3,
        "question": "do t55 and t6 compose additively at the whole-leg level?",
        "base_sha": "45b4f3a800f879e3579ca27ef0b1c0ef40e4473d",
        "host": "Apple M4 Pro, 48 GiB",
        "decode_tokens": 512,
        "mtp_depth": 8,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "design": {
            "order": [t for t, _, _ in PLAN],
            "warm_up_legs_declared_and_discarded": [l["tag"] for l in warm],
            "timed_legs": len(timed),
            "leg_positions_by_arm": {
                arm: [l["position"] for l in timed if l["arm"] == arm]
                for arm in ARMS},
            "position_sum_by_arm": position_sums,
            "position_mean_by_arm": position_means,
            "position_balanced": len(set(position_sums.values())) == 1,
        },
        "thermal": {
            "protocol": "MLXFAST_LOCAL_COOL_GATE=0, ABBA-counterbalanced",
            "entry_gpu_temp_spread_c": entry_spread,
            "entry_gpu_temp_mean_c_by_arm": {
                arm: by_arm[arm]["entry_gpu_temp_mean_c"] for arm in ARMS},
        },
        "legs": legs,
        "by_arm": by_arm,
        "regression": headline,
        "session_null": {
            "conservative_null_pct": session_null,
            "definition": "largest same-arm spread in this session",
            "retracted_floor_pct": 0.0629,
            "detail": null,
        },
        "measured_contrasts_pct": measured,
        "measured_contrast_stderr_pct": measured_se,
        "preregistered_comparison": {
            "B_minus_A": {"predicted_pct": PREREG["B_minus_A"]["pct"],
                          "predicted_stderr_pct": prereg_se["B_minus_A"],
                          "measured_pct": measured["B_minus_A"],
                          "measured_stderr_pct": measured_se["B_minus_A"],
                          "residual_pct": measured["B_minus_A"] - PREREG["B_minus_A"]["pct"],
                          "source": PREREG["B_minus_A"]["source"],
                          "replicates_within_session_null": replication_ok},
            "C_minus_B": {"predicted_pct": PREREG["C_minus_B"]["pct"],
                          "predicted_stderr_pct": prereg_se["C_minus_B"],
                          "measured_pct": measured["C_minus_B"],
                          "measured_stderr_pct": measured_se["C_minus_B"],
                          "residual_pct": measured["C_minus_B"] - PREREG["C_minus_B"]["pct"],
                          "source": PREREG["C_minus_B"]["source"]},
            "C_minus_A": {"predicted_pct_if_additive": PREREG_SUM_PCT,
                          "predicted_stderr_pct": prereg_sum_se,
                          "measured_pct": measured["C_minus_A"],
                          "measured_stderr_pct": measured_se["C_minus_A"],
                          "residual_pct": additivity_residual,
                          "combined_stderr_pct": combined_se,
                          "residual_in_combined_se":
                              additivity_residual / combined_se,
                          "additive_within_one_combined_se":
                              abs(additivity_residual) <= combined_se,
                          "verdict": ("additive"
                                      if abs(additivity_residual) <= combined_se
                                      else ("sub-additive (substitution)"
                                            if abs(measured["C_minus_A"])
                                            < abs(PREREG_SUM_PCT)
                                            else "super-additive")),
                          "shared_value_fraction":
                              1.0 - measured["C_minus_A"] / PREREG_SUM_PCT},
            "identity_caveat": (
                "Inside one linear model (C-A) = (B-A) + (C-B) by construction. "
                "Additivity is therefore tested only against the two independent "
                "single-mechanism measurements, never against this session alone."),
        },
        "per_leg_arm_effect": per_leg_effect,
        "sign_stability": {
            "c_leg_effects_pct": c_effects,
            "c_sign_stable_negative": all(e < 0 for e in c_effects),
            "b_leg_effects_pct": b_effects,
            "b_sign_stable_negative": all(e < 0 for e in b_effects),
        },
        "null_channels": channels,
        "rung1_worker_unchanged_across_every_leg": rung1_ok,
        "dispatch_live_every_leg": dispatch_live,
        "schedule_identical_across_all_legs": schedule_identical,
        "correctness_ok_every_leg": correctness_ok,
        "headline_excludes_prefill": headline_excludes_prefill,
        "wired_residency_active": False,
    }

    effect_multiple = abs(measured["C_minus_A"]) / session_null
    report["stop_rules"] = {
        "c_minus_a_negative": measured["C_minus_A"] < 0,
        "effect_multiple_of_session_null": effect_multiple,
        "beats_null_by_3x": effect_multiple >= 3.0,
        "c_sign_stable": all(e < 0 for e in c_effects),
        "b_minus_a_replicates": replication_ok,
    }
    if not correctness_ok or not rung1_ok or not dispatch_live:
        verdict = "invalid"
    elif not replication_ok:
        verdict = "stop: t6 did not replicate"
    elif (measured["C_minus_A"] < 0 and effect_multiple >= 3.0
          and all(e < 0 for e in c_effects)):
        verdict = "advance to rung 5"
    elif measured["C_minus_A"] < 0:
        verdict = "unclear"
    else:
        verdict = "stop"
    report["verdict"] = verdict

    pathlib.Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("pos tag    arm         mtp s/tok    serial s/tok  speedup   entry->exit GPU C")
    for l in legs:
        print("%3s %-5s %-11s %.8f  %.8f  %.6f  %s -> %s %s"
              % (l["position"] if l["position"] else "-", l["tag"], l["arm"],
                 l["mtp_seconds_per_token"], l["serial_seconds_per_token"],
                 l["mtp_decode_speedup"], gpu_temp(l["thermal_before"]),
                 gpu_temp(l["thermal_after"]),
                 "DISCARDED" if l["discarded"] else ""))
    print()
    for arm in ARMS:
        print("%-11s mean %.8f  (spread %.4f %%, %d legs, positions %s)"
              % (arm, by_arm[arm]["mtp_seconds_per_token_mean"],
                 by_arm[arm]["same_arm_spread_pct"], by_arm[arm]["legs"],
                 report["design"]["leg_positions_by_arm"][arm]))
    print("position sums %s  (balanced: %s)"
          % (position_sums, report["design"]["position_balanced"]))
    print()
    for k in ("B_minus_A", "C_minus_B", "C_minus_A"):
        c = headline["contrasts"][k]
        print("%-11s %+.8f s/tok = %+.4f %% +- %.4f  (t = %.2f, dof %d)"
              % (k, c["estimate"], c["pct_of_arm_a_mean"],
                 c["stderr_pct_of_arm_a_mean"], c["t"], headline["dof"]))
    print("position drift %+.5f %%/leg (t = %.2f)"
          % (headline["position_coef_pct_per_leg"], headline["position_coef_t"]))
    print("session null   %.4f %%   C-A is %.2fx the null"
          % (session_null, effect_multiple))
    print()
    pc = report["preregistered_comparison"]
    print("prereg B-A %+.4f -> measured %+.4f  (replicates: %s)"
          % (PREREG["B_minus_A"]["pct"], measured["B_minus_A"], replication_ok))
    print("prereg C-B %+.4f -> measured %+.4f" % (PREREG["C_minus_B"]["pct"],
                                                  measured["C_minus_B"]))
    print("prereg C-A %+.4f -> measured %+.4f  residual %+.4f = %.2f combined se -> %s"
          % (PREREG_SUM_PCT, measured["C_minus_A"], additivity_residual,
             pc["C_minus_A"]["residual_in_combined_se"], pc["C_minus_A"]["verdict"]))
    print()
    print("null channels (each against its own largest same-arm spread):")
    for k, v in channels.items():
        print("  %-32s C-A %+.4f %%  (own null %.4f %%)  %s"
              % (k, v["contrasts_pct"]["C_minus_A"],
                 v["own_largest_same_arm_spread_pct"],
                 "quiet" if v["quiet"] else "MOVED"))
    print()
    print("rung 1 worker unchanged every leg: %s" % rung1_ok)
    print("dispatch live every leg:           %s" % dispatch_live)
    print("correctness every leg:             %s" % correctness_ok)
    print("schedule identical every leg:      %s" % schedule_identical)
    print("verdict: %s -> %s" % (verdict, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
