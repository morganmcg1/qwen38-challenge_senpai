#!/usr/bin/env python3
"""E149 arm C1 -- price the shipped SDPA query split against the F4 thresholds.

Reads `research/e149-c1-sdpa.json` (written by `E149SdpaSplitCostTests`) and
turns the per-(kL, width) cells into the named frames the assignment asks for.

THE IDENTIFICATION PROBLEM. The probe measures `split - merged_forecast`, which
mixes two components:

  structural   one extra kernel dispatch plus a concat per layer. Does NOT
               depend on kL. The probe runs one `eval()` per arm, so its
               command buffer never fills and this term is inflated relative
               to the scored path, which queues 64 layers.
  streaming    the second sdpa call re-reads the whole KV of the layer. Grows
               strictly linearly in kL at 4096 bytes per key
               (K and V, 4 kv heads, 256 head dim, 2 bytes).

Only the streaming term is identified by this probe, and it is identified
cleanly: structure cannot depend on kL, so the kL slope of the raw difference
IS the streaming term. Its size is cross-checked against an independent
bandwidth calculation. A merged call reads the KV once instead of twice, so the
streaming term is exactly what a merged form would return.

Three decompositions are therefore reported, all frame-labelled:

  raw          split - extrapolated merged vector call. A hard UPPER BOUND
               that carries the probe's non-overlapped dispatch latency.
  streaming    16 * b * E[kL], from the kL regression. The identified,
               physically corroborated, recoverable term. HEADLINE.
  tinyKV       raw minus the kL == 8 structural control. Reported because the
               assignment asked for it, but it over-subtracts: the control
               exceeds the fitted kL-independent intercept.

Width 9 is void: the tiny-KV control loop runs `width <= kL` at kL == 8, so no
`split_m9` structural control exists, and `SEGMENTED_VERIFY_DEPTH_CAP = 7`
caps the scored verify width at 8 anyway.
"""

import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "e149-c1-sdpa.json"
OUT = HERE / "e149-c1-report.json"

# kL walks 513 -> 1024 across the 512-token decode window. Three sample points
# under trapezoid quadrature for a uniform density on that interval.
KL_WEIGHT = {513: 0.25, 769: 0.50, 1024: 0.25}

FULL_ATTENTION_LAYERS = 16
# K and V, 4 kv heads, 256 head dim, 2 bytes each.
BYTES_PER_KEY_PER_LAYER = 2 * 4 * 256 * 2
M4_PRO_PEAK_GB_PER_S = 273.0

# F4 supplied masses.
RANKED_WIDTH8_MASS = 0.5390
RANKED_P_M_GE_6 = 0.5861
LOCAL_BENCHFIXTURE_SHIP_P_M_GE_6 = 0.7692

# F235 median conversion: Arm A's -132.55 us/round read as +0.2573 % of median.
US_PER_ROUND_PER_PCT = 132.55 / 0.2573

DETECTION_FLOOR_US = 44.5          # Rule 147, essays_montaigne, 2 sigma
C2_TRIGGER_US = 150.0              # F4: C2 opens only above ~150 us/round

SCORED_WIDTHS = [6, 7, 8]


def ols(xs, ys):
    """Slope, intercept, residual rms and slope standard error."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / max(1, n - 2)
    return {
        "slope": b,
        "intercept": a,
        "slope_stderr": math.sqrt(s2 / sxx),
        "resid_rms": math.sqrt(sum(r * r for r in resid) / n),
    }


def main() -> None:
    src = json.loads(SRC.read_text())
    per_round = src["per_round"]
    control = src["control_host_microseconds"]

    def row(kL, w):
        return next(r for r in per_round if r["kL"] == kL and r["width"] == w)

    # Within-cell repeatability: the probe measures every arm forward and then
    # in reverse order, so the half-spread is a drift-inclusive noise proxy.
    spreads = [
        abs(c["forward_microseconds"] - c["reverse_microseconds"]) / 2.0 / c["microseconds"]
        for c in src["cells"] if c["microseconds"] > 0
    ]
    noise = {
        "palindrome_half_spread_pct_mean": 100.0 * sum(spreads) / len(spreads),
        "palindrome_half_spread_pct_worst": 100.0 * max(spreads),
    }

    # ---- identification: regress the raw per-layer difference on kL ----------
    xs, ys = [], []
    for kL in KL_WEIGHT:
        for w in SCORED_WIDTHS:
            xs.append(float(kL))
            ys.append(row(kL, w)["upper_bound_saving_us_per_layer"])
    pooled = ols(xs, ys)

    per_width_fit = {
        w: ols([float(k) for k in KL_WEIGHT],
               [row(k, w)["upper_bound_saving_us_per_layer"] for k in KL_WEIGHT])
        for w in SCORED_WIDTHS
    }

    b = pooled["slope"]
    b_se = pooled["slope_stderr"]
    mean_kL = sum(w * k for k, w in KL_WEIGHT.items())
    implied_bandwidth_gbs = BYTES_PER_KEY_PER_LAYER / b / 1000.0

    streaming_per_layer = b * mean_kL
    streaming_per_round = streaming_per_layer * FULL_ATTENTION_LAYERS
    streaming_per_round_se = b_se * mean_kL * FULL_ATTENTION_LAYERS

    # ---- the three decompositions, per M >= 6 round -------------------------
    # The streaming term does not depend on the verify width: the extra read is
    # the whole KV whatever M is. Only raw and tinyKV need a width mix.
    w8_share = RANKED_WIDTH8_MASS / RANKED_P_M_GE_6
    residual = 1.0 - w8_share
    mix = {6: residual / 2.0, 7: residual / 2.0, 8: w8_share}

    by_width = {}
    for w in SCORED_WIDTHS:
        by_width[w] = {
            "raw_us_per_round": sum(
                wt * row(k, w)["upper_bound_saving_us_per_round"]
                for k, wt in KL_WEIGHT.items()),
            "tinyKV_us_per_round": sum(
                wt * row(k, w)["gpu_attributable_us_per_round"]
                for k, wt in KL_WEIGHT.items()),
        }

    base = {
        "raw": sum(mix[w] * by_width[w]["raw_us_per_round"] for w in SCORED_WIDTHS),
        "streaming": streaming_per_round,
        "tinyKV": sum(mix[w] * by_width[w]["tinyKV_us_per_round"] for w in SCORED_WIDTHS),
    }

    frames = {}
    for fname, mass in [
        ("frameA_per_M_ge_6_round", 1.0),
        ("frameB_per_drafting_round_ranked", RANKED_P_M_GE_6),
        ("frameC_per_drafting_round_local_benchfixture_ship",
         LOCAL_BENCHFIXTURE_SHIP_P_M_GE_6),
    ]:
        f = {"mass_applied": mass,
             "streaming_us_per_round_stderr": streaming_per_round_se * mass}
        for k, v in base.items():
            f[f"{k}_us_per_round"] = v * mass
            f[f"{k}_pct_of_median"] = v * mass / US_PER_ROUND_PER_PCT
        frames[fname] = f

    headline = frames["frameB_per_drafting_round_ranked"]["streaming_us_per_round"]
    headline_se = frames["frameB_per_drafting_round_ranked"]["streaming_us_per_round_stderr"]

    control_structural = {
        str(w): control[f"split_m{w}"] - control["single_q5"] for w in SCORED_WIDTHS
    }
    exact = [e for e in src["exactness"] if e["width"] in SCORED_WIDTHS]

    report = {
        "arm": "e149_c1_sdpa_query_split",
        "harness": "local",
        "cool_gate_passed_real_gate": src["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": src["gate_qualified_for_timing"],
        "host": "Apple M4 Pro Mac16,11 48 GB",
        "source_json": SRC.name,

        "e149_sdpa_split_us_per_round": headline,
        "e149_sdpa_split_us_per_round_stderr": headline_se,
        "e149_sdpa_split_frame": "frameB_per_drafting_round_ranked_streaming",
        "e149_sdpa_split_us_per_round_raw_upper_bound":
            frames["frameB_per_drafting_round_ranked"]["raw_us_per_round"],
        "e149_sdpa_split_us_per_round_tinyKV_corrected":
            frames["frameB_per_drafting_round_ranked"]["tinyKV_us_per_round"],

        "frames": frames,
        "by_width_kL_averaged": by_width,
        "width_mix_within_M_ge_6": mix,
        "kL_quadrature_weights": {str(k): v for k, v in KL_WEIGHT.items()},
        "mean_kL": mean_kL,

        "identification": {
            "pooled_slope_us_per_key_per_layer": b,
            "pooled_slope_stderr": b_se,
            "pooled_intercept_us_per_layer": pooled["intercept"],
            "pooled_resid_rms_us_per_layer": pooled["resid_rms"],
            "per_width_slope": {str(w): per_width_fit[w]["slope"] for w in SCORED_WIDTHS},
            "bytes_per_key_per_layer": BYTES_PER_KEY_PER_LAYER,
            "implied_effective_bandwidth_gb_per_s": implied_bandwidth_gbs,
            "implied_fraction_of_m4_pro_peak":
                implied_bandwidth_gbs / M4_PRO_PEAK_GB_PER_S,
            "streaming_us_per_layer_at_mean_kL": streaming_per_layer,
            "tinyKV_control_structural_us_per_layer": control_structural,
            "tinyKV_over_subtraction_us_per_layer": {
                str(w): control_structural[str(w)] - pooled["intercept"]
                for w in SCORED_WIDTHS
            },
            "note": (
                "The kL slope is common across widths, as an extra whole-KV "
                "read must be. It implies an effective bandwidth that is a "
                "plausible fraction of the M4 Pro 273 GB/s peak, which is an "
                "independent corroboration the probe did not fit. The tiny-KV "
                "control sits above the fitted kL-independent intercept, so "
                "subtracting it drives the residual to about zero and destroys "
                "a term the regression resolves cleanly."
            ),
        },

        "e149_c1_width9_status":
            "void_no_structural_control_and_unreachable_under_depth_cap_7",
        "noise": noise,

        "exactness": {
            "split_vs_split_max_abs": max(e["split_vs_split_max_abs"] for e in exact),
            "split_vs_fallback_max_abs": max(e["split_vs_fallback_max_abs"] for e in exact),
            "positive_control_min_abs": min(e["positive_control_max_abs"] for e in exact),
            "note": (
                "split reproduces itself exactly and the perturbation control "
                "always fires, so the comparison is shown able to fail. The "
                "composed fallback is NOT bit-exact with the shipped split, so "
                "deleting the AttentionUtils guard would change tokens."
            ),
        },

        "thresholds": {
            "detection_floor_us_per_round": DETECTION_FLOOR_US,
            "c2_trigger_us_per_round": C2_TRIGGER_US,
            "headline_clears_floor": headline >= DETECTION_FLOOR_US,
            "headline_clears_c2_trigger": headline >= C2_TRIGGER_US,
            "headline_minus_2se_clears_floor":
                headline - 2 * headline_se >= DETECTION_FLOOR_US,
            "headline_minus_2se_clears_c2_trigger":
                headline - 2 * headline_se >= C2_TRIGGER_US,
        },

        "e149_c2_opens": False,
        "e149_c2_blocked_reason": (
            "unreachable_not_editable. benchmark.json editablePaths carries the "
            "sdpa KERNEL sources (sdpa_vector.h, "
            "scaled_dot_product_attention.metal, steel/attn) but NO .cpp under "
            "backend/metal. The routing gate that picks vector against composed "
            "lives in backend/metal/scaled_dot_product_attention.cpp:631-639, "
            "which is host code we may not edit. head_dim 256 with gqa 6 makes "
            "supports_sdpa_full permanently false and caps supports_sdpa_vector "
            "at floor(32/6) = 5 query rows, which is exactly the shipped split "
            "point. Editing the kernel cannot change which kernel is dispatched."
        ),
        "us_per_round_per_pct_of_median": US_PER_ROUND_PER_PCT,
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {OUT}\n")
    print(f"palindrome half-spread: mean {noise['palindrome_half_spread_pct_mean']:.3f} %  "
          f"worst {noise['palindrome_half_spread_pct_worst']:.3f} %\n")
    print("raw per-layer difference, us/layer")
    for kL in KL_WEIGHT:
        vals = " ".join(f"M{w}={row(kL, w)['upper_bound_saving_us_per_layer']:7.2f}"
                        for w in SCORED_WIDTHS)
        print(f"  kL={kL:5d}  {vals}")
    print()
    print(f"pooled fit   y = {pooled['intercept']:.2f} + {b:.6f} * kL   "
          f"(se {b_se:.6f}, rms {pooled['resid_rms']:.2f} us/layer)")
    print(f"  implied effective bandwidth {implied_bandwidth_gbs:.1f} GB/s "
          f"({100 * implied_bandwidth_gbs / M4_PRO_PEAK_GB_PER_S:.0f} % of M4 Pro peak)")
    print("  per-width slopes " +
          " ".join(f"M{w}={per_width_fit[w]['slope']:.6f}" for w in SCORED_WIDTHS))
    print(f"  tiny-KV control structural {control_structural}")
    print(f"  fitted kL-independent intercept {pooled['intercept']:.2f} us/layer\n")
    for name, f in frames.items():
        print(name)
        print(f"   raw        {f['raw_us_per_round']:9.1f} us/round = "
              f"{f['raw_pct_of_median']:+.4f} % of median   [UPPER BOUND]")
        print(f"   streaming  {f['streaming_us_per_round']:9.1f} "
              f"+/- {f['streaming_us_per_round_stderr']:.1f} us/round = "
              f"{f['streaming_pct_of_median']:+.4f} % of median   [HEADLINE]")
        print(f"   tinyKV     {f['tinyKV_us_per_round']:9.1f} us/round = "
              f"{f['tinyKV_pct_of_median']:+.4f} % of median   [over-subtracted]")
    print()
    t = report["thresholds"]
    print(f"e149_sdpa_split_us_per_round = {headline:.1f} +/- {headline_se:.1f}")
    print(f"  clears {DETECTION_FLOOR_US} floor: {t['headline_clears_floor']} "
          f"(at -2se {t['headline_minus_2se_clears_floor']})")
    print(f"  clears {C2_TRIGGER_US} C2 trigger: {t['headline_clears_c2_trigger']} "
          f"(at -2se {t['headline_minus_2se_clears_c2_trigger']})")
    print("e149_c2_opens = False -- routing gate is host .cpp, not editable")


if __name__ == "__main__":
    main()
